# -*- coding: utf-8 -*-
"""HMM live 引擎的看護 —— 有問題就送 Discord，沒問題就安靜。

    python research/ops/hmm_watch.py --pair AERO
    python research/ops/hmm_watch.py --pair AERO --heartbeat   # 強制送一次現況

===========================================================================
為什麼在 flow_system 而不是 arb（2026-09-14）
===========================================================================
CLAUDE.md §第 4 線的隔離：arb 是單向的 —— **flow_system 讀它，它不讀
flow_system**。`freshness_board` 與 `prereg_publish` 已經是這個形狀
（讀 `arb/engine/logs/*/minutes.csv`）。告警管線（`notify.py`，Discord 主
Telegram 備）住在這裡，所以看護也住在這裡，而不是讓 arb 去 import 它。

===========================================================================
它盯什麼（每一條都有一個真實的來歷）
===========================================================================
    行程不在        引擎死了而看門狗還沒補上 —— 或者它被停了而沒人知道
    HALT            一次性事件，而 HALT 是單向的：**要人去重啟**
    帳戶層拒絕      交易所因保證金／持倉限制撤我們的單 —— 不是行情問題
    裸曝險過大      net 超過 max_net_base 的八成 = 快要 HALT 了
    停止報價        活著但一小時沒報價 = 帶設錯或市場沒機會（FIL 的病）
    API 斷線頻繁    WAF 在擋，而它擋的是**重連**，那時引擎手上有部位
    旗標過期        status.json 不動 = 引擎卡住，而行程還在

**狀態轉換才送，不是每次都送**（`_last.json` 記上次送的是什麼）——
一個每五分鐘說一次「還好」的頻道，出事那次會被當成雜訊
（transition-only，freshness_board 的同一條）。

===========================================================================
訊息裡不放什麼
===========================================================================
不放美元金額、帳戶權益、webhook。**放的是狀態、方向、時間與計數** ——
跟公開面同一條規矩，而理由不同：這個頻道在手機上，手機會掉。
唯一的例外是裸曝險，它以**佔上限的百分比**表示，不以金額。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import notify                                            # noqa: E402
from live_hmm import live_hmm_pairs                      # noqa: E402

ARB = os.path.join("C:", os.sep, "Users", "rfo", "Desktop", "flowbot", "arb")
LOGS = os.path.join(ARB, "engine", "logs")
# 可覆寫,唯一的用途是**驗這支自己**：抑制/放行那兩個分支要靠狀態檔累積的
# 計數,所以測試必須能給它一個乾淨的、不會污染正式狀態的檔案
# （2026-09-14：加「連續兩次才報」時,沒有這個覆寫就只能拿真的告警去驗）。
STATE = os.environ.get("HMM_WATCH_STATE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hmm_watch_last.json")

STALE_FLAG_SEC = 180.0        # status.json 多久沒動就算引擎卡住
QUIET_QUOTE_SEC = 3600.0      # 活著但這麼久沒報價 = 帶設錯（FIL 的病）
NET_WARN_FRAC = 0.8           # 裸曝險到上限的八成就先說
WAF_WINDOW_MIN = 20           # 只看最近這麼久的 WAF 命中
WAF_LIMIT = 4                 # 窗內超過這個數才算「現在有事」
# 撤單確認不了要多久才算「卡住」。引擎每 ~60 秒印一次 CRITICAL，所以 10 分鐘
# 的窗代表「剛剛還在卡」；用整份 log 會把幾小時前已經重啟解決的那次也算進來。
UNRESOLVED_WINDOW_MIN = 10


def _window_ok(txt: str, window_min: int) -> bool:
    """這份 log 算不算得出「最近 window_min 分鐘」。

    **為什麼要把這件事分出來（2026-09-15 00:10 的事故）**：`_count_recent`
    在跨午夜時退回「數全部」，而那個退路對兩個呼叫端的**安全方向是相反的**：

        WAF（「最近有沒有被擋」）  數全部 -> 多報 -> 保守，沒問題
        quotes（「還有沒有在報價」）數全部 -> **永遠不是 0 -> 警報永遠不響**

    而那晚它真的沒響：MON 在 22:58 卡住（撤單確認不了）、**72 分鐘沒報一張
    單**，而看護每五分鐘說一次 🟢，因為 00:0x 的那幾輪退回數了大半天的 427。

    同一個退路，一邊是保守一邊是致命 —— 所以呼叫端必須自己決定，
    而不是共用一個「看起來安全」的預設（mistake.md 2026-09-14：
    未知狀態不可以長得像一個已知狀態）。
    """
    import re
    ts = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")
    for ln in reversed(txt.split("\n")):
        m = ts.match(ln)
        if m:
            last = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    + int(m.group(3)))
            return last - window_min * 60 >= 0
    return False


def _count_recent(txt: str, needle: str, window_min: int) -> int:
    """數**最近 window_min 分鐘**內出現幾次,不是整份 log。

    引擎的 log 每行開頭是 `HH:MM:SS.mmm`（本地時間,沒有日期）。所以用
    最後一行的時刻當「現在」,往回推 —— 不用牆鐘,避免時鐘與 log 不同源。
    跨午夜時最後一行的時刻會小於前面的,那時就退回數全部（保守方向：
    多報一次總比在跨日那幾分鐘瞎掉好）。
    """
    import re
    ts = re.compile(r"^(\d{2}):(\d{2}):(\d{2})")
    lines = txt.split("\n")
    last = None
    for ln in reversed(lines):
        m = ts.match(ln)
        if m:
            last = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            break
    if last is None:
        return txt.count(needle)
    cut = last - window_min * 60
    if cut < 0:
        return txt.count(needle)          # 跨午夜,不要假裝算得準
    n = 0
    for ln in lines:
        if needle not in ln:
            continue
        m = ts.match(ln)
        if m is None:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if t >= cut:
            n += 1
    return n


def _proc_alive(pair: str) -> bool:
    """引擎在不在。用指令列比對，跟看門狗同一個判準。"""
    import subprocess
    # `@(...)` 不是裝飾：Windows PowerShell 5.1 對**單一物件**回傳的 `.Count`
    # 是空的,於是 int("") -> 0 -> 「行程不在」。第一版就是這樣對一個正在
    # 報價的引擎報死 —— 而這支會把那個假警報送到 Discord,每五分鐘一次。
    # 抓到它的是「一個正在報價的引擎不可能不在」這個矛盾,不是我的警覺。
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'symbol %s' }).Count" % pair],
            capture_output=True, text=True, timeout=60)
        s = (out.stdout or "").strip()
        if not s.isdigit():
            return True    # 讀不懂就**不要**宣告它死了
        return int(s) >= 1
    except Exception:
        return True        # 查不到也一樣（fail-safe，mistake.md 2026-09-11：
                           # 看門狗的例外分支預設必須是不動手）


def look(pair: str) -> tuple[list, dict]:
    """回傳 (問題清單, 現況)。問題是字串，現況給心跳用。"""
    d = os.path.join(LOGS, pair)
    probs, now = [], time.time()
    sj = os.path.join(d, "status.json")
    st = {}
    if not os.path.exists(sj):
        return ["沒有 status.json —— 這個配對從來沒起來過"], {}
    age = now - os.path.getmtime(sj)
    try:
        st = json.load(io.open(sj, encoding="utf-8"))
    except Exception as e:
        return ["status.json 讀不了：%s" % e], {}
    pub = st.get("public") or {}
    pri = st.get("private") or {}
    cnt = pub.get("counts") or {}

    if not _proc_alive(pair):
        probs.append("**引擎行程不在**（看門狗五分鐘內會補；沒補就是註冊被拿掉了）")
    if age > STALE_FLAG_SEC:
        probs.append("**status.json %.0f 分鐘沒動** —— 行程可能卡住了" % (age / 60))
    if not st.get("ok", True):
        probs.append("**ok=False**：%s" % str(st.get("reason", ""))[:120])
    if cnt.get("stale_episodes", 0) >= (cnt.get("stale_episode_limit", 5) - 1):
        probs.append("**stale episodes %s/%s** —— 再一次就 HALT"
                     % (cnt.get("stale_episodes"), cnt.get("stale_episode_limit")))
    if cnt.get("post_only_rejects", 0) > 20:
        probs.append("post-only 被拒 %s 次 —— 我們一直掛進對手價裡"
                     % cnt["post_only_rejects"])

    net = abs(float(pri.get("net_base") or 0.0))
    cap = float((pub.get("guards_cap") or 0) or 0)
    if not cap:                                   # 設定值不在 status 裡就讀 yaml
        try:
            import yaml
            y = yaml.safe_load(io.open(os.path.join(
                ARB, "engine", "config_%s.yaml" % pair), encoding="utf-8"))
            cap = float((y.get("risk") or {}).get("max_net_base") or 0)
        except Exception:
            cap = 0.0
    if cap and net >= cap * NET_WARN_FRAC:
        probs.append("**裸曝險 %.0f%% 上限**（對沖沒跟上，超過就 HALT）"
                     % (net / cap * 100))

    rl = os.path.join(d, "runner.log")
    quotes = fills = waf = 0
    last_quote_age = None
    if os.path.exists(rl):
        try:
            txt = io.open(rl, encoding="utf-8", errors="replace").read()[-400000:]
            if "HALTED" in txt:
                probs.append("**HALTED** —— 單向的，要人去重啟（重啟會用 "
                             "strict=True 重讀真實部位）")
            if "MAKER ORDER CANCELLED BY THE VENUE FOR AN ACCOUNT" in txt:
                probs.append("**交易所因帳戶原因撤我們的單** —— 不是行情，"
                             "查保證金／持倉限制")
            # **速率不是累計。** 第一版數整份 log 的 "API unreachable",而
            # 累計數遲早一定會跨過任何固定門檻 -> 一盞永遠亮的紅燈,然後被
            # 當成雜訊（mistake.md 2026-09-03：永遠紅的燈跟壞掉的燈一樣沒用）。
            # 改成只數最後 WAF_WINDOW_MIN 分鐘 —— 那才是「現在有沒有在擋」。
            waf = _count_recent(txt, "API unreachable", WAF_WINDOW_MIN)
            # **2026-09-14：上面那個修法只套用到 waf，而 quotes / fills 就在
            # 它上面兩行、犯的是同一個病。** 兩者都是 `txt.count(...)`，
            # 而 `txt` 是 runner.log 的最後 40 萬個字元 ≈ 大半天 ——
            #
            #   後果一（嚴重）：`quotes == 0` 幾乎不可能成立，於是下面那盞
            #     「活著但沒報價」**結構上點不亮**，而它正是為了 FIL 的病建的。
            #     實測：引擎本次行程只掛了 36 張，而這裡數到 1103。
            #   後果二：印出來的「上線 N 小時｜報價 M」把**本次行程**的上線
            #     時間跟**大半天**的報價數放在同一行，而成交率是兩者相除 ——
            #     同一個容器裡兩個不同的窗（mistake.md 2026-09-13 的形狀）。
            #
            # 改成跟 waf 同一個做法：問「最近這段時間」。窗取
            # QUIET_QUOTE_SEC，因為那正是這盞燈要判的那段。
            #
            # **第二個錯,同一行:`"[QUOTE] "` 一張單會命中兩次。** 全 log 實測
            #     [QUOTE] sell / buy    619   <- 掛出去
            #     [QUOTE] cancelling    612   <- 撤掉,同一張單
            #     [QUOTE DONE]          612   <- 一張單剛好一次
            # 所以用 `[QUOTE DONE]`:它是「一張單解析完」的那一筆,跟 maker.csv
            # 的列一一對應（612 對 578,差的 34 正好是那批 csv 寫入失敗）。
            qwin = int(QUIET_QUOTE_SEC // 60)
            quotes = _count_recent(txt, "[QUOTE DONE]", qwin)
            fills = _count_recent(txt, "[QUOTE FILL]", qwin)
            # **算不出這個窗就不要判**（`_window_ok` 的 docstring 有事故經過）。
            # 退回「數全部」在這一格是致命的:它讓警報永遠不響,而且剛好在
            # 每天午夜後的第一個小時。
            qwin_ok = _window_ok(txt, qwin)
            if not qwin_ok:
                quotes = fills = None
            if qwin_ok and quotes == 0 and age < STALE_FLAG_SEC:
                up = float(pub.get("uptime_sec") or 0)
                if up > QUIET_QUOTE_SEC:
                    probs.append("**活著但 %.0f 小時沒報價** —— 帶設錯或這個"
                                 "市場沒機會（FIL 的病）" % (up / 3600))

            # **撤單確認不了 = 引擎會停在那裡,而它不會自己好。**
            # 2026-09-15 00:10 實際發生:Lighter 的帳戶 WS 串流在送出撤單的
            # **同一秒**重連,而 `LighterVenue.poll_order` 只讀那個串流的快取
            # （設定裡明寫 "There is deliberately no REST fallback here"）——
            # 單已經從交易所消失,所以不會再有它的更新進快取,於是那張單的
            # 終態**永遠確認不了**。引擎照設計保持悲觀（未確認的撤單不是
            # 已撤單),於是 723 次重試、72 分鐘一張新單都沒掛。
            #
            # 那 72 分鐘裡它印了 71 行 CRITICAL,而這支一行都沒看 ——
            # 它只認得 "HALTED" 與帳戶層撤單。重啟時的啟動掃單回報
            # **零張掛單**,證明那張單一直都撤掉了:沒有曝險,壞的是確認路徑。
            # 用短窗（不是整份 log）：要答的是「**現在**還卡著嗎」。
            if _window_ok(txt, UNRESOLVED_WINDOW_MIN) and _count_recent(
                    txt, "MAKER ORDER STILL UNRESOLVED",
                    UNRESOLVED_WINDOW_MIN) > 0:
                probs.append("**撤單確認不了,引擎卡著** —— 它不會自己好,"
                             "要重啟（啟動時的 REST 掃單會清掉）")
        except Exception:
            pass
    if waf >= WAF_LIMIT:
        probs.append("Lighter API 最近 %d 分鐘斷線 %d 次 —— WAF 在擋，"
                     "而它擋的是**重連**" % (WAF_WINDOW_MIN, waf))

    cur = {"pair": pair, "ok": st.get("ok"), "quotes": quotes, "fills": fills,
           "hedges": cnt.get("hedges", 0), "waf": waf,
           "net_frac": (net / cap * 100) if cap else None,
           "uptime_h": float(pub.get("uptime_sec") or 0) / 3600.0,
           "fill_rate": pub.get("fill_rate_pct"),
           "markout": (pub.get("markout") or {}).get("vs_mid_bps")}
    return probs, cur


def text_for(pair: str, probs: list, cur: dict) -> str:
    head = "🔴 HMM %s" % pair if probs else "🟢 HMM %s" % pair
    lines = [head]
    for p in probs:
        lines.append("• " + p)
    mk = cur.get("markout")
    # **報價／成交是「最近一小時」，上線與對沖是「本次行程」** —— 窗不同就
    # 要寫出來，否則讀的人會把它們相除（2026-09-14 那個 1103 就是這樣來的）。
    # `None` = 這份 log 算不出那個窗（跨午夜）。**印「未量」不要印 0** ——
    # 0 會被讀成「真的沒報價」,而那是完全相反的結論。
    def _n(v):
        return "未量" if v is None else "%d" % v
    lines.append("— 上線 %.1f 小時｜近 1 小時：報價 %s、成交 %s｜對沖 %d"
                 % (cur.get("uptime_h", 0), _n(cur.get("quotes")),
                    _n(cur.get("fills")), cur.get("hedges", 0)))
    bits = []
    if cur.get("fill_rate") is not None:
        bits.append("成交率 %.0f%%" % cur["fill_rate"])
    if mk is not None:
        bits.append("markout(vs mid) %+.1f bps" % mk)
    if cur.get("net_frac") is not None:
        bits.append("裸曝險 %.0f%% 上限" % cur["net_frac"])
    if bits:
        lines.append("— " + "｜".join(bits))
    return "\n".join(lines)


def _watch_one(pair: str, a, last: dict) -> int:
    """盯一個標的。`last` 會被就地更新,**寫檔由呼叫端負責**（一次寫完）。"""
    probs, cur = look(pair)

    # 「行程不在」要連續兩次才報（2026-09-14）。
    #
    # 啟動器的迴圈 30 秒就會把引擎拉回來,而這支每 300 秒看一次 —— 所以
    # **「死了」跟「正在重啟」在單次檢查裡長得一模一樣**,而重啟是我們自己
    # 每天都會做的事。今天因此誤報三次(AERO、XPL、以及重啟 MON 裝儀器
    # 那 30 秒),而代價不是噪音而已:**真的紅燈會被埋在假的紅燈裡** ——
    # scan_pull 連死 31 次那一下午,唯一在響的頻道報的是別的東西。
    #
    # 這是今天寫進 mistake.md 那條的鏡像:「停止不是瞬間的」,所以
    # 「不在」也不是單次可判定的。兩次沒看到 = 跨過了一個 .bat 迴圈週期,
    # 那才是證據。
    #
    # **只放寬這一條。** HALT、帳戶拒絕、裸曝險那幾條是單次就算數的,
    # 它們描述的是一個持續狀態,不是一個可能正在恢復的瞬間。
    GONE = "**引擎行程不在**"
    miss_key = pair + ":procmiss"
    gone_now = any(p.startswith(GONE) for p in probs)
    misses = (int(last.get(miss_key, 0)) + 1) if gone_now else 0
    last[miss_key] = misses
    if gone_now and misses < 2:
        # **完全拿掉,不要換一句話留在 probs 裡** —— probs 就是狀態指紋,
        # 留一句「第 1 次沒看到」照樣是一次狀態轉換,照樣會送出去,
        # 那等於沒修。提示只印在本地。
        probs = [p for p in probs if not p.startswith(GONE)]
        print("[抑制] 引擎行程第 1 次沒看到 —— 啟動器 30 秒會補，"
              "連續兩次才算數")

    msg = text_for(pair, probs, cur)
    key = "|".join(sorted(probs))              # 狀態轉換的指紋
    changed = last.get(pair) != key
    print(msg)
    if a.dry:
        print("\n[乾跑] 轉換=%s，沒有送出" % changed)
        # 乾跑預設**不動狀態**（預覽就該沒有副作用）。唯一的例外是狀態檔
        # 被明確覆寫的時候 —— 那只發生在測試裡,而「連續兩次才報」這條
        # 沒有累積的狀態就驗不了(彩排跳過的那一段,正是要驗的那一段)。
        # **寫檔搬到 main()**（多個標的時只寫一次）,這裡只更新 dict;
        # 那個「乾跑要不要寫」的判斷跟著搬,語意一個字沒變。
        last[pair] = key
        return 0
    if changed or a.heartbeat:
        r = notify.send(msg, source="hmm_watch")
        print("[送出] delivered=%s tried=%s" % (r.get("delivered"), r.get("tried")))
        if not r.get("delivered"):
            # 送不出去本身是一盞紅燈 —— notify 會寫 alert_last.json，
            # 而那個旗標已經在新鮮度看板上（2026-09-13 那條的修法）。
            print("**告警送不出去** —— 看 alert_last.json", file=sys.stderr)
    else:
        print("[安靜] 狀態沒變，不重複送")
    last[pair] = key
    return 0


def _save(last: dict) -> None:
    io.open(STATE, "w", encoding="utf-8").write(
        json.dumps(last, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None,
                    help="省略 = 自己推導現在在跑的 live 標的（排程用這個）")
    ap.add_argument("--heartbeat", action="store_true",
                    help="不管有沒有轉換都送一次（開場／收工用）")
    ap.add_argument("--dry", action="store_true", help="印出來但不送")
    a = ap.parse_args(argv)

    # **排程不可以寫死標的（2026-09-14）。** 這個形狀當天出現四次,
    # 而代價不是噪音 —— AERO 停掉而看護沒停,整個下午每五分鐘一則假警報,
    # 於是 scan_pull 連死 31 次那個真紅燈沒有人看見。
    # 真相源是看門狗的 `$Members` 減去 STOP 旗標,**跟 freshness_board
    # 讀同一支** `live_hmm.live_hmm_pairs()` —— 兩份實作會安靜地不同意。
    pairs = [a.pair] if a.pair else live_hmm_pairs()
    if not pairs:
        # **這不是「沒事」,是「不知道」。** 但不在這裡另開一條告警路徑:
        # freshness_board 的 HMM 那一列讀同一支函式,推導不出來時它會紅,
        # 而那一列已經接上通知了（少一個沒驗過的送出路徑,就少一個
        # 「守衛存在但從沒開火過」的候選）。
        print("**推導不出在跑的 live HMM 標的** —— 這一輪不盯任何東西。"
              "這一格由 freshness_board 的 HMM 那一列負責（同一支函式）。")
        return 0

    last = {}
    if os.path.exists(STATE):
        try:
            last = json.load(io.open(STATE, encoding="utf-8"))
        except Exception:
            last = {}

    rc = 0
    for pair in pairs:
        rc |= _watch_one(pair, a, last)

    # 乾跑預設**不動狀態**（預覽就該沒有副作用）—— 這一條原本寫在
    # `_watch_one` 裡,搬上來之後語意一個字沒變。唯一的例外仍然是狀態檔
    # 被明確覆寫的時候,那只發生在測試裡。
    if not a.dry or os.environ.get("HMM_WATCH_STATE"):
        _save(last)
        if a.dry:
            print("[乾跑] 狀態已寫入沙盒 %s" % STATE)
    return rc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
