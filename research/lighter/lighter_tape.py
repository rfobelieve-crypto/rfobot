# -*- coding: utf-8 -*-
"""Lighter 全市場逐筆成交帶（常駐 WS，2026-09-12）

===========================================================================
為什麼要錄
===========================================================================
兩個開著的問題都卡在同一份資料上，而它**不可回填**（WS 串流，沒有歷史端點）：

  1. **我們在 Lighter 上做市有沒有毛利**（TODO §1.29）。§1.28 的 +0.58 bps
     是在 Binance 11 個厚標的上量的，而 §1.27/§1.28 的判決是**場館特定的**。
  2. **300ms 取消延遲值幾 bps**（TODO §1.31）—— 要逐筆成交才算得出
     「決定要撤之後的那段時間內價格逆向走多遠」。

===========================================================================
這份 payload 比 HL 的豐富得多，而多出來的正好是做市要的
===========================================================================
實測一筆（2026-09-12 探測，`scratchpad/probe_lighter_ws.py`）帶 31 個欄位。
關鍵的幾個：

    maker_fee / taker_fee          **雙方實付費率**
    taker_position_size_before     吃單方成交前的部位
    maker_position_size_before     掛單方成交前的部位
    taker/maker_initial_margin_fraction_before
    ask_account_id / bid_account_id   雙方帳戶（等價於 HL 的地址）
    transaction_time               **微秒**（timestamp 是毫秒）
    block_height / tx_hash

**`maker_fee` 的單位是 1e-6 的費率，這是量出來的不是猜的**：清算單的
`taker_fee` 是 `10000`，而 `/api/v1/orderBooks` 的 `liquidation_fee` 是
`"1.0000"`（= 1%）。10000 / 1e6 = 0.01 = 1% -> 對上。所以一般成交看到的
`maker_fee: 28` = 0.0028% = **0.28 bps**，正好是官方費率表上
「Premium ＋ 質押 500k LIT」那一格。

**這件事解掉一個我們以為解不掉的問題**：`arb/arblib/fee_receipts.py` 一直
卡在「收據查證要自己的帳戶成交，我們沒有」，而**公開成交帶上每一筆都帶
雙方實付費率** —— 於是「Lighter 上實際有多少流量是付 0 的、多少付 0.40、
多少質押到 0.28」變成可量測的，不需要我們自己成交。
（CLAUDE.md §HFT 的 (c) 那一條因此有了出路，但**本檔只負責錄，不做判決**。）

`taker/maker_position_size_before` 是第二個：它直接分得出
**「在管庫存的人」與「在下方向的人」**，而那正是倉位型做市的核心假設。

===========================================================================
凍結的設計決定
===========================================================================
    宇宙      **永續、按日成交額前 `TOP_N`**。原始量測（不是拍的）：
              227 個 active 市場裡，前 40 名 = 98.0% 成交額 / 94.0% 筆數，
              **前 80 名 = 99.6% / 98.7%**，前 150 名之後幾乎全是 0 筆。
              取到 80 是為了讓「逐標的 markout 篩選」有足夠多的標的
              越過 200 筆的門檻（`hl/flow_toxicity.py` 的 `--min-trades`）。

              > **⚠ 2026-09-13：預設從 80 放寬到 250（= 全部 active，現為 217）。**
              > 上面那個量測**仍然正確**，變的是目的。「前 80 名 = 99.6% 的
              > 成交額」對「用最少訂閱抓住大部分流量」是對的理由；而 TODO
              > §1.40 要問的是**哪些市場沒人在掛單**，那件事住在剩下的 0.4%
              > 裡面 —— 對那個問題，成交額佔比是錯的篩選軸。
              > 實測佐證：前 80 名的半價差 BTC 0.03 / ETH 0.20 bps，而長尾
              > 是 20–60 bps；而 rank 81+ 攤到 145 個市場仍有每個 42 筆/天，
              > 是 G1 門檻（10 筆/天）的 4 倍 —— **長尾不是死市場，是看不見。**
              > 代價：磁碟從約 6.6 MB/日 到估約 20 MB/日（D 槽剩 673 GB），
              > 以及**訂閱必須分批**（見 SUB_BATCH，實測 burst 會被關連線）。
    **排除現貨** 現貨與永續的名目語意不同，混進同一張表是
              mistake.md 2026-09-03 那個坑（單位相近但差幾個數量級）。
              hl_tape 當初排除現貨也是同一個理由。
    存放      **D 槽**（`LIGHTER_TAPE_DIR`，預設 D:\\flowbot_data\\lighter\\trades）
              量級：1.29M 筆/日、31 欄 -> 原始約 0.4 GB/日，
              parquet 壓縮後約 60-100 MB/日（HL 的十分之一，因為筆數少 10 倍）
    輪替      UTC 小時，parquet
    去重      **`trade_id`** —— `subscribed/trade` 每次（重）訂閱會重播
              **50 筆**近期成交，所以每次重連都會帶一批舊的
    清算       `liquidation_trades` 與 `trades` 同形，**一起錄並帶 `is_liq`**。
              清算在這裡有明確旗標（HL 沒有，只能事後推定）—— 不要丟掉。
    斷線      自動重連 + 指數退避，旗標帶重連次數
    判準      freshness 看旗標的 `asof` 與 `ok`，**不看行程在不在**
              （mistake.md 2026-07-28）
    單實例    **檔案鎖**。兩個寫入者對同一批 parquet 做 read-modify-write
              會靜默掉列，而且事後算不出掉了多少
              （mistake.md 2026-09-11，我自己起第二個實例踩過）
    啟動旗標  **啟動當下就寫一次 `ok=True`**。看門狗每 5 分鐘看旗標，而第一次
              落盤要 300 秒 —— 不先舉手，剛起來的健康行程會被殺
              （mistake.md 2026-09-11，hl_mid 被殺了 70 次）

===========================================================================
跑法
===========================================================================
    python research/lighter/lighter_tape.py                  常駐
    python research/lighter/lighter_tape.py --seconds 120    跑一段就停（驗收）
    python research/lighter/lighter_tape.py --top 40         縮小宇宙
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAPE_DIR = Path(os.environ.get("LIGHTER_TAPE_DIR",
                               r"D:\flowbot_data\lighter\trades"))
FLAG = ROOT / "research" / "results" / "lighter_tape_last.json"
LOCK = ROOT / "research" / "results" / ".lighter_tape.lock"
WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
API = "https://mainnet.zklighter.elliot.ai"
FLUSH_SEC = 300            # 每 5 分鐘落盤（斷電最多損失 5 分鐘）
# 2026-09-13 從 80 放寬到 250（= 全部 active，現為 217）。理由見檔頭
# 「宇宙」那一段的警示框：80 對「抓住大部分流量」是對的，對「找出沒人
# 掛單的市場」是錯的篩選軸（TODO §1.40）。放寬的前提是訂閱要分批。
TOP_N_DEFAULT = 250

# ── 訂閱節流與自我修復（2026-09-13，TODO §1.40）─────────────────────────
# **把全部訂閱一次 burst 出去會被伺服器關連線。** 實測（`ws_sub_limit.py`，
# 217 個 active 市場、獨立連線）：
#     無間隔 213 ack 撐滿 102 秒 ／ 間隔 20ms 200 ack 撐滿 ／
#     無間隔 **189 ack 在第 7 秒被斷**（WebSocketConnectionClosedException，
#     而且是在還在送訂閱的時候斷的）
# 三次有一次被斷 —— 間歇性，而**失敗方式是連線被關，不是回錯誤訊息**。
# 對常駐錄製器來說那是最糟的形狀：watchdog 重啟 -> 重新訂閱 -> 再被斷
# （mistake.md 2026-09-11 `hl_mid` 被殺 70 次、丟 6 小時的同一個形狀）。
#
# 所以三件事：
#   1. **分批**，批間有間隔，而且**從背景執行緒送** —— 原本是在 `on_msg` 裡
#      同步迴圈，那會擋住自己的訊息泵，讓 burst 變成可能的最快速度。
#   2. **數 ack**。`subscribed/trade` 回來的 channel 是 **`trade:N`**
#      （訂閱時送的是 `trade/N`，分隔符不同 —— 這個差異咬過我一次，
#      用 "/" 切會拿到 0 個 ack 而同時有資料在進來）。
#   3. **缺的自動補訂**，並把 `subs/acks` 寫進旗標，少訂到不會是隱形的。
#      注意 `ok` **刻意不因為缺 ack 而變 False** —— 那會讓 watchdog 殺掉
#      一個正在正常錄的行程，而那正是上面那個 70 次的病。
SUB_BATCH = 20             # 每批幾個訂閱
SUB_PAUSE = 1.0            # 批與批之間（秒）
SUB_REPAIR_SEC = 90        # 每隔這麼久補訂一次沒有 ack 的
SUB_REPAIR_MAX = 5         # 最多補幾輪

COLS = ["ts", "tx_us", "market_id", "coin", "trade_id", "is_liq",
        "px", "sz", "usd", "is_maker_ask",
        "maker_fee", "taker_fee",
        "ask_acct", "bid_acct",
        "taker_pos_before", "maker_pos_before",
        "taker_imf_before", "maker_imf_before",
        "block_height",
        # 2026-09-13：**我們收到那則訊息的本機時刻**（毫秒）。
        # 跨場館 lead-lag 不可以用兩個交易所各自的時戳比——實測偏移 +371 ms，
        # 比效應本身（100-250 ms）還大。而照文章 3c，收到的時刻本來就是
        # 決策該用的那個時鐘。每則訊息取一次，同訊息內的成交共用。
        "rx_ms"]

_buf = []
_accts = set()
_lock = threading.Lock()
_stat = dict(trades=0, liqs=0, reconnects=0, flushes=0, rows_written=0,
             started=time.time(), last_trade_ms=0, markets=0,
             subs=0, repairs=0)
_id2sym = {}
_acked = set()          # 收到 subscribed/trade 的 market_id


def _chan_mid(ch):
    """`trade:231` -> 231。**回來的分隔符是 `:`，送出去的是 `/`。**"""
    s = str(ch or "")
    for sep in (":", "/"):
        if sep in s:
            try:
                return int(s.rsplit(sep, 1)[-1])
            except ValueError:
                return -1
    return -1


def _subscribe_all(ws, mids):
    """分批訂閱，**在背景執行緒跑**，不擋訊息泵。"""
    mids = list(mids)
    for i in range(0, len(mids), SUB_BATCH):
        batch = mids[i:i + SUB_BATCH]
        for mid in batch:
            try:
                ws.send(json.dumps({"type": "subscribe",
                                    "channel": "trade/%d" % mid}))
                _stat["subs"] += 1
            except Exception as e:                          # noqa: BLE001
                print("訂閱中斷於第 %d 個：%r" % (_stat["subs"], e))
                return
        time.sleep(SUB_PAUSE)


def _repair_subs(ws, mids):
    """補訂沒有 ack 的。間歇性的上限靠重試吃掉，而不是靠祈禱。"""
    for _ in range(SUB_REPAIR_MAX):
        time.sleep(SUB_REPAIR_SEC)
        missing = [m for m in mids if m not in _acked]
        if not missing:
            return
        _stat["repairs"] += 1
        print("補訂 %d 個沒有 ack 的市場（第 %d 輪）"
              % (len(missing), _stat["repairs"]))
        _subscribe_all(ws, missing)


def _get(path):
    """一次性的 REST 讀取（只在啟動時取宇宙用）。

    **必須用 aiohttp，不可以用 urllib。** 2026-09-12 實測：這個 CDN 對
    urllib 一律回 **HTTP 405**，而且與標頭無關 —— 四種組合都試過
    （瀏覽器 UA、aiohttp 風格的 UA、完全無標頭、python UA），全部 405；
    同一時刻同一個 URL 用 aiohttp 是 **HTTP 200**。所以差別在 client 層
    （TLS／連線指紋），不在標頭。不要「順手簡化成 urllib」。
    """
    import asyncio

    import aiohttp

    async def _go():
        async with aiohttp.ClientSession() as s:
            async with s.get(API + path,
                             timeout=aiohttp.ClientTimeout(total=30)) as r:
                r.raise_for_status()
                return await r.json()

    return asyncio.run(_go())


def universe(top_n):
    """永續、按日成交額取前 top_n。回傳 {market_id: symbol}。

    **兩個端點都要**：`orderBooks` 有 market_id 與 market_type（現貨要濾掉），
    `exchangeStats` 有日成交額。只用一個會少掉其中一半資訊。
    """
    books = (_get("/api/v1/orderBooks").get("order_books") or [])
    perp = {b["symbol"]: int(b["market_id"]) for b in books
            if b.get("status") == "active" and b.get("market_type") == "perp"}
    stats = (_get("/api/v1/exchangeStats").get("order_book_stats") or [])
    rows = [(x["symbol"], perp[x["symbol"]],
             float(x.get("daily_quote_token_volume") or 0))
            for x in stats if x.get("symbol") in perp]
    rows.sort(key=lambda r: -r[2])
    keep = rows[:top_n]
    tot = sum(r[2] for r in rows) or 1.0
    cov = sum(r[2] for r in keep) / tot
    print("永續 active %d 個；取前 %d = 日成交額的 %.1f%%"
          % (len(rows), len(keep), 100 * cov))
    return {m: s for s, m, _ in keep}


def flush():
    """落盤：按 UTC 小時分檔 append，`trade_id` 去重。"""
    with _lock:
        rows, _buf[:] = list(_buf), []
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=COLS)
        for hr, part in df.groupby(df.ts // 3_600_000):
            d = TAPE_DIR / time.strftime("%Y%m%d", time.gmtime(hr * 3600))
            d.mkdir(parents=True, exist_ok=True)
            p = d / (time.strftime("%H", time.gmtime(hr * 3600)) + ".parquet")
            if p.exists():
                part = pd.concat([pd.read_parquet(p), part], ignore_index=True)
            # 每次（重）訂閱會重播 50 筆近期成交 -> 必須去重，
            # 而且第一次落盤會產生一批舊小時的檔（那是真資料不是 bug）。
            part = part.drop_duplicates(subset=["trade_id", "is_liq"],
                                        keep="first")
            part.to_parquet(p, index=False)
        _stat["rows_written"] += len(rows)
    _stat["flushes"] += 1
    write_flag()


def write_flag(starting=False):
    """寫新鮮度旗標。

    `starting=True` 是**啟動那一瞬間**用的：此刻 trades 必然是 0，但行程是
    健康的（WS 已訂閱）。`ok` 的語意是「連得上且設定對」，不是「已經有資料」
    —— 後者在合法的空狀態下與故障無法區分（mistake.md 2026-09-03）。
    不寫這一份，看門狗會在第一次落盤（300 秒）之前把剛起來的行程殺掉
    （mistake.md 2026-09-11，hl_mid 被殺 70 次、近 6 小時不可回填的資料沒了）。
    """
    up = time.time() - _stat["started"]
    lag = ((time.time() * 1000 - _stat["last_trade_ms"]) / 1000
           if _stat["last_trade_ms"] else None)
    ok = (True if starting
          else bool(_stat["trades"] > 0 and (lag is None or lag < 300)))
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=ok,
        reason=("啟動中（已訂閱，還沒有成交）" if starting else
                "成交 %d 筆（%.1f/秒）、清算 %d、帳戶 %d、市場 %d、重連 %d、"
                "最後一筆 %s 秒前%s"
                % (_stat["trades"], _stat["trades"] / max(up, 1),
                   _stat["liqs"], len(_accts), _stat["markets"],
                   _stat["reconnects"],
                   "—" if lag is None else "%.0f" % lag,
                   # **訂閱缺口要看得見。** 伺服器對一次 burst 會關連線
                   # （實測三次有一次），所以「少訂到幾個市場」是真實的
                   # 失敗模式，而它不會讓任何別的數字變難看。
                   # `ok` 刻意不因此變 False —— 那會讓 watchdog 殺掉一個
                   # 正在正常錄的行程（mistake.md 2026-09-11）。
                   "" if len(_acked) >= _stat["subs"] > 0 else
                   "｜**訂閱 %d 送出 / %d 確認，缺 %d**（補訂 %d 輪）"
                   % (_stat["subs"], len(_acked),
                      _stat["subs"] - len(_acked), _stat["repairs"]))),
        trades=_stat["trades"], liqs=_stat["liqs"], accounts=len(_accts),
        subs=_stat["subs"], acks=len(_acked), sub_repairs=_stat["repairs"],
        markets=_stat["markets"], reconnects=_stat["reconnects"],
        rows_written=_stat["rows_written"], uptime_sec=round(up),
        tape_dir=str(TAPE_DIR),
        asof=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False,
        indent=2), encoding="utf-8")


_LOCK_FH = None          # 故意是模組層的全域：句柄要活到行程結束


def acquire_single_instance():
    """取得獨佔鎖；已經有一個實例在跑就回 False。

    用檔案鎖不用 pid 檔 —— 行程被殺時 OS 自動釋放，不留孤兒鎖。
    """
    global _LOCK_FH
    if _LOCK_FH is not None:
        return True
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK, "a+b")
    try:
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        return False
    except ImportError:                      # 非 Windows
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
    _LOCK_FH = fh
    return True


def _num(d, k):
    v = d.get(k)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row(t, is_liq, rx_ms):
    mid = int(t["market_id"])
    return (int(t["timestamp"]), int(t.get("transaction_time") or 0), mid,
            _id2sym.get(mid), int(t["trade_id"]), bool(is_liq),
            float(t["price"]), float(t["size"]), _num(t, "usd_amount"),
            bool(t.get("is_maker_ask")),
            _num(t, "maker_fee"), _num(t, "taker_fee"),
            t.get("ask_account_id"), t.get("bid_account_id"),
            _num(t, "taker_position_size_before"),
            _num(t, "maker_position_size_before"),
            _num(t, "taker_initial_margin_fraction_before"),
            _num(t, "maker_initial_margin_fraction_before"),
            t.get("block_height"), rx_ms)


def run(seconds=None, top_n=TOP_N_DEFAULT):
    import websocket
    if not acquire_single_instance():
        print("已經有一個 Lighter 成交帶在跑（檔案鎖 %s）。不啟動第二個 —— "
              "兩個寫入者會靜默破壞 parquet。" % LOCK.name)
        return 2

    _id2sym.update(universe(top_n))
    _stat["markets"] = len(_id2sym)
    print("訂閱 %d 個永續市場的成交帶 -> %s" % (len(_id2sym), TAPE_DIR))
    # **先舉手再連線**（理由見 write_flag）
    write_flag(starting=True)
    stop_at = (time.time() + seconds) if seconds else None

    def on_open(ws):
        # **不在這裡訂閱** —— Lighter 要等 {"type":"connected"} 才收訂閱，
        # 太早送會被靜默忽略（arb/engine/entropy_arb/feeds.py 的實測註解）。
        pass

    def on_msg(ws, m):
        rx_ms = int(time.time() * 1000)      # 先取，解析之前
        try:
            d = json.loads(m)
        except Exception:
            return
        t = d.get("type")
        if t == "connected":
            # **分批、在背景執行緒送** —— 原本是在這裡同步迴圈，那會擋住
            # 自己的訊息泵，而一次 burst 全部訂閱會被伺服器關連線（見檔頭
            # SUB_BATCH 那一段的實測）。
            mids = list(_id2sym)
            threading.Thread(target=_subscribe_all, args=(ws, mids),
                             daemon=True).start()
            threading.Thread(target=_repair_subs, args=(ws, mids),
                             daemon=True).start()
            return
        if t == "ping":
            ws.send(json.dumps({"type": "pong"}))
            return
        if t == "subscribed/trade":
            _acked.add(_chan_mid(d.get("channel")))
            # 不 return —— 這則訊息會重播近期成交，下面照收
        if t not in ("subscribed/trade", "update/trade"):
            return
        with _lock:
            for key, is_liq in (("trades", False), ("liquidation_trades", True)):
                for tr in (d.get(key) or []):
                    try:
                        _buf.append(_row(tr, is_liq, rx_ms))
                    except (KeyError, TypeError, ValueError):
                        continue
                    for a in (tr.get("ask_account_id"), tr.get("bid_account_id")):
                        if a is not None:
                            _accts.add(int(a))
                    if is_liq:
                        _stat["liqs"] += 1
                    else:
                        _stat["trades"] += 1
                    _stat["last_trade_ms"] = int(tr["timestamp"])
        if stop_at and time.time() > stop_at:
            ws.close()

    while True:
        ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_msg)
        ws.run_forever(ping_interval=20, ping_timeout=10)
        flush()
        if stop_at and time.time() > stop_at:
            break
        _stat["reconnects"] += 1
        # **新連線要重新訂閱，所以 ack 必須歸零** —— 不清的話旗標會顯示
        # 「訂閱齊了」而實際上新連線上一個都沒訂，那是隱形的資料缺口。
        _acked.clear()
        _stat["subs"] = 0
        print("WS 斷線，第 %d 次重連" % _stat["reconnects"])
        time.sleep(min(30, 2 ** min(_stat["reconnects"], 5)))

    flush()
    up = time.time() - _stat["started"]
    print("停止：成交 %d 筆（%.1f/秒）、清算 %d、寫入 %d 列、帳戶 %d、"
          "重連 %d、%.0f 秒"
          % (_stat["trades"], _stat["trades"] / max(up, 1), _stat["liqs"],
             _stat["rows_written"], len(_accts), _stat["reconnects"], up))
    return 0


def ticker():
    while True:
        time.sleep(FLUSH_SEC)
        try:
            flush()
        except Exception as e:
            print("flush 失敗:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None,
                    help="跑幾秒就停（驗收用；預設常駐）")
    ap.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                    help="錄日成交額前幾名的永續（預設 %d）" % TOP_N_DEFAULT)
    a = ap.parse_args()
    # **先拿鎖再開 ticker**：被拒絕的實例不該有任何會寫檔的執行緒在跑。
    if not acquire_single_instance():
        print("已經有一個 Lighter 成交帶在跑（檔案鎖 %s）。不啟動第二個。"
              % LOCK.name)
        return 2
    if not a.seconds:
        threading.Thread(target=ticker, daemon=True).start()
    # 回傳碼要傳出去：退出碼 0 是「跑完了」不是「做了事」
    # （mistake.md 2026-08-26，那次一個什麼都沒做的指令回了 0）。
    return run(a.seconds, a.top) or 0


if __name__ == "__main__":
    raise SystemExit(main())
