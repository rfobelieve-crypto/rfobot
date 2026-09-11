# -*- coding: utf-8 -*-
"""Hyperliquid 分鐘級中價與佇列（常駐 WS，2026-09-11）

===========================================================================
為什麼要另外錄這個，而不是用成交價
===========================================================================
外部閱讀（`docs/external_reading.md`，HFT Alpha Research 101）：

> 報酬目標**不能用成交價 bar 算，要用 mid**。薄的標的有買賣價跳動，
> 會讓成交價 bar 呈現**比實際強得多的反轉效應**。

這對我們不是理論問題，是**現在就會發生的污染**：

- `hl_tape.py` 錄的是**成交**，所以 HL 的分鐘級報酬只能用成交價算
- 而 §4.65 的結論正是「分鐘級是**反著做**（均值回歸）」——那是在 BTC/ETH 的
  `ohlcv_1m` 上量的（夠厚），**把同一套搬到 HL 長尾，量到的「反轉」會有一部分
  是跳動不是市場**
- `l2Book` 的 REST 快照目前**每小時才一筆**，做不出分鐘級 mid

而且這一段**不可回填**：`candleSnapshot` 只保留 5000 根（1m = 3.5 天），
簿口根本沒有歷史端點。每過一小時就永久少一小時。

===========================================================================
宇宙：為什麼是前 40 名（量出來的，不是拍的）
===========================================================================
2026-09-11 實測主場 178 個有量的市場，日成交額集中度：

    前 5 名   85.2%      前 10 名  92.4%      前 20 名  95.9%
    前 30 名  97.6%      前 50 名  98.9%      前 80 名  99.6%

頂檔價差中位：**前 30 名 2.9 bps、尾 50 名 38.5 bps（13 倍）**。

所以跳動偏誤最嚴重的地方，**正好是沒有量、也交易不了的地方**。
取 **前 40 名**（約 98.3%）：把「我們可能真的交易得到」的市場全部蓋住，
而不為交易不了的長尾付訂閱與磁碟成本。

**連帶的判準含意（Gate 0）**：§1.10 機制關的宇宙應該跟著收到這 40 個。
在交易不了的市場上量到的機制，沒有下一步。

===========================================================================
凍結的決定
===========================================================================
    取樣      **牆鐘固定 60 秒**，不是「有更新就記」。
              我們要的是一個規則網格去算報酬；事件驅動的取樣會讓
              報酬區間長度隨活躍度變動，而活躍度跟報酬相關。
    存什麼    最佳買賣（px/sz/**n**）、mid、價差 bps、前 5 檔的 sz 與 n 合計。
              **`n` 是每檔的掛單筆數** —— CEX 公開簿口沒有這一欄，
              它讓佇列位置從估計變成數得出來（§1.19 HFT 的核心輸入）。
    book_time HL 自己在訊息裡給的 `time` 一起存 —— 用來查我們取樣時
              那本簿口有多舊。**不存它就無法事後分辨「市場沒動」與
              「我們的連線斷了」。**
    落盤      UTC 小時 parquet，放 D 槽（與 tape 同一顆磁碟）
    單一實例  檔案鎖。兩個寫入者對同一批 parquet 做 read-modify-write
              會靜默掉列，而且事後算不出掉多少（2026-09-11 已經踩過一次）
    判準      freshness 看旗標的 asof 與 ok，不看行程在不在

    python research/hl/hl_mid.py                 常駐
    python research/hl/hl_mid.py --seconds 120   跑一段就停（驗收用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MID_DIR = Path(os.environ.get("HL_MID_DIR", r"D:\flowbot_data\hl\mid"))
FLAG = ROOT / "research" / "results" / "hl_mid_last.json"
WS_URL = "wss://api.hyperliquid.xyz/ws"
INFO_URL = "https://api.hyperliquid.xyz/info"

# 2026-09-11 從 40 加寬到 100（使用者同意）。原因是 §1.23b：那個 1 小時橫斷面
# alpha 的綁束**不是訊號是橫斷面的寬度** —— 11 個名字的 z 分數讓 SE 變成
# 2.2 bps/h，而效應量級是 0.x。他用 50 個，我們現在有 100 個的空間。
#
# 量出來的覆蓋（2026-09-11 實查 `metaAndAssetCtxs`，主場 178 個活躍永續、
# 日成交額合計 $56.3 億）：
#     前 40  -> 98.53%   第 40 名 VIRTUAL $3.5M/天
#     前 100 -> 99.82%   第 100 名 ZK     $331k/天
#     全 178 -> 100%     最後一名 TNSR    $9k/天
# **多拿的 60 個幣合起來只有 1.3% 的成交額。** 所以加寬買的不是「更多量」，
# 是**橫斷面的寬度**——那是 SE 的來源。
#
# 兩個要記住的代價：
#  1. 薄的標的有買賣價跳動，而那會偽裝成均值回歸（mistake.md 2026-09-11）。
#     **這支錄的是 mid 不是成交價**，所以錄是安全的；但用它做分析時
#     報酬目標一律用 mid，不可以換回成交價。
#  2. **能不能交易是另一關。** $331k/天的名字頂檔可能只有 $100，
#     所以就算 alpha 成立，容量可能是零（§1.20 的教訓：容量是獨立的一關）。
#     加寬是為了**量得準**，不是假設它們可交易。
# 一條 WS 吃得下：2026-09-11 實測訂 234 個 l2Book -> 234 個全部回資料。
TOP_N = 100
SAMPLE_SEC = 60         # 牆鐘固定取樣
FLUSH_SEC = 300         # 每 5 分鐘落盤
DEPTH_LEVELS = 5        # 合計前幾檔的 sz / n
BANDS = (1, 2, 5, 10, 25, 50)     # 距 mid 幾 bps 內的累計深度
OLD_LEVEL_USD = 100.0             # 「上一快照這一檔有超過這個金額」-> 算舊單
                                  # 這個門檻抄自外部來源（$100），**未經我們驗證**；
                                  # 它是代理的參數，之後要做敏感度

_books = {}             # coin -> (book_time_ms, bids, asks)
_prev = {}              # coin -> {("b"|"a", px_str): usd}  上一次取樣的逐檔
                        # **新舊拆分只能在錄的當下算**：它要比對上一個快照，
                        # 而落盤之後那個狀態就不存在了
_rows = []
_lock = threading.Lock()
_stat = dict(msgs=0, samples=0, rows=0, reconnects=0, started=time.time(),
             last_msg=0.0)
_LOCK_FH = None


def acquire_single_instance():
    """檔案鎖。行程被殺時 OS 自動釋放，不會留下指向已死 pid 的孤兒鎖。"""
    global _LOCK_FH
    if _LOCK_FH is not None:
        return True
    lk = ROOT / "research" / "results" / ".hl_mid.lock"
    lk.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lk, "a+b")
    try:
        import msvcrt
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        return False
    except ImportError:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
    _LOCK_FH = fh
    return True


def post(body):
    r = urllib.request.Request(INFO_URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=25).read())


def top_coins(n=TOP_N):
    """主場永續，按日成交額取前 n。**排除已下市**（它們沒有活簿口）。"""
    meta, ctxs = post({"type": "metaAndAssetCtxs"})
    out = []
    for u, c in zip(meta["universe"], ctxs):
        if u.get("isDelisted"):
            continue
        try:
            out.append((float(c.get("dayNtlVlm") or 0), u["name"]))
        except Exception:
            pass
    out.sort(reverse=True)
    return [name for _, name in out[:n]]


def sample_once():
    """把當下每個幣的簿口壓成一列。**只讀記憶體，不發請求。**"""
    ts = int(time.time() * 1000)
    got = 0
    with _lock:
        snap = dict(_books)
    rows = []
    for coin, (btime, bids, asks) in snap.items():
        if not bids or not asks:
            continue
        try:
            bpx, apx = float(bids[0]["px"]), float(asks[0]["px"])
            if not (bpx > 0 and apx > bpx):
                continue          # 交叉或空簿口：不寫，寧可少一列
            mid = (bpx + apx) / 2.0
            pm = _prev.get(coin, {})
            bd, bn_, bnew5, bold5, bnew10, bold10 = _bands(bids, mid, "b", pm)
            ad, an_, anew5, aold5, anew10, aold10 = _bands(asks, mid, "a", pm)
            rows.append((
                ts, coin, btime, bpx, apx, mid,
                round(1e4 * (apx - bpx) / mid, 4),
                float(bids[0]["sz"]), float(asks[0]["sz"]),
                int(bids[0].get("n") or 0), int(asks[0].get("n") or 0),
                sum(float(x["sz"]) for x in bids[:DEPTH_LEVELS]),
                sum(float(x["sz"]) for x in asks[:DEPTH_LEVELS]),
                sum(int(x.get("n") or 0) for x in bids[:DEPTH_LEVELS]),
                sum(int(x.get("n") or 0) for x in asks[:DEPTH_LEVELS]),
                *[bd[k] for k in BANDS], *[ad[k] for k in BANDS],
                *[bn_[k] for k in BANDS], *[an_[k] for k in BANDS],
                bnew5, bold5, anew5, aold5, bnew10, bold10, anew10, aold10,
            ))
            # 這一輪的逐檔金額留給下一輪當「舊」的判準
            nxt = {}
            for lv in bids:
                try:
                    nxt[("b", lv["px"])] = float(lv["px"]) * float(lv["sz"])
                except Exception:
                    pass
            for lv in asks:
                try:
                    nxt[("a", lv["px"])] = float(lv["px"]) * float(lv["sz"])
                except Exception:
                    pass
            _prev[coin] = nxt
            got += 1
        except Exception:
            continue
    with _lock:
        _rows.extend(rows)
        _stat["samples"] += 1
        _stat["rows"] += len(rows)
    return got


def _bands(levels, mid, side, prev_map):
    """回傳 (各帶 size, 各帶筆數, 5bps 新/舊, 10bps 新/舊)。

    「新/舊」用上一快照同一個價位有沒有超過 OLD_LEVEL_USD 當代理 ——
    掛單年齡看不到，這是外部來源提出的替代品（見本檔頭）。
    """
    dsz = {b_: 0.0 for b_ in BANDS}
    dn = {b_: 0 for b_ in BANDS}
    new5 = old5 = new10 = old10 = 0.0
    for lv in levels:
        try:
            px, sz, n = float(lv["px"]), float(lv["sz"]), int(lv.get("n") or 0)
        except Exception:
            continue
        if px <= 0 or mid <= 0:
            continue
        dist = abs(px - mid) / mid * 1e4          # 距 mid 幾 bps
        for b_ in BANDS:
            if dist <= b_:
                dsz[b_] += sz
                dn[b_] += n
        usd = px * sz
        was_old = prev_map.get((side, lv["px"]), 0.0) > OLD_LEVEL_USD
        if dist <= 5:
            if was_old:
                old5 += usd
            else:
                new5 += usd
        if dist <= 10:
            if was_old:
                old10 += usd
            else:
                new10 += usd
    return dsz, dn, new5, old5, new10, old10


COLS = (["ts", "coin", "book_time", "bid", "ask", "mid", "spread_bps",
         "bid_sz", "ask_sz", "bid_n", "ask_n",
         "bid_sz5", "ask_sz5", "bid_n5", "ask_n5"]
        + ["bid_d%d" % b_ for b_ in BANDS] + ["ask_d%d" % b_ for b_ in BANDS]
        + ["bid_nb%d" % b_ for b_ in BANDS] + ["ask_nb%d" % b_ for b_ in BANDS]
        + ["bid_new5", "bid_old5", "ask_new5", "ask_old5",
           "bid_new10", "bid_old10", "ask_new10", "ask_old10"])


def flush():
    with _lock:
        rows, _rows[:] = list(_rows), []
    if rows:
        import pandas as pd
        d = pd.DataFrame(rows, columns=COLS)
        for hr, part in d.groupby(d.ts // 3_600_000):
            dd = MID_DIR / time.strftime("%Y%m%d", time.gmtime(hr * 3600))
            dd.mkdir(parents=True, exist_ok=True)
            p = dd / (time.strftime("%H", time.gmtime(hr * 3600)) + ".parquet")
            if p.exists():
                part = pd.concat([pd.read_parquet(p), part], ignore_index=True)
            part = part.drop_duplicates(subset=["ts", "coin"], keep="last")
            part.to_parquet(p, index=False)
    write_flag()


def write_flag(starting=False):
    """啟動時也要寫一次，否則看門狗會在第一次落盤前把新行程殺掉
    （2026-09-11 hl_tape 實際發生過）。"""
    up = time.time() - _stat["started"]
    lag = (time.time() - _stat["last_msg"]) if _stat["last_msg"] else None
    ok = (True if starting
          else bool(_stat["samples"] > 0 and (lag is None or lag < 120)))
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=ok,
        reason=("啟動中（已訂閱，還沒取樣）" if starting else
                "取樣 %d 次、寫 %d 列、訊息 %d、重連 %d、最後訊息 %s 秒前"
                % (_stat["samples"], _stat["rows"], _stat["msgs"],
                   _stat["reconnects"],
                   "—" if lag is None else "%.0f" % lag)),
        coins=len(_books), samples=_stat["samples"], rows=_stat["rows"],
        reconnects=_stat["reconnects"], uptime_sec=round(up),
        mid_dir=str(MID_DIR), top_n=TOP_N, sample_sec=SAMPLE_SEC,
        asof=time.strftime("%Y-%m-%d %H:%M:%S")),
        ensure_ascii=False, indent=2), encoding="utf-8")


def sampler(stop_at=None):
    """**牆鐘對齊**到整分鐘取樣，不是 sleep(60) 漂移。"""
    while True:
        now = time.time()
        time.sleep(max(0.1, SAMPLE_SEC - (now % SAMPLE_SEC)))
        try:
            sample_once()
        except Exception as e:
            print("取樣失敗:", e)
        if stop_at and time.time() > stop_at:
            return


def ticker():
    while True:
        time.sleep(FLUSH_SEC)
        try:
            flush()
        except Exception as e:
            print("flush 失敗:", e)


def run(seconds=None):
    import websocket
    coins = top_coins()
    print("訂閱 %d 個幣的 l2Book -> %s（每 %d 秒取樣）"
          % (len(coins), MID_DIR, SAMPLE_SEC))
    print("前 10:", coins[:10])
    write_flag(starting=True)
    stop_at = (time.time() + seconds) if seconds else None

    def on_open(ws):
        for c in coins:
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": "l2Book", "coin": c}}))

    def on_msg(ws, m):
        try:
            d = json.loads(m)
        except Exception:
            return
        if d.get("channel") != "l2Book":
            return
        x = d.get("data") or {}
        lv = x.get("levels") or []
        if len(lv) < 2:
            return
        with _lock:
            _books[x.get("coin")] = (int(x.get("time") or 0), lv[0], lv[1])
            _stat["msgs"] += 1
            _stat["last_msg"] = time.time()
        if stop_at and time.time() > stop_at:
            ws.close()

    threading.Thread(target=sampler, args=(stop_at,), daemon=True).start()
    while True:
        ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_msg)
        ws.run_forever(ping_interval=20, ping_timeout=10)
        flush()
        if stop_at and time.time() > stop_at:
            break
        _stat["reconnects"] += 1
        print("WS 斷線，第 %d 次重連" % _stat["reconnects"])
        time.sleep(min(30, 2 ** min(_stat["reconnects"], 5)))

    flush()
    print("停止：取樣 %d 次、寫 %d 列、%d 個幣、重連 %d"
          % (_stat["samples"], _stat["rows"], len(_books), _stat["reconnects"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not acquire_single_instance():
        print("已經有一個中價錄製器在跑（research/results/.hl_mid.lock）。"
              "不啟動第二個 —— 兩個寫入者會靜默破壞 parquet。")
        return 2
    if not a.seconds:
        threading.Thread(target=ticker, daemon=True).start()
    return run(a.seconds) or 0


if __name__ == "__main__":
    raise SystemExit(main())
