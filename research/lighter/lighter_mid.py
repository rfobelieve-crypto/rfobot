# -*- coding: utf-8 -*-
"""Lighter 分鐘級中價與深度（常駐 WS，2026-09-12）

===========================================================================
為什麼要另外錄中價，而不是拿成交帶的成交價算
===========================================================================
**報酬目標用成交價算會被買賣價跳動污染，薄的標的會呈現比實際強得多的
反轉效應**（外部閱讀 HFT Alpha Research 101；mistake.md 2026-09-11 全文）。
成交價在薄簿口上是一個在買賣價之間跳的東西 —— 用它算報酬，相鄰兩期的
誤差**負相關**，那在統計上長得跟均值回歸一模一樣。而「反轉」正好是分鐘級
最常見的發現，所以這個偏誤會**確認你的假說**，那是最危險的方向。

`lighter_tape.py` 錄的是成交價。**做市 markout 必須用中價**，所以這一支
是它的必要配套，不是選配。

**現有的替代品不夠**：`../arb/engine/logs/*/minutes.csv` 確實有 Lighter
的逐分鐘頂檔，但**只有 9 個 symbol**（§0.75 家族），而 tape 的宇宙是 80 個。

===========================================================================
Lighter 與 HL 的能力差異 —— 兩個方向都有，都要寫下來
===========================================================================
**Lighter 沒有的：每檔掛單筆數。** `hl_mid.py` 有 `bid_n / ask_n / bid_nb*`
以及 new/old 單的代理，因為 HL 的 L2 每一檔帶 `n`（掛單筆數）。
**Lighter 的 L2 每一檔只有 `price` 與 `size`** —— 實測（2026-09-12
`scratchpad/probe_lighter_ws.py`）檔位欄位就是 `['price', 'size']`，沒有第三欄。

所以：**佇列位置那一族研究在 Lighter 上做不了**，而 hl_mid 上做得了。
本檔提供 `bid_levels / ask_levels`（**價格檔數**）當最接近的東西，
但**它是檔數不是筆數，兩者不可互換** —— 一檔可以是 1 筆也可以是 50 筆。
任何從 hl_mid 搬過來的分析，只要用到 `_n` 欄位，在這裡就要停下來重新設計。

**Lighter 多的：整本簿口。** 訂閱快照實測 BTC 是 **549 檔買 / 350 檔賣**，
而 HL 的 L2 端點只給 20 檔。所以深度帶可以拉到 100 bps 還是實數，
不像 HL 到 50 bps 就撞到端點上限。

===========================================================================
凍結的設計決定
===========================================================================
    宇宙      與 `lighter_tape.py` **同一組**（永續、日成交額前 `TOP_N`，
              2026-09-13 起預設 250 = 全部 active；原為 80，理由與放寬
              的理由都在 lighter_tape 檔頭的警示框）。
              刻意相同：兩份資料要能逐 coin join，宇宙不同就會在 join 上
              安靜地掉標的。
    取樣      **牆鐘對齊 5 秒**（`LIGHTER_MID_SAMPLE_SEC` 可覆寫）。
              hl_mid 是 60 秒；這裡縮到 5 秒的理由是外部量到的最強簿口
              alpha（obi_1bp 等）的 IC 對 5 秒報酬 0.126–0.139、對 15 秒
              0.107 —— 衰減很陡。**60 秒等於把它丟掉，而且錄不回來。**
    成本      實測（`scratchpad/probe_lighter_book_rate.py`，80 市場 75 秒）：
              **208 訊息/秒、6.09 MB/分 = 8.8 GB/日頻寬、CPU 4.2%**。
              **落盤每市場每 5 秒一列**（約 1.4M 列/日）。深度計算已
              向量化（numpy 累積和 + searchsorted 取代逐帶掃全簿，
              與迴圈版逐位元等價，800 組對照誤差 0.0），所以 12 倍的
              取樣頻率不是 12 倍的 CPU。
    簿口      snapshot + diff，**`begin_nonce` 斷裂就清簿並重訂閱** ——
              少一個檔位更新之後的簿口是虛構的，寧可空著也不要報假的頂檔。
              斷裂次數逐市場累計並寫進每一列（`resyncs`）。
    `stale_ms` 每一列都記 `ts - book_time`。**沒有它就分不出「市場沒動」與
              「我們斷線」** —— 那是 hl_mid 檔頭記下來的同一條。
    存放      **D 槽**（`LIGHTER_MID_DIR`，預設 D:\\flowbot_data\\lighter\\mid）
    輪替      UTC 小時，parquet
    單實例    檔案鎖（兩個寫入者對同一批 parquet 做 read-modify-write 會
              靜默掉列，mistake.md 2026-09-11）
    啟動旗標  **啟動當下就寫一次 `ok=True`** —— 不寫的話看門狗會在第一次
              落盤（300 秒）前把剛起來的健康行程殺掉（hl_mid 被殺 70 次）
    自曝關    每次取樣都檢查 **best_bid < best_ask**；交錯就不寫那一列並計數，
              旗標帶 `crossed`。交錯代表簿口維護壞了，而它**不會拋例外**。

===========================================================================
跑法
===========================================================================
    python research/lighter/lighter_mid.py                  常駐
    python research/lighter/lighter_mid.py --seconds 60     跑一段就停（驗收）
    LIGHTER_MID_SAMPLE_SEC=60 python research/lighter/lighter_mid.py   回到 60 秒
    python research/lighter/lighter_mid.py --top 40         縮小宇宙
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MID_DIR = Path(os.environ.get("LIGHTER_MID_DIR",
                              r"D:\flowbot_data\lighter\mid"))
FLAG = ROOT / "research" / "results" / "lighter_mid_last.json"
LOCK = ROOT / "research" / "results" / ".lighter_mid.lock"
WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
API = "https://mainnet.zklighter.elliot.ai"
# **5 秒不是 60 秒**（2026-09-12 改）：外部量到的最強簿口 alpha
# （obi_1bp / best_size_imbalance / obi_5bp / obi_10bp）的 IC 對 5 秒報酬是
# 0.126–0.139，對 15 秒掉到 0.107 —— 衰減很陡，60 秒取樣等於把它丟掉。
# 而簿口沒有歷史端點，**錄得不夠細是不可回填的**（mistake.md 2026-09-11）。
SAMPLE_SEC = int(os.environ.get("LIGHTER_MID_SAMPLE_SEC", "5"))
# ── 頂檔事件錄製（2026-09-13）────────────────────────────────────────
# TODO §1.37 的 L3/L4（文章 3b 的成交額主張）**用逐筆帶永遠答不了**：
# 250 ms 一格之下只有 0.8% 的格子兩邊都有成交。只有簿口答得了，因為
# **買賣報價不需要有人成交就會變**。
#
# 成本是量出來的不是估的（`research/ops/tob_event_rate.py`，180 秒、33 個
# 共同標的）：每則訊息都記 143.2/秒（618 MB/日）、頂檔價或量變了 59.0/秒、
# **只有頂檔價變了 57.0/秒 = 246 MB/日**。我原本估 1-2 GB/日，**高估 4-8 倍**。
#
# 而邊際成本接近零：這支**本來就**在每一則訊息上維護整本簿口（snapshot+diff），
# 我們只是把已經算出來的頂檔丟掉而已。文章 `data-pre-processing-guide` 的
# Reduction 那節正是這個處方：「remove new quotes where the data that is
# relevant to us has not changed」「we drop the duplicate midprices」。
#
# 只看**價**不看量：量變了而價沒變的那 60%（143.2 -> 57.0）對 lead-lag 與
# 中價報酬沒有資訊，而它們佔了大半的位元組。
TOB_ON = os.environ.get("LIGHTER_TOB", "1") == "1"
TOB_DIR = Path(os.environ.get("LIGHTER_TOB_DIR",
                              r"D:\flowbot_data\lighter\tob"))
TOB_COLS = ["rx_ms", "book_time", "coin", "market_id",
            "bid", "ask", "bid_sz", "ask_sz", "nonce"]
FLUSH_SEC = 300
DEPTH_LEVELS = 5
BANDS = (1, 2, 5, 10, 25, 50, 100)     # 距 mid 幾 bps 內的累計名目
# 2026-09-13 從 80 放寬到 250（= 全部 active，現為 217）。tob 事件是
# TODO §1.40 算 markout 的 mid 來源，而長尾的半價差才是那條線的標的
# （前 80 名 BTC 0.03 / ETH 0.20 bps，長尾 20-60）。前提是訂閱分批。
TOP_N_DEFAULT = 250

def _to_ms(v):
    """把場館的時間戳正規化成毫秒。**不假設單位。**

    實測（2026-09-12）`last_updated_at` 是**微秒**：ts(ms) 減它得到
    −1.787e15，而 1.787e15 us = 2026 年。第一版寫死當毫秒，於是
    `stale_ms` 是垃圾 —— mistake.md 2026-04-12 那條的第三次
    （同一個 API provider 的不同端點可以有不同單位）。

    門檻用量級判斷：秒 ~1.8e9、毫秒 ~1.8e12、微秒 ~1.8e15、奈秒 ~1.8e18。
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0
    if x <= 0:
        return 0
    if x > 1e17:            # 奈秒
        return int(x / 1e6)
    if x > 1e14:            # 微秒
        return int(x / 1e3)
    if x > 1e11:            # 毫秒
        return int(x)
    return int(x * 1000)    # 秒


COLS = (["ts", "bucket_ts", "coin", "market_id", "book_time", "stale_ms",
         "bid", "ask", "mid", "spread_bps",
         "bid_sz", "ask_sz", "bid_sz5", "ask_sz5",
         "bid_levels", "ask_levels", "nonce", "resyncs"]
        + ["bid_d%d" % b for b in BANDS] + ["ask_d%d" % b for b in BANDS])

_bids: dict[int, dict] = {}
_asks: dict[int, dict] = {}
_nonce: dict[int, int | None] = {}
_btime: dict[int, int] = {}
_resync: dict[int, int] = {}
_id2sym: dict[int, str] = {}
_rows: list = []
_tob: list = []                       # 頂檔事件緩衝
_tob_last: dict[int, tuple] = {}      # market_id -> (bid, ask) 上次寫的價
_stat = dict(samples=0, rows=0, flushes=0, reconnects=0, crossed=0,
             gaps=0, msgs=0, started=time.time(), last_sample=0, markets=0,
             tob_rows=0, tob_written=0)


def _chan_id(channel: str):
    """頻道回音的實際格式是 `order_book:181`（**冒號**），而訂閱送的是
    `order_book/181`。兩種都認 —— 抄 arb/engine/entropy_arb/feeds.py 的
    `_chan_id`，它已經踩過這個。"""
    for sep in (":", "/"):
        if sep in channel:
            try:
                return int(channel.rsplit(sep, 1)[1])
            except ValueError:
                return None
    return None


async def universe(top_n: int):
    """永續、按日成交額前 top_n。與 lighter_tape.universe 同一套規則。

    **必須用 aiohttp 不可以用 urllib**：這個 CDN 對 urllib 一律回 HTTP 405，
    與標頭無關（2026-09-12 四種組合實測）。
    """
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(API + "/api/v1/orderBooks",
                         timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            books = (await r.json()).get("order_books") or []
        async with s.get(API + "/api/v1/exchangeStats",
                         timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            stats = (await r.json()).get("order_book_stats") or []
    perp = {b["symbol"]: int(b["market_id"]) for b in books
            if b.get("status") == "active" and b.get("market_type") == "perp"}
    rows = [(x["symbol"], perp[x["symbol"]],
             float(x.get("daily_quote_token_volume") or 0))
            for x in stats if x.get("symbol") in perp]
    rows.sort(key=lambda r: -r[2])
    keep = rows[:top_n]
    tot = sum(r[2] for r in rows) or 1.0
    print("永續 active %d 個；取前 %d = 日成交額的 %.1f%%"
          % (len(rows), len(keep), 100 * sum(r[2] for r in keep) / tot))
    return {m: s for s, m, _ in keep}


def _apply(mid: int, ob: dict, snapshot: bool) -> None:
    b, a = _bids.setdefault(mid, {}), _asks.setdefault(mid, {})
    if snapshot:
        b.clear()
        a.clear()
    for side, d in (("bids", b), ("asks", a)):
        for lv in (ob.get(side) or []):
            try:
                px, sz = float(lv["price"]), float(lv["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if sz <= 0:
                d.pop(px, None)      # size 0 = 這一檔被清掉
            else:
                d[px] = sz


def _band_depth(levels: dict, mid_px: float, side: str):
    """距 mid 各 bps 帶內的**累計名目 USD**（price x size 相加）。

    **向量化，不是逐帶掃全簿**（2026-09-12 改）：取樣週期從 60 秒縮到 5 秒
    之後，原本「每個帶掃一遍全部檔位」的寫法是 80 市場 x ~900 檔 x 7 帶
    x 2 側 ≈ 每次取樣 100 萬次比較，乘 12 倍就吃掉可觀的 CPU。
    改成每次取樣每個市場只轉一次 numpy 陣列，再用累積和 + searchsorted：
    O(n log n) 一次，取代 O(n x 帶數)。
    """
    if not levels:
        return [0.0] * len(BANDS)
    px = np.fromiter(levels.keys(), dtype=float, count=len(levels))
    sz = np.fromiter(levels.values(), dtype=float, count=len(levels))
    ntl = px * sz
    if side == "bid":
        # 買側：價格高的先算。降冪排序後，帶越寬納入越多前綴。
        o = np.argsort(-px)
        p, c = px[o], np.cumsum(ntl[o])
        # 每個帶的下限價；找出 >= 下限 的檔位數（p 是降冪，用 -p 找）
        lims = mid_px * (1.0 - np.asarray(BANDS, dtype=float) / 1e4)
        k = np.searchsorted(-p, -lims, side="right")
    else:
        o = np.argsort(px)
        p, c = px[o], np.cumsum(ntl[o])
        lims = mid_px * (1.0 + np.asarray(BANDS, dtype=float) / 1e4)
        k = np.searchsorted(p, lims, side="right")
    return [float(c[i - 1]) if i > 0 else 0.0 for i in k]


def sample_once() -> int:
    """把當下每個市場的簿口壓成一列。**只讀記憶體，不發請求。**"""
    ts = int(time.time() * 1000)
    # 四捨五入到最近的取樣週期：早醒 0.2 秒不該讓這一筆掉到前一分鐘
    period_ms = int(round(ts / (SAMPLE_SEC * 1000.0))) * SAMPLE_SEC * 1000
    got = 0
    for mid, sym in _id2sym.items():
        b, a = _bids.get(mid) or {}, _asks.get(mid) or {}
        if not b or not a:
            continue                      # 還沒有快照，或 nonce 斷裂後清空了
        bb, ba = max(b), min(a)
        if bb >= ba:
            # 自曝關：簿口交錯 = 維護壞了。**不寫這一列**，計數讓它現形。
            _stat["crossed"] += 1
            continue
        m = (bb + ba) / 2.0
        bsz, asz = b[bb], a[ba]
        b5 = sum(b[p] for p in sorted(b, reverse=True)[:DEPTH_LEVELS])
        a5 = sum(a[p] for p in sorted(a)[:DEPTH_LEVELS])
        bt = _to_ms(_btime.get(mid) or 0)
        # `bucket_ts` 是**明確的桶鍵**，不靠 `ts // 週期`：取樣可能落在
        # 邊界前幾十毫秒（計時器精度），那時整除會把它分到前一格。
        # 桶鍵由週期序號四捨五入算，與抖動無關。
        # **名字是 `bucket_ts` 不是 `minute_ts`** —— 週期現在是 5 秒，
        # 叫 minute 會害下一個人（mistake.md 2026-08-26：詞比結論活得久）。
        _rows.append([ts, period_ms, sym, mid, bt, (ts - bt) if bt else None,
                      bb, ba, m, (ba - bb) / m * 1e4,
                      bsz, asz, b5, a5,
                      len(b), len(a), _nonce.get(mid), _resync.get(mid, 0)]
                     + _band_depth(b, m, "bid") + _band_depth(a, m, "ask"))
        got += 1
    _stat["samples"] += 1
    _stat["last_sample"] = ts
    return got


def tob_capture(mid: int) -> None:
    """頂檔**價**變了就記一列。時鐘用 `rx_ms`（我們自己的），不用交易所的。

    2026-09-13 實測：兩個交易所的時戳無法互比（偏移 +168 ~ +371 ms，
    而跨場館 lead-lag 的效應本身只有 100-270 ms）。而照
    `small-trader-alpha-6` Part 3c，收到的時刻本來就是**決策該用的**那個：
    「the message that is received first is the most accurate price」。
    """
    if not TOB_ON:
        return
    b, a = _bids.get(mid), _asks.get(mid)
    if not b or not a:
        return
    bb, aa = max(b), min(a)
    if bb >= aa:
        return                      # 交錯的簿口不寫（同取樣那側的自曝關）
    if _tob_last.get(mid) == (bb, aa):
        return                      # 價沒變 -> 不是事件
    _tob_last[mid] = (bb, aa)
    # **`_to_ms` 不是可選的**：Lighter 的 `last_updated_at` 是**微秒**，
    # 而 `rx_ms` 是毫秒。第一版直接存原始值，於是 rx_ms − book_time 的中位
    # 是 −1.787e15 —— 跟這個專案今天稍早在 stale_ms 上踩到的**同一個數字**。
    # 取樣那一側早就走 `_to_ms`（第 277 行），我寫新路徑時沒沿用。
    _tob.append((int(time.time() * 1000), _to_ms(_btime.get(mid) or 0),
                 _id2sym.get(mid), mid, bb, aa, b[bb], a[aa],
                 _nonce.get(mid)))
    _stat["tob_rows"] += 1


def flush() -> None:
    rows, _rows[:] = list(_rows), []
    if rows:
        import pandas as pd
        d = pd.DataFrame(rows, columns=COLS)
        for hr, part in d.groupby(d.bucket_ts // 3_600_000):
            dd = MID_DIR / time.strftime("%Y%m%d", time.gmtime(hr * 3600))
            dd.mkdir(parents=True, exist_ok=True)
            p = dd / (time.strftime("%H", time.gmtime(hr * 3600)) + ".parquet")
            if p.exists():
                part = pd.concat([pd.read_parquet(p), part], ignore_index=True)
            part = part.drop_duplicates(subset=["bucket_ts", "coin"],
                                        keep="last")
            part.to_parquet(p, index=False)
        _stat["rows"] += len(rows)

    tob, _tob[:] = list(_tob), []
    if tob:
        import pandas as pd
        d = pd.DataFrame(tob, columns=TOB_COLS)
        for hr, part in d.groupby(d.rx_ms // 3_600_000):
            dd = TOB_DIR / time.strftime("%Y%m%d", time.gmtime(hr * 3600))
            dd.mkdir(parents=True, exist_ok=True)
            p = dd / (time.strftime("%H", time.gmtime(hr * 3600)) + ".parquet")
            if p.exists():
                part = pd.concat([pd.read_parquet(p), part], ignore_index=True)
            # 只去掉**完全相同**的列（同毫秒同市場同價）。不要用
            # (rx_ms, market_id) 當鍵 —— 同一毫秒內的兩個不同頂檔是兩個事件。
            part = part.drop_duplicates(
                subset=["rx_ms", "market_id", "bid", "ask"], keep="first")
            part.to_parquet(p, index=False)
        _stat["tob_written"] += len(tob)

    _stat["flushes"] += 1
    write_flag()


def write_flag(starting: bool = False) -> None:
    """寫新鮮度旗標。`starting=True` 的理由見 lighter_tape.write_flag ——
    看門狗每 5 分鐘看旗標，而第一次落盤要 300 秒。"""
    up = time.time() - _stat["started"]
    lag = ((time.time() * 1000 - _stat["last_sample"]) / 1000
           if _stat["last_sample"] else None)
    live = sum(1 for m in _id2sym if (_bids.get(m) and _asks.get(m)))
    # **`ok` 的語意是「連得上、簿口在、取樣在跑」，不是「已經落盤」**
    # （mistake.md 2026-09-03）。用 `rows > 0` 會讓第一次落盤（300 秒）之前
    # 的每一次刷旗標都判紅，而那 5 分鐘是完全健康的。
    # 落盤本身另外看 `rows_written`，它是報表數字不是存活判準。
    ok = (True if starting
          else bool(_stat["samples"] > 0 and live > 0
                    and (lag is None or lag < 3 * SAMPLE_SEC)))
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=ok,
        reason=("啟動中（已訂閱，還沒取樣）" if starting else
                "取樣 %d 次、寫入 %d 列、有簿口的市場 %d/%d、交錯 %d、"
                "nonce 斷 %d、重連 %d、最後取樣 %s 秒前"
                % (_stat["samples"], _stat["rows"], live, len(_id2sym),
                   _stat["crossed"], _stat["gaps"], _stat["reconnects"],
                   "—" if lag is None else "%.0f" % lag)),
        samples=_stat["samples"], rows_written=_stat["rows"],
        markets_live=live, markets=len(_id2sym),
        crossed=_stat["crossed"], nonce_gaps=_stat["gaps"],
        reconnects=_stat["reconnects"], msgs=_stat["msgs"],
        uptime_sec=round(up), mid_dir=str(MID_DIR),
        sample_sec=SAMPLE_SEC,
        asof=time.strftime("%Y-%m-%d %H:%M:%S")),
        ensure_ascii=False, indent=2), encoding="utf-8")


_LOCK_FH = None


def acquire_single_instance() -> bool:
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
    except ImportError:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
    _LOCK_FH = fh
    return True


async def sampler(stop: asyncio.Event) -> None:
    """**牆鐘對齊**到整分鐘取樣，不是 sleep(60) 漂移。

    **為什麼要記週期序號（2026-09-12 驗收時抓到）**：光靠
    `sleep(SAMPLE_SEC - now % SAMPLE_SEC)` 會在邊界附近開火兩次 ——
    Windows 上 `asyncio.sleep` 的計時器精度讓它可能早醒（`now % 60` = 59.99），
    於是下一次算出的睡眠是 0.01 秒、被下限夾成 0.1 秒，1 秒後又開一次。
    實測 200 秒就出現：22:48:00 / 22:48:59 / 22:49:00 三筆，
    而去重是 `(ts, coin)`、兩筆 ts 差 1 秒 -> **兩筆都留下來**，
    那個分鐘就有兩個樣本。

    `hl_mid.py` 用同一個算式但沒有這個病（167,740 列 0 重複）——
    它用的是執行緒裡的 `time.sleep`，精度行為不同。
    **所以這裡不是抄它，是修得比它硬**：週期序號讓重複在構造上不可能，
    不依賴計時器精度。
    """
    last_period = None
    while not stop.is_set():
        d = SAMPLE_SEC - (time.time() % SAMPLE_SEC)
        if d < 1.0:                      # 早醒的那 <1 秒不值得再等一輪
            d += SAMPLE_SEC
        await asyncio.sleep(d)
        if stop.is_set():
            return
        period = int(time.time() // SAMPLE_SEC)
        if period == last_period:
            continue                     # 同一個週期已經取過 —— 不再取第二筆
        last_period = period
        try:
            n = sample_once()
            print("取樣 %s：%d 個市場" % (time.strftime("%H:%M:%S"), n))
            # **每次取樣就刷旗標**，不要只在 flush（300 秒）時刷 ——
            # 否則新鮮度訊號的解析度比取樣還粗，前 5 分鐘看板只看得到
            # 「啟動中」那一份，分不出「還沒到第一次落盤」與「取樣執行緒死了」。
            write_flag()
        except Exception as e:                               # noqa: BLE001
            print("取樣失敗:", e)


async def flusher(stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(FLUSH_SEC)
        try:
            flush()
        except Exception as e:                               # noqa: BLE001
            print("flush 失敗:", e)


SUB_BATCH = 20             # 每批幾個訂閱（實測值，見 connected 那段）
SUB_PAUSE = 1.0            # 批與批之間（秒）


async def _subscribe_batched(ws) -> None:
    """分批訂閱簿口。**不要在讀訊息的迴圈裡 await sleep** —— 那會停住讀取。"""
    mids = list(_id2sym)
    for i in range(0, len(mids), SUB_BATCH):
        for mid in mids[i:i + SUB_BATCH]:
            try:
                await ws.send(json.dumps(
                    {"type": "subscribe", "channel": "order_book/%d" % mid}))
            except Exception as e:                           # noqa: BLE001
                print("訂閱中斷於第 %d 個：%r" % (i, e))
                return
        await asyncio.sleep(SUB_PAUSE)


async def ws_loop(stop: asyncio.Event) -> None:
    import websockets
    while not stop.is_set():
        try:
            async with websockets.connect(
                    WS_URL, ping_interval=20, ping_timeout=20,
                    max_size=16 * 1024 * 1024) as ws:
                # 每次重連都把簿口清空並重新訂閱 —— 舊簿口配新 nonce 是虛構的
                _bids.clear()
                _asks.clear()
                _nonce.clear()
                subbed = False
                async for raw in ws:
                    if stop.is_set():
                        break
                    _stat["msgs"] += 1
                    try:
                        m = json.loads(raw)
                    except Exception:                        # noqa: BLE001
                        continue
                    t = m.get("type")
                    if t == "connected":
                        # **要等 connected 才訂閱**，太早送會被靜默忽略
                        # （arb/engine/entropy_arb/feeds.py 的實測註解）
                        #
                        # **而且要分批、而且要丟成背景 task。** 2026-09-13 實測
                        # （`ws_sub_limit.py --channel order_book`，217 個市場）：
                        #     一次 burst  215/217 ack
                        #     20 個一批、批間 1 秒  **217/217 ack**
                        # 成交帶那支更嚴重（burst 有一次在第 7 秒被伺服器關連線）。
                        # 而 `order_book` 的訂閱回的是**整本簿口快照**
                        # （BTC 有上千檔），217 本一次湧進來比 217 個成交 ack 重
                        # 得多 —— 所以宇宙放寬到長尾之後，分批是必要的不是保險。
                        #
                        # 丟成 task 的理由：在 `async for raw in ws` 裡面 await
                        # sleep 會**停住讀取**，快照就會堆在緩衝區裡。
                        asyncio.create_task(_subscribe_batched(ws))
                        subbed = True
                        continue
                    if t == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                        continue
                    if t not in ("subscribed/order_book", "update/order_book"):
                        continue
                    mid = _chan_id(m.get("channel", ""))
                    if mid is None or mid not in _id2sym:
                        continue
                    ob = m.get("order_book") or {}
                    _btime[mid] = (m.get("last_updated_at")
                                   or ob.get("last_updated_at")
                                   or m.get("timestamp") or 0)
                    if t == "subscribed/order_book":
                        _nonce[mid] = ob.get("nonce")
                        _apply(mid, ob, True)
                        tob_capture(mid)
                        continue
                    prev, beg = _nonce.get(mid), ob.get("begin_nonce")
                    if prev is not None and beg is not None and beg > prev + 1:
                        # 少了一個檔位更新 -> 簿口是虛構的。清掉並重訂閱，
                        # 寧可這個市場空幾秒，也不要報一個假的頂檔。
                        _stat["gaps"] += 1
                        _resync[mid] = _resync.get(mid, 0) + 1
                        _nonce[mid] = None
                        _bids.get(mid, {}).clear()
                        _asks.get(mid, {}).clear()
                        await ws.send(json.dumps(
                            {"type": "unsubscribe",
                             "channel": "order_book/%d" % mid}))
                        await ws.send(json.dumps(
                            {"type": "subscribe",
                             "channel": "order_book/%d" % mid}))
                        continue
                    if _nonce.get(mid) is None:
                        continue          # 等新快照
                    _nonce[mid] = ob.get("nonce")
                    _apply(mid, ob, False)
                    tob_capture(mid)
        except Exception as e:                               # noqa: BLE001
            if stop.is_set():
                return
            print("WS 例外：%s" % str(e)[:120])
        if stop.is_set():
            return
        _stat["reconnects"] += 1
        print("WS 斷線，第 %d 次重連" % _stat["reconnects"])
        await asyncio.sleep(min(30, 2 ** min(_stat["reconnects"], 5)))


async def main_async(seconds, top_n) -> int:
    _id2sym.update(await universe(top_n))
    _stat["markets"] = len(_id2sym)
    print("訂閱 %d 個永續市場的簿口 -> %s（每 %d 秒取樣一次）"
          % (len(_id2sym), MID_DIR, SAMPLE_SEC))
    write_flag(starting=True)          # 先舉手再連線
    stop = asyncio.Event()
    tasks = [asyncio.create_task(ws_loop(stop)),
             asyncio.create_task(sampler(stop)),
             asyncio.create_task(flusher(stop))]
    if seconds:
        await asyncio.sleep(seconds)
        stop.set()
    else:
        await asyncio.gather(*tasks)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    flush()
    up = time.time() - _stat["started"]
    print("停止：取樣 %d 次、寫入 %d 列、訊息 %d（%.0f/秒）、交錯 %d、"
          "nonce 斷 %d、重連 %d、%.0f 秒"
          % (_stat["samples"], _stat["rows"], _stat["msgs"],
             _stat["msgs"] / max(up, 1), _stat["crossed"], _stat["gaps"],
             _stat["reconnects"], up))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None,
                    help="跑幾秒就停（驗收用；預設常駐）")
    ap.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                    help="錄日成交額前幾名的永續（預設 %d）" % TOP_N_DEFAULT)
    a = ap.parse_args()
    if not acquire_single_instance():
        print("已經有一個 Lighter 中價錄製器在跑（檔案鎖 %s）。不啟動第二個 —— "
              "兩個寫入者會靜默破壞 parquet。" % LOCK.name)
        return 2
    # 回傳碼要傳出去：退出碼 0 是「跑完了」不是「做了事」
    # （mistake.md 2026-08-26）。
    return asyncio.run(main_async(a.seconds, a.top))


if __name__ == "__main__":
    raise SystemExit(main())
