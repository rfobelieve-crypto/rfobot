# -*- coding: utf-8 -*-
"""量「頂檔事件式錄製」要花多少磁碟 —— 量，不要估。

===========================================================================
為什麼要這支
===========================================================================
TODO §1.37 的 L3/L4（文章 3b 的成交額主張）用逐筆帶**永遠**答不了：
250 ms 一格之下只有 0.8% 的格子兩邊都有成交。只有簿口答得了，因為
**買賣報價不需要有人成交就會變**。

但「要不要開次秒級簿口錄製」這個決定不能用我估的「1-2 GB/日」去做 ——
mistake.md 2026-09-03 那條就是「頻率 x 幅度 x 規模」形狀的估計錯一到兩個
數量級。所以先量真實的**頂檔變動率**。

===========================================================================
本支只讀，不寫任何 parquet
===========================================================================
它開自己的 WS 連線、跑完就結束、**不碰任何錄製器的輸出目錄**。
（mistake.md 2026-09-11：我自己起第二個實例對同一批 parquet 做
read-modify-write，後果事後無法量化。這支沒有那個風險，因為它不寫。）

量三個數，因為它們對應三種設計：
  every_msg   每一則訊息都記           （上界）
  tob_any     頂檔的價或量變了才記
  tob_px      **只有頂檔的價變了才記**  （文章 Reduction 那節的處方：
              「remove new quotes where the data that is relevant to us has
              not changed」「we drop the duplicate midprices」）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

API = "https://mainnet.zklighter.elliot.ai"
WS = "wss://mainnet.zklighter.elliot.ai/stream"

# §1.37 的 39 個共同標的（Lighter 與 HL 都有）。寫死而不是當場算，
# 因為宇宙要事前凍結（factor-research 第 2 條）。
SHARED = ["AAVE", "ADA", "AERO", "ARB", "ASTER", "AVAX", "BNB", "BTC",
          "CASHCAT", "DOGE", "ENA", "ETH", "ETHFI", "FARTCOIN", "GRAM",
          "HYPE", "LINK", "LIT", "MET", "NEAR", "PAXG", "PENDLE", "PENGU",
          "PONS", "PUMP", "SOL", "SUI", "TAO", "TRUMP", "TRX", "USELESS",
          "XMR", "XPL", "ZEC", "ZK", "WLFI", "SKY", "OP", "JUP"]

# 一列的位元組估計：parquet + snappy 之下，12 個數值欄約 40-60 bytes/列。
# 拿我們自己的 lighter_mid 回推：157 MB/日 / (80 幣 x 17,280 取樣) = 113 B/列，
# 但那一列有 14 欄含深度。頂檔只要 6 欄 -> 取 **50 B/列**，並在輸出標明。
BYTES_PER_ROW = 50


def _chan_id(channel: str):
    """回音是 `order_book:181`（冒號），訂閱送的是 `order_book/181`。"""
    for sep in (":", "/"):
        if sep in channel:
            try:
                return int(channel.rsplit(sep, 1)[1])
            except ValueError:
                return None
    return None


async def main(seconds: int):
    # **市場表從自己的 parquet 拿，不打 REST。** `/api/v1/orderBooks` 在限流時
    # 回 text/html 而不是 JSON（2026-09-13 實測，同一家 CDN 之前也給過
    # urllib 一路 405）。而正在跑的 `lighter_tape` 每一列都帶 market_id 與
    # coin —— 那份對照表已經在手上，沒有理由再去問一次。
    import glob

    import pandas as pd
    fs = sorted(glob.glob("D:/flowbot_data/lighter/trades/*/*.parquet"),
                key=os.path.getmtime)[-6:]
    if not fs:
        raise RuntimeError("沒有 Lighter 逐筆帶 parquet —— 拿不到市場對照表")
    m = pd.concat([pd.read_parquet(f, columns=["market_id", "coin"])
                   for f in fs], ignore_index=True).dropna().drop_duplicates()
    sym2id = {str(c).upper(): int(i) for i, c in
              zip(m.market_id.values, m.coin.values)}
    print("市場對照表：從 %d 個 parquet 取到 %d 個市場" % (len(fs), len(sym2id)))
    want = {sym2id[c]: c for c in SHARED if c in sym2id}
    print("共同標的 %d 個，Lighter 上找到 %d 個" % (len(SHARED), len(want)))
    missing = [c for c in SHARED if c not in sym2id]
    if missing:
        print("  Lighter 沒有：%s" % ", ".join(missing))

    best = {}                       # mid -> (bid, ask, bid_sz, ask_sz)
    cnt = {"msg": 0, "any": 0, "px": 0, "book_msg": 0}
    per_sym = {}

    import websockets
    t0 = time.time()
    async with websockets.connect(WS, ping_interval=20,
                                  max_size=8 * 1024 * 1024) as ws:
        while time.time() - t0 < seconds:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                continue
            cnt["msg"] += 1
            try:
                m = json.loads(raw)
            except Exception:
                continue
            t = m.get("type")
            if t == "connected":
                for mid in want:
                    await ws.send(json.dumps({"type": "subscribe",
                                              "channel": "order_book/%d" % mid}))
                print("已訂閱 %d 個市場，開始計時 %d 秒" % (len(want), seconds))
                continue
            if t not in ("subscribed/order_book", "update/order_book"):
                continue
            mid = _chan_id(m.get("channel", ""))
            if mid not in want:
                continue
            cnt["book_msg"] += 1
            ob = m.get("order_book") or {}
            bids, asks = ob.get("bids") or [], ob.get("asks") or []
            if not bids or not asks:
                continue
            try:
                nb = (float(bids[0]["price"]), float(asks[0]["price"]),
                      float(bids[0].get("size") or 0),
                      float(asks[0].get("size") or 0))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            old = best.get(mid)
            best[mid] = nb
            if old is None:
                continue
            sym = want[mid]
            d = per_sym.setdefault(sym, {"any": 0, "px": 0})
            if nb != old:
                cnt["any"] += 1
                d["any"] += 1
            if nb[0] != old[0] or nb[1] != old[1]:
                cnt["px"] += 1
                d["px"] += 1

    el = time.time() - t0
    print("\n量測 %.0f 秒｜%d 個市場" % (el, len(best)))
    print("-" * 72)
    hdr = "%-26s %12s %12s %14s"
    print(hdr % ("設計", "事件/秒", "列/日", "MB/日 @%dB" % BYTES_PER_ROW))
    print("-" * 72)
    for lab, k in (("每則訊息都記（上界）", "msg"),
                   ("簿口訊息都記", "book_msg"),
                   ("頂檔價或量變了才記", "any"),
                   ("**只有頂檔價變了才記**", "px")):
        rate = cnt[k] / el
        per_day = rate * 86400
        print(hdr % (lab, "%.1f" % rate, "%.2fM" % (per_day / 1e6),
                     "%.0f" % (per_day * BYTES_PER_ROW / 1e6)))
    print("-" * 72)
    top = sorted(per_sym.items(), key=lambda z: -z[1]["px"])[:8]
    print("\n頂檔價變動最多的 8 個（事件/秒）：")
    for sym, d in top:
        print("  %-10s 價 %6.2f｜價或量 %6.2f" % (sym, d["px"] / el, d["any"] / el))
    quiet = [s for s, d in per_sym.items() if d["px"] / el < 0.05]
    print("\n幾乎不動的（價變 < 0.05/秒）：%d 個%s"
          % (len(quiet), "  " + ", ".join(sorted(quiet)[:12]) if quiet else ""))
    print("\n（列大小取 %d B —— 從 lighter_mid 實際的 113 B/列 回推，"
          "因為頂檔只要 6 欄而那支有 14 欄含深度）" % BYTES_PER_ROW)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    a = ap.parse_args()
    asyncio.run(main(a.seconds))
