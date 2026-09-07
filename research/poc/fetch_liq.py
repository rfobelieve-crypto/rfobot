# -*- coding: utf-8 -*-
"""清算歷史回填（Coinglass，1 小時，core9）—— 讓清算線不必等 6-12 個月。

為什麼換資料源
    本 repo 自己的 `liquidation_1m`（WS 錄製）有兩個獨立的殘缺：
      1. 只訂 BTC/ETH 兩個幣（`liquidation_collector.py` 的 URL 寫死）
      2. Binance `<sym>@forceOrder` **每秒最多推一筆**——級聯時每秒有
         上百筆清算，只有一筆會到我們手上。**漏失隨強度惡化**，而強度
         正是我們要研究的東西。這是交易所限制，不是可以修好的 bug。
    Coinglass 的 `/futures/liquidation/history` 是交易所回報的**彙總**，
    沒有逐筆節流的問題。代價是**只有 1 小時**（1m/5m 被方案擋，實測 403）。

    1 小時剛好對上凍結掃單引擎的 bar，所以兩條線可以放在同一支尺上比
    ——這正是使用者要的「價格幾何當對照組」。

單位與時點（踩過的坑，寫死在這裡）
    · `time` 是**毫秒**（實測 max > 1e12）。Coinglass 不同端點單位不同，
      不要假設（mistake.md 2026-04-12）。
    · 一列的 time 是**該小時的開始**，值涵蓋整個小時 -> 在 t 這一刻只能用
      `time + 3600s <= t` 的列。這與 OI 那次的前視同構
      （`event_census.py` 檔頭：metrics 的一列不是 create_time 當下的快照）。
      本檔只負責**存**，時點規矩由使用端負責，但欄位名刻意留 `time` 原樣
      以免有人以為它是「收盤時刻」。

輸出：research/poc/data/liq/{SYM}.parquet
    time(ms, 小時開始) / long_usd / short_usd / total_usd
"""
from __future__ import annotations

import sys
import time as _t
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from market_data.backfill import coinglass_backfill as cb  # noqa: E402

OUT = HERE / "data" / "liq"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FLOOR_MS = 1_704_067_200_000        # 2024-01-01，對齊 1h bar 快取的起點
PAGE = 1000


def fetch_symbol(sym):
    frames, end = [], None
    while True:
        p = {"symbol": f"{sym}USDT", "exchange": "Binance",
             "interval": "1h", "limit": PAGE}
        if end is not None:
            p["end_time"] = int(end)
        rows = cb._fetch("/futures/liquidation/history", p)
        if not rows:
            break
        d = pd.DataFrame(rows)
        d["time"] = pd.to_numeric(d["time"])
        if d["time"].max() < 1e12:                 # 秒 -> 毫秒（自動偵測）
            d["time"] = d["time"] * 1000
        frames.append(d)
        oldest = int(d["time"].min())
        if oldest <= FLOOR_MS or len(d) < PAGE or (end is not None and oldest >= end):
            break
        end = oldest
        _t.sleep(0.35)                             # 對 API 客氣一點
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"long_liquidation_usd": "long_usd",
                          "short_liquidation_usd": "short_usd"})
    for c in ("long_usd", "short_usd"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["total_usd"] = d["long_usd"] + d["short_usd"]
    d = (d[["time", "long_usd", "short_usd", "total_usd"]]
         .drop_duplicates("time").sort_values("time").reset_index(drop=True))
    return d[d["time"] >= FLOOR_MS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'幣':6s} {'列數':>8s} {'起':>12s} {'迄':>12s} {'缺口':>8s} "
          f"{'非零時數':>9s} {'總清算':>14s}")
    for s in CORE9:
        d = fetch_symbol(s)
        if d.empty:
            print(f"{s:6s} (空)")
            continue
        d.to_parquet(OUT / f"{s}.parquet", index=False)
        span_h = (d["time"].max() - d["time"].min()) / 3_600_000 + 1
        gap = 1 - len(d) / span_h
        nz = float((d["total_usd"] > 0).mean())
        a = pd.Timestamp(int(d["time"].min()), unit="ms", tz="UTC").date()
        b = pd.Timestamp(int(d["time"].max()), unit="ms", tz="UTC").date()
        print(f"{s:6s} {len(d):8,d} {str(a):>12s} {str(b):>12s} "
              f"{gap*100:7.3f}% {nz*100:8.2f}% "
              f"${d['total_usd'].sum()/1e9:13,.2f}B")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
