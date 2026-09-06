# -*- coding: utf-8 -*-
"""Fetch Binance USD-M futures `metrics` (5-minute open interest) for core9.

Why this and not liquidations (measured 2026-09-06)
    The branching-ratio idea needs a COMPLETE flow series.  The liquidation
    data in this repo is not one:
      · our own recorder captures 23.6% of Binance's liquidation notional,
        and the shortfall scales WITH intensity -- 11.0% in the top quintile,
        2.3% in Coinglass's twenty largest hours.  Under-counting that grows
        with intensity biases a branching ratio DOWN exactly at n -> 1.
      · it covers 159 days (BTC/ETH only) and starts 2026-03-31, missing five
        of the six days that carry the tail.
      · Coinglass cannot backfill: the plan's earliest allowed start_time is
        2026-03-10.  The 320-day parquet is accumulated, not retrievable.
      · Binance's historical liquidationSnapshot archive returns 404 (retired).

    Open interest is a STOCK reported by the exchange, not a throttled event
    stream, so it has no completeness problem, and data.binance.vision serves
    it for every coin back to listing at 5-minute granularity for ~10 KB/day.
    Position destruction is the FOOTPRINT of forced flow rather than a count
    of it -- noisier (voluntary closes are in there too) but complete, and
    completeness is the property the estimate actually needs.

    Honest limit, stated up front: this supports the COARSE acceleration proxy
    only.  A Hawkes fit needs event TIMES; a 5-minute stock is not a point
    process, and no historical liquidation event stream exists for this
    market.  "Coarse first, then Hawkes" has no second step on this data.

Columns: create_time, symbol, sum_open_interest, sum_open_interest_value,
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
count_long_short_ratio, sum_taker_long_short_vol_ratio
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "oi"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def day_frame(sym, day):
    url = f"{BASE}/{sym}USDT/{sym}USDT-metrics-{day}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "poc-oi/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        return None, e.code
    z = zipfile.ZipFile(io.BytesIO(blob))
    d = pd.read_csv(z.open(z.namelist()[0]))
    return d, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    ap.add_argument("--start", default="2024-02-15")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    end = a.end or (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(a.start, end, freq="D")]
    for sym in [s.strip().upper() for s in a.syms.split(",") if s.strip()]:
        p = OUT / f"{sym}.parquet"
        have = set()
        old = None
        if p.exists():
            old = pd.read_parquet(p)
            have = set(pd.to_datetime(old["create_time"]).dt.strftime("%Y-%m-%d"))
        frames, miss, t0 = [], [], time.time()
        for d in days:
            if d in have:
                continue
            f, code = day_frame(sym, d)
            if f is None:
                miss.append((d, code))
                continue
            frames.append(f)
            time.sleep(0.02)
        if frames:
            new = pd.concat(frames, ignore_index=True)
            allf = pd.concat([old, new], ignore_index=True) if old is not None else new
            allf["create_time"] = pd.to_datetime(allf["create_time"])
            allf = allf.drop_duplicates("create_time").sort_values("create_time")
            allf.to_parquet(p, index=False)
        else:
            allf = old if old is not None else pd.DataFrame()
        n404 = sum(1 for _, c in miss if c == 404)
        print(f"{sym:5s} rows={len(allf):8,d}  +{sum(len(f) for f in frames):7,d}  "
              f"missing_days={len(miss)} (404={n404})  {time.time()-t0:5.0f}s",
              flush=True)
        if miss[:3]:
            print(f"      first missing: {miss[:3]}", flush=True)


if __name__ == "__main__":
    main()
