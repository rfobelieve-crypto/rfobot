# -*- coding: utf-8 -*-
"""Stage 1 support — download SPOT aggTrades days for the tick-truth check.

Why spot and not the perp aggTrades already in the repo: those files
(market_data/raw_data/aggtrades/binance) are USD-M PERPETUAL, verified
2026-09-06 by matching a minute's print quantity against fapi's 1m volume
(ratio 1.000, versus 10.3x for spot).  The bar table this pipeline builds is
SPOT, because the frozen sweep engine and every Gate-F number are spot.  Truth
has to come from the same instrument as the thing being checked.

Files land in data/ticks/{DAY}.zip and are read straight out of the zip.
Each sampled day needs its predecessor too, for a 24h lookback.

Sampling spreads the days over the whole bar coverage rather than one quarter,
so the error estimate is not a snapshot of one volatility regime.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TICKS = HERE / "data" / "ticks"
BARS = HERE / "data" / "bars"
URL = ("https://data.binance.vision/data/spot/daily/aggTrades/{sym}USDT/"
       "{sym}USDT-aggTrades-{day}.zip")


def download(sym, day):
    p = TICKS / f"{sym}_{day}.zip"
    if p.exists() and p.stat().st_size > 0:
        return p, 0
    url = URL.format(sym=sym, day=day)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "poc-stage1/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return None, e.code
    p.write_bytes(data)
    return p, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="BTC")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()
    TICKS.mkdir(parents=True, exist_ok=True)

    bp = BARS / f"{a.sym}.parquet"
    if not bp.exists():
        sys.exit(f"missing {bp} -- run bars.py first")
    ts = pd.read_parquet(bp, columns=["ts"])["ts"]
    days = pd.to_datetime(ts, unit="ms", utc=True).dt.strftime("%Y-%m-%d").unique()
    days = days[2:-1]                       # need a predecessor and a full day
    rng = np.random.default_rng(a.seed)
    pick = sorted(str(x) for x in rng.choice(days, size=min(a.days, len(days)),
                                             replace=False))
    # L3 looks back 72h and the volume window L1 has been observed out to 54h,
    # so a reference day needs THREE predecessors, not one.  Without them the
    # tick "truth" would be computed on a truncated window -- silently wrong.
    want = []
    for d in pick:
        want.append(d)
        for k in (1, 2, 3):
            want.append((pd.Timestamp(d) - pd.Timedelta(days=k)).strftime("%Y-%m-%d"))
    want = sorted(set(want))

    got, mb, fail = 0, 0.0, []
    for d in want:
        p, code = download(a.sym, d)
        if p is None:
            fail.append((d, code))
            continue
        got += 1
        mb += p.stat().st_size / 1e6
    print(f"{a.sym}: {got}/{len(want)} day files, {mb:,.0f} MB -> {TICKS}")
    if fail:
        print("  missing:", fail[:10])
    print("sampled reference days:", ", ".join(pick))


if __name__ == "__main__":
    main()
