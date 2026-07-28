"""Download 1h klines for the 9-symbol basket from Binance public REST.

Writes {SYM}USDT_1h.csv (time,open,high,low,close,volume; ts in seconds)
into --out-dir (default: research/sweep_failure/.cache/, gitignored by the
repo's *.csv-under-.cache convention — do NOT commit data).

Usage:
    python research/sweep_failure/fetch_klines.py
    python research/sweep_failure/fetch_klines.py --days 900 --out-dir .cache
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
BASE = "https://api.binance.com/api/v3/klines"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sweep-research/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch_symbol(sym, days):
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    rows = {}
    cur = start
    while cur < end:
        d = get(f"{BASE}?symbol={sym}USDT&interval=1h&startTime={cur}&limit=1000")
        if not d:
            break
        for k in d:
            ts = int(k[0]) // 1000
            rows[ts] = (ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        cur = int(d[-1][0]) + 3600_000
        time.sleep(0.15)  # stay well under rate limit
    return [rows[t] for t in sorted(rows)][:-1]  # drop unfinished bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=930)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / ".cache"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in SYMS:
        bars = fetch_symbol(s, args.days)
        p = out / f"{s}USDT_1h.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "open", "high", "low", "close", "volume"])
            w.writerows(bars)
        print(f"{s}: {len(bars)} bars -> {p}")


if __name__ == "__main__":
    main()
