"""Download sub-hourly klines (with volume) for the POC/volume-profile study.

Why this exists (TODO §1.00): the existing `.cache/m1/*_1m.csv` files carry
only `minute,high,low,close` — no volume — so a volume profile cannot be
built from them.  `.cache/tf/*_15m.csv` has volume but stops 2026-07-30 and
only covers 400 days.

Writes, gitignored under .cache/ per this directory's convention:
    .cache/m5/{SYM}_5m.csv    core9, full window (matches the 1h cache)
    .cache/m1v/{SYM}_1m.csv   BTC/ETH only, calibration set

Both are `time,open,high,low,close,volume` with ts in SECONDS, same shape as
fetch_klines.py so sweep_core.load_csv reads them unchanged.

Resumable: an existing file is extended from its last bar.

Usage:
    python research/sweep_failure/fetch_klines_sub.py --interval 5m
    python research/sweep_failure/fetch_klines_sub.py --interval 1m --syms BTC,ETH --days 365
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
BASE = "https://api.binance.com/api/v3/klines"
MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "poc-study/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return []


def last_ts(path):
    """Last bar's ts in seconds, or None."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode(errors="ignore").strip().splitlines()
    for line in reversed(tail):
        part = line.split(",")[0]
        try:
            return int(float(part))
        except ValueError:
            continue
    return None


def fetch(sym, interval, start_ms, end_ms, out_path):
    step = MS[interval]
    new = 0
    cur = start_ms
    fresh = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["time", "open", "high", "low", "close", "volume"])
        while cur < end_ms:
            d = get(f"{BASE}?symbol={sym}USDT&interval={interval}"
                    f"&startTime={cur}&limit=1000")
            if not d:
                break
            rows = []
            for k in d:
                t0 = int(k[0])
                if t0 + step > end_ms:      # drop the unfinished bar
                    continue
                rows.append((t0 // 1000, float(k[1]), float(k[2]),
                             float(k[3]), float(k[4]), float(k[5])))
            w.writerows(rows)
            new += len(rows)
            f.flush()
            nxt = int(d[-1][0]) + step
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.12)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="5m", choices=list(MS))
    ap.add_argument("--syms", default=",".join(CORE9))
    ap.add_argument("--days", type=int, default=0,
                    help="0 = match the 1h cache window (2024-02-15 onward)")
    a = ap.parse_args()

    here = Path(__file__).parent
    sub = {"5m": "m5", "1m": "m1v", "15m": "m15"}[a.interval]
    out = here / ".cache" / sub
    out.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1000)
    default_start = 1707973200_000   # 2024-02-15 05:00 UTC, the 1h cache start

    for sym in [s.strip().upper() for s in a.syms.split(",") if s.strip()]:
        p = out / f"{sym}_{a.interval}.csv"
        lt = last_ts(p)
        if lt is not None:
            start = (lt + MS[a.interval] // 1000) * 1000
            mode = f"resume from {lt}"
        elif a.days:
            start = end_ms - a.days * 86400_000
            mode = f"last {a.days}d"
        else:
            start = default_start
            mode = "full window"
        if start >= end_ms:
            print(f"{sym:5s} {a.interval} up to date")
            continue
        t0 = time.time()
        n = fetch(sym, a.interval, start, end_ms, p)
        print(f"{sym:5s} {a.interval} +{n:7d} bars ({mode}) "
              f"{time.time()-t0:.0f}s -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
