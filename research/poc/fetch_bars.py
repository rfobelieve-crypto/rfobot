# -*- coding: utf-8 -*-
"""Stage 0a — pull raw 1-minute klines for the core9 basket.

Writes research/poc/data/raw/{COIN}_1m.csv (gitignored), columns
    open_ms, open, high, low, close, volume, n_trades
open_ms is the bar's OPEN time in milliseconds, exactly as Binance returns it;
no unit guessing downstream (mistake.md 2026-04-12).

Window matches the frozen 1h cache so the event population stays comparable:
2024-02-15 05:00 UTC -> now.  Resumable: an existing file is extended from its
last bar, so an interrupted run costs only what it had not yet written.

This script does NOT clean, align, or fill.  That is Stage 0b (bars.py), and
keeping them apart is deliberate: the raw file stays a faithful record of what
the exchange returned, so any later disagreement can be traced to one layer.
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
STEP_MS = 60_000
WINDOW_START_MS = 1_707_973_200_000        # 2024-02-15 05:00 UTC
RAW = Path(__file__).resolve().parent / "data" / "raw"


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "poc-stage0/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    return []


def last_open_ms(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 8192))
        tail = f.read().decode(errors="ignore").strip().splitlines()
    for line in reversed(tail):
        try:
            return int(line.split(",")[0])
        except (ValueError, IndexError):
            continue
    return None


def fetch(sym, start_ms, end_ms, path):
    new = 0
    cur = start_ms
    fresh = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["open_ms", "open", "high", "low", "close", "volume", "n_trades"])
        while cur < end_ms:
            d = get(f"{BASE}?symbol={sym}USDT&interval=1m&startTime={cur}&limit=1000")
            if not d:
                break
            rows = [(int(k[0]), k[1], k[2], k[3], k[4], k[5], k[8])
                    for k in d if int(k[0]) + STEP_MS <= end_ms]
            w.writerows(rows)
            new += len(rows)
            f.flush()
            nxt = int(d[-1][0]) + STEP_MS
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.10)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    end_ms = int(time.time() * 1000)
    for sym in [s.strip().upper() for s in a.syms.split(",") if s.strip()]:
        p = RAW / f"{sym}_1m.csv"
        lo = last_open_ms(p)
        start = (lo + STEP_MS) if lo is not None else WINDOW_START_MS
        if start >= end_ms:
            print(f"{sym:5s} up to date", flush=True)
            continue
        t0 = time.time()
        n = fetch(sym, start, end_ms, p)
        print(f"{sym:5s} +{n:8d} bars  {time.time() - t0:6.0f}s  -> {p.name}", flush=True)


if __name__ == "__main__":
    main()
