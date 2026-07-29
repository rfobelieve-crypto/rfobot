# -*- coding: utf-8 -*-
"""Fetch Binance USDT-M 1-minute klines covering the walk-forward OOS window.

Why this exists: the intra-bar volume-distribution hypothesis (see
research/intrabar_volume_ic.py) needs minute resolution, but the `ohlcv_1m`
table stops at 2026-03-30 while the OOS window runs to 2026-07-26 — an overlap
of only ~37 days / 2 calendar months, too thin for the per-month consistency
check the screening protocol requires.

Output goes to a parquet, NOT into `ohlcv_1m`. A research experiment should not
write to a production table before the feature has earned its place; if the
screen passes, backfilling the table properly becomes its own task.

Same endpoint the live system uses (indicator/data_fetcher.py:20), so the bars
are the same instrument and venue V7 already trains on.

Run: python research/fetch_1m_for_intrabar.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
LIMIT = 1500          # fapi max
OUT = ROOT / "research/results/binance_1m_oos_window.parquet"

# Cover the whole OOS window with a margin, so the first bar has its full
# complement of minutes rather than a truncated one.
START = pd.Timestamp("2026-02-22 00:00:00")
END = pd.Timestamp("2026-07-27 00:00:00")

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote", "ignore"]


def main() -> int:
    rows = []
    cur = int(START.timestamp() * 1000)
    end_ms = int(END.timestamp() * 1000)
    n_req = 0
    while cur < end_ms:
        try:
            r = requests.get(URL, params=dict(symbol=SYMBOL, interval="1m",
                                              startTime=cur, limit=LIMIT),
                             timeout=30)
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            print(f"  request failed at {pd.to_datetime(cur, unit='ms')}: {exc}")
            time.sleep(3)
            continue
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 60_000
        n_req += 1
        if n_req % 20 == 0:
            print(f"  {n_req} requests, {len(rows):,} bars, "
                  f"at {pd.to_datetime(cur, unit='ms')}", flush=True)
        time.sleep(0.12)          # stay well inside the weight limit

    df = pd.DataFrame(rows, columns=COLS)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df[["dt", "open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote"]]
          .drop_duplicates("dt").set_index("dt").sort_index())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    gaps = df.index.to_series().diff().dt.total_seconds().div(60)
    print(f"\n{len(df):,} minute bars  {df.index.min()} -> {df.index.max()}")
    print(f"missing minutes: {int((gaps > 1).sum())} gaps, "
          f"largest {gaps.max():.0f} min")
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
