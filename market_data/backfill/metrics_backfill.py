"""
Download Binance Futures metrics (OI, long/short ratios, taker ratios).

Source: https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/
  - 5-minute intervals
  - Columns: create_time, symbol, sum_open_interest, sum_open_interest_value,
             count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
             count_long_short_ratio, sum_taker_long_short_vol_ratio
  - No date limit (unlike REST API's 30-day cap)

Output: Parquet file per symbol in market_data/raw_data/metrics/

Usage:
    python -m market_data.backfill.metrics_backfill
    python -m market_data.backfill.metrics_backfill --start 2026-01-01 --end 2026-03-30
    python -m market_data.backfill.metrics_backfill --symbol ETHUSDT
"""
from __future__ import annotations

import io
import time
import logging
import argparse
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "market_data" / "raw_data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"


def download_day(symbol: str, d: date) -> pd.DataFrame | None:
    ds   = d.strftime("%Y-%m-%d")
    url  = f"{BASE_URL}/{symbol}/{symbol}-metrics-{ds}.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            logger.warning("  404: %s %s", symbol, ds)
            return None
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)
        return df
    except Exception:
        logger.exception("Failed: %s %s", symbol, ds)
        return None


def backfill(symbol: str = "BTCUSDT",
             start: date | None = None,
             end: date | None = None) -> Path:
    start = start or date(2026, 1, 1)
    end   = end or date.today() - timedelta(days=1)

    logger.info("Metrics backfill: %s  %s ~ %s", symbol, start, end)

    frames = []
    d = start
    ok = fail = 0
    while d <= end:
        df = download_day(symbol, d)
        if df is not None and not df.empty:
            frames.append(df)
            ok += 1
        else:
            fail += 1
        d += timedelta(days=1)
        time.sleep(0.15)

        if ok % 10 == 0 and ok > 0:
            logger.info("  %s: %d days downloaded...", symbol, ok)

    if not frames:
        logger.error("No data for %s", symbol)
        return Path()

    full = pd.concat(frames, ignore_index=True)
    full["create_time"] = pd.to_datetime(full["create_time"], utc=True)
    full = full.sort_values("create_time").drop_duplicates(subset=["create_time"])

    out = METRICS_DIR / f"{symbol}_metrics.parquet"
    full.to_parquet(out, index=False)
    logger.info("Saved %s: %d rows  (%s ~ %s)",
                out.name, len(full),
                full["create_time"].min(), full["create_time"].max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start",  type=date.fromisoformat, default=None)
    ap.add_argument("--end",    type=date.fromisoformat, default=None)
    args = ap.parse_args()

    backfill(args.symbol, args.start, args.end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
