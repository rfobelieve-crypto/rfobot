"""
Download Binance BVOL (BTC Volatility Index) and aggregate into 1h bars.

Downloads daily zip files from data.binance.vision, resamples 1-second
data to 1h OHLC, computes derived features, and saves as parquet.

Output: research/ml_data/binance_bvol_1h.parquet
  Columns: bvol_open, bvol_high, bvol_low, bvol_close, bvol_std,
           bvol_change_1h, bvol_change_4h, bvol_change_8h, bvol_change_24h,
           bvol_percentile, bvol_ma_24h, bvol_rv_spread

Usage:
    python research/backfill_bvol.py
    python research/backfill_bvol.py --start 2025-10-17 --end 2026-04-01
"""
from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/option/daily/BVOLIndex/BTCBVOLUSDT"
OUT_PATH = Path("research/ml_data/binance_bvol_1h.parquet")
CACHE_DIR = Path("research/ml_data/.bvol_cache")


def download_day(date_str: str) -> pd.DataFrame | None:
    """Download one day's BVOL data and resample to 1h."""
    cache_path = CACHE_DIR / f"{date_str}_1h.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    fname = f"BTCBVOLUSDT-BVOLIndex-{date_str}.zip"
    url = f"{BASE_URL}/{fname}"

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            logger.warning("  Not found: %s", date_str)
            return None
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)

        # Columns: calc_time, symbol, base_asset, quote_asset, index_value
        df["dt"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
        df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
        df = df.set_index("dt").sort_index()

        # Resample to 1h OHLC
        hourly = df["index_value"].resample("1h").agg(
            bvol_open="first",
            bvol_high="max",
            bvol_low="min",
            bvol_close="last",
            bvol_std="std",
        )
        hourly = hourly.dropna(subset=["bvol_close"])

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(cache_path)

        logger.info("  %s: %d 1h bars, BVOL=%.1f~%.1f",
                     date_str, len(hourly),
                     hourly["bvol_close"].min(), hourly["bvol_close"].max())
        return hourly

    except Exception as e:
        logger.error("  Failed %s: %s", date_str, e)
        return None


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the 1h BVOL data."""
    c = df["bvol_close"]

    # Change over multiple horizons
    df["bvol_change_1h"] = c.pct_change(1)
    df["bvol_change_4h"] = c.pct_change(4)
    df["bvol_change_8h"] = c.pct_change(8)
    df["bvol_change_24h"] = c.pct_change(24)

    # Moving averages
    df["bvol_ma_24h"] = c.rolling(24, min_periods=6).mean()
    df["bvol_ma_72h"] = c.rolling(72, min_periods=24).mean()

    # Ratio to MA (mean-reversion signal)
    df["bvol_ratio_ma24"] = c / df["bvol_ma_24h"].replace(0, np.nan)

    # Expanding percentile (historical rank)
    df["bvol_percentile"] = c.expanding(min_periods=168).rank(pct=True)

    # Intra-bar range (high - low) as volatility of vol
    df["bvol_intra_range"] = df["bvol_high"] - df["bvol_low"]
    df["bvol_intra_range_pct"] = df["bvol_intra_range"] / c.replace(0, np.nan)

    # Z-score (24h)
    mu = c.rolling(24, min_periods=6).mean()
    sd = c.rolling(24, min_periods=6).std().replace(0, np.nan)
    df["bvol_zscore"] = (c - mu) / sd

    # Momentum: slope of BVOL over 4h
    def _slope(arr):
        if len(arr) < 2 or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    df["bvol_slope_4h"] = c.rolling(4, min_periods=2).apply(_slope, raw=True)

    # Acceleration: slope change
    df["bvol_accel_4h"] = df["bvol_slope_4h"].diff()

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-17")
    ap.add_argument("--end", default="2026-04-01")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    n_days = (end - start).days + 1

    print(f"Downloading Binance BVOL: {args.start} ~ {args.end} ({n_days} days)")
    print()

    all_bars = []
    failed_days = []

    for i in range(n_days):
        date = start + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        bars = download_day(date_str)
        if bars is None:
            failed_days.append(date_str)
            continue

        all_bars.append(bars)

        if (i + 1) % 20 == 0:
            logger.info("[%d/%d] processed", i + 1, n_days)

        time.sleep(0.3)  # Rate limit

    if not all_bars:
        print("No data!")
        return

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Add features
    combined = compute_features(combined)

    print(f"\n{'='*60}")
    print(f"RESULT: {len(combined)} 1h bars")
    print(f"Range: {combined.index[0]} ~ {combined.index[-1]}")
    print(f"Failed days: {len(failed_days)}")
    print(f"\nBVOL stats:")
    print(f"  Mean: {combined['bvol_close'].mean():.1f}")
    print(f"  Min:  {combined['bvol_close'].min():.1f}")
    print(f"  Max:  {combined['bvol_close'].max():.1f}")
    print(f"  Columns: {list(combined.columns)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
