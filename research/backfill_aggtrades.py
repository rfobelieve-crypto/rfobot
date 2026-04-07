"""
Download Binance aggTrades and aggregate into 1h large/small trade flow bars.

Downloads daily zip files from data.binance.vision, processes each day
into 1h aggregated bars with large/small trade separation, then saves
the result as a single parquet file.

Output: research/ml_data/binance_aggtrades_1h.parquet
  Columns: large_buy_usd, large_sell_usd, large_delta, large_count,
           small_buy_usd, small_sell_usd, small_delta, small_count,
           total_usd, large_ratio, large_buy_ratio, large_small_div,
           large_small_div_zscore, avg_large_size

Usage:
    python research/backfill_aggtrades.py
    python research/backfill_aggtrades.py --start 2025-10-17 --end 2026-04-01
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"
OUT_PATH = Path("research/ml_data/binance_aggtrades_1h.parquet")
CACHE_DIR = Path("research/ml_data/.aggtrades_cache")

LARGE_THRESHOLD_USD = 50_000  # $50K notional = large trade


def download_day(date_str: str) -> pd.DataFrame | None:
    """Download and parse one day's aggTrades zip. Returns raw trades DF."""
    fname = f"BTCUSDT-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{fname}"

    # Check cache first
    cache_path = CACHE_DIR / f"{date_str}.parquet"
    if cache_path.exists():
        logger.info("  Cache hit: %s", date_str)
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            logger.warning("  Not found: %s (might not exist yet)", date_str)
            return None
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                # Some files have header row, some don't
                df = pd.read_csv(f, low_memory=False)
                if "price" in str(df.columns[0]).lower() or df.columns[0] == "agg_trade_id":
                    # Has header — already parsed correctly
                    df.columns = [
                        "agg_trade_id", "price", "quantity", "first_trade_id",
                        "last_trade_id", "transact_time", "is_buyer_maker",
                    ]
                else:
                    # No header — reset with names
                    df.columns = [
                        "agg_trade_id", "price", "quantity", "first_trade_id",
                        "last_trade_id", "transact_time", "is_buyer_maker",
                    ]

        df["price"] = df["price"].astype(float)
        df["quantity"] = df["quantity"].astype(float)
        df["notional_usd"] = df["price"] * df["quantity"]

        logger.info("  Downloaded %s: %d trades, $%.0fM notional",
                     date_str, len(df),
                     df["notional_usd"].sum() / 1e6)

        return df

    except Exception as e:
        logger.error("  Failed %s: %s", date_str, e)
        return None


def aggregate_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw trades into 1h bars with large/small separation."""
    df["dt"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df["hour"] = df["dt"].dt.floor("1h")

    # is_buyer_maker = True → taker is SELLER
    # is_buyer_maker = False → taker is BUYER
    df["is_taker_buy"] = ~df["is_buyer_maker"]
    df["is_large"] = df["notional_usd"] >= LARGE_THRESHOLD_USD

    rows = []
    for hour, group in df.groupby("hour"):
        large = group[group["is_large"]]
        small = group[~group["is_large"]]

        large_buy = large[large["is_taker_buy"]]["notional_usd"].sum()
        large_sell = large[~large["is_taker_buy"]]["notional_usd"].sum()
        small_buy = small[small["is_taker_buy"]]["notional_usd"].sum()
        small_sell = small[~small["is_taker_buy"]]["notional_usd"].sum()

        total = large_buy + large_sell + small_buy + small_sell

        rows.append({
            "dt": hour,
            "large_buy_usd": large_buy,
            "large_sell_usd": large_sell,
            "large_delta": large_buy - large_sell,
            "large_count": len(large),
            "small_buy_usd": small_buy,
            "small_sell_usd": small_sell,
            "small_delta": small_buy - small_sell,
            "small_count": len(small),
            "total_usd": total,
            "large_ratio": (large_buy + large_sell) / total if total > 0 else 0,
            "large_buy_ratio": large_buy / (large_buy + large_sell)
                               if (large_buy + large_sell) > 0 else 0.5,
            "avg_large_size": large["notional_usd"].mean() if len(large) > 0 else 0,
            "total_trades": len(group),
        })

    result = pd.DataFrame(rows).set_index("dt").sort_index()
    return result


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the aggregated 1h data."""
    # Large vs small delta divergence (the key feature)
    df["large_small_div"] = df["large_delta"] - df["small_delta"]

    # Normalized divergence (z-score over 24h)
    mu = df["large_small_div"].rolling(24, min_periods=4).mean()
    sd = df["large_small_div"].rolling(24, min_periods=4).std().replace(0, np.nan)
    df["large_small_div_zscore"] = (df["large_small_div"] - mu) / sd

    # Large delta as fraction of total delta
    total_delta = df["large_delta"] + df["small_delta"]
    df["large_delta_frac"] = df["large_delta"] / total_delta.replace(0, np.nan)

    # Large delta z-score
    mu_ld = df["large_delta"].rolling(24, min_periods=4).mean()
    sd_ld = df["large_delta"].rolling(24, min_periods=4).std().replace(0, np.nan)
    df["large_delta_zscore"] = (df["large_delta"] - mu_ld) / sd_ld

    # Large trade concentration: are large trades more one-sided than small?
    large_imb = (df["large_buy_usd"] - df["large_sell_usd"]) / \
                (df["large_buy_usd"] + df["large_sell_usd"]).replace(0, np.nan)
    small_imb = (df["small_buy_usd"] - df["small_sell_usd"]) / \
                (df["small_buy_usd"] + df["small_sell_usd"]).replace(0, np.nan)
    df["imbalance_divergence"] = large_imb - small_imb

    # Rolling momentum of large delta (4h / 8h)
    df["large_delta_ma_4h"] = df["large_delta"].rolling(4, min_periods=2).mean()
    df["large_delta_ma_8h"] = df["large_delta"].rolling(8, min_periods=2).mean()

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-17", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-04-01", help="End date (YYYY-MM-DD)")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    n_days = (end - start).days + 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Binance aggTrades: {args.start} ~ {args.end} ({n_days} days)")
    print(f"Large trade threshold: ${LARGE_THRESHOLD_USD:,.0f}")
    print()

    all_bars = []
    failed_days = []

    for i in range(n_days):
        date = start + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        # Check if already aggregated in cache
        agg_cache = CACHE_DIR / f"{date_str}_1h.parquet"
        if agg_cache.exists():
            bars = pd.read_parquet(agg_cache)
            all_bars.append(bars)
            logger.info("[%d/%d] %s: cached (%d bars)", i+1, n_days, date_str, len(bars))
            continue

        logger.info("[%d/%d] %s: downloading...", i+1, n_days, date_str)
        raw = download_day(date_str)

        if raw is None:
            failed_days.append(date_str)
            continue

        bars = aggregate_1h(raw)
        all_bars.append(bars)

        # Cache the aggregated result
        bars.to_parquet(agg_cache)

        logger.info("  → %d 1h bars, large_ratio=%.1f%%, large_delta=$%.0fK",
                     len(bars),
                     bars["large_ratio"].mean() * 100,
                     bars["large_delta"].sum() / 1000)

        # Rate limit
        time.sleep(0.5)

    if not all_bars:
        print("No data downloaded!")
        return

    # Combine all days
    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Add derived features
    combined = compute_derived_features(combined)

    print(f"\n{'='*60}")
    print(f"RESULT: {len(combined)} 1h bars")
    print(f"Range: {combined.index[0]} ~ {combined.index[-1]}")
    print(f"Failed days: {len(failed_days)}")
    if failed_days:
        print(f"  {failed_days[:10]}{'...' if len(failed_days) > 10 else ''}")

    # Stats
    print(f"\nLarge trade stats:")
    print(f"  Avg large_ratio: {combined['large_ratio'].mean():.1%}")
    print(f"  Avg large_count/bar: {combined['large_count'].mean():.0f}")
    print(f"  Avg large_delta: ${combined['large_delta'].mean():,.0f}")
    print(f"  Large_small_div std: ${combined['large_small_div'].std():,.0f}")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
