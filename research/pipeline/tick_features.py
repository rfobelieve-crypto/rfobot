"""
Tick-level feature builder: process raw aggTrades into 15m bar features.

Features computed (per 15m bar):
  Category 1 — Basic Trade Stats:
    trade_count, avg_trade_size, total_volume, large_trade_ratio,
    inter_trade_dur_mean, inter_trade_dur_std

  Category 2 — Order Flow Imbalance (BVC + VPIN):
    bvc_buy_vol, bvc_sell_vol, bvc_delta, bvc_cvd,
    bvc_cvd_zscore, bvc_cvd_accel,
    vpin, vpin_ma, vpin_std, vpin_extreme,
    large_cluster_count, large_cluster_delta

  Category 3 — Volume Dynamics:
    vol_acceleration, price_impact, vol_entropy

Input:  market_data/raw_data/aggtrades/binance/{month}/BTCUSDT-aggTrades-{date}.csv
Output: research/ml_data/BTC_USD_15m_tick_features.parquet

Usage:
    python -m research.pipeline.tick_features
    python -m research.pipeline.tick_features --dry-run
"""
from __future__ import annotations

import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
AGG_DIR    = ROOT / "market_data" / "raw_data" / "aggtrades" / "binance"
OUT_PATH   = ROOT / "research" / "ml_data" / "BTC_USD_15m_tick_features.parquet"

# BVC bucket size (USD notional per bucket for VPIN)
VPIN_BUCKET_N  = 50       # number of buckets for VPIN
LARGE_THR_BTC  = 1.0      # "large trade" threshold in BTC
CLUSTER_GAP_MS = 500      # trades within 500ms = same cluster
ZSCORE_WIN     = 96        # 96 x 15m = 24h


def _load_all_aggtrades() -> pd.DataFrame:
    """Load all aggTrade CSVs, sorted by time."""
    frames = []
    for month_dir in sorted(AGG_DIR.iterdir()):
        if not month_dir.is_dir():
            continue
        csv_files = sorted(month_dir.glob("BTCUSDT-aggTrades-*.csv"))
        logger.info("  Loading %s: %d files", month_dir.name, len(csv_files))
        for f in csv_files:
            df = pd.read_csv(f, usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
            frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("transact_time").reset_index(drop=True)
    logger.info("Total aggTrades loaded: %d rows", len(full))
    return full


def _bvc_sign(prices: np.ndarray) -> np.ndarray:
    """
    Bulk Volume Classification: assign trade sign based on price change.
    Uses tick rule: if price up from previous trade -> buy, down -> sell.
    Zero change -> carry forward previous sign.
    """
    dp = np.diff(prices, prepend=prices[0])
    signs = np.sign(dp)
    # Forward-fill zeros
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    # First trade: use is_buyer_maker as fallback (handled in caller)
    return signs


def compute_bar_features(trades: pd.DataFrame) -> dict:
    """Compute all tick-level features for a single 15m bar."""
    n = len(trades)
    if n == 0:
        return {}

    prices   = trades["price"].values
    qty      = trades["quantity"].values
    times    = trades["transact_time"].values
    is_maker = trades["is_buyer_maker"].values
    notional = prices * qty

    # ── Category 1: Basic Trade Stats ────────────────────────────────────
    total_vol    = qty.sum()
    trade_count  = n
    avg_size     = qty.mean()
    large_mask   = qty >= LARGE_THR_BTC
    large_ratio  = large_mask.mean()

    # Inter-trade duration (ms)
    if n > 1:
        durations = np.diff(times).astype(float)
        dur_mean  = durations.mean()
        dur_std   = durations.std()
    else:
        dur_mean = dur_std = 0.0

    # ── Category 2: BVC + signed flow ────────────────────────────────────
    # Use exchange-provided taker side (is_buyer_maker=False means taker is buyer)
    taker_buy = ~is_maker
    buy_vol   = qty[taker_buy].sum()
    sell_vol  = qty[~taker_buy].sum()

    # BVC classification (alternative signed volume using price changes)
    bvc_signs     = _bvc_sign(prices)
    bvc_buy_mask  = bvc_signs > 0
    bvc_buy_vol   = qty[bvc_buy_mask].sum()
    bvc_sell_vol  = qty[~bvc_buy_mask].sum()
    bvc_delta     = bvc_buy_vol - bvc_sell_vol

    # Also compute taker-based delta for comparison
    taker_delta = buy_vol - sell_vol

    # VPIN (Volume-synchronized Probability of Informed Trading)
    # Split volume into equal-sized buckets, measure abs(buy-sell)/total per bucket
    bucket_size = max(total_vol / VPIN_BUCKET_N, 0.001)
    cum_vol = np.cumsum(qty)
    bucket_ids = (cum_vol / bucket_size).astype(int)
    n_buckets = bucket_ids[-1] + 1 if len(bucket_ids) > 0 else 1

    if n_buckets > 1:
        bucket_buy  = np.zeros(n_buckets)
        bucket_sell = np.zeros(n_buckets)
        for i in range(n):
            bid = min(bucket_ids[i], n_buckets - 1)
            if taker_buy[i]:
                bucket_buy[bid] += qty[i]
            else:
                bucket_sell[bid] += qty[i]
        bucket_total = bucket_buy + bucket_sell
        bucket_imbal = np.where(bucket_total > 0,
                                np.abs(bucket_buy - bucket_sell) / bucket_total, 0)
        vpin = bucket_imbal.mean()
    else:
        vpin = abs(bvc_delta) / max(total_vol, 0.001)

    # Large trade clusters
    if large_mask.any():
        large_times = times[large_mask]
        large_qty   = qty[large_mask]
        large_buy   = taker_buy[large_mask]

        # Cluster by time proximity
        if len(large_times) > 1:
            gaps = np.diff(large_times)
            cluster_breaks = np.where(gaps > CLUSTER_GAP_MS)[0] + 1
            cluster_ids = np.zeros(len(large_times), dtype=int)
            for idx, brk in enumerate(cluster_breaks):
                cluster_ids[brk:] = idx + 1
            n_clusters = cluster_ids[-1] + 1

            # Signed cluster delta
            cluster_delta = 0.0
            for c in range(n_clusters):
                mask_c = cluster_ids == c
                c_buy  = large_qty[mask_c & large_buy].sum()
                c_sell = large_qty[mask_c & ~large_buy].sum()
                cluster_delta += (c_buy - c_sell)
        else:
            n_clusters = 1
            cluster_delta = large_qty[0] if large_buy[0] else -large_qty[0]
    else:
        n_clusters = 0
        cluster_delta = 0.0

    # ── Category 3: Volume Dynamics ──────────────────────────────────────
    # Price impact = price range / volume (how much price moves per unit volume)
    price_range = prices.max() - prices.min()
    price_impact = price_range / max(total_vol, 0.001)

    # Volume entropy (distribution of trade sizes)
    if n > 1:
        # Discretize trade sizes into 10 bins
        try:
            hist, _ = np.histogram(qty, bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            vol_entropy = -np.sum(hist * np.log(hist))
        except Exception:
            vol_entropy = 0.0
    else:
        vol_entropy = 0.0

    return {
        # Cat 1: Basic stats
        "trade_count":          trade_count,
        "avg_trade_size":       round(avg_size, 6),
        "tick_total_volume":    round(total_vol, 4),
        "large_trade_ratio":    round(large_ratio, 6),
        "inter_trade_dur_mean": round(dur_mean, 2),
        "inter_trade_dur_std":  round(dur_std, 2),
        # Cat 2: BVC signed flow
        "bvc_buy_vol":          round(bvc_buy_vol, 4),
        "bvc_sell_vol":         round(bvc_sell_vol, 4),
        "bvc_delta":            round(bvc_delta, 4),
        "taker_delta":          round(taker_delta, 4),
        "vpin":                 round(vpin, 6),
        "large_cluster_count":  n_clusters,
        "large_cluster_delta":  round(cluster_delta, 4),
        # Cat 3: Volume dynamics
        "price_impact":         round(price_impact, 6),
        "vol_entropy":          round(vol_entropy, 6),
    }


def build_all_bars(trades: pd.DataFrame) -> pd.DataFrame:
    """Group trades into 15m bars and compute features."""
    trades["dt"] = pd.to_datetime(trades["transact_time"], unit="ms", utc=True)
    trades["bar"] = trades["dt"].dt.floor("15min")

    logger.info("Computing tick features per 15m bar...")
    results = []
    groups = trades.groupby("bar")
    total = len(groups)

    for i, (bar_time, grp) in enumerate(groups):
        feats = compute_bar_features(grp)
        if feats:
            feats["ts_open"] = int(bar_time.timestamp() * 1000)
            feats["dt"] = bar_time
            results.append(feats)
        if (i + 1) % 1000 == 0:
            logger.info("  %d / %d bars processed", i + 1, total)

    df = pd.DataFrame(results).sort_values("dt").reset_index(drop=True)

    # ── Rolling / derived features ───────────────────────────────────────
    # BVC CVD (cumulative, daily reset at 00:00 UTC)
    df["bvc_cvd"] = df.groupby(df["dt"].dt.date)["bvc_delta"].cumsum()

    # CVD z-score (rolling 24h)
    df["bvc_cvd_zscore"] = (
        (df["bvc_cvd"] - df["bvc_cvd"].rolling(ZSCORE_WIN, min_periods=4).mean())
        / df["bvc_cvd"].rolling(ZSCORE_WIN, min_periods=4).std().replace(0, np.nan)
    )

    # CVD acceleration (delta of delta)
    df["bvc_delta_accel"] = df["bvc_delta"].diff()

    # VPIN rolling stats
    df["vpin_ma"]  = df["vpin"].rolling(ZSCORE_WIN, min_periods=4).mean()
    df["vpin_std"] = df["vpin"].rolling(ZSCORE_WIN, min_periods=4).std()
    df["vpin_extreme"] = (df["vpin"] > 0.7).astype(int)

    # Volume acceleration (short vs long volume ratio)
    vol_short = df["tick_total_volume"].rolling(4, min_periods=1).mean()   # 1h
    vol_long  = df["tick_total_volume"].rolling(24, min_periods=4).mean()  # 6h
    df["vol_acceleration"] = vol_short / vol_long.replace(0, np.nan)

    # Drop temp column
    df = df.drop(columns=["dt"])

    logger.info("Final tick features: %d bars x %d columns", len(df), len(df.columns))
    return df


def run(dry_run: bool = False):
    logger.info("=== Tick Feature Builder ===")

    trades = _load_all_aggtrades()
    df = build_all_bars(trades)

    logger.info("\nFeature summary:")
    for col in sorted(df.columns):
        if col == "ts_open":
            continue
        s = df[col]
        logger.info("  %-25s  mean=%10.4f  std=%10.4f  NaN=%.1f%%",
                     col, s.mean(), s.std(), s.isna().mean() * 100)

    if dry_run:
        logger.info("DRY RUN — not saving.")
        return df

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved to %s", OUT_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
