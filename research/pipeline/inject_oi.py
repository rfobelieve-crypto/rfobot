"""
Inject OI features from Binance metrics parquet into the tabular parquet.

Reads:  market_data/raw_data/metrics/BTCUSDT_metrics.parquet  (5-min intervals)
Updates: research/ml_data/BTC_USD_15m_tabular.parquet

OI features added/updated:
  - oi               : OI in USD (sum_open_interest_value)
  - oi_delta         : bar-over-bar change in OI
  - oi_accel         : delta of delta (acceleration)
  - oi_divergence    : CVD direction vs OI direction mismatch
  - cvd_x_oi_delta   : cvd_zscore * oi_delta (interaction)
  - cvd_oi_ratio     : CVD / OI (normalized)
  - toptrader_ls     : top trader long/short ratio
  - taker_ls         : taker buy/sell volume ratio

Metrics are 5-min; we resample to 15-min (last value per bar).

Usage:
    python -m research.pipeline.inject_oi
    python -m research.pipeline.inject_oi --dry-run
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
METRICS_IN  = ROOT / "market_data" / "raw_data" / "metrics" / "BTCUSDT_metrics.parquet"
TABULAR     = ROOT / "research" / "ml_data" / "BTC_USD_15m_tabular.parquet"
ZSCORE_WIN  = 96  # 96 x 15m = 24h


def run(dry_run: bool = False):
    # ── Load metrics ─────────────────────────────────────────────────────────
    metrics = pd.read_parquet(METRICS_IN)
    metrics["create_time"] = pd.to_datetime(metrics["create_time"], utc=True)
    metrics = metrics.sort_values("create_time").drop_duplicates("create_time")

    logger.info("Metrics loaded: %d rows  %s ~ %s",
                len(metrics), metrics["create_time"].min(), metrics["create_time"].max())

    # Resample 5min -> 15min (use last value in each 15m window)
    metrics = metrics.set_index("create_time")
    oi_15m = metrics.resample("15min").last().dropna(subset=["sum_open_interest_value"])

    logger.info("After 15m resample: %d rows", len(oi_15m))

    # ── Load tabular ─────────────────────────────────────────────────────────
    tab = pd.read_parquet(TABULAR)
    tab["dt"] = pd.to_datetime(tab["ts_open"], unit="ms", utc=True)
    n_before = tab["oi"].notna().sum()

    logger.info("Tabular loaded: %d rows, OI non-NaN before: %d", len(tab), n_before)

    # ── ASOF merge: align metrics to bar open time ───────────────────────────
    tab_sorted = tab.sort_values("dt")
    oi_sorted  = oi_15m.reset_index().rename(columns={"create_time": "metrics_time"})
    oi_sorted  = oi_sorted.sort_values("metrics_time")

    merged = pd.merge_asof(
        tab_sorted[["dt"]],
        oi_sorted[["metrics_time", "sum_open_interest_value",
                    "sum_open_interest",
                    "count_toptrader_long_short_ratio",
                    "sum_taker_long_short_vol_ratio"]],
        left_on="dt",
        right_on="metrics_time",
        direction="backward",
        tolerance=pd.Timedelta("20min"),  # max 20min staleness
    )

    # ── Compute OI features ──────────────────────────────────────────────────
    oi_usd = merged["sum_open_interest_value"].values.astype(float)

    # OI delta (bar-over-bar change)
    oi_delta = np.diff(oi_usd, prepend=np.nan)
    # First bar has no delta
    oi_delta[0] = np.nan

    # OI acceleration (delta of delta)
    oi_accel = np.diff(oi_delta, prepend=np.nan)
    oi_accel[0] = np.nan

    # CVD-OI divergence: sign(CVD change) != sign(OI change)
    # positive divergence = CVD up but OI down (short covering / squeeze signal)
    cvd_vals = tab_sorted["cvd"].values if "cvd" in tab_sorted.columns else np.full(len(tab_sorted), np.nan)
    cvd_delta = np.diff(cvd_vals, prepend=np.nan)
    oi_divergence = np.sign(cvd_delta) * -np.sign(oi_delta)
    # +1 = CVD up + OI down (short covering) or CVD down + OI up (new shorts)
    # -1 = CVD & OI same direction (trend confirmation)

    # Cross features
    cvd_zscore = tab_sorted["cvd_zscore"].values if "cvd_zscore" in tab_sorted.columns else np.full(len(tab_sorted), np.nan)
    cvd_x_oi_delta = cvd_zscore * oi_delta

    # CVD / OI ratio (normalized)
    cvd_oi_ratio = np.where(oi_usd > 0, cvd_vals / oi_usd, np.nan)

    # ── Write back ───────────────────────────────────────────────────────────
    tab_sorted["oi"]             = oi_usd
    tab_sorted["oi_delta"]       = oi_delta
    tab_sorted["oi_accel"]       = oi_accel
    tab_sorted["oi_divergence"]  = oi_divergence
    tab_sorted["cvd_x_oi_delta"] = cvd_x_oi_delta
    tab_sorted["cvd_oi_ratio"]   = cvd_oi_ratio

    # Bonus features from metrics
    tab_sorted["toptrader_ls"]   = merged["count_toptrader_long_short_ratio"].values.astype(float)
    tab_sorted["taker_ls"]       = merged["sum_taker_long_short_vol_ratio"].values.astype(float)

    # Restore original order
    tab_sorted = tab_sorted.sort_index()

    n_after = tab_sorted["oi"].notna().sum()
    coverage = n_after / len(tab_sorted) * 100

    logger.info("OI coverage: %d / %d (%.1f%%)", n_after, len(tab_sorted), coverage)
    logger.info("OI stats: mean=%.0f  std=%.0f  min=%.0f  max=%.0f",
                tab_sorted["oi"].mean(), tab_sorted["oi"].std(),
                tab_sorted["oi"].min(), tab_sorted["oi"].max())
    logger.info("oi_delta stats: mean=%.0f  std=%.0f",
                tab_sorted["oi_delta"].mean(), tab_sorted["oi_delta"].std())
    logger.info("toptrader_ls: mean=%.3f  taker_ls: mean=%.3f",
                tab_sorted["toptrader_ls"].mean(), tab_sorted["taker_ls"].mean())

    if dry_run:
        logger.info("DRY RUN — not saving.")
        return

    # Drop temp column and save
    out = tab_sorted.drop(columns=["dt"])
    out.to_parquet(TABULAR, index=False)
    logger.info("Saved to %s", TABULAR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
