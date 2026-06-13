"""
Multi-Timeframe CVD Consensus features (Phase 1, Idea 1).

Hypothesis (user, 2026-05-12):
    Single-timeframe CVD is noisy.  Cross-timeframe agreement (short / med /
    long all moving same direction) is a stronger filter for the
    "v9 binary win" target, especially after seed sensitivity showed v9 is
    fragile on the current feature set.

Design:
    Base: cg_fcvd_delta (1h futures CVD).  Already z-scored in features_all
    as cg_fcvd_delta_zscore (this is the "short" / 1h leg).

    Add 12 features built from rolling aggregations over multiple horizons:

    Tier 1 — Futures CVD multi-TF z-scores:
        cvd_mtf_2h_z       — rolling 2-bar sum, z-scored (60-bar window)
        cvd_mtf_6h_z       — rolling 6-bar sum, z-scored
        cvd_mtf_24h_z      — rolling 24-bar sum, z-scored

    Tier 2 — Spot CVD multi-TF z-scores (different liquidity source):
        scvd_mtf_2h_z      — rolling 2-bar spot CVD
        scvd_mtf_6h_z      — rolling 6-bar
        scvd_mtf_24h_z     — rolling 24-bar

    Tier 3 — Consensus across timeframes:
        cvd_consensus_count    — 0-3, # of (1h, 6h, 24h) aligning with sign(1h)
        cvd_consensus_strength — signed sum of |z| when ALL aligned, else 0

    Tier 4 — Cross-TF dynamics:
        cvd_short_long_div     — 2h_z - 24h_z  (divergence: short leading or lagging long)
        cvd_spot_futures_div   — scvd_6h_z - cvd_6h_z  (spot leading futures, or vice versa)
        cvd_accel_short_med    — 2h_z - 6h_z   (acceleration of short vs medium)
        cvd_accel_med_long     — 6h_z - 24h_z  (acceleration of medium vs long)

All features are trailing-only (no look-ahead).

Note on 15min timeframe (user's original idea #1):
    Production 1h cache doesn't include sub-hour aggregates.  True 15min CVD
    requires re-aggregating from flow_bars_1m (Service 2 table) — leave for
    iteration 2 if multi-TF on 1h base already shows promise.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROLLING_NORM_WINDOW = 60  # 60h ≈ 2.5 days for z-score normalization

EXTRA_FEATURES = [
    "cvd_mtf_2h_z", "cvd_mtf_6h_z", "cvd_mtf_24h_z",
    "scvd_mtf_2h_z", "scvd_mtf_6h_z", "scvd_mtf_24h_z",
    "cvd_consensus_count", "cvd_consensus_strength",
    "cvd_short_long_div", "cvd_spot_futures_div",
    "cvd_accel_short_med", "cvd_accel_med_long",
]


def _rolling_z(s: pd.Series, window: int = ROLLING_NORM_WINDOW) -> pd.Series:
    """Trailing-only z-score over rolling window."""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return ((s - mean) / std.replace(0, np.nan)).fillna(0)


def add_multitf_cvd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment df with 12 multi-TF CVD features.

    Requires columns: cg_fcvd_delta, cg_scvd_delta
    Returns: a new DataFrame (does not modify input).
    """
    df = df.copy()

    if "cg_fcvd_delta" not in df.columns or "cg_scvd_delta" not in df.columns:
        logger.warning("Missing base CVD columns; multi-TF features will be NaN")
        for f in EXTRA_FEATURES:
            df[f] = np.nan
        return df

    f = df["cg_fcvd_delta"]   # 1h futures CVD net
    s = df["cg_scvd_delta"]   # 1h spot CVD net

    # Tier 1: futures CVD multi-TF z-scores
    df["cvd_mtf_2h_z"] = _rolling_z(f.rolling(2, min_periods=2).sum())
    df["cvd_mtf_6h_z"] = _rolling_z(f.rolling(6, min_periods=6).sum())
    df["cvd_mtf_24h_z"] = _rolling_z(f.rolling(24, min_periods=24).sum())

    # Tier 2: spot CVD multi-TF z-scores
    df["scvd_mtf_2h_z"] = _rolling_z(s.rolling(2, min_periods=2).sum())
    df["scvd_mtf_6h_z"] = _rolling_z(s.rolling(6, min_periods=6).sum())
    df["scvd_mtf_24h_z"] = _rolling_z(s.rolling(24, min_periods=24).sum())

    # Tier 3: Consensus across 1h / 6h / 24h
    sign_1h = np.sign(df["cg_fcvd_delta"])
    sign_6h = np.sign(df["cvd_mtf_6h_z"])
    sign_24h = np.sign(df["cvd_mtf_24h_z"])
    df["cvd_consensus_count"] = (
        (sign_1h == sign_1h).astype(int)        # always 1 (self)
        + (sign_6h == sign_1h).astype(int)
        + (sign_24h == sign_1h).astype(int)
    )  # 1, 2, or 3

    # consensus_strength: signed sum of |z| when all 3 align, else 0
    all_aligned = (
        (sign_1h == sign_6h) & (sign_6h == sign_24h) & (sign_1h != 0)
    )
    df["cvd_consensus_strength"] = np.where(
        all_aligned,
        sign_1h * (
            df["cg_fcvd_delta_zscore"].abs()
            + df["cvd_mtf_6h_z"].abs()
            + df["cvd_mtf_24h_z"].abs()
        ),
        0.0,
    )

    # Tier 4: Cross-TF dynamics
    df["cvd_short_long_div"] = df["cvd_mtf_2h_z"] - df["cvd_mtf_24h_z"]
    df["cvd_spot_futures_div"] = df["scvd_mtf_6h_z"] - df["cvd_mtf_6h_z"]
    df["cvd_accel_short_med"] = df["cvd_mtf_2h_z"] - df["cvd_mtf_6h_z"]
    df["cvd_accel_med_long"] = df["cvd_mtf_6h_z"] - df["cvd_mtf_24h_z"]

    return df


def summarize(df: pd.DataFrame) -> None:
    """Print sanity stats for the newly added features."""
    print(f"\nMulti-TF CVD features summary (n={len(df)}):")
    print(f"  {'feature':<28} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} "
          f"{'%NaN':>6}")
    for f in EXTRA_FEATURES:
        s = df[f]
        nan_pct = s.isna().mean() * 100
        print(f"  {f:<28} {s.mean():>+10.3f} {s.std():>10.3f} "
              f"{s.min():>+10.3f} {s.max():>+10.3f} {nan_pct:>5.1f}%")

    # Consensus distribution
    counts = df["cvd_consensus_count"].value_counts().sort_index()
    print(f"\n  cvd_consensus_count distribution:")
    for k, v in counts.items():
        pct = v / len(df) * 100
        print(f"    count={int(k)}: {v} ({pct:.1f}%)")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache_path = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
    df = pd.read_parquet(cache_path)
    df = add_multitf_cvd_features(df)
    summarize(df.dropna(subset=["cvd_mtf_24h_z"]))  # drop warmup
