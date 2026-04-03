"""
Magnitude label construction for the 4h magnitude model.

Targets:
  1. abs_return_4h       — |close[t+4] / close[t] - 1|  (regression)
  2. vol_adj_abs_return  — abs_return_4h / realized_vol  (normalized regression)
  3. large_move          — binary: 1 if vol_adj_abs > 1.0 (classification alt)

The magnitude model answers: "How big will the next 4h move be?"
regardless of direction.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_BARS = 4


def build_magnitude_labels(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    vol_col: str = "realized_vol_20b",
    large_move_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Build magnitude targets.

    Parameters
    ----------
    df : Feature DataFrame with 'close' column.
    horizon_bars : Forward horizon in bars.
    vol_col : Realized vol column for normalization.
    large_move_threshold : Vol-adjusted threshold for large_move binary.

    Returns
    -------
    DataFrame with columns:
        return_4h           : Raw forward return (signed)
        y_abs_return        : |return_4h|  (regression target)
        y_vol_adj_abs       : abs_return / vol  (vol-normalized regression target)
        y_large_move        : 1 if vol_adj_abs > threshold, else 0
    """
    close = df["close"].values.astype(float)
    n = len(close)

    return_4h = np.full(n, np.nan)
    for i in range(n - horizon_bars):
        return_4h[i] = close[i + horizon_bars] / close[i] - 1

    abs_return = np.abs(return_4h)

    if vol_col in df.columns:
        vol = df[vol_col].values.astype(float)
    else:
        returns = pd.Series(close).pct_change().values
        vol = pd.Series(returns).rolling(20, min_periods=10).std().values
        logger.warning("vol_col '%s' not found, computing from returns", vol_col)

    vol_safe = np.where(vol > 0, vol, np.nan)
    vol_adj_abs = abs_return / vol_safe

    large_move = np.where(np.isnan(vol_adj_abs), np.nan,
                          np.where(vol_adj_abs > large_move_threshold, 1.0, 0.0))

    result = pd.DataFrame({
        "return_4h": return_4h,
        "y_abs_return": abs_return,
        "y_vol_adj_abs": vol_adj_abs,
        "y_large_move": large_move,
    }, index=df.index)

    valid = ~np.isnan(abs_return)
    logger.info("Magnitude labels: %d valid, abs_return mean=%.4f, "
                "vol_adj_abs mean=%.2f, large_move rate=%.1f%%",
                valid.sum(), np.nanmean(abs_return),
                np.nanmean(vol_adj_abs),
                100 * np.nanmean(large_move[~np.isnan(large_move)]))

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from shared_data import load_and_cache_data

    df = load_and_cache_data()
    labels = build_magnitude_labels(df)

    print("\nMagnitude label stats:")
    print(labels.describe())
