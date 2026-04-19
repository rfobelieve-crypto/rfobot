"""
Direction label construction for the 4h direction model.

Labeling strategy: deadzone (vol-adjusted threshold) on PATH-INTEGRATED return.

Label metric (new, 2026-04-13):
    path_return_4h[t] = mean(close[t+1..t+4]) / close[t] - 1

This is equivalent to "average unrealized PnL over the next 4 hours if you
opened a position at close[t]". Unlike the endpoint return close[t+4]/close[t],
it is path-dependent: whipsaw paths that happen to end at the same price as
a clean trend get a smaller magnitude, so the model learns to distinguish
sustained moves from noise that ends at the right price.

    - UP   (1) if path_return_4h >  k * realized_vol
    - DOWN (0) if path_return_4h < -k * realized_vol
    - NaN       if |path_return_4h| <= k * realized_vol (excluded from training)

Two return columns are exposed:
    path_return_4h : used for y_dir labeling (TWAP-based, path-dependent)
    return_4h      : endpoint close[t+4]/close[t] - 1, kept for IC tracking
                     and compatibility with production outcome_tracker which
                     measures realized PnL at the endpoint.

Note: TWAP magnitudes are ~0.5x endpoint magnitudes for a linear trend, so k
calibrated for endpoint (k=0.5) will be too strict for TWAP. Sweep k to keep
UP/DOWN frequencies comparable to the endpoint baseline (~35% each).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_BARS = 4  # 4h = 4 x 1h


def build_direction_labels(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    k: float = 0.5,
    vol_col: str = "realized_vol_20b",
) -> pd.DataFrame:
    """
    Build deadzone direction labels.

    Parameters
    ----------
    df : Feature DataFrame with 'close' column and datetime index.
    horizon_bars : Forward prediction horizon in bars (4 = 4h).
    k : Deadzone multiplier. Higher k = wider deadzone = fewer but cleaner samples.
    vol_col : Column name for realized volatility.

    Returns
    -------
    DataFrame with columns:
        y_dir          : 1=UP, 0=DOWN, NaN=deadzone (excluded)
        return_4h      : Raw forward return
        threshold      : Per-bar deadzone threshold
        vol            : Realized volatility used
    """
    close = df["close"].values.astype(float)
    n = len(close)

    # Endpoint return: close[t+horizon]/close[t] - 1.
    # Kept for production outcome_tracker compatibility and IC tracking.
    # Not used for label assignment any more.
    return_4h = np.full(n, np.nan)
    for i in range(n - horizon_bars):
        return_4h[i] = close[i + horizon_bars] / close[i] - 1

    # Path-integrated return (TWAP-style): mean(close[t+1..t+horizon]) / close[t] - 1.
    # Path-dependent — whipsaws get smaller magnitude than clean trends.
    # This is the metric used for y_dir labeling.
    path_return_4h = np.full(n, np.nan)
    for i in range(n - horizon_bars):
        future_sum = 0.0
        for k_off in range(1, horizon_bars + 1):
            future_sum += close[i + k_off]
        path_return_4h[i] = (future_sum / horizon_bars) / close[i] - 1

    # Volatility for threshold
    if vol_col in df.columns:
        vol = df[vol_col].values.astype(float)
    else:
        # Fallback: compute realized vol from returns
        returns = pd.Series(close).pct_change().values
        vol = pd.Series(returns).rolling(20, min_periods=10).std().values
        logger.warning("vol_col '%s' not found, computing from returns", vol_col)

    threshold = k * vol

    # Label assignment against PATH return (not endpoint)
    y_dir = np.full(n, np.nan)
    y_dir[path_return_4h > threshold] = 1.0
    y_dir[path_return_4h < -threshold] = 0.0
    # Remaining NaN = deadzone (ambiguous direction, excluded)

    result = pd.DataFrame({
        "y_dir": y_dir,
        "return_4h": return_4h,              # endpoint, kept for compatibility
        "path_return_4h": path_return_4h,    # TWAP-based, used for y_dir
        "threshold": threshold,
        "vol": vol,
    }, index=df.index)

    # Stats
    n_up = int(np.nansum(y_dir == 1))
    n_down = int(np.nansum(y_dir == 0))
    n_dead = int(np.isnan(y_dir).sum()) - horizon_bars  # exclude tail NaN
    total = n_up + n_down + n_dead
    logger.info("Direction labels (k=%.2f): UP=%d (%.1f%%) DOWN=%d (%.1f%%) "
                "deadzone=%d (%.1f%%) tail_nan=%d",
                k, n_up, 100 * n_up / max(total, 1),
                n_down, 100 * n_down / max(total, 1),
                n_dead, 100 * n_dead / max(total, 1),
                horizon_bars)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from shared_data import load_and_cache_data

    df = load_and_cache_data()
    labels = build_direction_labels(df)

    print("\nLabel distribution:")
    print(labels["y_dir"].value_counts(dropna=False))
    print(f"\nReturn 4h stats (labeled samples only):")
    mask = labels["y_dir"].notna()
    print(labels.loc[mask, "return_4h"].describe())
