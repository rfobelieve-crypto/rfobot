"""
Direction label construction for the 4h direction model.

Labeling strategy: deadzone (vol-adjusted threshold).
  - UP   (1) if return_4h >  k * realized_vol
  - DOWN (0) if return_4h < -k * realized_vol
  - NaN       if |return_4h| <= k * realized_vol (excluded from training)

This avoids training on noisy, near-zero returns that have no
tradeable directional signal.
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

    # Forward return (no look-ahead: label at time t uses close at t+horizon)
    return_4h = np.full(n, np.nan)
    for i in range(n - horizon_bars):
        return_4h[i] = close[i + horizon_bars] / close[i] - 1

    # Volatility for threshold
    if vol_col in df.columns:
        vol = df[vol_col].values.astype(float)
    else:
        # Fallback: compute realized vol from returns
        returns = pd.Series(close).pct_change().values
        vol = pd.Series(returns).rolling(20, min_periods=10).std().values
        logger.warning("vol_col '%s' not found, computing from returns", vol_col)

    threshold = k * vol

    # Label assignment
    y_dir = np.full(n, np.nan)
    y_dir[return_4h > threshold] = 1.0
    y_dir[return_4h < -threshold] = 0.0
    # Remaining NaN = deadzone (ambiguous direction, excluded)

    result = pd.DataFrame({
        "y_dir": y_dir,
        "return_4h": return_4h,
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
