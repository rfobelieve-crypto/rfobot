"""
Direction REGRESSION label construction (4h horizon, path-integrated).

Produces a continuous signed target — the 4h path-integrated (TWAP) return.
The model learns the full return distribution, and Strong signals are
extracted at inference time by thresholding the predicted value.

Label (path / TWAP):
    y_path_ret_4h[t] = mean(close[t+1..t+4]) / close[t] - 1

Why path return instead of endpoint:
    Endpoint return (close[t+4]/close[t]-1) treats a whipsaw that happens to
    close at the same price as a clean trend identically. Path return assigns
    smaller magnitude to whipsaws (the mean of the intra-horizon path is
    pulled back toward close[t] when the path oscillates), so the model
    learns to distinguish sustained moves from noise. For a linear trend,
    TWAP magnitude is ~0.5x endpoint, so the Strong threshold (±1.0%) is
    calibrated against this TWAP scale, not endpoint scale.

Design notes:
    - No deadzone — regression uses every labeled bar.
    - Tail bars (last 4 rows) get NaN via forward close indexing and are
      excluded from training by the notna() mask.
    - No look-ahead: close[t+k] for k in [1, 4] is strictly future-only and
      only used where it has already materialized.
    - `endpoint_ret_4h` is also returned for diagnostic parity with the
      production outcome tracker (which measures realized PnL at endpoint).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON_BARS = 4


def build_direction_reg_labels(
    df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
) -> pd.DataFrame:
    """
    Build signed 4h path-integrated return as regression target.

    Parameters
    ----------
    df : Feature DataFrame with 'close' column and datetime index.
    horizon_bars : Forward horizon in 1h bars (default 4).

    Returns
    -------
    DataFrame with:
        y_path_ret_4h : mean(close[t+1..t+h]) / close[t] - 1  (TWAP, signed)
                        THIS IS THE TRAINING TARGET.
        endpoint_ret_4h : close[t+h] / close[t] - 1
                          kept for outcome-tracker parity and IC diagnostics.
        abs_path_ret  : |y_path_ret_4h|
    """
    close = df["close"].values.astype(float)
    n = len(close)

    y_path = np.full(n, np.nan)
    y_endpoint = np.full(n, np.nan)

    for i in range(n - horizon_bars):
        # Path return: arithmetic mean of the forward close[1..h] over close[0]
        future_sum = 0.0
        for k_off in range(1, horizon_bars + 1):
            future_sum += close[i + k_off]
        y_path[i] = (future_sum / horizon_bars) / close[i] - 1.0

        # Endpoint return (diagnostic only)
        y_endpoint[i] = close[i + horizon_bars] / close[i] - 1.0

    result = pd.DataFrame({
        "y_path_ret_4h": y_path,
        "endpoint_ret_4h": y_endpoint,
        "abs_path_ret": np.abs(y_path),
    }, index=df.index)

    valid = np.isfinite(y_path)
    if valid.any():
        logger.info(
            "Direction reg labels (PATH/TWAP): n=%d valid=%d "
            "mean=%+.5f std=%.5f | P(|y|>=0.006)=%.1f%% "
            "P(|y|>=0.008)=%.1f%% P(|y|>=0.010)=%.1f%% P(|y|>=0.015)=%.1f%% "
            "| endpoint/path ratio=%.2f",
            n, int(valid.sum()),
            float(np.nanmean(y_path)), float(np.nanstd(y_path)),
            100 * float(np.nanmean(np.abs(y_path[valid]) >= 0.006)),
            100 * float(np.nanmean(np.abs(y_path[valid]) >= 0.008)),
            100 * float(np.nanmean(np.abs(y_path[valid]) >= 0.010)),
            100 * float(np.nanmean(np.abs(y_path[valid]) >= 0.015)),
            float(np.nanstd(y_endpoint[valid]) / max(np.nanstd(y_path[valid]), 1e-9)),
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from research.dual_model.shared_data import load_and_cache_data

    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)

    print("\nLabel describe (valid rows):")
    print(labels.dropna().describe())
