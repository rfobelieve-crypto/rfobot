"""
CVD (Cumulative Volume Delta) feature computation.
Pure functions — no DB access, no side effects.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_cvd(
    lookback_df: pd.DataFrame | None,
    window_df: pd.DataFrame | None,
) -> dict:
    """
    Compute CVD momentum features.

    CVD change  = net delta (buy − sell) summed over the current window.
                  This equals the change in cumulative delta during the window.

    CVD slope   = linear regression slope over the lookback CVD series.
                  Positive = upward momentum, negative = downward.

    CVD flip    = 1 if slope sign reversed compared to previous bar's slope,
                  else 0.  Indicates momentum inflection.

    Args:
        lookback_df: Historical tf-aggregated bars (delta_usd column).
                     Used to compute rolling CVD slope.
        window_df:   Current window's 1m flow bars (delta_usd column).

    Returns:
        dict:
            cvd_change – net delta in current window (USD)
            cvd_slope  – linear regression slope of CVD over lookback
            cvd_flip   – 1 if momentum direction flipped, else 0, else None
    """
    result: dict = {"cvd_change": None, "cvd_slope": None, "cvd_flip": None}

    # CVD change: sum of deltas within the window
    if window_df is not None and not window_df.empty and "delta_usd" in window_df.columns:
        result["cvd_change"] = round(float(window_df["delta_usd"].sum()), 4)

    # CVD slope + flip: need at least 3 lookback bars
    if lookback_df is None or len(lookback_df) < 3 or "delta_usd" not in lookback_df.columns:
        return result

    # Cumulative sum of lookback deltas represents the CVD trajectory
    cvd_series = lookback_df["delta_usd"].cumsum().values
    x = np.arange(len(cvd_series), dtype=float)

    # Linear regression slope
    slope = float(np.polyfit(x, cvd_series, 1)[0])
    result["cvd_slope"] = round(slope, 4)

    # Flip detection: compare slope of last half vs first half
    if len(cvd_series) >= 4:
        mid = len(cvd_series) // 2
        first_half = cvd_series[:mid]
        second_half = cvd_series[mid:]
        x_half = np.arange(len(first_half), dtype=float)
        slope_first  = np.polyfit(x_half, first_half, 1)[0]
        x_half2 = np.arange(len(second_half), dtype=float)
        slope_second = np.polyfit(x_half2, second_half, 1)[0]
        flip = (slope_first > 0) != (slope_second > 0)
        result["cvd_flip"] = 1 if flip else 0

    return result
