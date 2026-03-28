"""
Statistical features: rolling bands, z-score.

Dynamic Bands (mean ± Nσ) live HERE — not in bar_generator, not in score_engine.
These are statistical features applied to a score series.
"""
from __future__ import annotations
import pandas as pd


def compute_bands(
    series: pd.Series,
    window: int = 20,
    n_sigma: float = 2.0,
) -> dict:
    """
    Compute rolling mean ± n_sigma bands and z-score for the latest value.

    Uses the tail of `series` (up to `window` values) as the rolling window.
    The last value in `series` is treated as the current bar's score.

    Args:
        series:  Time-ordered Series of score values (oldest → newest).
                 Should include the current bar's score as the last element.
        window:  Rolling window size (number of bars).
        n_sigma: Number of standard deviations for band width.

    Returns:
        dict:
            rolling_mean – mean of the rolling window
            rolling_std  – std of the rolling window
            upper_band   – rolling_mean + n_sigma * rolling_std
            lower_band   – rolling_mean − n_sigma * rolling_std
            z_score      – (current − rolling_mean) / rolling_std
    """
    empty = {
        "rolling_mean": None,
        "rolling_std":  None,
        "upper_band":   None,
        "lower_band":   None,
        "z_score":      None,
    }

    if series is None or len(series) < 2:
        return empty

    s = pd.Series(series).dropna()
    if len(s) < 2:
        return empty

    tail    = s.tail(window)
    mean    = float(tail.mean())
    std     = float(tail.std(ddof=1)) if len(tail) >= 2 else 0.0
    current = float(s.iloc[-1])

    return {
        "rolling_mean": round(mean, 4),
        "rolling_std":  round(std, 4),
        "upper_band":   round(mean + n_sigma * std, 4),
        "lower_band":   round(mean - n_sigma * std, 4),
        "z_score":      round((current - mean) / std, 4) if std > 0 else 0.0,
    }
