"""
Delta feature computation.
Pure functions — no DB access, no side effects.
"""
from __future__ import annotations
import pandas as pd


def compute_delta(df: pd.DataFrame | None) -> dict:
    """
    Compute delta features from a window's flow bars.

    Args:
        df: DataFrame with columns [delta_usd, volume_usd].
            May be None or empty if no flow data exists.

    Returns:
        dict:
            delta_usd       – net USD flow (buy − sell)
            delta_ratio     – delta_usd / volume_usd  (−1 to +1 imbalance ratio)
            delta_direction – "buy" | "sell" | "neutral" | None
    """
    if df is None or df.empty:
        return {"delta_usd": None, "delta_ratio": None, "delta_direction": None}

    delta_usd  = float(df["delta_usd"].sum())
    volume_usd = float(df["volume_usd"].sum())

    if volume_usd <= 0:
        return {"delta_usd": round(delta_usd, 4), "delta_ratio": None, "delta_direction": None}

    ratio = delta_usd / volume_usd

    if ratio > 0.03:
        direction = "buy"
    elif ratio < -0.03:
        direction = "sell"
    else:
        direction = "neutral"

    return {
        "delta_usd":       round(delta_usd, 4),
        "delta_ratio":     round(ratio, 6),
        "delta_direction": direction,
    }
