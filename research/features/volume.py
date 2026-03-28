"""
Volume feature computation.
Pure functions — no DB access, no side effects.
"""
from __future__ import annotations
import pandas as pd


def compute_volume(
    df: pd.DataFrame | None,
    lookback_df: pd.DataFrame | None = None,
) -> dict:
    """
    Compute volume features for a time window.

    Args:
        df:          Current window's flow bars (delta_usd, volume_usd columns).
        lookback_df: Historical bars for relative volume baseline.
                     Each row should be one aggregated tf-bar.

    Returns:
        dict:
            volume_usd  – total traded USD volume in window
            rel_volume  – volume_usd / rolling_mean(lookback volumes)
                          None if no lookback data
    """
    if df is None or df.empty:
        return {"volume_usd": None, "rel_volume": None}

    volume_usd = float(df["volume_usd"].sum())

    rel_volume = None
    if lookback_df is not None and not lookback_df.empty and "volume_usd" in lookback_df.columns:
        avg = float(lookback_df["volume_usd"].mean())
        if avg > 0:
            rel_volume = round(volume_usd / avg, 4)

    return {
        "volume_usd": round(volume_usd, 4),
        "rel_volume": rel_volume,
    }
