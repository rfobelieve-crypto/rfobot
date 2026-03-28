"""
Open Interest feature computation.
Pure functions — no DB access, no side effects.
"""
from __future__ import annotations


def compute_oi(
    baseline_oi: float | None,
    snapshot_oi: float | None,
) -> dict:
    """
    Compute OI change features between window start and end.

    Args:
        baseline_oi: Combined OI notional (USD) near window start.
        snapshot_oi: Combined OI notional (USD) near window end.

    Returns:
        dict:
            oi_change      – absolute USD change
            oi_change_pct  – % change from baseline
            oi_direction   – "increasing" | "decreasing" | "flat" | None
    """
    if baseline_oi is None or snapshot_oi is None or baseline_oi == 0:
        return {"oi_change": None, "oi_change_pct": None, "oi_direction": None}

    change     = snapshot_oi - baseline_oi
    change_pct = round(change / baseline_oi * 100, 4)

    if change_pct > 0.5:
        direction = "increasing"
    elif change_pct < -0.5:
        direction = "decreasing"
    else:
        direction = "flat"

    return {
        "oi_change":     round(change, 4),
        "oi_change_pct": change_pct,
        "oi_direction":  direction,
    }
