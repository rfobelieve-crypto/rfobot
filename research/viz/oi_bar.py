"""OI change subplot trace. Reserved for future use."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from research.config.settings import ChartConfig


def make_oi_trace(score_df: pd.DataFrame, config: ChartConfig) -> list[go.Bar]:
    """
    OI change % bar trace. Green = increasing, red = decreasing.

    Args:
        score_df: DataFrame from market_state_repository (needs oi_change_pct)
        config:   ChartConfig
    """
    c = config.colors
    df = score_df.dropna(subset=["oi_change_pct"]).copy()
    if df.empty:
        return []

    df["oi_change_pct"] = pd.to_numeric(df["oi_change_pct"], errors="coerce").fillna(0)
    bar_colors = df["oi_change_pct"].apply(
        lambda v: c.reversal if v < 0 else c.continuation
    )

    return [go.Bar(
        x=df["timestamp"],
        y=df["oi_change_pct"],
        name="OI Δ%",
        marker_color=bar_colors,
        opacity=0.8,
        hovertemplate="<b>%{x}</b><br>OI Δ%: %{y:.2f}%<extra></extra>",
    )]
