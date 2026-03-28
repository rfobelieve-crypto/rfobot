"""CVD change subplot trace. Reserved for future use."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from research.config.settings import ChartConfig


def make_cvd_trace(score_df: pd.DataFrame, config: ChartConfig) -> list[go.Scatter]:
    """
    CVD change line trace with zero baseline.

    Args:
        score_df: DataFrame from market_state_repository (needs cvd_change)
        config:   ChartConfig
    """
    df = score_df.dropna(subset=["cvd_change"]).copy()
    if df.empty:
        return []

    df["cvd_change"] = pd.to_numeric(df["cvd_change"], errors="coerce").fillna(0)
    c = config.colors

    line = go.Scatter(
        x=df["timestamp"],
        y=df["cvd_change"],
        mode="lines",
        line=dict(color="#42a5f5", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(66,165,245,0.12)",
        name="CVD Δ",
        hovertemplate="<b>%{x}</b><br>CVD Δ: $%{y:,.0f}<extra></extra>",
    )

    zero = go.Scatter(
        x=[df["timestamp"].iloc[0], df["timestamp"].iloc[-1]],
        y=[0, 0],
        mode="lines",
        line=dict(color=c.grid_color, width=1, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    )

    return [zero, line]
