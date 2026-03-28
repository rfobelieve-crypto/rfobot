"""
Score bar trace + dynamic band traces.
Pure rendering — no calculations, no DB access.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from research.config.settings import ChartConfig


def make_score_trace(score_df: pd.DataFrame, config: ChartConfig) -> list[go.Bar]:
    """
    Build score bar traces, one per bias group (reversal / continuation / neutral).
    Grouping allows independent color + legend entries per bias.

    Args:
        score_df: DataFrame from market_state_repository.query_bars()
        config:   ChartConfig

    Returns:
        List of go.Bar traces (one per bias value present in data)
    """
    field  = config.score_field
    colors = config.colors

    bias_color = {
        "reversal":     colors.reversal,
        "continuation": colors.continuation,
        "neutral":      colors.neutral,
    }

    df = score_df.copy()

    # Fill missing score with 0 so bars are always present
    if field not in df.columns:
        df[field] = 0.0
    df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0)

    if "final_bias" not in df.columns:
        df["final_bias"] = "neutral"
    df["final_bias"] = df["final_bias"].fillna("neutral")

    traces = []
    for bias, color in bias_color.items():
        mask = df["final_bias"] == bias
        if not mask.any():
            continue

        sub = df[mask]
        traces.append(go.Bar(
            x=sub["timestamp"],
            y=sub[field],
            name=bias.capitalize(),
            marker_color=color,
            opacity=0.85,
            showlegend=True,
            customdata=sub[["reversal_score", "continuation_score",
                            "confidence", "final_bias",
                            "upper_band", "lower_band",
                            "event_count"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{field}: %{{y:.1f}}<br>"
                "rev: %{customdata[0]:.2f}  "
                "cont: %{customdata[1]:.2f}<br>"
                "conf: %{customdata[2]:.2f}<br>"
                "bias: %{customdata[3]}<br>"
                "upper: %{customdata[4]:.1f}  "
                "lower: %{customdata[5]:.1f}<br>"
                "events: %{customdata[6]}<extra></extra>"
            ),
        ))

    return traces


def make_band_traces(score_df: pd.DataFrame, config: ChartConfig) -> list[go.Scatter]:
    """
    Build upper / lower band line traces and a filled area between them.

    Returns:
        List of go.Scatter traces: [upper_line, lower_line, fill_area]
    """
    c = config.colors
    df = score_df.dropna(subset=["upper_band", "lower_band"])
    if df.empty:
        return []

    upper = go.Scatter(
        x=df["timestamp"],
        y=df["upper_band"],
        mode="lines",
        line=dict(color=c.band_line, width=1, dash="dot"),
        name=f"+{config.band_n_sigma}σ",
        showlegend=True,
        hoverinfo="skip",
    )

    lower = go.Scatter(
        x=df["timestamp"],
        y=df["lower_band"],
        mode="lines",
        line=dict(color=c.band_line, width=1, dash="dot"),
        fill="tonexty",
        fillcolor=c.band_fill,
        name=f"−{config.band_n_sigma}σ",
        showlegend=True,
        hoverinfo="skip",
    )

    return [upper, lower]
