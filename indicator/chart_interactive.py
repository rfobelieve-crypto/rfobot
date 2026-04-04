"""
Interactive indicator chart — Plotly-based HTML with zoom, pan, crosshair.

Serves the same 4-panel layout as chart_renderer.py but interactive:
  1. Confidence heatmap (top)
  2. Candlestick + direction markers
  3. Magnitude bars (signed by direction)
  4. Bull/Bear Power bars
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Colors (match static chart) ──
GREEN_STRONG = "#004d40"
GREEN_MOD    = "#26a69a"
RED_STRONG   = "#b71c1c"
RED_MOD      = "#ef5350"
GRAY         = "#9e9e9e"
BG_COLOR     = "#0d1117"
CARD_COLOR   = "#161b22"
GRID_COLOR   = "#1c222b"
TEXT_COLOR   = "#b0b8c4"


def render_interactive_chart(ind: pd.DataFrame, last_n: int = 200) -> str:
    """
    Render interactive indicator chart. Returns HTML string.

    ind must have: open, high, low, close, pred_direction, confidence_score,
                   strength_score, pred_return_4h, bull_bear_power, regime
    Optional: mag_pred
    """
    sig = ind.tail(last_n).copy()
    sig = sig.dropna(subset=["open", "high", "low", "close"])

    if len(sig) == 0:
        return "<h3>No data</h3>"

    # UTC+8
    from datetime import timezone, timedelta
    TZ_UTC8 = timezone(timedelta(hours=8))
    sig.index = sig.index.tz_convert(TZ_UTC8)
    dates = sig.index

    has_mag = "mag_pred" in sig.columns and sig["mag_pred"].notna().any()

    # ── Subplot layout ──
    row_count = 4 if has_mag else 3
    row_heights = [0.06, 0.58, 0.18, 0.18] if has_mag else [0.06, 0.7, 0.24]
    subplot_titles = (
        "Confidence", "BTC Price", "Magnitude (%)", "Bull/Bear Power"
    ) if has_mag else (
        "Confidence", "BTC Price", "Bull/Bear Power"
    )

    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Panel 1: Confidence heatmap ──
    conf = sig["confidence_score"].fillna(0).values.astype(float)
    fig.add_trace(go.Bar(
        x=dates, y=[1] * len(dates),
        marker=dict(
            color=conf,
            colorscale=[[0, "#1a1a2e"], [0.5, "#6a1b9a"], [1, "#e040fb"]],
            cmin=0, cmax=100,
            showscale=False,
        ),
        hovertemplate="Confidence: %{marker.color:.0f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ── Panel 2: Candlestick ──
    fig.add_trace(go.Candlestick(
        x=dates,
        open=sig["open"], high=sig["high"],
        low=sig["low"], close=sig["close"],
        increasing_line_color=GREEN_MOD,
        increasing_fillcolor=GREEN_MOD,
        decreasing_line_color=RED_MOD,
        decreasing_fillcolor=RED_MOD,
        name="BTC",
        showlegend=False,
    ), row=2, col=1)

    # Direction triangles (Moderate/Strong only)
    for strength, direction, marker_sym, color, name in [
        ("Strong", "UP", "triangle-up", GREEN_STRONG, "Strong UP"),
        ("Moderate", "UP", "triangle-up", GREEN_MOD, "Moderate UP"),
        ("Moderate", "DOWN", "triangle-down", RED_MOD, "Moderate DOWN"),
        ("Strong", "DOWN", "triangle-down", RED_STRONG, "Strong DOWN"),
    ]:
        mask = (sig["strength_score"] == strength) & (sig["pred_direction"] == direction)
        if not mask.any():
            continue
        subset = sig[mask]
        y_vals = subset["low"] * 0.999 if direction == "UP" else subset["high"] * 1.001
        fig.add_trace(go.Scatter(
            x=subset.index, y=y_vals,
            mode="markers",
            marker=dict(
                symbol=marker_sym,
                size=12 if strength == "Strong" else 9,
                color=color,
                line=dict(width=1.5, color="white") if strength == "Strong" else dict(width=0),
            ),
            name=name,
            hovertemplate=(
                f"{name}<br>"
                "Conf: %{customdata[0]:.0f}<br>"
                "Mag: %{customdata[1]:.2%}<extra></extra>"
            ),
            customdata=np.column_stack([
                subset["confidence_score"].fillna(0).values,
                subset.get("mag_pred", pd.Series(0, index=subset.index)).fillna(0).values,
            ]),
        ), row=2, col=1)

    # ── Panel 3: Magnitude bars ──
    if has_mag:
        mag_raw = sig["mag_pred"].fillna(0).values.astype(float) * 100
        mag_dirs = sig["pred_direction"].fillna("NEUTRAL").values

        mag_signed = np.zeros(len(mag_raw))
        mag_colors = []
        for i in range(len(mag_raw)):
            d = mag_dirs[i]
            m = mag_raw[i]
            if d == "UP":
                mag_signed[i] = m
                mag_colors.append(GREEN_STRONG if m > 0.5 else GREEN_MOD)
            elif d == "DOWN":
                mag_signed[i] = -m
                mag_colors.append(RED_STRONG if m > 0.5 else RED_MOD)
            else:
                mag_signed[i] = m
                mag_colors.append(GRAY)

        mag_panel_row = 3
        fig.add_trace(go.Bar(
            x=dates, y=mag_signed,
            marker_color=mag_colors,
            name="Magnitude",
            showlegend=False,
            hovertemplate="Mag: %{y:.2f}%<extra></extra>",
        ), row=mag_panel_row, col=1)
        fig.add_hline(y=0, line_width=0.5, line_color="white",
                      row=mag_panel_row, col=1)

    # ── Panel 4 (or 3): BBP bars ──
    bbp_row = 4 if has_mag else 3
    bbp = sig["bull_bear_power"].fillna(0).values.astype(float)
    bbp_colors = [GREEN_MOD if v > 0 else RED_MOD for v in bbp]

    fig.add_trace(go.Bar(
        x=dates, y=bbp,
        marker_color=bbp_colors,
        name="BBP",
        showlegend=False,
        hovertemplate="BBP: %{y:.3f}<extra></extra>",
    ), row=bbp_row, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color="white",
                  row=bbp_row, col=1)

    # ── Layout ──
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR, size=11),
        title=dict(
            text="BTC Market Intelligence Indicator (4h prediction)",
            font=dict(size=16, color="white"),
            x=0.5,
        ),
        height=850,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # Style all axes
    for i in range(1, row_count + 1):
        yaxis = f"yaxis{i}" if i > 1 else "yaxis"
        xaxis = f"xaxis{i}" if i > 1 else "xaxis"
        fig.update_layout(**{
            yaxis: dict(gridcolor=GRID_COLOR, zeroline=False),
            xaxis: dict(gridcolor=GRID_COLOR),
        })

    # Hide confidence y-axis ticks
    fig.update_yaxes(showticklabels=False, row=1, col=1)

    # Crosshair cursor
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#888", spikethickness=0.5, spikedash="dot",
    )
    fig.update_yaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#888", spikethickness=0.5, spikedash="dot",
    )

    # Range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=24, label="24h", step="hour", stepmode="backward"),
                dict(count=72, label="3d", step="hour", stepmode="backward"),
                dict(count=168, label="7d", step="hour", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR, size=10),
            activecolor="#3f51b5",
        ),
        row=1, col=1,
    )

    return fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect"],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
