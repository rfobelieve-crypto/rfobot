"""
Interactive indicator chart — Plotly-based HTML with zoom, pan, crosshair.

Layout matches Telegram static chart (chart_renderer.py):
  1. Confidence heatmap (top, thin)
  2. Candlestick + direction triangles (main, large)
  3. Magnitude bars (signed by direction)
  4. Bull/Bear Power bars (bottom)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Colors (match static chart exactly) ──
GREEN_STRONG = "#004d40"
GREEN_MOD    = "#26a69a"
RED_STRONG   = "#b71c1c"
RED_MOD      = "#ef5350"
GRAY         = "#9e9e9e"
BG_COLOR     = "#0d1117"
CARD_COLOR   = "#161b22"
GRID_COLOR   = "#1c222b"
TEXT_COLOR   = "#b0b8c4"
CONF_LOW     = "#1a1a2e"
CONF_MID     = "#6a1b9a"
CONF_HIGH    = "#e040fb"


def render_interactive_chart(ind: pd.DataFrame, last_n: int = 200) -> str:
    """
    Render interactive indicator chart. Returns full HTML string.

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
    try:
        if sig.index.tz is None:
            sig.index = sig.index.tz_localize("UTC")
        sig.index = sig.index.tz_convert(TZ_UTC8)
    except Exception:
        pass
    dates = sig.index

    has_mag = "mag_pred" in sig.columns and sig["mag_pred"].notna().any()

    # ── Subplot layout (match static: 0.8 / 8 / 2 / 2 = total 12.8) ──
    if has_mag:
        row_count = 4
        row_heights = [0.06, 0.62, 0.16, 0.16]
    else:
        row_count = 3
        row_heights = [0.07, 0.70, 0.23]

    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=row_heights,
    )

    # ════════════════════════════════════════════════════════════════
    # Panel 1: Confidence heatmap (thin purple bar)
    # ════════════════════════════════════════════════════════════════
    conf = sig["confidence_score"].fillna(0).values.astype(float)
    fig.add_trace(go.Bar(
        x=dates, y=[1] * len(dates),
        marker=dict(
            color=conf,
            colorscale=[[0, CONF_LOW], [0.5, CONF_MID], [1, CONF_HIGH]],
            cmin=0, cmax=100,
            showscale=False,
        ),
        hovertemplate="Confidence: %{marker.color:.0f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ════════════════════════════════════════════════════════════════
    # Panel 2: Candlestick + direction triangles
    # ════════════════════════════════════════════════════════════════
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

    # Direction triangles — Moderate and Strong only (same filter as static)
    price_range = sig["high"].max() - sig["low"].min()
    offset = price_range * 0.015

    for strength, direction, marker_sym, color, name in [
        ("Strong", "UP", "triangle-up", GREEN_STRONG, "Strong UP"),
        ("Moderate", "UP", "triangle-up", GREEN_MOD, "Moderate UP"),
        ("Moderate", "DOWN", "triangle-down", RED_MOD, "Moderate DOWN"),
        ("Strong", "DOWN", "triangle-down", RED_STRONG, "Strong DOWN"),
    ]:
        mask = (
            (sig["strength_score"] == strength) &
            (sig["pred_direction"] == direction) &
            sig["confidence_score"].notna()
        )
        if not mask.any():
            continue
        subset = sig[mask]
        if direction == "UP":
            y_vals = subset["low"] - offset
        else:
            y_vals = subset["high"] + offset

        fig.add_trace(go.Scatter(
            x=subset.index, y=y_vals,
            mode="markers",
            marker=dict(
                symbol=marker_sym,
                size=14 if strength == "Strong" else 10,
                color=color,
                line=dict(width=1.5, color="white") if strength == "Strong" else dict(width=0),
            ),
            name=name,
            hovertemplate=(
                f"{name}<br>"
                "Price: $%{y:,.0f}<br>"
                "Conf: %{customdata[0]:.0f}%<br>"
                "Mag: %{customdata[1]:.2%}<extra></extra>"
            ),
            customdata=np.column_stack([
                subset["confidence_score"].fillna(0).values,
                subset.get("mag_pred", pd.Series(0, index=subset.index)).fillna(0).values,
            ]),
        ), row=2, col=1)

    # ════════════════════════════════════════════════════════════════
    # Panel 3: Magnitude bars (signed by direction, same as static)
    # ════════════════════════════════════════════════════════════════
    if has_mag:
        mag_raw = sig["mag_pred"].fillna(0).values.astype(float) * 100
        mag_dirs = sig["pred_direction"].fillna("NEUTRAL").values

        mag_signed = np.zeros(len(mag_raw))
        mag_colors = []
        for i in range(len(mag_raw)):
            d, m = mag_dirs[i], mag_raw[i]
            if d == "UP":
                mag_signed[i] = m
                mag_colors.append(GREEN_STRONG if m > 0.5 else GREEN_MOD)
            elif d == "DOWN":
                mag_signed[i] = -m
                mag_colors.append(RED_STRONG if m > 0.5 else RED_MOD)
            else:
                mag_signed[i] = m  # NEUTRAL: grey above zero
                mag_colors.append(GRAY)

        fig.add_trace(go.Bar(
            x=dates, y=mag_signed,
            marker_color=mag_colors,
            name="Magnitude",
            showlegend=False,
            hovertemplate="Mag: %{y:.3f}%<extra></extra>",
        ), row=3, col=1)

    # ════════════════════════════════════════════════════════════════
    # Panel 4 (or 3): Bull/Bear Power bars
    # ════════════════════════════════════════════════════════════════
    bbp_row = 4 if has_mag else 3
    bbp = sig["bull_bear_power"].fillna(0).values.astype(float)
    bbp_colors = [GREEN_MOD if v > 0 else RED_MOD for v in bbp]

    fig.add_trace(go.Bar(
        x=dates, y=bbp,
        marker_color=bbp_colors,
        name="BBP",
        showlegend=False,
        hovertemplate="BBP: %{y:+.3f}<extra></extra>",
    ), row=bbp_row, col=1)

    # ════════════════════════════════════════════════════════════════
    # Layout — match Telegram style
    # ════════════════════════════════════════════════════════════════
    last_time = dates[-1].strftime("%Y-%m-%d %H:%M UTC+8")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR, size=11, family="Calibri, sans-serif"),
        title=dict(
            text=f"BTC Market Intelligence Indicator (4h prediction)  |  Updated: {last_time}",
            font=dict(size=14, color="white"),
            x=0.5,
        ),
        height=900,
        margin=dict(l=60, r=30, t=45, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    # ── Axis styling ──
    # Panel 1: confidence — hide y ticks
    fig.update_yaxes(showticklabels=False, fixedrange=True, row=1, col=1)

    # Panel 2: price
    fig.update_yaxes(title_text="Price (USD)", title_font_size=10, row=2, col=1)

    # Panel 3: magnitude
    if has_mag:
        fig.update_yaxes(title_text="Magnitude (%)", title_font_size=10, row=3, col=1)
        fig.add_hline(y=0, line_width=0.5, line_color="white", row=3, col=1)

    # Panel 4: BBP
    fig.update_yaxes(title_text="Bull/Bear Power", title_font_size=10, row=bbp_row, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color="white", row=bbp_row, col=1)

    # Grid color for all panels
    for r in range(1, row_count + 1):
        fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, row=r, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, row=r, col=1)

    # ── Crosshair cursor (all panels) ──
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=0.5, spikedash="dot",
    )
    fig.update_yaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=0.5, spikedash="dot",
    )

    # ── Range selector (top panel) ──
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

    # ── Watermark ──
    fig.add_annotation(
        text="source@rfo",
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=9, color=GRAY),
        opacity=0.5,
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
