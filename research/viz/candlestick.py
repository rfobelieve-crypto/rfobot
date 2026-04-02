"""
Candlestick trace + signal marker overlay.
Pure rendering — no calculations, no DB access.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from research.config.settings import ChartConfig


def make_candle_trace(ohlc_df: pd.DataFrame, config: ChartConfig) -> go.Candlestick:
    """
    Build a Candlestick trace from OHLC data.

    Args:
        ohlc_df: DataFrame with columns [timestamp, open, high, low, close]
        config:  ChartConfig for color settings

    Returns:
        go.Candlestick trace
    """
    c = config.colors
    return go.Candlestick(
        x=ohlc_df["timestamp"],
        open=ohlc_df["open"],
        high=ohlc_df["high"],
        low=ohlc_df["low"],
        close=ohlc_df["close"],
        increasing_line_color=c.candle_up,
        decreasing_line_color=c.candle_down,
        increasing_fillcolor=c.candle_up,
        decreasing_fillcolor=c.candle_down,
        name=config.symbol,
        showlegend=True,
    )


def make_signal_trace(
    ohlc_df: pd.DataFrame,
    score_df: pd.DataFrame,
    config: ChartConfig,
) -> go.Scatter | None:
    """
    Build a scatter trace marking bars where signal == 1.

    Signal dots are placed at the bar's close price.
    ohlc_df and score_df are aligned on timestamp before filtering.

    Returns None if no signals exist.
    """
    if "signal" not in score_df.columns:
        return None

    # Align on timestamp
    sig = score_df[score_df["signal"] == 1][["timestamp", "final_bias"]].copy()
    if sig.empty:
        return None

    merged = pd.merge_asof(
        sig.sort_values("timestamp"),
        ohlc_df[["timestamp", "close"]].sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    ).dropna(subset=["close"])

    if merged.empty:
        return None

    return go.Scatter(
        x=merged["timestamp"],
        y=merged["close"],
        mode="markers",
        marker=dict(
            symbol="circle",
            size=14,
            color=config.colors.signal,
            line=dict(color="#000", width=1),
        ),
        name="Signal ●",
        showlegend=True,
        hovertemplate=(
            "<b>Signal</b><br>"
            "%{x}<br>"
            "Close: %{y:,.2f}<br>"
            "Bias: %{customdata}<extra></extra>"
        ),
        customdata=merged["final_bias"],
    )
