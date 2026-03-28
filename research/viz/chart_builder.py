"""
Chart builder — the ONLY public entry point for visualization.

Responsibilities:
- Load data from storage (market_state_repository + OHLC)
- Assemble subplot layout
- Delegate trace creation to candlestick / score_bar / oi_bar / cvd_line
- Apply dark theme + layout

NO calculations. NO DB logic beyond data loading.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from research.config.settings import ChartConfig, ColorConfig
from research.storage.market_state_repository import query_bars
from research.viz.candlestick import make_candle_trace, make_signal_trace
from research.viz.score_bar import make_score_trace, make_band_traces

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────

def build_chart(
    ohlc_df: pd.DataFrame,
    score_df: pd.DataFrame,
    config: ChartConfig | None = None,
) -> go.Figure:
    """
    Assemble the full dual-panel (+ optional) research chart.

    Args:
        ohlc_df:  DataFrame with [timestamp, open, high, low, close]
        score_df: DataFrame from market_state_repository.query_bars()
        config:   ChartConfig (uses defaults if None)

    Returns:
        plotly Figure ready for .show() / .write_html()
    """
    if config is None:
        config = ChartConfig()

    # ── Subplot structure ────────────────────────────────────────────────────
    extra_rows  = (1 if config.show_oi else 0) + (1 if config.show_cvd else 0)
    n_rows      = 2 + extra_rows
    row_heights, subtitles = _subplot_spec(config, n_rows)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subtitles,
    )

    # ── Row 1: OHLC + signal markers ────────────────────────────────────────
    if not ohlc_df.empty:
        fig.add_trace(make_candle_trace(ohlc_df, config), row=1, col=1)
        sig = make_signal_trace(ohlc_df, score_df, config)
        if sig:
            fig.add_trace(sig, row=1, col=1)

    # ── Row 2: Score bars + dynamic bands ───────────────────────────────────
    if not score_df.empty:
        for t in make_score_trace(score_df, config):
            fig.add_trace(t, row=2, col=1)
        for t in make_band_traces(score_df, config):
            fig.add_trace(t, row=2, col=1)

    # ── Optional rows ────────────────────────────────────────────────────────
    cur_row = 3
    if config.show_oi and not score_df.empty:
        from research.viz.oi_bar import make_oi_trace
        for t in make_oi_trace(score_df, config):
            fig.add_trace(t, row=cur_row, col=1)
        cur_row += 1

    if config.show_cvd and not score_df.empty:
        from research.viz.cvd_line import make_cvd_trace
        for t in make_cvd_trace(score_df, config):
            fig.add_trace(t, row=cur_row, col=1)

    _apply_layout(fig, config, n_rows)
    return fig


def load_and_build(
    symbol: str = "BTC-USD",
    timeframe: str = "1h",
    lookback_days: int = 7,
    config: ChartConfig | None = None,
    ohlc_source: str = "trades",   # "trades" | callable
) -> go.Figure:
    """
    Convenience wrapper: load data then call build_chart().

    Args:
        symbol:       canonical symbol
        timeframe:    e.g. "1h"
        lookback_days: history window
        config:       ChartConfig (symbol/timeframe overridden by args)
        ohlc_source:  "trades" = normalize from normalized_trades table
                      Pass a callable(symbol, start_ms, end_ms) → DataFrame
                      for custom loaders (e.g. Binance REST, CSV).
    """
    if config is None:
        config = ChartConfig(symbol=symbol, timeframe=timeframe)

    now_ms    = int(time.time() * 1000)
    start_ms  = now_ms - lookback_days * 86_400_000

    score_df = query_bars(symbol, timeframe, start_ms, now_ms)

    if callable(ohlc_source):
        ohlc_df = ohlc_source(symbol, start_ms, now_ms)
    else:
        ohlc_df = _load_ohlc_from_trades(symbol, timeframe, start_ms, now_ms)

    return build_chart(ohlc_df, score_df, config)


# ── Private helpers ──────────────────────────────────────────────────────────

def _subplot_spec(config: ChartConfig, n_rows: int):
    """Return (row_heights, subplot_titles) based on active panels."""
    base_heights = {2: [0.70, 0.30], 3: [0.55, 0.25, 0.20], 4: [0.50, 0.22, 0.15, 0.13]}
    heights = base_heights.get(n_rows, [0.5] + [0.5 / (n_rows - 1)] * (n_rows - 1))

    titles = [
        f"{config.symbol} ({config.timeframe})",
        f"Risk-Adjusted Reversal Score  [{config.score_field}]  "
        f"± {config.band_n_sigma}σ bands",
    ]
    if config.show_oi:
        titles.append("OI Change %")
    if config.show_cvd:
        titles.append("CVD Change")

    return heights, titles


def _apply_layout(fig: go.Figure, config: ChartConfig, n_rows: int):
    c = config.colors
    fig.update_layout(
        title=dict(
            text=(
                f"{config.symbol} — Risk-Adjusted Reversal  "
                f"(Dynamic Bands: {config.band_n_sigma}σ)"
            ),
            font=dict(size=16, color=c.text_color),
        ),
        template="plotly_dark",
        paper_bgcolor=c.bg_color,
        plot_bgcolor=c.bg_color,
        font=dict(color=c.text_color, size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left",   x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=820 if n_rows == 2 else 920,
        margin=dict(l=60, r=40, t=80, b=60),
        annotations=[dict(
            text=config.annotation,
            xref="paper", yref="paper",
            x=0.0, y=-0.06,
            showarrow=False,
            font=dict(size=10, color="#555"),
        )],
    )
    fig.update_xaxes(
        tickformat="%m/%d %H:%M",
        tickangle=45,
        gridcolor=c.grid_color,
        linecolor=c.grid_color,
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor=c.grid_color,
        linecolor=c.grid_color,
        showgrid=True,
    )


def _load_ohlc_from_trades(
    symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """
    Aggregate OHLC from normalized_trades (3-day retention).
    Returns empty DataFrame if no data.
    """
    from research.config.settings import TF_MS
    from shared.db import get_db_conn

    tf_ms = TF_MS.get(timeframe, 3_600_000)

    # MySQL doesn't have FIRST/LAST — use subquery with MIN/MAX ts_exchange
    sql = """
    SELECT
        t.bar_start_ms            AS bar_start_ms,
        t.low, t.high,
        f.price                   AS open,
        l.price                   AS close
    FROM (
        SELECT
            FLOOR(ts_exchange / %s) * %s AS bar_start_ms,
            MIN(price)                    AS low,
            MAX(price)                    AS high,
            MIN(ts_exchange)              AS first_ts,
            MAX(ts_exchange)              AS last_ts
        FROM normalized_trades
        WHERE canonical_symbol = %s
          AND ts_exchange >= %s AND ts_exchange < %s
        GROUP BY bar_start_ms
    ) t
    JOIN normalized_trades f ON f.canonical_symbol = %s AND f.ts_exchange = t.first_ts
    JOIN normalized_trades l ON l.canonical_symbol = %s AND l.ts_exchange = t.last_ts
    ORDER BY t.bar_start_ms ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tf_ms, tf_ms, symbol, start_ms, end_ms, symbol, symbol))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = (
            pd.to_datetime(df["bar_start_ms"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Taipei")
        )
        return df
    except Exception:
        logger.exception("_load_ohlc_from_trades failed")
        return pd.DataFrame()
    finally:
        conn.close()
