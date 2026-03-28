"""
Query flow_bars_1m to provide market context around liquidity sweep events.

Used by the main bot to enrich TradingView webhook notifications.
"""

import time
import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def get_pre_sweep_context(canonical_symbol: str, lookback_minutes: int = 5) -> dict | None:
    """
    Get flow context for the N minutes before now (pre-sweep snapshot).

    Returns aggregated stats over the lookback window, or None if no data.
    """
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (lookback_minutes * 60_000)

    return _query_flow_range(canonical_symbol, start_ms, now_ms, label=f"pre-sweep {lookback_minutes}m")


def get_event_flow_context(canonical_symbol: str, event_ts_unix: int, duration_minutes: int) -> dict | None:
    """
    Get flow context for a specific time range after an event.

    event_ts_unix: event trigger time in unix seconds
    duration_minutes: how many minutes after event to look at
    """
    start_ms = event_ts_unix * 1000
    end_ms = start_ms + (duration_minutes * 60_000)

    return _query_flow_range(canonical_symbol, start_ms, end_ms, label=f"post-event {duration_minutes}m")


def _query_flow_range(canonical_symbol: str, start_ms: int, end_ms: int, label: str = "") -> dict | None:
    """Query flow_bars_1m for a time range and aggregate."""
    sql = """
    SELECT
        COUNT(*) AS bar_count,
        COALESCE(SUM(buy_notional_usd), 0) AS total_buy,
        COALESCE(SUM(sell_notional_usd), 0) AS total_sell,
        COALESCE(SUM(delta_usd), 0) AS total_delta,
        COALESCE(SUM(volume_usd), 0) AS total_volume,
        COALESCE(SUM(trade_count), 0) AS total_trades
    FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s
      AND window_start < %s
      AND exchange_scope = 'all'
    """

    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical_symbol, start_ms, end_ms))
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query flow context (%s)", label)
        return None

    if not row or row["bar_count"] == 0:
        return None

    total_buy = float(row["total_buy"])
    total_sell = float(row["total_sell"])
    total_delta = float(row["total_delta"])
    total_volume = float(row["total_volume"])

    # Get latest CVD from the most recent bar in range
    cvd = _get_latest_cvd(canonical_symbol, start_ms, end_ms)

    return {
        "canonical_symbol": canonical_symbol,
        "bar_count": row["bar_count"],
        "buy_usd": total_buy,
        "sell_usd": total_sell,
        "delta_usd": total_delta,
        "volume_usd": total_volume,
        "trade_count": int(row["total_trades"]),
        "cvd_usd": cvd,
        "buy_sell_ratio": round(total_buy / total_sell, 2) if total_sell > 0 else None,
    }


def _get_latest_cvd(canonical_symbol: str, start_ms: int, end_ms: int) -> float:
    """Get CVD from the latest bar in the range."""
    sql = """
    SELECT cvd_usd FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s
      AND window_start < %s
      AND exchange_scope = 'all'
    ORDER BY window_start DESC
    LIMIT 1
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical_symbol, start_ms, end_ms))
                row = cur.fetchone()
                return float(row["cvd_usd"]) if row else 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0


def format_flow_context(ctx: dict, title: str = "Market Flow") -> str:
    """Format a flow context dict into readable text for Telegram."""
    if ctx is None:
        return f"[{title}] 無資料（market data 可能尚未累積）"

    delta = ctx["delta_usd"]
    delta_emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "🟡"

    volume = ctx["volume_usd"]
    buy = ctx["buy_usd"]
    sell = ctx["sell_usd"]
    ratio = ctx["buy_sell_ratio"]
    ratio_text = f"{ratio:.2f}" if ratio is not None else "N/A"

    def fmt(x):
        ax = abs(x)
        if ax >= 1e9:
            return f"{x / 1e9:.2f}B"
        if ax >= 1e6:
            return f"{x / 1e6:.2f}M"
        if ax >= 1e3:
            return f"{x / 1e3:.2f}K"
        return f"{x:,.0f}"

    lines = [
        f"[{title}] (OKX+Binance, {ctx['bar_count']}bars)",
        f"  delta: {fmt(delta)} {delta_emoji}",
        f"  volume: {fmt(volume)}",
        f"  buy/sell: {fmt(buy)} / {fmt(sell)} (ratio {ratio_text})",
        f"  trades: {ctx['trade_count']}",
        f"  cvd: {fmt(ctx['cvd_usd'])}",
    ]
    return "\n".join(lines)
