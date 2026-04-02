"""
Feature assembler: queries raw DB tables for a time window,
calls feature modules, returns features_dict.

IMPORTANT: No scoring logic here. No strategy. Query + compute only.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import pandas as pd

from research.features.delta import compute_delta
from research.features.volume import compute_volume
from research.features.oi import compute_oi
from research.features.cvd import compute_cvd
from research.bar_generator.aligner import count_events
from research.config.settings import TF_MS

logger = logging.getLogger(__name__)

_LOOKBACK_BARS = 20   # historical bars used for CVD slope, rel_volume


def assemble_features(
    symbol: str,
    timeframe: str,
    window_start_ms: int,
    window_end_ms: int,
) -> dict:
    """
    Build the complete features_dict for one time window.

    Queries flow_bars_1m, oi_snapshots, funding_rates, liquidation_1m,
    event_registry — then delegates to feature modules for computation.

    Args:
        symbol:          canonical symbol (e.g. "BTC-USD")
        timeframe:       e.g. "1h"
        window_start_ms: Unix ms (inclusive)
        window_end_ms:   Unix ms (exclusive)

    Returns:
        dict with all feature keys (None where data is unavailable).
        Ready to be passed directly to score_engine.
    """
    # ── Raw data ────────────────────────────────────────────────────────────
    flow_df      = _query_flow_bars(symbol, window_start_ms, window_end_ms)
    lookback_df  = _query_lookback_flow(symbol, window_start_ms, timeframe, _LOOKBACK_BARS)
    oi_base, oi_snap = _query_oi(symbol, window_start_ms, window_end_ms)
    funding      = _query_funding(symbol, window_start_ms)
    liq_total    = _query_liq(symbol, window_start_ms, window_end_ms)
    ev_count     = count_events(symbol, window_start_ms, window_end_ms)
    ohlc         = _query_ohlc(symbol, window_start_ms, window_end_ms)

    # ── Features ────────────────────────────────────────────────────────────
    delta_f  = compute_delta(flow_df)
    volume_f = compute_volume(flow_df, lookback_df)
    oi_f     = compute_oi(oi_base, oi_snap)
    cvd_f    = compute_cvd(lookback_df, flow_df)

    return {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "window_start":  window_start_ms,
        "window_end":    window_end_ms,
        # delta
        **delta_f,
        # volume
        **volume_f,
        # OI
        "oi_change_pct": oi_f.get("oi_change_pct"),
        "oi_direction":  oi_f.get("oi_direction"),
        # CVD
        **cvd_f,
        # macro
        "funding_rate":  funding,
        "liq_total_usd": liq_total,
        # event overlay
        "event_count":   ev_count,
        # OHLC
        **ohlc,
    }


# ── Private DB helpers ───────────────────────────────────────────────────────

def _query_flow_bars(
    symbol: str, start_ms: int, end_ms: int
) -> Optional[pd.DataFrame]:
    """1m flow bars inside the window."""
    from shared.db import get_db_conn
    sql = """
    SELECT delta_usd, volume_usd, buy_notional_usd, sell_notional_usd,
           window_start, cvd_usd
    FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND exchange_scope   = 'all'
      AND window_start >= %s AND window_start < %s
    ORDER BY window_start ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, start_ms, end_ms))
            rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        logger.exception("_query_flow_bars failed")
        return None
    finally:
        conn.close()


def _query_lookback_flow(
    symbol: str, before_ms: int, timeframe: str, n_bars: int
) -> Optional[pd.DataFrame]:
    """
    Fetch the N completed tf-bars immediately before this window,
    aggregated from flow_bars_1m.
    """
    from shared.db import get_db_conn
    tf_ms = TF_MS.get(timeframe, 3_600_000)
    lookback_start = before_ms - n_bars * tf_ms

    sql = """
    SELECT
        FLOOR(window_start / %s) * %s AS tf_window,
        SUM(delta_usd)           AS delta_usd,
        SUM(volume_usd)          AS volume_usd,
        SUM(buy_notional_usd)    AS buy_notional_usd,
        SUM(sell_notional_usd)   AS sell_notional_usd
    FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND exchange_scope   = 'all'
      AND window_start >= %s AND window_start < %s
    GROUP BY tf_window
    ORDER BY tf_window ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tf_ms, tf_ms, symbol, lookback_start, before_ms))
            rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None
    finally:
        conn.close()


def _query_oi(
    symbol: str, start_ms: int, end_ms: int
) -> Tuple[Optional[float], Optional[float]]:
    """Combined OI (OKX + Binance) at window start and end."""
    from shared.db import get_db_conn
    margin_ms = 5 * 60 * 1000  # ±5 min tolerance

    def _sum_near(target_ms: int) -> Optional[float]:
        sql = """
        SELECT oi_notional_usd FROM oi_snapshots
        WHERE canonical_symbol = %s
          AND ts_exchange BETWEEN %s AND %s
        ORDER BY ABS(ts_exchange - %s) ASC
        LIMIT 4
        """
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    symbol,
                    target_ms - margin_ms, target_ms + margin_ms,
                    target_ms,
                ))
                rows = cur.fetchall()
                total = sum(float(r["oi_notional_usd"]) for r in rows)
                return total if total > 0 else None
        except Exception:
            return None

    conn = get_db_conn()
    try:
        return _sum_near(start_ms), _sum_near(end_ms)
    finally:
        conn.close()


def _query_funding(symbol: str, ts_ms: int) -> Optional[float]:
    """Most recent funding rate at or before ts_ms."""
    from shared.db import get_db_conn
    sql = """
    SELECT funding_rate FROM funding_rates
    WHERE canonical_symbol = %s AND ts_exchange <= %s
    ORDER BY ts_exchange DESC LIMIT 1
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, ts_ms))
            row = cur.fetchone()
            return float(row["funding_rate"]) if row else None
    except Exception:
        return None
    finally:
        conn.close()


def _query_ohlc(symbol: str, start_ms: int, end_ms: int) -> dict:
    """
    OHLC prices from normalized_trades for the bar window.
    Returns empty dict if no data (bar will have NULL prices).
    Uses separate open/close queries to avoid duplicate-row issues
    when multiple trades share the same ts_exchange.
    """
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(price) AS bar_low, MAX(price) AS bar_high, "
                "MIN(ts_exchange) AS first_ts, MAX(ts_exchange) AS last_ts "
                "FROM normalized_trades "
                "WHERE canonical_symbol=%s AND ts_exchange>=%s AND ts_exchange<%s",
                (symbol, start_ms, end_ms),
            )
            mm = cur.fetchone()
        if not mm or mm["bar_low"] is None:
            return {}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT price FROM normalized_trades "
                "WHERE canonical_symbol=%s AND ts_exchange=%s LIMIT 1",
                (symbol, mm["first_ts"]),
            )
            open_row = cur.fetchone()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT price FROM normalized_trades "
                "WHERE canonical_symbol=%s AND ts_exchange=%s LIMIT 1",
                (symbol, mm["last_ts"]),
            )
            close_row = cur.fetchone()

        return {
            "bar_open":  float(open_row["price"]) if open_row else None,
            "bar_high":  float(mm["bar_high"]),
            "bar_low":   float(mm["bar_low"]),
            "bar_close": float(close_row["price"]) if close_row else None,
        }
    except Exception:
        logger.exception("_query_ohlc failed @ %d", start_ms)
        return {}
    finally:
        conn.close()


def _query_liq(symbol: str, start_ms: int, end_ms: int) -> Optional[float]:
    """Total liquidation USD in window; None if zero."""
    from shared.db import get_db_conn
    sql = """
    SELECT COALESCE(SUM(liq_total_usd), 0) AS total
    FROM liquidation_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s AND window_start < %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, start_ms, end_ms))
            row = cur.fetchone()
            val = float(row["total"]) if row else 0.0
            return val if val > 0 else None
    except Exception:
        return None
    finally:
        conn.close()
