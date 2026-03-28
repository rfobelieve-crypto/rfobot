"""
Event aligner: count / describe liquidity events in a time window.

Events are a FEATURE — they do not control whether a bar is produced.
A bar with event_count=0 is valid and must still be computed.
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)


def count_events(symbol: str, start_ms: int, end_ms: int) -> int:
    """
    Count liquidity events (event_registry) whose trigger falls inside [start, end).

    Args:
        symbol:   canonical symbol (e.g. "BTC-USD")
        start_ms: window start Unix ms (inclusive)
        end_ms:   window end Unix ms (exclusive)

    Returns:
        Number of events; 0 if none or on DB error.
    """
    from shared.db import get_db_conn

    start_s = start_ms // 1000
    end_s   = end_ms   // 1000
    sql = """
    SELECT COUNT(*) AS cnt FROM event_registry
    WHERE symbol = %s AND trigger_ts >= %s AND trigger_ts < %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, start_s, end_s))
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0
    except Exception:
        logger.debug("count_events failed: %s", symbol)
        return 0
    finally:
        conn.close()


def get_event_types(symbol: str, start_ms: int, end_ms: int) -> List[str]:
    """
    Return distinct event types (e.g. ["BSL", "SSL"]) in the window.
    Used for richer hover tooltips.
    """
    from shared.db import get_db_conn

    start_s = start_ms // 1000
    end_s   = end_ms   // 1000
    sql = """
    SELECT DISTINCT event_type FROM event_registry
    WHERE symbol = %s AND trigger_ts >= %s AND trigger_ts < %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, start_s, end_s))
            return [r["event_type"] for r in cur.fetchall() if r.get("event_type")]
    except Exception:
        return []
    finally:
        conn.close()
