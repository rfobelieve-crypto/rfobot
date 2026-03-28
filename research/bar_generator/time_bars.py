"""
Time bar generation: determine which (symbol, timeframe, window) pairs
need to be computed.

Key rule: every window in the lookback period must exist in market_state_bars,
regardless of whether any liquidity event occurred during that window.
"""
from __future__ import annotations
import time
import logging
from typing import List, Tuple

from research.config.settings import TF_SECONDS

logger = logging.getLogger(__name__)


def get_pending_windows(
    symbol: str,
    timeframe: str,
    lookback_seconds: int,
) -> List[Tuple[int, int]]:
    """
    Return all (window_start_ms, window_end_ms) pairs in the lookback period
    that are not yet computed (missing from market_state_bars or score is NULL).

    Windows are aligned to timeframe boundaries (e.g. 1h bars start on the hour).
    The current (incomplete) bar is excluded.

    Args:
        symbol:           e.g. "BTC-USD"
        timeframe:        e.g. "1h"
        lookback_seconds: total seconds of history to cover

    Returns:
        List of (start_ms, end_ms) tuples, sorted oldest-first.
    """
    tf_sec = TF_SECONDS.get(timeframe)
    if not tf_sec:
        raise ValueError(f"Unknown timeframe: {timeframe!r}")

    now_sec = int(time.time())
    current_bar_start = (now_sec // tf_sec) * tf_sec  # incomplete — exclude

    # Oldest aligned boundary within the lookback
    oldest = ((now_sec - lookback_seconds) // tf_sec) * tf_sec

    all_windows: List[Tuple[int, int]] = []
    t = oldest
    while t < current_bar_start:
        all_windows.append((t * 1000, (t + tf_sec) * 1000))
        t += tf_sec

    if not all_windows:
        return []

    existing = _get_existing_starts(symbol, timeframe, [w[0] for w in all_windows])
    pending  = [w for w in all_windows if w[0] not in existing]

    logger.debug(
        "%s %s: %d total windows, %d already computed, %d pending",
        symbol, timeframe, len(all_windows), len(existing), len(pending),
    )
    return pending


def _get_existing_starts(
    symbol: str,
    timeframe: str,
    start_ms_list: List[int],
) -> set:
    """Query market_state_bars for windows already computed (score NOT NULL)."""
    if not start_ms_list:
        return set()

    from shared.db import get_db_conn

    placeholders = ",".join(["%s"] * len(start_ms_list))
    sql = f"""
    SELECT window_start FROM market_state_bars
    WHERE symbol = %s AND timeframe = %s
      AND window_start IN ({placeholders})
      AND reversal_score IS NOT NULL
    """
    params = [symbol, timeframe] + start_ms_list

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return {row["window_start"] for row in cur.fetchall()}
    except Exception:
        logger.exception("Failed to query existing windows")
        return set()
    finally:
        conn.close()
