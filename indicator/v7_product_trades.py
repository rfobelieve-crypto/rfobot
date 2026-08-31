# -*- coding: utf-8 -*-
"""V7 product-side (jarvis / Bitget) executions for the chart overlay.

The live-execution overlay on both V7 charts historically read
`v7_okx_positions`. That table froze at 2026-08-11 when execution migrated
to the jarvis product side (CLAUDE.md 2026-08-21) — real entries/exits now
happen on Bitget and land in jarvis's per-user `v7_trades.jsonl`.
`research/v7_product_trades_publish.py` pulls that ledger hourly (same
export channel + token as the mill pipeline, TODO §0.78) and upserts it
into `v7_product_trades`; this module is the read side.

Returns the SAME dict shape as
`indicator.okx.state.fetch_okx_positions_for_chart` (entry_time /
direction / entry_price / entry_tier / status / exit_time / exit_price /
exit_reason / win, naive-UTC datetimes) so both charts can simply
concatenate the two lists. Never raises; missing table or empty ledger is
an empty list (the overlay is decorative — it must not break the chart).
"""
from __future__ import annotations

import logging
from datetime import datetime

from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def fetch_v7_product_trades_for_chart(start_dt: datetime,
                                      end_dt: datetime) -> list:
    """Product-side V7 trades whose holding window intersects the chart."""
    try:
        conn = get_db_conn()
    except Exception as exc:                                # pragma: no cover
        logger.warning("v7_product_trades: no DB conn (%s)", exc)
        return []
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT entry_time, direction, entry_price, "
                    "       'Strong' AS entry_tier, status, exit_time, "
                    "       exit_price, exit_reason, win "
                    "FROM `v7_product_trades` "
                    "WHERE source <> 'selftest' "
                    "  AND entry_time <= %s "
                    "  AND (exit_time IS NULL OR exit_time >= %s) "
                    "ORDER BY entry_time ASC",
                    (end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                     start_dt.strftime("%Y-%m-%d %H:%M:%S")))
                return list(cur.fetchall())
            except Exception as exc:
                # table not created yet (token not set) → overlay simply
                # shows the OKX era only; log at debug-ish level.
                logger.warning("v7_product_trades fetch skipped: %s", exc)
                return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
