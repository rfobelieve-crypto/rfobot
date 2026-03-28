"""
Batch insert normalized trades into MySQL.
"""

import logging
from market_data.storage.db import get_conn

logger = logging.getLogger(__name__)

INSERT_SQL = """
INSERT INTO normalized_trades (
    exchange, raw_symbol, canonical_symbol, instrument_type,
    price, size, size_unit, taker_side, notional_usd,
    trade_id, ts_exchange, ts_received, is_aggregated_trade
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s
)
"""


def insert_trades(trades: list[dict]):
    """Batch insert a list of normalized trade dicts."""
    if not trades:
        return

    rows = []
    for t in trades:
        rows.append((
            t["exchange"],
            t["raw_symbol"],
            t["canonical_symbol"],
            t["instrument_type"],
            t["price"],
            t["size"],
            t["size_unit"],
            t["taker_side"],
            t["notional_usd"],
            t.get("trade_id"),
            t["ts_exchange"],
            t["ts_received"],
            t.get("is_aggregated_trade", False),
        ))

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(INSERT_SQL, rows)
        logger.debug("Inserted %d trades", len(rows))
    except Exception:
        logger.exception("Failed to insert %d trades", len(rows))
    finally:
        conn.close()
