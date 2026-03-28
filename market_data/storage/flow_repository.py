"""
Upsert flow_bars_1m into MySQL.
"""

import logging
from market_data.storage.db import get_conn

logger = logging.getLogger(__name__)

UPSERT_SQL = """
INSERT INTO flow_bars_1m (
    canonical_symbol, instrument_type, exchange_scope,
    window_start, window_end,
    buy_notional_usd, sell_notional_usd, delta_usd, volume_usd,
    trade_count, cvd_usd, source_count, quality_score
) VALUES (
    %s, %s, %s,
    %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    buy_notional_usd = VALUES(buy_notional_usd),
    sell_notional_usd = VALUES(sell_notional_usd),
    delta_usd = VALUES(delta_usd),
    volume_usd = VALUES(volume_usd),
    trade_count = VALUES(trade_count),
    cvd_usd = VALUES(cvd_usd),
    source_count = VALUES(source_count),
    quality_score = VALUES(quality_score)
"""


def upsert_flow_bars(bars: list[dict]):
    """Upsert a list of flow bar dicts (keyed by unique constraint)."""
    if not bars:
        return

    rows = []
    for b in bars:
        rows.append((
            b["canonical_symbol"],
            b["instrument_type"],
            b["exchange_scope"],
            b["window_start"],
            b["window_end"],
            b["buy_notional_usd"],
            b["sell_notional_usd"],
            b["delta_usd"],
            b["volume_usd"],
            b["trade_count"],
            b["cvd_usd"],
            b["source_count"],
            b["quality_score"],
        ))

    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(UPSERT_SQL, rows)
        logger.debug("Upserted %d flow bars", len(rows))
    except Exception:
        logger.exception("Failed to upsert %d flow bars", len(rows))
    finally:
        conn.close()
