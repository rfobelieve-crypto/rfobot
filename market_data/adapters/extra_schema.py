"""
Create funding_rates and liquidation_1m tables,
and add corresponding columns to event_feature_snapshots.
Called on startup alongside ensure_oi_schema().
"""

import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def ensure_extra_schema():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # ── funding_rates ──────────────────────────────────────────────
            cur.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                id               BIGINT AUTO_INCREMENT PRIMARY KEY,
                exchange         VARCHAR(20)    NOT NULL,
                canonical_symbol VARCHAR(20)    NOT NULL,
                funding_rate     DECIMAL(20, 8) NOT NULL,
                next_funding_ts  BIGINT         DEFAULT NULL,
                ts_exchange      BIGINT         NOT NULL,
                ts_received      BIGINT         NOT NULL,
                INDEX idx_lookup (canonical_symbol, exchange, ts_exchange)
            )
            """)
            logger.info("funding_rates table ready")

            # ── liquidation_1m ─────────────────────────────────────────────
            cur.execute("""
            CREATE TABLE IF NOT EXISTS liquidation_1m (
                id               BIGINT AUTO_INCREMENT PRIMARY KEY,
                canonical_symbol VARCHAR(20)    NOT NULL,
                window_start     BIGINT         NOT NULL,
                liq_buy_usd      DECIMAL(30, 4) NOT NULL DEFAULT 0,
                liq_sell_usd     DECIMAL(30, 4) NOT NULL DEFAULT 0,
                liq_total_usd    DECIMAL(30, 4) NOT NULL DEFAULT 0,
                liq_count        INT            NOT NULL DEFAULT 0,
                UNIQUE KEY uk_bar (canonical_symbol, window_start),
                INDEX idx_ts (window_start)
            )
            """)
            logger.info("liquidation_1m table ready")

            # ── New columns on event_feature_snapshots ─────────────────────
            new_columns = [
                ("final_score",      "DECIMAL(10, 4) DEFAULT NULL"),
                ("normalized_score", "DECIMAL(10, 4) DEFAULT NULL"),
                ("funding_rate",     "DECIMAL(20, 8) DEFAULT NULL"),
                ("liq_buy_usd",      "DECIMAL(30, 4) DEFAULT NULL"),
                ("liq_sell_usd",     "DECIMAL(30, 4) DEFAULT NULL"),
                ("liq_total_usd",    "DECIMAL(30, 4) DEFAULT NULL"),
                ("liq_count",        "INT DEFAULT NULL"),
            ]
            for col_name, col_def in new_columns:
                try:
                    cur.execute(
                        f"ALTER TABLE event_feature_snapshots ADD COLUMN {col_name} {col_def}"
                    )
                except Exception as e:
                    if "duplicate" in str(e).lower():
                        pass
                    else:
                        logger.debug("Column %s may already exist: %s", col_name, e)
            logger.info("event_feature_snapshots funding/liquidation columns ready")

    except Exception:
        logger.exception("ensure_extra_schema failed")
    finally:
        conn.close()
