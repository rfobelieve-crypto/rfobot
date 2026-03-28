"""
Create OI-related tables and columns. Called on startup.
"""

import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def ensure_oi_schema():
    """Create oi_snapshots table and add OI columns to event_feature_snapshots."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # 1. OI time-series table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS oi_snapshots (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(20) NOT NULL,
                canonical_symbol VARCHAR(20) NOT NULL,
                oi_contracts DECIMAL(30,8) NOT NULL,
                oi_notional_usd DECIMAL(30,4) NOT NULL DEFAULT 0,
                ts_exchange BIGINT NOT NULL,
                ts_received BIGINT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_lookup (canonical_symbol, exchange, ts_exchange),
                INDEX idx_ts (ts_exchange)
            )""")
            logger.info("oi_snapshots table ready")

            # 2. Add lifecycle columns to event_registry
            lifecycle_columns = [
                ("status", "VARCHAR(20) DEFAULT 'active'"),
                ("result_1h", "VARCHAR(50) DEFAULT NULL"),
                ("result_4h", "VARCHAR(50) DEFAULT NULL"),
                ("return_1h", "DECIMAL(10,4) DEFAULT NULL"),
                ("return_4h", "DECIMAL(10,4) DEFAULT NULL"),
                ("finished_at", "DATETIME DEFAULT NULL"),
            ]
            for col_name, col_def in lifecycle_columns:
                try:
                    cur.execute(
                        f"ALTER TABLE event_registry ADD COLUMN {col_name} {col_def}"
                    )
                except Exception:
                    pass  # already exists

            # 3. Add OI columns to event_feature_snapshots
            oi_columns = [
                ("oi_baseline_okx", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_baseline_binance", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_snapshot_okx", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_snapshot_binance", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_change_okx", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_change_binance", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_change_okx_pct", "DECIMAL(10,4) DEFAULT NULL"),
                ("oi_change_binance_pct", "DECIMAL(10,4) DEFAULT NULL"),
                ("oi_change_total", "DECIMAL(30,4) DEFAULT NULL"),
                ("oi_change_total_pct", "DECIMAL(10,4) DEFAULT NULL"),
            ]
            for col_name, col_def in oi_columns:
                try:
                    cur.execute(
                        f"ALTER TABLE event_feature_snapshots ADD COLUMN {col_name} {col_def}"
                    )
                except Exception as e:
                    if "duplicate" in str(e).lower():
                        pass
                    else:
                        logger.debug("Column %s may already exist: %s", col_name, e)

            logger.info("event_feature_snapshots OI columns ready")

    except Exception:
        logger.exception("ensure_oi_schema failed")
    finally:
        conn.close()
