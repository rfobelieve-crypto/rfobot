"""Create/verify market_state_bars schema."""
from __future__ import annotations
import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS market_state_bars (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol              VARCHAR(20)    NOT NULL,
    timeframe           VARCHAR(10)    NOT NULL,
    window_start        BIGINT         NOT NULL,
    window_end          BIGINT         NOT NULL,
    delta_usd           DECIMAL(30,4)  DEFAULT NULL,
    delta_ratio         DECIMAL(10,6)  DEFAULT NULL,
    delta_direction     VARCHAR(10)    DEFAULT NULL,
    volume_usd          DECIMAL(30,4)  DEFAULT NULL,
    rel_volume          DECIMAL(10,4)  DEFAULT NULL,
    cvd_change          DECIMAL(30,4)  DEFAULT NULL,
    cvd_slope           DECIMAL(20,8)  DEFAULT NULL,
    cvd_flip            TINYINT        DEFAULT NULL,
    oi_change_pct       DECIMAL(10,4)  DEFAULT NULL,
    oi_direction        VARCHAR(20)    DEFAULT NULL,
    funding_rate        DECIMAL(20,8)  DEFAULT NULL,
    liq_total_usd       DECIMAL(30,4)  DEFAULT NULL,
    rolling_mean        DECIMAL(10,4)  DEFAULT NULL,
    rolling_std         DECIMAL(10,4)  DEFAULT NULL,
    upper_band          DECIMAL(10,4)  DEFAULT NULL,
    lower_band          DECIMAL(10,4)  DEFAULT NULL,
    z_score             DECIMAL(10,4)  DEFAULT NULL,
    reversal_score      DECIMAL(10,4)  DEFAULT NULL,
    continuation_score  DECIMAL(10,4)  DEFAULT NULL,
    confidence          DECIMAL(10,4)  DEFAULT NULL,
    final_bias          VARCHAR(20)    DEFAULT NULL,
    risk_adj_score      DECIMAL(10,4)  DEFAULT NULL,
    signal              TINYINT        NOT NULL DEFAULT 0,
    score_model         VARCHAR(30)    DEFAULT NULL,
    event_count         INT            NOT NULL DEFAULT 0,
    computed_at         BIGINT         DEFAULT NULL,
    UNIQUE KEY uk_bar (symbol, timeframe, window_start),
    INDEX idx_lookup (symbol, timeframe, window_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def ensure_schema():
    """Create market_state_bars if it does not exist."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        logger.info("market_state_bars schema ready")
    except Exception:
        logger.exception("ensure_schema failed")
    finally:
        conn.close()
