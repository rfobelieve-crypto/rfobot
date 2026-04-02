-- Migration 009: flow_bars_15m + unique keys for backfill safety

-- 15-minute flow bars (ML training / inference base)
CREATE TABLE IF NOT EXISTS flow_bars_15m (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    canonical_symbol  VARCHAR(64)    NOT NULL,
    exchange_scope    VARCHAR(32)    NOT NULL,
    window_start      BIGINT         NOT NULL,   -- Unix ms, 15m bucket start
    window_end        BIGINT         NOT NULL,
    buy_notional_usd  DECIMAL(30,4)  NOT NULL DEFAULT 0,
    sell_notional_usd DECIMAL(30,4)  NOT NULL DEFAULT 0,
    delta_usd         DECIMAL(30,4)  NOT NULL DEFAULT 0,
    volume_usd        DECIMAL(30,4)  NOT NULL DEFAULT 0,
    trade_count       INT            NOT NULL DEFAULT 0,
    cvd_usd           DECIMAL(30,4)  NOT NULL DEFAULT 0,
    bar_open          DECIMAL(20,4)  DEFAULT NULL,
    bar_high          DECIMAL(20,4)  DEFAULT NULL,
    bar_low           DECIMAL(20,4)  DEFAULT NULL,
    bar_close         DECIMAL(20,4)  DEFAULT NULL,
    source            VARCHAR(32)    NOT NULL DEFAULT 'live',  -- 'csv_backfill' | 'live'
    UNIQUE KEY uk_bar (canonical_symbol, exchange_scope, window_start),
    INDEX idx_lookup (canonical_symbol, exchange_scope, window_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Add unique keys to oi_snapshots (ignore error if already exists)
ALTER TABLE oi_snapshots
    ADD UNIQUE KEY uk_oi (exchange, canonical_symbol, ts_exchange);

-- Add unique keys to funding_rates (ignore error if already exists)
ALTER TABLE funding_rates
    ADD UNIQUE KEY uk_funding (exchange, canonical_symbol, ts_exchange);
