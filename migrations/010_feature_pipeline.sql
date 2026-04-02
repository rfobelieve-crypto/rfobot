-- Migration 010: Event-driven feature pipeline
-- Layer 2: clean aligned 15m features
-- Layer 3: event-based research dataset

-- ── LAYER 2: feature_bars_15m ─────────────────────────────────────────────
-- One row per (symbol, 15m bucket). Merges flow + OI + funding.
-- No forward-looking data. Every column uses only info available at bar close.
CREATE TABLE IF NOT EXISTS feature_bars_15m (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    bucket_15m      BIGINT         NOT NULL,   -- window_start ms

    -- Flow (from flow_bars_15m)
    buy_notional    DECIMAL(30,4)  DEFAULT NULL,
    sell_notional   DECIMAL(30,4)  DEFAULT NULL,
    delta_usd       DECIMAL(30,4)  DEFAULT NULL,
    volume_usd      DECIMAL(30,4)  DEFAULT NULL,
    delta_ratio     DECIMAL(10,6)  DEFAULT NULL,  -- delta/volume [-1,+1]
    cvd_usd         DECIMAL(30,4)  DEFAULT NULL,
    trade_count     INT            DEFAULT NULL,

    -- OHLCV
    bar_open        DECIMAL(20,4)  DEFAULT NULL,
    bar_high        DECIMAL(20,4)  DEFAULT NULL,
    bar_low         DECIMAL(20,4)  DEFAULT NULL,
    bar_close       DECIMAL(20,4)  DEFAULT NULL,

    -- OI (last snapshot at or before bucket close)
    oi_usd          DECIMAL(30,4)  DEFAULT NULL,
    oi_change_usd   DECIMAL(30,4)  DEFAULT NULL,  -- vs previous 15m bucket
    oi_change_pct   DECIMAL(10,6)  DEFAULT NULL,

    -- Funding (last rate at or before bucket close)
    funding_rate    DECIMAL(14,10) DEFAULT NULL,
    funding_zscore  DECIMAL(8,4)   DEFAULT NULL,  -- rolling 48-bar (~12h) z-score

    -- Quality
    data_source     VARCHAR(32)    DEFAULT NULL,   -- 'live' | 'binance_coinm'
    computed_at     BIGINT         DEFAULT NULL,

    UNIQUE KEY uk_bar  (symbol, bucket_15m),
    INDEX      idx_ts  (symbol, bucket_15m)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ── LAYER 3: event_features ───────────────────────────────────────────────
-- One row per liquidity sweep event.
-- Pre-event features use T-1, T-2, T-4 bars (strictly before the event).
-- Post-event outcomes use T+1, T+2, T+4 bar closes.
CREATE TABLE IF NOT EXISTS event_features (
    id                     BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_uuid             VARCHAR(64)    NOT NULL,
    symbol                 VARCHAR(20)    NOT NULL,
    event_side             VARCHAR(10)    NOT NULL,  -- 'BSL' | 'SSL'
    trigger_ts_ms          BIGINT         NOT NULL,
    trigger_price          DECIMAL(20,4)  DEFAULT NULL,
    trigger_bucket_15m     BIGINT         NOT NULL,

    -- Pre-event features (T = trigger_bucket, T-1 = bar before)
    delta_1bar             DECIMAL(10,6)  DEFAULT NULL,  -- delta_ratio at T-1
    delta_2bar             DECIMAL(10,6)  DEFAULT NULL,  -- delta_ratio at T-2
    delta_4bar             DECIMAL(10,6)  DEFAULT NULL,  -- delta_ratio at T-4
    oi_change_1bar         DECIMAL(10,6)  DEFAULT NULL,  -- oi_change_pct at T-1
    oi_change_2bar         DECIMAL(10,6)  DEFAULT NULL,
    funding_rate           DECIMAL(14,10) DEFAULT NULL,
    funding_zscore         DECIMAL(8,4)   DEFAULT NULL,

    -- Interaction features
    pressure               DECIMAL(10,6)  DEFAULT NULL,  -- delta_ratio × oi_change_pct
    delta_price_divergence DECIMAL(10,6)  DEFAULT NULL,  -- delta vs price direction
    flow_acceleration      DECIMAL(10,6)  DEFAULT NULL,  -- delta_1bar - delta_2bar

    -- Post-event outcomes
    return_1bar            DECIMAL(10,6)  DEFAULT NULL,  -- 15m fwd return %
    return_2bar            DECIMAL(10,6)  DEFAULT NULL,  -- 30m
    return_4bar            DECIMAL(10,6)  DEFAULT NULL,  -- 1h

    -- Label (BSL: down=reversal, up=continuation; SSL: up=reversal, down=continuation)
    label                  VARCHAR(20)    DEFAULT NULL,  -- 'reversal'|'continuation'|'neutral'
    labeled_at             BIGINT         DEFAULT NULL,

    UNIQUE KEY uk_event   (event_uuid),
    INDEX      idx_lookup (symbol, event_side, trigger_bucket_15m)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
