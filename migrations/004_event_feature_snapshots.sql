-- Event registry: written immediately at webhook time
-- Provides event context for snapshot runner without waiting for 4h observation

CREATE TABLE IF NOT EXISTS event_registry (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_uuid VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) DEFAULT NULL,
    symbol VARCHAR(50) NOT NULL,
    liquidity_side VARCHAR(20) NOT NULL COMMENT 'buy=BSL / sell=SSL',
    entry_price DECIMAL(18,8) NOT NULL,
    trigger_ts INT NOT NULL,
    sweep_ref_price DECIMAL(18,8) DEFAULT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_event_uuid (event_uuid),
    INDEX idx_trigger_ts (trigger_ts)
);

-- Event feature snapshots: one row per (event, time window)
-- Records how the event looks at 15m / 1h / 4h after trigger

CREATE TABLE IF NOT EXISTS event_feature_snapshots (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_uuid VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) DEFAULT NULL,
    canonical_symbol VARCHAR(20) NOT NULL,
    liquidity_side VARCHAR(20) NOT NULL COMMENT 'buy=BSL / sell=SSL',
    trigger_price DECIMAL(18,8) NOT NULL,
    trigger_ts INT NOT NULL,
    snapshot_type VARCHAR(10) NOT NULL COMMENT '15m / 1h / 4h',
    snapshot_ts INT NOT NULL COMMENT 'actual computation unix timestamp',

    -- Features
    delta_value DECIMAL(30,10) DEFAULT NULL COMMENT 'net delta (buy-sell) in window',
    cvd_change DECIMAL(30,10) DEFAULT NULL COMMENT 'CVD change from trigger to snapshot end',
    cvd_sign_flip BOOLEAN DEFAULT NULL COMMENT 'CVD flipped to reversal direction',
    price_change_pct DECIMAL(10,4) DEFAULT NULL COMMENT '% price change from entry',
    reclaim_flag BOOLEAN DEFAULT NULL COMMENT 'price reclaimed sweep_ref level',
    break_again_flag BOOLEAN DEFAULT NULL COMMENT 'price broke further past entry',

    -- Scoring
    reversal_score DECIMAL(10,4) NOT NULL DEFAULT 0,
    continuation_score DECIMAL(10,4) NOT NULL DEFAULT 0,
    confidence_score DECIMAL(10,4) NOT NULL DEFAULT 0 COMMENT 'abs(rev-cont)/max(rev+cont,1)',
    bias VARCHAR(20) NOT NULL DEFAULT 'neutral' COMMENT 'reversal / continuation / neutral',
    label VARCHAR(20) DEFAULT NULL COMMENT 'ground truth, only 4h has final label',

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uk_event_snapshot (event_uuid, snapshot_type),
    INDEX idx_trigger_ts (trigger_ts),
    INDEX idx_snapshot_type (snapshot_type),
    INDEX idx_bias (bias)
);
