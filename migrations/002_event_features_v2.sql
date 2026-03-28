-- Event Features V2: computed features + scoring for each liquidity event
-- Linked to liquidity_events via event_uuid

CREATE TABLE IF NOT EXISTS event_features_v2 (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_uuid VARCHAR(64) NOT NULL,

    -- Event context (denormalized for query convenience)
    symbol VARCHAR(50) NOT NULL,
    liquidity_side VARCHAR(20) NOT NULL COMMENT 'buy=BSL / sell=SSL',
    entry_price DECIMAL(18,8) NOT NULL,
    trigger_ts INT NOT NULL,
    session VARCHAR(20) DEFAULT NULL,

    -- ── Flow features: pre-sweep (5m before event) ──
    pre_delta_usd DECIMAL(30,10) DEFAULT NULL,
    pre_volume_usd DECIMAL(30,10) DEFAULT NULL,
    pre_buy_sell_ratio DECIMAL(10,4) DEFAULT NULL,
    pre_cvd_usd DECIMAL(30,10) DEFAULT NULL,
    pre_trade_count INT DEFAULT NULL,

    -- ── Flow features: post 2h ──
    post_2h_delta_usd DECIMAL(30,10) DEFAULT NULL,
    post_2h_volume_usd DECIMAL(30,10) DEFAULT NULL,
    post_2h_buy_sell_ratio DECIMAL(10,4) DEFAULT NULL,
    post_2h_cvd_usd DECIMAL(30,10) DEFAULT NULL,
    post_2h_trade_count INT DEFAULT NULL,

    -- ── Flow features: post 4h ──
    post_4h_delta_usd DECIMAL(30,10) DEFAULT NULL,
    post_4h_volume_usd DECIMAL(30,10) DEFAULT NULL,
    post_4h_buy_sell_ratio DECIMAL(10,4) DEFAULT NULL,
    post_4h_cvd_usd DECIMAL(30,10) DEFAULT NULL,
    post_4h_trade_count INT DEFAULT NULL,

    -- ── Flow features: post 6h ──
    post_6h_delta_usd DECIMAL(30,10) DEFAULT NULL,
    post_6h_volume_usd DECIMAL(30,10) DEFAULT NULL,
    post_6h_buy_sell_ratio DECIMAL(10,4) DEFAULT NULL,
    post_6h_cvd_usd DECIMAL(30,10) DEFAULT NULL,
    post_6h_trade_count INT DEFAULT NULL,

    -- ── Derived features ──
    cvd_slope_2h DECIMAL(20,10) DEFAULT NULL COMMENT 'CVD change per minute (2h window)',
    cvd_slope_4h DECIMAL(20,10) DEFAULT NULL COMMENT 'CVD change per minute (4h window)',
    delta_imbalance_2h DECIMAL(10,4) DEFAULT NULL COMMENT '(buy-sell)/volume ratio, 2h',
    delta_imbalance_4h DECIMAL(10,4) DEFAULT NULL COMMENT '(buy-sell)/volume ratio, 4h',
    price_return_2h DECIMAL(10,4) DEFAULT NULL COMMENT '% return 2h after event',
    price_return_4h DECIMAL(10,4) DEFAULT NULL COMMENT '% return 4h after event',
    price_return_6h DECIMAL(10,4) DEFAULT NULL COMMENT '% return 6h after event',
    delta_divergence_2h BOOLEAN DEFAULT NULL COMMENT 'price and delta move opposite directions',
    delta_divergence_4h BOOLEAN DEFAULT NULL COMMENT 'price and delta move opposite directions',
    absorption_detected BOOLEAN DEFAULT NULL COMMENT 'large delta but small price move',

    -- ── V2 scoring features ──
    cvd_zscore_2h DECIMAL(10,4) DEFAULT NULL COMMENT 'z-score of post-2h CVD vs trailing 24h',
    cvd_turned_reversal BOOLEAN DEFAULT NULL COMMENT 'CVD turned in reversal direction post-sweep',
    reclaim_detected BOOLEAN DEFAULT NULL COMMENT 'price reclaimed sweep ref level within 4h',
    rebreak_detected BOOLEAN DEFAULT NULL COMMENT 'price broke further past entry within 4h',

    -- ── Reserved: OI / Liquidation / Orderbook (Phase 3) ──
    oi_change_2h DECIMAL(30,10) DEFAULT NULL,
    oi_change_4h DECIMAL(30,10) DEFAULT NULL,
    liq_buy_usd_2h DECIMAL(30,10) DEFAULT NULL,
    liq_sell_usd_2h DECIMAL(30,10) DEFAULT NULL,
    liq_buy_usd_4h DECIMAL(30,10) DEFAULT NULL,
    liq_sell_usd_4h DECIMAL(30,10) DEFAULT NULL,
    orderbook_imbalance_2h DECIMAL(10,4) DEFAULT NULL,
    orderbook_imbalance_4h DECIMAL(10,4) DEFAULT NULL,

    -- ── Scoring ──
    reversal_score DECIMAL(10,4) NOT NULL DEFAULT 0,
    continuation_score DECIMAL(10,4) NOT NULL DEFAULT 0,
    confidence_score DECIMAL(10,4) NOT NULL DEFAULT 0 COMMENT '0~1, how many features available',
    bias VARCHAR(20) NOT NULL DEFAULT 'neutral' COMMENT 'reversal / continuation / neutral',
    label VARCHAR(20) DEFAULT NULL COMMENT 'actual outcome (backfilled from liquidity_events)',

    -- ── Meta ──
    computed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    scorer_version VARCHAR(16) NOT NULL DEFAULT 'v1',

    UNIQUE KEY uk_event_uuid (event_uuid),
    INDEX idx_trigger_ts (trigger_ts),
    INDEX idx_bias (bias),
    INDEX idx_label (label),
    INDEX idx_session (session)
);
