-- Migration 008: market_state_bars
-- Time-driven continuous score series. One row per (symbol, timeframe, window).
-- Single source of truth for visualization.

CREATE TABLE IF NOT EXISTS market_state_bars (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol              VARCHAR(20)    NOT NULL,
    timeframe           VARCHAR(10)    NOT NULL,
    window_start        BIGINT         NOT NULL,   -- Unix ms
    window_end          BIGINT         NOT NULL,
    -- Flow features
    delta_usd           DECIMAL(30,4)  DEFAULT NULL,
    delta_ratio         DECIMAL(10,6)  DEFAULT NULL,
    delta_direction     VARCHAR(10)    DEFAULT NULL,
    volume_usd          DECIMAL(30,4)  DEFAULT NULL,
    rel_volume          DECIMAL(10,4)  DEFAULT NULL,
    -- CVD features
    cvd_change          DECIMAL(30,4)  DEFAULT NULL,
    cvd_slope           DECIMAL(20,8)  DEFAULT NULL,
    cvd_flip            TINYINT        DEFAULT NULL,
    -- OI features
    oi_change_pct       DECIMAL(10,4)  DEFAULT NULL,
    oi_direction        VARCHAR(20)    DEFAULT NULL,
    -- Macro features
    funding_rate        DECIMAL(20,8)  DEFAULT NULL,
    liq_total_usd       DECIMAL(30,4)  DEFAULT NULL,
    -- Statistical bands (features/statistics.py)
    rolling_mean        DECIMAL(10,4)  DEFAULT NULL,
    rolling_std         DECIMAL(10,4)  DEFAULT NULL,
    upper_band          DECIMAL(10,4)  DEFAULT NULL,
    lower_band          DECIMAL(10,4)  DEFAULT NULL,
    z_score             DECIMAL(10,4)  DEFAULT NULL,
    -- Scores
    reversal_score      DECIMAL(10,4)  DEFAULT NULL,
    continuation_score  DECIMAL(10,4)  DEFAULT NULL,
    confidence          DECIMAL(10,4)  DEFAULT NULL,
    final_bias          VARCHAR(20)    DEFAULT NULL,
    risk_adj_score      DECIMAL(10,4)  DEFAULT NULL,
    `signal`            TINYINT        NOT NULL DEFAULT 0,
    score_model         VARCHAR(30)    DEFAULT NULL,
    -- Event overlay (event is a feature, not a condition for bar existence)
    event_count         INT            NOT NULL DEFAULT 0,
    bar_open            DECIMAL(20,2)  DEFAULT NULL,
    bar_high            DECIMAL(20,2)  DEFAULT NULL,
    bar_low             DECIMAL(20,2)  DEFAULT NULL,
    bar_close           DECIMAL(20,2)  DEFAULT NULL,
    computed_at         BIGINT         DEFAULT NULL,
    UNIQUE KEY uk_bar (symbol, timeframe, window_start),
    INDEX idx_lookup (symbol, timeframe, window_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
