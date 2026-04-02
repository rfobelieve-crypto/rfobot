-- Migration 011: ML Feature Pipeline Tables
-- ohlcv_1m     : 1m klines from Binance USDT-M futures
-- flow_bars_1m_ml : 1m aggregated taker flow from coin-M aggTrades
-- features_5m / features_15m / features_1h : computed ML feature tables

-- ── ohlcv_1m ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    ts_open         BIGINT         NOT NULL,
    open            DECIMAL(20,4)  DEFAULT NULL,
    high            DECIMAL(20,4)  DEFAULT NULL,
    low             DECIMAL(20,4)  DEFAULT NULL,
    close           DECIMAL(20,4)  DEFAULT NULL,
    volume          DECIMAL(30,4)  DEFAULT NULL,
    quote_vol       DECIMAL(30,4)  DEFAULT NULL,
    trade_count     INT            DEFAULT NULL,
    taker_buy_vol   DECIMAL(30,4)  DEFAULT NULL,
    taker_buy_quote DECIMAL(30,4)  DEFAULT NULL,
    UNIQUE KEY uk_bar (symbol, ts_open),
    INDEX idx_sym_ts (symbol, ts_open)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── flow_bars_1m_ml ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flow_bars_1m_ml (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol          VARCHAR(20)    NOT NULL,
    ts_open         BIGINT         NOT NULL,
    buy_vol         DECIMAL(30,4)  DEFAULT NULL,
    sell_vol        DECIMAL(30,4)  DEFAULT NULL,
    delta           DECIMAL(30,4)  DEFAULT NULL,
    volume          DECIMAL(30,4)  DEFAULT NULL,
    delta_ratio     DECIMAL(10,6)  DEFAULT NULL,
    large_buy_vol   DECIMAL(30,4)  DEFAULT NULL,
    large_sell_vol  DECIMAL(30,4)  DEFAULT NULL,
    trade_count     INT            DEFAULT NULL,
    buy_count       INT            DEFAULT NULL,
    sell_count      INT            DEFAULT NULL,
    UNIQUE KEY uk_bar (symbol, ts_open),
    INDEX idx_sym_ts (symbol, ts_open)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── features_5m ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features_5m (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol              VARCHAR(20)    NOT NULL,
    ts_open             BIGINT         NOT NULL,
    open                DECIMAL(20,4)  DEFAULT NULL,
    high                DECIMAL(20,4)  DEFAULT NULL,
    low                 DECIMAL(20,4)  DEFAULT NULL,
    close               DECIMAL(20,4)  DEFAULT NULL,
    volume              DECIMAL(30,4)  DEFAULT NULL,
    return_1b           DECIMAL(10,6)  DEFAULT NULL,
    realized_vol_20b    DECIMAL(10,6)  DEFAULT NULL,
    buy_vol             DECIMAL(30,4)  DEFAULT NULL,
    sell_vol            DECIMAL(30,4)  DEFAULT NULL,
    delta               DECIMAL(30,4)  DEFAULT NULL,
    delta_ratio         DECIMAL(10,6)  DEFAULT NULL,
    cvd                 DECIMAL(30,4)  DEFAULT NULL,
    cvd_zscore          DECIMAL(10,6)  DEFAULT NULL,
    large_buy_vol       DECIMAL(30,4)  DEFAULT NULL,
    large_sell_vol      DECIMAL(30,4)  DEFAULT NULL,
    large_delta         DECIMAL(30,4)  DEFAULT NULL,
    oi                  DECIMAL(30,4)  DEFAULT NULL,
    oi_delta            DECIMAL(30,4)  DEFAULT NULL,
    oi_accel            DECIMAL(30,4)  DEFAULT NULL,
    oi_divergence       TINYINT        DEFAULT NULL,
    funding_rate        DECIMAL(14,10) DEFAULT NULL,
    funding_deviation   DECIMAL(14,10) DEFAULT NULL,
    funding_zscore      DECIMAL(8,4)   DEFAULT NULL,
    cvd_x_oi_delta      DECIMAL(10,6)  DEFAULT NULL,
    funding_x_cvd       DECIMAL(10,6)  DEFAULT NULL,
    cvd_oi_ratio        DECIMAL(10,6)  DEFAULT NULL,
    ema_9               DECIMAL(20,4)  DEFAULT NULL,
    ema_21              DECIMAL(20,4)  DEFAULT NULL,
    macd                DECIMAL(10,6)  DEFAULT NULL,
    macd_signal         DECIMAL(10,6)  DEFAULT NULL,
    return_lag_1        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_2        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_3        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_4        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_5        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_6        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_7        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_8        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_9        DECIMAL(10,6)  DEFAULT NULL,
    return_lag_10       DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_1         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_2         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_3         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_4         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_5         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_6         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_7         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_8         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_9         DECIMAL(10,6)  DEFAULT NULL,
    delta_lag_10        DECIMAL(10,6)  DEFAULT NULL,
    future_return_5m    DECIMAL(10,6)  DEFAULT NULL,
    future_return_15m   DECIMAL(10,6)  DEFAULT NULL,
    future_return_1h    DECIMAL(10,6)  DEFAULT NULL,
    label_5m            VARCHAR(10)    DEFAULT NULL,
    label_15m           VARCHAR(10)    DEFAULT NULL,
    label_1h            VARCHAR(10)    DEFAULT NULL,
    bull_bear_score     DECIMAL(6,2)   DEFAULT NULL,
    computed_at         BIGINT         DEFAULT NULL,
    UNIQUE KEY uk_bar (symbol, ts_open),
    INDEX idx_sym_ts (symbol, ts_open)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── features_15m (identical structure) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS features_15m LIKE features_5m;
ALTER TABLE features_15m ADD UNIQUE KEY uk_bar (symbol, ts_open);

-- ── features_1h (identical structure) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features_1h LIKE features_5m;
ALTER TABLE features_1h ADD UNIQUE KEY uk_bar (symbol, ts_open);
