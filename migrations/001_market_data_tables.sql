-- Market Data Layer: Phase 1 tables
-- Run this migration against your MySQL database before starting market_data services.

CREATE TABLE IF NOT EXISTS instruments (
    exchange VARCHAR(32) NOT NULL,
    raw_symbol VARCHAR(64) NOT NULL,
    canonical_symbol VARCHAR(64) NOT NULL,
    instrument_type VARCHAR(16) NOT NULL,
    contract_size DECIMAL(30,10) NULL,
    size_unit VARCHAR(16) NOT NULL,
    PRIMARY KEY (exchange, raw_symbol)
);

CREATE TABLE IF NOT EXISTS normalized_trades (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(32) NOT NULL,
    raw_symbol VARCHAR(64) NOT NULL,
    canonical_symbol VARCHAR(64) NOT NULL,
    instrument_type VARCHAR(16) NOT NULL,
    price DECIMAL(30,10) NOT NULL,
    size DECIMAL(30,10) NOT NULL,
    size_unit VARCHAR(16) NOT NULL,
    taker_side VARCHAR(8) NOT NULL,
    notional_usd DECIMAL(30,10) NOT NULL,
    trade_id VARCHAR(128) NULL,
    ts_exchange BIGINT NOT NULL,
    ts_received BIGINT NOT NULL,
    is_aggregated_trade BOOLEAN NOT NULL DEFAULT FALSE,
    INDEX idx_symbol_time (canonical_symbol, ts_exchange),
    INDEX idx_exchange_time (exchange, ts_exchange)
);

CREATE TABLE IF NOT EXISTS flow_bars_1m (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    canonical_symbol VARCHAR(64) NOT NULL,
    instrument_type VARCHAR(16) NOT NULL,
    exchange_scope VARCHAR(32) NOT NULL,
    window_start BIGINT NOT NULL,
    window_end BIGINT NOT NULL,
    buy_notional_usd DECIMAL(30,10) NOT NULL,
    sell_notional_usd DECIMAL(30,10) NOT NULL,
    delta_usd DECIMAL(30,10) NOT NULL,
    volume_usd DECIMAL(30,10) NOT NULL,
    trade_count INT NOT NULL,
    cvd_usd DECIMAL(30,10) NOT NULL,
    source_count INT NOT NULL,
    quality_score DECIMAL(10,4) NOT NULL DEFAULT 1.0,
    UNIQUE KEY uq_flow_bar (canonical_symbol, instrument_type, exchange_scope, window_start)
);

-- Seed instruments for Phase 1
INSERT IGNORE INTO instruments (exchange, raw_symbol, canonical_symbol, instrument_type, contract_size, size_unit) VALUES
('binance', 'BTCUSDT', 'BTC-USD', 'perp', NULL, 'base'),
('binance', 'ETHUSDT', 'ETH-USD', 'perp', NULL, 'base'),
('okx', 'BTC-USDT-SWAP', 'BTC-USD', 'perp', 0.01, 'contract'),
('okx', 'ETH-USDT-SWAP', 'ETH-USD', 'perp', 0.1, 'contract');
