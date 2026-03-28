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
);
