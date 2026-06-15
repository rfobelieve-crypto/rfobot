-- Stage 3 manual-approval queue (2026-05-28).
-- Each PENDING row = a signal awaiting Telegram /yes or /no.
-- After AUTO_MODE_THRESHOLD APPROVED+EXECUTED rows, gate switches to auto.

CREATE TABLE IF NOT EXISTS `v7_okx_approvals` (
  `id`              BIGINT PRIMARY KEY AUTO_INCREMENT,
  `created_at`      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `expires_at`      DATETIME NOT NULL,
  -- PENDING / APPROVED / DENIED / EXPIRED / EXECUTED / STALE
  `status`          VARCHAR(16) NOT NULL DEFAULT 'PENDING',
  -- JSON-serialised intent: direction, size_contracts, entry_price,
  -- stop_price, atr, size_frac, notional_usd, equity_before, tier,
  -- model_version, bar_ts
  `intent`          JSON NOT NULL,
  `decided_at`      DATETIME,
  `decided_by`      VARCHAR(64),
  -- v7_okx_positions row inserted after EXECUTED
  `exec_position_id` BIGINT,
  KEY `idx_status`     (`status`),
  KEY `idx_created`    (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
