-- Stage 2 OKX executor tables (skeleton, 2026-05-25).
-- See docs/okx_integration_design.md §7 for design rationale.
-- See indicator/okx/state.py for the DB layer that uses these.

-- ── Positions: parallel to v7_paper_positions, plus OKX-specific cols ─

CREATE TABLE IF NOT EXISTS `v7_okx_positions` (
  `id`              BIGINT PRIMARY KEY AUTO_INCREMENT,
  `entry_time`      DATETIME NOT NULL,
  `direction`       VARCHAR(8) NOT NULL,           -- LONG / SHORT
  `entry_tier`      VARCHAR(16) NOT NULL,          -- Strong / Moderate
  `entry_price`     DOUBLE NOT NULL,
  `atr_at_entry`    DOUBLE NOT NULL,
  `stop_dist`       DOUBLE NOT NULL,
  `trail_extreme`   DOUBLE NOT NULL,
  `current_stop`    DOUBLE NOT NULL,
  `size_contracts`  DECIMAL(12,4) NOT NULL,        -- OKX contract count (fractional, lotSz 0.01)
  `size_frac`       DOUBLE NOT NULL,               -- fraction of equity
  `notional_usd`    DOUBLE NOT NULL,
  `equity_before`   DOUBLE NOT NULL,
  `model_version`   VARCHAR(64),
  `paused_at_signal` TINYINT NOT NULL DEFAULT 0,
  -- OKX identity (filled when REST/WS confirms)
  `entry_cl_ord_id`     VARCHAR(64) NOT NULL,
  `entry_ord_id`        VARCHAR(64),
  `stop_algo_cl_ord_id` VARCHAR(64) NOT NULL,
  `stop_algo_id`        VARCHAR(64),
  -- Close
  `status`          VARCHAR(16) NOT NULL DEFAULT 'OPEN',  -- OPEN / CLOSED / DEMOTED
  `exit_time`       DATETIME,
  `exit_price`      DOUBLE,
  `exit_reason`     VARCHAR(32),
  `exit_fees_usd`   DOUBLE DEFAULT 0,
  `gross_pct`       DOUBLE,
  `net_pct`         DOUBLE,
  `equity_ret_pct`  DOUBLE,
  `equity_after`    DOUBLE,
  -- Audit
  `created_at`      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `uniq_entry_cl_ord` (`entry_cl_ord_id`),
  UNIQUE KEY `uniq_stop_algo_cl_ord` (`stop_algo_cl_ord_id`),
  KEY `idx_status` (`status`),
  KEY `idx_entry_time` (`entry_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Kill trigger log ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS `v7_okx_kill_log` (
  `id`           BIGINT PRIMARY KEY AUTO_INCREMENT,
  `ts`           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `trigger_id`   VARCHAR(16) NOT NULL,           -- e.g. "A4", "B1", "CAP-3"
  `severity`     VARCHAR(16) NOT NULL,           -- HALT / DEMOTE / HARD_FREEZE
  `context`      JSON NOT NULL,
  `resolved_at`  DATETIME,
  `resolution`   TEXT,
  KEY `idx_ts` (`ts`),
  KEY `idx_trigger` (`trigger_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Reconciliation log ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS `v7_okx_reconciliation_log` (
  `id`     BIGINT PRIMARY KEY AUTO_INCREMENT,
  `ts`     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `result` VARCHAR(16) NOT NULL,                 -- CONSISTENT / MISMATCH / UNAVAILABLE
  `detail` JSON,
  KEY `idx_ts` (`ts`),
  KEY `idx_result` (`result`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Executor status (single-row state machine persistence) ──────────

CREATE TABLE IF NOT EXISTS `v7_okx_executor_status` (
  `id`              INT PRIMARY KEY,             -- always 1
  `status`          VARCHAR(16) NOT NULL,        -- INIT / CONNECTING / READY / ACTIVE / HALTED / DEMOTED
  `last_changed_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `reason`          TEXT,
  `trigger_id`      VARCHAR(16),
  `context`         JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── Balance snapshots (for daily_loss_cap day-start anchor) ─────────

CREATE TABLE IF NOT EXISTS `v7_okx_balance_snapshots` (
  `id`             BIGINT PRIMARY KEY AUTO_INCREMENT,
  `ts`             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `total_eq_usd`   DOUBLE NOT NULL,
  `available_usd` DOUBLE NOT NULL,
  `source`         VARCHAR(16) NOT NULL,        -- 'ws' / 'rest' / 'start'
  KEY `idx_ts` (`ts`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
