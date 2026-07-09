-- 014: Multi-account support for V7 OKX executor (friend follow-trading)
--
-- Each row = one OKX account authorized to auto-follow V7 signals.
-- Credentials are Fernet-encrypted with OKX_CRED_MASTER_KEY (Railway env);
-- rows are useless without that key.
--
-- The operator's own account stays on env vars for now (label 'main' is
-- reserved); friend accounts are managed via Telegram admin commands.

CREATE TABLE IF NOT EXISTS okx_accounts (
    id                  INT PRIMARY KEY AUTO_INCREMENT,
    label               VARCHAR(32) NOT NULL UNIQUE COMMENT 'short handle, e.g. friend_a',
    owner_chat_id       VARCHAR(32) DEFAULT NULL    COMMENT 'Telegram chat id of the account owner (optional, for routing alerts)',

    -- Fernet-encrypted credentials (never stored in plaintext)
    api_key_enc         VARBINARY(512) NOT NULL,
    api_secret_enc      VARBINARY(512) NOT NULL,
    passphrase_enc      VARBINARY(512) NOT NULL,

    -- Per-account risk parameters (defaults mirror Stage 3 hard rules;
    -- caps may be TIGHTER than main, never looser)
    initial_capital_usd DECIMAL(12,2) NOT NULL,
    notional_lev_mult   DECIMAL(4,2) NOT NULL DEFAULT 2.00,
    daily_loss_cap_pct  DECIMAL(6,2) NOT NULL DEFAULT -20.00,
    total_loss_cap_pct  DECIMAL(6,2) NOT NULL DEFAULT -30.00,

    -- Lifecycle: PENDING (validated, not trading) / ACTIVE (auto-following)
    -- / PAUSED (manual stop) / HALTED / DEMOTED (kill-switch outcomes)
    status              VARCHAR(16) NOT NULL DEFAULT 'PENDING',

    -- Validation snapshot at registration time
    validated_at        DATETIME DEFAULT NULL,
    perm_snapshot       VARCHAR(64) DEFAULT NULL COMMENT 'OKX API perm string at registration, e.g. read_only,trade',
    equity_at_reg       DECIMAL(12,2) DEFAULT NULL,

    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status (status)
);

-- Positions gain an account dimension. Existing rows belong to 'main'
-- (account_id NULL = main/env-var account, keeps backward compat).
ALTER TABLE v7_okx_positions
    ADD COLUMN account_id INT DEFAULT NULL COMMENT 'okx_accounts.id; NULL = main env-var account',
    ADD INDEX idx_account_id (account_id);
