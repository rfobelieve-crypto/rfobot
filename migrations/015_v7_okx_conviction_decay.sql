-- 015: conviction-decay exit support (2026-07-24).
-- See research/conviction_decay_exit.py for the validated backtest and
-- indicator/okx/executor.py for the live mechanism.
--
-- Unlike the backtest sim (a single in-process loop), the live executor
-- reloads `pos` from this table on every cycle — a "consecutive N bars"
-- counter has nowhere to live across cycles unless it's persisted here.
--
-- No automated runner executes migrations/012-015 (nothing in the repo
-- calls run_migration on these files — confirmed by grep before writing
-- this); this Railway MySQL also does not support `ADD COLUMN IF NOT
-- EXISTS` (errors with a syntax error, tested directly). This file is
-- documentation of the schema change; the actual apply was a one-off
-- Python script that checked SHOW COLUMNS first. Kept idempotent-in-spirit
-- via the guard below for anyone who DOES wire up a runner later.

ALTER TABLE `v7_okx_positions`
  ADD COLUMN `decay_streak_count` INT NOT NULL DEFAULT 0
    COMMENT 'consecutive bars where model pred_ret has disagreed with position side; resets to 0 whenever it agrees again';
