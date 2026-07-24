-- 016: shadow-mode logging for conviction-decay exit (2026-07-24).
-- Parallel to decay_streak_count (015) but ALWAYS computed regardless of
-- conviction_decay_bars — log-only, never triggers a real close. Lets the
-- mechanism accumulate real live-data evidence (would it have fired, and
-- how would that compare to the actual eventual exit) before it's ever
-- allowed to touch a real position. See indicator/okx/executor.py
-- _manage_position and TODO.md's shadow-mode step.
--
-- Same caveat as 015: no automated runner executes these files; this was
-- applied directly via a one-off Python script (SHOW COLUMNS guard, since
-- this MySQL doesn't support ADD COLUMN IF NOT EXISTS).

ALTER TABLE `v7_okx_positions`
  ADD COLUMN `shadow_decay_streak_count` INT NOT NULL DEFAULT 0
    COMMENT 'always-on shadow computation of conviction_decay streak, regardless of conviction_decay_bars config -- log-only, never triggers a real exit';
