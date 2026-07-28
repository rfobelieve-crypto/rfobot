-- 017: make cancel_playbook_events multi-symbol (2026-07-28).
--
-- The table was written for a single instrument, so it has no symbol column
-- and its uniqueness is (def_version, minute_start_ms, playbook). The moment
-- a second coin runs the same watcher, two coins firing the same playbook in
-- the same minute collide on that key and one row is silently dropped.
--
-- Scope note (measured before writing this): only BTC-USD and ETH-USD have
-- ALL THREE tables the watcher needs — depth_deltas_1m (skew/net), flow_bars_1m
-- (vshock, taker_ratio) and orderbook_snapshots_1m (mid -> ret_1m). The other
-- 8 alts currently have depth_deltas only, and every frozen playbook gates on
-- vshock, so they cannot be enabled until those two collectors are extended.
-- This migration unblocks ETH now and the alts later.
--
-- Existing 340 rows are all BTC-USD; they are backfilled as such so the frozen
-- def_version history stays intact (old rows are never re-labelled).
--
-- Same caveat as 015/016: no automated runner executes these files; applied
-- via a one-off Python script with SHOW COLUMNS / SHOW INDEX guards, since
-- this MySQL supports neither ADD COLUMN IF NOT EXISTS nor DROP INDEX IF EXISTS.

ALTER TABLE `cancel_playbook_events`
  ADD COLUMN `canonical_symbol` VARCHAR(20) NOT NULL DEFAULT 'BTC-USD'
    COMMENT 'instrument this playbook fired on; pre-2026-07-28 rows are all BTC-USD'
    AFTER `def_version`;

UPDATE `cancel_playbook_events` SET `canonical_symbol` = 'BTC-USD'
  WHERE `canonical_symbol` = '' OR `canonical_symbol` IS NULL;

ALTER TABLE `cancel_playbook_events` DROP INDEX `uq_evt`;

ALTER TABLE `cancel_playbook_events`
  ADD UNIQUE KEY `uq_evt` (`def_version`, `canonical_symbol`,
                           `minute_start_ms`, `playbook`);

ALTER TABLE `cancel_playbook_events`
  ADD INDEX `idx_sym_min` (`canonical_symbol`, `minute_start_ms`);
