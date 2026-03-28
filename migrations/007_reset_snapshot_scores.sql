-- Migration 007: Reset final_score on all snapshots so runner recomputes with correct logic
-- Reason: old snapshots were scored with a bogus price_change_pct (delta/volume ratio fallback)
-- The runner picks up rows where final_score IS NULL and rescores them.
-- Safe to run multiple times.

UPDATE event_feature_snapshots
SET final_score = NULL,
    normalized_score = NULL,
    price_change_pct = NULL,
    reclaim_flag = NULL,
    break_again_flag = NULL
WHERE price_change_pct IS NOT NULL
  AND snapshot_ts < UNIX_TIMESTAMP('2026-03-29 12:00:00');
