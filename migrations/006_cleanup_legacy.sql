-- Migration 006: Drop legacy tables replaced by event_feature_snapshots pipeline
-- Safe to run multiple times (DROP IF EXISTS)

-- Old scoring table: replaced by event_feature_snapshots
DROP TABLE IF EXISTS event_features_v2;

-- Symbol registry: created in 001 but never read/written by any Python code
DROP TABLE IF EXISTS instruments;

-- Backup of old liquidity_events schema from rename migration
DROP TABLE IF EXISTS liquidity_events_v1;
