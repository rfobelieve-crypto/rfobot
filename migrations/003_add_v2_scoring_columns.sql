-- Add v2 scoring columns to event_features_v2
-- Each ALTER is separate so already-existing columns don't block others

ALTER TABLE event_features_v2 ADD COLUMN cvd_zscore_2h DECIMAL(10,4) DEFAULT NULL COMMENT 'z-score of post-2h CVD vs trailing 24h' AFTER absorption_detected;
ALTER TABLE event_features_v2 ADD COLUMN cvd_turned_reversal BOOLEAN DEFAULT NULL COMMENT 'CVD turned in reversal direction post-sweep' AFTER cvd_zscore_2h;
ALTER TABLE event_features_v2 ADD COLUMN reclaim_detected BOOLEAN DEFAULT NULL COMMENT 'price reclaimed sweep ref level within 4h' AFTER cvd_turned_reversal;
ALTER TABLE event_features_v2 ADD COLUMN rebreak_detected BOOLEAN DEFAULT NULL COMMENT 'price broke further past entry within 4h' AFTER reclaim_detected;
