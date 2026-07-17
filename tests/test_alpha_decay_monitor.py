"""Regression tests for the 2026-07-17 alpha-decay false alarm.

Two stale-baseline bugs made the monitor cry wolf:
  1. CURRENT_MODEL_DEPLOY hardcoded "2026-04-17" while the live model was
     retrained 2026-05-01 → confidence-WR mixed model versions.
  2. Importance drift compared the current CSV against a snapshot taken
     BEFORE the retrain → permanently "critical" after every retrain.

These tests pin the fixes: deploy date is resolved from the model artifact
(single source of truth), and pre-retrain snapshots are rejected as a
comparison baseline.
"""
import json
import re
from pathlib import Path

import indicator.model_version as mv
from indicator.alpha_decay_monitor import _snapshot_predates_model
from indicator.model_version import get_current_model_deploy_date

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = (PROJECT_ROOT / "indicator" / "model_artifacts"
               / "dual_model" / "direction_reg_config.json")


class TestDeployDate:
    def test_matches_artifact_trained_at(self):
        cfg = json.loads(CONFIG_PATH.read_text())
        assert get_current_model_deploy_date() == cfg["trained_at"][:10]

    def test_iso_date_format(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", get_current_model_deploy_date())

    def test_not_the_stale_hardcode(self):
        # The bug: "2026-04-17" survived the 2026-05-01 retrain.
        assert get_current_model_deploy_date() > "2026-04-17"

    def test_fallback_when_version_unknown(self, monkeypatch):
        monkeypatch.setattr(mv, "get_current_model_version", lambda: "unknown")
        assert get_current_model_deploy_date() == mv._DEPLOY_DATE_FALLBACK


class TestSnapshotPredatesModel:
    def test_pre_retrain_snapshot_rejected(self):
        # The exact false-alarm pair: 04-19 snapshot vs 05-01 model.
        assert _snapshot_predates_model(
            "direction_importance_20260419_regv7_136f", "2026-05-01") is True

    def test_post_retrain_snapshot_accepted(self):
        assert _snapshot_predates_model(
            "direction_importance_20260717_abc1234", "2026-05-01") is False

    def test_same_day_snapshot_accepted(self):
        assert _snapshot_predates_model(
            "direction_importance_20260501_regv7", "2026-05-01") is False

    def test_unparseable_name_fails_open(self):
        # A malformed filename must not block the drift check.
        assert _snapshot_predates_model("weird", "2026-05-01") is False
