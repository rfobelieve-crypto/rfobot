"""Single source of truth for the currently deployed model version.

Returns a short ISO-like string (e.g. "2026-05-01T07:42:43") derived from
direction_reg_config.json::trained_at. Stamped on every prediction-writing
path (indicator_history, tracked_signals) so:

  1. Past predictions remain attributable to the model that produced them
     even after retrains — never overwritten.
  2. Performance reports can filter by model_version to compute true
     live OOS metrics for a single model lineage.

Per memory feedback_no_signal_overwrite: the past is read-only. New rows
get the current model_version; existing rows must never be UPDATEd by a
newer model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = (
    Path(__file__).parent / "model_artifacts" / "dual_model"
    / "direction_reg_config.json"
)

_cache: dict[str, Optional[str]] = {"version": None, "mtime": None}


def get_current_model_version() -> str:
    """Read trained_at from direction_reg_config.json. Cached by file mtime
    so a redeploy with a fresh config picks up the new version on next call
    without restart, while normal calls don't re-read the JSON every bar.
    Falls back to 'unknown' if the file is missing or malformed — never
    raises, so a stamping failure can't break update_cycle.
    """
    try:
        mtime = _CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        return "unknown"

    if _cache["mtime"] == mtime and _cache["version"]:
        return _cache["version"]

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        trained_at = cfg.get("trained_at")
        if not trained_at:
            return "unknown"
        # Truncate microseconds for compactness — schema stores VARCHAR(40)
        version = str(trained_at).split(".")[0]
        _cache["version"] = version
        _cache["mtime"] = mtime
        return version
    except Exception as e:
        logger.warning("get_current_model_version failed: %s", e)
        return "unknown"
