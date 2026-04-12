"""
Config Store — Persistent runtime config that agents can modify.

Provides a JSON-backed override layer on top of Python constants.
Inference reads from here; agents write to here during meetings.

Architecture:
  Python defaults (inference.py) ← config_overrides.json ← agent actions
  If a key exists in overrides, it wins. Otherwise, the Python default is used.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))
CONFIG_PATH = Path(__file__).parent.parent / "model_artifacts" / "config_overrides.json"

_lock = Lock()

# Keys that agents are allowed to modify (whitelist)
ALLOWED_KEYS = {
    # Dual model direction thresholds
    "DUAL_DIR_UP_TH",
    "DUAL_DIR_DN_TH",
    # Signal strength thresholds
    "STRONG_THRESHOLD",
    "MODERATE_THRESHOLD",
    # Deadzone
    "STRENGTH_DEADZONE",
    "CHOPPY_DEADZONE_MULT",
    "TREND_DEADZONE_MULT",
    "BULL_CONTRA_PENALTY",
    # BBP gate
    "BBP_CONFIRM_THRESHOLD",
    "BBP_CONFIRM_ENABLED",
    # Hysteresis / cooldown
    "HYSTERESIS_MULT",
    "FLIP_COOLDOWN_BARS",
    # Monitor thresholds
    "ALERT_COOLDOWN_H",
}


def load_overrides() -> dict:
    """Load current overrides from disk."""
    with _lock:
        if not CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load config overrides: %s", e)
            return {}


def get_override(key: str, default=None):
    """Get a single override value, or default if not set."""
    overrides = load_overrides()
    return overrides.get("values", {}).get(key, default)


def set_override(key: str, value, reason: str, agent: str = "unknown") -> dict:
    """
    Set a config override. Returns the change record.

    Only ALLOWED_KEYS can be modified. This is a safety whitelist
    so agents can't accidentally modify critical system constants.
    """
    if key not in ALLOWED_KEYS:
        return {"error": f"Key '{key}' is not in the allowed override list. Allowed: {sorted(ALLOWED_KEYS)}"}

    with _lock:
        data = {}
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}

        if "values" not in data:
            data["values"] = {}
        if "history" not in data:
            data["history"] = []

        old_value = data["values"].get(key)
        data["values"][key] = value

        record = {
            "key": key,
            "old": old_value,
            "new": value,
            "reason": reason,
            "agent": agent,
            "time": datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M:%S"),
        }
        data["history"].append(record)

        # Keep only last 50 history entries
        if len(data["history"]) > 50:
            data["history"] = data["history"][-50:]

        data["last_modified"] = record["time"]

        CONFIG_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger.info("Config override: %s = %s (was %s) — %s [%s]",
                key, value, old_value, reason, agent)
    return record


def remove_override(key: str, reason: str, agent: str = "unknown") -> dict:
    """Remove an override, reverting to Python default."""
    with _lock:
        if not CONFIG_PATH.exists():
            return {"error": "No overrides file"}

        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        old_value = data.get("values", {}).pop(key, None)
        if old_value is None:
            return {"error": f"Key '{key}' was not overridden"}

        record = {
            "key": key,
            "old": old_value,
            "new": "(reverted to default)",
            "reason": reason,
            "agent": agent,
            "time": datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M:%S"),
        }
        data.setdefault("history", []).append(record)
        data["last_modified"] = record["time"]

        CONFIG_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return record


def list_overrides() -> dict:
    """List all current overrides with their history."""
    data = load_overrides()
    return {
        "values": data.get("values", {}),
        "last_modified": data.get("last_modified"),
        "recent_changes": data.get("history", [])[-10:],
    }
