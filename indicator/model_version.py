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


# Last known retrain date. Fallback only — must never be EARLIER than the
# real deploy date, so a broken config narrows the sample window instead of
# silently mixing model versions (mistake.md 2026-04-13 calibration lesson).
_DEPLOY_DATE_FALLBACK = "2026-05-01"


def get_current_model_deploy_date() -> str:
    """Deploy date (YYYY-MM-DD) of the current direction model.

    Use this for every since-filter on live-performance queries
    (tracked_signals, indicator_history) instead of hardcoding a date —
    a hardcoded date goes stale at the next retrain and the query starts
    mixing model versions (2026-07-17 alpha-decay false alarm).
    """
    version = get_current_model_version()
    if version != "unknown" and len(version) >= 10:
        return version[:10]
    return _DEPLOY_DATE_FALLBACK


# ── Semantic epochs ──────────────────────────────────────────────────────────
# A model retrain is not the only thing that makes older rows incomparable.
# When the CODE that produces a stored column changes meaning, every row
# before that moment is measuring something else — and because the past is
# read-only (feedback_no_signal_overwrite), the two definitions coexist in
# one table forever.  Any query that pools them silently averages apples and
# oranges, which is exactly how 2026-08-08's level drift hid for months.
#
# Rule: a since-filter must floor at BOTH the model deploy date AND every
# epoch that touches the column it reads.  Use `sample_floor()`.

# Decoding: the rolling-percentile buffer stopped being an in-sample seed
# reset on every deploy and became a live-grown window rebuilt from the DB
# (commit c26b125).  Bars before this were decoded against a distribution
# that was not the one producing them — on the DOWN side the cutoff sat
# below the model's reachable range, so "UP:DOWN ratio" means something
# categorically different either side of this line.
DECODE_EPOCH = "2026-08-12 16:00:00"   # first bar decoded by a live-grown buffer

# Confidence: the scale's denominator changed from max(|up|,|dn|) on the raw
# quantiles to the bar's OWN effective Strong cutoff (commit 02876ba).  Under
# a skewed buffer the old form discounted the narrow side — a Strong DOWN
# scored 54.4 where its mirror UP scored 100 — so confidence values are not
# comparable across this line.
CONFIDENCE_EPOCH = "2026-08-13 00:00:00"


def sample_floor(*epochs: str) -> str:
    """Latest of the model deploy date and any semantic epochs given.

    Returns 'YYYY-MM-DD HH:MM:SS'.  Bare dates are read as midnight, so
    passing a deploy date and a mid-day epoch compares correctly.
    """
    from datetime import datetime

    def _parse(s: str) -> datetime:
        s = s.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        # Unparseable epoch must not silently widen the window.
        raise ValueError(f"sample_floor: cannot parse {s!r}")

    candidates = [get_current_model_deploy_date(), *epochs]
    return max((_parse(c) for c in candidates)).strftime("%Y-%m-%d %H:%M:%S")
