# -*- coding: utf-8 -*-
"""Upstream-outage degradation guard (2026-09-01).

Answers a question asked of the system on 2026-09-01: "what happens if
Coinglass stops, and what happens when it comes back?" The code walk found
the honest answers:

  * 91 of the 136 direction features (67%) and 60 of 76 magnitude features
    are `cg_*`. An outage does not raise — `fetch_coinglass` falls back to
    `.data_cache/*.parquet` and `merge_asof(direction="backward")` carries
    the last value forward with NO time limit.
  * Frozen inputs are worse than missing ones: the 17 change/diff features
    become exactly 0 ("funding never moved"), and the 20 z-scores DRIFT TO
    EXTREMES because a constant series' rolling std shrinks toward zero.
  * The dangerous moment is the RECOVERY: every frozen feature jumps a
    multi-day delta inside one bar, products of two z-scores blow up, the
    model lands on leaf nodes it never saw in training, and an extreme
    pred is — by construction of rolling-percentile decoding — a Strong
    signal. That signal carries zero information and the product side
    trades it with real money.

Same family as mistake.md 2026-08-08 / 2026-08-11: the decoder's reference
distribution stops describing what it is judging. Those were fixed with a
staleness limit and a post-reset silence; this is the data-source version
of both.

Three guards, in the order they fire:
  1. STALENESS TOLERANCE — feature_builder_live passes `tolerance` to
     merge_asof (CG_MERGE_TOLERANCE, default 6h). Past that the feature is
     NaN, not a lie. XGBoost has a trained default branch for NaN; it has
     none for "a constant that has been frozen for three days".
  2. DEGRADED FLAG — `assess(cg_status)` classifies the cycle OK /
     DEGRADED / OUTAGE from the number of empty endpoints; the caller
     alerts and can pause NEW ENTRIES (exits, kill checks and
     reconciliation always keep running — the OKX_ENTRY_PAUSED semantics).
  3. RECOVERY SILENCE — after returning from DEGRADED/OUTAGE the guard
     stays in RECOVERING for RECOVERY_SILENCE_BARS bars, during which
     signals are suppressed. This lets the rolling z-windows and the
     200-bar decode buffer flush the frozen stretch, exactly as the
     2026-08-11 retrain warm-up silence does.

State lives in MySQL (`data_degradation_state`), not in a container file:
the 2026-08-11 lesson is that rolling state on an ephemeral filesystem is
reset by every deploy, turning "temporary" into "every day".
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# merge_asof staleness limit for Coinglass columns.
CG_MERGE_TOLERANCE = os.environ.get("CG_MERGE_TOLERANCE", "6h")

# Endpoint-failure thresholds (out of ~24 CG endpoints).
DEGRADED_MIN_FAILED = 3          # a few flaky endpoints = degraded
OUTAGE_FRACTION = 0.5            # half or more down = outage

# Counting endpoints is the WRONG metric on its own (2026-09-01 review):
# `funding` and `oi` each feed ~20 derived features, so losing one of them
# hurts more than losing three peripheral feeds. These are weighted: any
# one of them empty is DEGRADED on its own, two or more is an OUTAGE —
# regardless of how many endpoints are up in total.
CRITICAL_ENDPOINTS = ("funding", "oi", "oi_agg", "taker")
CRITICAL_OUTAGE_MIN = 2

# Bars of silence after recovery. 24 = one day of hourly bars: enough for
# the 4h-horizon labels and the short rolling windows to clear the frozen
# stretch. Deliberately NOT tuned — a swept value here would be a
# threshold-sweep on a safety guard.
RECOVERY_SILENCE_BARS = 24

STATE_OK = "OK"
STATE_DEGRADED = "DEGRADED"
STATE_OUTAGE = "OUTAGE"
STATE_RECOVERING = "RECOVERING"


def classify(n_failed: int, n_total: int, failed_names=None) -> str:
    """Pure function — the whole policy, testable without a DB.

    Two axes: raw count, and criticality. One critical endpoint down is
    already DEGRADED even when 23 of 24 are up.
    """
    if n_total <= 0:
        return STATE_OUTAGE
    crit = [f for f in (failed_names or []) if f in CRITICAL_ENDPOINTS]
    if len(crit) >= CRITICAL_OUTAGE_MIN:
        return STATE_OUTAGE
    if n_failed >= max(1, int(round(n_total * OUTAGE_FRACTION))):
        return STATE_OUTAGE
    if n_failed >= DEGRADED_MIN_FAILED or crit:
        return STATE_DEGRADED
    return STATE_OK


def next_state(prev_state: str, prev_recovery_left: int,
               observed: str) -> tuple[str, int]:
    """State machine: (new_state, recovery_bars_left).

    OK/DEGRADED/OUTAGE come from `observed`. The only memory is the
    recovery countdown: leaving a bad state does not go straight to OK,
    it goes to RECOVERING for RECOVERY_SILENCE_BARS bars. A new outage
    during recovery restarts the whole countdown (the point is a clean
    window, not a fixed delay).
    """
    if observed in (STATE_DEGRADED, STATE_OUTAGE):
        return observed, RECOVERY_SILENCE_BARS
    # observed == OK
    if prev_state in (STATE_DEGRADED, STATE_OUTAGE):
        return STATE_RECOVERING, RECOVERY_SILENCE_BARS
    if prev_state == STATE_RECOVERING:
        left = max(0, int(prev_recovery_left) - 1)
        return (STATE_OK, 0) if left == 0 else (STATE_RECOVERING, left)
    return STATE_OK, 0


def should_suppress_signals(state: str) -> bool:
    """True = do not publish a tradable signal this cycle."""
    return state in (STATE_DEGRADED, STATE_OUTAGE, STATE_RECOVERING)


# ── persistence (MySQL, never a container file — mistake 2026-08-11) ─────

def _ensure_table(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS data_degradation_state (
            id TINYINT PRIMARY KEY,
            state VARCHAR(16) NOT NULL,
            recovery_left INT NOT NULL DEFAULT 0,
            n_failed INT NOT NULL DEFAULT 0,
            n_total INT NOT NULL DEFAULT 0,
            failed_names TEXT NULL,
            updated_at DATETIME NOT NULL
                DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")


def load_state() -> tuple[str, int]:
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                _ensure_table(cur)
                cur.execute("SELECT state, recovery_left FROM "
                            "data_degradation_state WHERE id=1")
                row = cur.fetchone()
            conn.commit()
        finally:
            conn.close()
        if row:
            return str(row["state"]), int(row["recovery_left"])
    except Exception as e:                                  # pragma: no cover
        logger.warning("degradation state load failed (assuming OK): %s", e)
    return STATE_OK, 0


def save_state(state: str, recovery_left: int, n_failed: int,
               n_total: int, failed_names: list | None = None) -> None:
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                _ensure_table(cur)
                cur.execute(
                    "INSERT INTO data_degradation_state (id, state, "
                    "recovery_left, n_failed, n_total, failed_names) "
                    "VALUES (1,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "state=VALUES(state), recovery_left=VALUES(recovery_left), "
                    "n_failed=VALUES(n_failed), n_total=VALUES(n_total), "
                    "failed_names=VALUES(failed_names)",
                    (state, int(recovery_left), int(n_failed), int(n_total),
                     ",".join(failed_names or [])[:2000]))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:                                  # pragma: no cover
        logger.warning("degradation state save failed: %s", e)


def assess(cg_status: dict) -> dict:
    """Run one cycle of the guard. Returns a dict for the caller/alerts.

    Never raises: a broken guard must not take the hot path down (it is
    called from update_cycle, mistake.md 2026-04-22).
    """
    try:
        failed = [k for k, v in (cg_status or {}).items() if v.get("empty")]
        n_total = len(cg_status or {})
        observed = classify(len(failed), n_total, failed)
        prev_state, prev_left = load_state()
        state, left = next_state(prev_state, prev_left, observed)
        save_state(state, left, len(failed), n_total, failed)
        changed = state != prev_state
        out = {
            "state": state, "prev_state": prev_state,
            "observed": observed, "recovery_left": left,
            "n_failed": len(failed), "n_total": n_total,
            "failed": failed, "changed": changed,
            "suppress": should_suppress_signals(state),
            "asof": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        }
        if changed:
            logger.warning("DATA DEGRADATION state %s -> %s (%d/%d endpoints "
                           "failed)", prev_state, state, len(failed), n_total)
        return out
    except Exception as e:                                  # pragma: no cover
        logger.exception("degradation guard failed (fail-open): %s", e)
        return {"state": STATE_OK, "prev_state": STATE_OK, "observed": STATE_OK,
                "recovery_left": 0, "n_failed": 0, "n_total": 0, "failed": [],
                "changed": False, "suppress": False, "error": str(e)}


def alert_text(res: dict) -> str | None:
    """Operator-facing alert for a state CHANGE; None when nothing to say."""
    if not res.get("changed"):
        return None
    st, prev = res["state"], res["prev_state"]
    if st in (STATE_DEGRADED, STATE_OUTAGE):
        _crit = [f for f in res.get("failed", []) if f in CRITICAL_ENDPOINTS]
        if _crit:
            logger.warning("critical CG endpoints down: %s", _crit)
        return (f"🟠 DATA {st}: Coinglass {res['n_failed']}/{res['n_total']} "
                f"endpoints down ({', '.join(res['failed'][:6])})\n"
                f"Signals suppressed; exits/kill checks unaffected.\n"
                f"Stale features are NaN past {CG_MERGE_TOLERANCE}, not "
                f"carried forward.")
    if st == STATE_RECOVERING:
        return (f"🟡 DATA RECOVERING: upstream is back — silencing signals "
                f"for {res['recovery_left']} bars so the rolling windows "
                f"and decode buffer flush the frozen stretch "
                f"(recovery-jump guard).")
    if st == STATE_OK and prev == STATE_RECOVERING:
        return "🟢 DATA OK: recovery silence finished, signals resume."
    return None
