# -*- coding: utf-8 -*-
"""Tests for the upstream-outage degradation guard (2026-09-01).

Known-answer first (mistake.md 2026-07-29): the frozen-then-jump behaviour
this guard exists to stop is reproduced directly on a synthetic series, so
the fix is verified against a病灶 it can actually demonstrate, not against
an assertion that it "looks right".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicator import data_degradation as dg


# ── 1. the disease, reproduced (known answer) ────────────────────────────

def test_frozen_series_makes_diffs_zero_and_zscores_explode():
    """Why a staleness tolerance beats carrying values forward.

    A constant stretch drives diff -> 0 and, because the rolling std
    collapses, the z-score of the recovery jump becomes enormous. This is
    the exact input shape that lands XGBoost on unseen leaves.
    """
    # A realistic live feature is a noisy walk, not a ramp — a ramp has a
    # constant diff, which itself collapses the rolling std and would make
    # the "normal" period look as broken as the frozen one (caught by this
    # test failing on its first draft: the fixture was the artefact).
    rng = np.random.default_rng(7)
    normal = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.3, 200)))
    frozen = pd.Series([float(normal.iloc[-1])] * 72)   # 3 days carried fwd
    jump = pd.Series([float(normal.iloc[-1]) + 14.0])   # upstream returns
    s = pd.concat([normal, frozen, jump], ignore_index=True)

    diff = s.diff()
    assert (diff.iloc[201:272] == 0).all(), "frozen stretch must show zero change"

    # Causal z: the std window must EXCLUDE the current bar, which is how a
    # trailing feature is actually computed — and it is precisely why the
    # recovery bar explodes (its reference std was built from constants).
    roll_std = diff.rolling(24).std().shift(1)
    z = diff / roll_std
    z_recovery = float(z.iloc[-1])
    z_normal = float(np.nanmedian(np.abs(z.iloc[30:200])))
    assert not np.isfinite(z_recovery) or abs(z_recovery) > 50 * z_normal, (
        f"recovery z ({z_recovery}) should dwarf normal z ({z_normal:.2f})")


def test_nan_is_survivable_where_frozen_is_not():
    """NaN propagates as NaN (a state the model was trained to handle);
    a frozen constant silently manufactures a false 'nothing moved'."""
    frozen = pd.Series([101.0] * 10).diff()
    nan_gap = pd.Series([101.0] + [np.nan] * 8 + [115.0]).diff()
    assert (frozen.iloc[1:] == 0).all()
    assert nan_gap.iloc[1:].isna().all() or np.isnan(nan_gap.iloc[-1])


# ── 2. classification policy ─────────────────────────────────────────────

@pytest.mark.parametrize("failed,total,expect", [
    (0, 24, dg.STATE_OK),
    (2, 24, dg.STATE_OK),            # a couple of flaky endpoints
    (3, 24, dg.STATE_DEGRADED),
    (11, 24, dg.STATE_DEGRADED),
    (12, 24, dg.STATE_OUTAGE),       # half down
    (24, 24, dg.STATE_OUTAGE),
    (0, 0, dg.STATE_OUTAGE),         # nothing fetched at all
])
def test_classify(failed, total, expect):
    assert dg.classify(failed, total) == expect


# ── 3. state machine, including the recovery silence ─────────────────────

def test_outage_then_recovery_silence_then_ok():
    state, left = dg.STATE_OK, 0
    state, left = dg.next_state(state, left, dg.STATE_OUTAGE)
    assert state == dg.STATE_OUTAGE and dg.should_suppress_signals(state)

    # upstream returns: must NOT go straight to OK
    state, left = dg.next_state(state, left, dg.STATE_OK)
    assert state == dg.STATE_RECOVERING and left == dg.RECOVERY_SILENCE_BARS
    assert dg.should_suppress_signals(state)

    for i in range(dg.RECOVERY_SILENCE_BARS - 1):
        state, left = dg.next_state(state, left, dg.STATE_OK)
        assert state == dg.STATE_RECOVERING, f"resumed too early at bar {i}"
    state, left = dg.next_state(state, left, dg.STATE_OK)
    assert state == dg.STATE_OK and left == 0
    assert not dg.should_suppress_signals(state)


def test_new_outage_during_recovery_restarts_the_countdown():
    state, left = dg.STATE_RECOVERING, 5
    state, left = dg.next_state(state, left, dg.STATE_OUTAGE)
    assert state == dg.STATE_OUTAGE and left == dg.RECOVERY_SILENCE_BARS
    state, left = dg.next_state(state, left, dg.STATE_OK)
    assert left == dg.RECOVERY_SILENCE_BARS, "countdown must restart, not resume"


def test_healthy_stays_healthy():
    state, left = dg.next_state(dg.STATE_OK, 0, dg.STATE_OK)
    assert state == dg.STATE_OK and left == 0 and not dg.should_suppress_signals(state)


# ── 4. alerts fire only on transitions ───────────────────────────────────

def test_alert_only_on_change():
    assert dg.alert_text({"changed": False}) is None
    txt = dg.alert_text({"changed": True, "state": dg.STATE_OUTAGE,
                         "prev_state": dg.STATE_OK, "n_failed": 20,
                         "n_total": 24, "failed": ["oi", "funding"],
                         "recovery_left": 24})
    assert txt and "OUTAGE" in txt and "20/24" in txt
    rec = dg.alert_text({"changed": True, "state": dg.STATE_RECOVERING,
                         "prev_state": dg.STATE_OUTAGE, "n_failed": 0,
                         "n_total": 24, "failed": [], "recovery_left": 24})
    assert rec and "RECOVERING" in rec
    done = dg.alert_text({"changed": True, "state": dg.STATE_OK,
                          "prev_state": dg.STATE_RECOVERING, "n_failed": 0,
                          "n_total": 24, "failed": [], "recovery_left": 0})
    assert done and "OK" in done


# ── 5. the guard never takes the hot path down ───────────────────────────

def test_assess_survives_garbage_input(monkeypatch):
    monkeypatch.setattr(dg, "load_state", lambda: (dg.STATE_OK, 0))
    monkeypatch.setattr(dg, "save_state", lambda *a, **k: None)
    res = dg.assess(None)
    assert res["state"] == dg.STATE_OUTAGE      # nothing fetched = outage
    assert isinstance(res["suppress"], bool)


def test_assess_flags_suppression_on_outage(monkeypatch):
    monkeypatch.setattr(dg, "load_state", lambda: (dg.STATE_OK, 0))
    saved = {}
    monkeypatch.setattr(dg, "save_state",
                        lambda *a, **k: saved.update({"a": a}))
    status = {f"e{i}": {"empty": i < 20} for i in range(24)}
    res = dg.assess(status)
    assert res["state"] == dg.STATE_OUTAGE and res["suppress"] is True
    assert res["n_failed"] == 20 and res["changed"] is True


# ── 6. the merge tolerance is actually wired into the feature builder ────

def test_feature_builder_passes_tolerance():
    import pathlib
    src = (pathlib.Path(__file__).parent.parent / "indicator"
           / "feature_builder_live.py").read_text(encoding="utf-8")
    assert "tolerance=pd.Timedelta(_CG_TOLERANCE)" in src, (
        "merge_asof must carry a staleness tolerance — without it a "
        "multi-day outage is carried forward as frozen constants")
