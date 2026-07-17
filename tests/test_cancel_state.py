"""Tests for the cancel-flow display state machine (A1, 2026-07-17).

classify_state must be a pure 1:1 relabelling of the frozen v1 playbook
classifier (classify_minute) — no new thresholds, no drift. Verified two
ways: hand-built rows for every state, plus a random fuzz asserting the
mapping agrees with classify_minute on arbitrary inputs.
"""
import numpy as np
import pandas as pd
import pytest

from market_data.tasks.cancel_playbook_watcher import (
    STATE_META, classify_minute, classify_state, state_color)


def row(shock=1.0, vshock=1.0, taker_ratio=0.0, ret_1m=0.0,
        skew15=0.0, net15=0.0) -> pd.Series:
    return pd.Series(dict(shock=shock, vshock=vshock,
                          taker_ratio=taker_ratio, ret_1m=ret_1m,
                          skew15=skew15, net15=net15))


class TestSixStates:
    def test_calm_when_gate_closed(self):
        s = classify_state(row(shock=2.9, vshock=50, taker_ratio=0.9,
                               skew15=0.9, net15=0.9))
        assert s["state"] == "calm" and s["direction"] == "NONE"
        assert s["emoji"] == "⚫"

    def test_absorption_sellers_absorbed_up(self):
        s = classify_state(row(shock=5, vshock=5, taker_ratio=-0.5,
                               ret_1m=0.0002))
        assert s["state"] == "absorption"
        assert s["direction"] == "UP" and s["emoji"] == "🟢"

    def test_absorption_buyers_absorbed_down(self):
        s = classify_state(row(shock=5, vshock=5, taker_ratio=+0.5,
                               ret_1m=-0.0002))
        assert s["state"] == "absorption"
        assert s["direction"] == "DOWN" and s["emoji"] == "🔴"

    def test_cascade_is_true_break_minute(self):
        s = classify_state(row(shock=5, vshock=5, taker_ratio=-0.5,
                               ret_1m=-0.002))
        assert s["state"] == "cascade" and s["zh"] == "瀑布中"
        assert s["emoji"] == "⚪"          # 資訊權在量價, 不上色判方向

    def test_vacuum_up_green(self):
        s = classify_state(row(shock=4, vshock=1.2, skew15=0.35, net15=0.40))
        assert s["state"] == "vacuum_up"
        assert s["direction"] == "UP" and s["emoji"] == "🟢"

    def test_vacuum_down_red(self):
        s = classify_state(row(shock=4, vshock=1.2, skew15=-0.35, net15=-0.40))
        assert s["state"] == "vacuum_down"
        assert s["direction"] == "DOWN" and s["emoji"] == "🔴"

    def test_rotation_gross_high_net_zero(self):
        s = classify_state(row(shock=4, vshock=1.2, skew15=0.05, net15=0.02))
        assert s["state"] == "rotation" and s["zh"] == "換防警戒"
        assert s["emoji"] == "⚪" and s["direction"] == "NONE"

    def test_surge_residual_gate_only(self):
        # gate fired, mid-zone skews — the frozen classifier's gate_only
        s = classify_state(row(shock=4, vshock=1.2, skew15=0.20, net15=0.20))
        assert s["state"] == "surge" and s["zh"] == "爆量未定"
        assert s["emoji"] == "⚪"


class TestNoDriftFromFrozenClassifier:
    """classify_state must agree with classify_minute on every input."""

    EXPECTED = {None: "calm", "absorption": "absorption",
                "true_break": "cascade", "two_sided": "rotation",
                "gate_only": "surge"}

    @pytest.mark.parametrize("seed", range(5))
    def test_fuzz_mapping_agrees(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(400):
            r = row(shock=rng.uniform(0, 8), vshock=rng.uniform(0, 8),
                    taker_ratio=rng.uniform(-1, 1),
                    ret_1m=rng.uniform(-0.003, 0.003),
                    skew15=rng.uniform(-1, 1), net15=rng.uniform(-1, 1))
            res = classify_minute(r)
            s = classify_state(r)
            if res is None:
                assert s["state"] == "calm"
                continue
            playbook, direction = res
            if playbook == "vacuum":
                assert s["state"] == ("vacuum_up" if direction == "UP"
                                      else "vacuum_down")
            else:
                assert s["state"] == self.EXPECTED[playbook]
            assert s["direction"] == direction

    def test_nan_features_degrade_to_calm_or_surge(self):
        # NaN skews at a gate minute must not crash (classify_minute → gate_only)
        s = classify_state(row(shock=4, vshock=np.nan, taker_ratio=np.nan,
                               ret_1m=np.nan, skew15=np.nan, net15=np.nan))
        assert s["state"] in ("calm", "surge")

    def test_every_state_has_meta(self):
        for key, (zh, emoji) in STATE_META.items():
            assert zh
            if key != "absorption":     # absorption is direction-coloured
                assert emoji


class TestStateColor:
    """Ribbon colours (A2): one colour language across every chart surface."""

    def test_calm_draws_nothing(self):
        assert state_color({"state": "calm", "direction": "NONE"}) is None

    def test_vacuum_green_red(self):
        assert state_color({"state": "vacuum_up", "direction": "UP"}) == "#26a269"
        assert state_color({"state": "vacuum_down", "direction": "DOWN"}) == "#e01b24"

    def test_absorption_direction_coloured(self):
        assert state_color({"state": "absorption", "direction": "UP"}) == "#26a269"
        assert state_color({"state": "absorption", "direction": "DOWN"}) == "#e01b24"

    def test_no_direction_states_gray(self):
        for s in ("rotation", "surge"):
            assert state_color({"state": s, "direction": "NONE"}) == "#8a919c"
        assert state_color({"state": "cascade", "direction": "DOWN"}) == "#c0c6cf"
