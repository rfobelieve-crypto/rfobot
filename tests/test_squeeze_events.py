"""Tests for research/squeeze_events.py — schema + settlement invariants.

Fixture: fixed-seed synthetic series alternating low-vol compression
segments with strong drift segments, guaranteeing squeeze windows and
breakouts in both directions. No network, no DB, no project data files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.squeeze_events import (
    EntryMode, SqueezeConfig, detect_events,
)

EXPECTED_COLUMNS = {
    "squeeze_start_ts", "squeeze_bars", "breakout_ts", "direction",
    "box_top", "box_bottom", "atr_bo",
    "filled", "fill_ts", "cancel_reason",
    "entry", "sl", "tp1", "tp2", "tp3", "risk",
    "sl_first", "tps_hit", "resolved", "resolved_ts",
    "r_tp1_all", "r_scaleout", "mfe_R", "mae_R", "horizon_R",
}

# Label columns that must be NaN when the event never filled
NAN_WHEN_UNFILLED = ["entry", "sl", "tp1", "tp2", "tp3", "risk",
                     "r_tp1_all", "r_scaleout", "mfe_R", "mae_R",
                     "horizon_R"]


def _synthetic_ohlcv(seed: int = 42, n_cycles: int = 12) -> pd.DataFrame:
    """Seeded random walk with alternating compression / expansion regimes.

    Each cycle: 80 bars of tiny noise (squeeze forms) then 25 bars of
    strong drift (breakout + follow-through), drift direction alternating
    so both long and short events appear.

    Every third cycle is a "runaway" expansion: bars GAP in the drift
    direction (open ≈ prev close + drift) with tiny wicks, so the price
    never returns to the box edge → RETEST-mode orders expire unfilled.
    This makes the filled=False invariants non-vacuous.
    """
    rng = np.random.default_rng(seed)
    opens: list[float] = []
    closes: list[float] = []
    prev = 100.0
    for cyc in range(n_cycles):
        for _ in range(80):                       # compression
            o_ = prev
            c_ = o_ + rng.normal(0, 0.03)
            opens.append(o_)
            closes.append(c_)
            prev = c_
        drift = 0.9 if cyc % 2 == 0 else -0.9     # expansion
        runaway = (cyc % 3 == 2)
        for _ in range(25):
            if runaway:
                o_ = prev + drift                  # gap away from the box
                c_ = o_ + drift * 0.5 + rng.normal(0, 0.05)
            else:
                o_ = prev                          # contiguous bars
                c_ = o_ + drift + rng.normal(0, 0.35)
            opens.append(o_)
            closes.append(c_)
            prev = c_
    o = np.array(opens, dtype=float)
    c = np.array(closes, dtype=float)
    spread = np.abs(rng.normal(0, 0.08, len(c)))
    # runaway gaps must keep tiny wicks or the gap fills anyway
    spread = np.minimum(spread, 0.04)
    h = np.maximum(o, c) + spread
    low = np.minimum(o, c) - spread
    idx = pd.date_range("2025-01-01", periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c},
                        index=idx)


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _synthetic_ohlcv()


@pytest.fixture(scope="module")
def events_retest(ohlcv) -> pd.DataFrame:
    return detect_events(ohlcv, SqueezeConfig())          # default RETEST


@pytest.fixture(scope="module")
def events_close(ohlcv) -> pd.DataFrame:
    return detect_events(ohlcv, SqueezeConfig(entry_mode=EntryMode.CLOSE))


# ── (a) schema ─────────────────────────────────────────────────────────

class TestSchema:
    def test_events_exist(self, events_retest, events_close):
        # The fixture must actually generate events in both modes,
        # otherwise every downstream assertion is vacuous.
        assert len(events_retest) >= 3
        assert len(events_close) >= 3

    def test_schema_complete(self, events_retest, events_close):
        for ev in (events_retest, events_close):
            assert EXPECTED_COLUMNS.issubset(set(ev.columns)), (
                EXPECTED_COLUMNS - set(ev.columns))

    def test_both_directions_sampled(self, events_close):
        dirs = set(events_close["direction"].unique())
        assert dirs == {1, -1}

    def test_fills_and_resolutions_present(self, events_close):
        assert events_close["filled"].any()
        assert events_close["resolved"].any()

    def test_empty_input_returns_empty(self, ohlcv):
        # Too short for warmup → no events, empty frame, no crash
        out = detect_events(ohlcv.iloc[:50], SqueezeConfig())
        assert out.empty


# ── (b) settlement invariants ──────────────────────────────────────────

class TestSettlementInvariants:
    @pytest.fixture(scope="class", params=["retest", "close"])
    def events(self, request, events_retest, events_close):
        return events_retest if request.param == "retest" else events_close

    def test_sl_first_implies_r_tp1_all_minus_one(self, events):
        sl_first = events[events["sl_first"] == True]  # noqa: E712
        assert len(sl_first) > 0, "fixture produced no SL-first events"
        assert (sl_first["r_tp1_all"] == -1.0).all()

    def test_three_tps_implies_resolved(self, events):
        full = events[events["tps_hit"] == 3]
        if len(full):
            assert full["resolved"].all()

    def test_sl_first_implies_mae_at_least_one(self, events):
        sl_first = events[events["sl_first"] == True]  # noqa: E712
        # MAE is updated with the SL-hit bar's extreme before the SL check,
        # so adverse excursion must reach at least the 1R stop distance.
        assert (sl_first["mae_R"] >= 1.0 - 1e-9).all()

    def test_unfilled_events_have_nan_labels(self, events):
        unfilled = events[events["filled"] == False]  # noqa: E712
        if not len(unfilled):
            pytest.skip("fixture produced no unfilled events in this mode")
        for col in NAN_WHEN_UNFILLED:
            assert unfilled[col].isna().all(), f"{col} not NaN for unfilled"
        assert unfilled["sl_first"].isna().all()
        assert (unfilled["tps_hit"] == 0).all()
        assert (unfilled["resolved"] == False).all()  # noqa: E712
        assert unfilled["fill_ts"].isna().all()
        assert unfilled["resolved_ts"].isna().all()

    def test_resolved_implies_terminal_fields(self, events):
        res = events[events["resolved"] == True]  # noqa: E712
        assert len(res) > 0
        assert res["sl_first"].notna().all()
        assert res["resolved_ts"].notna().all()
        assert res["mfe_R"].notna().all()
        assert res["mae_R"].notna().all()

    def test_determinism_same_seed(self, ohlcv):
        a = detect_events(ohlcv, SqueezeConfig())
        b = detect_events(ohlcv, SqueezeConfig())
        pd.testing.assert_frame_equal(a, b)


# ── adapter contract (no data file needed) ─────────────────────────────

class TestAdapterContract:
    def test_unknown_symbol_raises_no_download(self):
        from research.squeeze_events_cli import load_ohlcv
        with pytest.raises(SystemExit, match="does not download"):
            load_ohlcv("ETH-USD", "5m")

    def test_no_live_imports(self):
        # Research-only module: must not touch executor / live paths.
        import research.squeeze_events as mod
        import sys
        assert not any(m.startswith("indicator.okx") for m in sys.modules
                       if mod.__name__ in m)
        src = open(mod.__file__, encoding="utf-8").read()
        for banned in ("indicator.", "BTC_perp_data", "telegram", "okx"):
            assert banned not in src, f"live-path reference: {banned}"
