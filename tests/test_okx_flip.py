"""Flip-on-opposite-Strong (2026-07-10, strong_preempt_bt GO).

Verifies the executor's same-cycle flip: an opposite STRONG reading while
holding closes the position (opp_signal, unchanged) AND enters the new
direction in the same cycle. Guards verified: flag off, Moderate opposite,
daily-cap window.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from indicator.okx.types import ExecutorStatus

from tests.test_okx_mock_broker import _mk_cfg, _mk_harness, _mk_klines


def _open_long(exe):
    with patch("indicator.okx.executor.send_critical", return_value=True):
        r = exe._open_position(klines=_mk_klines(), signal_direction="UP",
                               signal_strength="Strong", model_version="v1")
    assert r.action == "open"
    return r


def _cycle(exe, direction, strength):
    with patch("indicator.okx.executor.send_critical", return_value=True):
        return exe.cycle(klines=_mk_klines(), signal_direction=direction,
                         signal_strength=strength)


class TestFlipOnOppositeStrong:
    def _harness(self, **cfg_overrides):
        cfg = _mk_cfg(initial_capital_usd=10000.0, **cfg_overrides)
        exe, client, store, cfg = _mk_harness(cfg=cfg, latest_equity=10000.0,
                                              day_start_equity=10000.0)
        exe._status = ExecutorStatus.ACTIVE
        return exe, client, store, cfg

    def test_opposite_strong_flips_same_cycle(self):
        exe, client, store, cfg = self._harness()  # flip default ON
        _open_long(exe)
        res = _cycle(exe, "DOWN", "Strong")
        assert res.action == "flip"
        assert res.detail["closed"]["exit_reason"] == "opp_signal"
        assert res.detail["open_action"] == "open"
        # store: old row closed as opp_signal, new OPEN row is SHORT
        assert store.closed and store.closed[-1]["exit_reason"] == "opp_signal"
        new_pos = store.get_open_position()
        assert new_pos is not None and new_pos["direction"] == "SHORT"
        # exchange: exactly one position, SHORT
        okx_pos = client.get_positions(inst_id=cfg.inst_id)
        assert len(okx_pos) == 1 and okx_pos[0].direction == "SHORT"

    def test_flag_off_close_only(self):
        exe, client, store, cfg = self._harness(flip_on_opp_strong=False)
        _open_long(exe)
        res = _cycle(exe, "DOWN", "Strong")
        assert res.action == "close"
        assert store.get_open_position() is None
        assert client.get_positions(inst_id=cfg.inst_id) == []

    def test_opposite_moderate_close_only(self):
        exe, client, store, cfg = self._harness()
        _open_long(exe)
        res = _cycle(exe, "DOWN", "Moderate")
        assert res.action == "close"
        assert store.get_open_position() is None
        assert client.get_positions(inst_id=cfg.inst_id) == []

    def test_daily_cap_guard_blocks_flip(self):
        """equity_after already through the daily cap -> guard skips the
        flip (kill checks ran before the close; don't front-run the HALT)."""
        exe, _, store, _ = self._harness()

        class _R:  # minimal CycleResult stand-in (duck-typed detail access)
            detail = {"equity_after": 7900.0}   # -21% vs day start 10000
        assert exe._flip_daily_cap_ok(_R()) is False

        class _R2:
            detail = {"equity_after": 8500.0}   # -15% — inside the cap
        assert exe._flip_daily_cap_ok(_R2()) is True

    def test_daily_cap_guard_fail_closed(self):
        exe, _, store, _ = self._harness()
        store._day_start_equity = None          # lookup gap → skip flip

        class _R:
            detail = {"equity_after": 9000.0}
        assert exe._flip_daily_cap_ok(_R()) is False
