"""Unit tests for the conviction-decay exit (2026-07-24).

See research/conviction_decay_exit.py for the backtest validation this
mechanism is based on, and indicator/okx/executor.py._manage_position for
the implementation. These tests exercise the live code path directly
(same _mk_executor/_open_pos fixtures as test_okx_executor_trading.py) —
not a re-simulation — so a bug in the actual production logic would show
up here, not just in the standalone research script.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.types import (
    AlgoOrderResult, Balance, CancelResult, OrderResult,
    ReconciliationResult, ReconciliationVerdict,
)


def _mk_klines(n: int = 40, start_price: float = 75000.0, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.003, n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    idx = pd.date_range("2026-05-25", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(500, 2000, n)}, index=idx)


def _mk_cfg(**overrides) -> OkxConfig:
    base = OkxConfig(api_key="k", api_secret="s", passphrase="p",
                     telegram_critical_chat_id="critical-chat",
                     initial_capital_usd=10000.0, contract_size_base=0.01,
                     is_simulated=1, leverage=10)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _mk_executor(cfg=None):
    cfg = cfg or _mk_cfg()
    client = MagicMock()
    client.submit_market_order.return_value = OrderResult(
        cl_ord_id="v7-x", ord_id="okx-ord-1", status="submitted")
    client.submit_algo_stop.return_value = AlgoOrderResult(
        algo_cl_ord_id="v7a-x", algo_id="okx-algo-1", status="submitted")
    client.cancel_algo_stop.return_value = CancelResult(algo_id="x", status="ok")
    client.amend_algo_stop.return_value = MagicMock(status="ok")
    client.get_balance.return_value = Balance(total_eq_usd=10000.0, available_usd=10000.0)
    client.get_order.return_value = None

    store = MagicMock()
    store.get_latest_balance.return_value = {"total_eq_usd": 10000.0, "available_usd": 10000.0}
    store.insert_open_position.return_value = 42
    store.get_open_position.return_value = None
    store.get_all_open_positions.return_value = []

    recon = MagicMock()
    recon.reconcile_cycle.return_value = ReconciliationResult(
        verdict=ReconciliationVerdict.CONSISTENT, detail={})

    exe = V7OkxExecutor(client=client, store=store, reconciler=recon, cfg=cfg)
    return exe, client, store, recon


def _open_pos(side="LONG", *, decay_streak_count=0, shadow_decay_streak_count=0,
             entry_price=75000.0, stop_dist=150.0, trail_extreme=None):
    from datetime import datetime
    return {
        "id": 1, "entry_time": datetime(2026, 5, 27, 0, 0, 0), "direction": side,
        "entry_price": entry_price, "atr_at_entry": 50.0, "stop_dist": stop_dist,
        "trail_extreme": trail_extreme or entry_price,
        "current_stop": (entry_price - stop_dist if side == "LONG" else entry_price + stop_dist),
        "size_contracts": 5, "size_frac": 0.5, "notional_usd": 50.0,
        "equity_before": 100.0, "stop_algo_id": "okx-algo-1",
        "entry_cl_ord_id": "v7-abc", "decay_streak_count": decay_streak_count,
        "shadow_decay_streak_count": shadow_decay_streak_count,
    }


class TestConvictionDecayDisabledByDefault:
    """conviction_decay_bars=0 (the OkxConfig default) must reproduce the
    exact pre-2026-07-24 behavior — no streak tracking, no new exit path."""

    def test_disagreeing_pred_ret_does_not_close_when_disabled(self):
        exe, client, store, _ = _mk_executor()  # default cfg: conviction_decay_bars=0
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]  # no trail advance
        pos = _open_pos(side="LONG")
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.01)  # strongly disagrees
        assert result.action == "hold"
        store.update_decay_streak.assert_not_called()

    def test_omitting_pred_ret_defaults_safely(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        pos = _open_pos(side="LONG")
        # No pred_ret kwarg at all — must not raise.
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL")
        assert result.action == "hold"


class TestConvictionDecayEnabled:
    def test_first_disagreeing_bar_increments_streak_no_exit(self):
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", decay_streak_count=0)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)  # LONG + negative pred = disagreeing
        assert result.action == "hold"
        store.update_decay_streak.assert_called_once_with(position_id=1, streak=1)

    def test_second_consecutive_disagreeing_bar_triggers_exit(self):
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        klines = _mk_klines()
        pos = _open_pos(side="LONG", decay_streak_count=1)  # already 1 from a prior cycle
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.action == "close"
        assert result.detail["exit_reason"] == "conviction_decay"
        # Position is closing — no need to persist a streak for a dead position.
        store.update_decay_streak.assert_not_called()

    def test_agreeing_bar_resets_streak_to_zero(self):
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", decay_streak_count=1)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=+0.001)  # LONG + positive pred = agreeing
        assert result.action == "hold"
        store.update_decay_streak.assert_called_once_with(position_id=1, streak=0)

    def test_short_side_mirrors_long(self):
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        klines = _mk_klines()
        pos = _open_pos(side="SHORT", decay_streak_count=1,
                        entry_price=75000.0, stop_dist=150.0, trail_extreme=75000.0)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=+0.001)  # SHORT + positive pred = disagreeing
        assert result.action == "close"
        assert result.detail["exit_reason"] == "conviction_decay"

    def test_takes_priority_over_opp_signal_when_enabled(self):
        """A full opposite-tier reclassification (signal_direction=DOWN for a
        LONG) would normally be opp_signal — but conviction_decay's looser
        trigger fires first once the streak requirement is already met."""
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        klines = _mk_klines()
        pos = _open_pos(side="LONG", decay_streak_count=1)
        result = exe._manage_position(pos, klines=klines, signal_direction="DOWN",
                                      pred_ret=-0.001)
        assert result.detail["exit_reason"] == "conviction_decay"  # not opp_signal

    def test_time_cap_still_takes_priority_over_conviction_decay(self):
        from datetime import datetime, timedelta
        exe, client, store, _ = _mk_executor(
            cfg=_mk_cfg(conviction_decay_bars=2, time_cap_hours=72))
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", decay_streak_count=1)
        pos["entry_time"] = bar_ts_naive - timedelta(hours=80)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.detail["exit_reason"] == "time_cap"


class TestConvictionDecayDbFailureIsNonFatal:
    def test_update_decay_streak_exception_does_not_crash_cycle(self):
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=2))
        store.update_decay_streak.side_effect = Exception("db down")
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", decay_streak_count=0)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.action == "hold"  # did not raise


class TestShadowMode:
    """Shadow computation must ALWAYS run (regardless of conviction_decay_bars,
    including the default-disabled 0) and must NEVER be able to close a real
    position — it only logs + persists shadow_decay_streak_count."""

    def test_shadow_runs_even_when_feature_disabled(self):
        exe, client, store, _ = _mk_executor()  # default: conviction_decay_bars=0
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", shadow_decay_streak_count=0)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)  # disagreeing
        assert result.action == "hold"
        store.update_shadow_decay_streak.assert_called_once_with(position_id=1, streak=1)
        # Real streak untouched — feature is off.
        store.update_decay_streak.assert_not_called()

    def test_shadow_reaching_threshold_does_not_close_position(self):
        exe, client, store, _ = _mk_executor()  # disabled
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        # shadow_decay_streak_count=1 -> this disagreeing bar makes it 2,
        # meeting SHADOW_CONVICTION_DECAY_BARS — must still just log, not close.
        pos = _open_pos(side="LONG", shadow_decay_streak_count=1)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.action == "hold"  # NEVER close from shadow alone
        client.submit_market_order.assert_not_called()
        store.update_shadow_decay_streak.assert_called_once_with(position_id=1, streak=2)

    def test_shadow_resets_on_agreement(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", shadow_decay_streak_count=1)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=+0.001)  # agreeing
        assert result.action == "hold"
        store.update_shadow_decay_streak.assert_called_once_with(position_id=1, streak=0)

    def test_shadow_also_runs_when_real_feature_enabled(self):
        """Shadow keeps testing the validated SHADOW_CONVICTION_DECAY_BARS=2
        constant independently, even if the live switch is set to some other
        value for a real A/B — the two are deliberately decoupled."""
        exe, client, store, _ = _mk_executor(cfg=_mk_cfg(conviction_decay_bars=4))
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = klines["close"].iloc[-1]
        pos = _open_pos(side="LONG", decay_streak_count=1, shadow_decay_streak_count=1)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.action == "hold"  # real streak only at 2/4, no exit
        store.update_decay_streak.assert_called_once_with(position_id=1, streak=2)
        store.update_shadow_decay_streak.assert_called_once_with(position_id=1, streak=2)

    def test_shadow_failure_does_not_crash_cycle(self):
        exe, client, store, _ = _mk_executor()
        store.update_shadow_decay_streak.side_effect = Exception("db hiccup")
        klines = _mk_klines()
        pos = _open_pos(side="LONG", shadow_decay_streak_count=0)
        result = exe._manage_position(pos, klines=klines, signal_direction="NEUTRAL",
                                      pred_ret=-0.001)
        assert result.action == "hold"  # did not raise despite shadow failure


class TestFirstConvictionDecayLiveAlert:
    """2026-07-25 go-live (0 shadow samples — see mistake.md): the first
    real conviction_decay close gets a distinct banner so a human notices
    immediately, without blocking/delaying the exit itself (unlike the
    entry-side ApprovalGate, which CAN block since no capital is at risk
    while pending)."""

    def test_first_ever_close_gets_banner(self):
        exe, client, store, _ = _mk_executor()
        store.count_closed_by_exit_reason.return_value = 0
        klines = _mk_klines()
        bar_ts = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            exe._close_position(pos, exit_price=76500.0,
                                exit_reason="conviction_decay", bar_ts=bar_ts)
        store.count_closed_by_exit_reason.assert_called_once_with("conviction_decay")
        sent_msg = tg.call_args.args[1]
        assert "FIRST LIVE conviction_decay EXIT" in sent_msg

    def test_second_close_no_banner(self):
        exe, client, store, _ = _mk_executor()
        store.count_closed_by_exit_reason.return_value = 3
        klines = _mk_klines()
        bar_ts = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            exe._close_position(pos, exit_price=76500.0,
                                exit_reason="conviction_decay", bar_ts=bar_ts)
        sent_msg = tg.call_args.args[1]
        assert "FIRST LIVE" not in sent_msg

    def test_non_conviction_decay_exit_never_checks(self):
        """opp_signal/trail_stop/time_cap exits must not even query the
        count — the banner is conviction_decay-specific."""
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        bar_ts = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._close_position(pos, exit_price=76500.0,
                                exit_reason="opp_signal", bar_ts=bar_ts)
        store.count_closed_by_exit_reason.assert_not_called()

    def test_db_failure_falls_back_to_plain_message(self):
        exe, client, store, _ = _mk_executor()
        store.count_closed_by_exit_reason.side_effect = Exception("db down")
        klines = _mk_klines()
        bar_ts = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            result = exe._close_position(pos, exit_price=76500.0,
                                         exit_reason="conviction_decay", bar_ts=bar_ts)
        # Did not raise, still sent the (unbannered) alert.
        assert result.action == "close"
        tg.assert_called_once()
