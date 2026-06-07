"""Unit tests for V7OkxExecutor._open_position and _manage_position.

These exercise the trading-logic mirror against v7_paper_executor:
ATR sizing, algo-stop B4 latency, manual exits (time_cap, opp_signal),
trail amend.  All OKX side effects mocked.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.types import (
    AlgoOrderResult,
    Balance,
    CancelResult,
    OrderResult,
    ReconciliationResult,
    ReconciliationVerdict,
    Side,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _mk_klines(n: int = 40, start_price: float = 75000.0,
                seed: int = 1) -> pd.DataFrame:
    """Synthetic 1h klines.  ATR will be non-zero."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.003, n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    idx = pd.date_range("2026-05-25", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(500, 2000, n),
    }, index=idx)


def _mk_cfg(**overrides) -> OkxConfig:
    # Default tests use $10k so size_contracts > 0 cleanly.  The real
    # Stage 3 ceiling is $100 + 10x leverage (informed override 2026-05-28).
    base = OkxConfig(
        api_key="k", api_secret="s", passphrase="p",
        telegram_critical_chat_id="critical-chat",
        initial_capital_usd=10000.0,
        contract_size_base=0.01,
        is_simulated=1,
        leverage=10,
    )
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
    client.cancel_algo_stop.return_value = CancelResult(algo_id="x",
                                                         status="ok")
    client.amend_algo_stop.return_value = MagicMock(status="ok")
    client.get_balance.return_value = Balance(total_eq_usd=10000.0,
                                              available_usd=10000.0)

    store = MagicMock()
    store.get_latest_balance.return_value = {"total_eq_usd": 10000.0,
                                              "available_usd": 10000.0}
    store.insert_open_position.return_value = 42
    store.get_open_position.return_value = None
    store.get_all_open_positions.return_value = []

    recon = MagicMock()
    recon.reconcile_cycle.return_value = ReconciliationResult(
        verdict=ReconciliationVerdict.CONSISTENT, detail={})

    exe = V7OkxExecutor(client=client, store=store, reconciler=recon,
                        cfg=cfg)
    return exe, client, store, recon


# ── _open_position ───────────────────────────────────────────────────


class TestOpenPosition:
    def test_insufficient_klines_returns_none(self):
        exe, client, store, _ = _mk_executor()
        result = exe._open_position(klines=_mk_klines(n=5),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "insufficient_klines"
        client.submit_market_order.assert_not_called()

    def test_happy_path_long_entry(self):
        exe, client, store, _ = _mk_executor()
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as tg:
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "open"
        assert result.detail["side"] == "LONG"
        # Submitted a BUY market order
        order_kwargs = client.submit_market_order.call_args.kwargs
        assert order_kwargs["side"] == Side.BUY
        # Submitted an algo stop on SELL side (closes the long)
        algo_kwargs = client.submit_algo_stop.call_args.kwargs
        assert algo_kwargs["side"] == Side.SELL
        # Algo trigger is below entry by 3xATR
        assert algo_kwargs["trigger_px"] < result.detail["entry_price"]
        # DB write happened
        store.insert_open_position.assert_called_once()
        ins_kwargs = store.insert_open_position.call_args.kwargs
        assert ins_kwargs["direction"] == "LONG"
        assert ins_kwargs["entry_tier"] == "Strong"
        # OKX IDs mapped back
        store.set_position_okx_ids.assert_called_once()
        # Entry alert sent
        tg.assert_called_once()

    def test_happy_path_short_entry(self):
        exe, client, store, _ = _mk_executor()
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="DOWN",
                                         signal_strength="Moderate",
                                         model_version="v1")
        assert result.action == "open"
        assert result.detail["side"] == "SHORT"
        # SHORT: entry side SELL, algo stop side BUY, trigger above entry
        assert client.submit_market_order.call_args.kwargs["side"] == Side.SELL
        algo_kwargs = client.submit_algo_stop.call_args.kwargs
        assert algo_kwargs["side"] == Side.BUY
        assert algo_kwargs["trigger_px"] > result.detail["entry_price"]

    def test_entry_rejected_no_algo_submitted(self):
        exe, client, store, _ = _mk_executor()
        client.submit_market_order.return_value = OrderResult(
            cl_ord_id="x", status="rejected", error="okx_code_51008")
        result = exe._open_position(klines=_mk_klines(),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "entry_rejected"
        client.submit_algo_stop.assert_not_called()
        store.insert_open_position.assert_not_called()

    def test_below_min_lot_skips(self):
        # Make equity so small that size_contracts rounds to 0
        cfg = _mk_cfg(initial_capital_usd=0.01)
        exe, client, store, _ = _mk_executor(cfg=cfg)
        store.get_latest_balance.return_value = {"total_eq_usd": 0.01,
                                                  "available_usd": 0.01}
        result = exe._open_position(klines=_mk_klines(),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "below_min_lot"
        client.submit_market_order.assert_not_called()

    def test_fractional_sizing_small_account(self):
        # B sizing (2026-06-06): $89 account -> notional ~2x equity ->
        # FRACTIONAL contracts (~0.24), NOT forced to 1 whole contract
        # (the 2026-06-05 over-leverage bug) and NOT 0.
        cfg = _mk_cfg(initial_capital_usd=89.0)
        exe, client, store, _ = _mk_executor(cfg=cfg)
        store.get_latest_balance.return_value = {"total_eq_usd": 89.0,
                                                  "available_usd": 89.0}
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "open"
        ins = store.insert_open_position.call_args.kwargs
        sz = float(ins["size_contracts"])
        # fractional, between 0 and 1 (whole-contract floor is gone)
        assert 0.0 < sz < 1.0
        # snapped to the 0.01 lot step
        assert abs(sz * 100 - round(sz * 100)) < 1e-6
        # notional ~ 2x equity, effective leverage ~2x (NOT ~7x)
        assert abs(float(ins["notional_usd"]) - 2 * 89.0) < 5.0
        assert 1.8 < float(ins["notional_usd"]) / 89.0 < 2.2
        # the order actually submitted the fractional size
        assert float(client.submit_market_order.call_args.kwargs["sz"]) == sz

    def test_b4_latency_violation_force_closes(self):
        # If algo stop submission raises, we have no stop → force close
        exe, client, store, _ = _mk_executor()
        client.submit_algo_stop.side_effect = RuntimeError("network")
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "b4_latency_violation"
        # Emergency close issued (2 market orders: entry + emergency close)
        assert client.submit_market_order.call_count == 2
        store.insert_open_position.assert_not_called()

    def test_algo_rejected_treated_as_no_stop(self):
        exe, client, store, _ = _mk_executor()
        client.submit_algo_stop.return_value = AlgoOrderResult(
            algo_cl_ord_id="x", status="rejected", error="okx_code_51400")
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        # No live stop → B4 violation → force close
        assert result.action == "none"
        assert result.detail["reason"] == "b4_latency_violation"

    def test_db_insert_duplicate_returns_none(self):
        exe, client, store, _ = _mk_executor()
        store.insert_open_position.return_value = None  # dup
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "duplicate_cl_ord_id"


# ── _manage_position ─────────────────────────────────────────────────


def _open_pos(side="LONG", *, entry_time=None, entry_price=75000.0,
              stop_dist=150.0, trail_extreme=None, size=5):
    if entry_time is None:
        entry_time = datetime(2026, 5, 27, 0, 0, 0)  # naive UTC
    return {
        "id": 1,
        "entry_time": entry_time,
        "direction": side,
        "entry_price": entry_price,
        "atr_at_entry": 50.0,
        "stop_dist": stop_dist,
        "trail_extreme": trail_extreme or entry_price,
        "current_stop": (entry_price - stop_dist if side == "LONG"
                         else entry_price + stop_dist),
        "size_contracts": size,
        "size_frac": 0.5,
        "notional_usd": 50.0,
        "equity_before": 100.0,
        "stop_algo_id": "okx-algo-1",
        "entry_cl_ord_id": "v7-abc",
    }


class TestManagePosition:
    def test_no_klines_holds(self):
        exe, client, store, _ = _mk_executor()
        result = exe._manage_position(_open_pos(), klines=pd.DataFrame(),
                                       signal_direction="NEUTRAL")
        assert result.action == "hold"
        assert result.detail["reason"] == "no_klines"

    def test_long_trail_advance_amends_algo(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        # Force this bar's high to be way above prev_extreme to trigger amend
        pos = _open_pos(side="LONG", entry_price=75000.0, stop_dist=150.0,
                        trail_extreme=75000.0)
        # bar_high will be > 75000 from the kline series
        klines.loc[klines.index[-1], "high"] = 76000.0
        result = exe._manage_position(pos, klines=klines,
                                       signal_direction="NEUTRAL")
        assert result.action == "hold"
        client.amend_algo_stop.assert_called_once()
        amend_kwargs = client.amend_algo_stop.call_args.kwargs
        assert amend_kwargs["algo_id"] == "okx-algo-1"
        # New trigger = new_extreme(76000) - stop_dist(150) = 75850
        assert amend_kwargs["new_trigger_px"] == pytest.approx(75850.0)
        store.update_trail.assert_called_once()

    def test_long_no_new_high_no_amend(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        # Force bar_high below prev_extreme — no ratchet
        pos = _open_pos(side="LONG", entry_price=75000.0,
                        trail_extreme=80000.0)
        klines.loc[klines.index[-1], "high"] = 79000.0
        result = exe._manage_position(pos, klines=klines,
                                       signal_direction="NEUTRAL")
        assert result.action == "hold"
        client.amend_algo_stop.assert_not_called()
        store.update_trail.assert_not_called()

    def test_short_trail_advance_amends_algo(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        pos = _open_pos(side="SHORT", entry_price=75000.0, stop_dist=150.0,
                        trail_extreme=75000.0)
        klines.loc[klines.index[-1], "low"] = 74000.0
        result = exe._manage_position(pos, klines=klines,
                                       signal_direction="NEUTRAL")
        assert result.action == "hold"
        amend_kwargs = client.amend_algo_stop.call_args.kwargs
        # SHORT: new_extreme=74000, new_stop=74000+150=74150
        assert amend_kwargs["new_trigger_px"] == pytest.approx(74150.0)

    def test_time_cap_triggers_close(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        # bar_ts in fixture = klines.index[-1] (UTC).  Entry 80h before.
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        entry = bar_ts_naive - timedelta(hours=80)
        pos = _open_pos(side="LONG", entry_time=entry)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._manage_position(pos, klines=klines,
                                           signal_direction="NEUTRAL")
        assert result.action == "close"
        assert result.detail["exit_reason"] == "time_cap"
        client.cancel_algo_stop.assert_called_once_with(algo_id="okx-algo-1")
        client.submit_market_order.assert_called_once()
        assert (client.submit_market_order.call_args.kwargs["side"]
                == Side.SELL)
        store.close_position.assert_called_once()

    def test_opp_signal_DOWN_closes_LONG(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        # Recent entry, not time-capped
        entry = bar_ts_naive - timedelta(hours=2)
        pos = _open_pos(side="LONG", entry_time=entry)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._manage_position(pos, klines=klines,
                                           signal_direction="DOWN")
        assert result.action == "close"
        assert result.detail["exit_reason"] == "opp_signal"

    def test_opp_signal_UP_closes_SHORT(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        entry = bar_ts_naive - timedelta(hours=2)
        pos = _open_pos(side="SHORT", entry_time=entry)
        with patch("indicator.okx.executor.send_critical",
                   return_value=True):
            result = exe._manage_position(pos, klines=klines,
                                           signal_direction="UP")
        assert result.action == "close"
        assert result.detail["exit_reason"] == "opp_signal"
        assert (client.submit_market_order.call_args.kwargs["side"]
                == Side.BUY)

    def test_amend_failure_does_not_halt(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        klines.loc[klines.index[-1], "high"] = 76000.0
        pos = _open_pos(side="LONG", entry_price=75000.0,
                        trail_extreme=75000.0)
        client.amend_algo_stop.side_effect = RuntimeError("network")
        # Must not raise
        result = exe._manage_position(pos, klines=klines,
                                       signal_direction="NEUTRAL")
        assert result.action == "hold"


# ── close_position P&L math ──────────────────────────────────────────


class TestApprovalGateRouting:
    """_open_position behavior when an ApprovalGate is attached."""

    def test_manual_mode_creates_approval_no_submit(self):
        exe, client, store, _ = _mk_executor()
        gate = MagicMock()
        gate.is_auto_mode.return_value = False
        gate.request_approval.return_value = 77
        exe._approval = gate

        result = exe._open_position(klines=_mk_klines(),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "pending_approval"
        assert result.detail["approval_id"] == 77
        # No order submitted
        client.submit_market_order.assert_not_called()
        # Intent passed to gate
        intent = gate.request_approval.call_args.args[0]
        assert intent.direction == "LONG"
        assert intent.tier == "Strong"

    def test_auto_mode_executes_directly(self):
        exe, client, store, _ = _mk_executor()
        gate = MagicMock()
        gate.is_auto_mode.return_value = True
        exe._approval = gate
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "open"
        client.submit_market_order.assert_called_once()
        # No approval requested
        gate.request_approval.assert_not_called()

    def test_no_gate_executes_directly(self):
        # Backward-compat: existing tests pass approval_gate=None
        exe, client, store, _ = _mk_executor()
        assert exe._approval is None
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                         signal_direction="UP",
                                         signal_strength="Strong",
                                         model_version="v1")
        assert result.action == "open"

    def test_approval_request_failure_returns_none(self):
        exe, client, store, _ = _mk_executor()
        gate = MagicMock()
        gate.is_auto_mode.return_value = False
        gate.request_approval.return_value = None   # DB or TG failure
        exe._approval = gate
        result = exe._open_position(klines=_mk_klines(),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "approval_request_failed"
        client.submit_market_order.assert_not_called()

    def test_below_min_lot_returns_before_approval(self):
        # If intent is invalid (too small to trade), don't waste an approval
        cfg = _mk_cfg(initial_capital_usd=0.01)
        exe, client, store, _ = _mk_executor(cfg=cfg)
        store.get_latest_balance.return_value = {"total_eq_usd": 0.01,
                                                  "available_usd": 0.01}
        gate = MagicMock()
        gate.is_auto_mode.return_value = False
        exe._approval = gate
        result = exe._open_position(klines=_mk_klines(),
                                     signal_direction="UP",
                                     signal_strength="Strong",
                                     model_version="v1")
        assert result.action == "none"
        assert result.detail["reason"] == "below_min_lot"
        gate.request_approval.assert_not_called()


class TestExecuteApprovedIntent:
    """execute_approved_intent: called by webhook after /yes."""

    def test_executes_and_marks_approval(self):
        from indicator.okx.approval import TradeIntent
        exe, client, store, _ = _mk_executor()
        gate = MagicMock()
        exe._approval = gate

        intent = TradeIntent(
            direction="LONG", tier="Strong",
            entry_price=75000.0, stop_price=74550.0,
            atr=150.0, stop_dist=450.0,
            size_contracts=5, size_frac=0.5,
            notional_usd=180.0, equity_before=100.0,
            bar_ts_iso="2026-05-28T12:00:00",
            model_version="v1",
        )
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe.execute_approved_intent(intent, approval_id=77)
        assert result.action == "open"
        # Approval marked executed with the position id
        gate.mark_executed.assert_called_once()
        kwargs = gate.mark_executed.call_args.kwargs
        assert kwargs["approval_id"] == 77
        assert kwargs["position_id"] == 42   # store fixture returns 42

    def test_no_approval_id_does_not_mark(self):
        from indicator.okx.approval import TradeIntent
        exe, client, store, _ = _mk_executor()
        gate = MagicMock()
        exe._approval = gate
        intent = TradeIntent(
            direction="LONG", tier="Strong",
            entry_price=75000.0, stop_price=74550.0,
            atr=150.0, stop_dist=450.0,
            size_contracts=5, size_frac=0.5,
            notional_usd=180.0, equity_before=100.0,
            bar_ts_iso="2026-05-28T12:00:00",
        )
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe.execute_approved_intent(intent, approval_id=None)
        gate.mark_executed.assert_not_called()


class TestLongShortModePosSide:
    """In long_short_mode, every order needs posSide + algo stops use reduceOnly."""

    def test_long_entry_passes_pos_side_long(self):
        cfg = _mk_cfg(pos_mode="long_short_mode")
        exe, client, store, _ = _mk_executor(cfg=cfg)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._open_position(klines=_mk_klines(),
                                signal_direction="UP",
                                signal_strength="Strong",
                                model_version="v1")
        entry_kwargs = client.submit_market_order.call_args.kwargs
        assert entry_kwargs["pos_side"] == "long"
        # Algo stop matches the LONG position and is reduce_only
        algo_kwargs = client.submit_algo_stop.call_args.kwargs
        assert algo_kwargs["pos_side"] == "long"
        assert algo_kwargs["reduce_only"] is True

    def test_short_entry_passes_pos_side_short(self):
        cfg = _mk_cfg(pos_mode="long_short_mode")
        exe, client, store, _ = _mk_executor(cfg=cfg)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._open_position(klines=_mk_klines(),
                                signal_direction="DOWN",
                                signal_strength="Moderate",
                                model_version="v1")
        assert (client.submit_market_order.call_args.kwargs["pos_side"]
                == "short")
        algo_kwargs = client.submit_algo_stop.call_args.kwargs
        assert algo_kwargs["pos_side"] == "short"
        assert algo_kwargs["reduce_only"] is True

    def test_net_mode_omits_pos_side(self):
        cfg = _mk_cfg(pos_mode="net_mode")
        exe, client, store, _ = _mk_executor(cfg=cfg)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._open_position(klines=_mk_klines(),
                                signal_direction="UP",
                                signal_strength="Strong",
                                model_version="v1")
        # pos_side kwarg is None (gets omitted from request body by REST)
        assert client.submit_market_order.call_args.kwargs["pos_side"] is None
        assert client.submit_algo_stop.call_args.kwargs["pos_side"] is None

    def test_manual_close_carries_pos_side_and_reduce_only(self):
        cfg = _mk_cfg(pos_mode="long_short_mode")
        exe, client, store, _ = _mk_executor(cfg=cfg)
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        from datetime import timedelta
        # Hours-old LONG → time_cap triggers close
        pos = _open_pos(side="LONG",
                         entry_time=bar_ts_naive - timedelta(hours=80))
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._manage_position(pos, klines=klines,
                                  signal_direction="NEUTRAL")
        # The close order is the only one this cycle
        close_kwargs = client.submit_market_order.call_args.kwargs
        assert close_kwargs["pos_side"] == "long"
        assert close_kwargs["reduce_only"] is True

    def test_force_close_all_carries_pos_side_and_reduce_only(self):
        cfg = _mk_cfg(pos_mode="long_short_mode")
        exe, client, store, _ = _mk_executor(cfg=cfg)
        store.get_all_open_positions.return_value = [{
            "id": 1, "direction": "SHORT", "size_contracts": 3,
            "entry_price": 75000.0, "equity_before": 100.0,
            "stop_algo_id": "algo-x",
        }]
        exe._force_close_all()
        close_kwargs = client.submit_market_order.call_args.kwargs
        assert close_kwargs["pos_side"] == "short"
        assert close_kwargs["reduce_only"] is True


class TestClosePnL:
    def test_long_profit(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="LONG", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._close_position(pos, exit_price=76500.0,
                                          exit_reason="opp_signal",
                                          bar_ts=bar_ts_naive)
        # gross = 76500/75000 - 1 = +2.0%, net = +2.0% - 0.08% = +1.92%
        assert result.detail["gross_pct"] == pytest.approx(0.02)
        assert result.detail["net_pct"] == pytest.approx(0.02 - 0.0008)
        # size_frac=0.5, equity_before=100 → equity_after = 100*(1+0.5*0.0192)=100.96
        assert result.detail["equity_after"] == pytest.approx(
            100.0 * (1 + 0.5 * (0.02 - 0.0008))
        )

    def test_short_profit(self):
        exe, client, store, _ = _mk_executor()
        klines = _mk_klines()
        bar_ts_naive = klines.index[-1].tz_convert("UTC").tz_localize(None)
        pos = _open_pos(side="SHORT", entry_price=75000.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._close_position(pos, exit_price=73500.0,
                                          exit_reason="opp_signal",
                                          bar_ts=bar_ts_naive)
        # SHORT: gross = -((73500/75000)-1) = +2.0%
        assert result.detail["gross_pct"] == pytest.approx(0.02)
