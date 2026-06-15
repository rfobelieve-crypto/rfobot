"""Unit tests for indicator/okx/executor.py.

We mock OkxClient + OkxStateStore + PositionReconciler entirely.  Goal
is to verify the executor's state-machine transitions and integration
of kill_checks + alerter + WS callbacks — not the underlying parts
(already tested separately).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.types import (
    Balance,
    BalanceEvent,
    CancelResult,
    ExecutorStatus,
    OrderEvent,
    OrderResult,
    Position,
    ReconciliationResult,
    ReconciliationVerdict,
)


def _mk_cfg(**overrides) -> OkxConfig:
    base = OkxConfig(
        api_key="k", api_secret="s", passphrase="p",
        telegram_critical_chat_id="critical-chat",
        initial_capital_usd=100.0,
        is_simulated=1,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _consistent_recon() -> ReconciliationResult:
    return ReconciliationResult(
        verdict=ReconciliationVerdict.CONSISTENT,
        detail={"state": "both_flat"},
    )


def _mk_executor(**kwargs):
    """Build executor with all dependencies mocked.

    Returns (executor, client_mock, store_mock, reconciler_mock).
    """
    cfg = kwargs.get("cfg") or _mk_cfg()
    client = MagicMock()
    # Default to a clean state: all healthy queries
    client.get_account_config.return_value = {
        "code": "0",
        "data": [{"perm": "trade,read"}],
    }
    client.get_server_time.return_value = None  # skip NTP probe
    client.get_balance.return_value = Balance(total_eq_usd=100.0,
                                              available_usd=100.0)
    client.connectivity.return_value = MagicMock(
        public_ws_ok=True, private_ws_ok=True,
        last_public_heartbeat_age_sec=1.0,
        last_private_heartbeat_age_sec=1.0,
        consecutive_reconnect_fails=0,
    )
    client.get_positions.return_value = []
    client.submit_market_order.return_value = OrderResult(
        cl_ord_id="x", ord_id="ox", status="submitted"
    )
    client.cancel_algo_stop.return_value = CancelResult(
        algo_id="a", status="ok"
    )

    store = MagicMock()
    store.get_open_position.return_value = None
    store.get_all_open_positions.return_value = []
    store.get_latest_balance.return_value = None
    store.get_day_start_equity.return_value = None

    recon = MagicMock()
    recon.reconcile_cycle.return_value = _consistent_recon()

    exe = V7OkxExecutor(client=client, store=store,
                        reconciler=recon, cfg=cfg)
    return exe, client, store, recon


# ── start() lifecycle ─────────────────────────────────────────────────


class TestStart:
    def test_clean_start_lands_in_ACTIVE(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        assert exe.get_status() == ExecutorStatus.ACTIVE
        # Cold-start reconciliation was called
        recon.reconcile_cycle.assert_called_once()
        # WS started
        client.start_ws.assert_called_once()
        # All 3 callbacks wired
        assert client.subscribe_orders.called
        assert client.subscribe_positions.called
        assert client.subscribe_balance.called

    def test_withdraw_perm_blocks_with_E4(self):
        exe, client, store, recon = _mk_executor()
        client.get_account_config.return_value = {
            "code": "0",
            "data": [{"perm": "trade,read,withdraw"}],
        }
        exe.start()
        assert exe.get_status() == ExecutorStatus.DEMOTED
        # No reconciliation should have been attempted after demote
        recon.reconcile_cycle.assert_not_called()

    def test_missing_perm_field_logs_warning_continues(self):
        # OKX main-account /account/config doesn't always expose `perm`
        exe, client, store, recon = _mk_executor()
        client.get_account_config.return_value = {
            "code": "0", "data": [{"posMode": "net_mode"}],
        }
        exe.start()
        # Best-effort: don't block startup on missing perm field
        assert exe.get_status() == ExecutorStatus.ACTIVE

    def test_ntp_drift_at_start_demotes(self):
        exe, client, store, recon = _mk_executor()
        # 100s drift > demote threshold 30s → C6
        with patch("indicator.okx.executor.time.time", return_value=1000.0):
            client.get_server_time.return_value = 900.0
            exe.start()
        assert exe.get_status() == ExecutorStatus.DEMOTED
        recon.reconcile_cycle.assert_not_called()

    def test_ntp_query_failure_does_not_block_start(self):
        exe, client, store, recon = _mk_executor()
        client.get_server_time.return_value = None
        exe.start()
        assert exe.get_status() == ExecutorStatus.ACTIVE

    def test_reconciliation_mismatch_halts(self):
        exe, client, store, recon = _mk_executor()
        recon.reconcile_cycle.return_value = ReconciliationResult(
            verdict=ReconciliationVerdict.MISMATCH,
            detail={"type": "orphan_exchange"},
        )
        exe.start()
        assert exe.get_status() == ExecutorStatus.HALTED

    def test_balance_snapshot_taken_when_no_day_start_anchor(self):
        exe, client, store, recon = _mk_executor()
        store.get_day_start_equity.return_value = None
        client.get_balance.return_value = Balance(total_eq_usd=100.0,
                                                  available_usd=95.0)
        exe.start()
        store.insert_balance_snapshot.assert_called_once()
        kwargs = store.insert_balance_snapshot.call_args.kwargs
        assert kwargs["total_eq_usd"] == 100.0
        assert kwargs["source"] == "start"


# ── cycle() ──────────────────────────────────────────────────────────


class TestCycle:
    def test_in_INIT_state_returns_none(self):
        exe, client, store, recon = _mk_executor()
        # Don't call start; stays in INIT
        result = exe.cycle(klines=pd.DataFrame(), signal_direction="UP",
                           signal_strength="Strong")
        assert result.action == "none"
        assert result.detail["status"] == "INIT"

    def test_halted_to_active_auto_resolves_kills(self):
        """Recovery path: HALTED → triggers cleared → ACTIVE → kill log
        auto-resolved (M6 milestone enabler)."""
        exe, client, store, recon = _mk_executor()
        exe.start()
        # Force HALTED state directly
        from indicator.okx.types import ExecutorStatus
        exe._status = ExecutorStatus.HALTED
        # Next cycle: clean reconciliation, no triggers → should resume
        recon.reconcile_cycle.return_value = _consistent_recon()
        result = exe.cycle(klines=pd.DataFrame(),
                            signal_direction="NEUTRAL",
                            signal_strength="Weak")
        assert exe.get_status() == ExecutorStatus.ACTIVE
        # resolve_open_kills should have been called
        store.resolve_open_kills.assert_called_once()
        kw = store.resolve_open_kills.call_args.kwargs
        assert "auto" in kw.get("resolution", "").lower()

    def test_kill_trigger_returns_halted(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        # Inject a transient mismatch (orphan_local: WS missed a fill) ->
        # A4 HALT.  Foreign-position mismatch types (size_diff etc.) instead
        # escalate to MANUAL-INTERFERENCE DEMOTE — covered separately.
        recon.reconcile_cycle.return_value = ReconciliationResult(
            verdict=ReconciliationVerdict.MISMATCH,
            detail={"type": "orphan_local"},
        )
        result = exe.cycle(klines=pd.DataFrame(), signal_direction="NEUTRAL",
                           signal_strength="Weak")
        assert result.action == "halted"
        # A4 trigger should appear
        assert "A4" in result.detail["triggers"]

    def test_equity_pulled_from_balance_snapshot(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        store.get_latest_balance.return_value = {"total_eq_usd": 99.0,
                                                  "available_usd": 95.0}
        store.get_day_start_equity.return_value = 100.0
        # Just hitting cycle exercises the equity path
        exe.cycle(klines=pd.DataFrame(), signal_direction="NEUTRAL",
                  signal_strength="Weak")
        # Cycle completed without exception is the test

    def test_local_positions_converted_for_max_pos_check(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        store.get_all_open_positions.return_value = [
            {"id": 1, "direction": "LONG", "size_contracts": 5,
             "entry_price": 75000.0},
        ]
        # cycle exercises _dicts_to_positions
        exe.cycle(klines=pd.DataFrame(), signal_direction="NEUTRAL",
                  signal_strength="Weak")
        # One Position with direction=LONG; check_max_position with
        # max_count=1 → not triggered.  Cycle should not halt.
        assert exe.get_status() == ExecutorStatus.ACTIVE


# ── strong_only_entry gate ───────────────────────────────────────────


class TestStrongOnlyEntry:
    """OKX_STRONG_ONLY_ENTRY gate: Moderate signals must not open a position
    when the flag is on (they crowd out Strong under 1-position occupancy)."""

    def test_moderate_skipped_when_flag_on(self):
        exe, client, store, recon = _mk_executor(
            cfg=_mk_cfg(strong_only_entry=True))
        exe.start()
        result = exe.cycle(klines=pd.DataFrame(), signal_direction="UP",
                           signal_strength="Moderate")
        assert result.action == "none"
        assert result.detail["reason"] == "moderate_skipped_strong_only"
        assert result.detail["tier"] == "Moderate"
        client.submit_market_order.assert_not_called()

    def test_strong_passes_gate_when_flag_on(self):
        exe, client, store, recon = _mk_executor(
            cfg=_mk_cfg(strong_only_entry=True))
        exe.start()
        result = exe.cycle(klines=pd.DataFrame(), signal_direction="UP",
                           signal_strength="Strong")
        # Gate passed → reached _open_position (which bails on empty klines).
        assert result.detail.get("reason") != "moderate_skipped_strong_only"

    def test_moderate_opens_when_flag_off(self):
        exe, client, store, recon = _mk_executor(
            cfg=_mk_cfg(strong_only_entry=False))
        exe.start()
        result = exe.cycle(klines=pd.DataFrame(), signal_direction="UP",
                           signal_strength="Moderate")
        # Flag off (default) → Moderate is NOT gated; reaches _open_position.
        assert result.detail.get("reason") != "moderate_skipped_strong_only"


# ── _force_close_all ─────────────────────────────────────────────────


class TestForceCloseAll:
    def test_flat_account_no_op(self):
        exe, client, store, recon = _mk_executor()
        store.get_all_open_positions.return_value = []
        exe._force_close_all()
        # Nothing to close: no OKX calls
        client.submit_market_order.assert_not_called()
        client.cancel_algo_stop.assert_not_called()

    def test_long_position_closed_with_sell(self):
        exe, client, store, recon = _mk_executor()
        store.get_all_open_positions.return_value = [{
            "id": 1, "direction": "LONG", "size_contracts": 5,
            "entry_price": 75000.0, "equity_before": 100.0,
            "stop_algo_id": "algo-1",
        }]
        exe._force_close_all()
        # Cancel algo first
        client.cancel_algo_stop.assert_called_once_with(algo_id="algo-1")
        # Then market sell to flatten
        client.submit_market_order.assert_called_once()
        call_kwargs = client.submit_market_order.call_args.kwargs
        assert call_kwargs["side"].value == "sell"
        assert call_kwargs["sz"] == 5
        # DB marked DEMOTED
        store.close_position.assert_called_once()
        db_kwargs = store.close_position.call_args.kwargs
        assert db_kwargs["new_status"] == "DEMOTED"
        assert db_kwargs["exit_reason"] == "force_close"

    def test_short_position_closed_with_buy(self):
        exe, client, store, recon = _mk_executor()
        store.get_all_open_positions.return_value = [{
            "id": 2, "direction": "SHORT", "size_contracts": 3,
            "entry_price": 75000.0, "equity_before": 100.0,
            "stop_algo_id": "algo-2",
        }]
        exe._force_close_all()
        assert client.submit_market_order.call_args.kwargs["side"].value == "buy"

    def test_missing_algo_id_skips_cancel(self):
        exe, client, store, recon = _mk_executor()
        store.get_all_open_positions.return_value = [{
            "id": 1, "direction": "LONG", "size_contracts": 5,
            "entry_price": 75000.0, "equity_before": 100.0,
            "stop_algo_id": None,
        }]
        exe._force_close_all()
        client.cancel_algo_stop.assert_not_called()
        # Market close still happens
        client.submit_market_order.assert_called_once()

    def test_cancel_failure_does_not_block_market_close(self):
        exe, client, store, recon = _mk_executor()
        store.get_all_open_positions.return_value = [{
            "id": 1, "direction": "LONG", "size_contracts": 5,
            "entry_price": 75000.0, "equity_before": 100.0,
            "stop_algo_id": "algo-1",
        }]
        client.cancel_algo_stop.side_effect = RuntimeError("network")
        exe._force_close_all()
        # Market close still proceeds despite cancel failing
        client.submit_market_order.assert_called_once()


# ── WS callbacks ─────────────────────────────────────────────────────


class TestWsCallbacks:
    def test_on_balance_persists_snapshot(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        # Inspect what callback was registered
        balance_cb = client.subscribe_balance.call_args.args[0]
        # Reset call history from start()
        store.insert_balance_snapshot.reset_mock()
        # Fire callback with a WS event
        balance_cb(BalanceEvent(total_eq_usd=103.5, available_usd=99.0))
        store.insert_balance_snapshot.assert_called_once_with(
            total_eq_usd=103.5, available_usd=99.0, source="ws",
        )

    def test_on_order_maps_cl_ord_id_to_ord_id(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        order_cb = client.subscribe_orders.call_args.args[0]
        store.get_open_position.return_value = {
            "id": 7, "entry_cl_ord_id": "v7-abc",
        }
        order_cb(OrderEvent(cl_ord_id="v7-abc", ord_id="okx-12345",
                             state="filled"))
        store.set_position_okx_ids.assert_called_once_with(
            position_id=7, entry_ord_id="okx-12345",
        )

    def test_on_order_skips_when_cl_ord_id_mismatches(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        order_cb = client.subscribe_orders.call_args.args[0]
        store.get_open_position.return_value = {
            "id": 7, "entry_cl_ord_id": "v7-different",
        }
        order_cb(OrderEvent(cl_ord_id="v7-abc", ord_id="okx-12345",
                             state="filled"))
        store.set_position_okx_ids.assert_not_called()

    def test_on_order_no_open_position_no_op(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        order_cb = client.subscribe_orders.call_args.args[0]
        store.get_open_position.return_value = None
        order_cb(OrderEvent(cl_ord_id="v7-abc", ord_id="okx-12345",
                             state="filled"))
        store.set_position_okx_ids.assert_not_called()

    def test_on_balance_db_failure_does_not_raise(self):
        exe, client, store, recon = _mk_executor()
        exe.start()
        balance_cb = client.subscribe_balance.call_args.args[0]
        store.insert_balance_snapshot.side_effect = RuntimeError("DB down")
        # Must not propagate
        balance_cb(BalanceEvent(total_eq_usd=100.0, available_usd=100.0))


# ── _alert_critical ──────────────────────────────────────────────────


class TestAlertCritical:
    def test_alert_send_invokes_telegram(self):
        exe, client, store, recon = _mk_executor()
        from indicator.okx.types import KillCheckResult, KillSeverity
        check = KillCheckResult(
            triggered=True, trigger_id="A4",
            severity=KillSeverity.HALT,
            reason="reconciliation mismatch",
            context={"type": "size_diff"},
        )
        with patch("indicator.okx.executor.send_critical",
                   return_value=True) as send_mock:
            exe._alert_critical(check, severity_label="HALT")
        send_mock.assert_called_once()
        call_args = send_mock.call_args
        # First positional arg is chat_id, second is the formatted message
        assert call_args.args[0] == "critical-chat"
        assert "A4" in call_args.args[1]
        assert "reconciliation mismatch" in call_args.args[1]

    def test_alert_swallows_send_failure(self):
        exe, client, store, recon = _mk_executor()
        from indicator.okx.types import KillCheckResult, KillSeverity
        check = KillCheckResult(triggered=True, trigger_id="A4",
                                 severity=KillSeverity.HALT,
                                 reason="r")
        with patch("indicator.okx.executor.send_critical",
                   side_effect=RuntimeError("net")):
            # Must not raise
            exe._alert_critical(check, severity_label="HALT")
