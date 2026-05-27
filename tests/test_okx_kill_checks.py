"""Unit tests for indicator/okx/kill_checks.py.

Verifies that every safety belt (#2-#7, #9, #10) plus the A1/A2/A3
connectivity triggers actually fire under the conditions documented in
stage2_kill_criteria.md, and stay silent on the happy path.

These tests are the gate before live $100 trading: per CLAUDE.md
"Hard kill switches 必須先驗證能觸發".
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from indicator.okx.kill_checks import (
    check_algo_stop_latency,
    check_api_permissions,
    check_capital_cap,
    check_connectivity,
    check_daily_loss_cap,
    check_max_position,
    check_ntp_drift,
    check_reconciliation,
    check_total_loss_cap,
    run_all_checks,
)
from indicator.okx.types import (
    ConnectivityStatus,
    KillSeverity,
    Position,
    ReconciliationResult,
    ReconciliationVerdict,
)


# ── #2 Capital cap ────────────────────────────────────────────────────


class TestCapitalCap:
    def test_under_cap_does_not_trigger(self):
        result = check_capital_cap(current_equity_usd=100.0,
                                   capital_hard_cap_usd=100.0)
        assert not result.triggered

    def test_at_1_5x_cap_does_not_trigger(self):
        # 1.5x is the threshold (strict >); equal must not fire
        result = check_capital_cap(current_equity_usd=150.0,
                                   capital_hard_cap_usd=100.0)
        assert not result.triggered

    def test_above_1_5x_cap_triggers_halt(self):
        result = check_capital_cap(current_equity_usd=151.0,
                                   capital_hard_cap_usd=100.0)
        assert result.triggered
        assert result.trigger_id == "CAP-2"
        assert result.severity == KillSeverity.HALT
        assert "equity" in result.context


# ── #3 Daily loss cap ─────────────────────────────────────────────────


class TestDailyLossCap:
    def test_under_cap_does_not_trigger(self):
        # 10% loss with 50% cap
        result = check_daily_loss_cap(day_start_equity_usd=100.0,
                                      current_equity_usd=90.0,
                                      daily_loss_cap_pct=-50.0)
        assert not result.triggered

    def test_exactly_at_cap_triggers(self):
        # -50% with -50% cap: comparison is <=, must trigger
        result = check_daily_loss_cap(day_start_equity_usd=100.0,
                                      current_equity_usd=50.0,
                                      daily_loss_cap_pct=-50.0)
        assert result.triggered
        assert result.trigger_id == "CAP-3"
        assert result.severity == KillSeverity.HALT

    def test_below_cap_triggers(self):
        result = check_daily_loss_cap(day_start_equity_usd=100.0,
                                      current_equity_usd=40.0,
                                      daily_loss_cap_pct=-50.0)
        assert result.triggered
        assert result.context["day_change_pct"] == pytest.approx(-60.0)

    def test_zero_day_start_does_not_divide_by_zero(self):
        # Edge: brand-new day, equity not yet known
        result = check_daily_loss_cap(day_start_equity_usd=0.0,
                                      current_equity_usd=0.0,
                                      daily_loss_cap_pct=-50.0)
        assert not result.triggered

    def test_profit_does_not_trigger(self):
        result = check_daily_loss_cap(day_start_equity_usd=100.0,
                                      current_equity_usd=110.0,
                                      daily_loss_cap_pct=-50.0)
        assert not result.triggered


# ── #4 Total loss cap ─────────────────────────────────────────────────


class TestTotalLossCap:
    def test_under_cap(self):
        result = check_total_loss_cap(initial_capital_usd=100.0,
                                      current_equity_usd=80.0,
                                      total_loss_cap_pct=-50.0)
        assert not result.triggered

    def test_at_cap_triggers_demote(self):
        result = check_total_loss_cap(initial_capital_usd=100.0,
                                      current_equity_usd=50.0,
                                      total_loss_cap_pct=-50.0)
        assert result.triggered
        assert result.trigger_id == "CAP-4"
        # Terminal: DEMOTE not HALT
        assert result.severity == KillSeverity.DEMOTE

    def test_zero_initial_no_div_zero(self):
        result = check_total_loss_cap(initial_capital_usd=0.0,
                                      current_equity_usd=0.0,
                                      total_loss_cap_pct=-50.0)
        assert not result.triggered


# ── #5 API permissions ────────────────────────────────────────────────


class TestApiPermissions:
    def test_trade_and_read_only_passes(self):
        result = check_api_permissions(perms=["trade", "read"])
        assert not result.triggered

    def test_case_insensitive_match(self):
        result = check_api_permissions(perms=["Trade", "READ"])
        assert not result.triggered

    def test_withdraw_perm_blocks_startup(self):
        result = check_api_permissions(perms=["trade", "read", "withdraw"])
        assert result.triggered
        assert result.trigger_id == "E4"
        assert result.severity == KillSeverity.DEMOTE
        assert "withdraw" in result.reason.lower()

    def test_transfer_perm_blocks_startup(self):
        result = check_api_permissions(perms=["trade", "read", "transfer"])
        assert result.triggered
        assert "transfer" in result.reason.lower()

    def test_missing_trade_blocks_startup(self):
        result = check_api_permissions(perms=["read"])
        assert result.triggered
        assert result.trigger_id == "E4"

    def test_missing_read_blocks_startup(self):
        result = check_api_permissions(perms=["trade"])
        assert result.triggered

    def test_empty_perms_blocks(self):
        result = check_api_permissions(perms=[])
        assert result.triggered


# ── #6 Reconciliation ──────────────────────────────────────────────────


class TestReconciliation:
    def test_consistent_does_not_trigger(self):
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.CONSISTENT,
            detail={"state": "both_flat"},
        )
        result = check_reconciliation(result=recon)
        assert not result.triggered

    def test_mismatch_triggers_halt(self):
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.MISMATCH,
            detail={"type": "size_diff", "local": 1, "okx": 2},
        )
        result = check_reconciliation(result=recon)
        assert result.triggered
        assert result.trigger_id == "A4"
        assert result.severity == KillSeverity.HALT

    def test_unavailable_triggers_halt(self):
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.UNAVAILABLE,
            detail={"error": "OKX timeout"},
        )
        result = check_reconciliation(result=recon)
        assert result.triggered
        assert result.trigger_id == "A4"
        # Unavailable also halts; recoverable
        assert result.severity == KillSeverity.HALT


# ── #7 Algo stop placement latency ────────────────────────────────────


class TestAlgoStopLatency:
    def test_fast_placement_passes(self):
        t0 = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=2)
        result = check_algo_stop_latency(entry_fill_ts=t0,
                                          stop_placed_ts=t1,
                                          max_latency_sec=5.0)
        assert not result.triggered

    def test_slow_placement_triggers(self):
        t0 = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=6)
        result = check_algo_stop_latency(entry_fill_ts=t0,
                                          stop_placed_ts=t1,
                                          max_latency_sec=5.0)
        assert result.triggered
        assert result.trigger_id == "B4"
        assert result.severity == KillSeverity.HALT
        assert result.context["latency_sec"] == pytest.approx(6.0)

    def test_no_timestamp_triggers(self):
        # stop placement returned nothing — treat as failure
        t0 = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
        result = check_algo_stop_latency(entry_fill_ts=t0,
                                          stop_placed_ts=None,
                                          max_latency_sec=5.0)
        assert result.triggered
        assert result.trigger_id == "B4"


# ── #9 NTP drift ──────────────────────────────────────────────────────


class TestNtpDrift:
    def test_no_drift_passes(self):
        result = check_ntp_drift(local_ts_sec=1000.0,
                                 server_ts_sec=1000.0,
                                 halt_threshold_sec=5.0,
                                 demote_threshold_sec=30.0)
        assert not result.triggered

    def test_under_halt_threshold_passes(self):
        result = check_ntp_drift(local_ts_sec=1000.0,
                                 server_ts_sec=1004.0,
                                 halt_threshold_sec=5.0,
                                 demote_threshold_sec=30.0)
        assert not result.triggered

    def test_above_halt_below_demote_triggers_C5(self):
        result = check_ntp_drift(local_ts_sec=1000.0,
                                 server_ts_sec=1010.0,
                                 halt_threshold_sec=5.0,
                                 demote_threshold_sec=30.0)
        assert result.triggered
        assert result.trigger_id == "C5"
        assert result.severity == KillSeverity.HALT

    def test_above_demote_triggers_C6(self):
        result = check_ntp_drift(local_ts_sec=1000.0,
                                 server_ts_sec=1040.0,
                                 halt_threshold_sec=5.0,
                                 demote_threshold_sec=30.0)
        assert result.triggered
        assert result.trigger_id == "C6"
        assert result.severity == KillSeverity.DEMOTE

    def test_negative_drift_uses_absolute_value(self):
        # Local clock ahead of server
        result = check_ntp_drift(local_ts_sec=1040.0,
                                 server_ts_sec=1000.0,
                                 halt_threshold_sec=5.0,
                                 demote_threshold_sec=30.0)
        assert result.triggered
        assert result.trigger_id == "C6"


# ── #10 Max position count ────────────────────────────────────────────


def _mk_pos(direction="LONG", size=1):
    return Position(inst_id="BTC-USDT-SWAP", direction=direction,
                    size_contracts=size, avg_price=75000.0)


class TestMaxPosition:
    def test_zero_positions_passes(self):
        result = check_max_position(local_open_positions=[], max_count=1)
        assert not result.triggered

    def test_one_position_passes(self):
        result = check_max_position(local_open_positions=[_mk_pos()],
                                    max_count=1)
        assert not result.triggered

    def test_two_positions_triggers_demote(self):
        # Bug in our own state machine — demote, don't trust ourselves
        result = check_max_position(
            local_open_positions=[_mk_pos(), _mk_pos("SHORT")],
            max_count=1,
        )
        assert result.triggered
        assert result.trigger_id == "MAX-POS"
        assert result.severity == KillSeverity.DEMOTE


# ── Connectivity (A1/A2/A3) ────────────────────────────────────────────


def _healthy_conn() -> ConnectivityStatus:
    return ConnectivityStatus(
        public_ws_ok=True, private_ws_ok=True,
        last_public_heartbeat_age_sec=1.0,
        last_private_heartbeat_age_sec=1.0,
        consecutive_reconnect_fails=0,
    )


class TestConnectivity:
    def test_healthy_passes(self):
        result = check_connectivity(status=_healthy_conn())
        assert not result.triggered

    def test_heartbeat_age_above_A3_threshold_halts(self):
        # 31s > 30s heartbeat timeout → A3 HALT
        # (but still < 5min A1 threshold)
        conn = replace(_healthy_conn(), last_private_heartbeat_age_sec=31.0)
        result = check_connectivity(status=conn,
                                    heartbeat_timeout_sec=30.0,
                                    ws_disconnect_demote_sec=300)
        assert result.triggered
        assert result.trigger_id == "A3"
        assert result.severity == KillSeverity.HALT

    def test_heartbeat_age_above_A1_threshold_demotes(self):
        # 6 min > 5 min → A1 DEMOTE
        conn = replace(_healthy_conn(),
                       last_private_heartbeat_age_sec=360.0)
        result = check_connectivity(status=conn,
                                    ws_disconnect_demote_sec=300)
        assert result.triggered
        assert result.trigger_id == "A1"
        assert result.severity == KillSeverity.DEMOTE

    def test_3_reconnect_fails_demotes_A2(self):
        conn = replace(_healthy_conn(), consecutive_reconnect_fails=3)
        result = check_connectivity(status=conn,
                                    reconnect_fail_demote_count=3)
        assert result.triggered
        assert result.trigger_id == "A2"
        assert result.severity == KillSeverity.DEMOTE

    def test_A2_takes_priority_over_A1(self):
        # If both fire, A2 (reconnect fails) is checked first per impl
        conn = replace(_healthy_conn(),
                       last_private_heartbeat_age_sec=999.0,
                       consecutive_reconnect_fails=5)
        result = check_connectivity(status=conn)
        assert result.triggered
        assert result.trigger_id == "A2"


# ── Aggregator: run_all_checks ────────────────────────────────────────


class _MockCfg:
    """Minimal cfg duck-type for run_all_checks."""
    initial_capital_usd: float = 100.0
    daily_loss_cap_pct: float = -50.0
    total_loss_cap_pct: float = -50.0
    max_position_count: int = 1
    ws_disconnect_demote_sec: int = 300
    reconnect_fail_demote_count: int = 3
    heartbeat_timeout_sec: float = 30.0
    ntp_drift_halt_sec: float = 5.0
    ntp_drift_demote_sec: float = 30.0


class TestRunAllChecks:
    def test_all_healthy_returns_empty(self):
        cfg = _MockCfg()
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.CONSISTENT, detail={},
        )
        triggered = run_all_checks(
            cfg=cfg,
            equity_usd=100.0,
            day_start_equity_usd=100.0,
            local_positions=[],
            reconciliation=recon,
            connectivity=_healthy_conn(),
            ntp_drift_sec=None,
        )
        assert triggered == []

    def test_multiple_triggers_all_returned(self):
        cfg = _MockCfg()
        # Daily loss + total loss + reconciliation mismatch all fire
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.MISMATCH,
            detail={"type": "orphan_exchange"},
        )
        triggered = run_all_checks(
            cfg=cfg,
            equity_usd=40.0,            # -60% from initial 100 + day_start
            day_start_equity_usd=100.0,
            local_positions=[],
            reconciliation=recon,
            connectivity=_healthy_conn(),
            ntp_drift_sec=None,
        )
        trigger_ids = {r.trigger_id for r in triggered}
        assert "CAP-3" in trigger_ids   # daily
        assert "CAP-4" in trigger_ids   # total
        assert "A4" in trigger_ids      # reconciliation

    def test_skips_ntp_when_no_drift_provided(self):
        cfg = _MockCfg()
        recon = ReconciliationResult(
            verdict=ReconciliationVerdict.CONSISTENT, detail={},
        )
        triggered = run_all_checks(
            cfg=cfg,
            equity_usd=100.0,
            day_start_equity_usd=100.0,
            local_positions=[],
            reconciliation=recon,
            connectivity=_healthy_conn(),
            ntp_drift_sec=None,
        )
        # No NTP check ran
        assert all(r.trigger_id not in ("C5", "C6") for r in triggered)
