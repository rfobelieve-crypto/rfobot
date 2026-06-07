"""End-to-end tests using MockOkxClient — the testnet we never ran.

Two jobs:
  1. Prove MockOkxClient faithfully stands in for OkxClient (open/close/algo/
     positions/balance/server-time behave with real state).
  2. Drive the REAL V7OkxExecutor.cycle() against the fake + a real
     PositionReconciler + an in-memory store, and DELIBERATELY trip every
     kill switch — satisfying the CLAUDE.md hard rule that kill switches
     must be verified to fire, not just written into code.

No real money, no network.  See indicator/okx/mock_client.py.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from indicator.okx.config import OkxConfig
from indicator.okx.executor import V7OkxExecutor
from indicator.okx.mock_client import MockOkxClient
from indicator.okx.reconciler import PositionReconciler
from indicator.okx.types import ExecutorStatus, Side


# ── Fixtures ─────────────────────────────────────────────────────────


def _mk_klines(n: int = 40, start_price: float = 75000.0,
               seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.003, n)
    close = start_price * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    idx = pd.date_range("2026-06-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(500, 2000, n),
    }, index=idx)


def _mk_cfg(**overrides) -> OkxConfig:
    base = OkxConfig(
        api_key="k", api_secret="s", passphrase="p",
        telegram_critical_chat_id="critical-chat",
        initial_capital_usd=155.0,
        contract_size_base=0.01,
        is_simulated=1, leverage=10,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class FakeStore:
    """Minimal in-memory OkxStateStore stand-in.

    Stores a single position (we enforce max 1) plus the balance/equity
    anchors the kill checks read.  Write methods record so tests can
    assert.  This is the in-memory half of the testnet harness.
    """

    def __init__(self, *, latest_equity=None, day_start_equity=None):
        self._latest_equity = latest_equity
        self._day_start_equity = day_start_equity
        self._pos: dict | None = None
        self._next_id = 41
        self.statuses: list[dict] = []
        self.kill_logs: list[dict] = []
        self.balance_snapshots: list[dict] = []
        self.closed: list[dict] = []

    # reads used by kill checks / start()
    def get_latest_balance(self):
        if self._latest_equity is None:
            return None
        return {"total_eq_usd": self._latest_equity,
                "available_usd": self._latest_equity}

    def get_day_start_equity(self):
        return self._day_start_equity

    def set_equity(self, latest, day_start):
        self._latest_equity = latest
        self._day_start_equity = day_start

    # position lifecycle
    def get_open_position(self):
        return self._pos if (self._pos and self._pos["status"] == "OPEN") \
            else None

    def get_all_open_positions(self):
        return [self._pos] if self.get_open_position() else []

    def insert_open_position(self, **k):
        self._next_id += 1
        self._pos = {"id": self._next_id, "status": "OPEN", **k}
        return self._next_id

    def set_position_okx_ids(self, *, position_id, entry_ord_id=None,
                             stop_algo_id=None):
        if self._pos and self._pos["id"] == position_id:
            if entry_ord_id is not None:
                self._pos["entry_ord_id"] = entry_ord_id
            if stop_algo_id is not None:
                self._pos["stop_algo_id"] = stop_algo_id

    def close_position(self, *, position_id, new_status="CLOSED", **k):
        if self._pos and self._pos["id"] == position_id:
            self._pos["status"] = new_status
            self.closed.append({"position_id": position_id,
                                "new_status": new_status, **k})

    def update_trail(self, **k):
        if self._pos:
            self._pos["trail_extreme"] = k.get("trail_extreme")
            self._pos["current_stop"] = k.get("current_stop")

    # bookkeeping (no-ops that record)
    def insert_balance_snapshot(self, **k):
        self.balance_snapshots.append(k)

    def save_executor_status(self, **k):
        self.statuses.append(k)

    def log_kill_trigger(self, **k):
        self.kill_logs.append(k)

    def log_reconciliation(self, **k):
        pass

    def resolve_open_kills(self, **k):
        return 0

    def get_consecutive_clean_days(self):
        return 0


def _mk_harness(cfg=None, *, latest_equity=None, day_start_equity=None,
                mark_price=75000.0):
    cfg = cfg or _mk_cfg()
    client = MockOkxClient(cfg, mark_price=mark_price)
    store = FakeStore(latest_equity=latest_equity,
                      day_start_equity=day_start_equity)
    recon = PositionReconciler(client=client, store=store,
                               inst_id=cfg.inst_id)
    exe = V7OkxExecutor(client=client, store=store, reconciler=recon, cfg=cfg)
    return exe, client, store, cfg


# ════════════════════════════════════════════════════════════════════
#  1. MockOkxClient faithfulness
# ════════════════════════════════════════════════════════════════════


class TestMockBrokerFidelity:
    def test_market_order_opens_and_reports_position(self):
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        assert client.get_positions(inst_id=cfg.inst_id) == []
        res = client.submit_market_order(
            inst_id=cfg.inst_id, side=Side.BUY, sz=0.5,
            td_mode="cross", cl_ord_id="x", pos_side="long")
        assert res.status == "filled"
        assert res.fill_size == 0.5
        pos = client.get_positions(inst_id=cfg.inst_id)
        assert len(pos) == 1
        assert pos[0].direction == "LONG"
        assert pos[0].size_contracts == 0.5

    def test_reduce_only_flattens(self):
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        client.submit_market_order(inst_id=cfg.inst_id, side=Side.BUY,
                                   sz=0.5, td_mode="cross", pos_side="long")
        client.submit_market_order(inst_id=cfg.inst_id, side=Side.SELL,
                                   sz=0.5, td_mode="cross", pos_side="long",
                                   reduce_only=True)
        assert client.get_positions(inst_id=cfg.inst_id) == []

    def test_algo_register_amend_cancel(self):
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        algo = client.submit_algo_stop(
            inst_id=cfg.inst_id, side=Side.SELL, sz=0.5,
            trigger_px=70000.0, td_mode="cross", algo_cl_ord_id="a1")
        assert algo.status == "live" and algo.algo_id
        amend = client.amend_algo_stop(algo_id=algo.algo_id,
                                       new_trigger_px=71000.0)
        assert amend.status == "ok"
        cancel = client.cancel_algo_stop(algo_id=algo.algo_id)
        assert cancel.status == "ok"
        # cancel again -> not found
        assert client.cancel_algo_stop(algo_id=algo.algo_id).status \
            == "not_found"

    def test_market_rejection_injection(self):
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        client.reject_market_order("mock_code_51008")
        res = client.submit_market_order(inst_id=cfg.inst_id, side=Side.BUY,
                                         sz=0.5, td_mode="cross")
        assert res.status == "rejected"
        assert res.error == "mock_code_51008"

    def test_server_time_offset(self):
        import time as _t
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        client.set_server_time_offset(40.0)
        drift = client.get_server_time() - _t.time()
        assert 39 < drift < 41

    def test_perms_config_for_e4(self):
        cfg = _mk_cfg()
        client = MockOkxClient(cfg)
        client.set_perms(["read", "trade", "withdraw"])
        cfgd = client.get_account_config()
        assert "withdraw" in cfgd["data"][0]["perm"]


# ════════════════════════════════════════════════════════════════════
#  2. Full lifecycle through the real executor
# ════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_start_reaches_active_when_healthy(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        exe.start()
        assert exe.get_status() == ExecutorStatus.ACTIVE

    def test_open_through_mock_reflects_in_positions(self):
        # $10k so size is clean & well above min lot
        cfg = _mk_cfg(initial_capital_usd=10000.0)
        exe, client, store, _ = _mk_harness(cfg=cfg, latest_equity=10000.0,
                                            day_start_equity=10000.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe._open_position(klines=_mk_klines(),
                                        signal_direction="UP",
                                        signal_strength="Strong",
                                        model_version="v1")
        assert result.action == "open"
        pos = client.get_positions(inst_id=cfg.inst_id)
        assert len(pos) == 1
        assert pos[0].direction == "LONG"
        assert pos[0].size_contracts == pytest.approx(
            result.detail["size_contracts"])
        # an algo stop was placed
        assert len(client.algo_orders) == 1
        # DB row exists
        assert store.get_open_position() is not None

    def test_gap_stop_out_closes_via_ws(self):
        """Open, then OKX trail stop fires at a -10% gap -> WS closes DB row.

        This is the path that, if broken, leaves the DB OPEN forever after
        OKX auto-closes (the orphan_local HALT loop).
        """
        cfg = _mk_cfg(initial_capital_usd=10000.0)
        exe, client, store, _ = _mk_harness(cfg=cfg, latest_equity=10000.0,
                                            day_start_equity=10000.0)
        exe._wire_ws_callbacks()   # wire on_order so algo fills route in
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe._open_position(klines=_mk_klines(), signal_direction="UP",
                               signal_strength="Strong", model_version="v1")
            assert store.get_open_position() is not None
            # -10% gap stop-out
            client.fire_algo_stop(fill_price=75000.0 * 0.90)
        # DB row closed by the WS algo-fill handler
        assert store.get_open_position() is None
        assert store.closed and store.closed[-1]["new_status"] == "CLOSED"
        # OKX is flat too
        assert client.get_positions(inst_id=cfg.inst_id) == []


# ════════════════════════════════════════════════════════════════════
#  3. Every kill switch actually fires (the whole point)
# ════════════════════════════════════════════════════════════════════


def _active(exe):
    exe._status = ExecutorStatus.ACTIVE
    return exe


class TestKillSwitchesFire:
    """Drive cycle() with a NEUTRAL signal (no open) after injecting each
    adverse condition; assert the executor transitions to HALTED / DEMOTED.
    """

    def _cycle_neutral(self, exe):
        with patch("indicator.okx.executor.send_critical", return_value=True):
            return exe.cycle(klines=_mk_klines(), signal_direction="NEUTRAL",
                             signal_strength="Weak")

    def test_manual_interference_orphan_exchange_demotes_without_closing(self):
        """Operator opens a trade on the same account -> OKX shows a
        position the executor never opened -> MANUAL-INTERFERENCE DEMOTE
        (sticky), and the executor must NOT touch that foreign position.
        This is the 2026-06-05 manual-blowup detection vector.
        """
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_okx_position("LONG", 1.0)   # foreign position
        result = self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert result.action == "demoted"
        assert any(k["trigger_id"] == "MANUAL-INTERFERENCE"
                   for k in store.kill_logs)
        # executor did NOT try to close the foreign position
        assert client.market_orders == []
        assert client.get_positions(inst_id=cfg.inst_id)[0].direction == "LONG"

    def test_direction_diff_demotes(self):
        # Local says LONG, OKX says SHORT -> direction_diff -> manual interfere
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        store.insert_open_position(
            entry_time=datetime.utcnow(), direction="LONG", entry_tier="Strong",
            entry_price=75000.0, atr_at_entry=100.0, stop_dist=300.0,
            current_stop=74700.0, size_contracts=1.0, size_frac=0.5,
            notional_usd=75000.0, equity_before=155.0,
            entry_cl_ord_id="v7-x", stop_algo_cl_ord_id="v7a-x",
            model_version="v1")
        client.set_okx_position("SHORT", 1.0)
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert any(k["trigger_id"] == "MANUAL-INTERFERENCE"
                   for k in store.kill_logs)

    def test_ntp_drift_demote_c6(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_server_time_offset(40.0)   # > 30s demote
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert any(k["trigger_id"] == "C6" for k in store.kill_logs)

    def test_ntp_drift_halt_c5(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_server_time_offset(10.0)   # 5s < drift < 30s -> halt
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.HALTED
        assert any(k["trigger_id"] == "C5" for k in store.kill_logs)

    def test_daily_loss_cap_halts(self):
        # -21% on the day (> -20% cap) but not past the -30% total cap
        exe, client, store, cfg = _mk_harness(latest_equity=110.0,
                                              day_start_equity=140.0)
        _active(exe)
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.HALTED
        assert any(k["trigger_id"] == "CAP-3" for k in store.kill_logs)

    def test_total_loss_cap_demotes(self):
        # -35% from initial 155 (past -30% total cap); flat on the day
        exe, client, store, cfg = _mk_harness(latest_equity=100.0,
                                              day_start_equity=100.0)
        _active(exe)
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert any(k["trigger_id"] == "CAP-4" for k in store.kill_logs)

    def test_capital_cap_halts_on_overfund(self):
        # equity > 1.5x the $155 hard cap -> CAP-2 (accidental over-funding)
        exe, client, store, cfg = _mk_harness(latest_equity=300.0,
                                              day_start_equity=300.0)
        _active(exe)
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.HALTED
        assert any(k["trigger_id"] == "CAP-2" for k in store.kill_logs)

    def test_connectivity_a3_heartbeat_halt(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_connectivity(last_private_heartbeat_age_sec=35.0)  # >30
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.HALTED
        assert any(k["trigger_id"] == "A3" for k in store.kill_logs)

    def test_connectivity_a1_disconnect_demote(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_connectivity(last_private_heartbeat_age_sec=400.0)  # >300
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert any(k["trigger_id"] == "A1" for k in store.kill_logs)

    def test_connectivity_a2_reconnect_fails_demote(self):
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_connectivity(consecutive_reconnect_fails=3)
        self._cycle_neutral(exe)
        assert exe.get_status() == ExecutorStatus.DEMOTED
        assert any(k["trigger_id"] == "A2" for k in store.kill_logs)

    def test_withdraw_perm_demotes_at_start(self):
        # E4: API key must never carry withdraw/transfer permission
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        client.set_perms(["read", "trade", "withdraw"])
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe.start()
        assert exe.get_status() == ExecutorStatus.DEMOTED


class TestPresubmitGuard:
    """Defense-in-depth: a buggy oversized intent must NOT reach OKX."""

    def test_overleveraged_intent_blocked_before_submit(self):
        from indicator.okx.approval import TradeIntent
        cfg = _mk_cfg(initial_capital_usd=89.0, max_effective_leverage=3.0)
        exe, client, store, _ = _mk_harness(cfg=cfg, latest_equity=89.0,
                                            day_start_equity=89.0)
        # Simulate the 2026-06-05 bug: int()-floor produced 1 whole contract
        # ($750 notional) on an $89 account -> ~8.4x effective leverage.
        bad_intent = TradeIntent(
            direction="LONG", tier="Strong",
            entry_price=75000.0, stop_price=74550.0,
            atr=150.0, stop_dist=450.0,
            size_contracts=1.0, size_frac=8.4,
            notional_usd=750.0, equity_before=89.0,
            bar_ts_iso="2026-06-05T12:00:00", model_version="v1")
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe.execute_approved_intent(bad_intent, approval_id=None)
        assert result.action == "none"
        assert result.detail["reason"] == "presubmit_guard_blocked"
        assert result.detail["trigger_id"] == "PRESUBMIT-LEV"
        # NOTHING was sent to OKX
        assert client.market_orders == []
        assert client.get_positions(inst_id=cfg.inst_id) == []

    def test_within_leverage_intent_passes(self):
        from indicator.okx.approval import TradeIntent
        cfg = _mk_cfg(initial_capital_usd=89.0, max_effective_leverage=3.0)
        exe, client, store, _ = _mk_harness(cfg=cfg, latest_equity=89.0,
                                            day_start_equity=89.0)
        # Correct B sizing: ~2x notional, fractional contracts
        ok_intent = TradeIntent(
            direction="LONG", tier="Strong",
            entry_price=75000.0, stop_price=74550.0,
            atr=150.0, stop_dist=450.0,
            size_contracts=0.24, size_frac=2.02,
            notional_usd=180.0, equity_before=89.0,
            bar_ts_iso="2026-06-06T12:00:00", model_version="v1")
        with patch("indicator.okx.executor.send_critical", return_value=True):
            result = exe.execute_approved_intent(ok_intent, approval_id=None)
        assert result.action == "open"
        assert len(client.market_orders) == 1

    def test_pure_function_size_and_leverage(self):
        from indicator.okx.kill_checks import check_presubmit_order
        # bad leverage
        r = check_presubmit_order(size_contracts=1.0, notional_usd=750.0,
                                  equity_usd=89.0, max_effective_leverage=3.0,
                                  min_size_contracts=0.01)
        assert r.triggered and r.trigger_id == "PRESUBMIT-LEV"
        # sub-min size
        r2 = check_presubmit_order(size_contracts=0.0, notional_usd=0.0,
                                   equity_usd=89.0, max_effective_leverage=3.0,
                                   min_size_contracts=0.01)
        assert r2.triggered and r2.trigger_id == "PRESUBMIT-SIZE"
        # fine
        r3 = check_presubmit_order(size_contracts=0.24, notional_usd=180.0,
                                   equity_usd=89.0, max_effective_leverage=3.0,
                                   min_size_contracts=0.01)
        assert not r3.triggered


class TestKillSwitchRecovery:
    def test_halt_auto_resumes_when_condition_clears(self):
        # NTP halt this cycle, clears next cycle -> back to ACTIVE
        exe, client, store, cfg = _mk_harness(latest_equity=155.0,
                                              day_start_equity=155.0)
        _active(exe)
        client.set_server_time_offset(10.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe.cycle(klines=_mk_klines(), signal_direction="NEUTRAL",
                      signal_strength="Weak")
        assert exe.get_status() == ExecutorStatus.HALTED
        # clear drift; NTP probe is rate-limited so it won't re-probe, but
        # the other checks pass and HALT auto-resumes to ACTIVE
        client.set_server_time_offset(0.0)
        with patch("indicator.okx.executor.send_critical", return_value=True):
            exe.cycle(klines=_mk_klines(), signal_direction="NEUTRAL",
                      signal_strength="Weak")
        assert exe.get_status() == ExecutorStatus.ACTIVE
