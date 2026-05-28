"""Unit tests for indicator/okx/approval.py.

Mock OkxStateStore entirely.  Verifies state-machine transitions
(pending → approved/denied/expired), auto-mode threshold flip, and
the JSON round-trip safety of TradeIntent.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from indicator.okx.approval import (
    APPROVAL_TTL_MINUTES,
    MAX_DRIFT_PCT,
    ApprovalDecision,
    ApprovalGate,
    TradeIntent,
)


# ── TradeIntent JSON round-trip ──────────────────────────────────────


class TestTradeIntent:
    def test_round_trip(self):
        intent = TradeIntent(
            direction="LONG", tier="Strong",
            entry_price=75000.0, stop_price=74550.0,
            atr=150.0, stop_dist=450.0,
            size_contracts=5, size_frac=0.5,
            notional_usd=375.0, equity_before=100.0,
            bar_ts_iso="2026-05-28T12:00:00+00:00",
            model_version="v9_20260512",
        )
        rehydrated = TradeIntent.from_json(intent.to_json())
        assert rehydrated == intent

    def test_extra_dict_preserved(self):
        intent = TradeIntent(
            direction="SHORT", tier="Moderate",
            entry_price=75000.0, stop_price=75450.0,
            atr=150.0, stop_dist=450.0, size_contracts=3,
            size_frac=0.3, notional_usd=225.0, equity_before=100.0,
            bar_ts_iso="2026-05-28T12:00:00+00:00",
            extra={"sig_id": 99, "regime": "TRENDING_BEAR"},
        )
        rehydrated = TradeIntent.from_json(intent.to_json())
        assert rehydrated.extra == {"sig_id": 99, "regime": "TRENDING_BEAR"}


# ── Helpers ──────────────────────────────────────────────────────────


def _mk_gate(store=None, chat_id="critical-chat") -> ApprovalGate:
    if store is None:
        store = MagicMock()
        store.get_executed_approval_count.return_value = 0
        store.expire_old_approvals.return_value = 0
    return ApprovalGate(store=store, alert_chat_id=chat_id)


def _sample_intent() -> TradeIntent:
    return TradeIntent(
        direction="LONG", tier="Strong",
        entry_price=75000.0, stop_price=74550.0,
        atr=150.0, stop_dist=450.0, size_contracts=5,
        size_frac=0.5, notional_usd=375.0, equity_before=100.0,
        bar_ts_iso="2026-05-28T12:00:00+00:00",
    )


# ── Auto-mode threshold ──────────────────────────────────────────────


class TestAutoMode:
    def test_zero_count_is_manual(self):
        gate = _mk_gate()
        assert not gate.is_auto_mode()

    def test_below_threshold_still_manual(self):
        store = MagicMock()
        store.get_executed_approval_count.return_value = 4
        gate = _mk_gate(store=store)
        assert not gate.is_auto_mode()

    def test_at_threshold_flips_to_auto(self):
        store = MagicMock()
        store.get_executed_approval_count.return_value = 5
        gate = _mk_gate(store=store)
        assert gate.is_auto_mode()

    def test_above_threshold_still_auto(self):
        store = MagicMock()
        store.get_executed_approval_count.return_value = 25
        gate = _mk_gate(store=store)
        assert gate.is_auto_mode()

    def test_db_failure_fails_closed_to_manual(self):
        # Safer to ask than to silently auto-trade on DB blip
        store = MagicMock()
        store.get_executed_approval_count.side_effect = RuntimeError("db")
        gate = _mk_gate(store=store)
        assert not gate.is_auto_mode()


# ── request_approval ─────────────────────────────────────────────────


class TestRequestApproval:
    def test_inserts_and_sends_telegram(self):
        store = MagicMock()
        store.get_executed_approval_count.return_value = 2
        store.insert_pending_approval.return_value = 42
        gate = _mk_gate(store=store)
        with patch("indicator.okx.approval.send_critical",
                   return_value=True) as tg:
            approval_id = gate.request_approval(_sample_intent())
        assert approval_id == 42
        # Persisted with future expires_at
        ins_kwargs = store.insert_pending_approval.call_args.kwargs
        intent_json = ins_kwargs["intent_json"]
        decoded = json.loads(intent_json)
        assert decoded["direction"] == "LONG"
        # Telegram fired with the right chat id and id in body
        tg.assert_called_once()
        chat_id_arg, msg = tg.call_args.args
        assert chat_id_arg == "critical-chat"
        assert "42" in msg
        assert "/yes_42" in msg or "yes_42" in msg

    def test_db_failure_returns_none(self):
        store = MagicMock()
        store.get_executed_approval_count.return_value = 0
        store.insert_pending_approval.side_effect = RuntimeError("db")
        gate = _mk_gate(store=store)
        with patch("indicator.okx.approval.send_critical",
                   return_value=True):
            assert gate.request_approval(_sample_intent()) is None

    def test_telegram_failure_does_not_lose_approval(self):
        # If TG send fails, the PENDING row still exists — operator can
        # /yes manually if they happen to see it.  Caller should be aware.
        store = MagicMock()
        store.get_executed_approval_count.return_value = 0
        store.insert_pending_approval.return_value = 99
        gate = _mk_gate(store=store)
        with patch("indicator.okx.approval.send_critical",
                   return_value=False):
            approval_id = gate.request_approval(_sample_intent())
        assert approval_id == 99


# ── approve / deny ───────────────────────────────────────────────────


class TestApprove:
    def test_pending_to_approved_returns_intent(self):
        store = MagicMock()
        store.expire_old_approvals.return_value = 0
        store.get_approval.return_value = {
            "id": 1, "status": "PENDING",
            "intent": _sample_intent().to_json(),
        }
        store.decide_approval.return_value = True
        gate = _mk_gate(store=store)
        result = gate.approve(1, decided_by="op1")
        assert result.ok
        assert result.status == "APPROVED"
        assert result.intent.direction == "LONG"
        store.decide_approval.assert_called_once_with(
            approval_id=1, decision="APPROVED", decided_by="op1",
        )

    def test_not_found(self):
        store = MagicMock()
        store.expire_old_approvals.return_value = 0
        store.get_approval.return_value = None
        gate = _mk_gate(store=store)
        result = gate.approve(99)
        assert not result.ok
        assert result.status == "NOT_FOUND"

    def test_already_executed_rejected(self):
        store = MagicMock()
        store.expire_old_approvals.return_value = 0
        store.get_approval.return_value = {
            "id": 1, "status": "EXECUTED",
            "intent": _sample_intent().to_json(),
        }
        gate = _mk_gate(store=store)
        result = gate.approve(1)
        assert not result.ok
        assert result.status == "NOT_PENDING"

    def test_race_lost_to_expiry(self):
        # Row was PENDING when we read, but expired by the time we
        # tried to flip → decide_approval returns False
        store = MagicMock()
        store.expire_old_approvals.return_value = 0
        store.get_approval.return_value = {
            "id": 1, "status": "PENDING",
            "intent": _sample_intent().to_json(),
        }
        store.decide_approval.return_value = False
        gate = _mk_gate(store=store)
        result = gate.approve(1)
        assert not result.ok
        assert result.status == "EXPIRED"


class TestDeny:
    def test_pending_to_denied(self):
        store = MagicMock()
        store.expire_old_approvals.return_value = 0
        store.get_approval.return_value = {
            "id": 1, "status": "PENDING",
            "intent": _sample_intent().to_json(),
        }
        store.decide_approval.return_value = True
        gate = _mk_gate(store=store)
        result = gate.deny(1, decided_by="op1")
        assert result.ok
        assert result.status == "DENIED"
        store.decide_approval.assert_called_once_with(
            approval_id=1, decision="DENIED", decided_by="op1",
        )


# ── Drift check ──────────────────────────────────────────────────────


class TestDriftCheck:
    def test_no_drift_returns_none(self):
        gate = _mk_gate()
        intent = _sample_intent()
        assert gate.check_drift(intent, current_price=75000.0) is None

    def test_within_threshold_returns_none(self):
        gate = _mk_gate()
        intent = _sample_intent()
        # 0.3% drift, under 0.5% threshold
        assert gate.check_drift(intent, current_price=75225.0) is None

    def test_above_threshold_returns_drift(self):
        gate = _mk_gate()
        intent = _sample_intent()
        # 1% drift up — exceeds threshold
        drift = gate.check_drift(intent, current_price=75750.0)
        assert drift is not None
        assert drift == pytest.approx(0.01)

    def test_above_threshold_down_also_caught(self):
        gate = _mk_gate()
        intent = _sample_intent()
        drift = gate.check_drift(intent, current_price=74250.0)
        assert drift is not None

    def test_zero_intent_price_returns_none(self):
        # Edge case: avoid div-by-zero
        gate = _mk_gate()
        intent = _sample_intent()
        intent.entry_price = 0.0
        assert gate.check_drift(intent, current_price=75000.0) is None


# ── mark_executed / mark_stale ───────────────────────────────────────


class TestMarkExecuted:
    def test_calls_store(self):
        store = MagicMock()
        gate = _mk_gate(store=store)
        gate.mark_executed(approval_id=1, position_id=42)
        store.mark_approval_executed.assert_called_once_with(
            approval_id=1, position_id=42,
        )

    def test_swallows_failure(self):
        store = MagicMock()
        store.mark_approval_executed.side_effect = RuntimeError("db")
        gate = _mk_gate(store=store)
        # Must not raise — caller can't recover anyway
        gate.mark_executed(approval_id=1, position_id=42)


class TestMarkStale:
    def test_calls_store(self):
        store = MagicMock()
        gate = _mk_gate(store=store)
        gate.mark_stale(1, reason="price_drift")
        store.mark_approval_stale.assert_called_once_with(
            approval_id=1, reason="price_drift",
        )
