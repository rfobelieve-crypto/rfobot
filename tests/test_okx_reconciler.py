"""Unit tests for indicator/okx/reconciler.py.

Verifies the 5 mismatch shapes (orphan_local, orphan_exchange, size_diff,
direction_diff, multiple_exchange) + consistent + OKX-unavailable paths.

Mocks OkxClient.get_positions() and OkxStateStore.get_open_position() —
the reconciler logic is pure compare; we never touch real OKX or DB.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from indicator.okx.reconciler import PositionReconciler
from indicator.okx.types import (
    Position,
    ReconciliationVerdict,
)


INST = "BTC-USDT-SWAP"


def _mk_okx_pos(direction="LONG", size=1, price=75000.0) -> Position:
    return Position(inst_id=INST, direction=direction,
                    size_contracts=size, avg_price=price)


def _mk_reconciler(local_open: dict | None,
                   okx_positions: list[Position] | Exception):
    """Build a reconciler with mocked client + store.

    If okx_positions is an Exception, client.get_positions raises it.
    """
    client = MagicMock()
    store = MagicMock()
    store.get_open_position.return_value = local_open

    if isinstance(okx_positions, Exception):
        client.get_positions.side_effect = okx_positions
    else:
        client.get_positions.return_value = okx_positions

    return PositionReconciler(client=client, store=store, inst_id=INST)


# ── CONSISTENT paths ───────────────────────────────────────────────────


class TestConsistent:
    def test_both_flat(self):
        rec = _mk_reconciler(local_open=None, okx_positions=[])
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.CONSISTENT
        assert result.detail["state"] == "both_flat"

    def test_both_one_long_match(self):
        local = {"id": 1, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="LONG", size=5, price=75000.0)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.CONSISTENT

    def test_both_one_short_match(self):
        local = {"id": 2, "direction": "SHORT",
                 "size_contracts": 3, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="SHORT", size=3, price=75000.0)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.CONSISTENT

    def test_price_drift_warns_but_consistent(self):
        # Avg price differs > 1% → logged as warning, but still CONSISTENT
        local = {"id": 1, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="LONG", size=5, price=77000.0)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        # Per reconciler.py: price diff is warning, not mismatch
        assert result.verdict == ReconciliationVerdict.CONSISTENT


# ── MISMATCH paths ─────────────────────────────────────────────────────


class TestMismatch:
    def test_orphan_exchange_okx_has_local_does_not(self):
        # We are flat locally, OKX has a position — UNKNOWN cold-start
        rec = _mk_reconciler(local_open=None,
                             okx_positions=[_mk_okx_pos()])
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.MISMATCH
        assert result.detail["type"] == "orphan_exchange"
        assert result.detail["count"] == 1

    def test_orphan_local_local_has_okx_does_not(self):
        # Local thinks open, OKX flat — possibly stop hit and we missed it
        local = {"id": 7, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        rec = _mk_reconciler(local_open=local, okx_positions=[])
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.MISMATCH
        assert result.detail["type"] == "orphan_local"
        assert result.detail["local_id"] == 7

    def test_size_diff(self):
        local = {"id": 1, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="LONG", size=8)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.MISMATCH
        assert result.detail["type"] == "size_diff"
        assert result.detail["local_size"] == 5
        assert result.detail["okx_size"] == 8

    def test_direction_diff(self):
        local = {"id": 1, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="SHORT", size=5)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.MISMATCH
        assert result.detail["type"] == "direction_diff"
        assert result.detail["local_dir"] == "LONG"
        assert result.detail["okx_dir"] == "SHORT"

    def test_multiple_exchange_positions(self):
        # We never allow > 1 position; if OKX has multiple, halt
        local = {"id": 1, "direction": "LONG",
                 "size_contracts": 5, "entry_price": 75000.0}
        okx = [_mk_okx_pos(direction="LONG", size=5),
               _mk_okx_pos(direction="SHORT", size=3)]
        rec = _mk_reconciler(local_open=local, okx_positions=okx)
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.MISMATCH
        assert result.detail["type"] == "multiple_exchange_positions"
        assert result.detail["count"] == 2

    def test_other_inst_id_ignored(self):
        # OKX returns positions for ETH-USDT-SWAP — should be filtered out
        local = None
        eth_pos = Position(inst_id="ETH-USDT-SWAP", direction="LONG",
                           size_contracts=10, avg_price=3000.0)
        rec = _mk_reconciler(local_open=local, okx_positions=[eth_pos])
        result = rec.reconcile_cycle()
        # Both flat (after filtering)
        assert result.verdict == ReconciliationVerdict.CONSISTENT

    def test_zero_size_okx_position_treated_as_flat(self):
        # OKX may return position with size=0 after close — treat as flat
        local = None
        zero_pos = Position(inst_id=INST, direction="LONG",
                            size_contracts=0, avg_price=75000.0)
        rec = _mk_reconciler(local_open=local, okx_positions=[zero_pos])
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.CONSISTENT
        assert result.detail["state"] == "both_flat"


# ── UNAVAILABLE path ───────────────────────────────────────────────────


class TestUnavailable:
    def test_okx_query_raises(self):
        rec = _mk_reconciler(local_open=None,
                             okx_positions=RuntimeError("OKX 5xx"))
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.UNAVAILABLE
        assert "OKX 5xx" in str(result.detail.get("error", ""))


# ── Logging side-effect ────────────────────────────────────────────────


class TestLogging:
    def test_every_reconcile_writes_log(self):
        rec = _mk_reconciler(local_open=None, okx_positions=[])
        rec.reconcile_cycle()
        # store.log_reconciliation MUST be called exactly once
        rec._store.log_reconciliation.assert_called_once()
        call_kwargs = rec._store.log_reconciliation.call_args.kwargs
        assert call_kwargs["verdict"] == "CONSISTENT"

    def test_log_failure_does_not_propagate(self):
        # If DB write fails, reconcile_cycle must still return a verdict
        client = MagicMock()
        store = MagicMock()
        store.get_open_position.return_value = None
        client.get_positions.return_value = []
        store.log_reconciliation.side_effect = RuntimeError("DB down")
        rec = PositionReconciler(client=client, store=store, inst_id=INST)
        # Must not raise
        result = rec.reconcile_cycle()
        assert result.verdict == ReconciliationVerdict.CONSISTENT
