"""PositionReconciler — compares OkxStateStore (local) vs OkxClient (OKX).

P2: exchange is source of truth.
P4: never auto-mutate local to match OKX without explicit caller intent.

Verdicts:
  CONSISTENT  — local + OKX agree (or both flat)
  MISMATCH    — disagreement; caller should halt
  UNAVAILABLE — OKX query failed; caller should halt

Mismatch types we detect (kill_criteria.md A4):
  - orphan_local:    local has OPEN, OKX has no position
  - orphan_exchange: OKX has position, local has none
  - size_diff:       both have positions, contracts differ
  - direction_diff:  both have positions, LONG vs SHORT differ
  - price_diff:      avg price differs > 1% (warning only, not mismatch)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from indicator.okx.client import OkxClient
from indicator.okx.state import OkxStateStore
from indicator.okx.types import (
    Position, ReconciliationResult, ReconciliationVerdict,
)

logger = logging.getLogger(__name__)


# Tolerance for "size equal".  Sizes are now fractional (OKX lotSz 0.01),
# so use half a lot step to absorb float dust without masking real mismatches
# (e.g. the 0.29-vs-1.0 / 1-vs-12 kind still trips it).
SIZE_TOLERANCE_CONTRACTS = 0.005


class PositionReconciler:

    def __init__(self, client: OkxClient, store: OkxStateStore,
                 inst_id: str):
        self._client = client
        self._store = store
        self._inst_id = inst_id

    def reconcile_cycle(self) -> ReconciliationResult:
        """Run a single reconciliation pass.  Logs result; returns verdict."""
        local_open = self._store.get_open_position()
        try:
            okx_positions = self._client.get_positions(inst_id=self._inst_id)
        except Exception as e:
            logger.exception("reconciler_okx_query_failed")
            result = ReconciliationResult(
                verdict=ReconciliationVerdict.UNAVAILABLE,
                detail={"error": str(e)},
            )
            self._log(result)
            return result

        # Filter to non-flat OKX positions on our inst_id
        okx_active = [
            p for p in okx_positions
            if p.inst_id == self._inst_id and p.size_contracts != 0
        ]

        result = self._compare(local_open, okx_active)
        self._log(result)
        return result

    def _compare(self, local_open: Optional[dict],
                 okx_active: list[Position]) -> ReconciliationResult:
        # Both flat
        if local_open is None and not okx_active:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.CONSISTENT,
                detail={"state": "both_flat"},
            )

        # Local flat, OKX has position(s) → orphan_exchange
        if local_open is None and okx_active:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.MISMATCH,
                detail={
                    "type": "orphan_exchange",
                    "count": len(okx_active),
                    "okx": [
                        {"direction": p.direction,
                         "size": p.size_contracts,
                         "avg_price": p.avg_price}
                        for p in okx_active
                    ],
                },
            )

        # Local has position, OKX flat → orphan_local
        if local_open is not None and not okx_active:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.MISMATCH,
                detail={
                    "type": "orphan_local",
                    "local_id": local_open.get("id"),
                    "local_dir": local_open.get("direction"),
                    "local_size": local_open.get("size_contracts"),
                },
            )

        # Multiple OKX positions → unexpected (we only allow 1)
        if len(okx_active) > 1:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.MISMATCH,
                detail={
                    "type": "multiple_exchange_positions",
                    "count": len(okx_active),
                },
            )

        # Both have exactly one — compare
        okx = okx_active[0]
        local_size = float(local_open["size_contracts"] or 0)
        if abs(okx.size_contracts - local_size) > SIZE_TOLERANCE_CONTRACTS:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.MISMATCH,
                detail={
                    "type": "size_diff",
                    "local_size": local_size,
                    "okx_size": okx.size_contracts,
                },
            )

        if okx.direction != local_open["direction"]:
            return ReconciliationResult(
                verdict=ReconciliationVerdict.MISMATCH,
                detail={
                    "type": "direction_diff",
                    "local_dir": local_open["direction"],
                    "okx_dir": okx.direction,
                },
            )

        # Price diff is warning, not mismatch
        local_price = float(local_open.get("entry_price") or 0)
        if local_price > 0:
            diff_pct = abs(okx.avg_price - local_price) / local_price
            if diff_pct > 0.01:
                logger.warning("reconciler_price_drift local=%.2f okx=%.2f "
                               "diff=%.2f%%",
                               local_price, okx.avg_price, diff_pct * 100)

        return ReconciliationResult(
            verdict=ReconciliationVerdict.CONSISTENT,
            detail={"state": "both_one_position_match"},
        )

    def _log(self, result: ReconciliationResult) -> None:
        try:
            self._store.log_reconciliation(
                verdict=result.verdict.value,
                detail=result.detail,
            )
        except Exception:
            logger.exception("reconciler_log_failed")

    # ── Health/graduation surface ──────────────────────────────────────

    def get_consecutive_clean_days(self) -> int:
        return self._store.get_consecutive_clean_days()
