"""V7OkxExecutor — Stage 2 testnet / Stage 3 live executor.

Mirrors V7PaperExecutor.cycle() logic, but talks to OKX through OkxClient.

Per okx_integration_design.md P8: paper executor keeps running in parallel
as the baseline.  This executor does NOT modify paper; ExecutorRouter
invokes both.

State machine — see types.ExecutorStatus.
Kill triggers — see kill_checks.py, mapped to docs/stage2_kill_criteria.md.

Skeleton status (2026-05-25):
  - Cycle skeleton: lays out the steps + guards + kill_check integration
  - Real order placement / amend / close: marked TODO until OKX API key
    available
  - All connection-resilience pieces (REST retry, WS reconnect, kill
    checks) are real implementations
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from indicator.okx.client import OkxClient
from indicator.okx.config import OkxConfig
from indicator.okx.kill_checks import run_all_checks
from indicator.okx.reconciler import PositionReconciler
from indicator.okx.rest import make_cl_ord_id
from indicator.okx.state import OkxStateStore
from indicator.okx.types import (
    ExecutorStatus, KillCheckResult, KillSeverity, Side,
)

logger = logging.getLogger(__name__)


# ── Strategy constants (mirror v7_paper_executor.py) ──────────────────

ATR_PERIOD = 14
TRAIL_MULT = 3.0
TIME_CAP_HOURS = 72


@dataclass
class CycleResult:
    action: str                  # open / close / hold / halted / demoted / none
    detail: Optional[dict] = None


class V7OkxExecutor:

    def __init__(self, *, client: OkxClient, store: OkxStateStore,
                 reconciler: PositionReconciler, cfg: OkxConfig):
        self._client = client
        self._store = store
        self._reconciler = reconciler
        self._cfg = cfg
        # In-memory status cache; persisted via store on every change.
        self._status: ExecutorStatus = ExecutorStatus.INIT

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Bring executor from INIT → READY (if all checks pass)."""
        self._set_status(ExecutorStatus.CONNECTING, reason="start() called")
        # Start WS
        self._client.start_ws()
        # Wire WS callbacks to write into the store
        self._wire_ws_callbacks()
        # Validate API key perms (safety belt #5)
        # TODO(stage2-impl): query OKX, run check_api_permissions
        # NTP check (safety belt #9)
        # TODO(stage2-impl): query get_server_time, compute drift
        # Cold-start reconciliation
        recon = self._reconciler.reconcile_cycle()
        from indicator.okx.types import ReconciliationVerdict
        if recon.verdict != ReconciliationVerdict.CONSISTENT:
            self._set_status(ExecutorStatus.HALTED,
                             reason="cold_start_reconciliation_failed",
                             trigger_id="A4",
                             context=recon.detail)
            return
        self._set_status(ExecutorStatus.READY, reason="cold_start_ok")
        self._set_status(ExecutorStatus.ACTIVE, reason="ready_to_trade")

    def stop(self) -> None:
        self._client.close()
        self._set_status(ExecutorStatus.HALTED, reason="stop() called")

    # ── Per-cycle entry ────────────────────────────────────────────────

    def cycle(self, *, klines: pd.DataFrame, signal_direction: str,
              signal_strength: str,
              model_version: Optional[str] = None) -> CycleResult:
        """One cycle on the latest closed bar.

        Steps:
          1. Status guard
          2. Reconciliation guard
          3. Kill checks aggregate
          4. If open position: manage exit / amend trailing stop
          5. If flat + actionable signal: submit entry + algo stop
        """
        # 1. Status guard
        if self._status in (ExecutorStatus.INIT, ExecutorStatus.CONNECTING,
                            ExecutorStatus.DEMOTED):
            return CycleResult(action="none",
                               detail={"status": self._status.value})

        # 2. Pre-cycle reconciliation (every cycle, per design P2)
        recon = self._reconciler.reconcile_cycle()

        # 3. Kill checks aggregate
        connectivity = self._client.connectivity()
        local_positions = self._store.get_all_open_positions()
        # Compute equity from balance (stub for now)
        # TODO(stage2-impl): use real balance & track day_start_equity
        equity_usd = self._cfg.initial_capital_usd
        day_start_equity = self._cfg.initial_capital_usd

        # NTP drift check uses absolute diff; compute lazily
        ntp_drift: Optional[float] = None
        # TODO(stage2-impl): periodic NTP probe (every cfg.ntp_check_interval_sec)

        triggered = run_all_checks(
            cfg=self._cfg, equity_usd=equity_usd,
            day_start_equity_usd=day_start_equity,
            local_positions=[],   # TODO: convert dicts to Position
            reconciliation=recon,
            connectivity=connectivity,
            ntp_drift_sec=ntp_drift,
        )

        if triggered:
            self._handle_triggers(triggered)
            return CycleResult(action="halted" if self._status == ExecutorStatus.HALTED
                              else "demoted",
                              detail={"triggers":
                                      [t.trigger_id for t in triggered]})

        # If we're in HALTED state but no triggers now → auto-resume to ACTIVE
        if self._status == ExecutorStatus.HALTED:
            self._set_status(ExecutorStatus.ACTIVE, reason="triggers_cleared")

        # 4. Manage existing position
        pos = self._store.get_open_position()
        if pos:
            return self._manage_position(pos, klines=klines,
                                         signal_direction=signal_direction)

        # 5. Open new position if signal valid
        if signal_direction in ("UP", "DOWN"):
            return self._open_position(klines=klines,
                                       signal_direction=signal_direction,
                                       signal_strength=signal_strength,
                                       model_version=model_version)

        return CycleResult(action="none", detail={"reason": "no_signal_no_pos"})

    # ── Cycle steps ────────────────────────────────────────────────────

    def _manage_position(self, pos: dict, *, klines: pd.DataFrame,
                         signal_direction: str) -> CycleResult:
        """Existing-position branch.

        Mirrors v7_paper_executor logic:
          - trail extreme update each bar
          - amend algo stop to new trigger price
          - exit on time_cap (72h)
          - exit on opposite signal
        """
        # TODO(stage2-impl): compute new trailing extreme + amend algo stop
        # via self._client.amend_algo_stop(...)
        # If opposite signal or time cap → market close + cancel algo stop
        return CycleResult(action="hold",
                           detail={"reason": "manage_not_implemented"})

    def _open_position(self, *, klines: pd.DataFrame,
                       signal_direction: str, signal_strength: str,
                       model_version: Optional[str]) -> CycleResult:
        """Flat → open position branch.

        Steps (must be in this order; B4 / safety belt #7 depends on it):
          1. Compute ATR, stop_dist, size_contracts
          2. Submit market entry with clOrdId
          3. Wait for fill confirmation (REST poll fallback if WS slow)
          4. Submit algo stop (must be within 5s of fill)
          5. Insert row into v7_okx_positions
          6. Push Telegram entry alert (safety belt #8)
        """
        # TODO(stage2-impl): full implementation
        # For now log intent and return
        logger.info("okx_open_position_skeleton dir=%s str=%s",
                    signal_direction, signal_strength)
        return CycleResult(action="none",
                           detail={"reason": "open_not_implemented"})

    # ── Kill-trigger handling ──────────────────────────────────────────

    def _handle_triggers(self, triggered: list[KillCheckResult]) -> None:
        """Apply the most severe trigger's action.

        Priority: HARD_FREEZE > DEMOTE > HALT.
        """
        severity_rank = {
            KillSeverity.HARD_FREEZE: 3,
            KillSeverity.DEMOTE: 2,
            KillSeverity.HALT: 1,
        }
        triggered_sorted = sorted(
            triggered,
            key=lambda r: severity_rank.get(r.severity, 0),
            reverse=True,
        )
        worst = triggered_sorted[0]

        # Log every triggered to kill_log
        for t in triggered:
            try:
                self._store.log_kill_trigger(
                    trigger_id=t.trigger_id or "?",
                    severity=(t.severity.value if t.severity else "?"),
                    context={**t.context, "reason": t.reason},
                )
            except Exception:
                logger.exception("log_kill_trigger_failed")

        # TODO(stage2-impl): Telegram critical alert with worst.trigger_id

        if worst.severity == KillSeverity.HARD_FREEZE:
            # Pause paper too — coordinated via ExecutorRouter
            self._set_status(ExecutorStatus.DEMOTED,
                             reason=f"hard_freeze: {worst.reason}",
                             trigger_id=worst.trigger_id,
                             context=worst.context)
            self._force_close_all()

        elif worst.severity == KillSeverity.DEMOTE:
            self._set_status(ExecutorStatus.DEMOTED,
                             reason=f"demote: {worst.reason}",
                             trigger_id=worst.trigger_id,
                             context=worst.context)
            self._force_close_all()

        elif worst.severity == KillSeverity.HALT:
            self._set_status(ExecutorStatus.HALTED,
                             reason=f"halt: {worst.reason}",
                             trigger_id=worst.trigger_id,
                             context=worst.context)
            # HALT keeps positions; algo stop continues to protect

    def _force_close_all(self) -> None:
        """Cancel all algo orders + market-close all positions.

        Used on DEMOTE.  Best-effort; logs failures.
        """
        # TODO(stage2-impl): iterate open positions, cancel stop algo,
        # submit opposite-side market order, update DB.
        logger.warning("force_close_all_skeleton_not_implemented")

    # ── Status mgmt ────────────────────────────────────────────────────

    def _set_status(self, status: ExecutorStatus, *, reason: str,
                    trigger_id: Optional[str] = None,
                    context: Optional[dict] = None) -> None:
        prev = self._status
        self._status = status
        logger.info("okx_executor_status %s -> %s reason=%s trigger=%s",
                    prev.value, status.value, reason, trigger_id)
        try:
            self._store.save_executor_status(
                status=status.value, reason=reason,
                trigger_id=trigger_id, context=context,
            )
        except Exception:
            logger.exception("save_executor_status_failed")

    def get_status(self) -> ExecutorStatus:
        return self._status

    # ── WS callback wiring ─────────────────────────────────────────────

    def _wire_ws_callbacks(self) -> None:
        """Wire WS event callbacks to store-level persistence.

        Callbacks run on WS thread — do NOT mutate in-memory state here.
        Only persist via store; cycle reads from store on next tick.
        """
        from indicator.okx.types import OrderEvent, PositionEvent, BalanceEvent

        def on_order(evt: OrderEvent) -> None:
            # TODO(stage2-impl): on filled, update v7_okx_positions
            # entry_ord_id and (if stop) algo_id mappings.
            logger.info("ws_order_event cl_ord=%s state=%s",
                        evt.cl_ord_id, evt.state)

        def on_position(evt: PositionEvent) -> None:
            # We don't auto-mutate local from this.  Reconciler does
            # the comparison on the next cycle (P2 + P4).
            logger.debug("ws_position_event pos=%s avg=%.2f",
                         evt.pos, evt.avg_price)

        def on_balance(evt: BalanceEvent) -> None:
            # Used by daily_loss_cap check; persist latest
            # TODO(stage2-impl): write balance snapshot to a separate table
            logger.debug("ws_balance_event eq=%.2f", evt.total_eq_usd)

        self._client.subscribe_orders(on_order)
        self._client.subscribe_positions(on_position)
        self._client.subscribe_balance(on_balance)
