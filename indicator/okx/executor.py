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

from indicator.okx.alerter import format_kill_alert, send_critical
from indicator.okx.client import OkxClient
from indicator.okx.config import OkxConfig
from indicator.okx.kill_checks import (
    check_api_permissions, check_ntp_drift, run_all_checks,
)
from indicator.okx.reconciler import PositionReconciler
from indicator.okx.rest import make_cl_ord_id
from indicator.okx.state import OkxStateStore
from indicator.okx.types import (
    ExecutorStatus, KillCheckResult, KillSeverity, Position, Side,
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
        # NTP probe rate-limit: monotonic seconds of last probe.  0 = never.
        self._last_ntp_probe: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Bring executor from INIT → READY (if all checks pass)."""
        self._set_status(ExecutorStatus.CONNECTING, reason="start() called")
        # Start WS
        self._client.start_ws()
        # Wire WS callbacks to write into the store
        self._wire_ws_callbacks()

        # Safety belt #5: API key permission check
        # Best-effort: OKX /account/config doesn't always expose perm field
        # for main-account keys.  If we can't determine perms, log a warning
        # but DON'T block startup — the credential set itself was already
        # validated by validate_okx_config().
        try:
            acct_cfg = self._client.get_account_config() or {}
            data_list = acct_cfg.get("data") or []
            perm_str = ""
            if data_list:
                perm_str = (data_list[0].get("perm")
                            or data_list[0].get("permissions") or "")
            if perm_str:
                perms = [p.strip() for p in perm_str.split(",") if p.strip()]
                api_check = check_api_permissions(perms=perms)
                if api_check.triggered:
                    self._set_status(
                        ExecutorStatus.DEMOTED,
                        reason=f"api_perm_check_failed: {api_check.reason}",
                        trigger_id="E4",
                        context=api_check.context,
                    )
                    self._alert_critical(api_check, severity_label="DEMOTE")
                    return
            else:
                logger.warning(
                    "okx_api_perm_check_skipped_no_perm_field "
                    "verify withdraw=OFF manually in OKX UI"
                )
        except Exception:
            logger.exception("okx_api_perm_query_failed_continuing")

        # Safety belt #9: NTP drift check at boot
        ntp_check = self._probe_ntp(force=True)
        if ntp_check is not None and ntp_check.triggered:
            severity = ntp_check.severity
            self._set_status(
                ExecutorStatus.DEMOTED if severity == KillSeverity.DEMOTE
                else ExecutorStatus.HALTED,
                reason=f"ntp_at_start: {ntp_check.reason}",
                trigger_id=ntp_check.trigger_id,
                context=ntp_check.context,
            )
            self._alert_critical(ntp_check,
                                  severity_label=(severity.value
                                                  if severity else "?"))
            return

        # Cold-start reconciliation
        recon = self._reconciler.reconcile_cycle()
        from indicator.okx.types import ReconciliationVerdict
        if recon.verdict != ReconciliationVerdict.CONSISTENT:
            self._set_status(ExecutorStatus.HALTED,
                             reason="cold_start_reconciliation_failed",
                             trigger_id="A4",
                             context=recon.detail)
            return

        # Anchor day_start_equity with a snapshot if none for today
        self._snapshot_balance_if_missing()

        self._set_status(ExecutorStatus.READY, reason="cold_start_ok")
        self._set_status(ExecutorStatus.ACTIVE, reason="ready_to_trade")

    def _probe_ntp(self, *, force: bool = False) -> Optional[KillCheckResult]:
        """Query OKX server time, run check_ntp_drift.

        Returns None if the probe itself failed (network/parse) — caller
        treats absence as "no new evidence", not a trigger.  Rate-limited
        to cfg.ntp_check_interval_sec between probes unless force=True.
        """
        now = time.monotonic()
        if not force:
            interval = float(self._cfg.ntp_check_interval_sec)
            if now - self._last_ntp_probe < interval:
                return None
        self._last_ntp_probe = now
        server_ts = self._client.get_server_time()
        if server_ts is None:
            logger.warning("okx_ntp_probe_failed_skipping_check")
            return None
        local_ts = time.time()
        return check_ntp_drift(
            local_ts_sec=local_ts,
            server_ts_sec=server_ts,
            halt_threshold_sec=self._cfg.ntp_drift_halt_sec,
            demote_threshold_sec=self._cfg.ntp_drift_demote_sec,
        )

    def _snapshot_balance_if_missing(self) -> None:
        """Write a 'start' balance snapshot if none recorded yet today.

        Anchors get_day_start_equity().  Safe to skip if balance query
        fails — daily_loss_cap will silently default to no-anchor.
        """
        try:
            day_start = self._store.get_day_start_equity()
            latest = self._store.get_latest_balance() if day_start else None
            if day_start is not None and latest is not None:
                return
            balance = self._client.get_balance()
            if balance is None:
                return
            self._store.insert_balance_snapshot(
                total_eq_usd=balance.total_eq_usd,
                available_usd=balance.available_usd,
                source="start",
            )
        except Exception:
            logger.exception("okx_balance_snapshot_at_start_failed")

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
        local_position_dicts = self._store.get_all_open_positions() or []
        local_positions = self._dicts_to_positions(local_position_dicts)

        # Equity from latest balance snapshot (WS-pushed); day_start_equity
        # from pre-today snapshot.  Both fall back to initial_capital_usd
        # when no snapshot exists yet (boot day before first WS event).
        equity_usd = self._cfg.initial_capital_usd
        day_start_equity = self._cfg.initial_capital_usd
        try:
            latest = self._store.get_latest_balance()
            if latest and latest.get("total_eq_usd") is not None:
                equity_usd = float(latest["total_eq_usd"])
            day_start = self._store.get_day_start_equity()
            if day_start is not None:
                day_start_equity = day_start
            else:
                # No pre-today anchor yet — use current as the day's anchor
                day_start_equity = equity_usd
        except Exception:
            logger.exception("equity_lookup_failed_using_initial")

        # Periodic NTP probe (rate-limited inside _probe_ntp).
        # If the probe itself failed we skip the NTP check this cycle.
        ntp_result = self._probe_ntp(force=False)
        # run_all_checks expects either None (skip NTP) or (server_ts_sec).
        # We surface the already-computed trigger separately to preserve
        # the trigger_id (C5 vs C6) instead of re-running check_ntp_drift.
        triggered = run_all_checks(
            cfg=self._cfg, equity_usd=equity_usd,
            day_start_equity_usd=day_start_equity,
            local_positions=local_positions,
            reconciliation=recon,
            connectivity=connectivity,
            ntp_drift_sec=None,
        )
        if ntp_result is not None and ntp_result.triggered:
            triggered.append(ntp_result)

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

        # Telegram critical alert — severity carries from `worst`
        self._alert_critical(
            worst,
            severity_label=(worst.severity.value if worst.severity else "?"),
        )

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

        Used on DEMOTE.  Best-effort; each step is independently
        try/except'd so a failure in one position doesn't block the
        others.  We're already in a terminal state; can't make it worse.
        """
        try:
            rows = self._store.get_all_open_positions() or []
        except Exception:
            logger.exception("force_close_all_load_positions_failed")
            return

        for row in rows:
            pos_id = row.get("id")
            direction = row.get("direction") or "FLAT"
            size = int(row.get("size_contracts") or 0)
            algo_id = row.get("stop_algo_id")

            # Step 1: cancel algo stop (if we have its OKX id)
            if algo_id:
                try:
                    self._client.cancel_algo_stop(algo_id=str(algo_id))
                except Exception:
                    logger.exception("force_close_cancel_algo_failed pos=%s",
                                     pos_id)

            # Step 2: submit opposite-side market order to flatten
            if size > 0 and direction in ("LONG", "SHORT"):
                close_side = Side.SELL if direction == "LONG" else Side.BUY
                try:
                    self._client.submit_market_order(
                        inst_id=self._cfg.inst_id, side=close_side,
                        sz=size, td_mode=self._cfg.td_mode,
                        cl_ord_id=make_cl_ord_id(prefix="v7close"),
                    )
                except Exception:
                    logger.exception("force_close_market_close_failed pos=%s",
                                     pos_id)

            # Step 3: mark closed in local DB (best-effort)
            if pos_id is not None:
                try:
                    self._store.close_position(
                        position_id=int(pos_id),
                        exit_time=datetime.now(tz=timezone.utc),
                        exit_price=float(row.get("entry_price") or 0),
                        exit_reason="force_close",
                        gross_pct=0.0, net_pct=0.0,
                        equity_ret_pct=0.0,
                        equity_after=float(row.get("equity_before") or 0),
                        new_status="DEMOTED",
                    )
                except Exception:
                    logger.exception("force_close_db_update_failed pos=%s",
                                     pos_id)

        logger.warning("force_close_all_complete n=%d", len(rows))

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
            # On fill, map OKX ord_id back into v7_okx_positions row by
            # cl_ord_id.  Full position open/close lifecycle is owned by
            # _open_position/_manage_position (next iteration).
            logger.info("ws_order_event cl_ord=%s state=%s fill_px=%s",
                        evt.cl_ord_id, evt.state, evt.fill_price)
            try:
                pos = self._store.get_open_position()
                if not pos:
                    return
                pos_id = pos.get("id")
                if pos_id is None:
                    return
                # Match by cl_ord_id to avoid cross-row contamination
                if pos.get("entry_cl_ord_id") == evt.cl_ord_id and evt.ord_id:
                    self._store.set_position_okx_ids(
                        position_id=int(pos_id),
                        entry_ord_id=evt.ord_id,
                    )
            except Exception:
                logger.exception("on_order_ws_persist_failed")

        def on_position(evt: PositionEvent) -> None:
            # We don't auto-mutate local from this.  Reconciler does
            # the comparison on the next cycle (P2 + P4).
            logger.debug("ws_position_event pos=%s avg=%.2f",
                         evt.pos, evt.avg_price)

        def on_balance(evt: BalanceEvent) -> None:
            # Persist for daily_loss_cap check; insert is small (single
            # row), runs on WS thread but doesn't block reads.
            try:
                self._store.insert_balance_snapshot(
                    total_eq_usd=evt.total_eq_usd,
                    available_usd=evt.available_usd,
                    source="ws",
                )
            except Exception:
                logger.exception("on_balance_ws_persist_failed")

        self._client.subscribe_orders(on_order)
        self._client.subscribe_positions(on_position)
        self._client.subscribe_balance(on_balance)

    # ── Internal helpers ──────────────────────────────────────────────

    def _dicts_to_positions(self,
                             rows: list[dict]) -> list[Position]:
        """Convert v7_okx_positions row dicts → typed Position objects.

        Used only as input to check_max_position; we drop columns we
        don't need.  Direction is stored as the human label so no
        sign-inference needed.
        """
        out: list[Position] = []
        for r in rows:
            try:
                out.append(Position(
                    inst_id=self._cfg.inst_id,
                    direction=r.get("direction") or "FLAT",
                    size_contracts=int(r.get("size_contracts") or 0),
                    avg_price=float(r.get("entry_price") or 0),
                    raw=r,
                ))
            except Exception:
                logger.exception("dicts_to_positions_row_failed row_id=%s",
                                 r.get("id"))
        return out

    def _alert_critical(self, check: KillCheckResult, *,
                         severity_label: str) -> None:
        """Send Telegram critical alert.  Never raises."""
        try:
            msg = format_kill_alert(
                trigger_id=check.trigger_id or "?",
                severity=severity_label,
                reason=check.reason,
                stage_label=self._cfg.stage_label,
                context=check.context,
            )
            send_critical(self._cfg.telegram_critical_chat_id, msg)
        except Exception:
            logger.exception("alert_critical_send_failed")
