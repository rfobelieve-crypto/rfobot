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

import numpy as np

from indicator.okx.alerter import (
    format_entry_alert, format_exit_alert, format_kill_alert, send_critical,
)
from indicator.okx.approval import ApprovalGate, TradeIntent
from indicator.okx.client import OkxClient
from indicator.okx.config import OkxConfig
from indicator.okx.kill_checks import (
    check_algo_stop_latency, check_api_permissions, check_ntp_drift,
    check_presubmit_order, run_all_checks,
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
TIME_CAP_HOURS = 72        # canonical re-enable value; live cap = cfg.time_cap_hours (0=off)
RISK_FRAC = 0.02            # 2% of equity per trade (legacy; unused by B sizing)
MAX_LEVERAGE = 1.0

# ── Fixed-notional sizing "B" (2026-06-06) ────────────────────────────
# Position notional = NOTIONAL_LEV_MULT × equity, rounded to OKX's 0.01-contract
# lot step.  Replaces the old 2%-risk × leverage formula whose int()-floor to
# WHOLE contracts forced ~7x leverage on a small account (the 2026-06-05 setup).
# Scales with equity (no leverage creep).  Sim (169-trade WF + injected -10% gap):
# 2x survives the gap (-20% on it); 10x = ruin.  LOT_STEP/MIN_SZ verified via
# OKX public instruments API (BTC-USDT-SWAP lotSz=minSz=0.01).
NOTIONAL_LEV_MULT = 2.0
LOT_STEP = 0.01
MIN_SZ = 0.01


def _atr_wilder(klines, period: int = ATR_PERIOD) -> float:
    """Wilder ATR of the latest bar (matches Pine ta.atr / RMA of TR).

    Mirrors v7_paper_executor._atr_wilder so paper/live sizes match.
    """
    import pandas as _pd
    high = klines["high"].astype(float)
    low = klines["low"].astype(float)
    close = klines["close"].astype(float)
    pc = close.shift(1)
    tr = _pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                    axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False,
                 min_periods=period).mean()
    val = atr.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


@dataclass
class CycleResult:
    action: str                  # open / close / hold / halted / demoted / none
    detail: Optional[dict] = None


class V7OkxExecutor:

    def __init__(self, *, client: OkxClient, store: OkxStateStore,
                 reconciler: PositionReconciler, cfg: OkxConfig,
                 approval_gate: Optional[ApprovalGate] = None):
        self._client = client
        self._store = store
        self._reconciler = reconciler
        self._cfg = cfg
        # Optional: when set, _open_position routes through approval gate
        # until N successful manual approvals have accumulated.
        self._approval = approval_gate
        # In-memory status cache; persisted via store on every change.
        self._status: ExecutorStatus = ExecutorStatus.INIT
        # NTP probe rate-limit: monotonic seconds of last probe.  0 = never.
        self._last_ntp_probe: float = 0.0
        # Orphan-local auto-heal: track consecutive reconcile cycles where
        # DB has OPEN but OKX is flat.  After AUTO_HEAL_AFTER_CYCLES, we
        # auto-close DB to clear the HALT loop (WS algo-fill events are
        # known to miss; reconciler is the safety net).  Reset on any
        # consistent or non-orphan_local result.
        self._orphan_local_streak: int = 0

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

        # 2b. Auto-heal persistent orphan_local.
        # WS algo-fill events (OKX trail stop firing) sometimes arrive
        # late or get dropped during brief WS hiccups, leaving DB OPEN
        # while OKX is actually flat. Without this, each occurrence
        # halts the executor and requires manual /okx-admin/heal.
        # Logic: after 2 consecutive cycles of orphan_local (both via
        # REST query in reconciler), auto-close the DB row(s) and
        # override recon to CONSISTENT so kill_checks doesn't HALT.
        # Only acts when OKX is genuinely flat — orphan_exchange /
        # size_diff / direction_diff still HALT (those are unsafe).
        recon = self._maybe_auto_heal_orphan_local(recon)

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
            # Mark prior unresolved kill_log entries as resolved
            # (M6 milestone + dashboard 'Recovery History' reflect this)
            try:
                n_resolved = self._store.resolve_open_kills(
                    resolution="auto_recovery_to_active",
                )
                if n_resolved > 0:
                    logger.info("okx_kill_log_auto_resolved n=%d", n_resolved)
            except Exception:
                logger.exception("kill_log_auto_resolve_failed")

        # 4. Manage existing position
        pos = self._store.get_open_position()
        if pos:
            return self._manage_position(pos, klines=klines,
                                         signal_direction=signal_direction)

        # 5. Open new position if signal valid. (Paper cohort removed
        # 2026-06-05 — LIVE OKX is now the sole cohort, so there is no
        # paper-sync gate; OKX opens directly on a Strong/Moderate signal.)
        if signal_direction in ("UP", "DOWN"):
            # Strong-only entry gate (reversible: OKX_STRONG_ONLY_ENTRY).
            # Moderate signals do NOT take the single slot — they crowd out
            # higher-WR Strong entries and roughly double MaxDD at 2x (see
            # research/dual_model/entry_policy_real_exit_bt).  Managing an
            # already-open Moderate position is unaffected (handled above).
            if self._cfg.strong_only_entry and signal_strength != "Strong":
                logger.info("strong_only_entry: skip %s %s signal",
                            signal_strength, signal_direction)
                return CycleResult(
                    action="none",
                    detail={"reason": "moderate_skipped_strong_only",
                            "tier": signal_strength,
                            "direction": signal_direction})
            return self._open_position(klines=klines,
                                       signal_direction=signal_direction,
                                       signal_strength=signal_strength,
                                       model_version=model_version)

        return CycleResult(action="none", detail={"reason": "no_signal_no_pos"})

    # ── Auto-heal ──────────────────────────────────────────────────────

    # Wait this many consecutive cycles before auto-healing. 2 = give the
    # late WS event one extra cycle to arrive before we mutate state.
    AUTO_HEAL_AFTER_CYCLES: int = 2

    def _maybe_auto_heal_orphan_local(self, recon):
        """If orphan_local persists for N cycles, auto-close DB rows.

        Returns: the recon result, possibly overridden to CONSISTENT
        after a successful heal so kill_checks doesn't fire A4 HALT.

        Why this is safe: orphan_local means OKX is FLAT (reconciler
        queried REST). Closing the DB row aligns our state to truth.
        No risk of opening a hidden naked position.

        Why this is NOT done immediately: WS events sometimes arrive
        late (network jitter). Waiting one extra cycle gives a real
        algo-fill event time to land via the normal path (which
        produces accurate P&L from fill_price). Auto-heal is the
        last-resort safety net with zeroed P&L.
        """
        from indicator.okx.types import (
            ReconciliationResult, ReconciliationVerdict,
        )

        if recon.verdict != ReconciliationVerdict.MISMATCH:
            self._orphan_local_streak = 0
            return recon
        if (recon.detail or {}).get("type") != "orphan_local":
            # Other mismatch types (orphan_exchange / size_diff /
            # direction_diff) are NOT safe to auto-heal — keep HALT path.
            self._orphan_local_streak = 0
            return recon

        self._orphan_local_streak += 1
        if self._orphan_local_streak < self.AUTO_HEAL_AFTER_CYCLES:
            logger.warning(
                "orphan_local streak=%d (will auto-heal at %d); "
                "this cycle still HALTs",
                self._orphan_local_streak, self.AUTO_HEAL_AFTER_CYCLES,
            )
            return recon  # let A4 fire this cycle

        # Persistent → auto-heal
        logger.warning(
            "orphan_local streak=%d ≥ %d → AUTO-HEALING",
            self._orphan_local_streak, self.AUTO_HEAL_AFTER_CYCLES,
        )
        healed_n = self._auto_heal_orphan_local()
        self._orphan_local_streak = 0
        return ReconciliationResult(
            verdict=ReconciliationVerdict.CONSISTENT,
            detail={"auto_healed_from_orphan_local": True,
                    "healed_n": healed_n},
        )

    def _auto_heal_orphan_local(self) -> int:
        """Close all DB OPEN rows + push informational Telegram.

        Returns count of rows actually closed.
        """
        from datetime import datetime as _dt
        rows = self._store.get_all_open_positions() or []
        n = 0
        for row in rows:
            pos_id = int(row.get("id") or 0)
            try:
                # Use entry price as exit (zeroed P&L). OKX cash balance
                # is authoritative for true equity; this just clears the
                # DB row so reconciler returns CONSISTENT.
                fallback_exit = float(
                    row.get("current_stop") or row.get("entry_price") or 0
                )
                self._store.close_position(
                    position_id=pos_id,
                    exit_time=_dt.utcnow(),
                    exit_price=fallback_exit,
                    exit_reason="auto_heal_orphan_local",
                    gross_pct=0.0, net_pct=0.0,
                    equity_ret_pct=0.0,
                    equity_after=float(row.get("equity_before") or 0),
                    new_status="CLOSED",
                )
                n += 1
            except Exception:
                logger.exception("auto_heal_close_failed pos=%s", pos_id)
        if n > 0:
            try:
                send_critical(
                    self._cfg.telegram_critical_chat_id,
                    f"🟡 <b>AUTO-HEAL orphan_local</b>\n"
                    f"OKX confirmed flat for "
                    f"{self.AUTO_HEAL_AFTER_CYCLES} cycles.\n"
                    f"Closed {n} stuck DB row(s) — executor continues.\n"
                    f"Likely cause: WS algo-fill event missed.\n"
                    f"True P&L: see OKX wallet (DB row's P&L zeroed).",
                )
            except Exception:
                logger.exception("auto_heal_alert_failed")
        return n

    # ── Cycle steps ────────────────────────────────────────────────────

    def _manage_position(self, pos: dict, *, klines: pd.DataFrame,
                         signal_direction: str) -> CycleResult:
        """Existing-position branch.

        Mirrors v7_paper_executor exit/trail logic, but the *intrabar
        stop-out* itself is handled by OKX (the algo stop placed at
        entry).  We only need to:
          - ratchet the trailing extreme bar-by-bar + amend the algo
            stop trigger to the new level
          - manually close on opposite signal (or time_cap if cfg enables it)
            (those are conditions OKX doesn't know about)

        Algo-stop fills arrive via WS `orders` event and reconciliation;
        we don't poll for them here.
        """
        if klines is None or klines.empty:
            return CycleResult(action="hold",
                               detail={"reason": "no_klines"})

        side = pos.get("direction") or "FLAT"
        if side not in ("LONG", "SHORT"):
            return CycleResult(action="hold",
                               detail={"reason": "bad_side",
                                       "side": side})

        bar_ts = klines.index[-1]
        if getattr(bar_ts, "tzinfo", None) is not None:
            bar_ts_utc = bar_ts.tz_convert("UTC")
        else:
            bar_ts_utc = bar_ts
        bar_ts_naive = (bar_ts_utc.tz_localize(None)
                        if getattr(bar_ts_utc, "tzinfo", None) is not None
                        else bar_ts_utc)
        bar_high = float(klines["high"].iloc[-1])
        bar_low = float(klines["low"].iloc[-1])
        last_close = float(klines["close"].iloc[-1])

        # bars_held — entry_time stored naive UTC by store layer
        entry_ts = pos.get("entry_time")
        if isinstance(entry_ts, str):
            entry_ts = pd.Timestamp(entry_ts)
        if entry_ts is not None and getattr(entry_ts, "tzinfo", None) is not None:
            entry_ts = entry_ts.tz_convert("UTC").tz_localize(None)
        bars_held = (max(0, int((bar_ts_naive - entry_ts).total_seconds() / 3600))
                     if entry_ts is not None else 0)

        stop_dist = float(pos.get("stop_dist") or 0)
        prev_extreme = float(pos.get("trail_extreme")
                             or pos.get("entry_price") or 0)

        # Compute new extreme + new stop
        if side == "LONG":
            new_extreme = max(prev_extreme, bar_high)
            new_stop = new_extreme - stop_dist
        else:
            new_extreme = (min(prev_extreme, bar_low)
                           if prev_extreme > 0 else bar_low)
            new_stop = new_extreme + stop_dist

        # Manual exits — OKX doesn't know about time_cap / opp_signal
        # time_cap_hours=0 disables the cap (removed 2026-06-10) → exits come
        # only from the trailing stop or an opposite signal.
        exit_reason: Optional[str] = None
        if (self._cfg.time_cap_hours > 0
                and bars_held >= self._cfg.time_cap_hours):
            exit_reason = "time_cap"
        elif side == "LONG" and signal_direction == "DOWN":
            exit_reason = "opp_signal"
        elif side == "SHORT" and signal_direction == "UP":
            exit_reason = "opp_signal"

        if exit_reason is not None:
            return self._close_position(pos, exit_price=last_close,
                                         exit_reason=exit_reason,
                                         bar_ts=bar_ts_naive)

        # No exit — ratchet trail if extreme advanced
        if new_extreme != prev_extreme:
            amended_ok = False
            try:
                res = self._client.amend_algo_stop(
                    inst_id=self._cfg.inst_id,
                    algo_id=str(pos.get("stop_algo_id") or ""),
                    new_trigger_px=new_stop,
                )
                amended_ok = getattr(res, "status", None) == "ok"
                if not amended_ok:
                    logger.warning("amend_algo_stop_not_ok pos=%s err=%s",
                                   pos.get("id"), getattr(res, "error", res))
            except Exception:
                logger.exception("amend_algo_stop_failed pos=%s",
                                 pos.get("id"))
            # Only advance the DB trail if OKX actually moved the stop — else the
            # DB would claim a protection level the exchange does not have (the
            # 2026-06-10 missing-instId bug: DB trailed to 62560 while OKX stayed
            # at the 60181 entry stop). Retry on the next cycle if it failed.
            if amended_ok:
                try:
                    self._store.update_trail(
                        position_id=int(pos["id"]),
                        trail_extreme=new_extreme,
                        current_stop=new_stop,
                    )
                except Exception:
                    logger.exception("update_trail_db_failed pos=%s",
                                     pos.get("id"))

        return CycleResult(action="hold",
                           detail={"position_id": pos.get("id"),
                                   "direction": side,
                                   "bars_held": bars_held,
                                   "current_stop": new_stop})

    def _close_position(self, pos: dict, *, exit_price: float,
                         exit_reason: str,
                         bar_ts) -> CycleResult:
        """Cancel algo + market close + DB close + Telegram alert.

        Used for manual exits (time_cap, opp_signal) — algo-stop fills
        are recorded by the WS callback / reconciler instead.
        """
        pos_id = int(pos.get("id") or 0)
        side = pos.get("direction") or "FLAT"
        size = float(pos.get("size_contracts") or 0)
        algo_id = pos.get("stop_algo_id")
        entry_price = float(pos.get("entry_price") or 0)
        equity_before = float(pos.get("equity_before") or 0)
        size_frac = float(pos.get("size_frac") or 0)

        # 1. Cancel algo stop (best-effort)
        if algo_id:
            try:
                self._client.cancel_algo_stop(algo_id=str(algo_id))
            except Exception:
                logger.exception("close_cancel_algo_failed pos=%s", pos_id)

        # 2. Submit opposite-side market order to flatten.
        # Capture the result — unlike open path which checks rejection,
        # the prior close path discarded it.  A 4xx from OKX returns
        # status="rejected" WITHOUT raising, so the try/except below
        # would not catch it.  Failing to detect this caused the
        # "V7 says closed, OKX still has the position" bug.
        close_submit_failed = False
        close_error: Optional[str] = None
        if size > 0 and side in ("LONG", "SHORT"):
            close_side = Side.SELL if side == "LONG" else Side.BUY
            try:
                close_result = self._client.submit_market_order(
                    inst_id=self._cfg.inst_id, side=close_side,
                    sz=size, td_mode=self._cfg.td_mode,
                    cl_ord_id=make_cl_ord_id(prefix="v7close"),
                    pos_side=self._cfg.pos_side_for(side),
                    reduce_only=True,
                )
                if close_result.status == "rejected":
                    close_submit_failed = True
                    close_error = close_result.error or "okx_rejected"
                    logger.error(
                        "close_market_order_rejected pos=%s side=%s "
                        "size=%d posSide=%s err=%s",
                        pos_id, close_side.value, size,
                        self._cfg.pos_side_for(side), close_error,
                    )
            except Exception as exc:
                close_submit_failed = True
                close_error = f"exception: {exc}"
                logger.exception("close_market_order_failed pos=%s", pos_id)

        # If close submission failed, leave DB state OPEN so the next
        # cycle retries.  Cancel the algo-stop cancel we did above —
        # actually re-place it so the position keeps protection.
        if close_submit_failed:
            logger.error(
                "okx_close_FAILED pos=%s side=%s — OKX position likely "
                "still open. DB left as OPEN; next cycle will retry. "
                "Halting executor to prevent further state divergence.",
                pos_id, side,
            )
            # Try to re-place the algo stop we just canceled, otherwise
            # the position is now naked.
            try:
                algo_side = Side.SELL if side == "LONG" else Side.BUY
                self._client.submit_algo_stop(
                    inst_id=self._cfg.inst_id, side=algo_side, sz=size,
                    trigger_px=float(pos.get("current_stop") or 0),
                    td_mode=self._cfg.td_mode,
                    pos_side=self._cfg.pos_side_for(side),
                    reduce_only=True,
                )
            except Exception:
                logger.exception("close_failed_algo_replace_also_failed "
                                 "pos=%s — position now NAKED", pos_id)
            # Critical alert + halt
            try:
                send_critical(
                    self._cfg.telegram_critical_chat_id,
                    f"🔴 <b>OKX CLOSE FAILED</b> pos id={pos_id}\n"
                    f"side={side} size={size} reason={exit_reason}\n"
                    f"err: {close_error}\n"
                    f"Position likely still open on OKX. Executor halted; "
                    f"will retry next cycle. Manually verify OKX state.",
                )
            except Exception:
                logger.exception("close_failed_alert_send_also_failed")
            self._set_status(
                ExecutorStatus.HALTED,
                reason=f"close_market_order_failed: {close_error}",
                trigger_id="CLOSE-FAIL",
                context={"pos_id": pos_id, "side": side,
                         "exit_reason": exit_reason},
            )
            return CycleResult(action="close_failed",
                               detail={"position_id": pos_id,
                                       "error": close_error})

        # 3. Compute P&L using last_close as the exit-price estimate.
        # WS fill event for the close order can refine this later.
        if side == "LONG" and entry_price > 0:
            gross_pct = exit_price / entry_price - 1.0
        elif side == "SHORT" and entry_price > 0:
            gross_pct = -(exit_price / entry_price - 1.0)
        else:
            gross_pct = 0.0
        net_pct = gross_pct - self._cfg.taker_cost
        # implied leverage = notional / equity_before; with 10x cfg the
        # equity move is 10x the gross trade %.  Previously this missed
        # the leverage factor and under-recorded equity changes by 10x.
        notional = float(pos.get("notional_usd") or 0)
        if equity_before > 0 and notional > 0:
            equity_ret = (notional / equity_before) * net_pct
        else:
            equity_ret = size_frac * net_pct   # fallback for very old rows
        equity_after = max(equity_before * (1.0 + equity_ret), 0.0)

        try:
            self._store.close_position(
                position_id=pos_id,
                exit_time=bar_ts,
                exit_price=exit_price,
                exit_reason=exit_reason,
                gross_pct=gross_pct, net_pct=net_pct,
                equity_ret_pct=equity_ret * 100.0,
                equity_after=equity_after,
                new_status="CLOSED",
            )
        except Exception:
            logger.exception("close_position_db_failed pos=%s", pos_id)

        # 4. Telegram exit alert
        try:
            msg = format_exit_alert(
                stage_label=self._cfg.stage_label, direction=side,
                reason=exit_reason, entry_price=entry_price,
                exit_price=exit_price, gross_pct=gross_pct,
                net_pct=net_pct, equity_after=equity_after,
            )
            send_critical(self._cfg.telegram_critical_chat_id, msg)
        except Exception:
            logger.exception("close_telegram_alert_failed")

        logger.info("okx_close id=%d %s %.2f→%.2f net %+.3f%% reason=%s",
                    pos_id, side, entry_price, exit_price,
                    net_pct * 100, exit_reason)

        return CycleResult(action="close",
                           detail={"position_id": pos_id, "side": side,
                                   "exit_price": exit_price,
                                   "exit_reason": exit_reason,
                                   "gross_pct": gross_pct,
                                   "net_pct": net_pct,
                                   "equity_after": equity_after})

    def _open_position(self, *, klines: pd.DataFrame,
                       signal_direction: str, signal_strength: str,
                       model_version: Optional[str]) -> CycleResult:
        """Flat → open position branch.

        Routes through approval_gate if attached:
          - Auto mode (5+ manual approvals): execute immediately
          - Manual mode: build intent, push Telegram, return pending
        """
        intent = self._build_intent(
            klines=klines, signal_direction=signal_direction,
            signal_strength=signal_strength, model_version=model_version,
        )
        if isinstance(intent, CycleResult):
            return intent   # short-circuit from validation failure

        # Approval gate routing
        if self._approval is not None and not self._approval.is_auto_mode():
            approval_id = self._approval.request_approval(intent)
            if approval_id is None:
                return CycleResult(action="none",
                                   detail={"reason": "approval_request_failed",
                                           "intent": intent.direction})
            return CycleResult(
                action="pending_approval",
                detail={"approval_id": approval_id,
                        "direction": intent.direction,
                        "tier": intent.tier,
                        "size_contracts": intent.size_contracts,
                        "entry_price": intent.entry_price},
            )

        # Auto mode (or no gate) → execute now
        return self.execute_approved_intent(intent, approval_id=None)

    def _build_intent(self, *, klines: pd.DataFrame, signal_direction: str,
                       signal_strength: str,
                       model_version: Optional[str]):
        """Pure pre-trade computation: ATR / sizing / contract floor.

        Returns a TradeIntent on success, or a CycleResult(action='none')
        on validation failure (insufficient klines, bad ATR, below min lot).
        """
        if klines is None or klines.empty or len(klines) < ATR_PERIOD + 2:
            return CycleResult(action="none",
                               detail={"reason": "insufficient_klines"})

        side = "LONG" if signal_direction == "UP" else "SHORT"
        bar_ts = klines.index[-1]
        if getattr(bar_ts, "tzinfo", None) is not None:
            bar_ts = bar_ts.tz_convert("UTC").tz_localize(None)
        last_close = float(klines["close"].iloc[-1])

        atr = _atr_wilder(klines, ATR_PERIOD)
        if atr <= 0 or not np.isfinite(atr):
            return CycleResult(action="none",
                               detail={"reason": "atr_unavailable"})
        stop_dist = TRAIL_MULT * atr

        equity = self._cfg.initial_capital_usd
        try:
            latest = self._store.get_latest_balance()
            if latest and latest.get("total_eq_usd") is not None:
                equity = float(latest["total_eq_usd"])
        except Exception:
            logger.exception("open_equity_lookup_failed")

        # Fixed-notional sizing "B": notional = NOTIONAL_LEV_MULT × equity,
        # rounded to OKX's 0.01-contract lot step (NOT floored to whole
        # contracts).  This gives ~2x effective leverage that scales with
        # equity, instead of the old whole-contract floor that forced ~7x on
        # a small account.  See the NOTIONAL_LEV_MULT block above.
        per_contract_usd = last_close * self._cfg.contract_size_base
        if per_contract_usd <= 0:
            return CycleResult(action="none",
                               detail={"reason": "bad_contract_price"})
        target_notional = NOTIONAL_LEV_MULT * equity
        raw_contracts = target_notional / per_contract_usd
        # round to lot step, then snap to 2 dp to kill float dust (0.01 lots)
        size_contracts = round(round(raw_contracts / LOT_STEP) * LOT_STEP, 2)
        if size_contracts < MIN_SZ:
            logger.info("open_skip_min_lot dir=%s target_notional=%.2f per_ct=%.2f",
                        side, target_notional, per_contract_usd)
            return CycleResult(action="none",
                               detail={"reason": "below_min_lot",
                                       "target_notional": target_notional,
                                       "per_contract_usd": per_contract_usd})

        # Achieved notional after lot rounding — what OKX actually opens and
        # what P&L scales against.  size_frac is recorded as the achieved
        # effective leverage (notional / equity), purely for display/records.
        notional = size_contracts * per_contract_usd
        size_frac = notional / equity if equity > 0 else 0.0

        current_stop = (last_close - stop_dist if side == "LONG"
                        else last_close + stop_dist)
        tier = (signal_strength if signal_strength in ("Strong", "Moderate")
                else "Moderate")

        return TradeIntent(
            direction=side, tier=tier,
            entry_price=last_close, stop_price=current_stop,
            atr=atr, stop_dist=stop_dist,
            size_contracts=size_contracts, size_frac=size_frac,
            notional_usd=notional, equity_before=equity,
            bar_ts_iso=bar_ts.isoformat(),
            model_version=model_version,
        )

    def execute_approved_intent(self, intent: TradeIntent, *,
                                  approval_id: Optional[int]) -> CycleResult:
        """Submit OKX market entry + algo stop + DB insert per intent.

        Called from:
          - _open_position when in auto mode (approval_id=None)
          - app.py /yes webhook handler after gate.approve returns
            (approval_id=<id> so we can mark_executed)
        """
        side = intent.direction
        size_contracts = intent.size_contracts
        last_close = intent.entry_price
        current_stop = intent.stop_price
        atr = intent.atr
        tier = intent.tier
        notional = intent.notional_usd
        equity = intent.equity_before

        # Re-parse bar_ts from ISO; falls back to now if absent
        try:
            bar_ts = datetime.fromisoformat(intent.bar_ts_iso)
            if bar_ts.tzinfo is not None:
                bar_ts = bar_ts.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            bar_ts = datetime.utcnow()

        # 0. Pre-submit order guard (defense-in-depth).  Last line before a
        # real order hits OKX: refuse to send if the ACTUAL size/notional
        # implies leverage above the strategy cap — even if _build_intent's
        # sizing logic is buggy (this is the int()-floor 7x bug class from
        # 2026-06-05).  Skip the trade + alert; do NOT halt (single bad
        # intent is a sizing bug, not a system failure).
        presubmit = check_presubmit_order(
            size_contracts=size_contracts, notional_usd=notional,
            equity_usd=equity,
            max_effective_leverage=self._cfg.max_effective_leverage,
            min_size_contracts=MIN_SZ,
        )
        if presubmit.triggered:
            logger.error("presubmit_guard_blocked %s", presubmit.reason)
            try:
                send_critical(
                    self._cfg.telegram_critical_chat_id,
                    f"🛑 <b>PRE-SUBMIT GUARD blocked order</b>\n"
                    f"dir={side} size={size_contracts} contracts "
                    f"notional=${notional:,.2f} equity=${equity:,.2f}\n"
                    f"{presubmit.reason}\n"
                    f"No order sent — sizing produced an out-of-bounds order. "
                    f"Investigate sizing logic.",
                )
            except Exception:
                logger.exception("presubmit_guard_alert_failed")
            if approval_id is not None and self._approval is not None:
                # Don't leave the approval dangling as pending — it never
                # became a trade, so mark it stale (guard-rejected).
                try:
                    self._approval.mark_stale(approval_id,
                                              reason="presubmit_guard_blocked")
                except Exception:
                    logger.exception("presubmit_mark_approval_failed")
            return CycleResult(action="none",
                               detail={"reason": "presubmit_guard_blocked",
                                       "trigger_id": presubmit.trigger_id,
                                       "detail": presubmit.reason})

        # 1. Submit market entry
        entry_cl_ord_id = make_cl_ord_id(prefix="v7")
        order_side = Side.BUY if side == "LONG" else Side.SELL
        # posSide for long_short_mode; None in net_mode (omitted)
        pos_side = self._cfg.pos_side_for(side)

        # 0b. Isolated margin needs leverage configured per (instId, isolated,
        # posSide) BEFORE the order, or OKX rejects it.  Set it idempotently for
        # the side we're about to open; if it fails, ABORT the open (don't fire
        # an order destined for rejection) and alert.  Skipped for cross/cash.
        if self._cfg.td_mode == "isolated":
            try:
                lev_ok = self._client.set_leverage(
                    inst_id=self._cfg.inst_id, lever=self._cfg.leverage,
                    mgn_mode="isolated", pos_side=pos_side,
                )
            except Exception:
                logger.exception("set_leverage_exception")
                lev_ok = False
            if not lev_ok:
                logger.error("set_leverage_failed_abort_open posSide=%s", pos_side)
                try:
                    send_critical(
                        self._cfg.telegram_critical_chat_id,
                        f"🔴 <b>OKX OPEN ABORTED</b>\n"
                        f"set-leverage(isolated {self._cfg.leverage}x "
                        f"posSide={pos_side}) failed — no order sent.\n"
                        f"Isolated margin requires leverage configured first. "
                        f"Check OKX account/API; next signal will retry.",
                    )
                except Exception:
                    logger.exception("set_leverage_abort_alert_failed")
                if approval_id is not None and self._approval is not None:
                    try:
                        self._approval.mark_stale(
                            approval_id, reason="set_leverage_failed")
                    except Exception:
                        logger.exception("set_leverage_mark_approval_failed")
                return CycleResult(action="none",
                                   detail={"reason": "set_leverage_failed"})

        entry_t0 = datetime.now(tz=timezone.utc)
        try:
            entry_result = self._client.submit_market_order(
                inst_id=self._cfg.inst_id, side=order_side,
                sz=size_contracts, td_mode=self._cfg.td_mode,
                cl_ord_id=entry_cl_ord_id,
                pos_side=pos_side,
            )
        except Exception:
            logger.exception("open_submit_entry_failed")
            return CycleResult(action="none",
                               detail={"reason": "entry_submit_exception"})
        if entry_result.status == "rejected":
            logger.warning("open_entry_rejected err=%s", entry_result.error)
            # Surface to operator — without this, user sees a signal in
            # the indicator but no OKX position, and no idea why.
            try:
                send_critical(
                    self._cfg.telegram_critical_chat_id,
                    f"🔴 <b>OKX OPEN FAILED</b>\n"
                    f"dir={intent.direction} tier={intent.tier} "
                    f"size={intent.size_contracts} contracts\n"
                    f"intended entry: ${intent.entry_price:,.2f}\n"
                    f"posSide={self._cfg.pos_side_for(intent.direction)}\n"
                    f"err: {entry_result.error}\n"
                    f"No position opened. Next signal will retry.",
                )
            except Exception:
                logger.exception("open_failed_alert_send_failed")
            return CycleResult(action="none",
                               detail={"reason": "entry_rejected",
                                       "error": entry_result.error})

        # 2. Submit algo stop (Safety belt #7 / B4: within 5s of entry)
        stop_cl_ord_id = make_cl_ord_id(prefix="v7a")
        stop_side = Side.SELL if side == "LONG" else Side.BUY
        try:
            algo_result = self._client.submit_algo_stop(
                inst_id=self._cfg.inst_id, side=stop_side,
                sz=size_contracts, trigger_px=current_stop,
                td_mode=self._cfg.td_mode,
                algo_cl_ord_id=stop_cl_ord_id,
                pos_side=pos_side,           # match the position side
                reduce_only=True,            # algo only closes, never opens
            )
        except Exception:
            logger.exception("open_submit_algo_failed_emergency_close")
            algo_result = None

        stop_placed_ts = (datetime.now(tz=timezone.utc)
                          if algo_result and algo_result.status != "rejected"
                          else None)
        latency_check = check_algo_stop_latency(
            entry_fill_ts=entry_t0, stop_placed_ts=stop_placed_ts,
            max_latency_sec=self._cfg.algo_stop_max_latency_sec,
        )
        if latency_check.triggered:
            logger.error("open_algo_stop_b4_violation_force_close")
            if algo_result and algo_result.algo_id:
                try:
                    self._client.cancel_algo_stop(algo_id=algo_result.algo_id)
                except Exception:
                    logger.exception("emergency_cancel_algo_failed")
            emerg_close_ok = False
            emerg_close_err: Optional[str] = None
            try:
                close_side = Side.SELL if side == "LONG" else Side.BUY
                emerg_result = self._client.submit_market_order(
                    inst_id=self._cfg.inst_id, side=close_side,
                    sz=size_contracts, td_mode=self._cfg.td_mode,
                    cl_ord_id=make_cl_ord_id(prefix="v7emerg"),
                    pos_side=pos_side, reduce_only=True,
                )
                if emerg_result.status == "rejected":
                    emerg_close_err = emerg_result.error or "okx_rejected"
                else:
                    emerg_close_ok = True
            except Exception as exc:
                emerg_close_err = f"exception: {exc}"
                logger.exception("emergency_close_failed")
            # If emergency close FAILED, we have a NAKED position on OKX
            # with NO DB record (we haven't inserted yet) and NO algo stop.
            # This is the worst possible state — user must manually close.
            if not emerg_close_ok:
                try:
                    send_critical(
                        self._cfg.telegram_critical_chat_id,
                        f"🚨 <b>NAKED POSITION ON OKX</b> 🚨\n"
                        f"Entry submitted SUCCESSFULLY but algo stop failed\n"
                        f"AND emergency close failed.\n\n"
                        f"<b>OKX has an open {side} {size_contracts} contract "
                        f"position with NO stop loss and NO DB record.</b>\n\n"
                        f"Direction: {side}\n"
                        f"Size: {size_contracts} contracts\n"
                        f"Approx entry: ${last_close:,.2f}\n"
                        f"Entry clOrdId: {entry_cl_ord_id}\n"
                        f"posSide: {pos_side}\n"
                        f"Emergency close err: {emerg_close_err}\n\n"
                        f"⚠️ GO TO OKX IMMEDIATELY AND MANUALLY CLOSE.\n"
                        f"Executor will be HALTED.",
                    )
                except Exception:
                    logger.exception("naked_position_alert_failed")
            self._alert_critical(latency_check, severity_label="HALT")
            return CycleResult(action="none",
                               detail={"reason": "b4_latency_violation",
                                       "naked_position": not emerg_close_ok})

        # 3. Insert position row
        try:
            new_id = self._store.insert_open_position(
                entry_time=bar_ts, direction=side, entry_tier=tier,
                entry_price=last_close, atr_at_entry=atr,
                stop_dist=intent.stop_dist, current_stop=current_stop,
                size_contracts=size_contracts,
                size_frac=intent.size_frac,
                notional_usd=notional, equity_before=equity,
                entry_cl_ord_id=entry_cl_ord_id,
                stop_algo_cl_ord_id=stop_cl_ord_id,
                model_version=intent.model_version,
            )
        except Exception:
            logger.exception("open_db_insert_failed")
            return CycleResult(action="none",
                               detail={"reason": "db_insert_exception"})
        if new_id is None:
            logger.warning("open_duplicate_cl_ord_id=%s", entry_cl_ord_id)
            return CycleResult(action="none",
                               detail={"reason": "duplicate_cl_ord_id"})

        try:
            self._store.set_position_okx_ids(
                position_id=int(new_id),
                entry_ord_id=entry_result.ord_id,
                stop_algo_id=(algo_result.algo_id if algo_result else None),
            )
        except Exception:
            logger.exception("set_position_okx_ids_failed")

        # Mark approval EXECUTED if we went through the gate
        if approval_id is not None and self._approval is not None:
            self._approval.mark_executed(approval_id=approval_id,
                                          position_id=int(new_id))

        # 4. Telegram entry alert (safety belt #8)
        try:
            msg = format_entry_alert(
                stage_label=self._cfg.stage_label, direction=side,
                tier=tier, entry_price=last_close,
                size_contracts=size_contracts, notional_usd=notional,
                stop_price=current_stop, atr=atr,
            )
            send_critical(self._cfg.telegram_critical_chat_id, msg)
        except Exception:
            logger.exception("open_entry_alert_failed")

        logger.info("okx_open id=%d %s @ %.2f tier=%s size=%d notional=%.2f "
                    "stop=%.2f approval=%s",
                    new_id, side, last_close, tier, size_contracts,
                    notional, current_stop, approval_id)
        return CycleResult(action="open",
                           detail={"position_id": new_id, "side": side,
                                   "entry_price": last_close, "tier": tier,
                                   "size_contracts": size_contracts,
                                   "notional_usd": notional,
                                   "current_stop": current_stop,
                                   "approval_id": approval_id})

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

        # Manual interference takes precedence over everything else: OKX
        # carries a position the executor never opened.  DEMOTE (sticky —
        # needs a deliberate restart to re-enter) but DO NOT force-close it:
        # we must never touch a position we didn't create (wrong size/side
        # assumptions could make it worse).  The operator manages it by hand.
        # This is the code backstop for the 2026-06-05 manual-blowup vector;
        # the real fix is account isolation (docs/okx_account_isolation.md).
        manual = next((t for t in triggered
                       if t.trigger_id == "MANUAL-INTERFERENCE"), None)
        if manual is not None:
            self._alert_manual_interference(manual)
            self._set_status(
                ExecutorStatus.DEMOTED,
                reason=f"manual_interference: {manual.reason}",
                trigger_id="MANUAL-INTERFERENCE",
                context=manual.context,
            )
            return

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
            size = float(row.get("size_contracts") or 0)
            algo_id = row.get("stop_algo_id")

            # Step 1: cancel algo stop (if we have its OKX id)
            if algo_id:
                try:
                    self._client.cancel_algo_stop(algo_id=str(algo_id))
                except Exception:
                    logger.exception("force_close_cancel_algo_failed pos=%s",
                                     pos_id)

            # Step 2: submit opposite-side market order to flatten.
            # Capture result — silent close-fail here means OKX leaves
            # the position open while we mark it DEMOTED in DB.
            close_ok = False
            close_err: Optional[str] = None
            if size > 0 and direction in ("LONG", "SHORT"):
                close_side = Side.SELL if direction == "LONG" else Side.BUY
                try:
                    fc_result = self._client.submit_market_order(
                        inst_id=self._cfg.inst_id, side=close_side,
                        sz=size, td_mode=self._cfg.td_mode,
                        cl_ord_id=make_cl_ord_id(prefix="v7close"),
                        pos_side=self._cfg.pos_side_for(direction),
                        reduce_only=True,
                    )
                    if fc_result.status == "rejected":
                        close_err = fc_result.error or "okx_rejected"
                    else:
                        close_ok = True
                except Exception as exc:
                    close_err = f"exception: {exc}"
                    logger.exception("force_close_market_close_failed pos=%s",
                                     pos_id)
            else:
                # size==0 or direction missing → nothing to close, treat as OK
                close_ok = True

            # If force-close FAILED, leave DB OPEN so reconciler keeps
            # reporting orphan_local AND send a critical alert listing
            # exactly which position needs manual handling.
            if not close_ok:
                try:
                    send_critical(
                        self._cfg.telegram_critical_chat_id,
                        f"🚨 <b>FORCE-CLOSE FAILED on DEMOTE</b>\n"
                        f"Position id={pos_id} dir={direction} "
                        f"size={size} could NOT be closed during demote.\n"
                        f"DB left as OPEN; reconciler will keep alerting.\n"
                        f"err: {close_err}\n\n"
                        f"⚠️ GO TO OKX AND MANUALLY CLOSE.",
                    )
                except Exception:
                    logger.exception("force_close_alert_failed pos=%s", pos_id)
                # Skip DB update — leave row OPEN
                continue

            # Step 3: mark closed in local DB (only if close succeeded)
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

    def _sync_close_from_algo_fill(self, pos: dict, evt) -> None:
        """OKX trail-stop algo fired → close our DB row + alert.

        Called from on_order WS callback when an algo stop fill is
        detected.  Computes P&L from the fill price (preferred) or
        the current stop trigger (fallback when fill_price absent).

        Idempotent: close_position SQL is WHERE status='OPEN', so a
        duplicate WS event (or race with cycle close) is a no-op.
        """
        from datetime import datetime as _dt

        pos_id = int(pos.get("id") or 0)
        side = pos.get("direction") or "FLAT"
        entry_price = float(pos.get("entry_price") or 0)
        equity_before = float(pos.get("equity_before") or 0)
        notional = float(pos.get("notional_usd") or 0)
        # Prefer fill_price; fall back to the stop trigger if WS omitted it
        exit_price = (
            evt.fill_price
            if evt.fill_price is not None and evt.fill_price > 0
            else float(pos.get("current_stop") or entry_price)
        )

        if side == "LONG" and entry_price > 0:
            gross_pct = exit_price / entry_price - 1.0
        elif side == "SHORT" and entry_price > 0:
            gross_pct = -(exit_price / entry_price - 1.0)
        else:
            gross_pct = 0.0
        net_pct = gross_pct - self._cfg.taker_cost
        if equity_before > 0 and notional > 0:
            equity_ret = (notional / equity_before) * net_pct
        else:
            equity_ret = float(pos.get("size_frac") or 0) * net_pct
        equity_after = max(equity_before * (1.0 + equity_ret), 0.0)

        exit_time = evt.ts or _dt.utcnow()

        try:
            self._store.close_position(
                position_id=pos_id,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="trail_stop",
                gross_pct=gross_pct, net_pct=net_pct,
                equity_ret_pct=equity_ret * 100.0,
                equity_after=equity_after,
                new_status="CLOSED",
            )
        except Exception:
            logger.exception("ws_algo_fill_db_close_failed pos=%s", pos_id)
            return

        try:
            msg = format_exit_alert(
                stage_label=self._cfg.stage_label, direction=side,
                reason="trail_stop", entry_price=entry_price,
                exit_price=exit_price, gross_pct=gross_pct,
                net_pct=net_pct, equity_after=equity_after,
            )
            send_critical(self._cfg.telegram_critical_chat_id, msg)
        except Exception:
            logger.exception("ws_algo_fill_alert_failed pos=%s", pos_id)

        logger.info(
            "okx_close_via_ws_algo pos=%d %s %.2f→%.2f net %+.3f%%",
            pos_id, side, entry_price, exit_price, net_pct * 100,
        )

    def _wire_ws_callbacks(self) -> None:
        """Wire WS event callbacks to store-level persistence.

        Callbacks run on WS thread — do NOT mutate in-memory state here.
        Only persist via store; cycle reads from store on next tick.
        """
        from indicator.okx.types import OrderEvent, PositionEvent, BalanceEvent

        def on_order(evt: OrderEvent) -> None:
            # WS order events serve two purposes:
            # 1. Entry fill → record OKX ord_id back into our row
            # 2. Algo stop fill (OKX-triggered trail close) → close our row
            #    AND send the exit Telegram alert.  Without (2) the DB row
            #    stays OPEN forever after OKX auto-closes a position, and
            #    the reconciler flags orphan_local every cycle (HALT loop).
            logger.info("ws_order_event cl_ord=%s state=%s fill_px=%s",
                        evt.cl_ord_id, evt.state, evt.fill_price)
            try:
                pos = self._store.get_open_position()
                if not pos:
                    return
                pos_id = pos.get("id")
                if pos_id is None:
                    return
                # (1) Entry fill — match by entry cl_ord_id
                if pos.get("entry_cl_ord_id") == evt.cl_ord_id and evt.ord_id:
                    self._store.set_position_okx_ids(
                        position_id=int(pos_id),
                        entry_ord_id=evt.ord_id,
                    )
                    return
                # (2) Algo stop fill — when OKX's algo triggers and creates
                # the closing market order, the fill event carries
                # algoClOrdId / algoId, NOT the usual clOrdId (which may be
                # empty or auto-generated). Match on algo_cl_ord_id first,
                # fall back to algo_id (which we may have stored from the
                # submit response). Only act when state="filled".
                stop_cl_ord = pos.get("stop_algo_cl_ord_id") or ""
                stop_algo_id = str(pos.get("stop_algo_id") or "")
                algo_match = (
                    (stop_cl_ord and stop_cl_ord == evt.algo_cl_ord_id)
                    or (stop_algo_id and stop_algo_id == evt.algo_id)
                )
                if algo_match and evt.state == "filled":
                    logger.info(
                        "ws_algo_fill_matched pos_id=%s via=%s "
                        "evt_algo_cl_ord=%s evt_algo_id=%s",
                        pos.get("id"),
                        "cl_ord_id" if stop_cl_ord == evt.algo_cl_ord_id
                        else "algo_id",
                        evt.algo_cl_ord_id, evt.algo_id,
                    )
                    self._sync_close_from_algo_fill(pos, evt)
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
                    size_contracts=float(r.get("size_contracts") or 0),
                    avg_price=float(r.get("entry_price") or 0),
                    raw=r,
                ))
            except Exception:
                logger.exception("dicts_to_positions_row_failed row_id=%s",
                                 r.get("id"))
        return out

    def _alert_manual_interference(self, check: KillCheckResult) -> None:
        """Unmistakable alert for a foreign position on the bot account.

        Distinct from the generic kill alert so the operator immediately
        understands: someone (probably you) traded the executor's account
        by hand, the bot has STOPPED, and it will NOT touch that position.
        Never raises.
        """
        try:
            okx_detail = check.context or {}
            send_critical(
                self._cfg.telegram_critical_chat_id,
                f"🚨 <b>MANUAL INTERFERENCE DETECTED</b> 🚨\n"
                f"stage={self._cfg.stage_label}\n"
                f"OKX shows a position the executor never opened:\n"
                f"<code>{okx_detail}</code>\n\n"
                f"<b>Executor DEMOTED — it has stopped trading and will NOT "
                f"close this position.</b>\n"
                f"If you opened this by hand, this account is NOT isolated. "
                f"Close it yourself, move trading capital to a dedicated "
                f"sub-account the executor alone uses, then restart the "
                f"service to re-enter.\n"
                f"See docs/okx_account_isolation.md.",
            )
        except Exception:
            logger.exception("manual_interference_alert_send_failed")

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
