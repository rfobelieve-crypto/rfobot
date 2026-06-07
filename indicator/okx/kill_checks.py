"""10 hard safety belts (user accepted all on 2026-05-25).

Each check is a pure function: takes context, returns KillCheckResult.
Called from V7OkxExecutor.cycle() before any order action.

Mapping to docs/stage2_kill_criteria.md trigger IDs:

  Safety belt #1: clOrdId on every order
      → idempotency at submit time, enforced in rest.submit_market_order
      → no separate check needed (assert-at-submit pattern)

  Safety belt #2: capital hard cap
      → enforced by validate_okx_config + check_capital_cap below

  Safety belt #3: daily loss cap (-50% of capital)
      → check_daily_loss_cap, trigger ID D2-like (custom Stage 3 trigger)

  Safety belt #4: total loss cap (-50% of capital)
      → check_total_loss_cap, trigger ID D2-like

  Safety belt #5: API key permission verify
      → check_api_permissions, trigger ID E4

  Safety belt #6: every-cycle reconciliation
      → check_reconciliation, trigger ID A4 (delegates to PositionReconciler)

  Safety belt #7: algo stop placement latency 5s
      → check_algo_stop_latency, trigger ID B4

  Safety belt #8: every entry Telegram push
      → alerter side effect, see executor.cycle (no kill check)

  Safety belt #9: NTP drift > 5s halt / >30s demote
      → check_ntp_drift, trigger IDs C5 / C6

  Safety belt #10: MAX_POSITION_COUNT=1
      → check_max_position, custom Stage 3 trigger
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from indicator.okx.types import (
    ConnectivityStatus, KillCheckResult, KillSeverity,
    Position, ReconciliationResult, ReconciliationVerdict,
)

logger = logging.getLogger(__name__)


# ── #2 Capital hard cap ────────────────────────────────────────────────

def check_capital_cap(*, current_equity_usd: float,
                     capital_hard_cap_usd: float) -> KillCheckResult:
    """Equity must not exceed configured capital cap.

    Reason: prevents accidental over-funding into the bot account.  If
    you transfer in extra funds, system halts so you confirm intent.
    """
    if current_equity_usd > capital_hard_cap_usd * 1.5:
        return KillCheckResult(
            triggered=True, trigger_id="CAP-2",
            severity=KillSeverity.HALT,
            reason=f"equity ${current_equity_usd:.2f} > "
                   f"1.5x hard cap ${capital_hard_cap_usd:.2f}",
            context={"equity": current_equity_usd,
                     "cap": capital_hard_cap_usd},
        )
    return KillCheckResult(triggered=False)


# ── #3 Daily loss cap ──────────────────────────────────────────────────

def check_daily_loss_cap(*, day_start_equity_usd: float,
                        current_equity_usd: float,
                        daily_loss_cap_pct: float) -> KillCheckResult:
    """Today's drawdown from day-open must not exceed daily_loss_cap_pct.

    daily_loss_cap_pct is negative (e.g. -50.0 means -50%).
    """
    if day_start_equity_usd <= 0:
        return KillCheckResult(triggered=False)
    day_change_pct = ((current_equity_usd - day_start_equity_usd)
                      / day_start_equity_usd) * 100.0
    if day_change_pct <= daily_loss_cap_pct:
        return KillCheckResult(
            triggered=True, trigger_id="CAP-3",
            severity=KillSeverity.HALT,   # halt, not demote — tomorrow can resume
            reason=f"daily PnL {day_change_pct:+.1f}% ≤ "
                   f"cap {daily_loss_cap_pct:.1f}%",
            context={"day_start": day_start_equity_usd,
                     "current": current_equity_usd,
                     "day_change_pct": day_change_pct},
        )
    return KillCheckResult(triggered=False)


# ── #4 Total loss cap ──────────────────────────────────────────────────

def check_total_loss_cap(*, initial_capital_usd: float,
                        current_equity_usd: float,
                        total_loss_cap_pct: float) -> KillCheckResult:
    """Cumulative loss from initial capital must not exceed total_loss_cap_pct.

    Triggers DEMOTE (terminal) — needs human re-entry.
    """
    if initial_capital_usd <= 0:
        return KillCheckResult(triggered=False)
    total_pct = ((current_equity_usd - initial_capital_usd)
                 / initial_capital_usd) * 100.0
    if total_pct <= total_loss_cap_pct:
        return KillCheckResult(
            triggered=True, trigger_id="CAP-4",
            severity=KillSeverity.DEMOTE,
            reason=f"total PnL {total_pct:+.1f}% ≤ "
                   f"cap {total_loss_cap_pct:.1f}%",
            context={"initial": initial_capital_usd,
                     "current": current_equity_usd,
                     "total_pct": total_pct},
        )
    return KillCheckResult(triggered=False)


# ── #5 API key permission verify (startup only) ────────────────────────

def check_api_permissions(*, perms: list[str]) -> KillCheckResult:
    """API key MUST NOT have withdraw or transfer permission.

    Called once at startup after querying OKX /api/v5/account/config.
    OKX V5 returns perms as a comma-joined list including "read_only",
    "trade", "withdraw" — we normalise so callers can pass either form.
    """
    forbidden = ("withdraw", "transfer")
    perms_lower = [p.lower() for p in perms]
    bad = [p for p in forbidden if p in perms_lower]
    if bad:
        return KillCheckResult(
            triggered=True, trigger_id="E4",
            severity=KillSeverity.DEMOTE,
            reason=f"API key has forbidden permissions: {bad}",
            context={"perms": perms_lower, "forbidden": bad},
        )
    # OKX returns "read_only" not "read"; accept either as satisfying read.
    has_read = "read" in perms_lower or "read_only" in perms_lower
    has_trade = "trade" in perms_lower
    missing = []
    if not has_read:
        missing.append("read")
    if not has_trade:
        missing.append("trade")
    if missing:
        return KillCheckResult(
            triggered=True, trigger_id="E4",
            severity=KillSeverity.DEMOTE,
            reason=f"API key missing required permissions: {missing}",
            context={"perms": perms_lower, "missing": missing},
        )
    return KillCheckResult(triggered=False)


# ── #6 Reconciliation ──────────────────────────────────────────────────

def check_reconciliation(*, result: ReconciliationResult) -> KillCheckResult:
    """A4: any mismatch between local DB and OKX positions → halt."""
    if result.verdict == ReconciliationVerdict.MISMATCH:
        return KillCheckResult(
            triggered=True, trigger_id="A4",
            severity=KillSeverity.HALT,
            reason=f"reconciliation mismatch: {result.detail}",
            context=result.detail,
        )
    if result.verdict == ReconciliationVerdict.UNAVAILABLE:
        return KillCheckResult(
            triggered=True, trigger_id="A4",
            severity=KillSeverity.HALT,
            reason=f"reconciliation unavailable: {result.detail}",
            context=result.detail,
        )
    return KillCheckResult(triggered=False)


# ── #7 Algo stop placement latency ─────────────────────────────────────

def check_algo_stop_latency(*, entry_fill_ts: datetime,
                            stop_placed_ts: Optional[datetime],
                            max_latency_sec: float) -> KillCheckResult:
    """B4: after entry fill, algo stop must be live within max_latency_sec.

    Called after submit_algo_stop returns success.  If stop_placed_ts is
    None or > max_latency_sec after fill → halt the position (market
    close it ourselves) per safety belt #7.
    """
    if stop_placed_ts is None:
        return KillCheckResult(
            triggered=True, trigger_id="B4",
            severity=KillSeverity.HALT,
            reason="algo stop placement returned no timestamp",
        )
    latency = (stop_placed_ts - entry_fill_ts).total_seconds()
    if latency > max_latency_sec:
        return KillCheckResult(
            triggered=True, trigger_id="B4",
            severity=KillSeverity.HALT,
            reason=f"algo stop placement latency {latency:.1f}s > "
                   f"{max_latency_sec:.1f}s",
            context={"latency_sec": latency,
                     "max_latency_sec": max_latency_sec},
        )
    return KillCheckResult(triggered=False)


# ── #9 NTP drift ────────────────────────────────────────────────────────

def check_ntp_drift(*, local_ts_sec: float, server_ts_sec: float,
                   halt_threshold_sec: float = 5.0,
                   demote_threshold_sec: float = 30.0) -> KillCheckResult:
    """C5/C6: clock drift between local and OKX server.

    >5s → halt (orders may start rejecting)
    >30s → demote (OKX will hard-reject)
    """
    drift_sec = abs(local_ts_sec - server_ts_sec)
    if drift_sec > demote_threshold_sec:
        return KillCheckResult(
            triggered=True, trigger_id="C6",
            severity=KillSeverity.DEMOTE,
            reason=f"NTP drift {drift_sec:.1f}s > "
                   f"demote threshold {demote_threshold_sec:.1f}s",
            context={"drift_sec": drift_sec},
        )
    if drift_sec > halt_threshold_sec:
        return KillCheckResult(
            triggered=True, trigger_id="C5",
            severity=KillSeverity.HALT,
            reason=f"NTP drift {drift_sec:.1f}s > "
                   f"halt threshold {halt_threshold_sec:.1f}s",
            context={"drift_sec": drift_sec},
        )
    return KillCheckResult(triggered=False)


# ── Pre-submit order guard (defense-in-depth before every entry) ────────

def check_presubmit_order(*, size_contracts: float, notional_usd: float,
                          equity_usd: float,
                          max_effective_leverage: float,
                          min_size_contracts: float) -> KillCheckResult:
    """Final sanity on the ACTUAL order params, run right before submit.

    This is defense-in-depth, NOT a duplicate of sizing: even if
    _build_intent's sizing logic has a bug, this guard refuses to send an
    order whose effective leverage (notional / equity) blows past the
    strategy's intended ~2x (NOTIONAL_LEV_MULT).  It directly closes the
    bug class behind the 2026-06-05 over-leverage event, where an
    int()-floor to whole contracts forced ~7x on a small account
    (see CLAUDE.md §"分數合約 sizing B").

    NOTE: max_effective_leverage is the STRATEGY risk leverage cap, NOT the
    OKX account leverage (10x) — the latter only governs margin lockup.

    Returns triggered=True (caller skips the order + alerts) on:
      - non-finite / non-positive / sub-min size
      - effective leverage above max_effective_leverage
    Does NOT halt the executor: a single bad intent is a sizing bug to
    skip + surface, not a system-wide failure.
    """
    import math

    if (not math.isfinite(size_contracts) or size_contracts < min_size_contracts
            or not math.isfinite(notional_usd) or notional_usd <= 0):
        return KillCheckResult(
            triggered=True, trigger_id="PRESUBMIT-SIZE",
            severity=KillSeverity.HALT,   # severity unused (we skip, not halt)
            reason=f"order size {size_contracts} / notional {notional_usd} "
                   f"invalid (min lot {min_size_contracts})",
            context={"size_contracts": size_contracts,
                     "notional_usd": notional_usd,
                     "min_size_contracts": min_size_contracts},
        )
    if equity_usd > 0:
        eff_lev = notional_usd / equity_usd
        if eff_lev > max_effective_leverage:
            return KillCheckResult(
                triggered=True, trigger_id="PRESUBMIT-LEV",
                severity=KillSeverity.HALT,
                reason=f"effective leverage {eff_lev:.2f}x > cap "
                       f"{max_effective_leverage:.2f}x "
                       f"(notional ${notional_usd:.2f} / equity ${equity_usd:.2f})",
                context={"eff_leverage": eff_lev,
                         "max_effective_leverage": max_effective_leverage,
                         "notional_usd": notional_usd,
                         "equity_usd": equity_usd},
            )
    return KillCheckResult(triggered=False)


# ── #10 Max position count ──────────────────────────────────────────────

def check_max_position(*, local_open_positions: list[Position],
                       max_count: int = 1) -> KillCheckResult:
    """Safety belt #10: never have more than max_count open positions.

    If this fires we have a bug in our own state — refuse to open more.
    """
    if len(local_open_positions) > max_count:
        return KillCheckResult(
            triggered=True, trigger_id="MAX-POS",
            severity=KillSeverity.DEMOTE,
            reason=f"local DB has {len(local_open_positions)} open "
                   f"positions > max {max_count}",
            context={"count": len(local_open_positions),
                     "max": max_count},
        )
    return KillCheckResult(triggered=False)


# ── Connectivity (A1/A2/A3) ────────────────────────────────────────────

def check_connectivity(*, status: ConnectivityStatus,
                       ws_disconnect_demote_sec: int = 300,
                       reconnect_fail_demote_count: int = 3,
                       heartbeat_timeout_sec: float = 30.0) -> KillCheckResult:
    """A1 / A2 / A3 grouped:
      A1: private WS disconnect > 5 min → demote
      A2: 3 consecutive reconnect failures → demote
      A3: heartbeat lag > 30s → halt
    """
    # A2
    if status.consecutive_reconnect_fails >= reconnect_fail_demote_count:
        return KillCheckResult(
            triggered=True, trigger_id="A2",
            severity=KillSeverity.DEMOTE,
            reason=f"{status.consecutive_reconnect_fails} consecutive "
                   f"reconnect failures",
            context={"fails": status.consecutive_reconnect_fails},
        )
    # A1 (proxied by long private heartbeat age)
    if status.last_private_heartbeat_age_sec > ws_disconnect_demote_sec:
        return KillCheckResult(
            triggered=True, trigger_id="A1",
            severity=KillSeverity.DEMOTE,
            reason=f"private WS heartbeat age "
                   f"{status.last_private_heartbeat_age_sec:.0f}s > "
                   f"{ws_disconnect_demote_sec}s",
            context={"hb_age_sec": status.last_private_heartbeat_age_sec},
        )
    # A3
    if status.last_private_heartbeat_age_sec > heartbeat_timeout_sec:
        return KillCheckResult(
            triggered=True, trigger_id="A3",
            severity=KillSeverity.HALT,
            reason=f"private WS heartbeat age "
                   f"{status.last_private_heartbeat_age_sec:.1f}s > "
                   f"{heartbeat_timeout_sec:.0f}s",
            context={"hb_age_sec": status.last_private_heartbeat_age_sec},
        )
    return KillCheckResult(triggered=False)


# ── Aggregator (called every cycle) ────────────────────────────────────

def run_all_checks(*, cfg, equity_usd: float, day_start_equity_usd: float,
                   local_positions: list[Position],
                   reconciliation: ReconciliationResult,
                   connectivity: ConnectivityStatus,
                   ntp_drift_sec: Optional[float] = None) -> list[KillCheckResult]:
    """Run all per-cycle kill checks; return only triggered ones.

    Empty list = all clear.  Caller (executor) decides how to act on
    each triggered result based on severity.
    """
    triggered = []

    for check_result in (
        check_capital_cap(
            current_equity_usd=equity_usd,
            capital_hard_cap_usd=cfg.initial_capital_usd,
        ),
        check_daily_loss_cap(
            day_start_equity_usd=day_start_equity_usd,
            current_equity_usd=equity_usd,
            daily_loss_cap_pct=cfg.daily_loss_cap_pct,
        ),
        check_total_loss_cap(
            initial_capital_usd=cfg.initial_capital_usd,
            current_equity_usd=equity_usd,
            total_loss_cap_pct=cfg.total_loss_cap_pct,
        ),
        check_reconciliation(result=reconciliation),
        check_connectivity(
            status=connectivity,
            ws_disconnect_demote_sec=cfg.ws_disconnect_demote_sec,
            reconnect_fail_demote_count=cfg.reconnect_fail_demote_count,
            heartbeat_timeout_sec=cfg.heartbeat_timeout_sec,
        ),
        check_max_position(local_open_positions=local_positions,
                           max_count=cfg.max_position_count),
    ):
        if check_result.triggered:
            triggered.append(check_result)

    if ntp_drift_sec is not None:
        ntp_result = check_ntp_drift(
            local_ts_sec=0.0, server_ts_sec=ntp_drift_sec,
            halt_threshold_sec=cfg.ntp_drift_halt_sec,
            demote_threshold_sec=cfg.ntp_drift_demote_sec,
        )
        if ntp_result.triggered:
            triggered.append(ntp_result)

    return triggered
