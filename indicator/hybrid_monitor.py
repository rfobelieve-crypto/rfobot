"""
Hybrid forward window monitor + auto-pause (Path 1 Block 4).

Goal: enforce a HARD RULE that the production hybrid cohort must show
positive forward edge (net_bps > 0) over rolling 2-week windows.  If
edge degrades, automatically pause new signals — Block 1 retrain
pipeline can run and produce a fresher model, and the pause is lifted
when health recovers.

Acceptance for un-pause:
    Most-recent 2-week window net_bps >= +5 bps

Auto-pause trigger:
    Two consecutive 2-week windows with net_bps < 0
    (no false alarms on a single bad week)

State persisted at indicator/state/hybrid_pause.json.

This is independent from risk_manager (which is for live execution).
This module only controls whether `run_hybrid_cycle` records a signal
as "live" (paused_at_signal=0) or "shadowed" (paused_at_signal=1).
Shadowed signals still go to DB so we can compare paused-vs-unpaused
populations; they are excluded from paper PnL.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "indicator" / "state"
PAUSE_STATE_PATH = STATE_DIR / "hybrid_pause.json"

# ── Thresholds ──
ROLLING_WINDOW_DAYS = 14
MIN_TRADES_FOR_DECISION = 8      # need >= 8 closed trades in window to evaluate
PAUSE_NET_BPS_THRESHOLD = 0.0    # net < 0 over window -> potential pause
PAUSE_CONSECUTIVE_WINDOWS = 2    # 2 consecutive bad windows -> pause
UNPAUSE_NET_BPS_THRESHOLD = 5.0  # net >= +5 bps in current window -> unpause


@dataclass
class HybridHealth:
    window_days: int
    window_start: Optional[datetime]
    window_end: datetime
    n_trades: int
    n_wins: int
    wr: float
    net_bps: float
    gross_bps: float
    cum_net_pct: float


def compute_forward_health(window_days: int = ROLLING_WINDOW_DAYS) -> Optional[HybridHealth]:
    """Compute rolling N-day metrics on settled ldc_swing_positions (paused=0).

    Switched from hybrid_signals -> ldc_swing_positions on 2026-05-12 when
    Path 2 (jdehorty LDC swing) replaced the v9+LDC must-agree hybrid as
    the production strategy.
    """
    try:
        from indicator.paper_trading import fetch_ldc_swing_positions
    except Exception as exc:
        logger.warning("hybrid_monitor: cannot import fetch_ldc_swing_positions: %s", exc)
        return None

    df = fetch_ldc_swing_positions()
    if df.empty:
        return None
    df = df[df["paused_at_signal"] == 0]
    if df.empty:
        return None

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)
    # ldc_swing closes by exit_time; use entry_time for window classification
    # so a trade entered in-window is evaluated in-window even if it closes later
    window = df[df["entry_time"] >= cutoff.replace(tzinfo=None)]
    if len(window) < MIN_TRADES_FOR_DECISION:
        return HybridHealth(
            window_days=window_days,
            window_start=cutoff,
            window_end=now,
            n_trades=int(len(window)),
            n_wins=int((window["win"] == 1).sum()),
            wr=float((window["win"] == 1).mean()) if len(window) else 0.0,
            net_bps=float(window["net_pct_maker"].mean() * 10000) if len(window) else 0.0,
            gross_bps=float(window["gross_pct"].mean() * 10000) if len(window) else 0.0,
            cum_net_pct=float(window["net_pct_maker"].sum() * 100) if len(window) else 0.0,
        )

    return HybridHealth(
        window_days=window_days,
        window_start=cutoff,
        window_end=now,
        n_trades=int(len(window)),
        n_wins=int((window["win"] == 1).sum()),
        wr=float((window["win"] == 1).mean()),
        net_bps=float(window["net_pct_maker"].mean() * 10000),
        gross_bps=float(window["gross_pct"].mean() * 10000),
        cum_net_pct=float(window["net_pct_maker"].sum() * 100),
    )


def _load_state() -> dict:
    if not PAUSE_STATE_PATH.exists():
        return {"paused": False, "paused_since": None, "reason": None,
                 "consecutive_bad_windows": 0, "last_check": None,
                 "history": []}
    try:
        with open(PAUSE_STATE_PATH) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("hybrid_monitor: failed to load pause state: %s", exc)
        return {"paused": False, "paused_since": None, "reason": None,
                 "consecutive_bad_windows": 0, "last_check": None,
                 "history": []}


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    with open(PAUSE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def is_paused() -> bool:
    """Read-only check used by run_hybrid_cycle to decide paused flag."""
    return _load_state().get("paused", False)


def check_and_update_pause_state() -> dict:
    """Recompute forward health, update pause state if rules trigger,
    fire Telegram alerts on transitions."""
    state = _load_state()
    health = compute_forward_health()
    if health is None or health.n_trades < MIN_TRADES_FOR_DECISION:
        n = health.n_trades if health else 0
        logger.debug("hybrid_monitor: insufficient data (%d trades < %d), skip",
                      n, MIN_TRADES_FOR_DECISION)
        state["last_check"] = datetime.now(timezone.utc).isoformat()
        if health:
            state["last_window_summary"] = {
                "n": health.n_trades, "wr": health.wr, "net_bps": health.net_bps,
            }
        _save_state(state)
        return state

    summary = {
        "n": health.n_trades, "wr": health.wr,
        "net_bps": health.net_bps, "gross_bps": health.gross_bps,
        "window_days": health.window_days,
    }
    state["last_window_summary"] = summary
    was_paused = state.get("paused", False)
    consec_bad = state.get("consecutive_bad_windows", 0)
    now_iso = datetime.now(timezone.utc).isoformat()

    if health.net_bps < PAUSE_NET_BPS_THRESHOLD:
        consec_bad += 1
        logger.info("hybrid_monitor: bad window (net=%.1f bps, n=%d) — consecutive=%d/%d",
                     health.net_bps, health.n_trades, consec_bad, PAUSE_CONSECUTIVE_WINDOWS)
    else:
        consec_bad = 0  # any good window resets the counter

    state["consecutive_bad_windows"] = consec_bad

    # State transitions
    if not was_paused and consec_bad >= PAUSE_CONSECUTIVE_WINDOWS:
        state["paused"] = True
        state["paused_since"] = now_iso
        state["reason"] = (
            f"{consec_bad} consecutive {ROLLING_WINDOW_DAYS}d windows with net<0. "
            f"Latest window: n={health.n_trades} WR={health.wr*100:.1f}% net={health.net_bps:.1f}bp"
        )
        state["history"].append({
            "ts": now_iso, "event": "paused", "reason": state["reason"],
            "metrics": summary,
        })
        _alert_pause_change(paused=True, summary=summary, reason=state["reason"])
        logger.warning("hybrid_monitor: AUTO-PAUSED — %s", state["reason"])

    elif was_paused and health.net_bps >= UNPAUSE_NET_BPS_THRESHOLD:
        state["paused"] = False
        state["paused_since"] = None
        unpause_reason = (
            f"Recovery: latest {ROLLING_WINDOW_DAYS}d window n={health.n_trades} "
            f"WR={health.wr*100:.1f}% net={health.net_bps:.1f}bp >= +{UNPAUSE_NET_BPS_THRESHOLD:.0f}bp"
        )
        state["reason"] = unpause_reason
        state["history"].append({
            "ts": now_iso, "event": "unpaused", "reason": unpause_reason,
            "metrics": summary,
        })
        _alert_pause_change(paused=False, summary=summary, reason=unpause_reason)
        logger.info("hybrid_monitor: AUTO-UNPAUSED — %s", unpause_reason)

    _save_state(state)
    return state


def _alert_pause_change(paused: bool, summary: dict, reason: str) -> None:
    """Send Telegram alert on pause/unpause transition.  Failure is non-fatal."""
    try:
        from indicator.app import _send_telegram_text
        emoji = "\U0001f6d1" if paused else "✅"  # 🛑 / ✅
        title = "Hybrid AUTO-PAUSED" if paused else "Hybrid AUTO-UNPAUSED"
        msg = (
            f"{emoji} <b>{title}</b>\n\n"
            f"Reason: {reason}\n\n"
            f"Window ({summary.get('window_days', '?')}d): "
            f"n={summary['n']} WR={summary['wr']*100:.1f}% "
            f"net={summary['net_bps']:+.1f}bp gross={summary['gross_bps']:+.1f}bp\n\n"
            f"State file: indicator/state/hybrid_pause.json"
        )
        _send_telegram_text(msg)
    except Exception as exc:
        logger.warning("hybrid_monitor: telegram alert failed: %s", exc)


def get_status_report() -> str:
    """Human-readable status — used for /hybrid-status endpoint."""
    state = _load_state()
    health = compute_forward_health()
    lines = ["\U0001f3e5 <b>Hybrid Monitor Status</b>\n"]
    if state.get("paused"):
        lines.append(f"⛔ <b>PAUSED</b> since {state.get('paused_since', '?')}")
        lines.append(f"Reason: {state.get('reason', '?')}")
    else:
        lines.append("✅ <b>ACTIVE</b> — signals being recorded as live")

    if health is not None:
        lines.append(
            f"\n<b>Last {health.window_days}d window:</b>\n"
            f"  trades: {health.n_trades} (min {MIN_TRADES_FOR_DECISION} to evaluate)\n"
            f"  WR: {health.wr*100:.1f}%\n"
            f"  net: {health.net_bps:+.1f} bps/trade\n"
            f"  cum: {health.cum_net_pct:+.2f}%"
        )
    else:
        lines.append("\n<b>No hybrid trades yet</b> (waiting for first settled signal)")

    lines.append(
        f"\nConsecutive bad windows: {state.get('consecutive_bad_windows', 0)} / "
        f"{PAUSE_CONSECUTIVE_WINDOWS}"
    )
    return "\n".join(lines)
