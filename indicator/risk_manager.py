"""
Risk Manager — hard rules for staged auto-trading.

This module enforces drawdown/loss/position-count limits *before* any
order is placed, and tracks trade outcomes to update state.  All rules
are mandatory and applied as a chain of veto gates — any failing rule
blocks the trade.

Design:
    - Stateful `RiskManager` class.  Single instance per running process.
    - Pure-function rules: each rule returns (allowed, reason).
    - Persistence: state can be dumped to / loaded from JSON for restart safety.
    - Logging: every gate decision + state transition is logged for audit.

The same module is used in paper, testnet, and live stages.  Only the
`RiskConfig` values differ per stage (more conservative for smaller stages).

Used by:
    - research/simulate_risk_managed.py (offline paper simulation, this commit)
    - future:  indicator/trading_executor.py (live)

NOT used yet by indicator/inference.py — risk_manager only kicks in when
we actually trade.  v7 paper signals still flow through inference unchanged.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Per-stage risk parameters.  All values are fractions (0.005 = 0.5%)."""
    # Capital
    initial_equity: float = 1000.0
    # Position sizing
    max_risk_per_trade: float = 0.005        # 0.5% of equity per trade
    max_concurrent_positions: int = 1
    # Daily limits
    daily_loss_cap: float = 0.03             # halt today's signals if -3%
    daily_loss_cap_action: str = "halt_day"  # halt_day | halt_session
    # Consecutive losses
    consecutive_loss_threshold: int = 7
    consecutive_loss_pause_hours: int = 24
    # Drawdown
    max_drawdown: float = 0.20               # 20% of high-water mark
    max_drawdown_action: str = "halt_permanent"  # halt_permanent | downgrade_stage
    # Stage label (informational)
    stage: str = "paper"                     # paper | testnet | live_tiny | live_full

    @classmethod
    def from_stage(cls, stage: str) -> "RiskConfig":
        """Convenience constructor — sensible defaults per stage."""
        if stage == "paper":
            return cls(stage="paper", initial_equity=1000.0)
        if stage == "testnet":
            return cls(stage="testnet", initial_equity=10000.0)
        if stage == "live_tiny":
            return cls(stage="live_tiny", initial_equity=100.0,
                       max_risk_per_trade=0.01, max_drawdown=0.30)
        if stage == "live_full":
            return cls(stage="live_full", initial_equity=1000.0)
        raise ValueError(f"unknown stage: {stage}")


# ─── State ──────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """Stateful counters and tracking.  Persistable to JSON."""
    equity: float = 1000.0
    high_water_mark: float = 1000.0
    current_drawdown: float = 0.0           # fraction below HWM
    # Today's tracking (reset at UTC 00:00)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_daily_reset: Optional[str] = None  # ISO datetime
    # Consecutive losses
    consecutive_losses: int = 0
    last_trade_was_loss: bool = False
    # Halt state
    is_halted: bool = False
    halt_reason: Optional[str] = None
    halt_until: Optional[str] = None        # ISO datetime; None = permanent
    # Active positions (tracked by external id)
    active_positions: list = field(default_factory=list)
    # All trade history
    n_trades_total: int = 0
    n_wins: int = 0
    n_losses: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RiskState":
        return cls(**json.loads(s))


@dataclass
class RiskEvent:
    """One audit-log entry per risk decision."""
    ts: str
    event_type: str     # gate_allow | gate_block | halt_set | halt_cleared | daily_reset
    rule: Optional[str] = None
    reason: Optional[str] = None
    payload: Optional[dict] = None


# ─── RiskManager ────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self, config: RiskConfig, state: Optional[RiskState] = None):
        self.config = config
        if state is None:
            state = RiskState(equity=config.initial_equity,
                              high_water_mark=config.initial_equity)
        self.state = state
        self.events: list[RiskEvent] = []
        logger.info("RiskManager init: stage=%s equity=%.2f",
                    config.stage, state.equity)

    # ── Public API ────────────────────────────────────────────────────────

    def can_open_position(self, now: datetime, signal_id: str = "?") -> tuple[bool, str]:
        """Run all gates in order.  First failing gate blocks the trade."""
        self._maybe_reset_daily(now)
        self._maybe_clear_temporary_halt(now)

        # Gate 1: halt
        if self.state.is_halted:
            reason = f"halted: {self.state.halt_reason}"
            self._log("gate_block", "halt", reason, {"signal_id": signal_id})
            return False, reason

        # Gate 2: concurrent positions
        if len(self.state.active_positions) >= self.config.max_concurrent_positions:
            reason = (f"max_concurrent_positions "
                      f"({len(self.state.active_positions)}/{self.config.max_concurrent_positions})")
            self._log("gate_block", "concurrent", reason, {"signal_id": signal_id})
            return False, reason

        # Gate 3: daily loss cap
        if self.state.daily_pnl <= -self.config.daily_loss_cap * self.state.equity:
            reason = f"daily_loss_cap ({self.state.daily_pnl/self.state.equity*100:.2f}%)"
            self._log("gate_block", "daily_loss", reason, {"signal_id": signal_id})
            return False, reason

        # Gate 4: drawdown trigger
        if self.state.current_drawdown >= self.config.max_drawdown:
            reason = f"max_drawdown ({self.state.current_drawdown*100:.2f}%)"
            self.halt(now, "max_drawdown_triggered", permanent=True)
            self._log("gate_block", "drawdown", reason, {"signal_id": signal_id})
            return False, reason

        self._log("gate_allow", None, "ok", {"signal_id": signal_id})
        return True, "ok"

    def position_size_units(self, sl_dist: float, entry_price: float) -> float:
        """Return position size in base units (e.g. BTC)."""
        risk_dollars = self.config.max_risk_per_trade * self.state.equity
        # P&L per unit move = entry_price × sl_dist
        # Number of units to risk: risk_dollars / (entry_price × sl_dist)
        if entry_price <= 0 or sl_dist <= 0:
            return 0.0
        return risk_dollars / (entry_price * sl_dist)

    def position_size_notional(self, sl_dist: float) -> float:
        """Notional position value in USD."""
        if sl_dist <= 0:
            return 0.0
        return (self.config.max_risk_per_trade * self.state.equity) / sl_dist

    def record_position_open(self, now: datetime, position_id: str,
                              direction: str, entry_price: float,
                              size_notional: float) -> None:
        self.state.active_positions.append({
            "id": position_id,
            "ts_open": now.isoformat(),
            "direction": direction,
            "entry_price": entry_price,
            "size_notional": size_notional,
        })
        self._log("position_open", None, None,
                  {"id": position_id, "direction": direction,
                   "entry_price": entry_price, "notional": size_notional})

    def record_position_close(self, now: datetime, position_id: str,
                                pnl_pct: float, exit_reason: str = "") -> None:
        """pnl_pct = signed PnL as fraction of notional (e.g. +0.005 = +0.5%)."""
        # Remove from active list
        pos = next((p for p in self.state.active_positions if p["id"] == position_id), None)
        if pos is None:
            logger.warning("close called for unknown position %s", position_id)
            return
        self.state.active_positions = [p for p in self.state.active_positions if p["id"] != position_id]

        # PnL in dollars: pnl_pct × notional
        pnl_dollars = pnl_pct * pos["size_notional"]
        self.state.equity += pnl_dollars
        self.state.daily_pnl += pnl_dollars
        self.state.daily_trades += 1
        self.state.n_trades_total += 1

        if pnl_pct > 0:
            self.state.n_wins += 1
            self.state.consecutive_losses = 0
            self.state.last_trade_was_loss = False
        else:
            self.state.n_losses += 1
            self.state.consecutive_losses += 1
            self.state.last_trade_was_loss = True
            if self.state.consecutive_losses >= self.config.consecutive_loss_threshold:
                self.halt(now, "consecutive_loss_threshold",
                          duration_hours=self.config.consecutive_loss_pause_hours)

        # Update high-water mark + drawdown
        if self.state.equity > self.state.high_water_mark:
            self.state.high_water_mark = self.state.equity
            self.state.current_drawdown = 0.0
        else:
            self.state.current_drawdown = (
                self.state.high_water_mark - self.state.equity
            ) / max(self.state.high_water_mark, 1e-9)

        self._log("position_close", None, exit_reason,
                  {"id": position_id, "pnl_pct": pnl_pct, "pnl_dollars": pnl_dollars,
                   "equity_after": self.state.equity,
                   "current_dd": self.state.current_drawdown})

    def halt(self, now: datetime, reason: str,
             duration_hours: Optional[int] = None,
             permanent: bool = False) -> None:
        self.state.is_halted = True
        self.state.halt_reason = reason
        if permanent:
            self.state.halt_until = None  # never auto-clear
        elif duration_hours is not None:
            self.state.halt_until = (now + timedelta(hours=duration_hours)).isoformat()
        else:
            self.state.halt_until = None
        self._log("halt_set", reason, None,
                  {"permanent": permanent, "halt_until": self.state.halt_until})

    def kill_switch(self, now: datetime, reason: str) -> list[dict]:
        """Emergency stop: halt permanently + return list of active positions to close."""
        to_close = list(self.state.active_positions)
        self.halt(now, f"KILL_SWITCH: {reason}", permanent=True)
        self._log("kill_switch", reason, None, {"positions_to_close": len(to_close)})
        return to_close

    # ── Internal ──────────────────────────────────────────────────────────

    def _maybe_reset_daily(self, now: datetime) -> None:
        last = self.state.last_daily_reset
        if last is None:
            self.state.last_daily_reset = now.replace(
                hour=0, minute=0, second=0, microsecond=0).isoformat()
            return
        last_dt = datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now_aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        if now_aware.date() > last_dt.date():
            self._log("daily_reset", None, None,
                      {"daily_pnl_before": self.state.daily_pnl,
                       "daily_trades_before": self.state.daily_trades})
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_daily_reset = now_aware.replace(
                hour=0, minute=0, second=0, microsecond=0).isoformat()

    def _maybe_clear_temporary_halt(self, now: datetime) -> None:
        if not self.state.is_halted or self.state.halt_until is None:
            return
        halt_until_dt = datetime.fromisoformat(self.state.halt_until)
        if halt_until_dt.tzinfo is None:
            halt_until_dt = halt_until_dt.replace(tzinfo=timezone.utc)
        now_aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        if now_aware >= halt_until_dt:
            self._log("halt_cleared", self.state.halt_reason, None,
                      {"cleared_at": now_aware.isoformat()})
            self.state.is_halted = False
            self.state.halt_reason = None
            self.state.halt_until = None

    def _log(self, event_type: str, rule: Optional[str],
             reason: Optional[str], payload: Optional[dict]) -> None:
        ev = RiskEvent(ts=datetime.now(timezone.utc).isoformat(),
                        event_type=event_type, rule=rule,
                        reason=reason, payload=payload)
        self.events.append(ev)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.state.to_json())

    def load_state(self, path: Path) -> None:
        with open(path) as f:
            self.state = RiskState.from_json(f.read())

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        wr = self.state.n_wins / max(self.state.n_trades_total, 1)
        return {
            "stage": self.config.stage,
            "equity": self.state.equity,
            "high_water_mark": self.state.high_water_mark,
            "current_drawdown_pct": self.state.current_drawdown * 100,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.state.daily_pnl / max(self.state.equity, 1e-9) * 100,
            "daily_trades": self.state.daily_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "active_positions": len(self.state.active_positions),
            "halted": self.state.is_halted,
            "halt_reason": self.state.halt_reason,
            "halt_until": self.state.halt_until,
            "n_trades_total": self.state.n_trades_total,
            "winrate": wr,
        }
