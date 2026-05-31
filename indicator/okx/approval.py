"""Manual-approval Telegram gate for the first 5 Stage 3 trades.

Flow:
  1. Executor decides to open a position → builds an `intent` dict
     (everything _open_position needs to actually submit)
  2. ApprovalGate.request_approval persists the intent as PENDING and
     pushes a Telegram message with /yes_<id> /no_<id> commands
  3. Operator replies in Telegram; webhook handler in app.py calls
     ApprovalGate.approve / deny
  4. On approve, executor.execute_approved_intent(intent) submits
     the actual OKX order, then ApprovalGate.mark_executed records it
  5. After AUTO_MODE_THRESHOLD executed approvals, gate flips to auto:
     subsequent intents skip the Telegram round-trip

Staleness guard: if user takes too long and price drifts > MAX_DRIFT_PCT,
the approval is marked STALE instead of submitting at a now-bad price.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from indicator.okx.alerter import format_approval_request, send_critical
from indicator.okx.state import OkxStateStore

logger = logging.getLogger(__name__)


# Window during which an operator can /yes a pending approval.
APPROVAL_TTL_MINUTES = 30
# Maximum allowed price drift between intent and execution time.
# Above this, refuse to submit (the trade thesis was for a different
# price level).
MAX_DRIFT_PCT = 0.005   # 0.5 %


@dataclass
class TradeIntent:
    """Everything _execute_intent needs to actually submit an order.

    Kept JSON-serialisable so it round-trips through DB cleanly.
    """
    direction: str            # "LONG" / "SHORT"
    tier: str                 # "Strong" / "Moderate"
    entry_price: float        # last_close at signal time
    stop_price: float
    atr: float
    stop_dist: float
    size_contracts: int
    size_frac: float
    notional_usd: float
    equity_before: float
    bar_ts_iso: str           # ISO string for JSON safety
    model_version: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "direction": self.direction,
            "tier": self.tier,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "atr": self.atr,
            "stop_dist": self.stop_dist,
            "size_contracts": self.size_contracts,
            "size_frac": self.size_frac,
            "notional_usd": self.notional_usd,
            "equity_before": self.equity_before,
            "bar_ts_iso": self.bar_ts_iso,
            "model_version": self.model_version,
            "extra": self.extra,
        })

    @classmethod
    def from_json(cls, raw: str) -> "TradeIntent":
        d = json.loads(raw) if isinstance(raw, str) else raw
        return cls(
            direction=d["direction"],
            tier=d["tier"],
            entry_price=float(d["entry_price"]),
            stop_price=float(d["stop_price"]),
            atr=float(d["atr"]),
            stop_dist=float(d["stop_dist"]),
            size_contracts=int(d["size_contracts"]),
            size_frac=float(d["size_frac"]),
            notional_usd=float(d["notional_usd"]),
            equity_before=float(d["equity_before"]),
            bar_ts_iso=d["bar_ts_iso"],
            model_version=d.get("model_version"),
            extra=d.get("extra") or {},
        )


@dataclass
class ApprovalDecision:
    """Result of approve/deny, used by webhook caller for follow-up."""
    ok: bool
    status: str                       # "APPROVED" / "DENIED" / "EXPIRED" / "NOT_PENDING" / "NOT_FOUND"
    intent: Optional[TradeIntent] = None
    reason: str = ""


class ApprovalGate:
    """Manual-approval state machine.

    All persistence goes through OkxStateStore so restart-safe.
    Threshold lowered from 5 → 1 on 2026-05-31 (user override): a single
    first-trade confirmation is enough to spot gross code bugs; ongoing
    operation should remain fully automated to preserve the point of
    algorithmic trading.
    """

    AUTO_MODE_THRESHOLD = 1

    def __init__(self, *, store: OkxStateStore, alert_chat_id: str,
                 stage_label: str = "live",
                 threshold: Optional[int] = None):
        self._store = store
        self._chat_id = alert_chat_id
        self._stage_label = stage_label
        self._threshold = (threshold if threshold is not None
                            else self.AUTO_MODE_THRESHOLD)

    # ── Mode ──────────────────────────────────────────────────────────

    def is_auto_mode(self) -> bool:
        """True once threshold executed approvals have accumulated."""
        try:
            return (self._store.get_executed_approval_count()
                    >= self._threshold)
        except Exception:
            logger.exception("approval_count_failed_defaulting_manual")
            # Fail closed → stay in manual mode on DB error
            return False

    def progress(self) -> tuple[int, int]:
        """(approved_count, threshold) for display."""
        try:
            n = self._store.get_executed_approval_count()
        except Exception:
            n = 0
        return n, self._threshold

    # ── Request flow ──────────────────────────────────────────────────

    def request_approval(self, intent: TradeIntent) -> Optional[int]:
        """Persist intent as PENDING + push Telegram.  Returns approval id.

        Returns None if persistence or alert fails — caller should treat
        as "couldn't ask" and skip the trade.
        """
        expires = datetime.now(tz=timezone.utc) + timedelta(
            minutes=APPROVAL_TTL_MINUTES)
        try:
            approval_id = self._store.insert_pending_approval(
                intent_json=intent.to_json(),
                expires_at=expires,
            )
        except Exception:
            logger.exception("approval_insert_failed")
            return None

        # Push Telegram (best-effort; if it fails, the row stays PENDING
        # and will EXPIRE.  No trade happens — safe default.)
        try:
            count, threshold = self.progress()
            msg = format_approval_request(
                approval_id=approval_id, stage_label=self._stage_label,
                direction=intent.direction, tier=intent.tier,
                entry_price=intent.entry_price,
                size_contracts=intent.size_contracts,
                notional_usd=intent.notional_usd,
                stop_price=intent.stop_price, atr=intent.atr,
                count_so_far=count, threshold=threshold,
            )
            sent = send_critical(self._chat_id, msg)
            if not sent:
                logger.warning("approval_telegram_send_failed id=%d",
                               approval_id)
        except Exception:
            logger.exception("approval_telegram_format_failed")

        return approval_id

    # ── Decision flow (called from webhook handler) ───────────────────

    def approve(self, approval_id: int, *,
                 decided_by: str = "telegram") -> ApprovalDecision:
        """Transition PENDING → APPROVED.  Returns intent on success."""
        # First expire any stale ones, then attempt transition
        self._store.expire_old_approvals()
        row = self._store.get_approval(approval_id)
        if row is None:
            return ApprovalDecision(ok=False, status="NOT_FOUND")
        status = row.get("status") or ""
        if status != "PENDING":
            return ApprovalDecision(ok=False, status="NOT_PENDING",
                                     reason=f"current_status={status}")

        ok = self._store.decide_approval(
            approval_id=approval_id, decision="APPROVED",
            decided_by=decided_by,
        )
        if not ok:
            # Race: someone else flipped it (or it just expired)
            return ApprovalDecision(ok=False, status="EXPIRED")

        try:
            intent = TradeIntent.from_json(row["intent"])
        except Exception:
            logger.exception("approval_intent_parse_failed id=%d",
                             approval_id)
            return ApprovalDecision(ok=False, status="APPROVED",
                                     reason="intent_parse_failed")
        return ApprovalDecision(ok=True, status="APPROVED",
                                 intent=intent)

    def deny(self, approval_id: int, *,
              decided_by: str = "telegram") -> ApprovalDecision:
        self._store.expire_old_approvals()
        row = self._store.get_approval(approval_id)
        if row is None:
            return ApprovalDecision(ok=False, status="NOT_FOUND")
        status = row.get("status") or ""
        if status != "PENDING":
            return ApprovalDecision(ok=False, status="NOT_PENDING",
                                     reason=f"current_status={status}")
        ok = self._store.decide_approval(
            approval_id=approval_id, decision="DENIED",
            decided_by=decided_by,
        )
        return ApprovalDecision(ok=ok, status="DENIED" if ok else "EXPIRED")

    # ── Execution wrappers (called after approve, by executor) ────────

    def check_drift(self, intent: TradeIntent,
                     current_price: float) -> Optional[float]:
        """Return drift fraction if > MAX_DRIFT_PCT, else None.

        Caller uses this gate after /yes to decide whether to submit at
        a now-different price than the intent was approved at.
        """
        if intent.entry_price <= 0:
            return None
        drift = abs(current_price - intent.entry_price) / intent.entry_price
        if drift > MAX_DRIFT_PCT:
            return drift
        return None

    def mark_executed(self, *, approval_id: int,
                       position_id: int) -> None:
        try:
            self._store.mark_approval_executed(
                approval_id=approval_id, position_id=position_id,
            )
        except Exception:
            logger.exception("mark_approval_executed_failed id=%d",
                             approval_id)

    def mark_stale(self, approval_id: int,
                    reason: str = "price_drift") -> None:
        try:
            self._store.mark_approval_stale(
                approval_id=approval_id, reason=reason,
            )
        except Exception:
            logger.exception("mark_approval_stale_failed id=%d",
                             approval_id)
