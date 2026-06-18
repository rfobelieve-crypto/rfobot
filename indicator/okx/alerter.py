"""Telegram critical-alert helper for OKX executor.

Self-contained: doesn't import indicator/app.py to avoid circular deps.
Best-effort: logs and continues on any send failure (alerter must NEVER
itself crash the executor — that would silence the very alerts we want).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


def send_critical(chat_id: str, message: str, *,
                  timeout_sec: float = 5.0) -> bool:
    """Send a critical-channel Telegram message.

    Returns True on success, False on any failure (logged but swallowed).
    Reads TELEGRAM_BOT_TOKEN from env at call time so unit tests can
    patch it.
    """
    if not chat_id:
        logger.warning("telegram_critical_skipped no_chat_id")
        return False
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.warning("telegram_critical_skipped no_token_in_env")
        return False
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message,
                  "parse_mode": "Markdown"},
            timeout=timeout_sec,
        )
        if resp.status_code == 200:
            return True
        # A Markdown parse error (400 "can't parse entities") would otherwise
        # SILENTLY eat the alert — e.g. an exit reason like "opp_signal" /
        # "trail_stop" / "time_cap" carries an unbalanced '_' (italic marker in
        # legacy Markdown), which silently dropped EVERY exit notification live.
        # A critical alert must never be lost to a formatting bug, so retry once
        # as plain text (no parse_mode).  Other codes (429/5xx) aren't fixed by
        # dropping parse_mode, so don't double-send.
        if resp.status_code == 400:
            logger.warning(
                "telegram_critical_markdown_failed body=%s; retry plain",
                resp.text[:200])
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": message},
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                return True
        logger.warning("telegram_critical_failed status=%d body=%s",
                       resp.status_code, resp.text[:200])
    except Exception:
        logger.exception("telegram_critical_exception")
    return False


def format_kill_alert(*, trigger_id: str, severity: str, reason: str,
                       stage_label: str = "testnet",
                       context: Optional[dict] = None) -> str:
    """Compose a structured kill-trigger alert message."""
    ctx_lines = ""
    if context:
        items = list(context.items())[:5]   # cap noise
        ctx_lines = "\n" + "\n".join(f"  {k}: {v}" for k, v in items)
    return (
        f"*OKX {stage_label.upper()} kill trigger*\n"
        f"trigger: `{trigger_id}` ({severity})\n"
        f"reason: {reason}{ctx_lines}"
    )


def format_entry_alert(*, stage_label: str, direction: str, tier: str,
                        entry_price: float, size_contracts: int,
                        notional_usd: float, stop_price: float,
                        atr: float) -> str:
    """Safety belt #8 — every entry must produce a Telegram push."""
    return (
        f"*OKX {stage_label.upper()} ENTRY*\n"
        f"{direction} {tier} @ {entry_price:.2f}\n"
        f"size: {size_contracts} contracts  notional: ${notional_usd:.2f}\n"
        f"stop: {stop_price:.2f}  ATR(14): {atr:.2f}"
    )


def format_exit_alert(*, stage_label: str, direction: str, reason: str,
                       entry_price: float, exit_price: float,
                       gross_pct: float, net_pct: float,
                       equity_after: float) -> str:
    """Exit (manual close — algo-stop fills routed via WS event).

    `reason` is backtick-wrapped: exit reasons (opp_signal / trail_stop /
    time_cap) contain '_', which is an italic marker in legacy Markdown — an
    unbalanced '_' makes Telegram reject the whole message (400), silently
    dropping the alert.  A code span renders the underscore literally.
    """
    return (
        f"*OKX {stage_label.upper()} EXIT* (`{reason}`)\n"
        f"{direction} {entry_price:.2f} → {exit_price:.2f}\n"
        f"gross {gross_pct * 100:+.2f}%  net {net_pct * 100:+.2f}%\n"
        f"equity: ${equity_after:.2f}"
    )


def format_approval_request(*, approval_id: int, stage_label: str,
                              direction: str, tier: str,
                              entry_price: float,
                              size_contracts: int,
                              notional_usd: float,
                              stop_price: float,
                              atr: float,
                              count_so_far: int,
                              threshold: int) -> str:
    """Telegram message asking the operator to confirm a pending trade.

    The operator replies with /yes_<id> or /no_<id> (the suffix lets
    multiple pending approvals be unambiguous).  count_so_far/threshold
    shows progress toward auto-mode unlock.
    """
    return (
        f"*OKX {stage_label.upper()} PENDING #{approval_id}*\n"
        f"{direction} {tier} {size_contracts} contracts @ {entry_price:.2f}\n"
        f"notional: ${notional_usd:.2f}  stop: {stop_price:.2f}\n"
        f"ATR(14): {atr:.2f}\n"
        f"Reply: `/yes_{approval_id}` or `/no_{approval_id}`\n"
        f"(manual approvals: {count_so_far}/{threshold} before auto)"
    )
