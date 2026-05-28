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
