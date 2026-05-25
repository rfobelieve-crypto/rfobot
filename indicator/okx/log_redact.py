"""Runtime credential redaction for the logging system.

Defense in depth: our code is careful not to log api_key / secret /
passphrase, but this filter ensures that even if future code (or a
careless `logger.debug(f"submitting {body}")`) tries to, the actual
log line gets redacted before hitting stdout / log aggregator.

Install via:
    import logging
    from indicator.okx.log_redact import install_redaction_filter
    install_redaction_filter(logging.getLogger())  # root logger

Or per-logger:
    install_redaction_filter(logging.getLogger("indicator.okx"))
"""
from __future__ import annotations

import logging
import re

REDACTED = "***REDACTED***"


# Each pattern matches credential-shaped content and replaces the value
# while preserving the key name for debug context.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # api_key="..." / api_key: "..." / apiKey = "..."
    (re.compile(
        r"""(?i)(api[_-]?key|api_secret|apisecret|passphrase|api[_-]?passphrase)"""
        r"""(['"\s:=]+)"""
        r"""(['"]?)([A-Za-z0-9+/=_\-\.]{16,})\3"""
    ), rf"\1\2\3{REDACTED}\3"),

    # OKX request headers
    (re.compile(
        r"""(OK-ACCESS-(?:KEY|SIGN|PASSPHRASE|TIMESTAMP))"""
        r"""(['"\s:]+)"""
        r"""(['"]?)([^\s,'"}\]]+)\3"""
    ), rf"\1\2\3{REDACTED}\3"),

    # Anthropic key
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), REDACTED),

    # OpenAI key
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b"), REDACTED),

    # AWS / GitHub patterns
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), REDACTED),

    # Telegram bot token format
    (re.compile(r"\b\d{8,11}:[A-Za-z0-9_\-]{35}\b"), REDACTED),

    # OKX api_key UUID format when standalone
    (re.compile(
        r"""(['"])([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\1"""
    ), rf"\1{REDACTED}\1"),
]


def _redact(text: str) -> str:
    if not text:
        return text
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class CredentialRedactionFilter(logging.Filter):
    """logging.Filter that redacts credential-shaped substrings.

    Applied after the format step would have been the cleanest, but
    Python's logging only exposes pre-format intervention via Filter.
    We format eagerly here so the redaction sees the final string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Pre-format args into message; downstream formatter sees redacted str.
            if record.args:
                try:
                    formatted = record.msg % record.args
                except Exception:
                    formatted = f"{record.msg} {record.args}"
                record.msg = _redact(formatted)
                record.args = ()
            else:
                record.msg = _redact(str(record.msg))
        except Exception:
            # NEVER let the filter break logging.
            pass
        return True


def install_redaction_filter(logger: logging.Logger) -> None:
    """Idempotent: only installs once per logger."""
    for f in logger.filters:
        if isinstance(f, CredentialRedactionFilter):
            return
    logger.addFilter(CredentialRedactionFilter())


# Convenience: auto-install on root when this module is imported via
# OkxClient.  Users can also call explicitly for sub-loggers.
def auto_install_on_root() -> None:
    install_redaction_filter(logging.getLogger())
