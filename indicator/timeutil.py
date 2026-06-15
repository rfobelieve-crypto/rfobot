"""Display-layer timezone helper — UTC+8 Taipei.

Storage stays UTC everywhere (absolute time). This module is ONLY for
formatting timestamps shown to humans (dashboard labels, Telegram reports).
Do NOT use it for DB writes, query boundaries, API `_utc` fields, or any
analysis bucketing — those must remain UTC.

`to_tpe` is idempotent-safe: naive datetimes are treated as UTC, tz-aware
ones are converted. Works for both python datetime and pandas Timestamp.
"""
from datetime import timezone, timedelta

TZ_TPE = timezone(timedelta(hours=8))


def to_tpe(dt):
    """naive datetime -> assume UTC; tz-aware -> convert. Returns UTC+8 aware.

    None-safe (returns None). Accepts python datetime or pandas Timestamp.
    """
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_TPE)


def fmt_tpe(dt, fmt="%m/%d %H:%M"):
    """Format a UTC/naive/aware datetime as a UTC+8 string. '' if None."""
    d = to_tpe(dt)
    return d.strftime(fmt) if d is not None else ""
