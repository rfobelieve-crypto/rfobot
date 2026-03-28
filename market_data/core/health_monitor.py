"""
Health monitor: track data freshness and connectivity per source.
"""

import time
import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# {source_key: health_dict}
# source_key = "binance" or "okx"
_sources: dict[str, dict] = {}


def _get_or_create(source: str) -> dict:
    if source not in _sources:
        _sources[source] = {
            "source": source,
            "last_message_time": 0,
            "reconnect_count": 0,
            "gap_count": 0,
            "latency_ms": 0,
            "status": "down",
            "message_count": 0,
        }
    return _sources[source]


def on_trade(source: str, ts_exchange: int, ts_received: int):
    """Call this for every received trade to update health stats."""
    with _lock:
        h = _get_or_create(source)
        h["last_message_time"] = ts_received
        h["message_count"] += 1
        h["latency_ms"] = max(0, ts_received - ts_exchange)
        h["status"] = "healthy"


def on_reconnect(source: str):
    """Call when a WebSocket reconnects."""
    with _lock:
        h = _get_or_create(source)
        h["reconnect_count"] += 1


def on_gap(source: str):
    """Call when a gap in data is detected."""
    with _lock:
        h = _get_or_create(source)
        h["gap_count"] += 1


def check_staleness(stale_threshold_s: float = 30.0):
    """
    Check all sources for staleness.
    Mark as degraded if no message for stale_threshold_s,
    mark as down if > 3x threshold.
    """
    now_ms = int(time.time() * 1000)
    threshold_ms = int(stale_threshold_s * 1000)

    with _lock:
        for h in _sources.values():
            if h["last_message_time"] == 0:
                h["status"] = "down"
                continue

            gap = now_ms - h["last_message_time"]
            if gap > threshold_ms * 3:
                h["status"] = "down"
            elif gap > threshold_ms:
                h["status"] = "degraded"
            else:
                h["status"] = "healthy"


def get_status() -> dict[str, dict]:
    """Return snapshot of all source health."""
    with _lock:
        return {k: dict(v) for k, v in _sources.items()}


def get_source_status(source: str) -> dict:
    """Return health for a single source."""
    with _lock:
        h = _get_or_create(source)
        return dict(h)
