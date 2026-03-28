"""
1-minute flow bar aggregation.

Accumulates normalized trades into per-minute buckets keyed by
(canonical_symbol, minute_start). On flush, completed bars are returned
and removed from memory.
"""

import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# {(canonical_symbol, minute_start_ms): bucket_dict}
_buckets: dict[tuple[str, int], dict] = {}

# CVD carry-over per canonical_symbol: last flushed cvd value
_last_cvd: dict[str, float] = {}


def _minute_start(ts_ms: int) -> int:
    """Round down timestamp (ms) to the start of its minute."""
    return (ts_ms // 60_000) * 60_000


def _make_bucket(canonical_symbol: str, minute_start_ms: int) -> dict:
    return {
        "canonical_symbol": canonical_symbol,
        "instrument_type": "perp",
        "exchange_scope": "all",
        "window_start": minute_start_ms,
        "window_end": minute_start_ms + 60_000,
        "buy_notional_usd": 0.0,
        "sell_notional_usd": 0.0,
        "delta_usd": 0.0,
        "volume_usd": 0.0,
        "trade_count": 0,
        "cvd_usd": 0.0,
        "source_count": 0,
        "quality_score": 1.0,
        "_sources": set(),
    }


def add_trade(trade: dict):
    """
    Add a normalized trade to the aggregator.

    trade must have: canonical_symbol, taker_side, notional_usd,
                     ts_exchange, exchange
    """
    cs = trade["canonical_symbol"]
    ms = _minute_start(trade["ts_exchange"])
    key = (cs, ms)

    with _lock:
        if key not in _buckets:
            _buckets[key] = _make_bucket(cs, ms)

        b = _buckets[key]
        notional = trade["notional_usd"]

        if trade["taker_side"] == "buy":
            b["buy_notional_usd"] += notional
        else:
            b["sell_notional_usd"] += notional

        b["trade_count"] += 1
        b["_sources"].add(trade["exchange"])


def flush(now_ms: int | None = None) -> list[dict]:
    """
    Flush all completed bars (window_end <= now_ms).

    Returns list of flow bar dicts ready for DB insert.
    If now_ms is None, uses current time.
    """
    import time as _time
    if now_ms is None:
        now_ms = int(_time.time() * 1000)

    completed = []

    with _lock:
        keys_to_remove = []
        for key, b in _buckets.items():
            if b["window_end"] <= now_ms:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            b = _buckets.pop(key)
            cs = b["canonical_symbol"]

            # Finalize computed fields
            b["delta_usd"] = b["buy_notional_usd"] - b["sell_notional_usd"]
            b["volume_usd"] = b["buy_notional_usd"] + b["sell_notional_usd"]
            b["source_count"] = len(b["_sources"])

            # CVD: carry over from last flushed bar
            prev_cvd = _last_cvd.get(cs, 0.0)
            b["cvd_usd"] = prev_cvd + b["delta_usd"]
            _last_cvd[cs] = b["cvd_usd"]

            # Clean internal fields
            del b["_sources"]

            completed.append(b)

    if completed:
        logger.info("Flushed %d flow bars", len(completed))

    return completed


def stats() -> dict:
    """Return current aggregator state for debugging."""
    with _lock:
        return {
            "active_buckets": len(_buckets),
            "symbols": list(set(k[0] for k in _buckets)),
            "last_cvd": dict(_last_cvd),
        }
