"""
1-minute flow bar aggregation — multi-scope.

Each normalized trade is written to TWO buckets in parallel:
  * ("all", canonical_symbol, minute_start) — combined across all
    exchanges (preserves existing indicator-service behaviour).
  * (exchange, canonical_symbol, minute_start) — per-venue bucket
    (new; enables cross-venue divergence features).

On flush, both scopes emit flow bars with their own CVD stream carried
over across minutes. Rows are keyed in the DB by
(canonical_symbol, instrument_type, exchange_scope, window_start).
"""
from __future__ import annotations

import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# {(canonical_symbol, exchange_scope, minute_start_ms): bucket_dict}
_buckets: dict[tuple[str, str, int], dict] = {}

# CVD carry-over per (canonical_symbol, exchange_scope): last flushed cvd value
_last_cvd: dict[tuple[str, str], float] = {}


def _minute_start(ts_ms: int) -> int:
    """Round down timestamp (ms) to the start of its minute."""
    return (ts_ms // 60_000) * 60_000


def _make_bucket(canonical_symbol: str, exchange_scope: str,
                 minute_start_ms: int) -> dict:
    return {
        "canonical_symbol": canonical_symbol,
        "instrument_type": "perp",
        "exchange_scope": exchange_scope,
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


def _add_to_bucket(cs: str, scope: str, ms: int, notional: float,
                   side: str, exchange: str):
    key = (cs, scope, ms)
    if key not in _buckets:
        _buckets[key] = _make_bucket(cs, scope, ms)
    b = _buckets[key]
    if side == "buy":
        b["buy_notional_usd"] += notional
    else:
        b["sell_notional_usd"] += notional
    b["trade_count"] += 1
    b["_sources"].add(exchange)


def add_trade(trade: dict):
    """
    Add a normalized trade to the aggregator. Writes to both the combined
    "all" bucket and the per-exchange bucket.

    trade must have: canonical_symbol, taker_side, notional_usd,
                     ts_exchange, exchange
    """
    cs = trade["canonical_symbol"]
    ms = _minute_start(trade["ts_exchange"])
    notional = trade["notional_usd"]
    side = trade["taker_side"]
    exchange = trade["exchange"]

    with _lock:
        _add_to_bucket(cs, "all", ms, notional, side, exchange)
        _add_to_bucket(cs, exchange, ms, notional, side, exchange)


def flush(now_ms: int | None = None) -> list[dict]:
    """
    Flush all completed bars (window_end <= now_ms).

    Returns list of flow bar dicts ready for DB insert — one row per
    (canonical_symbol, exchange_scope, minute_start) tuple.
    """
    import time as _time
    if now_ms is None:
        now_ms = int(_time.time() * 1000)

    completed = []

    with _lock:
        keys_to_remove = [k for k, b in _buckets.items() if b["window_end"] <= now_ms]

        for key in keys_to_remove:
            b = _buckets.pop(key)
            cs = b["canonical_symbol"]
            scope = b["exchange_scope"]

            # Finalize computed fields
            b["delta_usd"] = b["buy_notional_usd"] - b["sell_notional_usd"]
            b["volume_usd"] = b["buy_notional_usd"] + b["sell_notional_usd"]
            b["source_count"] = len(b["_sources"])

            # CVD: carry over from last flushed bar for this (symbol, scope)
            cvd_key = (cs, scope)
            prev_cvd = _last_cvd.get(cvd_key, 0.0)
            b["cvd_usd"] = prev_cvd + b["delta_usd"]
            _last_cvd[cvd_key] = b["cvd_usd"]

            # Clean internal fields
            del b["_sources"]

            completed.append(b)

    if completed:
        logger.info("Flushed %d flow bars (combined + per-exchange)", len(completed))

    return completed


def stats() -> dict:
    """Return current aggregator state for debugging."""
    with _lock:
        scopes = set(k[1] for k in _buckets)
        return {
            "active_buckets": len(_buckets),
            "symbols": list(set(k[0] for k in _buckets)),
            "scopes": sorted(scopes),
            "last_cvd_keys": list(_last_cvd.keys()),
        }
