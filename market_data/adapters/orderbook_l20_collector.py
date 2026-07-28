"""
Orderbook L20 snapshot collector for Binance perp.

Fetches full L20 depth via REST every 60s and writes one row per
(exchange, symbol, timestamp) into `orderbook_snapshots_1m`.

This is the data-source for Phase 3 R&D features (orderbook L20 imbalance,
wall detection, depth velocity).  Production v7 indicator does NOT depend
on this table — pure historical accumulation for future feature engineering.

API:
    Binance: GET https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=20

Pattern follows oi_collector.py:
  - collect_once() : fetch + insert one row per tracked symbol
  - main()         : while True: collect_once(); sleep 60s
  - BTC_perp_data.py wraps this in a daemon thread

Schema (orderbook_snapshots_1m): see _ensure_schema() in this file.
"""
from __future__ import annotations
import json
import time
import logging
import requests

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

COLLECT_INTERVAL = 60  # seconds

# (exchange, api_symbol, canonical_symbol)
# 2026-07-28: extended from BTC+ETH to 11 instruments so the cancel-flow
# playbooks can run per coin (they need mid -> ret_1m from this table).
# Cost check before adding: fapi depth?limit=20 is weight 2, so 11 symbols
# every 60s is 22 weight/min against a 2400/min budget — not close.
# The real cost is storage: this is the largest table in the DB (~1.7 KB per
# row), so migration 018 gives it a 120-day retention to match flow_bars_1m.
TRACKED = [
    ("binance", "BTCUSDT", "BTC-USD"),
    ("binance", "ETHUSDT", "ETH-USD"),
    ("binance", "SOLUSDT", "SOL-USD"),
    ("binance", "XRPUSDT", "XRP-USD"),
    ("binance", "DOGEUSDT", "DOGE-USD"),
    ("binance", "ADAUSDT", "ADA-USD"),
    ("binance", "BNBUSDT", "BNB-USD"),
    ("binance", "LINKUSDT", "LINK-USD"),
    ("binance", "SUIUSDT", "SUI-USD"),
    ("binance", "UNIUSDT", "UNI-USD"),
    ("binance", "AAVEUSDT", "AAVE-USD"),
]

DEPTH_LIMIT = 20  # L20


def _ensure_schema():
    """Create orderbook_snapshots_1m table if missing.  Safe to call repeatedly."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots_1m (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(20) NOT NULL,
                canonical_symbol VARCHAR(20) NOT NULL,
                ts_ms BIGINT NOT NULL,
                mid_price DOUBLE NOT NULL,
                spread_bps DOUBLE NOT NULL,
                bid_l1_price DOUBLE,
                bid_l1_qty DOUBLE,
                ask_l1_price DOUBLE,
                ask_l1_qty DOUBLE,
                bid_depth_usd_l5 DOUBLE,
                ask_depth_usd_l5 DOUBLE,
                imbalance_l5 DOUBLE,
                bid_depth_usd_l20 DOUBLE,
                ask_depth_usd_l20 DOUBLE,
                imbalance_l20 DOUBLE,
                bid_max_wall_usd DOUBLE,
                ask_max_wall_usd DOUBLE,
                bid_wall_distance_bps DOUBLE,
                ask_wall_distance_bps DOUBLE,
                raw_levels JSON DEFAULT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_lookup (canonical_symbol, exchange, ts_ms),
                INDEX idx_ts (ts_ms),
                UNIQUE KEY uq_snapshot (exchange, canonical_symbol, ts_ms)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            logger.info("orderbook_snapshots_1m table ready")
    except Exception:
        logger.exception("ensure_schema failed for orderbook_snapshots_1m")
    finally:
        conn.close()


def _fetch_binance_depth(symbol: str) -> dict | None:
    """Fetch full L20 depth from Binance perp."""
    url = "https://fapi.binance.com/fapi/v1/depth"
    try:
        resp = requests.get(url, params={"symbol": symbol, "limit": DEPTH_LIMIT},
                              timeout=10)
        data = resp.json()
        if "bids" not in data or "asks" not in data:
            logger.warning("Binance depth: bad response for %s: %s",
                            symbol, str(data)[:200])
            return None
        return data
    except Exception:
        logger.exception("Failed to fetch Binance depth for %s", symbol)
        return None


def _compute_metrics(book: dict) -> dict | None:
    """Compute imbalance / depth / wall metrics from raw orderbook."""
    bids = [(float(p), float(q)) for p, q in book.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in book.get("asks", [])]
    if not bids or not asks:
        return None

    bid_l1_p, bid_l1_q = bids[0]
    ask_l1_p, ask_l1_q = asks[0]
    mid = (bid_l1_p + ask_l1_p) / 2
    if mid <= 0:
        return None
    spread_bps = (ask_l1_p - bid_l1_p) / mid * 10000

    def depth_usd(side: list, k: int) -> float:
        return sum(p * q for p, q in side[:k])

    bid_l5 = depth_usd(bids, 5)
    ask_l5 = depth_usd(asks, 5)
    bid_l20 = depth_usd(bids, 20)
    ask_l20 = depth_usd(asks, 20)

    def imb(b: float, a: float) -> float:
        return (b - a) / (b + a) if (b + a) > 0 else 0.0

    # Wall detection: largest single level in USD
    bid_levels_usd = [(p * q, p) for p, q in bids]
    ask_levels_usd = [(p * q, p) for p, q in asks]
    bid_max_usd, bid_max_price = max(bid_levels_usd, key=lambda x: x[0])
    ask_max_usd, ask_max_price = max(ask_levels_usd, key=lambda x: x[0])

    # Distance from mid (bps) — if wall is far away, less immediate influence
    bid_wall_dist = (mid - bid_max_price) / mid * 10000
    ask_wall_dist = (ask_max_price - mid) / mid * 10000

    return {
        "mid_price": mid,
        "spread_bps": spread_bps,
        "bid_l1_price": bid_l1_p,
        "bid_l1_qty": bid_l1_q,
        "ask_l1_price": ask_l1_p,
        "ask_l1_qty": ask_l1_q,
        "bid_depth_usd_l5": bid_l5,
        "ask_depth_usd_l5": ask_l5,
        "imbalance_l5": imb(bid_l5, ask_l5),
        "bid_depth_usd_l20": bid_l20,
        "ask_depth_usd_l20": ask_l20,
        "imbalance_l20": imb(bid_l20, ask_l20),
        "bid_max_wall_usd": bid_max_usd,
        "ask_max_wall_usd": ask_max_usd,
        "bid_wall_distance_bps": bid_wall_dist,
        "ask_wall_distance_bps": ask_wall_dist,
    }


def _insert(exchange: str, canonical: str, ts_ms: int, m: dict,
            raw_levels: dict | None = None) -> None:
    """Insert one snapshot row.  Idempotent via UNIQUE (exchange, canonical, ts_ms)."""
    sql = """
    INSERT IGNORE INTO orderbook_snapshots_1m
        (exchange, canonical_symbol, ts_ms, mid_price, spread_bps,
         bid_l1_price, bid_l1_qty, ask_l1_price, ask_l1_qty,
         bid_depth_usd_l5, ask_depth_usd_l5, imbalance_l5,
         bid_depth_usd_l20, ask_depth_usd_l20, imbalance_l20,
         bid_max_wall_usd, ask_max_wall_usd,
         bid_wall_distance_bps, ask_wall_distance_bps,
         raw_levels)
    VALUES (%s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s)
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                exchange, canonical, ts_ms,
                m["mid_price"], m["spread_bps"],
                m["bid_l1_price"], m["bid_l1_qty"],
                m["ask_l1_price"], m["ask_l1_qty"],
                m["bid_depth_usd_l5"], m["ask_depth_usd_l5"], m["imbalance_l5"],
                m["bid_depth_usd_l20"], m["ask_depth_usd_l20"], m["imbalance_l20"],
                m["bid_max_wall_usd"], m["ask_max_wall_usd"],
                m["bid_wall_distance_bps"], m["ask_wall_distance_bps"],
                json.dumps(raw_levels) if raw_levels is not None else None,
            ))
    except Exception:
        logger.exception("insert orderbook snapshot failed (%s %s %d)",
                          exchange, canonical, ts_ms)
    finally:
        conn.close()


def collect_once() -> int:
    """One-shot: fetch all tracked symbols, write one row each.  Returns insert count."""
    n_inserted = 0
    ts_ms = int(time.time() * 1000)
    for exchange, api_sym, canonical in TRACKED:
        if exchange != "binance":
            continue  # only Binance supported for now
        book = _fetch_binance_depth(api_sym)
        if not book:
            continue
        m = _compute_metrics(book)
        if not m:
            continue
        # raw_levels: keep compact form for replay (just price-qty pairs)
        raw_compact = {
            "bids": book.get("bids", [])[:20],
            "asks": book.get("asks", [])[:20],
        }
        _insert(exchange, canonical, ts_ms, m, raw_compact)
        n_inserted += 1
    return n_inserted


def main() -> None:
    """Standalone runner — kept for ad-hoc local testing.
    Production usage: BTC_perp_data.py wraps collect_once() in a daemon thread."""
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    _ensure_schema()
    logger.info("Orderbook L20 collector starting (every %ds)", COLLECT_INTERVAL)
    while True:
        try:
            n = collect_once()
            logger.info("Inserted %d orderbook snapshots", n)
        except Exception:
            logger.exception("Orderbook collect_once failed")
        time.sleep(COLLECT_INTERVAL)


if __name__ == "__main__":
    main()
