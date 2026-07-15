"""Depth-delta collector — per-side add/cancel aggregates from Binance
incremental book (BTC-USD), 1-minute buckets.

Purpose: forward-collect TRUE liquidity-withdrawal data for the squeeze
"path of least resistance" research (research/squeeze_flow_join.py uses
1m L20 snapshot diffs as a proxy today; this table replaces the proxy
once 3-6 months of history accumulate).

Decomposition per price level between consecutive book updates:
    Δqty = adds − cancels − fills
Fills are not separable from cancels without the trade tape at level
granularity, so we record the NET decrease that exceeds traded volume
as cancellation pressure — the standard L2-only approximation:
    cancel_proxy_side  = Σ max(0, −Δqty_level)   (liquidity removed)
    add_side           = Σ max(0, +Δqty_level)   (liquidity added)
Aggregated per side per minute. Interpretation stays comparative
(bid vs ask within the same minute), which is all the research needs.

Architecture compliance (.claude/rules/architecture.md):
  · additive — new module, no existing adapter touched
  · own WS connection, raw parsing, DB flush; daemon-thread pattern
  · all MySQL via shared/db.get_db_conn()
Run modes:
  python -m market_data.adapters.depth_delta_collector      # standalone
  (wire into the market_data service main once validated — NOT auto-wired)
"""
from __future__ import annotations

import json
import logging
import threading
import time

import websocket

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"   # spot
PERP_WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth@100ms"  # USDT-M perp
CANONICAL = "BTC-USD"
FLUSH_SEC = 10


def _ensure_schema() -> None:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS depth_deltas_1m (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(20) NOT NULL,
                canonical_symbol VARCHAR(20) NOT NULL,
                minute_start_ms BIGINT NOT NULL,
                bid_add_qty DOUBLE NOT NULL DEFAULT 0,
                bid_cancel_qty DOUBLE NOT NULL DEFAULT 0,
                ask_add_qty DOUBLE NOT NULL DEFAULT 0,
                ask_cancel_qty DOUBLE NOT NULL DEFAULT 0,
                update_count INT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_bucket (exchange, canonical_symbol, minute_start_ms),
                INDEX idx_ts (minute_start_ms)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        conn.commit()
        logger.info("depth_deltas_1m table ready")
    finally:
        conn.close()


class DepthDeltaCollector:
    """Tracks per-level qty between diff messages, buckets add/cancel by minute.

    Defaults preserve the original spot stream (exchange='binance') exactly —
    the spot series must stay unbroken for the pre-registered cancel tests.
    Pass ws_url=PERP_WS_URL, exchange='binance_perp' for the parallel
    futures-book instance (2026-07-15).
    """

    def __init__(self, ws_url: str = WS_URL, exchange: str = "binance") -> None:
        self._ws_url = ws_url
        self._exchange = exchange
        self._book: dict[str, dict[float, float]] = {"bid": {}, "ask": {}}
        self._buckets: dict[int, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None

    # ── WS plumbing ────────────────────────────────────────────────────
    def _on_message(self, _ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
            ts_ms = int(msg.get("E", time.time() * 1000))
            minute = (ts_ms // 60_000) * 60_000
            with self._lock:
                b = self._buckets.setdefault(minute, dict(
                    bid_add=0.0, bid_cancel=0.0, ask_add=0.0,
                    ask_cancel=0.0, n=0))
                b["n"] += 1
                for side, key in (("bid", "b"), ("ask", "a")):
                    levels = self._book[side]
                    for px_s, qty_s in msg.get(key, []):
                        px, qty = float(px_s), float(qty_s)
                        prev = levels.get(px, 0.0)
                        d = qty - prev
                        if d > 0:
                            b[f"{side}_add"] += d
                        elif d < 0:
                            b[f"{side}_cancel"] += -d
                        if qty <= 0:
                            levels.pop(px, None)
                        else:
                            levels[px] = qty
        except Exception:
            logger.exception("depth_delta_parse_error")

    def _on_error(self, _ws, err) -> None:
        logger.warning("depth_delta_ws_error: %s", err)

    def _flush_loop(self) -> None:
        while True:
            time.sleep(FLUSH_SEC)
            try:
                self._flush()
            except Exception:
                logger.exception("depth_delta_flush_failed")

    def _flush(self) -> None:
        now_min = (int(time.time() * 1000) // 60_000) * 60_000
        with self._lock:
            done = {m: v for m, v in self._buckets.items() if m < now_min}
            for m in done:
                self._buckets.pop(m, None)
        if not done:
            return
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for m, v in sorted(done.items()):
                    cur.execute(
                        """
                        INSERT INTO depth_deltas_1m
                            (exchange, canonical_symbol, minute_start_ms,
                             bid_add_qty, bid_cancel_qty,
                             ask_add_qty, ask_cancel_qty, update_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            bid_add_qty = bid_add_qty + VALUES(bid_add_qty),
                            bid_cancel_qty = bid_cancel_qty + VALUES(bid_cancel_qty),
                            ask_add_qty = ask_add_qty + VALUES(ask_add_qty),
                            ask_cancel_qty = ask_cancel_qty + VALUES(ask_cancel_qty),
                            update_count = update_count + VALUES(update_count)
                        """,
                        (self._exchange, CANONICAL, m, v["bid_add"],
                         v["bid_cancel"], v["ask_add"], v["ask_cancel"], v["n"]))
            conn.commit()
            logger.info("depth_delta_flush exchange=%s minutes=%d",
                        self._exchange, len(done))
        finally:
            conn.close()

    def start(self) -> None:
        _ensure_schema()
        threading.Thread(target=self._flush_loop, daemon=True,
                         name="depth-delta-flush").start()
        while True:   # reconnect loop — same pattern as trade adapters
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url, on_message=self._on_message,
                    on_error=self._on_error)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                logger.exception("depth_delta_ws_crashed")
            # book state is stale after a gap — reset so deltas restart clean
            with self._lock:
                self._book = {"bid": {}, "ask": {}}
            logger.warning("depth_delta_ws_reconnect_in_5s")
            time.sleep(5)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["spot", "perp"], default="spot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.market == "perp":
        DepthDeltaCollector(ws_url=PERP_WS_URL, exchange="binance_perp").start()
    else:
        DepthDeltaCollector().start()


if __name__ == "__main__":
    main()
