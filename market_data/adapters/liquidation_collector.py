"""
Liquidation data collector for OKX + Binance.

Subscribes to liquidation WebSocket feeds, aggregates into 1-minute buckets,
and flushes to liquidation_1m table every 10 seconds.

OKX:     wss://ws.okx.com:8443/ws/v5/public  (channel: liquidation-orders, instType: SWAP)
Binance: wss://fstream.binance.com/stream?streams=btcusdt@forceOrder/ethusdt@forceOrder

Bucket schema:
  liq_sell_usd  — notional of liquidated LONG positions (forced sell orders)
  liq_buy_usd   — notional of liquidated SHORT positions (forced buy orders)
  liq_total_usd — combined notional
"""

import json
import time
import threading
import logging

import websocket

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

OKX_WS_URL     = "wss://ws.okx.com:8443/ws/v5/public"
BINANCE_WS_URL = "wss://fstream.binance.com/stream?streams=btcusdt@forceOrder/ethusdt@forceOrder"

OKX_SYMBOL_MAP = {
    "BTC-USDT-SWAP": "BTC-USD",
    "ETH-USDT-SWAP": "ETH-USD",
}
BINANCE_SYMBOL_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}
OKX_CONTRACT_SIZE = {
    "BTC-USDT-SWAP": 0.01,
    "ETH-USDT-SWAP": 0.1,
}

# In-memory buckets: {canonical: {minute_start_ms: {liq_buy_usd, liq_sell_usd, ...}}}
_buckets: dict = {}
_lock = threading.Lock()

FLUSH_INTERVAL = 10  # seconds


def _minute_start(ts_ms: int) -> int:
    return (ts_ms // 60_000) * 60_000


def _add_liq(canonical: str, ts_ms: int, side: str, notional_usd: float):
    """
    Add a liquidation event to the in-memory bucket.

    side = "sell" → long position liquidated (sell order executed)
    side = "buy"  → short position liquidated (buy order executed)
    """
    minute = _minute_start(ts_ms)
    with _lock:
        sym = _buckets.setdefault(canonical, {})
        if minute not in sym:
            sym[minute] = {
                "liq_buy_usd":   0.0,
                "liq_sell_usd":  0.0,
                "liq_total_usd": 0.0,
                "liq_count":     0,
            }
        bucket = sym[minute]
        if side == "sell":
            bucket["liq_sell_usd"] += notional_usd
        else:
            bucket["liq_buy_usd"] += notional_usd
        bucket["liq_total_usd"] += notional_usd
        bucket["liq_count"] += 1


def _flush_completed_buckets():
    """Flush completed (past) minute buckets to DB."""
    now_ms = int(time.time() * 1000)
    current_minute = _minute_start(now_ms)

    to_flush = []
    with _lock:
        for canonical, minutes in list(_buckets.items()):
            for minute_start, data in list(minutes.items()):
                if minute_start < current_minute:
                    to_flush.append((canonical, minute_start, dict(data)))
                    del _buckets[canonical][minute_start]

    if not to_flush:
        return

    sql = """
    INSERT INTO liquidation_1m
        (canonical_symbol, window_start, liq_buy_usd, liq_sell_usd, liq_total_usd, liq_count)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        liq_buy_usd   = liq_buy_usd   + VALUES(liq_buy_usd),
        liq_sell_usd  = liq_sell_usd  + VALUES(liq_sell_usd),
        liq_total_usd = liq_total_usd + VALUES(liq_total_usd),
        liq_count     = liq_count     + VALUES(liq_count)
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for canonical, window_start, data in to_flush:
                    cur.execute(sql, (
                        canonical, window_start,
                        data["liq_buy_usd"], data["liq_sell_usd"],
                        data["liq_total_usd"], data["liq_count"],
                    ))
        finally:
            conn.close()
        logger.debug("Flushed %d liquidation buckets", len(to_flush))
    except Exception:
        logger.exception("Failed to flush liquidation buckets")


def flush_loop():
    logger.info("Liquidation flusher starting (every %ds)", FLUSH_INTERVAL)
    while True:
        time.sleep(FLUSH_INTERVAL)
        try:
            _flush_completed_buckets()
        except Exception:
            logger.exception("Liquidation flush error")


# ──────────────────────────────────────────────
# OKX WebSocket adapter
# ──────────────────────────────────────────────

class OKXLiquidationAdapter:
    def __init__(self):
        self._running = False

    def start(self):
        self._running = True
        reconnect_delay = 5
        while self._running:
            try:
                logger.info("[OKX-Liq] Connecting...")
                ws = websocket.WebSocketApp(
                    OKX_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
            except Exception:
                logger.exception("[OKX-Liq] Connection error")
            if not self._running:
                break
            logger.warning("[OKX-Liq] Reconnecting in %ds...", reconnect_delay)
            time.sleep(reconnect_delay)

    def _on_open(self, ws):
        ws.send(json.dumps({
            "op": "subscribe",
            "args": [{"channel": "liquidation-orders", "instType": "SWAP"}],
        }))
        logger.info("[OKX-Liq] Subscribed to liquidation-orders SWAP")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "data" not in data:
                return
            for item in data["data"]:
                inst_id   = item.get("instId", "")
                canonical = OKX_SYMBOL_MAP.get(inst_id)
                if not canonical:
                    continue
                contract_size = OKX_CONTRACT_SIZE.get(inst_id, 0.01)
                for detail in item.get("details", []):
                    side   = detail.get("side", "")  # "sell"=long liq, "buy"=short liq
                    sz     = float(detail.get("sz", 0))
                    bk_px  = float(detail.get("bkPx", 0))
                    ts_ms  = int(detail.get("ts", time.time() * 1000))
                    notional = sz * bk_px * contract_size
                    if notional > 0:
                        _add_liq(canonical, ts_ms, side, notional)
        except Exception:
            logger.exception("[OKX-Liq] on_message error")

    def _on_error(self, ws, error):
        logger.error("[OKX-Liq] WS error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.warning("[OKX-Liq] WS closed: code=%s msg=%s", code, msg)


# ──────────────────────────────────────────────
# Binance WebSocket adapter
# ──────────────────────────────────────────────

class BinanceLiquidationAdapter:
    def __init__(self):
        self._running = False

    def start(self):
        self._running = True
        reconnect_delay = 5
        while self._running:
            try:
                logger.info("[Binance-Liq] Connecting...")
                ws = websocket.WebSocketApp(
                    BINANCE_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
            except Exception:
                logger.exception("[Binance-Liq] Connection error")
            if not self._running:
                break
            logger.warning("[Binance-Liq] Reconnecting in %ds...", reconnect_delay)
            time.sleep(reconnect_delay)

    def _on_open(self, ws):
        logger.info("[Binance-Liq] Connected, receiving forceOrder streams")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "stream" in data:
                data = data["data"]
            if data.get("e") != "forceOrder":
                return
            order     = data.get("o", {})
            raw_sym   = order.get("s", "")
            canonical = BINANCE_SYMBOL_MAP.get(raw_sym)
            if not canonical:
                return
            # S: "SELL" = long liquidated, "BUY" = short liquidated
            raw_side = order.get("S", "").upper()
            side     = "sell" if raw_side == "SELL" else "buy"
            qty      = float(order.get("q", 0))
            avg_px   = float(order.get("ap") or order.get("p", 0))
            ts_ms    = int(order.get("T", time.time() * 1000))
            notional = qty * avg_px
            if notional > 0:
                _add_liq(canonical, ts_ms, side, notional)
        except Exception:
            logger.exception("[Binance-Liq] on_message error")

    def _on_error(self, ws, error):
        logger.error("[Binance-Liq] WS error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.warning("[Binance-Liq] WS closed: code=%s msg=%s", code, msg)


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def start_all():
    """Start OKX + Binance liquidation WS adapters and flush loop as daemon threads."""
    threading.Thread(target=OKXLiquidationAdapter().start,
                     daemon=True, name="okx-liq").start()
    threading.Thread(target=BinanceLiquidationAdapter().start,
                     daemon=True, name="binance-liq").start()
    threading.Thread(target=flush_loop,
                     daemon=True, name="liq-flusher").start()
    logger.info("Liquidation collectors started (OKX + Binance + flusher)")
