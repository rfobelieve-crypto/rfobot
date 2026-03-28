"""
OKX perpetual futures trade adapter.

Connects to OKX public WebSocket, subscribes to trades channel
for tracked symbols, and emits raw parsed trade dicts.
"""

import json
import time
import logging
import threading

import websocket

from market_data.core.symbol_mapper import tracked_symbols
from market_data.core.health_monitor import on_trade, on_reconnect

logger = logging.getLogger(__name__)

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"


class OKXTradeAdapter:
    def __init__(self, on_trades_callback):
        """
        on_trades_callback: callable(list[dict]) - called with a batch of raw trades.
        Each raw trade dict has:
            exchange, raw_symbol, price, size, taker_side,
            trade_id, ts_exchange, ts_received, is_aggregated_trade
        """
        self.on_trades = on_trades_callback
        self.symbols = tracked_symbols("okx")
        self._running = False

    def start(self):
        """Start WebSocket connection in a background thread (blocking reconnect loop)."""
        self._running = True
        reconnect_delay = 5

        while self._running:
            try:
                logger.info("[OKX] Connecting to %s ...", OKX_WS_URL)
                ws = websocket.WebSocketApp(
                    OKX_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
            except Exception as e:
                logger.exception("[OKX] Connection error: %s", e)

            if not self._running:
                break

            on_reconnect("okx")
            logger.warning("[OKX] Reconnecting in %ds ...", reconnect_delay)
            time.sleep(reconnect_delay)

    def stop(self):
        self._running = False

    def _on_open(self, ws):
        logger.info("[OKX] Connected, subscribing to %d symbols", len(self.symbols))
        args = [{"channel": "trades", "instId": s} for s in self.symbols]
        ws.send(json.dumps({"op": "subscribe", "args": args}))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            # Skip subscribe confirmations and other events
            if "event" in data:
                return

            if "data" not in data or "arg" not in data:
                return

            channel = data["arg"].get("channel")
            if channel != "trades":
                return

            ts_received = int(time.time() * 1000)
            raw_trades = []

            for t in data["data"]:
                inst_id = t.get("instId", "")
                if inst_id not in self.symbols:
                    continue

                side = t.get("side", "").lower()
                if side not in ("buy", "sell"):
                    continue

                ts_exchange = int(t.get("ts", 0))

                raw_trades.append({
                    "exchange": "okx",
                    "raw_symbol": inst_id,
                    "price": t["px"],
                    "size": t["sz"],
                    "taker_side": side,
                    "trade_id": str(t.get("tradeId", "")),
                    "ts_exchange": ts_exchange,
                    "ts_received": ts_received,
                    "is_aggregated_trade": False,
                })

                on_trade("okx", ts_exchange, ts_received)

            if raw_trades:
                self.on_trades(raw_trades)

        except Exception:
            logger.exception("[OKX] on_message error")

    def _on_error(self, ws, error):
        logger.error("[OKX] WebSocket error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.warning("[OKX] WebSocket closed: code=%s msg=%s", code, msg)
