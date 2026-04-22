"""
Bybit USDT-margined perpetual futures trade adapter.

Connects to Bybit v5 public WebSocket, subscribes to publicTrade topics
for tracked symbols, and emits raw parsed trade dicts.

Trade size is reported in base units (BTC for BTCUSDT), so contract_size=1.
"""

import json
import time
import logging

import websocket

from market_data.core.symbol_mapper import tracked_symbols
from market_data.core.health_monitor import on_trade, on_reconnect

logger = logging.getLogger(__name__)

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"


class BybitTradeAdapter:
    def __init__(self, on_trades_callback):
        """
        on_trades_callback: callable(list[dict]) — called with a batch of raw trades.
        Each raw trade dict has:
            exchange, raw_symbol, price, size, taker_side,
            trade_id, ts_exchange, ts_received, is_aggregated_trade
        """
        self.on_trades = on_trades_callback
        self.symbols = tracked_symbols("bybit")
        self._running = False

    def start(self):
        """Start WebSocket connection in a blocking reconnect loop."""
        self._running = True
        reconnect_delay = 5

        while self._running:
            try:
                logger.info("[Bybit] Connecting to %s ...", BYBIT_WS_URL)
                ws = websocket.WebSocketApp(
                    BYBIT_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
            except Exception as e:
                logger.exception("[Bybit] Connection error: %s", e)

            if not self._running:
                break

            on_reconnect("bybit")
            logger.warning("[Bybit] Reconnecting in %ds ...", reconnect_delay)
            time.sleep(reconnect_delay)

    def stop(self):
        self._running = False

    def _on_open(self, ws):
        logger.info("[Bybit] Connected, subscribing to %d symbols", len(self.symbols))
        args = [f"publicTrade.{s}" for s in self.symbols]
        ws.send(json.dumps({"op": "subscribe", "args": args}))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            # Skip subscribe/ping confirmations
            if data.get("op") in ("subscribe", "pong") or data.get("success") is not None:
                return

            topic = data.get("topic", "")
            if not topic.startswith("publicTrade."):
                return
            if "data" not in data:
                return

            ts_received = int(time.time() * 1000)
            raw_trades = []

            for t in data["data"]:
                sym = t.get("s", "")
                if sym not in self.symbols:
                    continue

                side_raw = t.get("S", "").lower()
                if side_raw == "buy":
                    taker_side = "buy"
                elif side_raw == "sell":
                    taker_side = "sell"
                else:
                    continue

                ts_exchange = int(t.get("T", 0))

                raw_trades.append({
                    "exchange": "bybit",
                    "raw_symbol": sym,
                    "price": t["p"],
                    "size": t["v"],
                    "taker_side": taker_side,
                    "trade_id": str(t.get("i", "")),
                    "ts_exchange": ts_exchange,
                    "ts_received": ts_received,
                    "is_aggregated_trade": False,
                })

                on_trade("bybit", ts_exchange, ts_received)

            if raw_trades:
                self.on_trades(raw_trades)

        except Exception:
            logger.exception("[Bybit] on_message error")

    def _on_error(self, ws, error):
        logger.error("[Bybit] WebSocket error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.warning("[Bybit] WebSocket closed: code=%s msg=%s", code, msg)
