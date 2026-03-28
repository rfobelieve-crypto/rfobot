"""
Binance perpetual futures trade adapter.

Connects to Binance fapi WebSocket, subscribes to aggTrade streams
for tracked symbols, and emits raw parsed trade dicts.

Binance aggTrade fields:
  e: "aggTrade"
  s: "BTCUSDT"
  p: "67000.50"  (price)
  q: "0.010"     (quantity in base asset)
  f: first trade ID
  l: last trade ID
  T: trade time (ms)
  m: is buyer maker (true = taker is sell, false = taker is buy)
"""

import json
import time
import logging

import websocket

from market_data.core.symbol_mapper import tracked_symbols
from market_data.core.health_monitor import on_trade, on_reconnect

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"


class BinanceTradeAdapter:
    def __init__(self, on_trades_callback):
        """
        on_trades_callback: callable(list[dict]) - called with raw trades.
        """
        self.on_trades = on_trades_callback
        self.symbols = tracked_symbols("binance")
        self._running = False

    def _build_url(self) -> str:
        """Build combined stream URL for all tracked symbols."""
        streams = [s.lower() + "@aggTrade" for s in self.symbols]
        return BINANCE_WS_BASE + "/" + "/".join(streams)

    def start(self):
        """Start WebSocket connection (blocking reconnect loop)."""
        self._running = True
        reconnect_delay = 5

        while self._running:
            try:
                url = self._build_url()
                logger.info("[Binance] Connecting to %s ...", url)
                ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
            except Exception as e:
                logger.exception("[Binance] Connection error: %s", e)

            if not self._running:
                break

            on_reconnect("binance")
            logger.warning("[Binance] Reconnecting in %ds ...", reconnect_delay)
            time.sleep(reconnect_delay)

    def stop(self):
        self._running = False

    def _on_open(self, ws):
        logger.info("[Binance] Connected, receiving aggTrade for %d symbols", len(self.symbols))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            # Combined stream wraps in {"stream": ..., "data": ...}
            if "stream" in data:
                data = data["data"]

            event_type = data.get("e")
            if event_type != "aggTrade":
                return

            raw_symbol = data.get("s", "")
            if raw_symbol not in self.symbols:
                return

            ts_received = int(time.time() * 1000)
            ts_exchange = int(data.get("T", 0))

            # m=true means buyer is maker => taker is seller
            is_buyer_maker = data.get("m", False)
            taker_side = "sell" if is_buyer_maker else "buy"

            raw_trade = {
                "exchange": "binance",
                "raw_symbol": raw_symbol,
                "price": data["p"],
                "size": data["q"],
                "taker_side": taker_side,
                "trade_id": str(data.get("a", "")),
                "ts_exchange": ts_exchange,
                "ts_received": ts_received,
                "is_aggregated_trade": True,
            }

            on_trade("binance", ts_exchange, ts_received)
            self.on_trades([raw_trade])

        except Exception:
            logger.exception("[Binance] on_message error")

    def _on_error(self, ws, error):
        logger.error("[Binance] WebSocket error: %s", error)

    def _on_close(self, ws, code, msg):
        logger.warning("[Binance] WebSocket closed: code=%s msg=%s", code, msg)
