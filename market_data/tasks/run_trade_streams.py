"""
Entry point: start OKX + Binance trade streams.

Receives raw trades -> normalizes -> aggregates -> batch inserts.

Usage:
    python -m market_data.tasks.run_trade_streams
"""

import os
import sys
import time
import threading
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.adapters.okx_trades import OKXTradeAdapter
from market_data.adapters.binance_trades import BinanceTradeAdapter
from market_data.core.trade_normalizer import normalize
from market_data.core.flow_aggregator import add_trade
from market_data.core.health_monitor import check_staleness, get_status
from market_data.storage.trade_repository import insert_trades

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Batch buffer for DB writes
_batch_lock = threading.Lock()
_trade_batch: list[dict] = []
BATCH_SIZE = 100
BATCH_FLUSH_INTERVAL = 5  # seconds


def on_raw_trades(raw_trades: list[dict]):
    """Callback shared by all adapters."""
    normalized = []
    for raw in raw_trades:
        t = normalize(raw)
        if t is not None:
            normalized.append(t)
            add_trade(t)

    if normalized:
        with _batch_lock:
            _trade_batch.extend(normalized)

            if len(_trade_batch) >= BATCH_SIZE:
                to_flush = list(_trade_batch)
                _trade_batch.clear()

        # Flush outside lock if threshold hit
        if len(normalized) >= BATCH_SIZE or _should_flush():
            _flush_trades()


def _should_flush() -> bool:
    with _batch_lock:
        return len(_trade_batch) >= BATCH_SIZE


def _flush_trades():
    with _batch_lock:
        if not _trade_batch:
            return
        to_flush = list(_trade_batch)
        _trade_batch.clear()

    try:
        insert_trades(to_flush)
    except Exception:
        logger.exception("Failed to flush %d trades to DB", len(to_flush))


def batch_flush_loop():
    """Periodically flush trade batch to DB."""
    while True:
        time.sleep(BATCH_FLUSH_INTERVAL)
        _flush_trades()


def health_check_loop():
    """Periodically check and log health status."""
    while True:
        time.sleep(30)
        check_staleness()
        status = get_status()
        for source, info in status.items():
            logger.info(
                "[Health] %s: status=%s latency=%dms msgs=%d reconnects=%d",
                source, info["status"], info["latency_ms"],
                info["message_count"], info["reconnect_count"],
            )


def main():
    logger.info("=== Market Data Trade Streams starting ===")

    # Background: batch flush
    threading.Thread(target=batch_flush_loop, daemon=True).start()

    # Background: health check
    threading.Thread(target=health_check_loop, daemon=True).start()

    # Start adapters in separate threads
    okx = OKXTradeAdapter(on_trades_callback=on_raw_trades)
    binance = BinanceTradeAdapter(on_trades_callback=on_raw_trades)

    threading.Thread(target=okx.start, daemon=True, name="okx-ws").start()
    threading.Thread(target=binance.start, daemon=True, name="binance-ws").start()

    logger.info("OKX and Binance adapters started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        okx.stop()
        binance.stop()
        _flush_trades()


if __name__ == "__main__":
    main()
