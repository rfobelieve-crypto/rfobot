"""
Single entry point: start trade streams + flow bar flusher together.

Usage:
    python -m market_data.tasks.start_all

This does NOT touch the main bot (BTC_perp_data.py).
Run this as a separate process alongside the main bot.
"""

import os
import sys
import threading
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.storage.db import run_migration
from market_data.tasks.run_trade_streams import main as start_streams
from market_data.tasks.flush_flow_bars import flush_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Market Data Layer starting ===")

    # Run migration
    migration_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations",
        "001_market_data_tables.sql",
    )
    if os.path.exists(migration_path):
        try:
            run_migration(migration_path)
            logger.info("Migration complete.")
        except Exception:
            logger.exception("Migration failed (tables may already exist)")

    # Start flow bar flusher in background
    threading.Thread(target=flush_loop, daemon=True, name="flow-flusher").start()
    logger.info("Flow bar flusher started.")

    # Start trade streams (blocking)
    start_streams()


if __name__ == "__main__":
    main()
