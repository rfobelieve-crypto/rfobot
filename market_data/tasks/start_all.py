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
from market_data.tasks.cleanup import cleanup_once
from market_data.features.snapshot_runner import process_once as snapshot_process_once

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Market Data Layer starting ===")

    # Run migrations
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations",
    )
    for mig_file in ("001_market_data_tables.sql", "004_event_feature_snapshots.sql"):
        mig_path = os.path.join(migrations_dir, mig_file)
        if os.path.exists(mig_path):
            try:
                run_migration(mig_path)
            except Exception:
                logger.exception("Migration %s failed (may already exist)", mig_file)
    logger.info("Migrations complete.")

    # Start flow bar flusher in background
    threading.Thread(target=flush_loop, daemon=True, name="flow-flusher").start()
    logger.info("Flow bar flusher started.")

    # Start data cleanup in background (every hour)
    def _cleanup_loop():
        import time
        while True:
            try:
                cleanup_once()
            except Exception:
                logger.exception("Cleanup error")
            time.sleep(3600)

    threading.Thread(target=_cleanup_loop, daemon=True, name="cleanup").start()
    logger.info("Data cleanup started (trades: 3d, flow_bars: 90d).")

    # Start snapshot runner in background (every 60s)
    def _snapshot_loop():
        import time
        while True:
            try:
                snapshot_process_once()
            except Exception:
                logger.exception("Snapshot runner error")
            time.sleep(60)

    threading.Thread(target=_snapshot_loop, daemon=True, name="snapshot-runner").start()
    logger.info("Snapshot runner started (every 60s).")

    # Start trade streams (blocking)
    start_streams()


if __name__ == "__main__":
    main()
