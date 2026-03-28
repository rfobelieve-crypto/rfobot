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
from market_data.adapters.oi_collector import collect_loop as oi_collect_loop
from market_data.adapters.oi_schema import ensure_oi_schema
from market_data.adapters.funding_collector import collect_loop as funding_collect_loop
from market_data.adapters.funding_collector import collect_once as funding_collect_once
from market_data.adapters.liquidation_collector import start_all as start_liquidation
from market_data.adapters.extra_schema import ensure_extra_schema

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

    # Ensure OI schema (table + columns)
    try:
        ensure_oi_schema()
    except Exception:
        logger.exception("OI schema setup failed (may already exist)")

    # Ensure funding + liquidation schema
    try:
        ensure_extra_schema()
    except Exception:
        logger.exception("Extra schema setup failed (may already exist)")

    # Start OI collector in background (every 60s)
    threading.Thread(target=oi_collect_loop, daemon=True, name="oi-collector").start()
    logger.info("OI collector started (every 60s).")

    # Start funding rate collector in background (every 60s)
    # Collect once immediately so baseline is available for the first event
    try:
        funding_collect_once()
    except Exception:
        logger.exception("Initial funding collection failed")
    threading.Thread(target=funding_collect_loop, daemon=True, name="funding-collector").start()
    logger.info("Funding rate collector started (every 60s).")

    # Start liquidation WS adapters + flusher
    start_liquidation()
    logger.info("Liquidation collectors started (OKX + Binance).")

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
