"""
Service 2 entry point: trade streams + flow bar aggregation only.

Responsibilities:
- OKX + Binance WebSocket trade streams
- 1-minute flow bar aggregation (delta, volume, CVD → flow_bars_1m)
- Periodic cleanup of old raw trades

Everything else (snapshot runner, OI/funding/liquidation collectors,
schema setup) runs in Service 1 (BTC_perp_data.py).

Usage:
    python -m market_data.tasks.start_all
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Market Data Layer (Service 2) starting ===")

    # Run base migrations
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations",
    )
    for mig_file in ("001_market_data_tables.sql", "004_event_feature_snapshots.sql",
                     "006_cleanup_legacy.sql"):
        mig_path = os.path.join(migrations_dir, mig_file)
        if os.path.exists(mig_path):
            try:
                run_migration(mig_path)
            except Exception:
                logger.exception("Migration %s failed (may already exist)", mig_file)
    logger.info("Migrations complete.")

    # Start flow bar flusher
    threading.Thread(target=flush_loop, daemon=True, name="flow-flusher").start()
    logger.info("Flow bar flusher started.")

    # Start data cleanup (every hour)
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

    # Start trade streams (blocking — must be last)
    start_streams()


if __name__ == "__main__":
    main()
