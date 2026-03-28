"""
Entry point: periodically flush completed 1m flow bars to MySQL.

Usage:
    python -m market_data.tasks.flush_flow_bars

Typically run alongside run_trade_streams. Can also be imported
and started as a background thread.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.core.flow_aggregator import flush, stats
from market_data.storage.flow_repository import upsert_flow_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

FLUSH_INTERVAL = 10  # seconds; checks every 10s, only flushes completed bars


def flush_loop():
    """Continuous loop: flush completed flow bars to DB."""
    while True:
        try:
            bars = flush()
            if bars:
                upsert_flow_bars(bars)
                for b in bars:
                    logger.info(
                        "[FlowBar] %s %d->%d delta=%.2f vol=%.2f trades=%d cvd=%.2f",
                        b["canonical_symbol"],
                        b["window_start"], b["window_end"],
                        b["delta_usd"], b["volume_usd"],
                        b["trade_count"], b["cvd_usd"],
                    )

            agg_stats = stats()
            if agg_stats["active_buckets"] > 0:
                logger.debug("[Aggregator] %s", agg_stats)

        except Exception:
            logger.exception("flush_loop error")

        time.sleep(FLUSH_INTERVAL)


def main():
    logger.info("=== Flow Bar Flusher starting ===")
    flush_loop()


if __name__ == "__main__":
    main()
