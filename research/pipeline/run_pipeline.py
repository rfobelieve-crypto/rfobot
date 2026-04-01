"""
Full pipeline runner.

Step 1: Apply migration 010
Step 2: Build feature_bars_15m   (Layer 2)
Step 3: Build event_features      (Layer 3)
Step 4: Print edge report

Usage:
    python -m research.pipeline.run_pipeline
    python -m research.pipeline.run_pipeline --report-only
"""
import sys, os, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.storage.db import run_migration
from research.pipeline.feature_builder import run as build_features
from research.pipeline.event_builder import build as build_events
from research.pipeline.edge_query import EdgeQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        # 1. Migration
        logger.info("=== Step 1: Migration 010 ===")
        mig = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "migrations", "010_feature_pipeline.sql")
        try:
            run_migration(mig)
        except Exception:
            logger.exception("Migration 010 failed (may already exist)")

        # 2. Feature layer
        logger.info("=== Step 2: feature_bars_15m ===")
        build_features()

        # 3. Event layer
        logger.info("=== Step 3: event_features ===")
        build_events()

    # 4. Report
    logger.info("=== Step 4: Edge Report ===")
    EdgeQuery().print_report()


if __name__ == "__main__":
    main()
