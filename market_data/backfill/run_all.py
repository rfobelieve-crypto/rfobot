"""
Run all backfill tasks in sequence.

Order:
  1. Apply migration 009 (flow_bars_15m + unique keys)
  2. Funding Rate backfill (Binance, from 2023-01-01)
  3. OI backfill (Binance, last 30 days)
  4. flow_bars_15m: CSV build + DB rollup

Usage:
    python -m market_data.backfill.run_all
    python -m market_data.backfill.run_all --skip-csv   # skip CSV (slow, ~26M rows)
"""
import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.storage.db import run_migration
from market_data.backfill.funding_backfill import run as run_funding
from market_data.backfill.oi_backfill import run as run_oi
from market_data.backfill.flow_bars_15m_builder import run as run_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-csv", action="store_true", help="Skip CSV flow bar build (faster)")
    ap.add_argument("--skip-funding", action="store_true")
    ap.add_argument("--skip-oi",      action="store_true")
    args = ap.parse_args()

    # 1. Migration
    logger.info("=== Step 1: Apply migration 009 ===")
    mig_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations", "009_flow_bars_15m.sql",
    )
    try:
        run_migration(mig_path)
        logger.info("Migration 009 applied.")
    except Exception:
        logger.exception("Migration failed (may already exist, continuing)")

    # 2. Funding Rate
    if not args.skip_funding:
        logger.info("=== Step 2: Funding Rate backfill ===")
        try:
            run_funding()
        except Exception:
            logger.exception("Funding backfill failed")
    else:
        logger.info("=== Step 2: Funding Rate backfill SKIPPED ===")

    # 3. OI
    if not args.skip_oi:
        logger.info("=== Step 3: OI backfill (30 days) ===")
        try:
            run_oi(days_back=30)
        except Exception:
            logger.exception("OI backfill failed")
    else:
        logger.info("=== Step 3: OI backfill SKIPPED ===")

    # 4. flow_bars_15m
    logger.info("=== Step 4: flow_bars_15m build ===")
    try:
        run_flow(csv=not args.skip_csv, db=True)
    except Exception:
        logger.exception("flow_bars_15m build failed")

    logger.info("=== All backfill tasks complete ===")


if __name__ == "__main__":
    main()
