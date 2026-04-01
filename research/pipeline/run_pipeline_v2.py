"""
ML Feature Pipeline v2 — full orchestrator.

Steps:
  1. Migration 011  (ohlcv_1m, flow_bars_1m_ml, features_5m/15m/1h)
  2. Import klines  → ohlcv_1m
  3. Import aggTrades → flow_bars_1m_ml
  4. Feature builder  (5m, 15m, 1h)
  5. Export ML Parquet (15m default, or all)
  6. Summary

Usage:
    python -m research.pipeline.run_pipeline_v2
    python -m research.pipeline.run_pipeline_v2 --skip-import
    python -m research.pipeline.run_pipeline_v2 --export-all-tf
"""
import sys
import os
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-import",   action="store_true",
                    help="Skip klines + aggTrade import")
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip feature computation")
    ap.add_argument("--export-all-tf", action="store_true",
                    help="Export Parquet for 5m, 15m, 1h (default: 15m only)")
    ap.add_argument("--timeframes",    default="15m,1h",
                    help="Comma-separated timeframes to build features for (default: 15m,1h — klines source is 15m)")
    args = ap.parse_args()

    tfs = [t.strip() for t in args.timeframes.split(",")]

    # ── Step 1: Migration ─────────────────────────────────────────────────────
    logger.info("=== Step 1: Migration 011 ===")
    from market_data.storage.db import run_migration
    mig = Path(__file__).resolve().parents[2] / "migrations" / "011_ml_tables.sql"
    try:
        run_migration(str(mig))
    except Exception:
        logger.exception("Migration 011 failed (may already exist)")

    if not args.skip_import:
        # ── Step 2: Import klines ─────────────────────────────────────────────
        logger.info("=== Step 2: Import klines → ohlcv_1m ===")
        from market_data.backfill.import_klines import import_klines
        n_klines = import_klines()
        logger.info("ohlcv_1m: %d rows", n_klines)

        # ── Step 3: Import aggTrades ──────────────────────────────────────────
        logger.info("=== Step 3: Import aggTrades → flow_bars_1m_ml ===")
        from market_data.backfill.import_aggtrades import import_aggtrades
        n_flow = import_aggtrades()
        logger.info("flow_bars_1m_ml: %d bars", n_flow)

        # ── Step 3b: Import funding rates ─────────────────────────────────────
        logger.info("=== Step 3b: Import funding rates ===")
        from market_data.backfill.funding_backfill import import_funding_local
        n_funding = import_funding_local()
        logger.info("funding_rates: %d rows", n_funding)

    if not args.skip_features:
        # ── Step 4: Feature builder ───────────────────────────────────────────
        logger.info("=== Step 4: Feature builder (%s) ===", ", ".join(tfs))
        from research.pipeline.feature_builder_v2 import run as build_features
        build_features(timeframes=tfs)

    # ── Step 5: Export ML Parquet ─────────────────────────────────────────────
    export_tfs = tfs if args.export_all_tf else ["15m"]
    logger.info("=== Step 5: Export ML Parquet (%s) ===", ", ".join(export_tfs))
    from research.pipeline.export_ml import export
    for tf in export_tfs:
        export("BTC-USD", tf)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    logger.info("=== Step 6: Summary ===")
    from shared.db import get_db_conn
    conn = get_db_conn()
    with conn.cursor() as cur:
        for t in ["ohlcv_1m", "flow_bars_1m_ml",
                  "features_5m", "features_15m", "features_1h",
                  "oi_snapshots", "funding_rates"]:
            try:
                cur.execute(f"SELECT COUNT(*) as n FROM `{t}`")
                n = cur.fetchone()["n"]
                logger.info("  %-25s %d rows", t, n)
            except Exception:
                logger.warning("  %-25s (not found)", t)
    conn.close()

    from pathlib import Path as P
    ml_dir = P(__file__).resolve().parents[1] / "ml_data"
    if ml_dir.exists():
        for f in sorted(ml_dir.glob("*.parquet")):
            size_mb = f.stat().st_size / 1024 / 1024
            logger.info("  ml_data/%-45s %.1f MB", f.name, size_mb)


if __name__ == "__main__":
    main()
