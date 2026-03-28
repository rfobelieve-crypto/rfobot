"""
Feature computation runner.

Scans liquidity_events, computes features + scores, writes to event_features_v2.
Skips events that already have features computed.

Usage:
    python -m market_data.features.runner              # run once
    python -m market_data.features.runner --loop       # run every 5 min
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.db import get_db_conn
from market_data.storage.db import run_migration
from market_data.features.extractor import extract_features
from market_data.features.scorer import score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

LOOP_INTERVAL = 300  # 5 minutes


def get_unprocessed_events() -> list[dict]:
    """Get liquidity_events that don't have features computed yet."""
    sql = """
    SELECT e.*
    FROM liquidity_events e
    LEFT JOIN event_features_v2 f ON e.event_uuid = f.event_uuid
    WHERE f.id IS NULL
    ORDER BY e.trigger_ts ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


def save_features(features: dict, scores: dict):
    """Insert one row into event_features_v2."""
    sql = """
    INSERT INTO event_features_v2 (
        event_uuid, symbol, liquidity_side, entry_price, trigger_ts, session,
        pre_delta_usd, pre_volume_usd, pre_buy_sell_ratio, pre_cvd_usd, pre_trade_count,
        post_2h_delta_usd, post_2h_volume_usd, post_2h_buy_sell_ratio, post_2h_cvd_usd, post_2h_trade_count,
        post_4h_delta_usd, post_4h_volume_usd, post_4h_buy_sell_ratio, post_4h_cvd_usd, post_4h_trade_count,
        post_6h_delta_usd, post_6h_volume_usd, post_6h_buy_sell_ratio, post_6h_cvd_usd, post_6h_trade_count,
        cvd_slope_2h, cvd_slope_4h,
        delta_imbalance_2h, delta_imbalance_4h,
        price_return_2h, price_return_4h, price_return_6h,
        delta_divergence_2h, delta_divergence_4h,
        absorption_detected,
        oi_change_2h, oi_change_4h,
        liq_buy_usd_2h, liq_sell_usd_2h, liq_buy_usd_4h, liq_sell_usd_4h,
        orderbook_imbalance_2h, orderbook_imbalance_4h,
        reversal_score, continuation_score, confidence_score, bias, label,
        scorer_version
    ) VALUES (
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s,
        %s,
        %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s, %s,
        %s
    )
    ON DUPLICATE KEY UPDATE
        reversal_score = VALUES(reversal_score),
        continuation_score = VALUES(continuation_score),
        confidence_score = VALUES(confidence_score),
        bias = VALUES(bias),
        label = VALUES(label),
        scorer_version = VALUES(scorer_version),
        computed_at = CURRENT_TIMESTAMP
    """

    f = features
    s = scores

    params = (
        f["event_uuid"], f["symbol"], f["liquidity_side"], f["entry_price"],
        f["trigger_ts"], f["session"],
        f["pre_delta_usd"], f["pre_volume_usd"], f["pre_buy_sell_ratio"],
        f["pre_cvd_usd"], f["pre_trade_count"],
        f["post_2h_delta_usd"], f["post_2h_volume_usd"], f["post_2h_buy_sell_ratio"],
        f["post_2h_cvd_usd"], f["post_2h_trade_count"],
        f["post_4h_delta_usd"], f["post_4h_volume_usd"], f["post_4h_buy_sell_ratio"],
        f["post_4h_cvd_usd"], f["post_4h_trade_count"],
        f["post_6h_delta_usd"], f["post_6h_volume_usd"], f["post_6h_buy_sell_ratio"],
        f["post_6h_cvd_usd"], f["post_6h_trade_count"],
        f["cvd_slope_2h"], f["cvd_slope_4h"],
        f["delta_imbalance_2h"], f["delta_imbalance_4h"],
        f["price_return_2h"], f["price_return_4h"], f["price_return_6h"],
        f["delta_divergence_2h"], f["delta_divergence_4h"],
        f["absorption_detected"],
        f["oi_change_2h"], f["oi_change_4h"],
        f["liq_buy_usd_2h"], f["liq_sell_usd_2h"],
        f["liq_buy_usd_4h"], f["liq_sell_usd_4h"],
        f["orderbook_imbalance_2h"], f["orderbook_imbalance_4h"],
        s["reversal_score"], s["continuation_score"],
        s["confidence_score"], s["bias"], f["label"],
        s["scorer_version"],
    )

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        logger.info("Saved features for event %s: bias=%s rev=%.1f cont=%.1f conf=%.2f",
                     f["event_uuid"][:8], s["bias"],
                     s["reversal_score"], s["continuation_score"], s["confidence_score"])
    finally:
        conn.close()


def process_once():
    """Process all unprocessed events."""
    events = get_unprocessed_events()
    if not events:
        logger.info("No unprocessed events.")
        return 0

    logger.info("Processing %d events...", len(events))
    count = 0

    for event in events:
        try:
            features = extract_features(event)
            scores = score(features)
            save_features(features, scores)
            count += 1
        except Exception:
            logger.exception("Failed to process event %s", event.get("event_uuid", "?")[:8])

    return count


def main():
    # Run migration first
    migration_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations", "002_event_features_v2.sql",
    )
    if os.path.exists(migration_path):
        try:
            run_migration(migration_path)
        except Exception:
            logger.exception("Migration failed (table may already exist)")

    loop_mode = "--loop" in sys.argv

    if loop_mode:
        logger.info("Feature runner starting in loop mode (every %ds)", LOOP_INTERVAL)
        while True:
            try:
                process_once()
            except Exception:
                logger.exception("process_once error")
            time.sleep(LOOP_INTERVAL)
    else:
        count = process_once()
        logger.info("Done. Processed %d events.", count)


if __name__ == "__main__":
    main()
