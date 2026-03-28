"""
DB operations for event registry and feature snapshots.
"""

import logging
import time
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def register_event(event_uuid: str, event_type: str, symbol: str,
                   liquidity_side: str, entry_price: float,
                   trigger_ts: int, sweep_ref_price: float = None):
    """Insert event into event_registry immediately at webhook time."""
    sql = """
    INSERT IGNORE INTO event_registry
        (event_uuid, event_type, symbol, liquidity_side, entry_price, trigger_ts, sweep_ref_price)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (event_uuid, event_type, symbol,
                              liquidity_side, entry_price, trigger_ts, sweep_ref_price))
        logger.info("Registered event %s to event_registry", event_uuid[:8])
    except Exception:
        logger.exception("Failed to register event %s", event_uuid[:8])
    finally:
        conn.close()


# Snapshot windows in seconds
SNAPSHOT_WINDOWS = {
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}


def get_pending_snapshots() -> list[dict]:
    """
    Find (event, snapshot_type) pairs that need computing:
    1. Snapshots that don't exist yet (s.id IS NULL)
    2. Snapshots that exist but predate the current scorer (final_score IS NULL)
       — falls back to (1) only if final_score column doesn't exist yet
    """
    now = int(time.time())
    results = []

    # Try with final_score check first; fall back if column not yet migrated
    sql_with_rescore = """
    SELECT r.*
    FROM event_registry r
    LEFT JOIN event_feature_snapshots s
        ON r.event_uuid = s.event_uuid AND s.snapshot_type = %s
    WHERE (s.id IS NULL OR s.final_score IS NULL)
      AND (r.trigger_ts + %s) <= %s
    ORDER BY r.trigger_ts ASC
    """
    sql_basic = """
    SELECT r.*
    FROM event_registry r
    LEFT JOIN event_feature_snapshots s
        ON r.event_uuid = s.event_uuid AND s.snapshot_type = %s
    WHERE s.id IS NULL
      AND (r.trigger_ts + %s) <= %s
    ORDER BY r.trigger_ts ASC
    """

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Probe once whether final_score column exists
            use_rescore = True
            try:
                cur.execute(
                    "SELECT final_score FROM event_feature_snapshots LIMIT 0"
                )
            except Exception:
                use_rescore = False

            sql = sql_with_rescore if use_rescore else sql_basic

            for snap_type, offset_sec in SNAPSHOT_WINDOWS.items():
                cur.execute(sql, (snap_type, offset_sec, now))
                rows = cur.fetchall()
                for row in rows:
                    row["_snapshot_type"] = snap_type
                    row["_offset_sec"] = offset_sec
                    results.append(row)
    finally:
        conn.close()

    return results


def save_snapshot(snapshot: dict):
    """Insert one snapshot row into event_feature_snapshots."""
    sql = """
    INSERT INTO event_feature_snapshots (
        event_uuid, event_type, canonical_symbol, liquidity_side,
        trigger_price, trigger_ts, snapshot_type, snapshot_ts,
        delta_value, cvd_change, cvd_sign_flip,
        price_change_pct, reclaim_flag, break_again_flag,
        oi_baseline_okx, oi_baseline_binance,
        oi_snapshot_okx, oi_snapshot_binance,
        oi_change_okx, oi_change_binance,
        oi_change_okx_pct, oi_change_binance_pct,
        oi_change_total, oi_change_total_pct,
        reversal_score, continuation_score, confidence_score, bias,
        final_score, normalized_score,
        funding_rate, liq_buy_usd, liq_sell_usd, liq_total_usd, liq_count,
        label
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s, %s,
        %s
    )
    ON DUPLICATE KEY UPDATE
        delta_value = VALUES(delta_value),
        cvd_change = VALUES(cvd_change),
        cvd_sign_flip = VALUES(cvd_sign_flip),
        price_change_pct = VALUES(price_change_pct),
        reclaim_flag = VALUES(reclaim_flag),
        break_again_flag = VALUES(break_again_flag),
        oi_baseline_okx = VALUES(oi_baseline_okx),
        oi_baseline_binance = VALUES(oi_baseline_binance),
        oi_snapshot_okx = VALUES(oi_snapshot_okx),
        oi_snapshot_binance = VALUES(oi_snapshot_binance),
        oi_change_okx = VALUES(oi_change_okx),
        oi_change_binance = VALUES(oi_change_binance),
        oi_change_okx_pct = VALUES(oi_change_okx_pct),
        oi_change_binance_pct = VALUES(oi_change_binance_pct),
        oi_change_total = VALUES(oi_change_total),
        oi_change_total_pct = VALUES(oi_change_total_pct),
        reversal_score = VALUES(reversal_score),
        continuation_score = VALUES(continuation_score),
        confidence_score = VALUES(confidence_score),
        bias = VALUES(bias),
        final_score = VALUES(final_score),
        normalized_score = VALUES(normalized_score),
        funding_rate = VALUES(funding_rate),
        liq_buy_usd = VALUES(liq_buy_usd),
        liq_sell_usd = VALUES(liq_sell_usd),
        liq_total_usd = VALUES(liq_total_usd),
        liq_count = VALUES(liq_count),
        label = VALUES(label)
    """
    s = snapshot
    params = (
        s["event_uuid"], s.get("event_type"), s["canonical_symbol"], s["liquidity_side"],
        s["trigger_price"], s["trigger_ts"], s["snapshot_type"], s["snapshot_ts"],
        s.get("delta_value"), s.get("cvd_change"), s.get("cvd_sign_flip"),
        s.get("price_change_pct"), s.get("reclaim_flag"), s.get("break_again_flag"),
        s.get("oi_baseline_okx"), s.get("oi_baseline_binance"),
        s.get("oi_snapshot_okx"), s.get("oi_snapshot_binance"),
        s.get("oi_change_okx"), s.get("oi_change_binance"),
        s.get("oi_change_okx_pct"), s.get("oi_change_binance_pct"),
        s.get("oi_change_total"), s.get("oi_change_total_pct"),
        s["reversal_score"], s["continuation_score"], s["confidence_score"], s["bias"],
        s.get("final_score"), s.get("normalized_score"),
        s.get("funding_rate"),
        s.get("liq_buy_usd"), s.get("liq_sell_usd"), s.get("liq_total_usd"), s.get("liq_count"),
        s.get("label"),
    )

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        logger.info("Saved %s snapshot for event %s: bias=%s rev=%.0f cont=%.0f conf=%.2f",
                     s["snapshot_type"], s["event_uuid"][:8],
                     s["bias"], s["reversal_score"], s["continuation_score"],
                     s["confidence_score"])
    finally:
        conn.close()
