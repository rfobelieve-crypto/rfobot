"""
Query functions for event snapshots, scores, and history.
Used by Telegram bot commands.
"""

import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def get_latest_snapshots(limit: int = 1) -> list[dict]:
    """
    Get snapshots for the most recent event(s).
    Returns all snapshot rows (15m/1h/4h) for the latest N events.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Step 1: find latest event UUIDs
            cur.execute("""
                SELECT event_uuid, MAX(trigger_ts) AS latest_ts
                FROM event_feature_snapshots
                GROUP BY event_uuid
                ORDER BY latest_ts DESC
                LIMIT %s
            """, (limit,))
            uuids = [r["event_uuid"] for r in cur.fetchall()]
            if not uuids:
                return []

            # Step 2: get all snapshots for those events
            placeholders = ",".join(["%s"] * len(uuids))
            cur.execute(f"""
                SELECT * FROM event_feature_snapshots
                WHERE event_uuid IN ({placeholders})
                ORDER BY trigger_ts DESC,
                    CASE snapshot_type
                        WHEN '15m' THEN 1 WHEN '1h' THEN 2 WHEN '4h' THEN 3 ELSE 99
                    END
            """, uuids)
            return cur.fetchall()
    finally:
        conn.close()


def get_snapshots_by_uuid(uuid_prefix: str) -> list[dict]:
    """Get all snapshots for a specific event by UUID prefix."""
    sql = """
    SELECT * FROM event_feature_snapshots
    WHERE event_uuid LIKE %s
    ORDER BY FIELD(snapshot_type, '15m', '1h', '4h')
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (uuid_prefix + "%",))
            return cur.fetchall()
    finally:
        conn.close()


def get_latest_scores(limit: int = 3) -> list[dict]:
    """Get the latest scored events from event_features_v2."""
    sql = """
    SELECT event_uuid, symbol, liquidity_side, entry_price, trigger_ts,
           reversal_score, continuation_score, confidence_score, bias, label,
           scorer_version, computed_at
    FROM event_features_v2
    ORDER BY trigger_ts DESC
    LIMIT %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_event_history(limit: int = 5) -> list[dict]:
    """Get recent events with their outcomes and snapshot bias evolution."""
    sql = """
    SELECT
        r.event_uuid, r.event_type, r.symbol, r.liquidity_side,
        r.entry_price, r.trigger_ts, r.sweep_ref_price,
        s15.bias AS bias_15m, s15.confidence_score AS conf_15m,
        s15.reversal_score AS rev_15m, s15.continuation_score AS cont_15m,
        s1h.bias AS bias_1h, s1h.confidence_score AS conf_1h,
        s1h.reversal_score AS rev_1h, s1h.continuation_score AS cont_1h,
        s4h.bias AS bias_4h, s4h.confidence_score AS conf_4h,
        s4h.reversal_score AS rev_4h, s4h.continuation_score AS cont_4h,
        s4h.label AS label
    FROM event_registry r
    LEFT JOIN event_feature_snapshots s15
        ON r.event_uuid = s15.event_uuid AND s15.snapshot_type = '15m'
    LEFT JOIN event_feature_snapshots s1h
        ON r.event_uuid = s1h.event_uuid AND s1h.snapshot_type = '1h'
    LEFT JOIN event_feature_snapshots s4h
        ON r.event_uuid = s4h.event_uuid AND s4h.snapshot_type = '4h'
    ORDER BY r.trigger_ts DESC
    LIMIT %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_pending_snapshot_count() -> dict:
    """Count how many snapshots are pending per type."""
    import time
    now = int(time.time())
    windows = {"15m": 900, "1h": 3600, "4h": 14400}
    counts = {}

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for snap_type, offset in windows.items():
                sql = """
                SELECT COUNT(*) AS cnt
                FROM event_registry r
                LEFT JOIN event_feature_snapshots s
                    ON r.event_uuid = s.event_uuid AND s.snapshot_type = %s
                WHERE s.id IS NULL AND (r.trigger_ts + %s) <= %s
                """
                cur.execute(sql, (snap_type, offset, now))
                row = cur.fetchone()
                counts[snap_type] = row["cnt"] if row else 0
    finally:
        conn.close()

    return counts
