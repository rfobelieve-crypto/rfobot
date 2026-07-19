"""
Data retention cleanup.

Retention policy:
  - normalized_trades: 3 days (debug/replay only)
  - flow_bars_1m: 90 days (main research data)
  - liquidity_events / sweep_outcomes: keep forever (tiny)

Usage:
  python -m market_data.tasks.cleanup          # run once
  python -m market_data.tasks.cleanup --loop   # run every hour
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.db import get_db_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Retention in days
RETENTION = {
    "normalized_trades": 3,
    # 120d (was 90): 保住 depth_deltas 時代(2026-07-09 起)的完整 taker
    # 對照到 2026-10-09 撤單 re-run 判決日 (PREREG 附錄 A 家族)
    "flow_bars_1m": 120,
    # 2026-07-19 稽核新增: 兩張無上限增長的活表封頂。讀取者全用近窗
    # (report/dashboard SINCE=07-14 基準; snapshot_builder 回看數小時)
    "v7_okx_balance_snapshots": 180,   # 84 萬行 @ ~11 rows/min
    "oi_snapshots": 90,                # 62 萬行 @ ~4 rows/min
}


def cleanup_once():
    """Delete rows older than retention policy."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # normalized_trades: ts_exchange is in milliseconds
            cutoff_ms = int((time.time() - RETENTION["normalized_trades"] * 86400) * 1000)
            cur.execute(
                "DELETE FROM normalized_trades WHERE ts_exchange < %s",
                (cutoff_ms,)
            )
            trades_deleted = cur.rowcount
            if trades_deleted:
                logger.info("Cleaned normalized_trades: deleted %d rows (older than %d days)",
                            trades_deleted, RETENTION["normalized_trades"])

            # flow_bars_1m: window_start is in milliseconds
            cutoff_ms = int((time.time() - RETENTION["flow_bars_1m"] * 86400) * 1000)
            cur.execute(
                "DELETE FROM flow_bars_1m WHERE window_start < %s",
                (cutoff_ms,)
            )
            bars_deleted = cur.rowcount
            if bars_deleted:
                logger.info("Cleaned flow_bars_1m: deleted %d rows (older than %d days)",
                            bars_deleted, RETENTION["flow_bars_1m"])

            # DATETIME-keyed tables (column differs per table)
            extra_deleted = 0
            for tbl, col in (("v7_okx_balance_snapshots", "ts"),
                             ("oi_snapshots", "created_at")):
                cur.execute(
                    f"DELETE FROM {tbl} WHERE {col} < NOW() - INTERVAL %s DAY",
                    (RETENTION[tbl],))
                if cur.rowcount:
                    extra_deleted += cur.rowcount
                    logger.info("Cleaned %s: deleted %d rows (older than %d days)",
                                tbl, cur.rowcount, RETENTION[tbl])

            if not trades_deleted and not bars_deleted and not extra_deleted:
                logger.info("Cleanup: nothing to delete")

    finally:
        conn.close()


def main():
    loop_mode = "--loop" in sys.argv

    if loop_mode:
        logger.info("Cleanup running in loop mode (every 1 hour)")
        while True:
            try:
                cleanup_once()
            except Exception:
                logger.exception("Cleanup error")
            time.sleep(3600)
    else:
        cleanup_once()


if __name__ == "__main__":
    main()
