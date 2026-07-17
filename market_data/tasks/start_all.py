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
from market_data.adapters.depth_delta_collector import (
    DepthDeltaCollector, PERP_WS_URL)

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

    # Start depth-delta collector (per-side add/cancel from Binance incremental
    # book → depth_deltas_1m). Forward-collection for the squeeze "path of least
    # resistance" research; self-contained (own schema, flush, WS reconnect).
    threading.Thread(target=DepthDeltaCollector().start, daemon=True,
                     name="depth-delta").start()
    logger.info("Depth-delta collector started (depth_deltas_1m).")

    # Parallel PERP-book instance (2026-07-15): same decomposition on the
    # Binance USDT-M futures incremental book, rows tagged exchange=
    # 'binance_perp'. The spot stream above is untouched — its series feeds
    # the pre-registered cancel tests and must stay unbroken.
    threading.Thread(
        target=DepthDeltaCollector(ws_url=PERP_WS_URL,
                                   exchange="binance_perp").start,
        daemon=True, name="depth-delta-perp").start()
    logger.info("Depth-delta PERP collector started (exchange=binance_perp).")

    # Cancel-playbook watcher (2026-07-16): machine-prospective event logger
    # for the cancel-flow playbooks (frozen defs v1) + rate-limited TG alerts.
    # Read-only on quant tables; writes only cancel_playbook_events. Research
    # aid — NOT a trading signal, never touches the executor.
    from market_data.tasks.cancel_playbook_watcher import watch_loop
    threading.Thread(target=watch_loop, daemon=True,
                     name="cancel-playbook").start()
    logger.info("Cancel-playbook watcher started (cancel_playbook_events).")

    # TV-alert poller (2026-07-17): consumes tv_alert_events written by the
    # main bot's /tv webhook (DB as bus), computes the cancel-flow display
    # state at trigger, pushes a simplified event card, and backfills the
    # 30/60/120m 判定窗 — the H-R sample clock starts here. Research aid,
    # never touches the executor.
    from market_data.tasks.tv_alert_poller import poll_loop
    threading.Thread(target=poll_loop, daemon=True,
                     name="tv-alert-poller").start()
    logger.info("TV-alert poller started (tv_alert_events).")

    # Start trade streams (blocking — must be last)
    start_streams()


if __name__ == "__main__":
    main()
