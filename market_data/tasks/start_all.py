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
    DepthDeltaCollector, PERP_WS_URL, ETH_WS_URL, ETH_PERP_WS_URL,
    XRP_WS_URL, XRP_PERP_WS_URL, DOGE_WS_URL, DOGE_PERP_WS_URL,
    ADA_WS_URL, ADA_PERP_WS_URL, SUI_WS_URL, SUI_PERP_WS_URL,
    BNB_WS_URL, BNB_PERP_WS_URL, LINK_WS_URL, LINK_PERP_WS_URL,
    UNI_WS_URL, UNI_PERP_WS_URL, AAVE_WS_URL, AAVE_PERP_WS_URL)

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

    # F7 second-resolution depth events (2026-08-13, TODO §0.46): BTC-only
    # 1s buckets with price bands + cancel→re-add matching — the structures
    # the 1m aggregation destroys. Additive third consumer of the same
    # streams; the frozen 1m BTC series above is untouched.
    from market_data.adapters.depth_events_1s import (
        DepthEvents1sCollector, PERP_WS_URL as F7_PERP_WS_URL)
    threading.Thread(target=DepthEvents1sCollector().start, daemon=True,
                     name="depth-events-1s").start()
    threading.Thread(
        target=DepthEvents1sCollector(ws_url=F7_PERP_WS_URL,
                                      exchange="binance_perp").start,
        daemon=True, name="depth-events-1s-perp").start()
    logger.info("F7 depth-events 1s collectors started (depth_events_1s, "
                "BTC spot+perp).")

    # ETH parallel series (2026-07-23, V7/cancel-flow multicoin override —
    # see CLAUDE.md "V7 多幣化提前啟動"): same decomposition, new coin, own
    # canonical_symbol so it writes to rows the BTC cancel tests never touch.
    # BTC's spot+perp threads above are byte-for-byte unchanged — this is
    # purely additive, starting ETH's data clock in parallel per TODO.md
    # "撤單流多幣化".
    threading.Thread(
        target=DepthDeltaCollector(ws_url=ETH_WS_URL, exchange="binance",
                                   canonical_symbol="ETH-USD").start,
        daemon=True, name="depth-delta-eth").start()
    logger.info("Depth-delta ETH collector started (canonical_symbol=ETH-USD).")

    threading.Thread(
        target=DepthDeltaCollector(ws_url=ETH_PERP_WS_URL, exchange="binance_perp",
                                   canonical_symbol="ETH-USD").start,
        daemon=True, name="depth-delta-eth-perp").start()
    logger.info("Depth-delta ETH PERP collector started (exchange=binance_perp, canonical_symbol=ETH-USD).")

    # Batch 2 (2026-07-23/24, same override): liquidity-ranked real-crypto
    # Binance USDT perpetuals — XRP/DOGE/ADA/SUI/BNB/LINK/UNI/AAVE. Identical
    # spot+perp decomposition, own canonical_symbol per coin, byte-for-byte
    # same pattern as the ETH threads above. BTC/ETH threads untouched.
    _BATCH2 = [
        ("XRP-USD", XRP_WS_URL, XRP_PERP_WS_URL),
        ("DOGE-USD", DOGE_WS_URL, DOGE_PERP_WS_URL),
        ("ADA-USD", ADA_WS_URL, ADA_PERP_WS_URL),
        ("SUI-USD", SUI_WS_URL, SUI_PERP_WS_URL),
        ("BNB-USD", BNB_WS_URL, BNB_PERP_WS_URL),
        ("LINK-USD", LINK_WS_URL, LINK_PERP_WS_URL),
        ("UNI-USD", UNI_WS_URL, UNI_PERP_WS_URL),
        ("AAVE-USD", AAVE_WS_URL, AAVE_PERP_WS_URL),
    ]
    for _canon, _spot_url, _perp_url in _BATCH2:
        _coin = _canon.split("-")[0].lower()
        threading.Thread(
            target=DepthDeltaCollector(ws_url=_spot_url, exchange="binance",
                                       canonical_symbol=_canon).start,
            daemon=True, name=f"depth-delta-{_coin}").start()
        threading.Thread(
            target=DepthDeltaCollector(ws_url=_perp_url, exchange="binance_perp",
                                       canonical_symbol=_canon).start,
            daemon=True, name=f"depth-delta-{_coin}-perp").start()
        logger.info("Depth-delta %s spot+perp collectors started (canonical_symbol=%s).",
                    _coin.upper(), _canon)

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
