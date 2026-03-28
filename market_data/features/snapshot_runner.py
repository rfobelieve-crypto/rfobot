"""
Snapshot runner: periodically check for pending snapshots and compute them.

Scans event_registry for events that are due for 15m/1h/4h snapshots
but haven't been computed yet. Builds features, scores, and saves.

Usage:
    python -m market_data.features.snapshot_runner              # run once
    python -m market_data.features.snapshot_runner --loop       # run every 60s
"""

import os
import sys
import time
import logging

import requests as _requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_data.storage.db import run_migration
from market_data.features.snapshot_repository import get_pending_snapshots, save_snapshot
from market_data.features.snapshot_builder import build_snapshot
from market_data.features.snapshot_scorer import score_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Telegram push config (from env)
_TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LOOP_INTERVAL = 60  # 1 minute


def _notify_telegram(features: dict):
    """Send snapshot result to Telegram (best-effort, never raises)."""
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    try:
        snap_type = features.get("snapshot_type", "?")
        uuid_short = features.get("event_uuid", "")[:8]
        bias = features.get("bias", "?")
        confidence = features.get("confidence_score", 0)
        rev = features.get("reversal_score", 0)
        cont = features.get("continuation_score", 0)
        side = features.get("liquidity_side", "?")
        symbol = features.get("canonical_symbol", "?")

        bias_icon = {"reversal": "🔄", "continuation": "➡️", "neutral": "⚖️"}.get(bias, "❓")

        lines = [
            f"📸 快照完成: {snap_type}",
            f"事件: {uuid_short} ({side.upper()})",
            f"幣種: {symbol}",
            f"─────────────",
            f"bias: {bias_icon} {bias}",
            f"reversal: {rev:.0f}  continuation: {cont:.0f}",
            f"confidence: {confidence:.2f}",
        ]

        # OI info if available
        oi_pct = features.get("oi_change_total_pct")
        if oi_pct is not None:
            oi_dir = "📈" if float(oi_pct) > 0 else "📉"
            lines.append(f"OI change: {oi_dir} {oi_pct:+.2f}%")

        price_pct = features.get("price_change_pct")
        if price_pct is not None:
            lines.append(f"price change: {price_pct:+.2f}%")

        text = "\n".join(lines)
        _requests.post(
            f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage",
            data={"chat_id": _TG_CHAT_ID, "text": text},
            timeout=10,
        )
    except Exception:
        logger.debug("Telegram notification failed (non-critical)")


def process_once() -> int:
    """Find and process all pending snapshots."""
    pending = get_pending_snapshots()
    if not pending:
        logger.info("No pending snapshots.")
        return 0

    logger.info("Found %d pending snapshots to compute.", len(pending))
    count = 0

    for item in pending:
        snap_type = item["_snapshot_type"]
        offset_sec = item["_offset_sec"]
        event_uuid = item["event_uuid"]

        try:
            features = build_snapshot(item, snap_type, offset_sec)
            scores = score_snapshot(features)

            # Merge scores into snapshot dict
            features["reversal_score"] = scores["reversal_score"]
            features["continuation_score"] = scores["continuation_score"]
            features["confidence_score"] = scores["confidence_score"]
            features["bias"] = scores["bias"]

            save_snapshot(features)
            _notify_telegram(features)
            count += 1

        except Exception:
            logger.exception("Failed to compute %s snapshot for event %s",
                             snap_type, event_uuid[:8])

    return count


def main():
    # Run migration
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "migrations",
    )
    mig_path = os.path.join(migrations_dir, "004_event_feature_snapshots.sql")
    if os.path.exists(mig_path):
        try:
            run_migration(mig_path)
        except Exception:
            logger.exception("Migration 004 failed (tables may already exist)")

    loop_mode = "--loop" in sys.argv

    if loop_mode:
        logger.info("Snapshot runner starting in loop mode (every %ds)", LOOP_INTERVAL)
        while True:
            try:
                process_once()
            except Exception:
                logger.exception("process_once error")
            time.sleep(LOOP_INTERVAL)
    else:
        count = process_once()
        logger.info("Done. Computed %d snapshots.", count)


if __name__ == "__main__":
    main()
