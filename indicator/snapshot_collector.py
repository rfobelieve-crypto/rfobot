"""
Snapshot data collector — accumulates order book depth and aggTrades
summaries over time for future model training.

Stores data in parquet files that grow incrementally:
  - model_artifacts/.snapshots/depth_snapshots.parquet
  - model_artifacts/.snapshots/aggtrades_snapshots.parquet

Called from app.py / auto_update.py after each hourly update cycle.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(__file__).parent / "model_artifacts" / ".snapshots"


def save_depth_snapshot(depth: dict, bar_time: datetime | None = None):
    """Append one depth snapshot row to the accumulating parquet."""
    if not depth or "depth_imbalance" not in depth:
        return

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / "depth_snapshots.parquet"

    ts = bar_time or datetime.now(timezone.utc)
    row = pd.DataFrame([{
        "dt": ts,
        "bid_depth_usd": depth.get("bid_depth_usd", 0),
        "ask_depth_usd": depth.get("ask_depth_usd", 0),
        "depth_imbalance": depth.get("depth_imbalance", 0),
        "near_bid_usd": depth.get("near_bid_usd", 0),
        "near_ask_usd": depth.get("near_ask_usd", 0),
        "near_imbalance": depth.get("near_imbalance", 0),
        "spread_bps": depth.get("spread_bps", 0),
        "mid_price": depth.get("mid_price", 0),
    }])
    row["dt"] = pd.to_datetime(row["dt"], utc=True)
    row = row.set_index("dt")

    _append_parquet(path, row)
    logger.info("Depth snapshot saved (imb=%.3f, spread=%.1fbps)",
                depth.get("depth_imbalance", 0), depth.get("spread_bps", 0))


def save_aggtrades_snapshot(aggtrades: dict, bar_time: datetime | None = None):
    """Append one aggTrades summary row to the accumulating parquet."""
    if not aggtrades or "large_ratio" not in aggtrades:
        return

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / "aggtrades_snapshots.parquet"

    ts = bar_time or datetime.now(timezone.utc)
    row = pd.DataFrame([{
        "dt": ts,
        "large_buy_usd": aggtrades.get("large_buy_usd", 0),
        "large_sell_usd": aggtrades.get("large_sell_usd", 0),
        "large_delta_usd": aggtrades.get("large_delta_usd", 0),
        "large_count": aggtrades.get("large_count", 0),
        "large_ratio": aggtrades.get("large_ratio", 0),
        "large_buy_ratio": aggtrades.get("large_buy_ratio", 0.5),
        "small_buy_usd": aggtrades.get("small_buy_usd", 0),
        "small_sell_usd": aggtrades.get("small_sell_usd", 0),
        "small_delta_usd": aggtrades.get("small_delta_usd", 0),
        "avg_trade_usd": aggtrades.get("avg_trade_usd", 0),
        "total_count": aggtrades.get("total_count", 0),
        "total_usd": aggtrades.get("total_usd", 0),
    }])
    row["dt"] = pd.to_datetime(row["dt"], utc=True)
    row = row.set_index("dt")

    _append_parquet(path, row)
    logger.info("AggTrades snapshot saved (large_ratio=%.1f%%, large_delta=$%.0f)",
                aggtrades.get("large_ratio", 0) * 100,
                aggtrades.get("large_delta_usd", 0))


def _append_parquet(path: Path, new_row: pd.DataFrame):
    """Append row to existing parquet or create new one."""
    try:
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_row])
            # Dedup by index (keep latest)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(path)
        else:
            new_row.to_parquet(path)
    except Exception as e:
        logger.error("Failed to save snapshot to %s: %s", path, e)


def get_snapshot_stats() -> dict:
    """Return stats about accumulated snapshot data (for health checks)."""
    stats = {}
    for name in ["depth_snapshots", "aggtrades_snapshots"]:
        path = SNAPSHOT_DIR / f"{name}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                stats[name] = {
                    "rows": len(df),
                    "first": str(df.index[0]),
                    "last": str(df.index[-1]),
                }
            except Exception:
                stats[name] = {"rows": 0, "error": "corrupt"}
        else:
            stats[name] = {"rows": 0}
    return stats
