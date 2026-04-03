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


def save_options_snapshot(options_data: dict, bar_time: datetime | None = None):
    """Append one options/ETF/DVOL snapshot row to the accumulating parquet."""
    if not options_data:
        return

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / "options_snapshots.parquet"

    ts = bar_time or datetime.now(timezone.utc)
    row = pd.DataFrame([{
        "dt": ts,
        # Deribit DVOL
        "dvol_value": options_data.get("dvol_value", 0),
        "dvol_open": options_data.get("dvol_open", 0),
        "dvol_high": options_data.get("dvol_high", 0),
        "dvol_low": options_data.get("dvol_low", 0),
        "dvol_change": options_data.get("dvol_change", 0),
        # Deribit options
        "pc_volume_ratio": options_data.get("pc_volume_ratio", 0),
        "pc_oi_ratio_deribit": options_data.get("pc_oi_ratio_deribit", 0),
        "iv_skew": options_data.get("iv_skew", 0),
        "mean_otm_put_iv": options_data.get("mean_otm_put_iv", 0),
        "mean_otm_call_iv": options_data.get("mean_otm_call_iv", 0),
        "total_call_oi": options_data.get("total_call_oi", 0),
        "total_put_oi": options_data.get("total_put_oi", 0),
        # Coinglass options
        "max_pain_price": options_data.get("max_pain_price", 0),
        "cg_pc_oi_ratio": options_data.get("pc_oi_ratio", 0),
        "call_oi_notional": options_data.get("call_oi_notional", 0),
        "put_oi_notional": options_data.get("put_oi_notional", 0),
        "opt_futures_ratio": options_data.get("opt_futures_ratio", 0),
        # ETF flows
        "etf_net_flow_usd": options_data.get("etf_net_flow_usd", 0),
        "etf_flow_ibit": options_data.get("etf_flow_ibit", 0),
        "etf_flow_fbtc": options_data.get("etf_flow_fbtc", 0),
        "etf_btc_price": options_data.get("etf_btc_price", 0),
    }])
    row["dt"] = pd.to_datetime(row["dt"], utc=True)
    row = row.set_index("dt")

    _append_parquet(path, row)
    logger.info("Options snapshot saved (DVOL=%.1f, IV_skew=%.1f, P/C=%.2f, ETF=$%.1fM)",
                options_data.get("dvol_value", 0),
                options_data.get("iv_skew", 0),
                options_data.get("pc_volume_ratio", 0),
                options_data.get("etf_net_flow_usd", 0) / 1e6)


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
    for name in ["depth_snapshots", "aggtrades_snapshots", "options_snapshots"]:
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
