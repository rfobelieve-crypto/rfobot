"""
Snapshot data collector — accumulates order book depth, aggTrades,
and options/ETF/DVOL summaries over time for future model training.

Dual-write: saves to both MySQL (persistent, survives redeploy) and
local parquet (fast reads for local analysis).

MySQL tables (auto-created on first write):
  - indicator_depth_snapshots
  - indicator_aggtrades_snapshots
  - indicator_options_snapshots
  - indicator_sentiment_snapshots
  - indicator_history

Called from app.py / auto_update.py after each hourly update cycle.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(__file__).parent / "model_artifacts" / ".snapshots"

# ── Schema definitions ────────────────────────────────────────────────────

DEPTH_COLUMNS = [
    "bid_depth_usd", "ask_depth_usd", "depth_imbalance",
    "near_bid_usd", "near_ask_usd", "near_imbalance",
    "spread_bps", "mid_price",
]

AGGTRADES_COLUMNS = [
    "large_buy_usd", "large_sell_usd", "large_delta_usd",
    "large_count", "large_ratio", "large_buy_ratio",
    "small_buy_usd", "small_sell_usd", "small_delta_usd",
    "avg_trade_usd", "total_count", "total_usd",
]

OPTIONS_COLUMNS = [
    "dvol_value", "dvol_open", "dvol_high", "dvol_low", "dvol_change",
    "pc_volume_ratio", "pc_oi_ratio_deribit",
    "iv_skew", "mean_otm_put_iv", "mean_otm_call_iv",
    "total_call_oi", "total_put_oi",
    "max_pain_price", "cg_pc_oi_ratio",
    "call_oi_notional", "put_oi_notional", "opt_futures_ratio",
    "etf_net_flow_usd", "etf_flow_ibit", "etf_flow_fbtc", "etf_btc_price",
]

# Field name mapping: source dict key → DB column name
OPTIONS_FIELD_MAP = {
    "dvol_value": "dvol_value", "dvol_open": "dvol_open",
    "dvol_high": "dvol_high", "dvol_low": "dvol_low",
    "dvol_change": "dvol_change",
    "pc_volume_ratio": "pc_volume_ratio",
    "pc_oi_ratio_deribit": "pc_oi_ratio_deribit",
    "iv_skew": "iv_skew", "mean_otm_put_iv": "mean_otm_put_iv",
    "mean_otm_call_iv": "mean_otm_call_iv",
    "total_call_oi": "total_call_oi", "total_put_oi": "total_put_oi",
    "max_pain_price": "max_pain_price",
    "pc_oi_ratio": "cg_pc_oi_ratio",  # rename on save
    "call_oi_notional": "call_oi_notional",
    "put_oi_notional": "put_oi_notional",
    "opt_futures_ratio": "opt_futures_ratio",
    "etf_net_flow_usd": "etf_net_flow_usd",
    "etf_flow_ibit": "etf_flow_ibit",
    "etf_flow_fbtc": "etf_flow_fbtc",
    "etf_btc_price": "etf_btc_price",
}

SENTIMENT_COLUMNS = [
    "fear_greed_value", "etf_aum_usd",
    "futures_netflow_5m", "futures_netflow_15m", "futures_netflow_1h",
    "futures_netflow_4h", "futures_netflow_24h",
    "spot_netflow_5m", "spot_netflow_15m", "spot_netflow_1h",
    "spot_netflow_4h", "spot_netflow_24h",
    "hl_whale_count", "hl_whale_net_usd", "hl_whale_long_pct",
]

SENTIMENT_FIELD_MAP = {c: c for c in SENTIMENT_COLUMNS}


# ── MySQL helpers ─────────────────────────────────────────────────────────

_tables_ensured: set[str] = set()


def _get_db_conn():
    """Get MySQL connection from shared pool."""
    from shared.db import get_db_conn
    return get_db_conn()


def _ensure_table(table_name: str, columns: list[str]):
    """Create table if not exists. Only runs once per process."""
    if table_name in _tables_ensured:
        return

    col_defs = ",\n    ".join(f"`{c}` DOUBLE DEFAULT 0" for c in columns)
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        `dt` DATETIME NOT NULL,
        {col_defs},
        PRIMARY KEY (`dt`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        _tables_ensured.add(table_name)
        logger.info("Table ensured: %s", table_name)
    except Exception as e:
        logger.error("Failed to ensure table %s: %s", table_name, e)
    finally:
        conn.close()


def _upsert_row(table_name: str, columns: list[str], ts: datetime, data: dict):
    """INSERT ... ON DUPLICATE KEY UPDATE one row to MySQL."""
    _ensure_table(table_name, columns)

    values = [ts.strftime("%Y-%m-%d %H:%M:%S")]
    for col in columns:
        values.append(float(data.get(col, 0)))

    col_list = ", ".join(f"`{c}`" for c in ["dt"] + columns)
    placeholders = ", ".join(["%s"] * (len(columns) + 1))
    update_clause = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns)

    sql = (
        f"INSERT INTO `{table_name}` ({col_list}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, values)
    except Exception as e:
        logger.error("MySQL upsert to %s failed: %s", table_name, e)
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────

def save_depth_snapshot(depth: dict, bar_time: datetime | None = None):
    """Save one depth snapshot row to MySQL + local parquet."""
    if not depth or "depth_imbalance" not in depth:
        return

    ts = bar_time or datetime.now(timezone.utc)

    # MySQL (persistent)
    try:
        _upsert_row("indicator_depth_snapshots", DEPTH_COLUMNS, ts, depth)
    except Exception as e:
        logger.error("Depth MySQL save failed: %s", e)

    # Local parquet (fast access)
    _append_parquet_row(
        SNAPSHOT_DIR / "depth_snapshots.parquet", ts,
        {c: depth.get(c, 0) for c in DEPTH_COLUMNS},
    )
    logger.info("Depth snapshot saved (imb=%.3f, spread=%.1fbps)",
                depth.get("depth_imbalance", 0), depth.get("spread_bps", 0))


def save_aggtrades_snapshot(aggtrades: dict, bar_time: datetime | None = None):
    """Save one aggTrades summary row to MySQL + local parquet."""
    if not aggtrades or "large_ratio" not in aggtrades:
        return

    ts = bar_time or datetime.now(timezone.utc)

    # MySQL (persistent)
    try:
        _upsert_row("indicator_aggtrades_snapshots", AGGTRADES_COLUMNS, ts, aggtrades)
    except Exception as e:
        logger.error("AggTrades MySQL save failed: %s", e)

    # Local parquet
    _append_parquet_row(
        SNAPSHOT_DIR / "aggtrades_snapshots.parquet", ts,
        {c: aggtrades.get(c, 0) for c in AGGTRADES_COLUMNS},
    )
    logger.info("AggTrades snapshot saved (large_ratio=%.1f%%, large_delta=$%.0f)",
                aggtrades.get("large_ratio", 0) * 100,
                aggtrades.get("large_delta_usd", 0))


def save_options_snapshot(options_data: dict, bar_time: datetime | None = None):
    """Save one options/ETF/DVOL snapshot row to MySQL + local parquet."""
    if not options_data:
        return

    ts = bar_time or datetime.now(timezone.utc)

    # Map field names from source dict to DB column names
    mapped = {}
    for src_key, db_col in OPTIONS_FIELD_MAP.items():
        mapped[db_col] = options_data.get(src_key, 0)

    # MySQL (persistent)
    try:
        _upsert_row("indicator_options_snapshots", OPTIONS_COLUMNS, ts, mapped)
    except Exception as e:
        logger.error("Options MySQL save failed: %s", e)

    # Local parquet
    _append_parquet_row(
        SNAPSHOT_DIR / "options_snapshots.parquet", ts, mapped,
    )
    logger.info("Options snapshot saved (DVOL=%.1f, IV_skew=%.1f, P/C=%.2f, ETF=$%.1fM)",
                options_data.get("dvol_value", 0),
                options_data.get("iv_skew", 0),
                options_data.get("pc_volume_ratio", 0),
                options_data.get("etf_net_flow_usd", 0) / 1e6)


def save_sentiment_snapshot(sentiment_data: dict, bar_time: datetime | None = None):
    """Save one sentiment/whale/netflow snapshot row to MySQL + local parquet."""
    if not sentiment_data:
        return

    ts = bar_time or datetime.now(timezone.utc)

    # Map field names
    mapped = {}
    for src_key, db_col in SENTIMENT_FIELD_MAP.items():
        mapped[db_col] = sentiment_data.get(src_key, 0)

    # MySQL (persistent)
    try:
        _upsert_row("indicator_sentiment_snapshots", SENTIMENT_COLUMNS, ts, mapped)
    except Exception as e:
        logger.error("Sentiment MySQL save failed: %s", e)

    # Local parquet (fast access)
    _append_parquet_row(
        SNAPSHOT_DIR / "sentiment_snapshots.parquet", ts, mapped,
    )
    logger.info("Sentiment snapshot saved (F&G=%.0f, AUM=$%.1fB, whales=%d)",
                sentiment_data.get("fear_greed_value", 0),
                sentiment_data.get("etf_aum_usd", 0) / 1e9,
                sentiment_data.get("hl_whale_count", 0))


def save_indicator_history(df: pd.DataFrame):
    """
    Save indicator prediction history to MySQL.
    Only saves the last row (most recent prediction) to avoid bulk writes.
    Full history is still in parquet for chart rendering.
    """
    if df.empty:
        return

    table = "indicator_history"
    columns = [
        "open", "high", "low", "close",
        "pred_return_4h", "pred_direction_code", "confidence_score",
        "strength_code", "bull_bear_power", "regime_code",
        "up_pred", "down_pred", "strength_raw", "dynamic_deadzone",
        "dir_prob_up",
    ]

    _ensure_indicator_history_table(table)

    last = df.iloc[-1]
    ts = df.index[-1]
    if hasattr(ts, 'to_pydatetime'):
        ts = ts.to_pydatetime()

    # Map string fields to numeric codes for MySQL DOUBLE columns
    dir_map = {"UP": 1, "DOWN": -1, "NEUTRAL": 0}
    str_map = {"Strong": 3, "Moderate": 2, "Weak": 1}
    regime_map = {"TRENDING_BULL": 2, "TRENDING_BEAR": -2, "CHOPPY": 0, "WARMUP": -99}

    values = [
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        float(last.get("open", 0) or 0),
        float(last.get("high", 0) or 0),
        float(last.get("low", 0) or 0),
        float(last.get("close", 0) or 0),
        float(last.get("pred_return_4h", 0) or 0),
        float(dir_map.get(str(last.get("pred_direction", "NEUTRAL")), 0)),
        float(last.get("confidence_score", 0) or 0),
        float(str_map.get(str(last.get("strength_score", "Weak")), 1)),
        float(last.get("bull_bear_power", 0) or 0),
        float(regime_map.get(str(last.get("regime", "CHOPPY")), 0)),
        float(last.get("up_pred", 0) or 0),
        float(last.get("down_pred", 0) or 0),
        float(last.get("strength_raw", 0) or 0),
        float(last.get("dynamic_deadzone", 0) or 0),
        float(last.get("dir_prob_up", 0.5) or 0.5),
    ]

    col_list = ", ".join(f"`{c}`" for c in ["dt"] + columns)
    placeholders = ", ".join(["%s"] * len(values))
    update_clause = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns)

    sql = (
        f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, values)
    except Exception as e:
        logger.error("Indicator history MySQL save failed: %s", e)
    finally:
        conn.close()


def _ensure_indicator_history_table(table: str):
    """Create indicator_history table with mixed types."""
    if table in _tables_ensured:
        return

    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        `dt` DATETIME NOT NULL,
        `open` DOUBLE DEFAULT 0,
        `high` DOUBLE DEFAULT 0,
        `low` DOUBLE DEFAULT 0,
        `close` DOUBLE DEFAULT 0,
        `pred_return_4h` DOUBLE DEFAULT 0,
        `pred_direction_code` DOUBLE DEFAULT 0,
        `confidence_score` DOUBLE DEFAULT 0,
        `strength_code` DOUBLE DEFAULT 0,
        `bull_bear_power` DOUBLE DEFAULT 0,
        `regime_code` DOUBLE DEFAULT 0,
        `up_pred` DOUBLE DEFAULT 0,
        `down_pred` DOUBLE DEFAULT 0,
        `strength_raw` DOUBLE DEFAULT 0,
        `dynamic_deadzone` DOUBLE DEFAULT 0,
        `dir_prob_up` DOUBLE DEFAULT 0.5,
        PRIMARY KEY (`dt`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    conn = _get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        _tables_ensured.add(table)
        logger.info("Table ensured: %s", table)
    except Exception as e:
        logger.error("Failed to ensure table %s: %s", table, e)
    finally:
        conn.close()


# ── Local parquet helpers (unchanged) ─────────────────────────────────────

def _append_parquet_row(path: Path, ts: datetime, data: dict):
    """Append one row to local parquet file."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([data])
    row["dt"] = pd.to_datetime(ts, utc=True)
    row = row.set_index("dt")

    try:
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, row])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(path)
        else:
            row.to_parquet(path)
    except Exception as e:
        logger.error("Failed to save parquet to %s: %s", path, e)


def get_snapshot_stats() -> dict:
    """Return stats about accumulated snapshot data (for health checks)."""
    stats = {}

    # Check MySQL tables
    try:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            for table in ["indicator_depth_snapshots", "indicator_aggtrades_snapshots",
                          "indicator_options_snapshots", "indicator_sentiment_snapshots",
                          "indicator_history"]:
                try:
                    cur.execute(f"SELECT COUNT(*) as cnt, MIN(dt) as first_dt, MAX(dt) as last_dt FROM `{table}`")
                    row = cur.fetchone()
                    stats[f"mysql_{table}"] = {
                        "rows": row["cnt"],
                        "first": str(row["first_dt"]) if row["first_dt"] else None,
                        "last": str(row["last_dt"]) if row["last_dt"] else None,
                    }
                except Exception:
                    stats[f"mysql_{table}"] = {"rows": 0, "error": "table not found"}
        conn.close()
    except Exception as e:
        stats["mysql_error"] = str(e)

    # Check local parquet files
    for name in ["depth_snapshots", "aggtrades_snapshots", "options_snapshots", "sentiment_snapshots"]:
        path = SNAPSHOT_DIR / f"{name}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                stats[f"parquet_{name}"] = {
                    "rows": len(df),
                    "first": str(df.index[0]),
                    "last": str(df.index[-1]),
                }
            except Exception:
                stats[f"parquet_{name}"] = {"rows": 0, "error": "corrupt"}
        else:
            stats[f"parquet_{name}"] = {"rows": 0}

    return stats
