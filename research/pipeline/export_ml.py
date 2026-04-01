"""
Export features_{tf} table to ML-ready Parquet files.

Outputs:
  research/ml_data/{symbol}_{tf}_tabular.parquet  — flat, for XGBoost/LightGBM
  research/ml_data/{symbol}_{tf}_sequence.parquet — windowed sequences, for LSTM/Transformer

Usage:
    python -m research.pipeline.export_ml
    python -m research.pipeline.export_ml --timeframe 15m
    python -m research.pipeline.export_ml --timeframe 15m --sequence-len 64
"""
from __future__ import annotations

import sys
import os
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "ml_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS   = ["label_5m", "label_15m", "label_1h"]
META_COLS    = ["id", "symbol", "ts_open", "computed_at",
                "future_return_5m", "future_return_15m", "future_return_1h"]
EXCLUDE_COLS = set(META_COLS) | set(LABEL_COLS)


def _load(symbol: str, timeframe: str) -> pd.DataFrame:
    table = f"features_{timeframe}"
    conn  = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {table} WHERE symbol = %s ORDER BY ts_open",
                (symbol,)
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    for col in df.columns:
        if col not in LABEL_COLS and col not in ("symbol",):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def export_tabular(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    """Flat Parquet for XGBoost / LightGBM."""
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Include forward returns + labels so user can pick target
    raw_keep = (["ts_open"] + feat_cols + LABEL_COLS +
                ["future_return_5m", "future_return_15m", "future_return_1h"])
    seen: set = set()
    keep = [c for c in raw_keep if c in df.columns and not (c in seen or seen.add(c))]

    out = df[keep].copy()

    # Drop rows where core features are all NaN
    core = [c for c in feat_cols if c not in
            {f"return_lag_{i}" for i in range(1, 11)} |
            {f"delta_lag_{i}"  for i in range(1, 11)}]
    out = out.dropna(subset=[c for c in core if c in out.columns], how="all")

    path = OUT_DIR / f"{symbol.replace('-','_')}_{timeframe}_tabular.parquet"
    out.to_parquet(path, index=False, compression="snappy")
    logger.info("Tabular: %s  (%d rows × %d cols)", path.name, len(out), len(out.columns))
    return path


def export_sequence(symbol: str, timeframe: str, df: pd.DataFrame,
                    seq_len: int = 64) -> Path:
    """Windowed sequence Parquet for LSTM / Transformer.

    Schema:
        ts_open      : int64  (last bar in window)
        label_15m    : str    (label of last bar)
        bull_bear_score : float
        features     : bytes  (float32 array, shape seq_len × n_features, row-major)
        n_features   : int
    """
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE_COLS and c != "bull_bear_score"]

    # Only use rows where all feature cols present
    clean = df[["ts_open"] + feat_cols + LABEL_COLS + ["bull_bear_score"]].copy()
    feat_arr = clean[feat_cols].values.astype(np.float32)
    n_feat   = len(feat_cols)

    records = []
    for end in range(seq_len - 1, len(clean)):
        start  = end - seq_len + 1
        window = feat_arr[start: end + 1]
        if np.isnan(window).all():
            continue
        window = np.nan_to_num(window, nan=0.0)
        row    = clean.iloc[end]
        records.append({
            "ts_open":         int(row["ts_open"]),
            "label_15m":       row.get("label_15m"),
            "label_5m":        row.get("label_5m"),
            "label_1h":        row.get("label_1h"),
            "bull_bear_score": float(row["bull_bear_score"]) if pd.notna(row.get("bull_bear_score")) else None,
            "features":        window.tobytes(),
            "n_features":      n_feat,
            "seq_len":         seq_len,
        })

    seq_df = pd.DataFrame(records)
    path   = OUT_DIR / f"{symbol.replace('-','_')}_{timeframe}_sequence.parquet"
    seq_df.to_parquet(path, index=False, compression="snappy")
    logger.info("Sequence: %s  (%d windows × %d bars × %d features)",
                path.name, len(seq_df), seq_len, n_feat)

    # Save feature column names for decoding
    meta_path = OUT_DIR / f"{symbol.replace('-','_')}_{timeframe}_sequence_meta.txt"
    meta_path.write_text("\n".join(feat_cols))

    return path


def export(symbol: str = "BTC-USD", timeframe: str = "15m",
           sequence_len: int = 64):
    df = _load(symbol, timeframe)
    if df.empty:
        logger.warning("No data in features_%s for %s", timeframe, symbol)
        return

    logger.info("Loaded %d rows from features_%s", len(df), timeframe)
    export_tabular(symbol, timeframe, df)
    export_sequence(symbol, timeframe, df, seq_len=sequence_len)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",       default="BTC-USD")
    ap.add_argument("--timeframe",    default="15m", choices=["5m","15m","1h"])
    ap.add_argument("--sequence-len", default=64, type=int)
    args = ap.parse_args()
    export(args.symbol, args.timeframe, args.sequence_len)
