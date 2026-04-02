"""
Import Binance USDT-M klines CSV → ohlcv_1m table.

Sources (in priority order):
  1. market_data/raw_data/klines/binance/{Month}/BTCUSDT-15m-*.csv  (daily 15m files)
  2. market_data/raw_data/klines/BTCUSDT-1m-*.csv  (legacy flat structure)

Note: if only 15m files are available, ohlcv_1m will contain 15m bars.
      The feature_builder can then produce 15m and 1h features (not 5m).

Usage:
    python -m market_data.backfill.import_klines
"""
from __future__ import annotations

import os
import sys
import time
import logging
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
KLINES_DIR = ROOT / "market_data" / "raw_data" / "klines"
SYMBOL     = "BTC-USD"
BATCH      = 2000

CSV_COLS = [
    "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]


def _find_klines_files() -> list[str]:
    """Find klines CSV files. Checks new subdirectory layout first, then legacy flat layout."""
    # New layout: klines/binance/{Month}/BTCUSDT-15m-*.csv
    new_layout = sorted(glob.glob(str(KLINES_DIR / "binance" / "*" / "BTCUSDT-15m-*.csv")))
    if new_layout:
        logger.info("Using new layout: %d daily 15m files", len(new_layout))
        return new_layout

    # Legacy layout: klines/BTCUSDT-1m-*.csv
    legacy = sorted(glob.glob(str(KLINES_DIR / "BTCUSDT-1m-*.csv")))
    if legacy:
        logger.info("Using legacy layout: %d 1m files", len(legacy))
        return legacy

    return []


def import_klines(symbol: str = SYMBOL) -> int:
    files = _find_klines_files()
    if not files:
        logger.warning("No klines CSV found in %s", KLINES_DIR)
        return 0

    logger.info("Found %d klines files", len(files))

    # Read all files into one DataFrame
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, header=None, names=CSV_COLS,
                             dtype={"open_time": np.int64})
        except Exception:
            # File has header row
            df = pd.read_csv(fp)
            df.columns = CSV_COLS[:len(df.columns)]

        # Ensure open_time is int64
        df["open_time"] = df["open_time"].astype(np.int64)
        frames.append(df[["open_time","open","high","low","close",
                           "volume","quote_volume","count",
                           "taker_buy_volume","taker_buy_quote_volume"]])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates("open_time").sort_values("open_time")
    logger.info("Total rows after dedup: %d", len(all_df))

    sql = """
        INSERT INTO ohlcv_1m
            (symbol, ts_open, open, high, low, close,
             volume, quote_vol, trade_count, taker_buy_vol, taker_buy_quote)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            open            = VALUES(open),
            high            = VALUES(high),
            low             = VALUES(low),
            close           = VALUES(close),
            volume          = VALUES(volume),
            quote_vol       = VALUES(quote_vol),
            trade_count     = VALUES(trade_count),
            taker_buy_vol   = VALUES(taker_buy_vol),
            taker_buy_quote = VALUES(taker_buy_quote)
    """

    def _v(x):
        if pd.isna(x):
            return None
        return float(x)

    total = 0
    for i in range(0, len(all_df), BATCH):
        chunk = all_df.iloc[i: i + BATCH]
        params = []
        for r in chunk.itertuples(index=False):
            params.append((
                symbol,
                np.int64(r.open_time),
                _v(r.open), _v(r.high), _v(r.low), _v(r.close),
                _v(r.volume), _v(r.quote_volume),
                int(r.count) if not pd.isna(r.count) else None,
                _v(r.taker_buy_volume), _v(r.taker_buy_quote_volume),
            ))
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
            conn.commit()
            total += len(params)
        except Exception:
            logger.exception("Batch insert failed at row %d", i)
        finally:
            conn.close()

        if (i // BATCH) % 50 == 0:
            logger.info("  progress: %d / %d rows", total, len(all_df))

    logger.info("ohlcv_1m: %d rows written for %s", total, symbol)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import_klines()
