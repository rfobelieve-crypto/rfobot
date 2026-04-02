"""
Download / import aggTrades → flow_bars_1m_ml.

Sources:
  --from-local  (default): read pre-downloaded files from raw_data/aggtrades/
                Binance: raw_data/aggtrades/binance/{Month}/BTCUSDT-aggTrades-YYYY-MM-DD.csv
                OKX:     raw_data/aggtrades/okx/{Month}/BTC-USDT-SWAP-trades-YYYY-MM-DD.csv

                Local mode stores:
                  - exchange='binance'
                  - exchange='okx'
                  - exchange='all' (binance + okx merged at 1m bar level)

  --download:   download from Binance Vision (USDT-M only) and aggregate.

Binance format:
  agg_trade_id, price, quantity, first_trade_id, last_trade_id,
  transact_time, is_buyer_maker
  notional_usd = price × quantity

OKX format:
  instrument_name, trade_id, side, price, size, created_time
  notional_usd = price × size × 0.01  (contract_size = 0.01 BTC/contract)

Usage:
    python -m market_data.backfill.import_aggtrades
    python -m market_data.backfill.import_aggtrades --download
    python -m market_data.backfill.import_aggtrades --months 2025-01 2026-02
    python -m market_data.backfill.import_aggtrades --daily-start 2026-03-01
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "market_data" / "raw_data" / "aggtrades_usdt"
TMP_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_AGGTRADES_DIR = ROOT / "market_data" / "raw_data" / "aggtrades"

SYMBOL = "BTC-USD"
BINANCE_SYMBOL = "BTCUSDT"
LARGE_THRESHOLD = 100_000.0   # $100k notional
OKX_CONTRACT_SIZE = 0.01      # BTC per contract for BTC-USDT-SWAP
CHUNK_SIZE = 1_000_000
BATCH = 2000
BASE_URL = "https://data.binance.vision/data/futures/um"


# ─── Download helper ──────────────────────────────────────────────────────────

def _download_zip(url: str, dest: Path, retries: int = 3) -> bool:
    """Download zip and extract CSV to dest. Returns True on success."""
    if dest.exists():
        return True

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 404:
                logger.warning("404: %s", url)
                return False
            resp.raise_for_status()

            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                logger.warning("No CSV found in zip: %s", url)
                return False

            zf.extract(csv_names[0], dest.parent)
            extracted = dest.parent / csv_names[0]
            if extracted != dest:
                if dest.exists():
                    dest.unlink()
                extracted.rename(dest)
            return True

        except Exception as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)

    return False


# ─── Binance aggregation ──────────────────────────────────────────────────────

def _agg_chunk_binance(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate one chunk of Binance USDT-M trades to 1m bars.
    USDT-M notional = price × quantity.
    """
    chunk["ts_open"] = (
        chunk["transact_time"].astype(np.int64) // np.int64(60_000)
    ) * np.int64(60_000)

    chunk["notional"] = chunk["price"].astype(float) * chunk["quantity"].astype(float)
    chunk["is_sell"] = chunk["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])

    chunk["buy_vol"] = np.where(~chunk["is_sell"], chunk["notional"], 0.0)
    chunk["sell_vol"] = np.where(chunk["is_sell"], chunk["notional"], 0.0)

    large = chunk["notional"] > LARGE_THRESHOLD
    chunk["large_buy"] = np.where((~chunk["is_sell"]) & large, chunk["notional"], 0.0)
    chunk["large_sell"] = np.where(chunk["is_sell"] & large, chunk["notional"], 0.0)

    chunk["buy_cnt"] = (~chunk["is_sell"]).astype(int)
    chunk["sell_cnt"] = chunk["is_sell"].astype(int)

    return chunk.groupby("ts_open", as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("notional", "sum"),
        large_buy_vol=("large_buy", "sum"),
        large_sell_vol=("large_sell", "sum"),
        trade_count=("notional", "count"),
        buy_count=("buy_cnt", "sum"),
        sell_count=("sell_cnt", "sum"),
    )


def _process_binance_file_to_df(fp: Path) -> pd.DataFrame:
    """Read one Binance CSV and return aggregated 1m bars."""
    chunks = []
    try:
        for chunk in pd.read_csv(
            fp,
            chunksize=CHUNK_SIZE,
            dtype={
                "transact_time": np.int64,
                "price": float,
                "quantity": float,
                "agg_trade_id": np.int64,
            },
        ):
            chunks.append(_agg_chunk_binance(chunk))
    except Exception:
        logger.exception("Error reading Binance file %s", fp.name)
        return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    bars = combined.groupby("ts_open", as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("volume", "sum"),
        large_buy_vol=("large_buy_vol", "sum"),
        large_sell_vol=("large_sell_vol", "sum"),
        trade_count=("trade_count", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
    )
    bars["ts_open"] = bars["ts_open"].astype(np.int64)
    return bars


def _process_binance_file_upsert(fp: Path, symbol: str) -> int:
    """Read one Binance CSV, aggregate to 1m bars, upsert exchange='all' for download mode."""
    bars = _process_binance_file_to_df(fp)
    if bars.empty:
        return 0

    bars = _finalize_bars(bars)
    _upsert(bars, symbol, exchange="all")
    return len(bars)


# ─── OKX aggregation ──────────────────────────────────────────────────────────

def _agg_chunk_okx(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate one chunk of OKX BTC-USDT-SWAP trades to 1m bars.
    OKX format: instrument_name, trade_id, side, price, size, created_time
    notional_usd = price × size × 0.01  (contract_size = 0.01 BTC/contract)
    """
    chunk["ts_open"] = (
        chunk["created_time"].astype(np.int64) // np.int64(60_000)
    ) * np.int64(60_000)

    chunk["notional"] = (
        chunk["price"].astype(float)
        * chunk["size"].astype(float)
        * OKX_CONTRACT_SIZE
    )
    chunk["is_sell"] = chunk["side"].astype(str).str.lower() == "sell"

    chunk["buy_vol"] = np.where(~chunk["is_sell"], chunk["notional"], 0.0)
    chunk["sell_vol"] = np.where(chunk["is_sell"], chunk["notional"], 0.0)

    large = chunk["notional"] > LARGE_THRESHOLD
    chunk["large_buy"] = np.where((~chunk["is_sell"]) & large, chunk["notional"], 0.0)
    chunk["large_sell"] = np.where(chunk["is_sell"] & large, chunk["notional"], 0.0)

    chunk["buy_cnt"] = (~chunk["is_sell"]).astype(int)
    chunk["sell_cnt"] = chunk["is_sell"].astype(int)

    return chunk.groupby("ts_open", as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("notional", "sum"),
        large_buy_vol=("large_buy", "sum"),
        large_sell_vol=("large_sell", "sum"),
        trade_count=("notional", "count"),
        buy_count=("buy_cnt", "sum"),
        sell_count=("sell_cnt", "sum"),
    )


def _process_file_okx(fp: Path) -> pd.DataFrame:
    """Read OKX trade CSV and return aggregated 1m bars (not upserted yet)."""
    chunks = []
    try:
        for chunk in pd.read_csv(
            fp,
            chunksize=CHUNK_SIZE,
            dtype={"created_time": np.int64, "price": float, "size": float},
        ):
            chunks.append(_agg_chunk_okx(chunk))
    except Exception:
        logger.exception("Error reading OKX file %s", fp.name)
        return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    bars = combined.groupby("ts_open", as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("volume", "sum"),
        large_buy_vol=("large_buy_vol", "sum"),
        large_sell_vol=("large_sell_vol", "sum"),
        trade_count=("trade_count", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
    )
    bars["ts_open"] = bars["ts_open"].astype(np.int64)
    return bars


# ─── Shared bar helpers ───────────────────────────────────────────────────────

def _finalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Add delta / delta_ratio to an aggregated bars DataFrame."""
    if bars.empty:
        return bars

    bars = bars.copy()
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]
    bars["delta_ratio"] = np.where(
        bars["volume"] > 0, bars["delta"] / bars["volume"], 0.0
    )
    bars["ts_open"] = bars["ts_open"].astype(np.int64)
    return bars


def _aggregate_files(files: list[Path], process_fn) -> pd.DataFrame:
    """
    Aggregate many CSV files using a per-file processing function
    that returns 1m aggregated bars DataFrame.
    """
    parts = []
    for fp in files:
        bars = process_fn(fp)
        if not bars.empty:
            parts.append(bars)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("ts_open", as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("volume", "sum"),
        large_buy_vol=("large_buy_vol", "sum"),
        large_sell_vol=("large_sell_vol", "sum"),
        trade_count=("trade_count", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
    )


# ─── Local import (Binance + OKX pre-downloaded, stored separately) ──────────

def import_from_local(symbol: str = SYMBOL) -> int:
    """
    Read pre-downloaded aggtrade files from raw_data/aggtrades/{binance,okx}/.
    Stores Binance and OKX bars separately (exchange='binance' / 'okx')
    plus a combined row (exchange='all') in flow_bars_1m_ml.
    Returns total bar count across all written scopes.
    """
    binance_dir = LOCAL_AGGTRADES_DIR / "binance"
    okx_dir = LOCAL_AGGTRADES_DIR / "okx"

    binance_files = sorted(binance_dir.glob("**/BTCUSDT-aggTrades-*.csv"))
    okx_files = sorted(okx_dir.glob("**/BTC-USDT-SWAP-trades-*.csv"))

    logger.info("Local files — Binance: %d, OKX: %d", len(binance_files), len(okx_files))

    bnc_bars = _finalize_bars(_aggregate_files(binance_files, _process_binance_file_to_df))
    okx_bars = _finalize_bars(_aggregate_files(okx_files, _process_file_okx))

    if not bnc_bars.empty and not okx_bars.empty:
        merged = pd.concat([bnc_bars, okx_bars], ignore_index=True)
        all_bars = _finalize_bars(
            merged.groupby("ts_open", as_index=False).agg(
                buy_vol=("buy_vol", "sum"),
                sell_vol=("sell_vol", "sum"),
                volume=("volume", "sum"),
                large_buy_vol=("large_buy_vol", "sum"),
                large_sell_vol=("large_sell_vol", "sum"),
                trade_count=("trade_count", "sum"),
                buy_count=("buy_count", "sum"),
                sell_count=("sell_count", "sum"),
            )
        )
    elif not bnc_bars.empty:
        all_bars = bnc_bars.copy()
    else:
        all_bars = okx_bars.copy()

    total = 0
    for exchange, bars in [
        ("binance", bnc_bars),
        ("okx", okx_bars),
        ("all", all_bars),
    ]:
        if bars.empty:
            logger.warning("No bars for exchange=%s", exchange)
            continue

        _upsert(bars, symbol, exchange)
        logger.info("exchange=%-8s  %d bars", exchange, len(bars))
        total += len(bars)

    return total


# ─── DB upsert ────────────────────────────────────────────────────────────────

def _upsert(bars: pd.DataFrame, symbol: str, exchange: str = "all") -> None:
    sql = """
        INSERT INTO flow_bars_1m_ml
            (symbol, exchange, ts_open, buy_vol, sell_vol, delta, volume, delta_ratio,
             large_buy_vol, large_sell_vol, trade_count, buy_count, sell_count)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            buy_vol        = VALUES(buy_vol),
            sell_vol       = VALUES(sell_vol),
            delta          = VALUES(delta),
            volume         = VALUES(volume),
            delta_ratio    = VALUES(delta_ratio),
            large_buy_vol  = VALUES(large_buy_vol),
            large_sell_vol = VALUES(large_sell_vol),
            trade_count    = VALUES(trade_count),
            buy_count      = VALUES(buy_count),
            sell_count     = VALUES(sell_count)
    """

    def _f(x):
        return None if pd.isna(x) else float(x)

    params = [
        (
            symbol,
            exchange,
            np.int64(r.ts_open),
            _f(r.buy_vol),
            _f(r.sell_vol),
            _f(r.delta),
            _f(r.volume),
            _f(r.delta_ratio),
            _f(r.large_buy_vol),
            _f(r.large_sell_vol),
            int(r.trade_count),
            int(r.buy_count),
            int(r.sell_count),
        )
        for r in bars.itertuples(index=False)
    ]

    for i in range(0, len(params), BATCH):
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, params[i : i + BATCH])
            conn.commit()
        except Exception:
            logger.exception("Upsert failed at row %d (exchange=%s)", i, exchange)
            raise
        finally:
            conn.close()


# ─── Month range helpers ──────────────────────────────────────────────────────

def _month_range(start: str = "2025-01", end: str = "2026-02") -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    months = []

    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    return months


# ─── Download-mode importers (Binance USDT-M only) ───────────────────────────

def import_monthly(
    months: list[str] | None = None,
    symbol: str = SYMBOL,
    keep_files: bool = False,
) -> int:
    months = months or _month_range()
    total = 0

    for ym in months:
        fname = f"{BINANCE_SYMBOL}-aggTrades-{ym}.csv"
        dest = TMP_DIR / fname
        url = f"{BASE_URL}/monthly/aggTrades/{BINANCE_SYMBOL}/{BINANCE_SYMBOL}-aggTrades-{ym}.zip"

        logger.info("=== %s ===", ym)
        if not _download_zip(url, dest):
            logger.warning("Skipped %s (download failed)", ym)
            continue

        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info("File: %.0f MB — aggregating...", size_mb)

        n = _process_binance_file_upsert(dest, symbol)
        total += n
        logger.info("→ %d 1m bars upserted", n)

        if not keep_files:
            dest.unlink(missing_ok=True)
            logger.info("Deleted %s", fname)

        time.sleep(0.5)

    return total


def import_daily(
    start: date | None = None,
    end: date | None = None,
    symbol: str = SYMBOL,
    keep_files: bool = False,
) -> int:
    start = start or date(2026, 3, 1)
    end = end or (date.today() - timedelta(days=1))
    total = 0
    d = start

    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        fname = f"{BINANCE_SYMBOL}-aggTrades-{ds}.csv"
        dest = TMP_DIR / fname
        url = f"{BASE_URL}/daily/aggTrades/{BINANCE_SYMBOL}/{BINANCE_SYMBOL}-aggTrades-{ds}.zip"

        if not _download_zip(url, dest):
            d += timedelta(days=1)
            time.sleep(0.2)
            continue

        n = _process_binance_file_upsert(dest, symbol)
        total += n
        logger.info("%s → %d bars", ds, n)

        if not keep_files:
            dest.unlink(missing_ok=True)

        d += timedelta(days=1)
        time.sleep(0.3)

    return total


# ─── Entry points ─────────────────────────────────────────────────────────────

def import_aggtrades(symbol: str = SYMBOL) -> int:
    """
    Entry point called by run_pipeline_v2.
    Uses local pre-downloaded files if available, otherwise falls back to download.
    """
    binance_local = LOCAL_AGGTRADES_DIR / "binance"
    okx_local = LOCAL_AGGTRADES_DIR / "okx"

    has_local = (
        (binance_local.exists() and any(binance_local.rglob("*.csv")))
        or (okx_local.exists() and any(okx_local.rglob("*.csv")))
    )

    if has_local:
        logger.info("Local aggTrade files found — importing from disk (Binance + OKX)")
        return import_from_local(symbol)

    logger.info("No local files found — downloading from Binance Vision")
    return run()


def run(
    months: list[str] | None = None,
    daily_start: str | None = None,
    keep_files: bool = False,
) -> int:
    """Full download import: monthly + daily. Use --download flag from CLI."""
    logger.info("Clearing flow_bars_1m_ml...")
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE flow_bars_1m_ml")
        conn.commit()
    finally:
        conn.close()

    logger.info("=== Monthly import ===")
    n_monthly = import_monthly(months=months, keep_files=keep_files)
    logger.info("Monthly total: %d bars", n_monthly)

    logger.info("=== Daily import ===")
    ds = date.fromisoformat(daily_start) if daily_start else date(2026, 3, 1)
    n_daily = import_daily(start=ds, keep_files=keep_files)
    logger.info("Daily total: %d bars", n_daily)

    total = n_monthly + n_daily
    logger.info("flow_bars_1m_ml grand total: %d bars", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--download",
        action="store_true",
        help="Force download from Binance Vision instead of reading local files",
    )
    ap.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="e.g. 2025-01 2026-02 (start end) — only used with --download",
    )
    ap.add_argument(
        "--daily-start",
        default=None,
        help="e.g. 2026-03-01 — only used with --download",
    )
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()

    if args.download:
        months = None
        if args.months and len(args.months) == 2:
            months = _month_range(args.months[0], args.months[1])
        elif args.months:
            months = args.months

        run(months=months, daily_start=args.daily_start, keep_files=args.keep_files)
    else:
        import_from_local()