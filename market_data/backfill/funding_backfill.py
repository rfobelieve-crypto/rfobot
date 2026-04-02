"""
Backfill historical funding rates.

Sources:
  --from-local (default): read OKX CSV files from raw_data/funding rate/
                Format: instrument_name, funding_rate, funding_time (ms)
                Saved with exchange='okx', canonical_symbol='BTC-USD'

  --download:  fetch from Binance FAPI (every 8h, USDT-M perpetual)

Usage:
    python -m market_data.backfill.funding_backfill
    python -m market_data.backfill.funding_backfill --download --start 2024-01-01
"""
import sys
import glob
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__file__))))
from shared.db import get_db_conn

ROOT             = Path(__file__).resolve().parents[2]
LOCAL_FUNDING_DIR = ROOT / "market_data" / "raw_data" / "funding rate"

logger = logging.getLogger(__name__)

BASE_URL   = "https://fapi.binance.com/fapi/v1/fundingRate"
LIMIT      = 1000
BATCH_SIZE = 500
SLEEP_S    = 0.3   # rate limit buffer

TARGETS = [
    ("BTCUSDT", "BTC-USD"),
    ("ETHUSDT", "ETH-USD"),
]

DEFAULT_START_MS = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_page(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    resp = requests.get(BASE_URL, params={
        "symbol":    symbol,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     LIMIT,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _save_batch(rows: list[dict]):
    if not rows:
        return
    sql = """
        INSERT INTO funding_rates
            (exchange, canonical_symbol, funding_rate, next_funding_ts, ts_exchange, ts_received)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE funding_rate = VALUES(funding_rate)
    """
    ts_recv = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
    finally:
        conn.close()


def backfill_symbol(binance_sym: str, canonical: str, start_ms: int, end_ms: int):
    logger.info("Backfilling funding: %s  %s ~ %s",
                binance_sym,
                datetime.utcfromtimestamp(start_ms / 1000).strftime('%Y-%m-%d'),
                datetime.utcfromtimestamp(end_ms   / 1000).strftime('%Y-%m-%d'))

    total = 0
    cursor = start_ms
    batch  = []

    while cursor < end_ms:
        try:
            page = _fetch_page(binance_sym, cursor, end_ms)
        except Exception:
            logger.exception("Fetch failed for %s at %d, retrying in 5s", binance_sym, cursor)
            time.sleep(5)
            continue

        if not page:
            break

        for item in page:
            batch.append((
                "binance",
                canonical,
                float(item["fundingRate"]),
                int(item.get("fundingTime", 0)) + 8 * 3_600_000,  # next = current + 8h
                int(item["fundingTime"]),
                int(time.time() * 1000),
            ))

        # flush batch
        if len(batch) >= BATCH_SIZE:
            _save_batch(batch)
            total += len(batch)
            batch = []
            logger.info("  %s  saved %d rows  cursor=%s",
                        binance_sym, total,
                        datetime.utcfromtimestamp(cursor / 1000).strftime('%Y-%m-%d %H:%M'))

        last_ts = int(page[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(SLEEP_S)

    if batch:
        _save_batch(batch)
        total += len(batch)

    logger.info("Done %s: %d funding rate rows inserted/updated", binance_sym, total)
    return total


def import_funding_local(symbol: str = "BTC-USD") -> int:
    """
    Read OKX funding rate CSV files from raw_data/funding rate/ and upsert to DB.
    Format: instrument_name, funding_rate, funding_time (ms, every 8h)
    """
    files = sorted(LOCAL_FUNDING_DIR.glob("BTC-USDT-SWAP-fundingrates-*.csv"))
    if not files:
        logger.warning("No funding rate CSV found in %s", LOCAL_FUNDING_DIR)
        return 0

    logger.info("Found %d funding rate files", len(files))
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        frames.append(df[["funding_rate", "funding_time"]])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates("funding_time").sort_values("funding_time")
    logger.info("Total funding rows after dedup: %d", len(all_df))

    sql = """
        INSERT INTO funding_rates
            (exchange, canonical_symbol, funding_rate, next_funding_ts, ts_exchange, ts_received)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE funding_rate = VALUES(funding_rate)
    """
    ts_recv = int(time.time() * 1000)
    params = [
        ("okx", symbol,
         float(r.funding_rate),
         int(r.funding_time) + 8 * 3_600_000,  # next = current + 8h
         int(r.funding_time),
         ts_recv)
        for r in all_df.itertuples(index=False)
    ]

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params)
        conn.commit()
    finally:
        conn.close()

    logger.info("Inserted/updated %d funding rate rows (exchange=okx)", len(params))
    return len(params)


def run(start_ms: int | None = None):
    now_ms = int(time.time() * 1000)
    start  = start_ms or DEFAULT_START_MS
    grand_total = 0
    for binance_sym, canonical in TARGETS:
        grand_total += backfill_symbol(binance_sym, canonical, start, now_ms)
    logger.info("Funding backfill complete. Total rows: %d", grand_total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="Download from Binance FAPI instead of reading local files")
    ap.add_argument("--start", default="2023-01-01", help="Start date (only used with --download)")
    args = ap.parse_args()

    if args.download:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        run(int(start_dt.timestamp() * 1000))
    else:
        import_funding_local()
