"""
Backfill historical Open Interest from Binance (30-day limit).

Source: GET https://fapi.binance.com/futures/data/openInterestHist
  - BTC/ETH USDT-M perpetual
  - Period: 15m (matches model timeframe)
  - 500 records per request, max 30 days back

Usage:
    python -m market_data.backfill.oi_backfill
    python -m market_data.backfill.oi_backfill --days 30
"""
import sys
import time
import logging
import argparse
import requests
from datetime import datetime, timezone, timedelta

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__file__))))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

BASE_URL   = "https://fapi.binance.com/futures/data/openInterestHist"
PERIOD     = "15m"
LIMIT      = 500
BATCH_SIZE = 500
SLEEP_S    = 0.3

TARGETS = [
    ("BTCUSDT", "BTC-USD"),
    ("ETHUSDT", "ETH-USD"),
]


def _fetch_page(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    resp = requests.get(BASE_URL, params={
        "symbol":    symbol,
        "period":    PERIOD,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     LIMIT,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _save_batch(rows: list[tuple]):
    if not rows:
        return
    sql = """
        INSERT INTO oi_snapshots
            (exchange, canonical_symbol, oi_contracts, oi_notional_usd, ts_exchange, ts_received)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            oi_contracts     = VALUES(oi_contracts),
            oi_notional_usd  = VALUES(oi_notional_usd)
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
    finally:
        conn.close()


def backfill_symbol(binance_sym: str, canonical: str, start_ms: int, end_ms: int):
    logger.info("Backfilling OI: %s  %s ~ %s",
                binance_sym,
                datetime.utcfromtimestamp(start_ms / 1000).strftime('%Y-%m-%d'),
                datetime.utcfromtimestamp(end_ms   / 1000).strftime('%Y-%m-%d'))

    total   = 0
    cursor  = start_ms
    batch   = []
    ts_recv = int(time.time() * 1000)

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
                float(item["sumOpenInterest"]),       # in base asset (BTC/ETH)
                float(item["sumOpenInterestValue"]),   # in USD
                int(item["timestamp"]),
                ts_recv,
            ))

        if len(batch) >= BATCH_SIZE:
            _save_batch(batch)
            total += len(batch)
            batch = []
            logger.info("  %s  saved %d rows  cursor=%s",
                        binance_sym, total,
                        datetime.utcfromtimestamp(cursor / 1000).strftime('%Y-%m-%d %H:%M'))

        last_ts = int(page[-1]["timestamp"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(SLEEP_S)

    if batch:
        _save_batch(batch)
        total += len(batch)

    logger.info("Done %s: %d OI rows inserted/updated", binance_sym, total)
    return total


def run(days_back: int = 30):
    now_ms   = int(time.time() * 1000)
    # Binance hard limit: 30 days
    days_back = min(days_back, 30)
    start_ms = now_ms - days_back * 86_400_000

    logger.info("OI backfill: last %d days", days_back)
    grand_total = 0
    for binance_sym, canonical in TARGETS:
        grand_total += backfill_symbol(binance_sym, canonical, start_ms, now_ms)
    logger.info("OI backfill complete. Total rows: %d", grand_total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="Days to backfill (max 30)")
    args = ap.parse_args()
    run(args.days)
