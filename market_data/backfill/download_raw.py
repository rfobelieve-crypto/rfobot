"""
Download raw market data from Binance data portal and REST API.

Downloads:
  1. BTCUSDT 1m klines (USDT-M futures) — monthly 2025-01~2026-02, daily 2026-03
  2. BTCUSD_PERP aggTrades (coin-M) — daily 2026-03
  3. OI snapshots — Binance USDT-M API (last 30 days, 15m)
  4. Funding rates — Binance USDT-M API (2023-01-01 ~)

Usage:
    python -m market_data.backfill.download_raw
    python -m market_data.backfill.download_raw --skip-klines
    python -m market_data.backfill.download_raw --skip-agg
"""
from __future__ import annotations

import os
import sys
import io
import time
import zipfile
import logging
import argparse
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[2]
KLINES_DIR   = ROOT / "market_data" / "raw_data" / "klines"
AGG_DIR      = ROOT / "market_data" / "raw_data" / "aggtrades"
EXISTING_AGG = ROOT / "market_data" / "aggtrades_data"

KLINES_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://data.binance.vision/data/futures"


# ─── Download helpers ─────────────────────────────────────────────────────────

def _download_zip(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a zip file and extract the CSV inside to dest."""
    if dest.exists():
        logger.info("  skip (exists): %s", dest.name)
        return True
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 404:
                logger.warning("  404 not found: %s", url)
                return False
            resp.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                logger.warning("  no CSV in zip: %s", url)
                return False
            zf.extract(csv_names[0], dest.parent)
            extracted = dest.parent / csv_names[0]
            if extracted != dest:
                extracted.rename(dest)
            logger.info("  downloaded: %s", dest.name)
            return True
        except Exception as e:
            logger.warning("  attempt %d failed: %s — %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)
    return False


# ─── 1m Klines (USDT-M) ───────────────────────────────────────────────────────

def download_klines_monthly(symbol: str = "BTCUSDT",
                             months: list[str] | None = None):
    """Download monthly 1m klines. months = ['2025-01', '2026-02', ...]"""
    if months is None:
        # 2025-01 through 2026-02
        months = []
        y, m = 2025, 1
        while (y, m) <= (2026, 2):
            months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m > 12:
                m = 1; y += 1

    logger.info("=== Klines monthly: %d files ===", len(months))
    ok = fail = 0
    for ym in months:
        fname = f"{symbol}-1m-{ym}.csv"
        dest  = KLINES_DIR / fname
        url   = f"{BASE}/um/monthly/klines/{symbol}/1m/{symbol}-1m-{ym}.zip"
        if _download_zip(url, dest):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)
    logger.info("Klines monthly done: %d ok / %d failed", ok, fail)


def download_klines_daily(symbol: str = "BTCUSDT",
                           start: date | None = None,
                           end: date | None = None):
    """Download daily 1m klines for a date range."""
    start = start or date(2026, 3, 1)
    end   = end   or date.today() - timedelta(days=1)

    logger.info("=== Klines daily: %s ~ %s ===", start, end)
    ok = fail = 0
    d = start
    while d <= end:
        ds    = d.strftime("%Y-%m-%d")
        fname = f"{symbol}-1m-{ds}.csv"
        dest  = KLINES_DIR / fname
        url   = f"{BASE}/um/daily/klines/{symbol}/1m/{symbol}-1m-{ds}.zip"
        if _download_zip(url, dest):
            ok += 1
        else:
            fail += 1
        d += timedelta(days=1)
        time.sleep(0.2)
    logger.info("Klines daily done: %d ok / %d failed", ok, fail)


# ─── AggTrades (coin-M) ───────────────────────────────────────────────────────

def download_aggtrades_daily(symbol: str = "BTCUSD_PERP",
                              start: date | None = None,
                              end: date | None = None):
    """Download daily coin-M aggTrade files for a date range."""
    start = start or date(2026, 3, 1)
    end   = end   or date.today() - timedelta(days=1)

    logger.info("=== AggTrades daily: %s ~ %s ===", start, end)
    ok = fail = 0
    d = start
    while d <= end:
        ds    = d.strftime("%Y-%m-%d")
        fname = f"{symbol}-aggTrades-{ds}.csv"
        dest  = AGG_DIR / fname
        url   = f"{BASE}/cm/daily/aggTrades/{symbol}/{symbol}-aggTrades-{ds}.zip"
        if _download_zip(url, dest):
            ok += 1
        else:
            fail += 1
        d += timedelta(days=1)
        time.sleep(0.3)
    logger.info("AggTrades daily done: %d ok / %d failed", ok, fail)


# ─── OI backfill (USDT-M API) ────────────────────────────────────────────────

def backfill_oi(symbol: str = "BTCUSDT", canonical: str = "BTC-USD"):
    """Fetch OI history from Binance USDT-M futures API (max 30 days, 15m)."""
    sys.path.insert(0, str(ROOT))
    from shared.db import get_db_conn

    url    = "https://fapi.binance.com/futures/data/openInterestHist"
    now_ms = int(time.time() * 1000)
    start  = now_ms - 30 * 24 * 3600 * 1000
    limit  = 500
    total  = 0

    logger.info("=== OI backfill: %s ===", symbol)
    conn = get_db_conn()
    try:
        cursor = start
        while cursor < now_ms:
            try:
                resp = requests.get(url, params={
                    "symbol": symbol, "period": "15m",
                    "startTime": cursor, "limit": limit,
                }, timeout=15)
                resp.raise_for_status()
                page = resp.json()
            except Exception:
                logger.exception("OI fetch failed"); break
            if not page:
                break
            now_ms = int(time.time() * 1000)
            rows = []
            for rec in page:
                rows.append((
                    "binance", canonical,
                    int(rec["timestamp"]),
                    float(rec["sumOpenInterest"]),
                    float(rec["sumOpenInterestValue"]),
                    now_ms,
                ))
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO oi_snapshots
                        (exchange, canonical_symbol, ts_exchange,
                         oi_contracts, oi_notional_usd, ts_received)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        oi_contracts    = VALUES(oi_contracts),
                        oi_notional_usd = VALUES(oi_notional_usd)
                """, rows)
            conn.commit()
            total  += len(rows)
            cursor  = int(page[-1]["timestamp"]) + 15 * 60 * 1000
            time.sleep(0.2)
    finally:
        conn.close()
    logger.info("OI backfill done: %d rows for %s", total, symbol)


# ─── Funding backfill (USDT-M API) ───────────────────────────────────────────

def backfill_funding(symbol: str = "BTCUSDT", canonical: str = "BTC-USD",
                     start_ms: int | None = None):
    """Fetch funding rate history from Binance (2023-01-01 ~)."""
    sys.path.insert(0, str(ROOT))
    from shared.db import get_db_conn

    if start_ms is None:
        import datetime
        start_ms = int(datetime.datetime(2023, 1, 1).timestamp() * 1000)

    url    = "https://fapi.binance.com/fapi/v1/fundingRate"
    now_ms = int(time.time() * 1000)
    limit  = 1000
    total  = 0

    logger.info("=== Funding backfill: %s ===", symbol)
    conn = get_db_conn()
    try:
        cursor = start_ms
        while cursor < now_ms:
            try:
                resp = requests.get(url, params={
                    "symbol": symbol, "startTime": cursor,
                    "endTime": now_ms, "limit": limit,
                }, timeout=15)
                resp.raise_for_status()
                page = resp.json()
            except Exception:
                logger.exception("Funding fetch failed"); break
            if not page:
                break
            now_ms = int(time.time() * 1000)
            rows = []
            for rec in page:
                rows.append((
                    "binance", canonical,
                    int(rec["fundingTime"]),
                    float(rec["fundingRate"]),
                    now_ms,
                ))
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO funding_rates
                        (exchange, canonical_symbol, ts_exchange,
                         funding_rate, ts_received)
                    VALUES (%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        funding_rate = VALUES(funding_rate)
                """, rows)
            conn.commit()
            total  += len(rows)
            last    = int(page[-1]["fundingTime"])
            if last <= cursor:
                break
            cursor  = last + 1
            time.sleep(0.1)
    finally:
        conn.close()
    logger.info("Funding backfill done: %d rows for %s", total, symbol)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-klines", action="store_true")
    ap.add_argument("--skip-agg",    action="store_true")
    ap.add_argument("--skip-oi",     action="store_true")
    ap.add_argument("--skip-funding",action="store_true")
    args = ap.parse_args()

    if not args.skip_klines:
        download_klines_monthly()
        download_klines_daily()

    if not args.skip_agg:
        download_aggtrades_daily()

    if not args.skip_oi:
        backfill_oi("BTCUSDT", "BTC-USD")
        backfill_oi("ETHUSDT", "ETH-USD")

    if not args.skip_funding:
        backfill_funding("BTCUSDT", "BTC-USD")
        backfill_funding("ETHUSDT", "ETH-USD")

    logger.info("=== All downloads complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
