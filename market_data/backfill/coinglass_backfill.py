"""
Coinglass API v4 data backfill — downloads historical data for model training.

Endpoints (requires paid plan):
  1. OI History (per exchange)    — OHLC OI, 30m+, ~83 days at 30m
  2. OI Aggregated History        — all-exchange OI, 30m+
  3. Liquidation History          — long/short liq USD, 30m+
  4. Long/Short Ratio             — global account L/S ratio, 30m+
  5. Funding Rate History         — OHLC funding, 30m+
  6. Taker Buy/Sell Volume        — taker buy/sell USD, 30m+

API v4 notes:
  - Base URL: https://open-api-v4.coinglass.com/api
  - Max limit per request: 4000
  - No pagination via startTime (API returns most recent N rows)
  - Symbol format: Binance=BTCUSDT, OKX=BTC-USDT-SWAP
  - Per-exchange endpoints require 'exchange' param

Output: market_data/raw_data/coinglass/{endpoint}_{symbol}.parquet

Usage:
    python -m market_data.backfill.coinglass_backfill
    python -m market_data.backfill.coinglass_backfill --endpoint oi
    python -m market_data.backfill.coinglass_backfill --interval 1h
    python -m market_data.backfill.coinglass_backfill --dry-run
"""
from __future__ import annotations

import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "market_data" / "raw_data" / "coinglass"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://open-api-v4.coinglass.com/api"
API_KEY  = os.environ.get("COINGLASS_API_KEY", "")

MAX_LIMIT  = 4000
RATE_LIMIT = 0.8  # seconds between requests (safe for 80/min)

# Exchange-specific symbol mapping
EXCHANGE_SYMBOLS = {
    "Binance": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "OKX":     {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP"},
    "Bybit":   {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
}


def _get_api_key() -> str:
    key = API_KEY
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("COINGLASS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise ValueError("COINGLASS_API_KEY not set. Set env var or add to .env")
    return key


def _headers() -> dict:
    return {"CG-API-KEY": _get_api_key(), "accept": "application/json"}


def _fetch(path: str, params: dict) -> list[dict] | None:
    """Fetch from Coinglass v4 API. Returns list of rows or None."""
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, headers=_headers(), timeout=30)
    resp.raise_for_status()
    body = resp.json()

    code = body.get("code")
    if code not in ("0", 0):
        logger.error("API error: code=%s msg=%s path=%s", code, body.get("msg"), path)
        return None

    data = body.get("data")
    if isinstance(data, list):
        return data
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint-specific backfillers
# ═════════════════════════════════════════════════════════════════════════════

def backfill_oi(symbol: str = "BTC", interval: str = "1h",
                exchange: str = "Binance") -> pd.DataFrame:
    """Open Interest OHLC History (per exchange)."""
    ex_sym = EXCHANGE_SYMBOLS.get(exchange, {}).get(symbol, f"{symbol}USDT")
    logger.info("=== OI History: %s (%s) %s %s ===", symbol, ex_sym, exchange, interval)

    rows = _fetch("/futures/open-interest/history", {
        "symbol": ex_sym, "exchange": exchange,
        "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = exchange
    df["symbol"] = symbol
    logger.info("OI: %d rows fetched", len(df))
    return df


def backfill_oi_aggregated(symbol: str = "BTC", interval: str = "1h",
                            **_kwargs) -> pd.DataFrame:
    """OI Aggregated History (all exchanges combined)."""
    logger.info("=== OI Aggregated: %s %s ===", symbol, interval)

    rows = _fetch("/futures/open-interest/aggregated-history", {
        "symbol": symbol, "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    logger.info("OI Aggregated: %d rows", len(df))
    return df


def backfill_liquidation(symbol: str = "BTC", interval: str = "1h",
                          exchange: str = "Binance") -> pd.DataFrame:
    """Liquidation History (long/short USD)."""
    ex_sym = EXCHANGE_SYMBOLS.get(exchange, {}).get(symbol, f"{symbol}USDT")
    logger.info("=== Liquidation: %s (%s) %s %s ===", symbol, ex_sym, exchange, interval)

    rows = _fetch("/futures/liquidation/history", {
        "symbol": ex_sym, "exchange": exchange,
        "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = exchange
    df["symbol"] = symbol
    logger.info("Liquidation: %d rows", len(df))
    return df


def backfill_long_short(symbol: str = "BTC", interval: str = "1h",
                         exchange: str = "Binance") -> pd.DataFrame:
    """Global Long/Short Account Ratio History."""
    ex_sym = EXCHANGE_SYMBOLS.get(exchange, {}).get(symbol, f"{symbol}USDT")
    logger.info("=== Long/Short Ratio: %s (%s) %s %s ===", symbol, ex_sym, exchange, interval)

    rows = _fetch("/futures/global-long-short-account-ratio/history", {
        "symbol": ex_sym, "exchange": exchange,
        "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = exchange
    df["symbol"] = symbol
    logger.info("Long/Short: %d rows", len(df))
    return df


def backfill_funding(symbol: str = "BTC", interval: str = "1h",
                      exchange: str = "Binance") -> pd.DataFrame:
    """Funding Rate OHLC History."""
    ex_sym = EXCHANGE_SYMBOLS.get(exchange, {}).get(symbol, f"{symbol}USDT")
    logger.info("=== Funding Rate: %s (%s) %s %s ===", symbol, ex_sym, exchange, interval)

    rows = _fetch("/futures/funding-rate/history", {
        "symbol": ex_sym, "exchange": exchange,
        "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = exchange
    df["symbol"] = symbol
    logger.info("Funding: %d rows", len(df))
    return df


def backfill_taker(symbol: str = "BTC", interval: str = "1h",
                    exchange: str = "Binance") -> pd.DataFrame:
    """Taker Buy/Sell Volume History."""
    ex_sym = EXCHANGE_SYMBOLS.get(exchange, {}).get(symbol, f"{symbol}USDT")
    logger.info("=== Taker Buy/Sell: %s (%s) %s %s ===", symbol, ex_sym, exchange, interval)

    rows = _fetch("/futures/taker-buy-sell-volume/history", {
        "symbol": ex_sym, "exchange": exchange,
        "interval": interval, "limit": MAX_LIMIT,
    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["exchange"] = exchange
    df["symbol"] = symbol
    logger.info("Taker: %d rows", len(df))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

ENDPOINT_MAP = {
    "oi":           backfill_oi,
    "oi_agg":       backfill_oi_aggregated,
    "liquidation":  backfill_liquidation,
    "long_short":   backfill_long_short,
    "funding":      backfill_funding,
    "taker":        backfill_taker,
}


def run(endpoints: list[str] | None = None, symbol: str = "BTC",
        interval: str = "1h", exchange: str = "Binance",
        dry_run: bool = False):
    if endpoints is None:
        endpoints = list(ENDPOINT_MAP.keys())

    for ep_name in endpoints:
        fn = ENDPOINT_MAP.get(ep_name)
        if fn is None:
            logger.warning("Unknown endpoint: %s", ep_name)
            continue

        df = fn(symbol=symbol, interval=interval, exchange=exchange)
        if df.empty:
            continue

        # Convert numeric columns from string
        for col in df.columns:
            if col in ("exchange", "symbol", "dt"):
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse timestamps
        if "time" in df.columns:
            df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df = df.sort_values("time").reset_index(drop=True)

        if dry_run:
            logger.info("DRY RUN -- %s: %d rows, columns: %s",
                         ep_name, len(df), list(df.columns))
            print(df.head(3).to_string())
            print(f"...\n{df.tail(1).to_string()}")
            span = ""
            if "dt" in df.columns:
                span = f"  range: {df['dt'].min()} -> {df['dt'].max()}"
            print(f"Total: {len(df)} rows{span}\n")
            continue

        out = OUT_DIR / f"{ep_name}_{symbol}_{exchange}_{interval}.parquet"
        df.to_parquet(out, index=False)
        logger.info("Saved: %s (%d rows)", out, len(df))

    # Rate limit info
    logger.info("Done. Files saved to %s", OUT_DIR)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", nargs="*", default=None,
                    choices=list(ENDPOINT_MAP.keys()),
                    help="Specific endpoints to backfill")
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--interval", default="1h",
                    choices=["30m", "1h", "2h", "4h", "6h", "12h", "1d"])
    ap.add_argument("--exchange", default="Binance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run(endpoints=args.endpoint, symbol=args.symbol,
        interval=args.interval, exchange=args.exchange,
        dry_run=args.dry_run)


if __name__ == "__main__":
    main()
