"""
Live data fetcher — Binance klines + Coinglass API.

v2: Retry with exponential backoff, response validation,
    fallback to last-known-good data on failure.
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
CG_BASE = "https://open-api-v4.coinglass.com/api"
CG_API_KEY = os.environ.get("COINGLASS_API_KEY", "")

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds, doubles each attempt

# Cache dir for fallback
CACHE_DIR = Path(__file__).parent / "model_artifacts" / ".data_cache"

CG_ENDPOINTS = {
    "oi":          ("/futures/open-interest/history", "Binance", "BTCUSDT"),
    "oi_agg":      ("/futures/open-interest/aggregated-history", None, "BTC"),
    "liquidation": ("/futures/liquidation/history", "Binance", "BTCUSDT"),
    "long_short":  ("/futures/top-long-short-account-ratio/history", "Binance", "BTCUSDT"),
    "global_ls":   ("/futures/global-long-short-account-ratio/history", "Binance", "BTCUSDT"),
    "funding":     ("/futures/funding-rate/history", "Binance", "BTCUSDT"),
    "taker":       ("/futures/taker-buy-sell-volume/history", "Binance", "BTCUSDT"),
}


def _retry_request(method: str, url: str, **kwargs) -> requests.Response:
    """Execute HTTP request with exponential backoff retry."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.Timeout) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Request failed (attempt %d/%d), retry in %ds: %s",
                               attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)
    raise last_err


def _save_cache(name: str, df: pd.DataFrame):
    """Cache successful fetch for fallback."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    try:
        df.to_parquet(path)
    except Exception:
        pass


def _load_cache(name: str) -> pd.DataFrame:
    """Load cached data as fallback."""
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
            logger.warning("Using cached data for %s (%d rows)", name, len(df))
            return df
        except Exception:
            pass
    return pd.DataFrame()


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                         limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance Futures REST API with retry."""
    try:
        resp = _retry_request("GET", BINANCE_KLINES_URL, params={
            "symbol": symbol, "interval": interval, "limit": limit,
        })
        rows = resp.json()

        if not isinstance(rows, list) or len(rows) == 0:
            logger.error("Binance klines: unexpected response format")
            return _load_cache("binance_klines")

        df = pd.DataFrame(rows, columns=[
            "ts_open", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trade_count", "taker_buy_vol",
            "taker_buy_quote", "ignore",
        ])
        for c in ["open", "high", "low", "close", "volume",
                   "taker_buy_vol", "taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
        df["ts_open"] = pd.to_numeric(df["ts_open"])
        df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
        df = df.set_index("dt").sort_index()

        # Drop the current incomplete bar
        df = df.iloc[:-1]

        # Validate: prices should be positive
        if (df["close"] <= 0).any():
            logger.error("Binance klines: invalid price data detected")
            return _load_cache("binance_klines")

        _save_cache("binance_klines", df)
        logger.info("Binance klines: %d bars, %s ~ %s",
                     len(df), df.index[0], df.index[-1])
        return df

    except Exception as e:
        logger.error("Binance klines failed after retries: %s", e)
        return _load_cache("binance_klines")


def _cg_fetch(path: str, exchange: str | None, symbol: str,
              interval: str = "30m", limit: int = 500) -> pd.DataFrame:
    """Fetch one Coinglass endpoint with retry."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if exchange:
        params["exchange"] = exchange

    headers = {"CG-API-KEY": CG_API_KEY}
    resp = _retry_request("GET", CG_BASE + path, params=params, headers=headers)
    data = resp.json()

    if data.get("code") != "0":
        logger.warning("CG %s: code=%s msg=%s", path, data.get("code"), data.get("msg"))
        return pd.DataFrame()

    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) == 0:
        logger.warning("CG %s: empty or invalid data", path)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Numeric conversion
    for col in df.columns:
        if col not in ("time", "t", "createTime"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp
    time_col = next((c for c in ["t", "time", "createTime"] if c in df.columns), None)
    if time_col:
        ts = df[time_col].astype(float)
        if ts.max() > 1e12:
            df["dt"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["dt"] = pd.to_datetime(ts, unit="s", utc=True)
        df = df.set_index("dt").sort_index()

    return df


def fetch_coinglass(interval: str = "1h", limit: int = 500) -> dict[str, pd.DataFrame]:
    """Fetch all Coinglass endpoints with retry and cache fallback."""
    if not CG_API_KEY:
        logger.error("COINGLASS_API_KEY is not set — all CG data will be empty!")

    result = {}
    failed = []

    for name, (path, exchange, symbol) in CG_ENDPOINTS.items():
        try:
            df = _cg_fetch(path, exchange, symbol, interval, limit)
            if df.empty:
                df = _load_cache(f"cg_{name}")
                if not df.empty:
                    failed.append(name)
            else:
                _save_cache(f"cg_{name}", df)
            result[name] = df
            logger.info("CG %s: %d rows", name, len(df))
        except Exception as e:
            logger.error("CG %s failed: %s", name, e)
            result[name] = _load_cache(f"cg_{name}")
            failed.append(name)
        time.sleep(1)  # Rate limit courtesy

    if failed:
        logger.warning("Coinglass endpoints using cached data: %s", failed)

    return result
