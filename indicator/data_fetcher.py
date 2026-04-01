"""
Live data fetcher — Binance klines + Coinglass API.
"""
from __future__ import annotations

import os
import time
import logging

import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
CG_BASE = "https://open-api-v4.coinglass.com/api"
CG_API_KEY = os.environ.get("COINGLASS_API_KEY", "")

CG_ENDPOINTS = {
    "oi":          ("/futures/open-interest/history", "Binance", "BTCUSDT"),
    "oi_agg":      ("/futures/open-interest/aggregated-history", None, "BTC"),
    "liquidation": ("/futures/liquidation/history", "Binance", "BTCUSDT"),
    "long_short":  ("/futures/top-long-short-account-ratio/history", "Binance", "BTCUSDT"),
    "funding":     ("/futures/funding-rate/history", "Binance", "BTCUSDT"),
    "taker":       ("/futures/taker-buy-sell-volume/history", "Binance", "BTCUSDT"),
}


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "15m",
                         limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance Futures REST API."""
    resp = requests.get(BINANCE_KLINES_URL, params={
        "symbol": symbol, "interval": interval, "limit": limit,
    }, timeout=30)
    resp.raise_for_status()
    rows = resp.json()

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

    logger.info("Binance klines: %d bars, %s ~ %s",
                len(df), df.index[0], df.index[-1])
    return df


def _cg_fetch(path: str, exchange: str | None, symbol: str,
              interval: str = "30m", limit: int = 500) -> pd.DataFrame:
    """Fetch one Coinglass endpoint."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if exchange:
        params["exchange"] = exchange

    headers = {"CG-API-KEY": CG_API_KEY}
    resp = requests.get(CG_BASE + path, params=params,
                        headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "0" or not data.get("data"):
        logger.warning("CG %s: code=%s msg=%s", path, data.get("code"), data.get("msg"))
        return pd.DataFrame()

    rows = data["data"]
    if isinstance(rows, list):
        df = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

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


def fetch_coinglass(interval: str = "30m", limit: int = 500) -> dict[str, pd.DataFrame]:
    """Fetch all Coinglass endpoints. Returns dict of DataFrames."""
    result = {}
    for name, (path, exchange, symbol) in CG_ENDPOINTS.items():
        try:
            df = _cg_fetch(path, exchange, symbol, interval, limit)
            result[name] = df
            logger.info("CG %s: %d rows", name, len(df))
        except Exception as e:
            logger.error("CG %s failed: %s", name, e)
            result[name] = pd.DataFrame()
        time.sleep(1)  # Rate limit courtesy
    return result
