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
                   "taker_buy_vol", "taker_buy_quote", "quote_vol"]:
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


def fetch_binance_depth(symbol: str = "BTCUSDT", limit: int = 20) -> dict:
    """
    Fetch order book depth snapshot from Binance Futures.

    Returns dict with 'bids' and 'asks' as lists of [price, qty],
    plus computed summary: bid_depth_usd, ask_depth_usd, imbalance.
    Returns empty dict on failure.
    """
    url = "https://fapi.binance.com/fapi/v1/depth"
    try:
        resp = _retry_request("GET", url, params={
            "symbol": symbol, "limit": limit,
        })
        data = resp.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            logger.warning("Binance depth: empty order book")
            return {}

        # Compute USD depth (price * qty for each level)
        bid_depth = sum(float(p) * float(q) for p, q in bids)
        ask_depth = sum(float(p) * float(q) for p, q in asks)
        total = bid_depth + ask_depth

        # Near-price depth (top 5 levels)
        near_bid = sum(float(p) * float(q) for p, q in bids[:5])
        near_ask = sum(float(p) * float(q) for p, q in asks[:5])

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2

        result = {
            "bid_depth_usd": bid_depth,
            "ask_depth_usd": ask_depth,
            "depth_imbalance": (bid_depth - ask_depth) / total if total > 0 else 0,
            "near_bid_usd": near_bid,
            "near_ask_usd": near_ask,
            "near_imbalance": (near_bid - near_ask) / (near_bid + near_ask)
                              if (near_bid + near_ask) > 0 else 0,
            "spread_bps": (best_ask - best_bid) / mid * 10000 if mid > 0 else 0,
            "mid_price": mid,
        }
        logger.info("Binance depth: bid=%.0f ask=%.0f imb=%.3f spread=%.1fbps",
                     bid_depth, ask_depth, result["depth_imbalance"], result["spread_bps"])
        return result

    except Exception as e:
        logger.error("Binance depth fetch failed: %s", e)
        return {}


def fetch_binance_aggtrades(symbol: str = "BTCUSDT",
                            lookback_hours: int = 2,
                            large_threshold_usd: float = 100_000) -> dict:
    """
    Fetch recent aggTrades and separate large vs small trades.

    Returns dict with per-hour aggregated stats:
        large_buy_usd, large_sell_usd, large_count, large_ratio,
        small_buy_usd, small_sell_usd, avg_trade_usd, total_count.
    Returns empty dict on failure.

    Note: Binance aggTrades limit=1000, so for high-activity periods
    this covers ~minutes. We make multiple requests to cover lookback.
    """
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_hours * 3600 * 1000

    all_trades = []
    current_end = now_ms

    try:
        # Paginate backwards (up to 5 requests to stay within rate limits)
        for _ in range(5):
            resp = _retry_request("GET", url, params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": current_end,
                "limit": 1000,
            })
            trades = resp.json()
            if not isinstance(trades, list) or len(trades) == 0:
                break

            all_trades.extend(trades)
            # Move end time before earliest trade in this batch
            earliest_ts = min(t["T"] for t in trades)
            if earliest_ts <= start_ms:
                break
            current_end = earliest_ts - 1
            time.sleep(0.2)  # Rate limit

        if not all_trades:
            logger.warning("Binance aggTrades: no trades fetched")
            return {}

        # Parse trades
        large_buy_usd = 0.0
        large_sell_usd = 0.0
        large_count = 0
        small_buy_usd = 0.0
        small_sell_usd = 0.0
        total_count = len(all_trades)

        for t in all_trades:
            price = float(t["p"])
            qty = float(t["q"])
            notional = price * qty
            is_seller_maker = t["m"]  # True = taker is buyer

            if notional >= large_threshold_usd:
                large_count += 1
                if not is_seller_maker:  # seller is taker
                    large_sell_usd += notional
                else:
                    large_buy_usd += notional
            else:
                if not is_seller_maker:
                    small_sell_usd += notional
                else:
                    small_buy_usd += notional

        total_usd = large_buy_usd + large_sell_usd + small_buy_usd + small_sell_usd
        result = {
            "large_buy_usd": large_buy_usd,
            "large_sell_usd": large_sell_usd,
            "large_delta_usd": large_buy_usd - large_sell_usd,
            "large_count": large_count,
            "large_ratio": (large_buy_usd + large_sell_usd) / total_usd if total_usd > 0 else 0,
            "large_buy_ratio": large_buy_usd / (large_buy_usd + large_sell_usd)
                               if (large_buy_usd + large_sell_usd) > 0 else 0.5,
            "small_buy_usd": small_buy_usd,
            "small_sell_usd": small_sell_usd,
            "small_delta_usd": small_buy_usd - small_sell_usd,
            "avg_trade_usd": total_usd / total_count if total_count > 0 else 0,
            "total_count": total_count,
            "total_usd": total_usd,
        }
        logger.info("aggTrades: %d trades, large=%d (%.1f%%), large_delta=$%.0f",
                     total_count, large_count,
                     result["large_ratio"] * 100, result["large_delta_usd"])
        return result

    except Exception as e:
        logger.error("Binance aggTrades fetch failed: %s", e)
        return {}


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
