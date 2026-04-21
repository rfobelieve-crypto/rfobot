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
    # --- Existing 7 endpoints ---
    "oi":          {"path": "/futures/open-interest/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "oi_agg":      {"path": "/futures/open-interest/aggregated-history", "symbol": "BTC"},
    "liquidation": {"path": "/futures/liquidation/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "long_short":  {"path": "/futures/top-long-short-account-ratio/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "global_ls":   {"path": "/futures/global-long-short-account-ratio/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "funding":     {"path": "/futures/funding-rate/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "taker":       {"path": "/futures/taker-buy-sell-volume/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    # --- New timeseries endpoints (Startup plan) ---
    "coinbase_premium": {"path": "/coinbase-premium-index"},
    "bitfinex_margin":  {"path": "/bitfinex-margin-long-short", "symbol": "BTC"},
    "top_ls_position":  {"path": "/futures/top-long-short-position-ratio/history", "exchange": "Binance", "symbol": "BTCUSDT"},
    "futures_cvd_agg":  {"path": "/futures/aggregated-cvd/history", "symbol": "BTC", "extra_params": {"exchange_list": "Binance"}},
    "spot_cvd_agg":     {"path": "/spot/aggregated-cvd/history", "symbol": "BTC", "extra_params": {"exchange_list": "Binance"}},
    "liq_agg":          {"path": "/futures/liquidation/aggregated-history", "symbol": "BTC", "extra_params": {"exchange_list": "Binance"}},
    "oi_coin_margin":   {"path": "/futures/open-interest/aggregated-coin-margin-history", "symbol": "BTC", "extra_params": {"exchange_list": "Binance"}},
}

# Deribit public API (no auth needed)
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


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


def _cg_fetch(path: str, exchange: str | None = None, symbol: str | None = None,
              interval: str = "30m", limit: int = 500,
              extra_params: dict | None = None) -> pd.DataFrame:
    """Fetch one Coinglass endpoint with retry."""
    params = {"interval": interval, "limit": limit}
    if symbol:
        params["symbol"] = symbol
    if exchange:
        params["exchange"] = exchange
    if extra_params:
        params.update(extra_params)

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

    for name, cfg in CG_ENDPOINTS.items():
        try:
            df = _cg_fetch(
                path=cfg["path"],
                exchange=cfg.get("exchange"),
                symbol=cfg.get("symbol"),
                interval=interval,
                limit=limit,
                extra_params=cfg.get("extra_params"),
            )
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


# ── Coinglass Options + ETF endpoints ────────────────────────────────────

def _cg_fetch_raw(path: str, params: dict | None = None) -> dict:
    """Fetch raw Coinglass API response (for non-timeseries endpoints)."""
    headers = {"CG-API-KEY": CG_API_KEY}
    resp = _retry_request("GET", CG_BASE + path, params=params or {}, headers=headers)
    data = resp.json()
    if data.get("code") != "0":
        logger.warning("CG %s: code=%s msg=%s", path, data.get("code"), data.get("msg"))
        return {}
    return data.get("data", {})


def fetch_cg_options() -> dict:
    """
    Fetch Coinglass options data: max pain, options OI, options/futures ratio.

    Returns dict with:
        max_pain_price, call_oi_notional, put_oi_notional, pc_oi_ratio,
        options_oi_total, opt_futures_ratio
    """
    result = {}

    # 1. Max Pain (nearest expiry)
    try:
        data = _cg_fetch_raw("/option/max-pain", {"symbol": "BTC", "exchange": "Deribit"})
        if isinstance(data, list) and len(data) > 0:
            nearest = data[0]
            call_oi = float(nearest.get("call_open_interest", 0))
            put_oi = float(nearest.get("put_open_interest", 0))
            result["max_pain_price"] = float(nearest.get("max_pain_price", 0))
            result["call_oi_notional"] = float(nearest.get("call_open_interest_notional", 0))
            result["put_oi_notional"] = float(nearest.get("put_open_interest_notional", 0))
            result["pc_oi_ratio"] = put_oi / call_oi if call_oi > 0 else 1.0
            result["nearest_expiry"] = nearest.get("date", "")
            logger.info("CG options max_pain: $%.0f, P/C OI=%.2f",
                        result["max_pain_price"], result["pc_oi_ratio"])
    except Exception as e:
        logger.error("CG options max_pain failed: %s", e)

    time.sleep(1)

    # 2. Options OI info (aggregated)
    try:
        data = _cg_fetch_raw("/option/info", {"symbol": "BTC"})
        if isinstance(data, list):
            total_oi = sum(float(d.get("openInterest", 0)) for d in data)
            result["options_oi_total"] = total_oi
    except Exception as e:
        logger.error("CG options info failed: %s", e)

    time.sleep(1)

    # 3. Options/Futures OI ratio
    try:
        data = _cg_fetch_raw("/index/option-vs-futures-oi-ratio", {"symbol": "BTC"})
        if isinstance(data, dict) and "ratio" in data:
            result["opt_futures_ratio"] = float(data["ratio"])
        elif isinstance(data, list) and len(data) > 0:
            result["opt_futures_ratio"] = float(data[-1].get("ratio", 0))
    except Exception as e:
        logger.error("CG opt/futures ratio failed: %s", e)

    if result:
        logger.info("CG options: %d fields fetched", len(result))
    return result


def fetch_cg_etf_flow() -> dict:
    """
    Fetch BTC ETF net flow data from Coinglass.

    Returns dict with:
        etf_net_flow_usd  — total daily net flow (all ETFs combined)
        etf_flow_ibit     — BlackRock IBIT net flow
        etf_flow_fbtc     — Fidelity FBTC net flow
        etf_btc_price     — BTC price at time of flow data
    """
    try:
        data = _cg_fetch_raw("/etf/bitcoin/flow-history")
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("CG ETF flow: empty response")
            return {}

        latest = data[-1]
        result = {
            "etf_net_flow_usd": float(latest.get("flow_usd", 0)),
            "etf_btc_price": float(latest.get("price_usd", 0)),
        }

        # Per-ticker breakdown
        tickers = latest.get("etf_flows", [])
        if isinstance(tickers, list):
            for t in tickers:
                name = t.get("etf_ticker", "").upper()
                flow = float(t.get("flow_usd", 0))
                if name == "IBIT":
                    result["etf_flow_ibit"] = flow
                elif name == "FBTC":
                    result["etf_flow_fbtc"] = flow

        logger.info("CG ETF flow: net=$%.1fM",
                     result["etf_net_flow_usd"] / 1e6)
        return result

    except Exception as e:
        logger.error("CG ETF flow failed: %s", e)
        return {}


def fetch_cg_fear_greed() -> dict:
    """Fetch Fear & Greed index (daily) from Coinglass."""
    try:
        data = _cg_fetch_raw("/index/fear-greed-history", {"limit": 2})
        if not isinstance(data, dict):
            return {}
        data_list = data.get("data_list", [])
        if not data_list:
            return {}
        latest = float(data_list[0])
        result = {"fear_greed_value": latest}
        logger.info("CG Fear & Greed: %.0f", latest)
        return result
    except Exception as e:
        logger.error("CG Fear & Greed failed: %s", e)
        return {}


def fetch_cg_etf_aum() -> dict:
    """Fetch BTC ETF total AUM from Coinglass."""
    try:
        data = _cg_fetch_raw("/etf/bitcoin/aum")
        if not isinstance(data, list) or len(data) == 0:
            return {}
        latest = data[-1]
        aum = float(latest.get("aum_usd", 0))
        result = {"etf_aum_usd": aum}
        logger.info("CG ETF AUM: $%.1fB", aum / 1e9)
        return result
    except Exception as e:
        logger.error("CG ETF AUM failed: %s", e)
        return {}


def _parse_netflow_row(data: list | dict, symbol: str = "BTC") -> dict:
    """Extract multi-timeframe netflow for a symbol from netflow-list response."""
    if not isinstance(data, list):
        return {}
    for row in data:
        if row.get("symbol", "").upper() == symbol.upper():
            result = {}
            for tf in ["5m", "15m", "1h", "4h", "24h"]:
                key = f"net_flow_usd_{tf}"
                if key in row:
                    result[tf] = float(row[key])
            return result
    return {}


def fetch_cg_futures_netflow() -> dict:
    """Fetch futures net flow (multi-timeframe) for BTC from Coinglass."""
    try:
        data = _cg_fetch_raw("/futures/netflow-list", {"symbol": "BTC"})
        nf = _parse_netflow_row(data, "BTC")
        result = {f"futures_netflow_{tf}": v for tf, v in nf.items()}
        if result:
            logger.info("CG Futures netflow: 1h=$%.0f, 24h=$%.0f",
                         result.get("futures_netflow_1h", 0),
                         result.get("futures_netflow_24h", 0))
        return result
    except Exception as e:
        logger.error("CG Futures netflow failed: %s", e)
        return {}


def fetch_cg_spot_netflow() -> dict:
    """Fetch spot net flow (multi-timeframe) for BTC from Coinglass."""
    try:
        data = _cg_fetch_raw("/spot/netflow-list", {"symbol": "BTC"})
        nf = _parse_netflow_row(data, "BTC")
        result = {f"spot_netflow_{tf}": v for tf, v in nf.items()}
        if result:
            logger.info("CG Spot netflow: 1h=$%.0f, 24h=$%.0f",
                         result.get("spot_netflow_1h", 0),
                         result.get("spot_netflow_24h", 0))
        return result
    except Exception as e:
        logger.error("CG Spot netflow failed: %s", e)
        return {}


def fetch_cg_hl_whale_positions() -> dict:
    """Fetch Hyperliquid whale position summary from Coinglass."""
    try:
        data = _cg_fetch_raw("/hyperliquid/whale-position", {"limit": 200})
        if not isinstance(data, list) or len(data) == 0:
            return {}
        # Aggregate: count, net USD, long fraction
        total = len(data)
        net_usd = 0.0
        long_count = 0
        for w in data:
            size = float(w.get("position_size", 0))
            entry = float(w.get("entry_price", 0))
            notional = abs(size) * entry if entry > 0 else abs(size)
            if size > 0:
                net_usd += notional
                long_count += 1
            else:
                net_usd -= notional
        result = {
            "hl_whale_count": total,
            "hl_whale_net_usd": net_usd,
            "hl_whale_long_pct": long_count / total if total > 0 else 0.5,
        }
        logger.info("CG HL whales: %d positions, net=$%.0f, long=%.1f%%",
                     total, net_usd, result["hl_whale_long_pct"] * 100)
        return result
    except Exception as e:
        logger.error("CG HL whale positions failed: %s", e)
        return {}


# ── Deribit public API (free, no auth) ───────────────────────────────────

def fetch_deribit_dvol() -> dict:
    """
    Fetch Deribit DVOL (BTC Volatility Index) via OHLC candle endpoint.

    Returns dict with dvol_value (close), dvol_open, dvol_high, dvol_low.
    """
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 7200_000  # last 2 hours
        resp = _retry_request("GET", DERIBIT_BASE + "/get_volatility_index_data", params={
            "currency": "BTC",
            "resolution": "3600",
            "start_timestamp": start_ms,
            "end_timestamp": now_ms,
        })
        data = resp.json().get("result", {}).get("data", [])

        if not data:
            logger.warning("Deribit DVOL: no data returned")
            return {}

        # Each entry: [timestamp, open, high, low, close]
        latest = data[-1]
        result = {
            "dvol_value": float(latest[4]),  # close
            "dvol_open": float(latest[1]),
            "dvol_high": float(latest[2]),
            "dvol_low": float(latest[3]),
            "dvol_change": float(latest[4]) - float(latest[1]),  # close - open
        }
        logger.info("Deribit DVOL: %.1f (change=%.2f)",
                     result["dvol_value"], result["dvol_change"])
        return result

    except Exception as e:
        logger.error("Deribit DVOL fetch failed: %s", e)
        return {}


def fetch_cross_market() -> dict:
    """
    Fetch daily SPX, DXY, Gold, US10Y from yfinance.

    Returns dict of {name: pd.Series} with daily close prices (last 30 days).
    Cached to .data_cache/cross_market.parquet with 6h TTL.
    Returns cached data or empty dict on failure.
    """
    cache_path = CACHE_DIR / "cross_market.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache TTL (6 hours)
    if cache_path.exists():
        try:
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_h < 6:
                cached = pd.read_parquet(cache_path)
                result = {}
                for col in cached.columns:
                    result[col] = cached[col].dropna()
                logger.info("Cross-market: loaded from cache (%d cols, %.1fh old)", len(result), age_h)
                return result
        except Exception:
            pass

    # Lazy import — may not be installed everywhere
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — cross-market data unavailable")
        # Try returning cached data even if stale
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                return {col: cached[col].dropna() for col in cached.columns}
            except Exception:
                pass
        return {}

    tickers = {
        "SPX": "^GSPC",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "US10Y": "^TNX",
    }

    try:
        symbols = list(tickers.values())
        data = yf.download(symbols, period="30d", interval="1d",
                           progress=False, threads=True)

        if data.empty:
            logger.warning("yfinance: empty response")
            if cache_path.exists():
                cached = pd.read_parquet(cache_path)
                return {col: cached[col].dropna() for col in cached.columns}
            return {}

        result = {}
        combined = pd.DataFrame()

        for name, ticker in tickers.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    col_data = data["Close"][ticker]
                else:
                    col_data = data["Close"]
                series = col_data.dropna()
                if len(series) > 0:
                    # Localize index to UTC if needed
                    if series.index.tz is None:
                        series.index = series.index.tz_localize("UTC")
                    elif str(series.index.tz) != "UTC":
                        series.index = series.index.tz_convert("UTC")
                    result[name] = series
                    combined[name] = series
            except Exception as e:
                logger.warning("Cross-market %s parse failed: %s", name, e)

        # Save cache
        if not combined.empty:
            try:
                combined.to_parquet(cache_path)
            except Exception:
                pass

        logger.info("Cross-market: fetched %d series via yfinance", len(result))
        return result

    except Exception as e:
        logger.error("Cross-market fetch failed: %s", e)
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
                return {col: cached[col].dropna() for col in cached.columns}
            except Exception:
                pass
        return {}


def fetch_fear_greed() -> pd.Series | None:
    """
    Fetch BTC Fear & Greed index from alternative.me API.

    Returns a Series indexed by date (UTC) with values 0-100 (last 30 days).
    Cached to .data_cache/fear_greed.json with 6h TTL.
    """
    import json as _json

    cache_path = CACHE_DIR / "fear_greed.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _parse_fng(data_list: list) -> pd.Series | None:
        """Parse alternative.me FNG response into a datetime-indexed Series."""
        records = []
        for item in data_list:
            try:
                ts = int(item["timestamp"])
                val = int(item["value"])
                dt = pd.Timestamp(ts, unit="s", tz="UTC")
                records.append((dt, val))
            except (KeyError, ValueError):
                continue
        if not records:
            return None
        s = pd.Series(
            [r[1] for r in records],
            index=pd.DatetimeIndex([r[0] for r in records], name="dt"),
            dtype=float,
            name="fng_value",
        ).sort_index()
        return s

    # Check cache TTL (6 hours)
    if cache_path.exists():
        try:
            age_h = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_h < 6:
                with open(cache_path) as f:
                    cached = _json.load(f)
                s = _parse_fng(cached)
                if s is not None:
                    logger.info("Fear & Greed: loaded from cache (%d days, %.1fh old)", len(s), age_h)
                    return s
        except Exception:
            pass

    # Fetch from API
    url = "https://api.alternative.me/fng/?limit=30&format=json"
    try:
        resp = _retry_request("GET", url)
        body = resp.json()
        data_list = body.get("data", [])

        if not data_list:
            logger.warning("Fear & Greed: empty response")
            # Fallback to cache
            if cache_path.exists():
                with open(cache_path) as f:
                    cached = _json.load(f)
                return _parse_fng(cached)
            return None

        # Save cache
        try:
            with open(cache_path, "w") as f:
                _json.dump(data_list, f)
        except Exception:
            pass

        s = _parse_fng(data_list)
        if s is not None:
            logger.info("Fear & Greed: fetched %d days (latest=%d)", len(s), int(s.iloc[-1]))
        return s

    except Exception as e:
        logger.error("Fear & Greed fetch failed: %s", e)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = _json.load(f)
                return _parse_fng(cached)
            except Exception:
                pass
        return None


def fetch_deribit_options_summary() -> dict:
    """
    Fetch Deribit BTC options summary — compute put/call ratio and IV skew.

    Returns dict with:
        pc_volume_ratio   — put/call volume ratio
        pc_oi_ratio       — put/call OI ratio
        iv_skew_25d       — 25-delta skew (OTM put IV - OTM call IV)
        mean_call_iv      — average call mark IV
        mean_put_iv       — average put mark IV
        total_call_oi     — total call OI in USD
        total_put_oi      — total put OI in USD
    """
    try:
        resp = _retry_request("GET", DERIBIT_BASE + "/get_book_summary_by_currency", params={
            "currency": "BTC", "kind": "option",
        })
        data = resp.json().get("result", [])

        if not isinstance(data, list) or len(data) == 0:
            return {}

        call_vol = 0.0
        put_vol = 0.0
        call_oi = 0.0
        put_oi = 0.0

        # For IV skew: collect OTM options near ATM
        # We need the index price to determine OTM
        # Parse strike from instrument name: BTC-DDMMMYY-STRIKE-C/P
        calls_by_strike = {}  # strike -> mark_iv
        puts_by_strike = {}

        for item in data:
            name = item.get("instrument_name", "")
            vol = float(item.get("volume_usd", 0) or 0)
            oi = float(item.get("open_interest", 0) or 0)
            mark_iv = item.get("mark_iv")

            # Parse strike price from name
            parts = name.split("-")
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
            except (ValueError, IndexError):
                continue

            if name.endswith("-C"):
                call_vol += vol
                call_oi += oi
                if mark_iv and mark_iv > 0:
                    calls_by_strike[strike] = float(mark_iv)
            elif name.endswith("-P"):
                put_vol += vol
                put_oi += oi
                if mark_iv and mark_iv > 0:
                    puts_by_strike[strike] = float(mark_iv)

        result = {
            "pc_volume_ratio": put_vol / call_vol if call_vol > 0 else 1.0,
            "pc_oi_ratio_deribit": put_oi / call_oi if call_oi > 0 else 1.0,
            "total_call_oi": call_oi,
            "total_put_oi": put_oi,
        }

        # IV skew: compare 25-delta proxy (OTM puts vs OTM calls near ATM)
        # Find strikes where both put and call have IV data
        common_strikes = sorted(set(calls_by_strike) & set(puts_by_strike))
        if common_strikes:
            # ATM ~ median strike; take strikes ~5-15% OTM
            mid_strike = common_strikes[len(common_strikes) // 2]
            # OTM puts: strikes below ATM; OTM calls: strikes above ATM
            otm_put_ivs = [puts_by_strike[s] for s in common_strikes
                           if s < mid_strike * 0.95 and s > mid_strike * 0.75]
            otm_call_ivs = [calls_by_strike[s] for s in common_strikes
                            if s > mid_strike * 1.05 and s < mid_strike * 1.25]

            if otm_put_ivs and otm_call_ivs:
                result["iv_skew"] = np.mean(otm_put_ivs) - np.mean(otm_call_ivs)
            else:
                result["iv_skew"] = 0.0
            result["mean_otm_put_iv"] = np.mean(otm_put_ivs) if otm_put_ivs else 0
            result["mean_otm_call_iv"] = np.mean(otm_call_ivs) if otm_call_ivs else 0
        else:
            result["iv_skew"] = 0.0

        logger.info("Deribit options: P/C vol=%.2f, P/C OI=%.2f, IV skew=%.1f",
                     result["pc_volume_ratio"], result["pc_oi_ratio_deribit"],
                     result["iv_skew"])
        return result

    except Exception as e:
        logger.error("Deribit options summary failed: %s", e)
        return {}
