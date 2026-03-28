"""
OI (Open Interest) collector for OKX + Binance.

Periodically fetches OI via REST API and writes to oi_snapshots table.
Runs every 60 seconds by default.

OKX API: GET /api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP
Binance API: GET /fapi/v1/openInterest?symbol=BTCUSDT
"""

import time
import logging
import requests

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

COLLECT_INTERVAL = 60  # seconds

# Symbols to track: (exchange, api_param, canonical_symbol)
TRACKED = [
    ("okx", "BTC-USDT-SWAP", "BTC-USD"),
    ("okx", "ETH-USDT-SWAP", "ETH-USD"),
    ("binance", "BTCUSDT", "BTC-USD"),
    ("binance", "ETHUSDT", "ETH-USD"),
]

# OKX contract sizes for notional calculation
OKX_CONTRACT_SIZE = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.1}


def fetch_okx_oi(inst_id: str) -> dict | None:
    """Fetch OI from OKX REST API."""
    url = "https://www.okx.com/api/v5/public/open-interest"
    try:
        resp = requests.get(url, params={"instType": "SWAP", "instId": inst_id}, timeout=10)
        data = resp.json()
        if data.get("code") != "0" or not data.get("data"):
            logger.warning("OKX OI response error for %s: %s", inst_id, data.get("msg"))
            return None
        item = data["data"][0]
        oi_contracts = float(item["oi"])
        ts_exchange = int(item["ts"])
        # notional = contracts * contract_size * mark_price (approximate with instId)
        # OKX returns oi in contracts, we need USD notional
        # Use oiCcy if available (OI in coin), else approximate
        oi_coin = float(item.get("oiCcy", 0))
        return {
            "oi_contracts": oi_contracts,
            "oi_coin": oi_coin,
            "ts_exchange": ts_exchange,
        }
    except Exception:
        logger.exception("Failed to fetch OKX OI for %s", inst_id)
        return None


def fetch_binance_oi(symbol: str) -> dict | None:
    """Fetch OI from Binance Futures REST API."""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        data = resp.json()
        if "openInterest" not in data:
            logger.warning("Binance OI response error for %s: %s", symbol, data)
            return None
        oi_coin = float(data["openInterest"])
        ts_exchange = int(data.get("time", time.time() * 1000))
        return {
            "oi_contracts": oi_coin,  # Binance returns in base asset (coin)
            "oi_coin": oi_coin,
            "ts_exchange": ts_exchange,
        }
    except Exception:
        logger.exception("Failed to fetch Binance OI for %s", symbol)
        return None


def fetch_mark_price_okx(inst_id: str) -> float | None:
    """Fetch mark price from OKX for notional calculation."""
    url = "https://www.okx.com/api/v5/public/mark-price"
    try:
        resp = requests.get(url, params={"instType": "SWAP", "instId": inst_id}, timeout=10)
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            return float(data["data"][0]["markPx"])
    except Exception:
        pass
    return None


def fetch_mark_price_binance(symbol: str) -> float | None:
    """Fetch mark price from Binance for notional calculation."""
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        data = resp.json()
        return float(data["markPrice"])
    except Exception:
        pass
    return None


def save_oi(exchange: str, canonical_symbol: str,
            oi_contracts: float, oi_notional_usd: float,
            ts_exchange: int):
    """Insert one OI snapshot into oi_snapshots."""
    sql = """
    INSERT INTO oi_snapshots (exchange, canonical_symbol, oi_contracts, oi_notional_usd,
                              ts_exchange, ts_received)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    ts_received = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (exchange, canonical_symbol, oi_contracts,
                              oi_notional_usd, ts_exchange, ts_received))
    finally:
        conn.close()


def collect_once():
    """Fetch OI from all sources and save to DB."""
    count = 0
    for exchange, api_param, canonical in TRACKED:
        try:
            if exchange == "okx":
                oi_data = fetch_okx_oi(api_param)
                if not oi_data:
                    continue
                mark = fetch_mark_price_okx(api_param)
                contract_size = OKX_CONTRACT_SIZE.get(api_param, 0.01)
                # OKX: notional = oi_contracts * contract_size * mark_price
                if mark:
                    notional = oi_data["oi_contracts"] * contract_size * mark
                else:
                    # Fallback: use oi_coin * mark (if oiCcy available)
                    notional = 0

            elif exchange == "binance":
                oi_data = fetch_binance_oi(api_param)
                if not oi_data:
                    continue
                mark = fetch_mark_price_binance(api_param)
                # Binance: openInterest is in base asset (BTC/ETH), notional = oi * mark
                if mark:
                    notional = oi_data["oi_coin"] * mark
                else:
                    notional = 0
            else:
                continue

            save_oi(exchange, canonical, oi_data["oi_contracts"],
                    notional, oi_data["ts_exchange"])
            count += 1

        except Exception:
            logger.exception("Failed to collect OI for %s %s", exchange, api_param)

    if count > 0:
        logger.debug("OI collected: %d sources", count)
    return count


def collect_loop():
    """Run OI collection in a loop."""
    logger.info("OI collector starting (every %ds)", COLLECT_INTERVAL)
    while True:
        try:
            collect_once()
        except Exception:
            logger.exception("OI collect_once error")
        time.sleep(COLLECT_INTERVAL)
