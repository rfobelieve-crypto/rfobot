"""
Funding Rate collector for OKX + Binance.

Polls funding rates every 60s via REST and writes to funding_rates table.
Also provides collect_once() for on-demand collection at event time.

OKX:     GET /api/v5/public/funding-rate?instId=BTC-USDT-SWAP
Binance: GET /fapi/v1/premiumIndex?symbol=BTCUSDT
"""

import time
import logging
import requests

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

COLLECT_INTERVAL = 60  # seconds

TRACKED = [
    ("okx",     "BTC-USDT-SWAP", "BTC-USD"),
    ("okx",     "ETH-USDT-SWAP", "ETH-USD"),
    ("binance", "BTCUSDT",       "BTC-USD"),
    ("binance", "ETHUSDT",       "ETH-USD"),
]


def fetch_okx_funding(inst_id: str) -> dict | None:
    url = "https://www.okx.com/api/v5/public/funding-rate"
    try:
        resp = requests.get(url, params={"instId": inst_id}, timeout=10)
        data = resp.json()
        if data.get("code") != "0" or not data.get("data"):
            logger.warning("OKX funding response error for %s: %s", inst_id, data.get("msg"))
            return None
        item = data["data"][0]
        return {
            "funding_rate":     float(item["fundingRate"]),
            "next_funding_ts":  int(item["nextFundingTime"]),
            "ts_exchange":      int(item["fundingTime"]),
        }
    except Exception:
        logger.exception("Failed to fetch OKX funding for %s", inst_id)
        return None


def fetch_binance_funding(symbol: str) -> dict | None:
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        data = resp.json()
        if "lastFundingRate" not in data:
            logger.warning("Binance funding response error for %s: %s", symbol, data)
            return None
        return {
            "funding_rate":    float(data["lastFundingRate"]),
            "next_funding_ts": int(data["nextFundingTime"]),
            "ts_exchange":     int(data.get("time", time.time() * 1000)),
        }
    except Exception:
        logger.exception("Failed to fetch Binance funding for %s", symbol)
        return None


def save_funding(exchange: str, canonical_symbol: str,
                 funding_rate: float, next_funding_ts: int, ts_exchange: int):
    sql = """
    INSERT INTO funding_rates
        (exchange, canonical_symbol, funding_rate, next_funding_ts, ts_exchange, ts_received)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    ts_received = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (exchange, canonical_symbol, funding_rate,
                              next_funding_ts, ts_exchange, ts_received))
    finally:
        conn.close()


def collect_once():
    """Fetch funding rates from all sources and save to DB."""
    count = 0
    for exchange, api_param, canonical in TRACKED:
        try:
            if exchange == "okx":
                data = fetch_okx_funding(api_param)
            else:
                data = fetch_binance_funding(api_param)

            if not data:
                continue

            save_funding(exchange, canonical,
                         data["funding_rate"], data["next_funding_ts"], data["ts_exchange"])
            count += 1
        except Exception:
            logger.exception("Failed to collect funding for %s %s", exchange, api_param)

    if count > 0:
        logger.debug("Funding rates collected: %d sources", count)
    return count


def collect_loop():
    """Run funding rate collection in a loop."""
    logger.info("Funding collector starting (every %ds)", COLLECT_INTERVAL)
    while True:
        try:
            collect_once()
        except Exception:
            logger.exception("Funding collect_once error")
        time.sleep(COLLECT_INTERVAL)
