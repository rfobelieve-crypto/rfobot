"""Value-differentiation check: for every coin-style endpoint, confirm the
ETH/SOL responses actually differ from BTC (guards against endpoints that
silently ignore the symbol param, as /coinbase-premium-index does).
Pair-style (BTCUSDT/ETHUSDT) endpoints error on bad symbols, so one
representative is included as control.
"""
import time
from audit_cg_endpoints import cg_get

CHECKS = {
    # coin-style endpoints
    "oi_agg":          ("/futures/open-interest/aggregated-history", {"symbol": "{c}"}),
    "bitfinex_margin": ("/bitfinex-margin-long-short", {"symbol": "{c}"}),
    "futures_cvd_agg": ("/futures/aggregated-cvd/history", {"symbol": "{c}", "exchange_list": "Binance"}),
    "spot_cvd_agg":    ("/spot/aggregated-cvd/history", {"symbol": "{c}", "exchange_list": "Binance"}),
    "liq_agg":         ("/futures/liquidation/aggregated-history", {"symbol": "{c}", "exchange_list": "Binance"}),
    "oi_coin_margin":  ("/futures/open-interest/aggregated-coin-margin-history", {"symbol": "{c}", "exchange_list": "Binance"}),
    # pair-style control
    "oi (pair ctrl)":  ("/futures/open-interest/history", {"symbol": "{c}USDT", "exchange": "Binance"}),
}

for name, (path, tmpl) in CHECKS.items():
    rows = {}
    for coin in ["BTC", "ETH", "SOL"]:
        params = {"interval": "1h", "limit": 3}
        params.update({k: v.format(c=coin) for k, v in tmpl.items()})
        status, data = cg_get(path, params)
        rows[coin] = str(data[-1]) if status == "OK" and isinstance(data, list) and data else status
        time.sleep(0.8)
    distinct = len(set(rows.values()))
    verdict = "OK distinct" if distinct == 3 else f"SUSPECT ({distinct} distinct)"
    print(f"{name:18s} -> {verdict}")
    if distinct < 3:
        for coin, v in rows.items():
            print(f"    {coin}: {v[:120]}")
