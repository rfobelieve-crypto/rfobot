"""Scan all Coinglass API endpoints to find what's available on Startup plan."""
import os, requests, time
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

KEY = os.environ.get("COINGLASS_API_KEY", "")
BASE = "https://open-api-v4.coinglass.com/api"
headers = {"CG-API-KEY": KEY, "accept": "application/json"}

tests = [
    # Futures extras
    ("Fut OI Exchange List",       "/futures/open-interest/exchange-list", {"symbol": "BTC"}),
    ("Fut OI Stablecoin Margin",   "/futures/open-interest/aggregated-stablecoin-margin-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Fut OI Coin Margin",         "/futures/open-interest/aggregated-coin-margin-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("FR OI Weight",               "/futures/funding-rate/oi-weight-ohlc-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("FR Vol Weight",              "/futures/funding-rate/vol-weight-ohlc-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("FR Arbitrage",               "/futures/funding-rate/arbitrage", {"symbol": "BTC"}),
    ("Top LS Position",            "/futures/top-long-short-position-ratio/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Liq Coin History",           "/futures/liquidation/aggregated-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Liq Order",                  "/futures/liquidation/order", {"symbol": "BTC", "limit": 2}),
    ("Liq Max Pain",               "/futures/liquidation/max-pain", {"symbol": "BTC"}),
    ("Liq Heatmap M1",            "/futures/liquidation-heatmap/model1", {"symbol": "BTC", "exchange": "Binance"}),
    ("Liq Heatmap Pair M1",       "/futures/liquidation-heatmap/pair-model1", {"symbol": "BTCUSDT", "exchange": "Binance"}),
    ("Orderbook History",          "/futures/orderbook/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Orderbook Agg",              "/futures/orderbook/aggregated-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Orderbook Heatmap",          "/futures/orderbook/heatmap", {"symbol": "BTCUSDT", "exchange": "Binance"}),
    ("Large Orderbook",            "/futures/orderbook/large", {"symbol": "BTC"}),
    ("Large Orderbook History",    "/futures/orderbook/large-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Fut Footprint",              "/futures/footprint", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Fut CVD Pair",               "/futures/cvd/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Fut CVD Agg",                "/futures/aggregated-cvd/history", {"symbol": "BTC", "interval": "1h", "limit": 2, "exchange_list": "Binance"}),
    ("Fut Coin Netflow",           "/futures/coin-netflow", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Fut Netflow List",           "/futures/netflow-list", {"symbol": "BTC"}),
    # Spot
    ("Spot Taker Pair",            "/spot/taker-buysell-ratio/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Spot Taker Agg",             "/spot/aggregated-taker-buysell/history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Spot Orderbook",             "/spot/orderbook/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Spot Orderbook Agg",         "/spot/orderbook/aggregated-history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Spot Large Orders",          "/spot/orderbook/large", {"symbol": "BTC"}),
    ("Spot Footprint",             "/spot/footprint", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Spot CVD Pair",              "/spot/cvd/history", {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 2}),
    ("Spot CVD Agg",               "/spot/aggregated-cvd/history", {"symbol": "BTC", "interval": "1h", "limit": 2, "exchange_list": "Binance"}),
    ("Spot Coin Netflow",          "/spot/coin-netflow", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Spot Netflow List",          "/spot/netflow-list", {"symbol": "BTC"}),
    # Options
    ("Options OI Exchange",        "/option/exchange-open-interest/history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Options Vol Exchange",       "/option/exchange-volume/history", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    # Indicators
    ("Coinbase Premium",           "/coinbase-premium-index", {"interval": "1h", "limit": 2}),
    ("Bitfinex Margin",            "/bitfinex-margin-long-short", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("BTC Dominance",              "/index/bitcoin-dominance", {"interval": "1h", "limit": 2}),
    ("Fear Greed",                 "/index/fear-greed-history", {"limit": 2}),
    ("Opt/Fut Ratio",              "/index/option-vs-futures-oi-ratio", {"symbol": "BTC", "interval": "1h", "limit": 2}),
    ("Borrow Rate",                "/borrow-interest-rate", {"symbol": "BTC"}),
    # On-chain
    ("Exchange Assets",            "/exchange/assets", {"symbol": "BTC"}),
    ("Exchange Balance List",      "/exchange/balance-list", {"symbol": "BTC"}),
    ("Exchange Balance Chart",     "/exchange/balance-chart", {"symbol": "BTC"}),
    ("Whale Transfer",             "/onchain/whale-transfer", {"symbol": "BTC", "limit": 2}),
    # ETF
    ("BTC ETF List",               "/etf/bitcoin/list", {}),
    ("BTC ETF Net Assets",         "/etf/bitcoin/netassets-history", {"limit": 2}),
    ("BTC ETF Premium",            "/etf/bitcoin/premium-discount-history", {"limit": 2}),
    ("BTC ETF AUM",                "/etf/bitcoin/aum", {}),
    ("HK ETF Flow",                "/etf/hong-kong-bitcoin/flow-history", {"limit": 2}),
    # On-chain indicators
    ("Whale Index",                "/indicator/whale-index", {"interval": "1h", "limit": 2}),
    ("CGDI Index",                 "/indicator/cgdi-index", {"limit": 2}),
    ("AHR999",                     "/indicator/ahr999", {"limit": 2}),
    ("Puell Multiple",             "/indicator/puell-multiple", {"limit": 2}),
    ("Stock to Flow",              "/indicator/stock-flow", {"limit": 2}),
    ("Altcoin Season",             "/indicator/altcoin-season-index", {"limit": 2}),
    ("STH SOPR",                   "/indicator/bitcoin-short-term-holder-sopr", {"limit": 2}),
    ("LTH SOPR",                   "/indicator/bitcoin-long-term-holder-sopr", {"limit": 2}),
    ("NUPL",                       "/indicator/bitcoin-nupl", {"limit": 2}),
    ("Active Addresses",           "/indicator/bitcoin-active-addresses", {"limit": 2}),
    ("Macro Oscillator",           "/indicator/bitcoin-macro-oscillator", {"limit": 2}),
    ("Futures/Spot Vol",           "/indicator/futures-spot-volume-ratio", {"limit": 2}),
    ("Reserve Risk",               "/indicator/bitcoin-reserve-risk", {"limit": 2}),
    # Hyperliquid
    ("HL Whale Alert",             "/hyperliquid/whale-alert", {"limit": 2}),
    ("HL Whale Position",          "/hyperliquid/whale-position", {"limit": 2}),
    ("HL LS Ratio",                "/hyperliquid/long-short-account-ratio/history", {"interval": "1h", "limit": 2}),
]

ok = []
upgrade = []
fail = []

for name, path, params in tests:
    try:
        resp = requests.get(BASE + path, headers=headers, params=params, timeout=10)
        data = resp.json()
        code = data.get("code", "?")
        msg = data.get("msg", "")[:50]
        d = data.get("data")
        if code == "0" and d:
            n = len(d) if isinstance(d, list) else "dict"
            keys = ""
            if isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                keys = str(list(d[0].keys())[:5])
            elif isinstance(d, dict):
                keys = str(list(d.keys())[:5])
            ok.append((name, path, n, keys))
        elif "401" in str(code) or "Upgrade" in msg or "upgrade" in msg.lower():
            upgrade.append((name, path, msg))
        else:
            fail.append((name, path, f"{code}: {msg}"))
    except Exception as e:
        fail.append((name, path, str(e)[:40]))
    time.sleep(0.8)

print("=" * 70)
print(f"AVAILABLE on Startup ({len(ok)} endpoints)")
print("=" * 70)
for n, p, cnt, keys in ok:
    print(f"  {n:30s} {p}")
    if keys:
        print(f"    {cnt} items, keys: {keys}")

print(f"\n{'=' * 70}")
print(f"NEED UPGRADE ({len(upgrade)} endpoints)")
print("=" * 70)
for n, p, msg in upgrade:
    print(f"  {n:30s} {p}  ({msg})")

print(f"\n{'=' * 70}")
print(f"NOT FOUND / ERROR ({len(fail)} endpoints)")
print("=" * 70)
for n, p, msg in fail:
    print(f"  {n:30s} {msg}")
