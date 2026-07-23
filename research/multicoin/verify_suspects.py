"""Follow-up checks for two suspicious audit results:
1. Does /coinbase-premium-index accept a symbol param (ETH)?
2. Does /index/option-vs-futures-oi-ratio actually vary by symbol,
   or does it ignore the param and always return BTC?
"""
import time
import requests
from audit_cg_endpoints import CG_BASE, HEADERS, cg_get

# 1. coinbase premium with explicit symbol
for sym in ["ETH", "SOL"]:
    status, data = cg_get("/coinbase-premium-index", {"interval": "1h", "limit": 3, "symbol": sym})
    print(f"coinbase_premium symbol={sym}: {status}", end=" ")
    if status == "OK" and isinstance(data, list) and data:
        print(f"last premium_rate={data[-1].get('premium_rate', data[-1])}")
    else:
        print(data if data else "")
    time.sleep(0.8)

# also BTC control to compare values
status, btc = cg_get("/coinbase-premium-index", {"interval": "1h", "limit": 3, "symbol": "BTC"})
if status == "OK" and btc:
    print(f"coinbase_premium symbol=BTC control: last premium_rate={btc[-1].get('premium_rate')}")
time.sleep(0.8)

# 2. option-vs-futures ratio: compare last values across symbols
vals = {}
for sym in ["BTC", "ETH", "SOL"]:
    status, data = cg_get("/index/option-vs-futures-oi-ratio", {"symbol": sym})
    if status == "OK" and isinstance(data, list) and data:
        vals[sym] = data[-1]
    else:
        vals[sym] = status
    time.sleep(0.8)
print("\nopt_fut_ratio last rows:")
for sym, v in vals.items():
    print(f"  {sym}: {v}")
identical = len({str(v) for v in vals.values()}) == 1
print(f"  -> identical across symbols: {identical} (True = endpoint ignores symbol, BTC-only)")
