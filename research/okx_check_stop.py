"""READ-ONLY: is the stop algo @63055 still on OKX for the orphan SHORT?"""
import os, sys
sys.path.insert(0, ".")
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
os.environ.setdefault("STAGE", "live")
from indicator.okx.config import load_okx_config_from_env
from indicator.okx.rest import OkxRestClient

cli = OkxRestClient(load_okx_config_from_env("live"))
for ot in ("conditional", "oco", "trigger", "move_order_stop"):
    r = cli._retry_get(
        path="/api/v5/trade/orders-algo-pending",
        params={"instId": "BTC-USDT-SWAP", "ordType": ot},
        retries=2, backoff_base=0.5,
    )
    rows = r.get("data", []) if isinstance(r, dict) else (r or [])
    print(f"\n=== ordType={ot}: {len(rows)} pending ===")
    for o in rows:
        print("  ", {k: o.get(k) for k in
              ("algoId", "algoClOrdId", "side", "posSide", "sz",
               "slTriggerPx", "tpTriggerPx", "state", "ordType")})
