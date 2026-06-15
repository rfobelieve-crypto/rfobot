"""READ-ONLY OKX live state check. Submits NO orders, closes NOTHING.
Only GET /account/positions + /account/balance + /public/time.
Purpose: see the real state of the orphan SHORT after the auto admin_heal."""
import os, sys
sys.path.insert(0, ".")
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
os.environ.setdefault("STAGE", "live")

from indicator.okx.config import load_okx_config_from_env
from indicator.okx.rest import OkxRestClient

cfg = load_okx_config_from_env("live")
print("=== credentials presence (live) ===")
print(f"  api_key set: {bool(cfg.api_key)}  secret set: {bool(cfg.api_secret)}  "
      f"passphrase set: {bool(cfg.passphrase)}  is_simulated={cfg.is_simulated}")
if not cfg.api_key:
    print("  !! OKX_API_KEY_LIVE not in env — cannot query. Stop.")
    sys.exit(1)

cli = OkxRestClient(cfg)

st = cli.get_server_time()
print(f"\n=== connectivity === server_time={st}")

bal = cli.get_balance()
print(f"\n=== balance ===")
print(f"  {bal}")

print(f"\n=== positions (BTC-USDT-SWAP) ===")
positions = cli.get_positions(inst_id="BTC-USDT-SWAP")
if not positions:
    print("  NO open positions on OKX. (orphan SHORT already gone — stop hit or closed)")
for p in positions:
    print(f"  direction={p.direction}  size={p.size_contracts}  avg_price={p.avg_price}  "
          f"upl_usd={p.unrealized_pnl_usd:+.2f}")
    # surface risk fields from raw OKX payload if present
    r = p.raw or {}
    for k in ("liqPx", "mgnRatio", "margin", "imr", "mmr", "lever", "markPx", "uplRatio"):
        if k in r and r[k] not in ("", None):
            print(f"      {k}={r[k]}")
