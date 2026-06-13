"""Strong-only vs Strong+Moderate: entry-edge comparison by tier.

Uses the V7 OOS predictions (pred_ret) + realized 4h TWAP path return saved by
longhorizon_phase0_diag. Tiers mirror production: Strong = top/bottom 5% of
pred_ret, Moderate = the 5-15% band (both directions = long+short).

PnL proxy = sign(pred) * y_path_ret_4h (directional path return), minus round-
trip taker cost (8 bps). This is ENTRY EDGE at the model's 4h horizon, NOT live
PnL — real exits use a 3xATR trailing/signal stop, so absolute bps differ.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
P = ROOT / "research" / "results" / "dual_model" / "v7_oos_for_horizon.parquet"
COST = 0.0008          # round-trip taker (cfg.taker_cost)
MONTHS = 5.5

df = pd.read_parquet(P)
pred = df["pred_ret"].values
y = df["y_path_ret_4h"].values

p95, p85 = np.nanpercentile(pred, 95), np.nanpercentile(pred, 85)
p05, p15 = np.nanpercentile(pred, 5), np.nanpercentile(pred, 15)

strong = (pred >= p95) | (pred <= p05)
moder = ((pred >= p85) & (pred < p95)) | ((pred > p05) & (pred <= p15))
both = strong | moder


def stats(mask, label):
    n = int(mask.sum())
    if n == 0:
        return dict(cfg=label, n=0)
    direction = np.sign(pred[mask])
    dret = direction * y[mask]            # directional path return
    wr = float((dret > 0).mean())
    gross = float(dret.mean())
    net = gross - COST
    total_net = float((dret - COST).sum())
    return dict(cfg=label, n=n, trades_per_day=n / (MONTHS * 30.4),
                WR=wr, gross_bps=gross * 1e4, net_bps=net * 1e4,
                total_net_pct=total_net * 100)


rows = [
    stats(strong, "Strong only"),
    stats(both, "Strong + Moderate"),
    stats(moder, "(Moderate band only)"),
]

print("=" * 96)
print(f"V7 OOS entry edge by tier — n={len(df)} bars, 4h path return, "
      f"cost={COST*1e4:.0f}bps round-trip")
print("=" * 96)
print(f"{'config':22s} {'n':>5s} {'tr/day':>7s} {'WR':>7s} "
      f"{'gross':>9s} {'net':>9s} {'totalNet':>10s}")
print("-" * 96)
for r in rows:
    if r.get("n", 0) == 0:
        continue
    print(f"{r['cfg']:22s} {r['n']:>5d} {r['trades_per_day']:>7.2f} "
          f"{r['WR']*100:>6.1f}% {r['gross_bps']:>+8.1f} {r['net_bps']:>+8.1f} "
          f"{r['total_net_pct']:>+9.2f}%")
print("-" * 96)

s, b, m = rows[0], rows[1], rows[2]
print(f"\nStrong-only:        {s['WR']*100:.1f}% WR, {s['net_bps']:+.1f} net bps/trade, "
      f"{s['n']} trades → {s['total_net_pct']:+.2f}% summed")
print(f"Strong+Moderate:    {b['WR']*100:.1f}% WR, {b['net_bps']:+.1f} net bps/trade, "
      f"{b['n']} trades → {b['total_net_pct']:+.2f}% summed")
print(f"Moderate band adds: {m['n']} trades @ {m['WR']*100:.1f}% WR, "
      f"{m['net_bps']:+.1f} net bps/trade → {m['total_net_pct']:+.2f}% summed")
print(f"\nDelta (both − strong): WR {(b['WR']-s['WR'])*100:+.1f}pp, "
      f"per-trade net {b['net_bps']-s['net_bps']:+.1f}bps, "
      f"total summed {b['total_net_pct']-s['total_net_pct']:+.2f}%")
