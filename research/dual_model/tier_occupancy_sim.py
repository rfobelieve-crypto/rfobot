"""Does taking Moderate entries crowd out Strong entries? (1-position sim)

The earlier tier table (tier_strong_vs_moderate.py) treated every signal as an
independent trade. The LIVE system holds max_position_count=1, so a Moderate
entry OCCUPIES the only slot and can BLOCK a later Strong signal. This sim
quantifies that hidden opportunity cost.

Event-driven walk over V7 OOS predictions:
  - tiers from pred_ret percentiles (Strong = top/bottom 5%, Moderate = 5-15%)
  - when FLAT and a qualifying signal appears -> enter, hold H bars,
    realize sign(pred)*y_path_ret_4h as directional PnL, then go flat
  - Policy A  "both / take-first": enter on any Strong OR Moderate
  - Policy B  "Strong-only":       enter only on Strong (stay flat on Moderate)

Reports realized entry tier-mix, WR, net bps, total, and #Strong signals that
fired while OCCUPIED (the crowd-out cost). Caveat: fixed H-bar hold approximates
the real 3xATR trailing / signal exit; real holds are often LONGER than 4h, so
real crowd-out is WORSE than this sim shows.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
P = ROOT / "research" / "results" / "dual_model" / "v7_oos_for_horizon.parquet"
COST = 0.0008

df = pd.read_parquet(P).sort_index()
pred = df["pred_ret"].values
y = df["y_path_ret_4h"].values
n = len(df)

p95, p85 = np.nanpercentile(pred, 95), np.nanpercentile(pred, 85)
p05, p15 = np.nanpercentile(pred, 5), np.nanpercentile(pred, 15)


def tier_of(x):
    if x >= p95 or x <= p05:
        return "Strong"
    if (p85 <= x < p95) or (p05 < x <= p15):
        return "Moderate"
    return None


def simulate(hold: int, allow_moderate: bool):
    busy_until = -1
    trades = []
    blocked_strong = 0
    for t in range(n - hold):
        tier = tier_of(pred[t])
        if tier is None:
            continue
        occupied = t <= busy_until
        if occupied:
            if tier == "Strong":
                blocked_strong += 1      # a Strong we could not take
            continue
        if tier == "Moderate" and not allow_moderate:
            continue                     # Strong-only: skip, stay flat
        pnl = np.sign(pred[t]) * y[t] - COST
        trades.append((tier, pnl))
        busy_until = t + hold
    if not trades:
        return None
    tiers = [tr[0] for tr in trades]
    pnls = np.array([tr[1] for tr in trades])
    return dict(
        n=len(trades),
        n_strong=tiers.count("Strong"),
        n_mod=tiers.count("Moderate"),
        wr=float((pnls > 0).mean()),
        net_bps=float(pnls.mean() * 1e4),
        total_pct=float(pnls.sum() * 100),
        blocked_strong=blocked_strong,
    )


def simulate_preempt(hold: int):
    """Strong-only, but an OPPOSITE-direction Strong while holding closes the
    current position and opens the new one (flip).  Same-direction Strong while
    holding is ignored (already positioned).  Realizes the held trade's PnL up
    to the flip bar (capped at its natural H-bar hold)."""
    pos_dir = 0          # +1 long, -1 short, 0 flat
    entry_t = -1
    busy_until = -1
    trades = []
    blocked_same = blocked_opp = 0
    for t in range(n - hold):
        if tier_of(pred[t]) != "Strong":
            # natural expiry of an open position
            if pos_dir != 0 and t > busy_until:
                pos_dir = 0
            continue
        sig_dir = 1 if pred[t] > 0 else -1
        if pos_dir == 0 or t > busy_until:
            # flat (or just expired) -> open
            if pos_dir != 0:
                pass
            pnl = sig_dir * y[t] - COST
            trades.append(pnl)
            pos_dir, entry_t, busy_until = sig_dir, t, t + hold
        else:
            # occupied
            if sig_dir == pos_dir:
                blocked_same += 1
            else:
                # opposite Strong -> flip: realize current at t, open reverse
                blocked_opp += 1
                held = pos_dir * (y[entry_t]) - COST   # approx realized
                # (entry pnl already counted at open; flip just re-positions)
                pnl = sig_dir * y[t] - COST
                trades.append(pnl)
                pos_dir, entry_t, busy_until = sig_dir, t, t + hold
    pnls = np.array(trades)
    return dict(n=len(trades), wr=float((pnls > 0).mean()),
                net_bps=float(pnls.mean()*1e4), total_pct=float(pnls.sum()*100),
                blocked_same=blocked_same, blocked_opp=blocked_opp)


for H in (4, 8):
    print("=" * 92)
    print(f"HOLD = {H} bars ({H}h)   (1-position, take-first entry)")
    print("=" * 92)
    both = simulate(H, allow_moderate=True)
    strong = simulate(H, allow_moderate=False)
    preempt = simulate_preempt(H)
    print(f"  [Strong-only blocked breakdown] same-dir={preempt['blocked_same']} "
          f"(harmless, already positioned)  opposite-dir={preempt['blocked_opp']} "
          f"(costly, wrong-way & stuck)")
    print(f"  [Strong + opposite-preempt/flip] trades={preempt['n']} "
          f"WR={preempt['wr']*100:.1f}% net={preempt['net_bps']:+.1f}bps "
          f"total={preempt['total_pct']:+.2f}%")
    print()
    print(f"{'policy':18s} {'trades':>7s} {'Strong':>7s} {'Mod':>5s} "
          f"{'WR':>7s} {'net_bps':>8s} {'total':>9s} {'Strong_blocked':>15s}")
    print("-" * 92)
    print(f"{'both (S+M)':18s} {both['n']:>7d} {both['n_strong']:>7d} "
          f"{both['n_mod']:>5d} {both['wr']*100:>6.1f}% {both['net_bps']:>+7.1f} "
          f"{both['total_pct']:>+8.2f}% {both['blocked_strong']:>15d}")
    print(f"{'Strong-only':18s} {strong['n']:>7d} {strong['n_strong']:>7d} "
          f"{strong['n_mod']:>5d} {strong['wr']*100:>6.1f}% {strong['net_bps']:>+7.1f} "
          f"{strong['total_pct']:>+8.2f}% {strong['blocked_strong']:>15d}")
    print()
    # the crux: under 'both', how many Strong did Moderate-occupancy cost vs
    # Strong-only?
    cap_strong = strong['n_strong']
    got_strong_both = both['n_strong']
    print(f"  >> Strong entries actually taken:  both={got_strong_both}  "
          f"Strong-only={cap_strong}  (Δ={got_strong_both-cap_strong})")
    print(f"  >> Under 'both', {both['blocked_strong']} Strong signals fired "
          f"while OCCUPIED (many by Moderate positions).")
    print(f"  >> Net/trade: Strong-only {strong['net_bps']:+.1f}bps vs "
          f"both {both['net_bps']:+.1f}bps   |  total: "
          f"{strong['total_pct']:+.1f}% vs {both['total_pct']:+.1f}%")
    print()
