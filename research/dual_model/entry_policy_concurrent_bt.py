"""K-concurrent extension of the real-exit entry-policy backtest.

Question (user): if we go Strong-only, what changes if we also allow TWO
positions at once (K=2) instead of one? Re-verify under the REAL exit logic
(3xATR TRAILING stop + opposite-signal; time-cap removed).

Engine: event-driven multi-position. Up to K positions open at once; each has
its own trailing stop. Entry = next bar open after a qualifying signal when a
slot is free. Opposite reading at a bar closes all positions on the wrong side.

Two risk interpretations of K>1, both reported:
  - DIVERSIFIED (metrics() / portfolio_daily): concurrent trades are equal-
    weighted -> each gets 1/K size, AGGREGATE exposure stays ~constant. This is
    the risk-sane way to run 2 positions (half size each).
  - AMPLIFIED: each position full size -> aggregate exposure = peak_concurrent x.
    Reported as peak_concurrent / peak_net_dir so the AMPLIFIED risk is visible.

Reuses verify_kernel_method_c primitives (decode_tiers rolling-percentile = live
decode, atr_wilder, _trade_pnl, FEE_RT, metrics).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from verify_kernel_method_c import (
    decode_tiers, atr_wilder, _trade_pnl, metrics, _strip_tz,
)

KLINES = ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
TRAIL_MULT = 3.0
SAFETY_CAP = 240          # hard safety cap (10d) — time_cap removed, this only
                          # bounds runaway/never-exited positions in the sim


def run(k, decoded, klines, atr, K, policy):
    idx = klines.index
    openp = klines["open"].to_numpy(float)
    high = klines["high"].to_numpy(float)
    low = klines["low"].to_numpy(float)
    close = klines["close"].to_numpy(float)
    dir_arr = decoded["direction"].reindex(idx).to_numpy()
    tier_arr = decoded["tier"].reindex(idx).to_numpy()
    atr_arr = atr.reindex(idx).to_numpy()
    n = len(idx)

    def qualifies(i):
        d, ti = dir_arr[i], tier_arr[i]
        if d == "NEUTRAL":
            return False
        return ti == "Strong" if policy == "STRONG" else ti in ("Strong", "Moderate")

    positions, pending, trades = [], [], []
    peak_concurrent = peak_net = 0

    for i in range(n):
        # 1) realize pending entries at this bar's open
        for (d, tier, a) in pending:
            ep = openp[i]
            sd = TRAIL_MULT * a
            positions.append(dict(dir=d, ei=i, ep=ep, ext=ep, sd=sd,
                                  sp=ep - sd if d == "UP" else ep + sd, tier=tier))
        pending = []

        # 2) manage exits for active positions
        still = []
        opp_here = dir_arr[i]
        for p in positions:
            d = p["dir"]
            exited = exit_px = reason = None
            if d == "UP" and low[i] <= p["sp"]:
                exited, exit_px, reason = True, p["sp"], "trail_stop"
            elif d == "DOWN" and high[i] >= p["sp"]:
                exited, exit_px, reason = True, p["sp"], "trail_stop"
            elif opp_here == ("DOWN" if d == "UP" else "UP"):
                exited, exit_px, reason = True, close[i], "opp_signal"
            elif i - p["ei"] >= SAFETY_CAP:
                exited, exit_px, reason = True, close[i], "safety_cap"
            if exited:
                g, net = _trade_pnl(d, p["ep"], exit_px)
                trades.append(dict(signal_ts=idx[p["ei"]], entry_ts=idx[p["ei"]],
                                   exit_ts=idx[i], direction=d, tier=p["tier"],
                                   entry_px=p["ep"], exit_px=exit_px,
                                   exit_reason=reason,
                                   hold_h=float(i - p["ei"]), gross=g, net=net))
            else:
                if d == "UP":
                    p["ext"] = max(p["ext"], high[i]); p["sp"] = p["ext"] - p["sd"]
                else:
                    p["ext"] = min(p["ext"], low[i]); p["sp"] = p["ext"] + p["sd"]
                still.append(p)
        positions = still

        # 3) exposure snapshot
        net = sum(1 if p["dir"] == "UP" else -1 for p in positions)
        peak_concurrent = max(peak_concurrent, len(positions))
        peak_net = max(peak_net, abs(net))

        # 4) entry if slot free
        if qualifies(i) and (len(positions) + len(pending)) < K and i + 1 < n:
            a = atr_arr[i]
            if np.isfinite(a) and a > 0:
                pending.append((dir_arr[i], tier_arr[i], a))

    tr = pd.DataFrame(trades)
    m = metrics(tr, klines) if len(tr) else {"n": 0}
    m["peak_concurrent"] = peak_concurrent
    m["peak_net_dir"] = peak_net
    return tr, m


def main():
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index); k = k[~k.index.duplicated(keep="last")].sort_index()
    v = pd.read_parquet(V71_OOS); v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    k = k.loc[v.index[0]:v.index[-1]]
    atr = atr_wilder(k["high"], k["low"], k["close"], 14)
    decoded = decode_tiers(v["pred_ret"])
    print(f"OOS {v.index[0]:%Y-%m-%d}→{v.index[-1]:%Y-%m-%d}  {len(v)} bars  "
          "(real 3xATR trail + opp-signal, time-cap OFF)\n")

    print("=" * 104)
    print("metrics() = DIVERSIFIED interpretation (concurrent trades equal-weighted = 1/K size each)")
    print("peak_concurrent / peak_net = AMPLIFIED risk if each position is full size")
    print("=" * 104)
    print(f"{'policy':10s} {'K':>2s} {'n':>4s} {'Strong':>6s} {'Mod':>4s} {'WR':>7s} "
          f"{'net/tr':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'cumNet':>8s} {'pkConc':>7s} {'pkNet':>6s}")
    print("-" * 104)
    rows = []
    for policy in ("STRONG", "BOTH"):
        for K in (1, 2):
            tr, m = run(k, decoded, k, atr, K, policy)
            ns = int((tr["tier"] == "Strong").sum()) if len(tr) else 0
            nm = int((tr["tier"] == "Moderate").sum()) if len(tr) else 0
            rows.append(dict(policy=policy, K=K, **{x: m.get(x) for x in
                        ("n", "wr", "avg_net_bps", "sharpe", "mdd_pct",
                         "cum_net_pct", "peak_concurrent", "peak_net_dir")}))
            print(f"{policy:10s} {K:>2d} {m.get('n',0):>4d} {ns:>6d} {nm:>4d} "
                  f"{m.get('wr',0)*100:>6.1f}% {m.get('avg_net_bps',0):>+7.1f} "
                  f"{m.get('sharpe',0):>7.2f} {m.get('mdd_pct',0):>6.2f}% "
                  f"{m.get('cum_net_pct',0):>+7.1f}% {m.get('peak_concurrent',0):>7d} "
                  f"{m.get('peak_net_dir',0):>6d}")
        print("-" * 104)

    pd.DataFrame(rows).to_csv(
        ROOT / "research" / "results" / "dual_model" / "entry_policy_concurrent.csv",
        index=False)
    print("\nNotes:")
    print("- DIVERSIFIED Sharpe/MaxDD: K=2 splits capital -> 2 half-size trades; tests whether")
    print("  the extra slot (less Strong crowd-out) helps WITHOUT adding exposure.")
    print("- AMPLIFIED: if each position is FULL size, peak_net same-dir positions = that many x")
    print("  aggregate exposure (e.g. peak_net=2 at 2x sizing = 4x). The per-order presubmit")
    print("  leverage guard does NOT cap aggregate -> K>1 needs a new aggregate-exposure guard")
    print("  and is a hard-rule change (max_position_count=1, Safety belt #10).")


if __name__ == "__main__":
    main()
