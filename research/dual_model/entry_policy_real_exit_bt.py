"""Entry-policy backtest under the REAL exit logic + 1-position occupancy.

Answers: under the live exit (3xATR TRAILING stop + opposite-signal exit,
optionally a fixed time cap) and max_position_count=1, does opening ONLY
Strong beat opening Strong+Moderate? The proxy sims (tier_occupancy_sim)
said yes on a fixed-hold path-return proxy; this uses the faithful exit
so the verdict is decision-grade.

Faithful to indicator/okx/executor.py:
  TRAIL_MULT=3.0, opp_signal = ANY opposite reading (Moderate or Strong),
  exit at that bar's close. Trailing stop ratchets the extreme each
  completed bar; the updated stop is active for the NEXT bar (no intrabar
  look-ahead). Entry = next bar open after the signal.

Time-cap policy (changed 2026-06-10):
  Default --time-cap 72 reproduces the original Strong-only 62% / both 53%
  comparison cited in indicator/okx/config.py.
  Use --time-cap 0 to disable the cap and reflect current live behaviour
  (cap removed per CLAUDE.md 2026-06-10 — winners run to trail/opp only).

Reuses verify_kernel_method_c primitives (decode_tiers rolling-percentile =
live decode, atr_wilder, _trade_pnl, FEE_RT, metrics) so numbers are directly
comparable with the V7.1 exit-variants report.

Policies (all under 1-position occupancy):
  STRONG      enter only on Strong tier
  BOTH        enter on Strong OR Moderate (take-first = current live behaviour)
  BOTH_PREempt enter S/M; if holding and an opposite STRONG fires, flip
  + a derived BOTH_MODHALF view (stage4 incumbent: Moderate sized 0.5)

Usage:
    python research/dual_model/entry_policy_real_exit_bt.py             # 72h cap (original)
    python research/dual_model/entry_policy_real_exit_bt.py --time-cap 0  # current live

Output: research/results/dual_model/entry_policy_real_exit.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from verify_kernel_method_c import (
    decode_tiers, atr_wilder, _trade_pnl, FEE_RT, metrics, _strip_tz,
)

KLINES = ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
V71_OOS = ROOT / "research" / "results" / "dual_model" / "direction_reg_oos_mse.parquet"
OUT = ROOT / "research" / "results" / "dual_model" / "entry_policy_real_exit.csv"

TRAIL_MULT = 3.0
DEFAULT_TIME_CAP = 72  # original comparison's cap; live is now 0 (disabled)


def run_policy(k, decoded, atr, policy: str, time_cap: int) -> pd.DataFrame:
    idx = k.index
    openp = k["open"].to_numpy(float)
    high = k["high"].to_numpy(float)
    low = k["low"].to_numpy(float)
    close = k["close"].to_numpy(float)
    dir_arr = decoded["direction"].reindex(idx).to_numpy()
    tier_arr = decoded["tier"].reindex(idx).to_numpy()
    atr_arr = atr.reindex(idx).to_numpy()
    n = len(idx)

    def qualifies(i):
        d, ti = dir_arr[i], tier_arr[i]
        if d == "NEUTRAL":
            return False
        if policy == "STRONG":
            return ti == "Strong"
        return ti in ("Strong", "Moderate")     # BOTH / BOTH_PREempt

    rows = []
    i = 0
    while i < n - 1:
        if not qualifies(i):
            i += 1
            continue
        d = dir_arr[i]
        a = atr_arr[i]
        entry_i = i + 1
        if entry_i >= n or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        entry_px = openp[entry_i]
        stop_dist = TRAIL_MULT * a
        extreme = entry_px
        stop_px = entry_px - stop_dist if d == "UP" else entry_px + stop_dist
        exit_i = exit_px = reason = None
        # When time_cap=0 (current live), let the loop scan to the data end —
        # exits then come from trail or opp signal only.
        end_j = min(entry_i + time_cap, n) if time_cap > 0 else n
        for j in range(entry_i, end_j):
            # 1) trailing stop active from prior bar's extreme — intrabar hit
            if d == "UP" and low[j] <= stop_px:
                exit_i, exit_px, reason = j, stop_px, "trail_stop"
                break
            if d == "DOWN" and high[j] >= stop_px:
                exit_i, exit_px, reason = j, stop_px, "trail_stop"
                break
            # 2) opposite signal (any opposite reading) -> exit at this close
            opp = "DOWN" if d == "UP" else "UP"
            is_opp = dir_arr[j] == opp
            if is_opp:
                # preempt-flip only switches the *next* entry; the exit itself
                # is identical (close on opposite). reason notes the tier.
                reason = ("opp_strong" if tier_arr[j] == "Strong"
                          else "opp_signal")
                exit_i, exit_px = j, close[j]
                break
            # 3) ratchet trail with this completed bar (active next bar)
            if d == "UP":
                extreme = max(extreme, high[j])
                stop_px = extreme - stop_dist
            else:
                extreme = min(extreme, low[j])
                stop_px = extreme + stop_dist
        if exit_i is None:
            # Two ways to land here:
            #   1) time_cap > 0 and the loop walked the full cap window → time_cap exit
            #   2) time_cap = 0 and we ran to the dataset end → data_end exit
            #      (flagged as not a real strategy exit)
            if time_cap > 0:
                exit_i = min(entry_i + time_cap - 1, n - 1)
                exit_px, reason = close[exit_i], "time_cap"
            else:
                exit_i = n - 1
                exit_px, reason = close[exit_i], "data_end"
        gross, net = _trade_pnl(d, entry_px, exit_px)
        rows.append(dict(
            signal_ts=idx[i], entry_ts=idx[entry_i], exit_ts=idx[exit_i],
            direction=d, tier=tier_arr[i], entry_px=entry_px, exit_px=exit_px,
            exit_reason=reason,
            hold_h=(idx[exit_i] - idx[entry_i]).total_seconds() / 3600.0,
            gross=gross, net=net,
        ))
        # occupancy: resume scanning AFTER the exit bar (slot freed)
        i = max(exit_i, entry_i)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-cap", type=int, default=DEFAULT_TIME_CAP,
                    help=("Time cap in hours. Default 72 reproduces the "
                          "original Strong-only 62%% / both 53%% comparison. "
                          "Use 0 to disable cap (current live behaviour)."))
    args = ap.parse_args()

    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    k.index = _strip_tz(k.index)
    k = k[~k.index.duplicated(keep="last")].sort_index()
    v = pd.read_parquet(V71_OOS)
    v.index = _strip_tz(v.index)
    v = v[~v.index.duplicated(keep="last")].sort_index()
    k = k.loc[v.index[0]:v.index[-1]]
    atr = atr_wilder(k["high"], k["low"], k["close"], 14)
    decoded = decode_tiers(v["pred_ret"])

    print(f"OOS span {v.index[0]:%Y-%m-%d} → {v.index[-1]:%Y-%m-%d}  ({len(v)} bars)")
    nS = int((decoded["tier"] == "Strong").sum())
    nM = int((decoded["tier"] == "Moderate").sum())
    cap_str = f"{args.time_cap}h" if args.time_cap > 0 else "disabled (live)"
    print(f"decoded signal bars: Strong={nS} Moderate={nM}  "
          f"(real live decode, rolling percentile)")
    print(f"time cap: {cap_str}\n")

    policies = ["STRONG", "BOTH", "BOTH_PREempt"]
    trades = {p: run_policy(k, decoded, atr, p, args.time_cap)
              for p in policies}
    M = {p: metrics(trades[p], k) for p in policies}

    print("=" * 104)
    print("ENTRY POLICY under REAL exit (3xATR trail + 72h cap + opp-signal) + 1-position occupancy")
    print("=" * 104)
    hdr = (f"{'policy':14s} {'n':>4s} {'Strong':>6s} {'Mod':>4s} {'WR':>7s} "
           f"{'net/tr':>8s} {'hold_h':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'cumNet':>8s}")
    print(hdr); print("-" * 104)
    for p in policies:
        tr, m = trades[p], M[p]
        ns = int((tr["tier"] == "Strong").sum()) if len(tr) else 0
        nm = int((tr["tier"] == "Moderate").sum()) if len(tr) else 0
        print(f"{p:14s} {m['n']:>4d} {ns:>6d} {nm:>4d} {m['wr']*100:>6.1f}% "
              f"{m['avg_net_bps']:>+7.1f} {m['avg_hold_h']:>7.1f} "
              f"{m['sharpe']:>7.2f} {m['mdd_pct']:>6.2f}% {m['cum_net_pct']:>+7.1f}%")

    # derived: BOTH but Moderate sized 0.5 (stage4 incumbent) — scale Moderate
    # trades' return; summed/cum only (Sharpe approx, so omit).
    tb = trades["BOTH"].copy()
    w = np.where(tb["tier"].to_numpy() == "Moderate", 0.5, 1.0)
    sized_net = tb["net"].to_numpy() * w
    eq = (1.0 + pd.Series(sized_net)).prod() - 1.0
    print("-" * 104)
    print(f"{'BOTH_MODHALF':14s} (derived) n={len(tb)}  "
          f"sized summed-net={sized_net.sum()*100:+.1f}%  compounded={eq*100:+.1f}%  "
          f"[stage4 incumbent: Strong 1.0 / Moderate 0.5]")

    # tier WR within BOTH (how good are the Moderate trades actually?)
    print("\nWithin BOTH — quality by tier (real exit):")
    for ti in ("Strong", "Moderate"):
        sub = trades["BOTH"][trades["BOTH"]["tier"] == ti]
        if len(sub):
            mm = metrics(sub, k)
            print(f"  {ti:9s} n={mm['n']:>3d}  WR={mm['wr']*100:.1f}%  "
                  f"net/tr={mm['avg_net_bps']:+.1f}bps  hold={mm['avg_hold_h']:.1f}h")

    # exit-reason mix per policy
    print("\nExit-reason mix:")
    for p in policies:
        tr = trades[p]
        if not len(tr):
            continue
        parts = [f"{r} {len(g)}({len(g)/len(tr)*100:.0f}%,{g['net'].mean()*1e4:+.0f}bps)"
                 for r, g in tr.groupby("exit_reason")]
        print(f"  {p:14s}: " + "  ".join(parts))

    allrows = []
    for p in policies:
        m = dict(M[p]); m["policy"] = p; allrows.append(m)
    pd.DataFrame(allrows).to_csv(OUT, index=False)
    print(f"\nWrote → {OUT}")

    # ---- per-month robustness: is STRONG's win stable or one-month-driven? ----
    print("\nPer-month robustness (net summed %, by entry month):")
    print(f"  {'month':9s} {'STRONG':>9s} {'BOTH':>9s} {'winner':>8s}")
    ts = trades["STRONG"].copy(); tb2 = trades["BOTH"].copy()
    ts["m"] = pd.to_datetime(ts["entry_ts"]).dt.to_period("M")
    tb2["m"] = pd.to_datetime(tb2["entry_ts"]).dt.to_period("M")
    months = sorted(set(ts["m"]) | set(tb2["m"]))
    strong_wins = 0
    for mo in months:
        sret = ts.loc[ts["m"] == mo, "net"].sum() * 100
        bret = tb2.loc[tb2["m"] == mo, "net"].sum() * 100
        win = "STRONG" if sret > bret else "BOTH"
        strong_wins += int(sret > bret)
        print(f"  {str(mo):9s} {sret:>+8.1f}% {bret:>+8.1f}% {win:>8s}")
    print(f"  STRONG beats BOTH in {strong_wins}/{len(months)} months")

    # verdict
    s, b = M["STRONG"], M["BOTH"]
    print("\n" + "=" * 104)
    print("VERDICT (real exit + occupancy)")
    print("=" * 104)
    print(f"  STRONG-only vs BOTH:  WR {s['wr']*100:.1f}% vs {b['wr']*100:.1f}%  |  "
          f"net/tr {s['avg_net_bps']:+.1f} vs {b['avg_net_bps']:+.1f}bps  |  "
          f"Sharpe {s['sharpe']:.2f} vs {b['sharpe']:.2f}  |  "
          f"MaxDD {s['mdd_pct']:.2f}% vs {b['mdd_pct']:.2f}%  |  "
          f"cum {s['cum_net_pct']:+.1f}% vs {b['cum_net_pct']:+.1f}%")


if __name__ == "__main__":
    main()
