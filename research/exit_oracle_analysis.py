"""Exit oracle analysis — Phase 0 ROI test for an ML exit model.

Roadmap §第二步. For every BASELINE trade, find the best exit point inside
its actual hold window and measure the gap vs the realised exit:

    gap = oracle_net - actual_net   (bps)

CRITICAL METHODOLOGY NOTE
-------------------------
Perfect MFE (max-favorable-excursion) is a LOOK-AHEAD upper bound — it knows
the future peak, so NO causal model can reach it. The perfect gap is inflated
by construction; quoting it alone would overstate ML's room (the same
over-extension trap as treating a univariate IC as deployable lift).

So we report TWO ceilings:
  1. PERFECT  — exit exactly at MFE bar (look-ahead, theoretical ceiling)
  2. CAUSAL   — exit L bars after the running favorable peak STOPS making new
                extremes (only uses past/current bars; a realistic proxy for
                what an exit model could actually learn: "momentum stalled, get out")

Plus diagnostics to tell "structurally capturable" from "unrepeatable noise":
  - capture ratio = realised_favorable / MFE_favorable
  - give-back     = MFE_favorable - realised_favorable (bps left on the table)
  - by exit_reason breakdown (is the gap concentrated in trail_stop?)
  - MFE timing within the hold window

DECISION (roadmap §第二步) — applied to the CAUSAL gap, not the perfect one:
  causal mean gap > 30 bps → ML exit has ROI → proceed to Phase 1
  causal mean gap < 10 bps → exit already near-optimal → problem is ENTRY
  10–30 bps                → marginal, weigh the trade-off

Caveats: same 5.5-mo WF-OOS window + same small N (~75 trades) as
exit_variants_backtest. This is a diagnostic, NOT a deploy decision.

Usage:
    python -m research.exit_oracle_analysis
    python -m research.exit_oracle_analysis --causal-lag 3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from research.v71_v7_sizing_1x import (
    ATR_PERIOD, TAKER_COST, load_data, decode_signals, _atr_wilder,
)
from research.exit_variants_backtest import simulate_with_policy, variants_catalog


def _causal_exit_price(side, entry_price, highs, lows, closes, lag):
    """Exit `lag` bars after the running favorable extreme stops improving.

    Pure causal: only looks at bars up to the current one. Returns the close
    at the stall-confirmation bar, or None if the peak never stalled for `lag`
    consecutive bars (then the realised exit stands).
    """
    n = len(closes)
    if side == "LONG":
        running = entry_price
    else:
        running = entry_price
    stall = 0
    for j in range(n):
        if side == "LONG":
            if highs[j] > running:
                running = highs[j]
                stall = 0
            else:
                stall += 1
        else:  # SHORT — favorable is DOWN
            if lows[j] < running:
                running = lows[j]
                stall = 0
            else:
                stall += 1
        if stall >= lag:
            return closes[j]
    return None


def analyse(df, trades, causal_lag):
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    idx = df.index
    cost = TAKER_COST

    recs = []
    for _, t in trades.iterrows():
        i0 = idx.get_loc(t["entry_ts"])
        i1 = idx.get_loc(t["exit_ts"])
        if isinstance(i0, slice) or isinstance(i1, slice):
            continue
        side = t["side"]
        ep = t["entry_price"]
        win_h = h[i0:i1 + 1]
        win_lo = lo[i0:i1 + 1]
        win_c = c[i0:i1 + 1]
        wlen = len(win_c)
        if wlen == 0:
            continue

        # ── Perfect MFE ──
        if side == "LONG":
            mfe_price = win_h.max()
            mfe_pos = int(win_h.argmax())
            mfe_fav = mfe_price / ep - 1.0
        else:
            mfe_price = win_lo.min()
            mfe_pos = int(win_lo.argmin())
            mfe_fav = -(mfe_price / ep - 1.0)
        perfect_net = mfe_fav - cost

        # ── Causal stall exit ──
        cx = _causal_exit_price(side, ep, win_h, win_lo, win_c, causal_lag)
        if cx is None:
            causal_net = t["net_pct"]          # never stalled → realised stands
        else:
            cg = (cx / ep - 1.0) if side == "LONG" else -(cx / ep - 1.0)
            causal_net = cg - cost

        actual_net = float(t["net_pct"])
        actual_gross = float(t["gross_pct"])
        recs.append(dict(
            exit_reason=t["exit_reason"], side=side, win_bars=wlen,
            actual_net=actual_net, actual_gross=actual_gross,
            mfe_fav=mfe_fav, mfe_pos_frac=(mfe_pos / max(wlen - 1, 1)),
            perfect_net=perfect_net,
            causal_net=causal_net,
            perfect_gap=(perfect_net - actual_net) * 1e4,
            causal_gap=(causal_net - actual_net) * 1e4,
            giveback=(mfe_fav - actual_gross) * 1e4,
            capture=(actual_gross / mfe_fav) if mfe_fav > 1e-9 else np.nan,
        ))
    return pd.DataFrame(recs)


def report(r, causal_lag):
    n = len(r)
    print(f"\nBaseline trades analysed: {n}  (causal lag = {causal_lag} bars)")
    print("=" * 78)

    pg, cg = r["perfect_gap"], r["causal_gap"]
    print(f"{'':22}{'mean':>9}{'median':>9}{'p75':>9}{'p95':>9}")
    print(f"{'PERFECT gap (bps)':22}{pg.mean():>9.1f}{pg.median():>9.1f}"
          f"{pg.quantile(.75):>9.1f}{pg.quantile(.95):>9.1f}   ← look-ahead UB")
    print(f"{'CAUSAL gap (bps)':22}{cg.mean():>9.1f}{cg.median():>9.1f}"
          f"{cg.quantile(.75):>9.1f}{cg.quantile(.95):>9.1f}   ← realistic ceiling")
    print()
    cap = r["capture"].dropna()
    print(f"Capture ratio (realised_fav / MFE_fav): "
          f"mean={cap.mean():.2f}  median={cap.median():.2f}  "
          f"(1.0 = perfect, already kept the whole move)")
    print(f"Give-back (MFE_fav - realised, bps):    "
          f"mean={r['giveback'].mean():+.1f}  median={r['giveback'].median():+.1f}")
    print(f"MFE position in hold window (0=entry,1=exit): "
          f"mean={r['mfe_pos_frac'].mean():.2f}  median={r['mfe_pos_frac'].median():.2f}")
    frac_causal_pos = (cg > 0).mean() * 100
    print(f"Trades where causal exit beats realised: {frac_causal_pos:.0f}%")

    print("\nBy exit_reason (causal gap):")
    print(f"  {'reason':<12}{'n':>4}{'caus_gap':>10}{'perf_gap':>10}"
          f"{'capture':>9}{'giveback':>10}")
    for reason, g in r.groupby("exit_reason"):
        capr = g["capture"].dropna()
        print(f"  {reason:<12}{len(g):>4}{g['causal_gap'].mean():>10.1f}"
              f"{g['perfect_gap'].mean():>10.1f}"
              f"{(capr.mean() if len(capr) else float('nan')):>9.2f}"
              f"{g['giveback'].mean():>+10.1f}")

    print("\n" + "=" * 78)
    cgm = cg.mean()
    print(f"VERDICT (on CAUSAL mean gap = {cgm:+.1f} bps):")
    if cgm > 30:
        print("  > +30 bps → ML exit model has REAL ROI. Proceed to Phase 1.")
        print("  A causal stall-detector already beats realised by a wide margin,")
        print("  so a learned model has structural room (not just look-ahead noise).")
    elif cgm < 10:
        print("  < +10 bps → exit is already near-optimal. The capturable room is")
        print("  tiny once you remove look-ahead. ML exit would chase noise; the")
        print("  bottleneck is ENTRY edge, not exit. Do NOT build the ML exit model.")
    else:
        print("  10–30 bps → MARGINAL. A causal model could recover some, but the")
        print("  trade-off (4-week advisory, retrain cost) may not clear the bar.")
        print("  Lean NO unless give-back is concentrated in one fixable exit_reason.")
    print(f"\n  (Perfect look-ahead gap was {pg.mean():+.1f} bps — do NOT use this")
    print("   number for the decision; it is unreachable by construction.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal-lag", type=int, default=2,
                    help="bars the favorable peak must stall before causal exit")
    ap.add_argument("--entry-mode", default="signal_close",
                    choices=["signal_close", "next_open"])
    args = ap.parse_args()

    print("Loading OOS data + decoding signals...")
    df = load_data()
    df["atr"] = _atr_wilder(df, ATR_PERIOD)
    direction, tier, warmup_n = decode_signals(df)
    df = df.iloc[warmup_n:].copy()
    direction = direction[warmup_n:]
    tier = tier[warmup_n:]
    span = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    print(f"OOS window: {df.index[0]} → {df.index[-1]}  ({span:.1f} days)")

    baseline = variants_catalog()["baseline"]
    trades = simulate_with_policy(df, direction, tier, baseline,
                                  entry_mode=args.entry_mode)
    print(f"Baseline produced {len(trades)} trades.")
    r = analyse(df, trades, args.causal_lag)
    report(r, args.causal_lag)


if __name__ == "__main__":
    main()
