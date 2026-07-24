"""
SHORT-only isolation test (2026-07-24).

Question from chat: the backtest has always shown SHORT (+115bps/70%WR) far
outperforming LONG (+23bps/46%WR) when the SAME both-directions strategy's
trades are split post-hoc by side. But that split was never turned into an
actual standalone strategy — LONG signals were still occupying the single
position slot (max_position_count=1) and blocking SHORT entries that would
have fired if LONG were disabled entirely. This script builds the REAL
short-only bot (LONG signals forced to NEUTRAL, not just filtered out of the
trade log after the fact) and tests it with the same rigor as every other
decision this project makes: bootstrap WR CI + first/second-half consistency
— not just a point estimate.

Reuses research/v71_v7_sizing_1x.py's load_data/decode_signals/simulate/
summarize UNCHANGED (this is the trusted, walk-forward-OOS, no-leakage
backtest harness that reconstructs V7.1's exact production logic) — this
script only adds the direction-masking + WR-CI + half-split layer on top.

Run: python research/short_only_isolation_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import research.v71_v7_sizing_1x as bt


def wr_bootstrap_ci(wins: np.ndarray, n_iter: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI on win rate (resample trade outcomes, not equity paths —
    this is the "is WR really > 50%" question, separate from bootstrap_trades'
    ROI/MDD CI in v71_v7_sizing_1x.py)."""
    rng = np.random.default_rng(seed)
    n = len(wins)
    boot_wr = np.array([wins[rng.integers(0, n, n)].mean() for _ in range(n_iter)])
    return (float(wins.mean()), float(np.percentile(boot_wr, 2.5)), float(np.percentile(boot_wr, 97.5)))


def half_split_wr(trades: pd.DataFrame) -> tuple[float, float]:
    """First-half vs second-half WR by entry time — same 'halves agree'
    discipline used throughout this project's cancel-flow research."""
    t = trades.sort_values("entry_ts")
    mid = len(t) // 2
    if mid < 3:
        return (float("nan"), float("nan"))
    return (float(t.iloc[:mid]["win"].mean()), float(t.iloc[mid:]["win"].mean()))


def run_variant(df, direction_masked, tier, label, span_days, entry_mode="signal_close"):
    trades = bt.simulate(df, direction_masked, tier, entry_mode=entry_mode)
    if trades.empty:
        print(f"\n  --- {label}: NO TRADES ---")
        return None
    s = bt.summarize(trades, span_days)
    wr, wr_lo, wr_hi = wr_bootstrap_ci(trades["win"].values)
    boot = bt.bootstrap_trades(trades, n_iter=5000)
    h1, h2 = half_split_wr(trades)

    print(f"\n  --- {label} ---")
    print(f"    n={s['n']}  WR={s['wr_pct']:.1f}%  "
          f"WR 95% CI=[{wr_lo*100:.1f}%, {wr_hi*100:.1f}%]")
    print(f"    avg_net_bps={s['avg_net_bps']:+.1f}  ROI={s['roi_pct']:+.1f}%  "
          f"MDD={s['mdd_pct']:.1f}%  Sharpe(cal)={s['sharpe_calendar_ann']:.2f}")
    print(f"    ROI bootstrap: p5={boot['roi_p5']:+.1f}% p50={boot['roi_p50']:+.1f}% "
          f"p95={boot['roi_p95']:+.1f}%  P(profitable)={boot['p_profitable']*100:.0f}%")
    print(f"    half-split WR: first={h1*100:.1f}%  second={h2*100:.1f}%  "
          f"agree={'YES' if (h1 - 0.5) * (h2 - 0.5) > 0 else 'NO — SIGN DISAGREES'}")
    return dict(label=label, n=s['n'], wr=s['wr_pct'], wr_ci=(wr_lo, wr_hi),
                avg_net_bps=s['avg_net_bps'], roi=s['roi_pct'], mdd=s['mdd_pct'],
                sharpe=s['sharpe_calendar_ann'], p_profitable=boot['p_profitable'],
                half1=h1, half2=h2)


def main():
    print("=" * 72)
    print("  SHORT-ONLY ISOLATION TEST — is the SHORT-side edge real when")
    print("  actually traded standalone (not just post-hoc trade-log split)?")
    print("=" * 72)

    df = bt.load_data()
    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400.0
    direction, tier, _ = bt.decode_signals(df)
    direction = np.asarray(direction, dtype=object)
    tier = np.asarray(tier, dtype=object)

    print(f"  OOS span: {df.index.min()} -> {df.index.max()}  ({span_days:.0f} days)")

    # Variant A: current production behaviour (both sides)
    results = []
    results.append(run_variant(df, direction, tier, "ALL (current, both LONG+SHORT)", span_days))

    # Variant B: SHORT-only — LONG (UP) signals masked to NEUTRAL at the
    # SIGNAL level (not filtered post-hoc), so SHORT entries can use position
    # slots that would otherwise be occupied by a LONG trade.
    direction_short_only = direction.copy()
    direction_short_only[direction_short_only == "UP"] = "NEUTRAL"
    results.append(run_variant(df, direction_short_only, tier, "SHORT-ONLY (LONG signals masked)", span_days))

    # Variant C: LONG-only, for symmetry/comparison
    direction_long_only = direction.copy()
    direction_long_only[direction_long_only == "DOWN"] = "NEUTRAL"
    results.append(run_variant(df, direction_long_only, tier, "LONG-ONLY (SHORT signals masked)", span_days))

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for r in results:
        if r is None:
            continue
        gate_pass = r['wr_ci'][0] > 0.52 and (r['half1'] - 0.5) * (r['half2'] - 0.5) > 0
        print(f"  {r['label']:35s} n={r['n']:4d}  WR={r['wr']:5.1f}%  "
              f"CI_lo={r['wr_ci'][0]*100:5.1f}%  P(profit)={r['p_profitable']*100:5.1f}%  "
              f"=> {'PASS (CI_lo>52%, halves agree)' if gate_pass else 'FAIL/inconclusive'}")


if __name__ == "__main__":
    main()
