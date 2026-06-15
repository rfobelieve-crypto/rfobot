"""Asymmetric LONG/SHORT threshold scan on canonical V7 OOS predictions.

Hypothesis (per [[project_up_down_asymmetry]] memory + V7 paper 13d data):
DOWN signals reach higher WR than UP at same |pred|.  So instead of
symmetric "|pred| >= 0.0008 → Strong", use:
    LONG  signal if pred >  +thr_long
    SHORT signal if pred < -thr_short

with thr_short < thr_long (lower bar for shorts since they're more reliable).

Method:
  1. Load 3696 OOS predictions
  2. For each (thr_long, thr_short) grid, compute:
     - LONG WR, n_long, fire_rate
     - SHORT WR, n_short, fire_rate
     - Combined WR + total signals
     - Strong-tier combined WR
  3. Pick Pareto-optimal: highest combined WR with ≥ certain n threshold

Output:
  research/results/asymmetric_threshold_grid.csv
  research/results/asymmetric_threshold_recommendation.json

NO retraining.  Pure thresholding optimization on existing OOS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OOS_PATH = PROJECT_ROOT / "research" / "results" / "dual_model" \
                       / "direction_reg_oos_mse.parquet"
GRID_OUT = PROJECT_ROOT / "research" / "results" \
                        / "asymmetric_threshold_grid.csv"
REC_OUT = PROJECT_ROOT / "research" / "results" \
                       / "asymmetric_threshold_recommendation.json"


def side_metrics(df: pd.DataFrame, thr_long: float,
                  thr_short: float) -> dict:
    long_sig = df[df["pred_ret"] > thr_long]
    short_sig = df[df["pred_ret"] < -thr_short]
    n_long, n_short = len(long_sig), len(short_sig)
    n_total = n_long + n_short
    if n_long > 0:
        long_wr = (long_sig["y_path_ret_4h"] > 0).mean()
    else:
        long_wr = np.nan
    if n_short > 0:
        short_wr = (short_sig["y_path_ret_4h"] < 0).mean()
    else:
        short_wr = np.nan
    if n_total > 0:
        long_correct = ((long_sig["y_path_ret_4h"] > 0).sum() if n_long
                         else 0)
        short_correct = ((short_sig["y_path_ret_4h"] < 0).sum() if n_short
                          else 0)
        combined_wr = (long_correct + short_correct) / n_total
    else:
        combined_wr = np.nan
    fire_rate = n_total / len(df)
    return {
        "thr_long": thr_long, "thr_short": thr_short,
        "n_long": n_long, "n_short": n_short, "n_total": n_total,
        "long_wr": long_wr, "short_wr": short_wr,
        "combined_wr": combined_wr,
        "fire_rate": fire_rate,
    }


def main() -> int:
    df = pd.read_parquet(OOS_PATH)
    print(f"Loaded {len(df)} OOS predictions, "
          f"pred range [{df['pred_ret'].min():+.5f}, "
          f"{df['pred_ret'].max():+.5f}]")

    # ── Step 1: Symmetric baseline (current production behavior)
    print()
    print("=" * 70)
    print("STEP 1: Symmetric baseline (current production-style)")
    print("=" * 70)
    print(f"{'thr':>10s}  {'LONG':>20s}  {'SHORT':>20s}  {'Combined':>16s}")
    print(f"{'':>10s}  {'n / wr':>20s}  {'n / wr':>20s}  {'n / wr':>16s}")
    print("-" * 70)
    for thr in (0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003):
        m = side_metrics(df, thr, thr)
        print(f"{thr:10.4f}  "
              f"{m['n_long']:6d} / {m['long_wr']*100 if not np.isnan(m['long_wr']) else 0:5.1f}%  "
              f"{m['n_short']:6d} / {m['short_wr']*100 if not np.isnan(m['short_wr']) else 0:5.1f}%  "
              f"{m['n_total']:5d} / {m['combined_wr']*100 if not np.isnan(m['combined_wr']) else 0:5.1f}%")

    # ── Step 2: Per-side scan independently
    print()
    print("=" * 70)
    print("STEP 2: Per-side scan — what does each direction need?")
    print("=" * 70)
    print("LONG side:")
    print(f"  {'thr':>10s}  {'n':>5s}  {'WR':>6s}  {'fire':>6s}")
    for thr in (0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003):
        sig = df[df["pred_ret"] > thr]
        if len(sig) == 0:
            continue
        wr = (sig["y_path_ret_4h"] > 0).mean()
        print(f"  {thr:10.4f}  {len(sig):5d}  {wr*100:5.1f}%  "
              f"{len(sig)/len(df)*100:5.1f}%")
    print("SHORT side:")
    for thr in (0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003):
        sig = df[df["pred_ret"] < -thr]
        if len(sig) == 0:
            continue
        wr = (sig["y_path_ret_4h"] < 0).mean()
        print(f"  {thr:10.4f}  {len(sig):5d}  {wr*100:5.1f}%  "
              f"{len(sig)/len(df)*100:5.1f}%")

    # ── Step 3: Asymmetric grid scan
    print()
    print("=" * 70)
    print("STEP 3: Asymmetric grid scan")
    print("=" * 70)
    thrs = [0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025,
            0.003]
    rows = []
    for tL in thrs:
        for tS in thrs:
            m = side_metrics(df, tL, tS)
            rows.append(m)
    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)

    # Filter to grid points with reasonable signal counts (≥ 50 trades over 5 mo)
    filtered = grid[grid["n_total"] >= 50].copy()
    if filtered.empty:
        print("WARN: no grid points have n_total >= 50")
        return 1

    # Top 5 by combined_wr
    top = filtered.sort_values("combined_wr", ascending=False).head(8)
    print()
    print("Top asymmetric configurations (n_total >= 50, sorted by combined_wr):")
    print(f"  {'thr_L':>8s}  {'thr_S':>8s}  {'n_L':>5s}  {'n_S':>5s}  "
          f"{'WR_L':>6s}  {'WR_S':>6s}  {'WR':>6s}  {'fire':>6s}")
    for _, r in top.iterrows():
        print(f"  {r['thr_long']:8.4f}  {r['thr_short']:8.4f}  "
              f"{int(r['n_long']):5d}  {int(r['n_short']):5d}  "
              f"{r['long_wr']*100:5.1f}%  {r['short_wr']*100:5.1f}%  "
              f"{r['combined_wr']*100:5.1f}%  "
              f"{r['fire_rate']*100:5.1f}%")

    # ── Step 4: Compare to current production threshold
    # Current production uses |pred| >= 0.0008 for Strong floor (per
    # CLAUDE.md absolute floor). Find symmetric baseline at 0.0008
    print()
    print("=" * 70)
    print("STEP 4: Recommendation")
    print("=" * 70)
    base = side_metrics(df, 0.0008, 0.0008)
    print(f"Current production (symmetric 0.0008):")
    print(f"  combined WR={base['combined_wr']*100:.1f}%  "
          f"n_total={base['n_total']}  "
          f"fire_rate={base['fire_rate']*100:.1f}%")

    # Pick best asymmetric where n_total is similar (within 20% of baseline)
    target_n = base["n_total"]
    similar_n = filtered[(filtered["n_total"] >= target_n * 0.8)
                         & (filtered["n_total"] <= target_n * 1.2)].copy()
    if not similar_n.empty:
        best = similar_n.sort_values("combined_wr", ascending=False).iloc[0]
        print()
        print(f"Best asymmetric at similar fire rate:")
        print(f"  thr_long={best['thr_long']:.4f}  "
              f"thr_short={best['thr_short']:.4f}")
        print(f"  combined WR={best['combined_wr']*100:.1f}%  "
              f"n_total={int(best['n_total'])}  "
              f"fire_rate={best['fire_rate']*100:.1f}%")
        lift = (best['combined_wr'] - base['combined_wr']) * 100
        print(f"  Lift: {lift:+.1f}pp WR  ({'WORTH IT' if lift >= 2 else 'marginal'})")

        recommendation = {
            "baseline_symmetric_thr": 0.0008,
            "baseline_combined_wr": float(base['combined_wr']),
            "baseline_n_total": int(base['n_total']),
            "recommended_thr_long": float(best['thr_long']),
            "recommended_thr_short": float(best['thr_short']),
            "recommended_combined_wr": float(best['combined_wr']),
            "recommended_n_total": int(best['n_total']),
            "wr_lift_pp": float(lift),
            "deploy_recommendation": ("YES" if lift >= 2 else "marginal — needs more data"),
        }
        REC_OUT.write_text(json.dumps(recommendation, indent=2))
        print()
        print(f"Wrote {REC_OUT.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
