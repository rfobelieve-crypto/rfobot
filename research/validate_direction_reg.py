"""
Rigorous validation battery for direction-regression WF OOS predictions.

Addresses the mistake-log lessons:
  - Never trust in-sample: all metrics come from direction_reg_oos_*.parquet
    (true walk-forward, 77 folds, purge=4, embargo=4)
  - Never trust Wilson CI as the only uncertainty check — bars within a fold
    are autocorrelated, so we do fold-level block bootstrap as well
  - Never trust an overall number without per-fold and time-slice breakdown

Checks:
  V1  Block bootstrap CI for top-5% rolling WR
  V2  Per-fold WR distribution (quartiles + worst folds)
  V3  Long vs Short asymmetry (with independent CIs)
  V4  Time-slice holdout: first half vs second half
  V6  Feature importance top-15 (visual sanity)
  V7  Monthly WR with Wilson CI lower bound

Usage:
    python -m research.validate_direction_reg --objective mse
"""
from __future__ import annotations

import sys
import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import RESULTS_DIR


ROLL_WINDOW = 500
STRONG_FRAC = 0.05


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def rolling_strong_signals(pred, y, window=ROLL_WINDOW, strong_frac=STRONG_FRAC):
    """Compute rolling top-strong_frac symmetric fires.

    Returns arrays of:
      fire (bool), win (bool), side (+1 long / -1 short / 0 none)
    """
    q_up_s = pd.Series(pred).rolling(window, min_periods=100).quantile(
        1 - strong_frac / 2).values
    q_dn_s = pd.Series(pred).rolling(window, min_periods=100).quantile(
        strong_frac / 2).values
    valid = ~np.isnan(q_up_s)

    long_fire = valid & (pred >= q_up_s)
    short_fire = valid & (pred <= q_dn_s)
    fire = long_fire | short_fire
    win = (long_fire & (y > 0)) | (short_fire & (y < 0))
    side = np.where(long_fire, 1, np.where(short_fire, -1, 0))
    return fire, win, side


def V1_block_bootstrap(oos, n_boot=5000, seed=42):
    """Block bootstrap by fold. Each resample picks N folds with replacement,
    pools their bars, and computes WR. This respects within-fold autocorrelation."""
    print("\n== V1: Block bootstrap CI on rolling top-5% WR ==")
    pred = oos["pred_ret"].values
    y = oos["y_path_ret_4h"].values
    fire, win, _ = rolling_strong_signals(pred, y)

    fold_ids = oos["fold"].values
    unique_folds = np.unique(fold_ids)

    # Point estimate
    n_pt = int(fire.sum())
    w_pt = int(win.sum())
    wr_pt = w_pt / n_pt if n_pt else 0
    print(f"  point: n={n_pt}  wins={w_pt}  WR={wr_pt * 100:.1f}%")

    rng = np.random.default_rng(seed)
    wrs = []
    n_folds = len(unique_folds)
    for _ in range(n_boot):
        pick = rng.choice(unique_folds, size=n_folds, replace=True)
        # Gather bars from picked folds (sum over picks, not unique)
        agg_fire = 0
        agg_win = 0
        for f in pick:
            mask = fold_ids == f
            agg_fire += int(fire[mask].sum())
            agg_win += int(win[mask].sum())
        if agg_fire:
            wrs.append(agg_win / agg_fire)
    wrs = np.array(wrs)
    lo, hi = np.quantile(wrs, [0.025, 0.975])
    median = np.quantile(wrs, 0.5)
    print(f"  block-bootstrap 95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]  "
          f"(median {median * 100:.1f}%, n_boot={n_boot})")
    p_below_60 = float((wrs < 0.60).mean())
    print(f"  P(WR < 60%) = {p_below_60:.3f}  "
          f"{'<-- RISK' if p_below_60 > 0.10 else '<-- OK'}")
    return wr_pt, (lo, hi)


def V2_per_fold_stability(oos):
    print("\n== V2: Per-fold WR stability ==")
    pred = oos["pred_ret"].values
    y = oos["y_path_ret_4h"].values
    fire, win, _ = rolling_strong_signals(pred, y)

    df = pd.DataFrame({"fold": oos["fold"].values, "fire": fire, "win": win})
    gb = df.groupby("fold").agg(n_fire=("fire", "sum"),
                                 n_win=("win", "sum"))
    gb = gb[gb["n_fire"] > 0]
    gb["wr"] = gb["n_win"] / gb["n_fire"]
    print(f"  folds with >=1 fire: {len(gb)}/{oos['fold'].nunique()}")
    print(f"  fold-level WR distribution (n >= 3):")
    gb3 = gb[gb["n_fire"] >= 3]
    if len(gb3):
        q = np.quantile(gb3["wr"], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"    min  = {q[0]*100:5.1f}%")
        print(f"    10%  = {q[1]*100:5.1f}%")
        print(f"    25%  = {q[2]*100:5.1f}%")
        print(f"    50%  = {q[3]*100:5.1f}%  <-- median")
        print(f"    75%  = {q[4]*100:5.1f}%")
        print(f"    90%  = {q[5]*100:5.1f}%")
        print(f"    max  = {q[6]*100:5.1f}%")
        bad = gb3[gb3["wr"] < 0.50]
        print(f"  folds with WR < 50%: {len(bad)}/{len(gb3)}")
        print(f"  folds with WR >= 60%: "
              f"{int((gb3['wr'] >= 0.60).sum())}/{len(gb3)}")


def V3_long_vs_short(oos):
    print("\n== V3: Long vs Short asymmetry ==")
    pred = oos["pred_ret"].values
    y = oos["y_path_ret_4h"].values
    fire, win, side = rolling_strong_signals(pred, y)

    for name, sign in [("Long", 1), ("Short", -1)]:
        m = side == sign
        n = int(m.sum())
        if n == 0:
            continue
        w = int((m & win).sum())
        wr = w / n
        lo, hi = wilson_ci(w, n)
        print(f"  {name:<6} n={n:4d}  W={w:4d}  WR={wr*100:5.1f}%  "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")

    # Base rate of realized ret>0 in the whole OOS
    base_up = float((y > 0).mean())
    print(f"  (realized base rate P(y>0) = {base_up*100:.1f}%)")


def V4_time_slice(oos):
    print("\n== V4: Time-slice first-half vs second-half ==")
    # Split by chronological order — oos is already in time order from WF
    n = len(oos)
    half = n // 2
    for name, sl in [("first half", slice(0, half)),
                      ("second half", slice(half, n))]:
        sub = oos.iloc[sl]
        pred = sub["pred_ret"].values
        y = sub["y_path_ret_4h"].values
        fire, win, _ = rolling_strong_signals(pred, y)
        n_f = int(fire.sum())
        n_w = int(win.sum())
        if n_f == 0:
            print(f"  {name}: no fires")
            continue
        wr = n_w / n_f
        lo, hi = wilson_ci(n_w, n_f)
        span = f"{sub.index[0]} -> {sub.index[-1]}"
        print(f"  {name:<13} {span}")
        print(f"    n_fire={n_f:4d}  wins={n_w:4d}  WR={wr*100:5.1f}%  "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")


def V6_feature_importance(objective):
    print("\n== V6: Top-15 feature importance sanity ==")
    path = RESULTS_DIR / f"direction_reg_importance_{objective}.csv"
    if not path.exists():
        print(f"  {path} not found")
        return
    imp = pd.read_csv(path).head(15)
    print(imp.to_string(index=False,
                         float_format=lambda x: f"{x:.4f}"))


def V7_monthly_wilson(oos):
    print("\n== V7: Monthly WR with Wilson CI ==")
    pred = oos["pred_ret"].values
    y = oos["y_path_ret_4h"].values
    fire, win, _ = rolling_strong_signals(pred, y)
    df = pd.DataFrame({"fire": fire, "win": win}, index=oos.index)
    df["month"] = pd.DatetimeIndex(df.index).strftime("%Y-%m")
    print(f"  {'month':<10}{'fires':>7}{'wins':>7}{'WR':>10}{'CI':>24}"
          f"{'flag':>8}")
    for m, g in df.groupby("month"):
        f = int(g["fire"].sum())
        if f == 0:
            continue
        w = int(g["win"].sum())
        wr = w / f
        lo, hi = wilson_ci(w, f)
        flag = ""
        if lo < 0.50:
            flag = "LOW"
        elif wr < 0.60:
            flag = "sub60"
        print(f"  {m:<10}{f:>7d}{w:>7d}{wr*100:>9.1f}%"
              f"  [{lo*100:5.1f}%, {hi*100:5.1f}%]{flag:>8}")


def main(objective):
    path = RESULTS_DIR / f"direction_reg_oos_{objective}.parquet"
    oos = pd.read_parquet(path).sort_index()

    print("=" * 76)
    print(f"  DIRECTION-REG VALIDATION BATTERY  objective={objective}")
    print(f"  OOS bars: {len(oos)}  folds: {oos['fold'].nunique()}")
    print(f"  window={ROLL_WINDOW}  strong_frac={STRONG_FRAC}")
    print("=" * 76)

    V1_block_bootstrap(oos)
    V2_per_fold_stability(oos)
    V3_long_vs_short(oos)
    V4_time_slice(oos)
    V6_feature_importance(objective)
    V7_monthly_wilson(oos)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--objective", default="mse", choices=["mse", "huber"])
    args = p.parse_args()
    main(args.objective)
