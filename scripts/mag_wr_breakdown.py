"""
How does Mag-model percentile interact with Direction-Reg Strong WR?

Joins direction_reg_oos_mse.parquet (77-fold WF) with
magnitude_oos_phase1_voladj.parquet (77-fold WF, same walk-forward split)
and computes:
  1. Mag model's own standalone evaluation (IC, decile ratios)
  2. Direction Strong WR broken down by Mag decile
     (answers: "when Mag says big move, is Direction more accurate?")
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROLL = 500
STRONG_FRAC = 0.05


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def rolling_top5(score, y):
    q_up = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        1 - STRONG_FRAC / 2).values
    q_dn = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        STRONG_FRAC / 2).values
    valid = ~np.isnan(q_up)
    long_fire = valid & (score >= q_up)
    short_fire = valid & (score <= q_dn)
    fire = long_fire | short_fire
    win = (long_fire & (y > 0)) | (short_fire & (y < 0))
    return fire, win


def main():
    dir_oos = pd.read_parquet(
        ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"
    ).sort_index()
    mag_oos = pd.read_parquet(
        ROOT / "research/results/dual_model/magnitude_oos_phase1_voladj.parquet"
    ).sort_index()

    print(f"Dir OOS: {len(dir_oos)} bars, {dir_oos['fold'].nunique()} folds")
    print(f"Mag OOS: {len(mag_oos)} bars, cols={list(mag_oos.columns)}")

    # Join on index (walk-forward splits align)
    joined = dir_oos.join(
        mag_oos[[c for c in mag_oos.columns if c not in dir_oos.columns]],
        how="inner"
    )
    print(f"Joined: {len(joined)} bars\n")

    # ── Mag model standalone eval ──
    mag_pred_col = next((c for c in ["pred_mag", "pred", "y_pred"]
                         if c in joined.columns), None)
    mag_y_col = next((c for c in ["y_vol_adj_abs", "y_mag", "y_true"]
                      if c in joined.columns), None)
    if mag_pred_col and mag_y_col:
        from scipy.stats import spearmanr
        mag_ic = float(spearmanr(joined[mag_pred_col], joined[mag_y_col]).correlation)
        print("=" * 70)
        print("  MAG MODEL STANDALONE")
        print("=" * 70)
        print(f"  Spearman IC (vs y_vol_adj_abs) : {mag_ic:+.4f}")

        joined["mag_decile"] = pd.qcut(joined[mag_pred_col], 10,
                                        labels=False, duplicates="drop")
        decile_stats = joined.groupby("mag_decile").agg(
            n=(mag_pred_col, "size"),
            pred_mean=(mag_pred_col, "mean"),
            realized_mean=(mag_y_col, "mean"),
        )
        print("\n  Mag decile breakdown (predict σ vs realized σ):")
        print(f"  {'decile':<8}{'n':>6}{'pred σ':>10}{'realized σ':>14}")
        print("  " + "-" * 40)
        for d, r in decile_stats.iterrows():
            print(f"  {int(d):<8d}{int(r['n']):>6d}{r['pred_mean']:>10.2f}"
                  f"{r['realized_mean']:>14.2f}")

    # ── Direction Strong WR by Mag decile ──
    dir_pred = joined["pred_ret"].values
    dir_y = joined["y_path_ret_4h"].values
    fire_dir, win_dir = rolling_top5(dir_pred, dir_y)

    joined["dir_fire"] = fire_dir
    joined["dir_win"] = win_dir

    # Convert mag pred to rolling percentile (same as production mag_score)
    mag_pred_arr = joined[mag_pred_col].values
    mag_pct = np.full(len(mag_pred_arr), np.nan)
    for i in range(len(mag_pred_arr)):
        lo_i = max(0, i - ROLL + 1)
        w = mag_pred_arr[lo_i:i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 50:
            mag_pct[i] = (w < mag_pred_arr[i]).sum() / len(w)
    joined["mag_pct"] = mag_pct

    print("\n" + "=" * 70)
    print("  DIRECTION STRONG WR × MAG PERCENTILE BIN")
    print("=" * 70)
    print(f"  {'mag bin':<14}{'n_fire':>9}{'wins':>7}{'WR':>10}"
          f"{'Wilson CI':>22}")
    print("  " + "-" * 62)

    # Overall baseline
    fires_all = joined["dir_fire"].sum()
    wins_all = joined["dir_win"].sum()
    wr_all = wins_all / fires_all if fires_all else 0
    lo, hi = wilson_ci(int(wins_all), int(fires_all))
    print(f"  {'ALL (base)':<14}{int(fires_all):>9d}{int(wins_all):>7d}"
          f"{wr_all*100:>9.1f}%  [{lo*100:5.1f}%, {hi*100:5.1f}%]")

    # Bins
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.70),
            (0.70, 0.85), (0.85, 1.01)]
    for lo_b, hi_b in bins:
        m = joined["mag_pct"].between(lo_b, hi_b, inclusive="left")
        sub = joined[m & joined["dir_fire"]]
        if len(sub) < 5:
            print(f"  {lo_b:.2f}-{hi_b:.2f}     (too few: {len(sub)})")
            continue
        n = len(sub)
        w = int(sub["dir_win"].sum())
        wr = w / n
        clo, chi = wilson_ci(w, n)
        print(f"  {lo_b:.2f}-{hi_b:<9.2f}{n:>9d}{w:>7d}"
              f"{wr*100:>9.1f}%  [{clo*100:5.1f}%, {chi*100:5.1f}%]")

    # ── Cumulative: require mag ≥ threshold ──
    print("\n  Cumulative: require mag percentile >= threshold")
    print(f"  {'threshold':<14}{'n_fire':>9}{'wins':>7}{'WR':>10}"
          f"{'Wilson CI':>22}{'delta':>10}")
    print("  " + "-" * 72)
    for thr in [0.0, 0.30, 0.50, 0.60, 0.70, 0.80]:
        m = joined["mag_pct"] >= thr
        sub = joined[m & joined["dir_fire"]]
        if len(sub) < 5:
            continue
        n = len(sub)
        w = int(sub["dir_win"].sum())
        wr = w / n
        clo, chi = wilson_ci(w, n)
        delta = (wr - wr_all) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  mag>={thr:<9.2f}{n:>9d}{w:>7d}{wr*100:>9.1f}%"
              f"  [{clo*100:5.1f}%, {chi*100:5.1f}%]  {sign}{delta:+.1f}pp")


if __name__ == "__main__":
    main()
