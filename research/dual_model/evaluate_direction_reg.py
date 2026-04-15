"""
Detailed evaluation for direction-regression OOS predictions.

Reads the walk-forward output from train_direction_reg_4h.py and produces:
  - MAE / RMSE / pred-vs-true distribution
  - Spearman IC (overall + monthly)
  - Sign-AUC
  - Strong WR with Wilson 95% CI at a user-chosen threshold
  - Reliability diagram (10 quantile bins of pred vs realized)
  - Monthly Strong performance table

Usage:
    python -m research.dual_model.evaluate_direction_reg
    python -m research.dual_model.evaluate_direction_reg --objective mse --threshold 0.010
    python -m research.dual_model.evaluate_direction_reg --objective huber --threshold 0.008
"""
from __future__ import annotations

import sys
import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import RESULTS_DIR


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def reliability_diagram(pred: np.ndarray, true: np.ndarray,
                        n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions into quantile bins and compare mean predicted return
    to mean realized path return. A well-calibrated regressor has
    pred_mean ≈ true_mean for every bin (identity line on a scatter plot).
    """
    q = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo = q[i]
        hi = q[i + 1]
        if i < n_bins - 1:
            mask = (pred >= lo) & (pred < hi)
        else:
            mask = (pred >= lo)
        if mask.sum() == 0:
            continue
        rows.append(dict(
            bin=i + 1,
            n=int(mask.sum()),
            pred_lo=float(lo),
            pred_hi=float(hi),
            pred_mean=float(pred[mask].mean()),
            true_mean=float(true[mask].mean()),
            hit_rate=float((np.sign(pred[mask]) == np.sign(true[mask])).mean()),
        ))
    return pd.DataFrame(rows)


def evaluate(objective: str, threshold: float):
    path = RESULTS_DIR / f"direction_reg_oos_{objective}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run train_direction_reg_4h first.")
        return
    oos = pd.read_parquet(path)

    pred = oos["pred_ret"].values.astype(float)
    true = oos["y_path_ret_4h"].values.astype(float)

    print("=" * 76)
    print(f"  DIRECTION-REG EVALUATION  objective={objective}  "
          f"threshold=±{threshold:.3f}")
    print("=" * 76)
    print(f"  n_oos = {len(oos)}   folds = {oos['fold'].nunique()}")

    # ── Regression metrics ──────────────────────────────────────────────
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    print(f"\n  MAE          : {mae:.5f}")
    print(f"  RMSE         : {rmse:.5f}")
    print(f"  pred  μ / σ  : {pred.mean():+.5f} / {pred.std():.5f}")
    print(f"  true  μ / σ  : {true.mean():+.5f} / {true.std():.5f}")

    from scipy.stats import spearmanr
    ic = float(spearmanr(pred, true).correlation)
    print(f"  Spearman IC  : {ic:+.4f}")

    from sklearn.metrics import roc_auc_score
    mask = true != 0
    auc = float(roc_auc_score((true[mask] > 0).astype(int), pred[mask]))
    print(f"  AUC (sign)   : {auc:.4f}")

    # ── Strong signal ───────────────────────────────────────────────────
    long_fire = pred >= threshold
    short_fire = pred <= -threshold
    n_long = int(long_fire.sum())
    n_short = int(short_fire.sum())
    n_total = n_long + n_short
    w_long = int((long_fire & (true > 0)).sum())
    w_short = int((short_fire & (true < 0)).sum())
    w_total = w_long + w_short

    print(f"\n  === Strong signal @ ±{threshold:.3f} ===")
    print(f"  fires       : total={n_total}  long={n_long}  short={n_short}")
    print(f"  fire rate   : {n_total / len(pred) * 100:.2f}%")
    if n_total > 0:
        wr = w_total / n_total
        lo, hi = wilson_ci(w_total, n_total)
        tag = "  <-- hits 60% target" if wr >= 0.60 else ""
        print(f"  WR overall  : {wr * 100:.1f}%  [{lo * 100:.1f}%, "
              f"{hi * 100:.1f}%]{tag}")
        if n_long > 0:
            wr_l = w_long / n_long
            lo_l, hi_l = wilson_ci(w_long, n_long)
            print(f"  WR long     : {wr_l * 100:.1f}%  "
                  f"[{lo_l * 100:.1f}%, {hi_l * 100:.1f}%]  (n={n_long})")
        if n_short > 0:
            wr_s = w_short / n_short
            lo_s, hi_s = wilson_ci(w_short, n_short)
            print(f"  WR short    : {wr_s * 100:.1f}%  "
                  f"[{lo_s * 100:.1f}%, {hi_s * 100:.1f}%]  (n={n_short})")

    # ── Reliability diagram ────────────────────────────────────────────
    print("\n  === Reliability (10 quantile bins of pred) ===")
    rel = reliability_diagram(pred, true, n_bins=10)
    if not rel.empty:
        print(rel.to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}",
        ))
        # Save for plotting
        rel.to_csv(RESULTS_DIR / f"direction_reg_reliability_{objective}.csv",
                   index=False)

    # ── Monthly Strong WR ──────────────────────────────────────────────
    if isinstance(oos.index, pd.DatetimeIndex) and n_total > 0:
        dfm = pd.DataFrame({
            "pred": pred,
            "true": true,
        }, index=oos.index)
        dfm["fire"] = (dfm["pred"] >= threshold) | (dfm["pred"] <= -threshold)
        dfm["win"] = (
            ((dfm["pred"] >= threshold) & (dfm["true"] > 0)) |
            ((dfm["pred"] <= -threshold) & (dfm["true"] < 0))
        )
        dfm["month"] = dfm.index.strftime("%Y-%m")
        print("\n  === Monthly Strong WR ===")
        print(f"  {'month':<10}{'fires':>7}{'wins':>7}{'WR':>10}{'CI':>22}")
        for m, g in dfm.groupby("month"):
            f = int(g["fire"].sum())
            if f == 0:
                continue
            w = int(g["win"].sum())
            wr = w / f
            lo, hi = wilson_ci(w, f)
            print(f"  {m:<10}{f:>7d}{w:>7d}{wr * 100:>9.1f}%"
                  f"  [{lo * 100:5.1f}%, {hi * 100:5.1f}%]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate direction-regression OOS predictions")
    p.add_argument("--objective", default="mse", choices=["mse", "huber"])
    p.add_argument("--threshold", type=float, default=0.010)
    args = p.parse_args()
    evaluate(args.objective, args.threshold)
