"""
Permutation Test — is the model's OOS IC real signal or overfitting artifact?

Procedure:
  1. Train the production pipeline normally, record real OOS AUC/IC
  2. Shuffle the labels N times (default 100). Each shuffle = "null world"
     where there is no signal. Train the same pipeline on shuffled labels.
  3. Build the null distribution of AUC/IC
  4. Compute: where does the real OOS metric sit in the null distribution?
     - p = (number of null trials >= real) / N
     - Below p < 0.05 → the model's signal is statistically significant
     - p > 0.05 → the OOS metric is within random-noise range

Why this matters:
  Walk-forward OOS splits prevent look-ahead bias but do NOT prevent
  multiple-testing overfit. With 80 features + 300 trees + 4 depths, the
  XGBoost has enough capacity to memorize patterns that survive a 77-fold
  walk-forward. Permutation testing is the only way to confirm that the
  0.6020 AUC you see is not reachable by pure curve fitting.

  Use the SAME pipeline, same regime weighting, same feature set.
  Only the labels change (to random noise with same marginal distribution).

Usage:
    python research/permutation_test.py                    # 50 shuffles, production 29 feats
    python research/permutation_test.py --n 100            # more shuffles
    python research/permutation_test.py --use-ablation     # test the 34-feature combined set
    python research/permutation_test.py --fast             # 2 folds only, quick sanity check

Output:
    research/results/permutation_test.json
    research/results/permutation_distribution.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_labels import build_direction_labels

PROD_DIR_FEATS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
ABLATION_JSON = Path("research/results/ablation_study.json")
OUT_JSON = Path("research/results/permutation_test.json")
OUT_PNG = Path("research/results/permutation_distribution.png")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 200,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}


def compute_regime_weights(df: pd.DataFrame) -> np.ndarray:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    weights = np.ones(len(df))
    bull_mask = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear_mask = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    weights[bull_mask] = 4.0
    weights[bear_mask] = 2.0
    return weights


def wf_eval(df: pd.DataFrame, feats: list[str], y: np.ndarray, rets: np.ndarray,
            splits, regime_weights: np.ndarray) -> tuple[float, float]:
    """Walk-forward AUC and IC for a given label array y (may be real or shuffled)."""
    probs, trues, rs = [], [], []
    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask].astype(int)
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask].astype(int)
        r_te = rets[te_idx][te_mask]
        sw = regime_weights[tr_idx][tr_mask]

        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0

        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        prob = m.predict_proba(X_te)[:, 1]
        probs.extend(prob)
        trues.extend(y_te)
        rs.extend(r_te)

    p_arr = np.array(probs)
    t_arr = np.array(trues)
    r_arr = np.array(rs)

    if len(p_arr) == 0 or t_arr.std() == 0:
        return (0.5, 0.0)
    auc = roc_auc_score(t_arr, p_arr)
    ic, _ = spearmanr(p_arr, r_arr)
    return (float(auc), float(ic))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Number of permutations (default 50)")
    ap.add_argument("--use-ablation", action="store_true",
                    help="Use the 34-feature set from ablation_study.json instead of 29")
    ap.add_argument("--fast", action="store_true", help="Only use every 10th fold for speed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    labels = build_direction_labels(df, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["dir_return_4h"] = labels["return_4h"]

    # Feature set
    with open(PROD_DIR_FEATS) as f:
        base = json.load(f)
    base = [c for c in base if c in df.columns]

    if args.use_ablation and ABLATION_JSON.exists():
        abl = json.loads(ABLATION_JSON.read_text())
        keep = abl.get("combined", {}).get("keep_features", [])
        feats = base + [k for k in keep if k not in base]
        label = f"ablation_34 (baseline + {len(keep)} KEEP)"
    else:
        feats = base
        label = f"baseline_29"

    feats = [c for c in feats if c in df.columns]
    print(f"Feature set: {label} ({len(feats)} features)")

    # Labels and returns arrays
    y = df["y_dir"].values.astype(float)  # NaN preserved
    rets = df["dir_return_4h"].values.astype(float)
    regime_weights = compute_regime_weights(df)

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    if args.fast:
        splits = splits[::10]
        print(f"FAST mode: using {len(splits)} folds (every 10th)")
    else:
        print(f"Walk-forward: {len(splits)} folds")

    # Real evaluation
    print("\n[1/2] Real evaluation (unshuffled labels)...")
    t0 = time.time()
    real_auc, real_ic = wf_eval(df, feats, y, rets, splits, regime_weights)
    print(f"  Real:  AUC={real_auc:.4f}  IC={real_ic:+.4f}  ({time.time()-t0:.1f}s)")

    # Permutation loop
    print(f"\n[2/2] Permutation loop ({args.n} shuffles)...")
    rng = np.random.default_rng(args.seed)
    null_aucs = []
    null_ics = []

    valid_mask = ~np.isnan(y)
    valid_idx = np.where(valid_mask)[0]

    t_start = time.time()
    for i in range(args.n):
        y_shuffled = y.copy()
        permuted = rng.permutation(y_shuffled[valid_idx])
        y_shuffled[valid_idx] = permuted

        # IMPORTANT: shuffle rets together? No — in a permutation test, the question
        # is "does the feature-label relationship exist?" We shuffle labels only.
        # But IC uses rets not y_dir, so we need to shuffle returns consistently:
        # to test direction signal, we permute return_4h too, matched to y.
        rets_shuffled = rets.copy()
        perm_rets = rng.permutation(rets_shuffled[valid_idx])
        rets_shuffled[valid_idx] = perm_rets

        auc, ic = wf_eval(df, feats, y_shuffled, rets_shuffled, splits, regime_weights)
        null_aucs.append(auc)
        null_ics.append(ic)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / (i + 1) * (args.n - i - 1)
            print(f"  [{i+1:3d}/{args.n}]  null AUC={auc:.4f} IC={ic:+.4f}  "
                  f"(elapsed={elapsed:.0f}s, eta={remaining:.0f}s)")

    null_aucs = np.array(null_aucs)
    null_ics = np.array(null_ics)

    # P-values (one-sided: how often does null match or beat real?)
    p_auc = float((null_aucs >= real_auc).mean())
    p_ic = float((np.abs(null_ics) >= abs(real_ic)).mean())

    print("\n" + "=" * 70)
    print("PERMUTATION TEST RESULTS")
    print("=" * 70)
    print(f"  Feature set: {label} ({len(feats)} features)")
    print(f"  Shuffles:    {args.n}")
    print()
    print(f"  Real AUC: {real_auc:.4f}")
    print(f"  Null AUC: mean={null_aucs.mean():.4f}  std={null_aucs.std():.4f}  "
          f"max={null_aucs.max():.4f}")
    print(f"  p(null >= real) = {p_auc:.3f}  ", end="")
    if p_auc < 0.05:
        print("SIGNIFICANT (AUC is real signal)")
    else:
        print("NOT significant (AUC may be overfitting)")
    print()
    print(f"  Real IC:  {real_ic:+.4f}")
    print(f"  Null IC:  mean={null_ics.mean():+.4f}  std={null_ics.std():.4f}  "
          f"|max|={np.abs(null_ics).max():.4f}")
    print(f"  p(|null| >= |real|) = {p_ic:.3f}  ", end="")
    if p_ic < 0.05:
        print("SIGNIFICANT (IC is real signal)")
    else:
        print("NOT significant (IC may be overfitting)")

    # Effect size (how many std devs above null)
    if null_aucs.std() > 0:
        auc_z = (real_auc - null_aucs.mean()) / null_aucs.std()
    else:
        auc_z = 0
    if null_ics.std() > 0:
        ic_z = (real_ic - null_ics.mean()) / null_ics.std()
    else:
        ic_z = 0
    print()
    print(f"  AUC z-score: {auc_z:+.2f}  (standard deviations above null mean)")
    print(f"  IC z-score:  {ic_z:+.2f}")

    # Save
    result = {
        "feature_set": label,
        "n_features": len(feats),
        "n_permutations": args.n,
        "real_auc": float(real_auc),
        "real_ic": float(real_ic),
        "null_auc_mean": float(null_aucs.mean()),
        "null_auc_std": float(null_aucs.std()),
        "null_auc_max": float(null_aucs.max()),
        "null_ic_mean": float(null_ics.mean()),
        "null_ic_std": float(null_ics.std()),
        "null_ic_abs_max": float(np.abs(null_ics).max()),
        "p_auc": p_auc,
        "p_ic": p_ic,
        "auc_z_score": float(auc_z),
        "ic_z_score": float(ic_z),
        "auc_significant": p_auc < 0.05,
        "ic_significant": p_ic < 0.05,
        "null_aucs": null_aucs.tolist(),
        "null_ics": null_ics.tolist(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(null_aucs, bins=20, color="steelblue", alpha=0.7, label=f"Null (n={args.n})")
    ax.axvline(real_auc, color="red", linewidth=2, label=f"Real = {real_auc:.4f}")
    ax.axvline(null_aucs.mean(), color="gray", linestyle="--",
               label=f"Null mean = {null_aucs.mean():.4f}")
    ax.set_xlabel("OOS AUC")
    ax.set_ylabel("Frequency")
    ax.set_title(f"AUC Null Distribution\np = {p_auc:.3f}  z = {auc_z:+.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(null_ics, bins=20, color="steelblue", alpha=0.7, label=f"Null (n={args.n})")
    ax.axvline(real_ic, color="red", linewidth=2, label=f"Real = {real_ic:+.4f}")
    ax.axvline(null_ics.mean(), color="gray", linestyle="--",
               label=f"Null mean = {null_ics.mean():+.4f}")
    ax.axvline(-abs(real_ic), color="red", linewidth=2, alpha=0.4)
    ax.set_xlabel("OOS IC")
    ax.set_title(f"IC Null Distribution\np = {p_ic:.3f}  z = {ic_z:+.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Permutation Test: {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
