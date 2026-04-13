"""
Ablation Study — honest feature addition validation.

Instead of "prune 80 → 29" (which hides which new features actually help),
this script does the reverse:
  1. Establish baseline with current production 29 features (OOS AUC/IC)
  2. Add each candidate new feature ONE AT A TIME to the baseline
  3. Measure walk-forward OOS delta for each
  4. Only features with positive contribution should be kept

This is how you should have added the 6 regime features on 2026-04-13.
Rule of thumb: if a feature's ablation delta is < +0.002 OOS AUC, drop it.

Usage:
    python research/ablation_study.py
    python research/ablation_study.py --candidates is_trending_bull,oi_8h_non_bull

Output:
    research/results/ablation_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

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
OUT = Path("research/results/ablation_study.json")

# Candidate features to test (the 6 added on 2026-04-13)
DEFAULT_CANDIDATES = [
    "is_trending_bull",
    "is_trending_bear",
    "vol_kurt_non_bear",
    "oi_8h_non_bull",
    "long_liq_exhaustion_4h",
    "cvd_persistence_12h",
]

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0, "early_stopping_rounds": 30,
}


def compute_regime_weights(df: pd.DataFrame) -> np.ndarray:
    """Reproduce production regime weighting (bull=4x, bear=2x)."""
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


def walk_forward_eval(df: pd.DataFrame, feats: list[str],
                      splits: list, regime_weights: np.ndarray) -> dict:
    """Walk-forward OOS evaluation with regime weighting."""
    probs, trues, rets = [], [], []
    fold_aucs = []

    for train_idx, test_idx in splits:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        tm = tr["y_dir"].notna()
        ttm = te["y_dir"].notna()
        X_tr = tr.loc[tm, feats].fillna(0).values
        y_tr = tr.loc[tm, "y_dir"].values.astype(int)
        X_te = te.loc[ttm, feats].fillna(0).values
        y_te = te.loc[ttm, "y_dir"].values.astype(int)
        r_te = te.loc[ttm, "dir_return_4h"].values
        sw = regime_weights[train_idx][tm.values]

        if len(y_tr) < 50 or len(y_te) < 5:
            continue

        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0

        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_te, y_te)], verbose=False)
        prob = m.predict_proba(X_te)[:, 1]

        probs.extend(prob)
        trues.extend(y_te)
        rets.extend(r_te)
        try:
            fold_aucs.append(roc_auc_score(y_te, prob))
        except Exception:
            pass

    p_arr = np.array(probs)
    t_arr = np.array(trues)
    r_arr = np.array(rets)

    auc = roc_auc_score(t_arr, p_arr)
    ic, _ = spearmanr(p_arr, r_arr)
    acc = ((p_arr > 0.5).astype(int) == t_arr).mean()

    return {
        "auc": float(auc),
        "ic": float(ic),
        "acc": float(acc),
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "n_folds": len(fold_aucs),
        "n_samples": len(p_arr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES),
                    help="Comma-separated candidate features")
    args = ap.parse_args()
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]

    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

    # Labels (direction)
    labels = build_direction_labels(df, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["dir_return_4h"] = labels["return_4h"]

    # Load production baseline features
    with open(PROD_DIR_FEATS) as f:
        baseline = json.load(f)
    baseline = [f for f in baseline if f in df.columns]
    print(f"Baseline: {len(baseline)} production features")

    # Check candidate availability
    available = [c for c in candidates if c in df.columns]
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        print(f"WARNING: missing candidates (skipped): {missing}")

    # Precompute regime weights and splits
    regime_weights = compute_regime_weights(df)
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    print(f"Walk-forward: {len(splits)} folds, regime weighting bull=4x bear=2x\n")

    # Baseline
    print("=" * 70)
    print("BASELINE (production 29 features)")
    print("=" * 70)
    base_metrics = walk_forward_eval(df, baseline, splits, regime_weights)
    print(f"  AUC={base_metrics['auc']:.4f}  IC={base_metrics['ic']:+.4f}  "
          f"Acc={base_metrics['acc']:.1%}  "
          f"fold_auc={base_metrics['fold_auc_mean']:.4f}±{base_metrics['fold_auc_std']:.4f}")

    # Ablation: baseline + 1 new feature at a time
    print("\n" + "=" * 70)
    print("ABLATION: baseline + 1 new feature")
    print("=" * 70)
    print(f"{'Feature':32s}  {'Δ AUC':>9s}  {'Δ IC':>9s}  {'Δ Acc':>8s}  Verdict")
    print("-" * 80)

    results = {
        "baseline": {
            "n_features": len(baseline),
            "features": baseline,
            **base_metrics,
        },
        "candidates": {},
    }

    for feat in available:
        if feat in baseline:
            print(f"{feat:32s}  (already in baseline — skipped)")
            continue
        feats = baseline + [feat]
        m = walk_forward_eval(df, feats, splits, regime_weights)

        d_auc = m["auc"] - base_metrics["auc"]
        d_ic = m["ic"] - base_metrics["ic"]
        d_acc = m["acc"] - base_metrics["acc"]

        # Verdict: positive if ΔAUC > +0.002 AND ΔIC > 0
        if d_auc > 0.002 and d_ic > 0:
            verdict = "KEEP"
        elif d_auc > 0 and d_ic > 0:
            verdict = "marginal"
        else:
            verdict = "DROP"

        print(f"{feat:32s}  {d_auc:+9.4f}  {d_ic:+9.4f}  {d_acc:+8.2%}  {verdict}")

        results["candidates"][feat] = {
            "metrics": m,
            "delta_auc": float(d_auc),
            "delta_ic": float(d_ic),
            "delta_acc": float(d_acc),
            "verdict": verdict,
        }

    # Combined: baseline + all KEEP features
    keep_feats = [f for f, r in results["candidates"].items()
                  if r["verdict"] in ("KEEP", "marginal")]
    if keep_feats:
        print("\n" + "=" * 70)
        print(f"COMBINED: baseline + {len(keep_feats)} KEEP features")
        print("=" * 70)
        print(f"  Features: {keep_feats}")
        combined = walk_forward_eval(df, baseline + keep_feats, splits, regime_weights)
        d_auc = combined["auc"] - base_metrics["auc"]
        d_ic = combined["ic"] - base_metrics["ic"]
        print(f"  AUC={combined['auc']:.4f} ({d_auc:+.4f})  "
              f"IC={combined['ic']:+.4f} ({d_ic:+.4f})  "
              f"Acc={combined['acc']:.1%}")
        print("  Note: combined delta may differ from sum of individual deltas")
        print("        due to feature correlation. Trust the combined number.")

        results["combined"] = {
            "keep_features": keep_feats,
            "metrics": combined,
            "delta_auc": float(d_auc),
            "delta_ic": float(d_ic),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {OUT}")


if __name__ == "__main__":
    main()
