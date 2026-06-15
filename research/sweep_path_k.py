"""
Sweep k multiplier for the new TWAP-based direction label.

For each k, report:
  - Label distribution (UP% / DOWN% / deadzone%)
  - Walk-forward OOS IC against BOTH path_return_4h and endpoint return_4h
  - OOS AUC

Uses the same ablation 35-feature set as current production.
"""
from __future__ import annotations

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

ABLATION = Path("research/results/ablation_study.json")
OUT = Path("research/results/sweep_path_k.json")

DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}


def get_feats(df: pd.DataFrame) -> list[str]:
    abl = json.loads(ABLATION.read_text())
    base = abl.get("baseline", {}).get("features", [])
    keep = abl.get("combined", {}).get("keep_features", [])
    seen = set()
    out = []
    for f in base + keep:
        if f in df.columns and f not in seen:
            out.append(f)
            seen.add(f)
    return out


def wf_eval(df: pd.DataFrame, feats: list[str], y: np.ndarray,
            path_ret: np.ndarray, end_ret: np.ndarray,
            weights: np.ndarray) -> dict:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    probs, labels, paths, ends = [], [], [], []

    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue

        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask].astype(int)
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask].astype(int)
        sw = weights[tr_idx][tr_mask]

        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0

        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pr = m.predict_proba(X_te)[:, 1]

        te_global = np.array(te_idx)[te_mask]
        probs.extend(pr.tolist())
        labels.extend(y_te.tolist())
        paths.extend(path_ret[te_global].tolist())
        ends.extend(end_ret[te_global].tolist())

    probs = np.array(probs)
    labels = np.array(labels)
    paths = np.array(paths)
    ends = np.array(ends)

    auc = float(roc_auc_score(labels, probs))
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    ic_path, _ = spearmanr(probs, paths)
    ic_end, _ = spearmanr(probs, ends)

    return {
        "n_oos": int(len(probs)),
        "auc": auc,
        "acc": acc,
        "ic_path": float(ic_path),
        "ic_endpoint": float(ic_end),
    }


def main() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    feats = get_feats(df)
    print(f"Features: {len(feats)} (ablation baseline + KEEP)")
    print(f"Rows: {len(df)}")

    # Regime weights (production-matched)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    weights = np.ones(len(df))
    bull_mask = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear_mask = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    weights[bull_mask] = 4.0
    weights[bear_mask] = 2.0

    # Baseline reference: endpoint k=0.5 (what production currently uses)
    # Build once for comparison. We'll compare every TWAP-k against this.
    k_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    results = []

    for k in k_grid:
        labels_df = build_direction_labels(df, k=k)
        y = labels_df["y_dir"].values.astype(float)
        path_ret = labels_df["path_return_4h"].values.astype(float)
        end_ret = labels_df["return_4h"].values.astype(float)

        n_valid = int(np.isfinite(y).sum())
        n_up = int(np.nansum(y == 1))
        n_down = int(np.nansum(y == 0))
        horizon = 4
        n_total = len(df) - horizon
        n_dead = n_total - n_up - n_down
        up_pct = n_up / n_total * 100
        down_pct = n_down / n_total * 100
        dead_pct = n_dead / n_total * 100

        print(f"\n=== k={k:.2f} ===")
        print(f"  UP={up_pct:.1f}% DOWN={down_pct:.1f}% dead={dead_pct:.1f}% (n_valid={n_valid})")

        metrics = wf_eval(df, feats, y, path_ret, end_ret, weights)
        print(f"  AUC={metrics['auc']:.4f} Acc={metrics['acc']:.1%} "
              f"IC_path={metrics['ic_path']:+.4f} IC_endpoint={metrics['ic_endpoint']:+.4f}")

        results.append({
            "k": k,
            "up_pct": up_pct,
            "down_pct": down_pct,
            "dead_pct": dead_pct,
            "n_valid": n_valid,
            **metrics,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'k':>6} {'UP%':>7} {'DOWN%':>7} {'dead%':>7} {'AUC':>8} {'Acc':>8} {'IC_path':>10} {'IC_end':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['k']:>6.2f} {r['up_pct']:>6.1f}% {r['down_pct']:>6.1f}% "
              f"{r['dead_pct']:>6.1f}% {r['auc']:>8.4f} {r['acc']:>7.1%} "
              f"{r['ic_path']:>+10.4f} {r['ic_endpoint']:>+10.4f}")


if __name__ == "__main__":
    main()
