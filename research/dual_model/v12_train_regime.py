"""
v12 3-class regime classifier (Idea 2, user 2026-05-12).

XGBClassifier with multi:softprob, num_class=3.  Same 136 production features
as v7/v9.  Target = y_regime (0=no_trend, 1=up, 2=down) built by
build_regime_labels.

Goal: see if reframing the prediction problem (binary 'TP before SL' →
3-class trend direction) breaks past the AUC 0.54 ceiling that v9/v10/v11
hit on binary target.

Evaluation:
    Per-class one-vs-rest AUC
    Trade-rule cohort (max(P_up, P_dn) >= threshold + argmax in trend class)
    WR on triggered trades vs base rate
    5-seed sensitivity
"""
from __future__ import annotations
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"
V9_FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"

BASE_PARAMS = dict(
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    early_stopping_rounds=30,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
)


def load_data(threshold_bp: int, horizon: int):
    features_df = pd.read_parquet(FEATURES_CACHE)
    labels_df = pd.read_parquet(
        LABEL_CACHE_DIR / f"labels_regime_T{threshold_bp}_H{horizon}.parquet"
    )
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_convert("UTC").tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)
    df = features_df.join(labels_df[["y_regime"]], how="inner")
    df = df.dropna(subset=["y_regime"])

    with open(V9_FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def train_one(X: np.ndarray, y: np.ndarray, splits: list, seed: int) -> np.ndarray:
    n = len(X)
    oos_p = np.full((n, 3), np.nan)
    for tr_idx, te_idx in splits:
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]
        params = dict(BASE_PARAMS, random_state=seed)
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_in], y[tr_in],
                   eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])
    return oos_p


def report_calibration(probas: np.ndarray, y: np.ndarray) -> None:
    """Multi-class AUC + per-class hit rates at threshold."""
    mask = np.isfinite(probas[:, 0])
    p = probas[mask]
    yv = y[mask]

    # One-vs-rest AUC per class
    print("\n--- Per-class one-vs-rest AUC ---")
    class_names = {0: "no_trend", 1: "up_trend", 2: "down_trend"}
    for cls in [0, 1, 2]:
        y_bin = (yv == cls).astype(int)
        if y_bin.sum() > 0:
            auc = roc_auc_score(y_bin, p[:, cls])
            print(f"  {cls} ({class_names[cls]}): AUC={auc:.4f}  base_rate={y_bin.mean()*100:.1f}%")

    # Trade rule: filter on max(P_up, P_dn) and predicted class
    print("\n--- Trade rule sweep ---")
    print(f"  Rule: if argmax in {{1,2}} AND max(P_up, P_dn) >= T → trade")
    print(f"  {'T':>5} {'n_signals':>10} {'WR':>7} {'avg_P':>7} "
          f"{'precision':>10} {'recall':>7}")

    for T in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        pred_class = np.argmax(p, axis=1)
        max_trend_p = np.maximum(p[:, 1], p[:, 2])
        signal = (pred_class != 0) & (max_trend_p >= T)
        n = int(signal.sum())
        if n < 5:
            print(f"  {T:>5.2f} {n:>10} (too few)")
            continue
        # Win = predicted class matches actual non-zero class
        sub_pred = pred_class[signal]
        sub_y = yv[signal]
        correct = (sub_pred == sub_y).sum()
        wr = correct / n
        # Precision per direction
        up_signals = (pred_class == 1) & (max_trend_p >= T)
        dn_signals = (pred_class == 2) & (max_trend_p >= T)
        up_prec = ((pred_class == 1) & (yv == 1) & (max_trend_p >= T)).sum() / max(up_signals.sum(), 1)
        dn_prec = ((pred_class == 2) & (yv == 2) & (max_trend_p >= T)).sum() / max(dn_signals.sum(), 1)
        recall_trend = correct / max((yv != 0).sum(), 1)
        avg_p = max_trend_p[signal].mean()
        print(f"  {T:>5.2f} {n:>10} {wr*100:>5.1f}% {avg_p:>6.3f}  "
              f"UP_prec={up_prec*100:>4.1f}% DN_prec={dn_prec*100:>4.1f}%  "
              f"recall_trend={recall_trend*100:>4.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.012,
                        help="Match the threshold used in build_regime_labels")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds (e.g. '42,1,7,123,2026')")
    args = parser.parse_args()

    threshold_bp = int(args.threshold * 10000)
    df, feature_cols = load_data(threshold_bp, args.horizon)
    print(f"\nData: n={len(df)}, features={len(feature_cols)}, "
          f"threshold={args.threshold*100:.1f}%, horizon={args.horizon}")

    y = df["y_regime"].values.astype(int)
    X = df[feature_cols].values.astype(np.float32)
    print(f"Class distribution: no_trend={(y==0).mean()*100:.1f}% "
          f"up={(y==1).mean()*100:.1f}% dn={(y==2).mean()*100:.1f}%")

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                   step=48, purge=4, embargo=4)

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\nTraining {len(seeds)} seed(s)...")

    all_probas = []
    for seed in seeds:
        logger.info("  seed=%d", seed)
        probas = train_one(X, y, splits, seed)
        all_probas.append(probas)
        print(f"\n=== Seed {seed} results ===")
        report_calibration(probas, y)

    # If 5 seeds run, also ensemble + Jaccard
    if len(seeds) >= 3:
        print("\n=== Ensemble (mean across seeds) ===")
        mean_probas = np.nanmean(all_probas, axis=0)
        report_calibration(mean_probas, y)

        # Cohort jaccard at T=0.5 between seeds
        print("\n--- Cohort Jaccard between seeds (at T=0.5) ---")
        cohorts = []
        for probas in all_probas:
            mask = np.isfinite(probas[:, 0])
            pred_class = np.argmax(probas, axis=1)
            max_p = np.maximum(probas[:, 1], probas[:, 2])
            signal = mask & (pred_class != 0) & (max_p >= 0.5)
            cohorts.append(set(np.where(signal)[0]))
        jaccards = []
        for i in range(len(cohorts)):
            for j in range(i + 1, len(cohorts)):
                a, b = cohorts[i], cohorts[j]
                jaccards.append(len(a & b) / max(len(a | b), 1))
        print(f"  mean Jaccard: {np.mean(jaccards):.3f}  "
              f"min: {min(jaccards):.3f}  max: {max(jaccards):.3f}")

    # Save first-seed OOS for further analysis
    if seeds:
        out_path = RESULTS_DIR / f"direction_v12_regime_T{threshold_bp}_H{args.horizon}_oos.parquet"
        first_probas = all_probas[0]
        mask = np.isfinite(first_probas[:, 0])
        out = pd.DataFrame({
            "ts": df.index[mask],
            "p_no_trend": first_probas[mask, 0],
            "p_up_trend": first_probas[mask, 1],
            "p_dn_trend": first_probas[mask, 2],
            "y_regime": y[mask],
        })
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        logger.info("Saved v12 OOS (seed=%d) -> %s", seeds[0], out_path)


if __name__ == "__main__":
    main()
