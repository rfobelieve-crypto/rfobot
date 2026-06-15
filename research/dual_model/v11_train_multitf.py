"""
v11 = v9 features (136) + 12 multi-TF CVD features = 148 features.

Same XGBClassifier setup as v9/v10.  Quick AB test vs v9 baseline at
seed=42, H=8.  If AUC delta > 0.005 + WR delta > 2pp, run 5-seed
sensitivity.
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
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.dual_model.features_multitf_cvd import (
    add_multitf_cvd_features, EXTRA_FEATURES as MTF_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"
V9_FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)

THRESHOLD = 0.65
BE_WR_TAKER = 0.51
BE_WR_MAKER = 0.414

BASE_PARAMS = dict(
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic", eval_metric="auc",
)


def load_data(horizon: int, include_mtf: bool):
    features_df = pd.read_parquet(FEATURES_CACHE)
    labels_df = pd.read_parquet(LABEL_CACHE_DIR / f"labels_winrate_TP50_SL30_H{horizon}.parquet")
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_convert("UTC").tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)

    if include_mtf:
        features_df = add_multitf_cvd_features(features_df)

    df = features_df.join(labels_df[["y_short_win", "y_long_win"]], how="inner")
    df = df.dropna(subset=["y_short_win"])

    with open(V9_FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if include_mtf:
        for ef in MTF_FEATURES:
            if ef in df.columns and ef not in feature_cols:
                feature_cols.append(ef)
    return df, feature_cols


def train_one(X, y, splits, seed):
    n = len(X)
    oos_p = np.full(n, np.nan)
    for tr_idx, te_idx in splits:
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]
        n_pos = int(y[tr_in].sum())
        n_neg = len(tr_in) - n_pos
        params = dict(BASE_PARAMS, random_state=seed,
                      scale_pos_weight=(n_neg / max(n_pos, 1)))
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_in], y[tr_in], eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oos_p


def report(p, y, threshold, label):
    mask = np.isfinite(p)
    p_v = p[mask]
    y_v = y[mask]
    sel = p_v >= threshold
    n = int(sel.sum())
    wins = int(y_v[sel].sum())
    wr = wins / n if n > 0 else 0
    auc = roc_auc_score(y_v, p_v)
    ci_str = ""
    if n >= 5:
        ci = binomtest(wins, n).proportion_ci(confidence_level=0.95)
        ci_str = f"[{ci.low*100:.1f}, {ci.high*100:.1f}]"
    print(f"  {label:<30} AUC={auc:.4f}  n>={threshold}: {n:>3}  "
          f"WR={wr*100:>5.1f}%  CI={ci_str}")
    return {"auc": auc, "n": n, "wr": wr}


def step1_ab_seed42(horizon=8):
    print(f"\n{'='*90}")
    print(f"STEP 1: v9 vs v11 (multi-TF CVD) AB test  seed=42, H={horizon}")
    print(f"{'='*90}")
    df_v9, cols_v9 = load_data(horizon, include_mtf=False)
    df_v11, cols_v11 = load_data(horizon, include_mtf=True)
    print(f"  v9 features:  {len(cols_v9)}")
    print(f"  v11 features: {len(cols_v11)} (+{len(cols_v11) - len(cols_v9)} multi-TF CVD)")

    splits = walk_forward_splits(len(df_v9), initial_train=288, test_size=48,
                                   step=48, purge=4, embargo=4)
    y = df_v9["y_short_win"].values.astype(int)
    X_v9 = df_v9[cols_v9].values.astype(np.float32)
    X_v11 = df_v11[cols_v11].values.astype(np.float32)

    logger.info("Training v9 (seed=42)...")
    p_v9 = train_one(X_v9, y, splits, 42)
    logger.info("Training v11 (seed=42)...")
    p_v11 = train_one(X_v11, y, splits, 42)

    print()
    r_v9 = report(p_v9, y, THRESHOLD, "v9  (136 feat)")
    r_v11 = report(p_v11, y, THRESHOLD, "v11 (148 feat, +12 MTF CVD)")
    auc_delta = r_v11["auc"] - r_v9["auc"]
    wr_delta = r_v11["wr"] - r_v9["wr"]
    print(f"\n  Delta: AUC {auc_delta:+.4f}, WR {wr_delta*100:+.2f}pp")
    return r_v9, r_v11, p_v11, df_v11, cols_v11, splits, y


def step2_seed_sensitivity(df, cols, splits, y, horizon=8):
    print(f"\n{'='*90}")
    print(f"STEP 2: v11 5-seed sensitivity @ H={horizon}")
    print(f"{'='*90}")
    X = df[cols].values.astype(np.float32)
    seeds = [42, 1, 7, 123, 2026]
    preds = {}
    aucs, wrs, ns = [], [], []
    for s in seeds:
        logger.info("  seed=%d ...", s)
        preds[s] = train_one(X, y, splits, s)
        mask = np.isfinite(preds[s])
        p_v = preds[s][mask]
        y_v = y[mask]
        auc = roc_auc_score(y_v, p_v)
        sel = p_v >= THRESHOLD
        n = int(sel.sum())
        wr = y_v[sel].mean() if n > 0 else 0
        aucs.append(auc); wrs.append(wr); ns.append(n)
        print(f"  seed={s:>5}: AUC={auc:.4f}  n>=0.65: {n:>3}  WR={wr*100:>5.1f}%")

    print(f"\n  AUC: mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
          f"min={min(aucs):.4f}  max={max(aucs):.4f}")
    print(f"  WR:  mean={np.mean(wrs)*100:.2f}%  std={np.std(wrs)*100:.2f}%  "
          f"min={min(wrs)*100:.1f}%  max={max(wrs)*100:.1f}%")
    seeds_above_be = sum(1 for wr in wrs if wr >= BE_WR_TAKER)
    print(f"  Seeds above 51% (taker BE): {seeds_above_be}/5")

    cohorts = [set(np.where(preds[s][np.isfinite(preds[s])] >= THRESHOLD)[0]) for s in seeds]
    jaccards = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = cohorts[i], cohorts[j]
            jaccards.append(len(a & b) / max(len(a | b), 1))
    print(f"  Cohort Jaccard: mean={np.mean(jaccards):.3f} "
          f"min={min(jaccards):.3f} max={max(jaccards):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--always-do-step2", action="store_true",
                        help="Run 5-seed sensitivity even if step 1 doesn't show clear improvement")
    args = parser.parse_args()

    r_v9, r_v11, p_v11, df, cols, splits, y = step1_ab_seed42(args.horizon)
    if r_v11["auc"] > r_v9["auc"] + 0.003 or args.always_do_step2:
        step2_seed_sensitivity(df, cols, splits, y, args.horizon)
    else:
        print("\n[Skip] v11 AUC not materially improved over v9, skipping seed sensitivity.")


if __name__ == "__main__":
    main()
