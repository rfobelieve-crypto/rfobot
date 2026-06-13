"""
Ensemble 5 seeds and test stability.

After A3 seed sensitivity fail showed Jaccard 0.113 between 5 single-seed models,
averaging their predictions might either:
    (a) Smooth out per-seed noise -> stable, broader cohort with maintained
         WR.  This would show that the underlying signal is real, just one
         seed's view of it is unstable.
    (b) Average down to base rate.  This would confirm signals are
         fundamentally noise — no underlying pattern, just lucky alignments.

Method:
    1. Walk-forward train 5 SHORT models with seeds [42, 1, 7, 123, 2026]
    2. For each OOS bar, compute mean P over 5 seeds
    3. Apply threshold to ensemble P, measure WR
    4. Compare to 5 single-seed performances
    5. Run permutation + temporal robustness on ensemble cohort

If ensemble:
    - WR >= 51% (BE) and Jaccard with single-seed cohorts is high
      -> ensemble works, deploy ensemble model
    - WR < 51%
      -> confirmed: signal is too weak, abandon v9 short-trade direction
"""
from __future__ import annotations
import sys
import json
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "labels_winrate_TP50_SL30_H8.parquet"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"

SEEDS = [42, 1, 7, 123, 2026]
THRESHOLD = 0.65
BE_WR = 0.51

BASE_PARAMS = dict(
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic", eval_metric="auc",
)


def train_one_seed(X: np.ndarray, y: np.ndarray, splits: list, seed: int) -> np.ndarray:
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
        model.fit(X[tr_in], y[tr_in],
                  eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oos_p


def main():
    features_df = pd.read_parquet(FEATURES_CACHE)
    labels_df = pd.read_parquet(LABEL_CACHE)
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_convert("UTC").tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)
    df = features_df.join(labels_df[["y_short_win"]], how="inner")
    df = df.dropna(subset=["y_short_win"])
    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_short_win"].values.astype(int)

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                   step=48, purge=4, embargo=4)
    logger.info("Training %d seeds, %d folds each...", len(SEEDS), len(splits))

    preds = {}
    for seed in SEEDS:
        logger.info("  seed=%d ...", seed)
        preds[seed] = train_one_seed(X, y, splits, seed)

    # Mask only bars where all 5 seeds have a prediction
    masks = [np.isfinite(preds[s]) for s in SEEDS]
    combined_mask = np.logical_and.reduce(masks)
    y_oos = y[combined_mask]
    n_oos = len(y_oos)

    # Per-seed performance at THRESHOLD
    print("\n" + "="*90)
    print("Per-seed cohort at P>=0.65")
    print("="*90)
    print(f"  {'seed':>6} {'n':>5} {'WR':>7} {'AUC':>7}")
    print("-" * 35)
    per_seed_pred = {}
    for s in SEEDS:
        p_v = preds[s][combined_mask]
        per_seed_pred[s] = p_v
        sel = p_v >= THRESHOLD
        n = int(sel.sum())
        wr = y_oos[sel].mean() if n > 0 else 0
        auc = roc_auc_score(y_oos, p_v)
        print(f"  {s:>6} {n:>5} {wr*100:>5.1f}% {auc:>6.4f}")

    # Ensemble: mean of 5 seeds' predictions
    ensemble_p = np.mean([per_seed_pred[s] for s in SEEDS], axis=0)

    print("\n" + "="*90)
    print("Ensemble (mean of 5 seeds) — cohort across thresholds")
    print("="*90)
    print(f"  {'threshold':>10} {'n_signals':>10} {'WR':>8} {'CI 95%':>20} {'AUC':>7}")
    print("-" * 65)
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = ensemble_p >= thr
        n = int(sel.sum())
        wins = int(y_oos[sel].sum())
        wr = wins / n if n > 0 else 0
        ci_str = ""
        if n >= 5:
            ci = binomtest(wins, n).proportion_ci(confidence_level=0.95)
            ci_str = f"[{ci.low*100:.1f}, {ci.high*100:.1f}]"
        print(f"  {thr:>10.2f} {n:>10} {wr*100:>6.2f}%  {ci_str:>20}")
    auc_ens = roc_auc_score(y_oos, ensemble_p)
    print(f"  Ensemble AUC: {auc_ens:.4f}")

    # Specific test: ensemble at THRESHOLD
    sel = ensemble_p >= THRESHOLD
    n = int(sel.sum())
    wins = int(y_oos[sel].sum())
    wr = wins / n if n > 0 else 0
    print(f"\n--- Ensemble @ P>={THRESHOLD} ---")
    print(f"  n={n}, WR={wr*100:.2f}%")
    if n >= 5:
        ci = binomtest(wins, n).proportion_ci(confidence_level=0.95)
        print(f"  95% CI: [{ci.low*100:.2f}, {ci.high*100:.2f}]")
        if ci.low > BE_WR:
            print(f"  PASS — CI lower {ci.low*100:.1f}% > 51% BE, robust edge")
        elif wr > BE_WR:
            print(f"  o — WR {wr*100:.1f}% > 51% BE but CI lower < 51%, weak evidence")
        else:
            print(f"  FAIL — WR < 51% BE, no edge")

    # Cohort overlap with individual seeds at THRESHOLD
    print("\n--- Cohort overlap (ensemble vs each seed @ P>=0.65) ---")
    ens_cohort = set(np.where(ensemble_p >= THRESHOLD)[0])
    for s in SEEDS:
        seed_cohort = set(np.where(per_seed_pred[s] >= THRESHOLD)[0])
        if len(ens_cohort | seed_cohort) > 0:
            jaccard = len(ens_cohort & seed_cohort) / len(ens_cohort | seed_cohort)
        else:
            jaccard = 0
        print(f"  seed={s:>5}: Jaccard={jaccard:.3f}")

    # Save ensemble OOS for downstream consumption
    out = pd.DataFrame({
        "ts": df.index[combined_mask],
        "p_ensemble": ensemble_p,
        "y_short_win": y_oos,
    })
    out_path = RESULTS_DIR / "direction_v9_ensemble5_oos.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved ensemble OOS -> %s", out_path)


if __name__ == "__main__":
    main()
