"""
v9 Dual Binary Win-Rate Classifier — walk-forward training.

Two independent XGBClassifiers:
    Model A: P(long_win | 136 features)  — predict TP-before-SL for long entry
    Model B: P(short_win | 136 features) — symmetric for short entry

Same walk-forward setup as v7/v8 (initial_train=288, test=48, step=48,
purge=4, embargo=4), same 136 features.  Hyperparams adapted for binary
classification (binary:logistic + scale_pos_weight to handle 36/64 imbalance).

Outputs:
    research/results/dual_model/direction_v9_winrate_oos.parquet
        columns: ts, p_long_win, p_short_win, y_long_win, y_short_win,
                 long_label_type, short_label_type
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"

BASE_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic",
    eval_metric="auc",
)


def train_one_side(X: np.ndarray, y: np.ndarray, splits: list,
                    side_label: str) -> np.ndarray:
    """Walk-forward train a binary classifier for one side (long or short)."""
    n = len(X)
    oos_p = np.full(n, np.nan)
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]
        y_tr_in = y[tr_in]
        n_pos = int(y_tr_in.sum())
        n_neg = len(y_tr_in) - n_pos
        spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

        params = dict(BASE_PARAMS)
        params["scale_pos_weight"] = spw

        model = xgb.XGBClassifier(**params)
        model.fit(
            X[tr_in], y_tr_in,
            eval_set=[(X[tr_val], y[tr_val])],
            verbose=False,
        )
        # Predict probability of class 1 (win)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]

        if (fold_i + 1) % 20 == 0 or fold_i == len(splits) - 1:
            logger.info("  [%s] fold %d/%d done (scale_pos_weight=%.2f)",
                         side_label, fold_i + 1, len(splits), spw)
    return oos_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    horizon = args.horizon
    label_cache = LABEL_CACHE_DIR / f"labels_winrate_TP50_SL30_H{horizon}.parquet"
    out_path = RESULTS_DIR / f"direction_v9_winrate_H{horizon}_oos.parquet"
    if not label_cache.exists():
        logger.error("Label cache not found: %s — run build_winrate_labels --horizon %d first",
                     label_cache, horizon)
        sys.exit(1)

    logger.info("Loading features and labels (H=%d)...", horizon)
    features_df = pd.read_parquet(FEATURES_CACHE)
    labels_df = pd.read_parquet(label_cache)
    logger.info("Features: n=%d, Labels: n=%d", len(features_df), len(labels_df))

    # Normalize tz: features is tz-aware, labels saved tz-naive — align both naive UTC
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_convert("UTC").tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)

    df = features_df.join(
        labels_df[["y_long_win", "y_short_win",
                    "long_label_type", "short_label_type"]],
        how="inner",
    )
    df = df.dropna(subset=["y_long_win", "y_short_win"])
    logger.info("Joined: n=%d", len(df))

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Using %d features", len(feature_cols))

    X = df[feature_cols].values.astype(np.float32)
    y_long = df["y_long_win"].values.astype(int)
    y_short = df["y_short_win"].values.astype(int)
    ts = df.index

    splits = walk_forward_splits(
        n_samples=len(df),
        initial_train=288, test_size=48, step=48,
        purge=4, embargo=4,
    )
    logger.info("Walk-forward folds: %d", len(splits))

    logger.info("Training LONG side...")
    p_long = train_one_side(X, y_long, splits, "LONG")
    logger.info("Training SHORT side...")
    p_short = train_one_side(X, y_short, splits, "SHORT")

    mask = np.isfinite(p_long) & np.isfinite(p_short)
    pL = p_long[mask]
    pS = p_short[mask]
    yL = y_long[mask]
    yS = y_short[mask]
    ts_oos = ts[mask]
    ltype_oos = df.loc[mask, "long_label_type"].values
    stype_oos = df.loc[mask, "short_label_type"].values

    # Metrics
    auc_long = roc_auc_score(yL, pL)
    auc_short = roc_auc_score(yS, pS)
    ic_long, _ = spearmanr(pL, yL)
    ic_short, _ = spearmanr(pS, yS)
    print(f"\n{'='*80}")
    print(f"v9 OOS metrics (n={len(pL)})")
    print(f"{'='*80}")
    print(f"  LONG  AUC = {auc_long:.4f}  Spearman = {ic_long:+.4f}")
    print(f"  SHORT AUC = {auc_short:.4f}  Spearman = {ic_short:+.4f}")
    print(f"  LONG  base WR = {yL.mean()*100:.2f}%, mean predicted P = {pL.mean()*100:.2f}%")
    print(f"  SHORT base WR = {yS.mean()*100:.2f}%, mean predicted P = {pS.mean()*100:.2f}%")

    # Quick calibration: WR at different P thresholds
    print(f"\n{'-'*80}")
    print(f"Calibration: actual win rate at P threshold (LONG)")
    print(f"{'-'*80}")
    print(f"  {'threshold':>10} {'n_signals':>10} {'actual_WR':>10} {'edge':>8}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = yL[pL >= thr]
        if len(sub) >= 20:
            wr = sub.mean()
            edge = (wr - 0.51) * 100  # 51% BE WR for cost
            print(f"  {thr:>10.2f} {len(sub):>10d} {wr*100:>9.2f}% {edge:>+7.2f}pp")
        else:
            print(f"  {thr:>10.2f} {len(sub):>10d} (too few)")

    print(f"\n{'-'*80}")
    print(f"Calibration: actual win rate at P threshold (SHORT)")
    print(f"{'-'*80}")
    print(f"  {'threshold':>10} {'n_signals':>10} {'actual_WR':>10} {'edge':>8}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sub = yS[pS >= thr]
        if len(sub) >= 20:
            wr = sub.mean()
            edge = (wr - 0.51) * 100
            print(f"  {thr:>10.2f} {len(sub):>10d} {wr*100:>9.2f}% {edge:>+7.2f}pp")
        else:
            print(f"  {thr:>10.2f} {len(sub):>10d} (too few)")

    # Save
    out = pd.DataFrame({
        "ts": ts_oos,
        "p_long_win": pL,
        "p_short_win": pS,
        "y_long_win": yL,
        "y_short_win": yS,
        "long_label_type": ltype_oos,
        "short_label_type": stype_oos,
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved v9 OOS -> %s", out_path)


if __name__ == "__main__":
    main()
