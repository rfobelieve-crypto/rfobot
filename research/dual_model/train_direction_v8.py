"""
Train v8 direction model with path-clipped target (Phase 1.2).

Mirrors train_direction_reg_4h.py exactly except for the target:
    target = y_path_clip_4h  (TP=0.5% / SL=0.3% / 4h horizon, first-touch)
    instead of y_path_ret_4h (4h TWAP endpoint)

Same XGBRegressor hyperparams, same 136 features, same
walk_forward_splits (initial_train=288, test=48, step=48, purge=4, embargo=4).

Outputs:
    research/results/dual_model/direction_v8_oos.parquet
        columns: ts, pred, y_path_clip_4h, label_type, first_touch_bar
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
    objective="reg:squarederror",
    eval_metric="mae",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=float, default=0.005,
                        help="TP distance (must match a built label cache)")
    parser.add_argument("--sl", type=float, default=0.003,
                        help="SL distance (must match a built label cache)")
    args = parser.parse_args()
    tp_bp = int(args.tp * 10000)
    sl_bp = int(args.sl * 10000)
    suffix = f"TP{tp_bp}_SL{sl_bp}"
    label_cache = LABEL_CACHE_DIR / f"labels_path_clip_{suffix}.parquet"
    out_path = RESULTS_DIR / f"direction_v8_{suffix}_oos.parquet"

    if not label_cache.exists():
        logger.error("Label cache not found: %s — run build_path_aware_labels --tp %.4f --sl %.4f",
                     label_cache, args.tp, args.sl)
        sys.exit(1)

    logger.info("Loading features...")
    features_df = pd.read_parquet(FEATURES_CACHE)
    logger.info("Features: n=%d, range %s ~ %s",
                len(features_df), features_df.index.min(), features_df.index.max())

    logger.info("Loading path-clip labels: %s", label_cache.name)
    labels_df = pd.read_parquet(label_cache)
    logger.info("Labels: n=%d, range %s ~ %s",
                len(labels_df), labels_df.index.min(), labels_df.index.max())

    # Inner join on datetime index
    df = features_df.join(labels_df[["y_path_clip_4h", "label_type",
                                       "first_touch_bar"]], how="inner")
    logger.info("Joined: n=%d", len(df))

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Using %d direction features", len(feature_cols))

    valid = df["y_path_clip_4h"].notna()
    df = df.loc[valid].copy()
    logger.info("Valid rows (label not-NaN): %d", len(df))

    X = df[feature_cols].values.astype(np.float32)
    y = df["y_path_clip_4h"].values.astype(np.float32)
    ts = df.index

    splits = walk_forward_splits(
        n_samples=len(df),
        initial_train=288,
        test_size=48,
        step=48,
        purge=4,
        embargo=4,
    )
    logger.info("Walk-forward folds: %d", len(splits))

    oos_pred = np.full(len(df), np.nan)
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]

        model = xgb.XGBRegressor(**BASE_PARAMS)
        model.fit(
            X[tr_in], y[tr_in],
            eval_set=[(X[tr_val], y[tr_val])],
            verbose=False,
        )
        oos_pred[te_idx] = model.predict(X[te_idx])

        if (fold_i + 1) % 20 == 0 or fold_i == len(splits) - 1:
            logger.info("  fold %d/%d done", fold_i + 1, len(splits))

    mask = np.isfinite(oos_pred)
    pred = oos_pred[mask]
    y_oos = y[mask]
    ts_oos = ts[mask]
    label_type_oos = df.loc[mask, "label_type"].values
    first_touch_oos = df.loc[mask, "first_touch_bar"].values

    overall_ic, _ = spearmanr(pred, y_oos)
    sign_acc = float((np.sign(pred) == np.sign(y_oos)).mean())
    logger.info(
        "v8 OVERALL OOS: n=%d  IC=%+.4f  sign_acc=%.3f  "
        "mean_pred=%+.5f  std_pred=%.5f",
        len(pred), overall_ic, sign_acc,
        float(pred.mean()), float(pred.std()),
    )

    out = pd.DataFrame({
        "ts": ts_oos,
        "pred": pred,
        "y_path_clip_4h": y_oos,
        "label_type": label_type_oos,
        "first_touch_bar": first_touch_oos,
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved v8 OOS → %s", out_path)


if __name__ == "__main__":
    main()
