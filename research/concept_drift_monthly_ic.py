"""
Concept-drift diagnostic: Direction Model walk-forward OOS IC sliced by month.

Question being answered (from 2026-05-09 alpha-decay critical alert):
    The 5/1 retrained Direction model has 7d IC = -0.066 (negative — predicting
    inverse). Hypothesis: the training data 11/15 → 4/30 itself contains a
    concept drift in the back half (Mar/Apr), which the model fit on. If true,
    retraining on the same distribution can never fix it — we have to drop
    the bad tail.

    Mag concept drift was already documented (mistake.md 2026-04-13: Mag IC
    halved at Feb/Mar boundary). This script tests whether Direction has the
    same drift, using strict walk-forward (NOT in-sample, per mistake.md).

What it does:
    1. Load the cached feature parquet (3999 bars, 11/15 → 4/30) and build
       y_path_ret_4h labels (same as production).
    2. Run expanding-window walk-forward with purge=4 + embargo=4 (production
       defaults from shared_data.walk_forward_splits — already leakage-safe).
    3. Train one XGB regressor per fold with the SAME hyperparams that
       train_direction_reg_4h.py uses, on the SAME 136 features that the
       5/1 production model uses.
    4. Stitch all fold OOS predictions, slice by month, report:
       - Spearman IC (pred vs y_path_ret_4h)
       - Sign accuracy (sign(pred) == sign(y))
       - Mean |y| (regime intensity context)
       - Bootstrap 95% CI on IC
       - n samples per month

This is the only valid way to answer "did concept drift happen inside the
training window" — in-sample monthly IC is meaningless (mistake.md 2026-04-13).
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)

# Mirror train_direction_reg_4h.py BASE_PARAMS exactly.
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


def bootstrap_ic_ci(pred: np.ndarray, y: np.ndarray, n_boot: int = 1000,
                    seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(pred)
    if n < 30:
        return (np.nan, np.nan)
    ics = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ic_i, _ = spearmanr(pred[idx], y[idx])
        ics[i] = ic_i if np.isfinite(ic_i) else 0.0
    return float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))


def main():
    logger.info("Loading cached features from %s", CACHE)
    df = pd.read_parquet(CACHE)
    logger.info("Features: %d bars x %d cols, range %s ~ %s",
                len(df), len(df.columns), df.index.min(), df.index.max())

    labels = build_direction_reg_labels(df)
    df = df.join(labels, how="left")

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Missing %d features: %s", len(missing), missing[:10])
        feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info("Using %d direction features (production set)", len(feature_cols))

    # XGBoost handles NaN natively (production does too) — only drop rows with
    # NaN label. Dropping on feature NaN would discard the entire warmup/early
    # period and concentrate OOS into the most recent month.
    valid = df["y_path_ret_4h"].notna()
    df_valid = df.loc[valid].copy()
    feat_nan_pct = df_valid[feature_cols].isna().mean().mean() * 100
    logger.info("Valid rows (label not-NaN): %d / %d | feature NaN avg=%.1f%%",
                len(df_valid), len(df), feat_nan_pct)

    X = df_valid[feature_cols].values.astype(np.float32)
    y = df_valid["y_path_ret_4h"].values.astype(np.float32)
    ts = df_valid.index

    splits = walk_forward_splits(
        n_samples=len(df_valid),
        initial_train=288,
        test_size=48,
        step=48,
        purge=4,
        embargo=4,
    )
    logger.info("Walk-forward folds: %d", len(splits))

    oos_pred = np.full(len(df_valid), np.nan)
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        # Internal val cut for early stopping
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

        if (fold_i + 1) % 10 == 0 or fold_i == len(splits) - 1:
            logger.info("  fold %d/%d done", fold_i + 1, len(splits))

    mask = np.isfinite(oos_pred)
    pred = oos_pred[mask]
    y_oos = y[mask]
    ts_oos = ts[mask]

    overall_ic, _ = spearmanr(pred, y_oos)
    overall_lo, overall_hi = bootstrap_ic_ci(pred, y_oos)
    overall_sign = float((np.sign(pred) == np.sign(y_oos)).mean())
    logger.info(
        "OVERALL OOS: n=%d  IC=%+.4f [%.3f, %.3f]  sign_acc=%.3f  mean|y|=%.4f",
        len(pred), overall_ic, overall_lo, overall_hi, overall_sign,
        float(np.mean(np.abs(y_oos))),
    )

    df_oos = pd.DataFrame({
        "ts": ts_oos,
        "pred": pred,
        "y": y_oos,
    })
    df_oos["month"] = df_oos["ts"].dt.to_period("M")

    print("\n" + "=" * 92)
    print(f"{'Month':<10} {'n':>5} {'IC':>8} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'sign_acc':>9} {'mean|y|':>9} {'mean_y':>10}")
    print("-" * 92)
    rows = []
    for month, g in df_oos.groupby("month", sort=True):
        if len(g) < 30:
            continue
        ic_m, _ = spearmanr(g["pred"], g["y"])
        lo, hi = bootstrap_ic_ci(g["pred"].values, g["y"].values)
        sign_acc = float((np.sign(g["pred"]) == np.sign(g["y"])).mean())
        mean_abs = float(np.mean(np.abs(g["y"])))
        mean_y = float(np.mean(g["y"]))
        print(f"{str(month):<10} {len(g):>5} {ic_m:>+8.4f} "
              f"{lo:>+8.3f} {hi:>+8.3f} {sign_acc:>9.3f} "
              f"{mean_abs:>9.5f} {mean_y:>+10.5f}")
        rows.append({
            "month": str(month),
            "n": len(g),
            "ic": ic_m,
            "ic_ci_lo": lo,
            "ic_ci_hi": hi,
            "sign_acc": sign_acc,
            "mean_abs_y": mean_abs,
            "mean_y": mean_y,
        })
    print("=" * 92)
    print(
        f"{'OVERALL':<10} {len(pred):>5} {overall_ic:>+8.4f} "
        f"{overall_lo:>+8.3f} {overall_hi:>+8.3f} {overall_sign:>9.3f} "
        f"{float(np.mean(np.abs(y_oos))):>9.5f} {float(np.mean(y_oos)):>+10.5f}"
    )
    print("=" * 92)

    out_path = (
        PROJECT_ROOT / "research" / "results" / "dual_model"
        / "direction_monthly_ic.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Saved monthly IC table to %s", out_path)

    oos_path = out_path.parent / "direction_concept_drift_oos.parquet"
    df_oos.to_parquet(oos_path, index=False)
    logger.info("Saved OOS predictions to %s", oos_path)


if __name__ == "__main__":
    main()
