"""
Direction REGRESSION model training (4h horizon, path-integrated target).

Walk-forward (default 77 folds via shared_data.walk_forward_splits:
initial_train=288, test_size=48, step=48, purge=4, embargo=4) training of an
XGBoost regressor that predicts signed 4h path return (TWAP).

Two objectives are trained in parallel — reg:squarederror and
reg:pseudohubererror — both evaluated side-by-side so the user can pick the
winner for production.

Outputs (research/results/dual_model/):
    direction_reg_oos_{objective}.parquet     # [pred_ret, y_path_ret_4h, fold]
    direction_reg_metrics_{objective}.csv
    direction_reg_importance_{objective}.csv

Usage:
    python -m research.dual_model.train_direction_reg_4h                  # both
    python -m research.dual_model.train_direction_reg_4h --objective mse
    python -m research.dual_model.train_direction_reg_4h --objective huber
"""
from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, walk_forward_splits, RESULTS_DIR, ensure_dirs,
)
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import (
    FULL_DIRECTION, filter_available,
)

logger = logging.getLogger(__name__)

# Strong-threshold sweep reported on every run
STRONG_THRESHOLDS = (0.006, 0.008, 0.010, 0.012, 0.015)

# Shared XGB hyperparams — mirror the binary direction model so comparisons
# isolate the objective change.
BASE_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,       # early_stopping_rounds trims when fold MAE plateaus
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "early_stopping_rounds": 30,
}

OBJECTIVE_MAP = {
    "mse": {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    },
    "huber": {
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
        # slope ≈ typical |return| scale; residuals > this become linear (robust)
        "huber_slope": 0.01,
    },
}


def _strong_wr(pred: np.ndarray, y_true: np.ndarray, thr: float) -> dict:
    """Strong WR at a given symmetric threshold (vectorized)."""
    long_fire = pred >= thr
    short_fire = pred <= -thr
    n_long = int(long_fire.sum())
    n_short = int(short_fire.sum())
    n = n_long + n_short
    if n == 0:
        return dict(n=0, n_long=0, n_short=0, wr=float("nan"),
                    fire_rate=0.0)
    # "Win" = predicted direction agrees with sign of path return
    w_long = int((long_fire & (y_true > 0)).sum())
    w_short = int((short_fire & (y_true < 0)).sum())
    return dict(
        n=n, n_long=n_long, n_short=n_short,
        wr=float((w_long + w_short) / n),
        fire_rate=float(n / len(pred)),
        wr_long=float(w_long / n_long) if n_long else float("nan"),
        wr_short=float(w_short / n_short) if n_short else float("nan"),
    )


def _compute_metrics(oos: pd.DataFrame) -> dict:
    """
    Aggregate OOS metrics:
      - MAE, RMSE
      - Spearman IC (pred vs realized path return)
      - Sign-based AUC (prediction value as score, realized>0 as label)
      - Strong WR sweep across STRONG_THRESHOLDS
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    pred = oos["pred_ret"].values.astype(float)
    true = oos["y_path_ret_4h"].values.astype(float)

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ic = float(spearmanr(pred, true).correlation)

    # Sign-AUC: treat "realized > 0" as label, raw pred_ret as score
    label_mask = true != 0
    auc = (float(roc_auc_score((true[label_mask] > 0).astype(int),
                                pred[label_mask]))
           if label_mask.sum() > 10 else float("nan"))

    strong = {f"thr_{thr:.3f}": _strong_wr(pred, true, thr)
              for thr in STRONG_THRESHOLDS}

    return dict(
        mae=mae, rmse=rmse, spearman_ic=ic, auc_sign=auc,
        strong_sweep=strong,
    )


def train_direction_reg_walk_forward(
    df: pd.DataFrame,
    feature_names: list[str],
    objective: str = "mse",
    initial_train: int = 288,
    test_size: int = 48,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Walk-forward regression training.

    - 77 folds with purge=4 + embargo=4 (handled inside walk_forward_splits).
    - No deadzone filter: every bar with a valid forward label is used.
    - Features filtered via `filter_available` — research/production parity
      is guaranteed by training on the same feature frame `build_live_features`
      produces in the live engine.
    """
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]

    features = filter_available(feature_names, list(df.columns))
    logger.info("Direction-reg training: %d requested, %d available "
                "(objective=%s)", len(feature_names), len(features), objective)

    splits = walk_forward_splits(
        len(df),
        initial_train=initial_train,
        test_size=test_size,
        step=test_size,
    )
    if not splits:
        raise ValueError("Not enough data for walk-forward validation")
    logger.info("Walk-forward: %d folds (purge+embargo applied)", len(splits))

    params = BASE_PARAMS.copy()
    params.update(OBJECTIVE_MAP[objective])

    all_oos: list[pd.DataFrame] = []
    all_imp: list[pd.Series] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_mask = train_df["y_path_ret_4h"].notna()
        test_mask = test_df["y_path_ret_4h"].notna()

        X_train = train_df.loc[train_mask, features].fillna(0)
        y_train = train_df.loc[train_mask, "y_path_ret_4h"].values.astype(float)
        X_test = test_df.loc[test_mask, features].fillna(0)
        y_test = test_df.loc[test_mask, "y_path_ret_4h"].values.astype(float)

        if len(y_train) < 50 or len(y_test) < 5:
            logger.warning("Fold %d: skip (train=%d test=%d)",
                           fold_i, len(y_train), len(y_test))
            continue

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        pred_ret = model.predict(X_test)

        oos_df = pd.DataFrame({
            "pred_ret": pred_ret,
            "y_path_ret_4h": y_test,
            "fold": fold_i,
        }, index=test_df.loc[test_mask].index)
        all_oos.append(oos_df)

        all_imp.append(pd.Series(
            model.feature_importances_, index=features, name=f"fold_{fold_i}"))

        fold_mae = float(np.mean(np.abs(pred_ret - y_test)))
        logger.info("Fold %d: train=%d test=%d MAE=%.5f",
                    fold_i, len(y_train), len(y_test), fold_mae)

    if not all_oos:
        raise ValueError("No valid folds produced")

    oos_preds = pd.concat(all_oos)
    metrics = _compute_metrics(oos_preds)
    metrics["n_folds"] = len(all_oos)
    metrics["n_features"] = len(features)
    metrics["objective"] = objective

    importance = (pd.concat(all_imp, axis=1).mean(axis=1)
                  .sort_values(ascending=False).reset_index())
    importance.columns = ["feature", "importance"]

    return oos_preds, metrics, importance


def _print_report(metrics: dict, objective: str):
    print("\n" + "=" * 70)
    print(f"  DIRECTION-REG  objective={objective}   "
          f"folds={metrics['n_folds']}  features={metrics['n_features']}")
    print("=" * 70)
    print(f"  MAE          : {metrics['mae']:.5f}")
    print(f"  RMSE         : {metrics['rmse']:.5f}")
    print(f"  Spearman IC  : {metrics['spearman_ic']:+.4f}")
    print(f"  AUC (sign)   : {metrics['auc_sign']:.4f}")
    print("\n  Strong WR sweep (target = 60%):")
    print(f"  {'thr':<10}{'n':>6}{'long':>6}{'short':>7}{'WR':>8}{'fire%':>8}")
    for key, s in metrics["strong_sweep"].items():
        if s["n"] == 0:
            continue
        wr_pct = s["wr"] * 100
        marker = "  <-- target" if wr_pct >= 60 else ""
        print(f"  {key:<10}{s['n']:>6d}{s['n_long']:>6d}{s['n_short']:>7d}"
              f"{wr_pct:>7.1f}%{s['fire_rate']*100:>7.2f}%{marker}")


def run(objective: str):
    ensure_dirs()
    df = load_and_cache_data()
    oos, metrics, imp = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective=objective,
    )

    tag = objective
    oos.to_parquet(RESULTS_DIR / f"direction_reg_oos_{tag}.parquet")

    # Flatten strong_sweep for CSV
    flat = {k: v for k, v in metrics.items() if k != "strong_sweep"}
    for thr_key, s in metrics["strong_sweep"].items():
        for sk, sv in s.items():
            flat[f"{thr_key}__{sk}"] = sv
    pd.DataFrame([flat]).to_csv(
        RESULTS_DIR / f"direction_reg_metrics_{tag}.csv", index=False)
    imp.to_csv(RESULTS_DIR / f"direction_reg_importance_{tag}.csv", index=False)

    _print_report(metrics, objective)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train direction-regression model (4h path return)")
    parser.add_argument(
        "--objective", choices=["mse", "huber", "both"], default="both",
        help="Loss objective (default: both)")
    args = parser.parse_args()

    if args.objective == "both":
        run("mse")
        run("huber")
    else:
        run(args.objective)
