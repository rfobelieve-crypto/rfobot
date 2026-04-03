"""
Magnitude model training (4h horizon) with walk-forward validation.

Usage:
    python -m research.dual_model.train_magnitude_model_4h
    python -m research.dual_model.train_magnitude_model_4h --feature-set expanded
    python -m research.dual_model.train_magnitude_model_4h --ablation

Outputs saved to research/results/dual_model/:
  - magnitude_oos_predictions_{feature_set}.parquet
  - magnitude_metrics_{feature_set}.csv
  - magnitude_feature_importance_{feature_set}.csv
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
from research.dual_model.build_magnitude_labels import build_magnitude_labels
from research.dual_model.magnitude_features_v2 import (
    MAGNITUDE_GROUPS, filter_available,
)
from research.dual_model.evaluate_magnitude_4h import (
    evaluate_magnitude, print_magnitude_report,
)

logger = logging.getLogger(__name__)

# XGBoost hyperparams for regression
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "early_stopping_rounds": 30,
}


def train_magnitude_walk_forward(
    df: pd.DataFrame,
    feature_names: list[str],
    target: str = "y_abs_return",
    initial_train: int = 288,
    test_size: int = 48,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Walk-forward training for magnitude regressor.

    Parameters
    ----------
    df : Feature DataFrame with 'close' column.
    feature_names : Feature columns to use.
    target : Target column name ('y_abs_return' or 'y_vol_adj_abs').
    initial_train, test_size : Walk-forward params.

    Returns
    -------
    oos_preds : DataFrame with OOS predictions.
    metrics : Aggregated OOS metrics dict.
    importance : Feature importance DataFrame.
    """
    # Build labels
    labels = build_magnitude_labels(df)
    df = df.copy()
    for col in labels.columns:
        df[col] = labels[col]

    # Filter features
    features = filter_available(feature_names, list(df.columns))
    logger.info("Magnitude training: %d features requested, %d available",
                len(feature_names), len(features))

    splits = walk_forward_splits(len(df), initial_train=initial_train,
                                  test_size=test_size, step=test_size)
    if not splits:
        raise ValueError("Not enough data for walk-forward validation")

    all_oos = []
    all_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_mask = train_df[target].notna()
        test_mask = test_df[target].notna()

        X_train = train_df.loc[train_mask, features].fillna(0)
        y_train = train_df.loc[train_mask, target].values
        X_test = test_df.loc[test_mask, features].fillna(0)
        y_test = test_df.loc[test_mask, target].values

        if len(y_train) < 50 or len(y_test) < 5:
            logger.warning("Fold %d: skipping (train=%d, test=%d)",
                           fold_i, len(y_train), len(y_test))
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)

        oos_df = pd.DataFrame({
            "y_pred": y_pred,
            "y_true": y_test,
            "return_4h": test_df.loc[test_mask, "return_4h"].values,
            "fold": fold_i,
        }, index=test_df.loc[test_mask].index)
        all_oos.append(oos_df)

        imp = pd.Series(model.feature_importances_, index=features, name=f"fold_{fold_i}")
        all_importances.append(imp)

        from scipy.stats import spearmanr
        fold_ic, _ = spearmanr(y_test, y_pred)
        logger.info("Fold %d: train=%d test=%d IC=%.4f",
                     fold_i, len(y_train), len(y_test), fold_ic)

    if not all_oos:
        raise ValueError("No valid folds produced")

    oos_preds = pd.concat(all_oos)
    metrics = evaluate_magnitude(oos_preds["y_true"].values, oos_preds["y_pred"].values)
    metrics["n_folds"] = len(all_oos)
    metrics["n_features"] = len(features)
    metrics["target"] = target

    importance = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    importance = importance.reset_index()
    importance.columns = ["feature", "importance"]

    return oos_preds, metrics, importance


def run_single(feature_set_name: str, feature_list: list[str], df: pd.DataFrame,
               target: str = "y_abs_return"):
    """Train, evaluate, and save results for one feature set."""
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("Magnitude model — feature set: %s (%d features), target: %s",
                feature_set_name, len(feature_list), target)

    oos, metrics, importance = train_magnitude_walk_forward(df, feature_list, target=target)

    tag = feature_set_name.replace(" ", "_")
    oos.to_parquet(RESULTS_DIR / f"magnitude_oos_{tag}.parquet")
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f"magnitude_metrics_{tag}.csv", index=False)
    importance.to_csv(RESULTS_DIR / f"magnitude_importance_{tag}.csv", index=False)

    print_magnitude_report(metrics, label=f"Magnitude [{feature_set_name}]")
    return metrics


def run_ablation(df: pd.DataFrame):
    """Run old vs expanded comparison."""
    all_results = []

    for name, features in MAGNITUDE_GROUPS.items():
        try:
            metrics = run_single(name, features, df)
            metrics["feature_set"] = name
            all_results.append(metrics)
        except Exception as e:
            logger.error("Magnitude '%s' failed: %s", name, e)

    if all_results:
        comparison = pd.DataFrame(all_results)
        cols = ["feature_set", "n_features", "ic", "icir", "rmse", "mae",
                "monotonicity_score", "top_bot_ratio"]
        available_cols = [c for c in cols if c in comparison.columns]
        comparison = comparison[available_cols].sort_values("ic", ascending=False)
        comparison.to_csv(RESULTS_DIR / "magnitude_ablation_comparison.csv", index=False)

        print("\n" + "=" * 70)
        print("  MAGNITUDE ABLATION COMPARISON")
        print("=" * 70)
        print(comparison.to_string(index=False))
        print()

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train magnitude model (4h)")
    parser.add_argument("--feature-set", default="expanded",
                        choices=list(MAGNITUDE_GROUPS.keys()),
                        help="Feature set to use")
    parser.add_argument("--target", default="y_abs_return",
                        choices=["y_abs_return", "y_vol_adj_abs"],
                        help="Target variable")
    parser.add_argument("--ablation", action="store_true",
                        help="Run old vs expanded comparison")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data")
    args = parser.parse_args()

    df = load_and_cache_data(force_refresh=args.refresh)

    if args.ablation:
        run_ablation(df)
    else:
        feature_list = MAGNITUDE_GROUPS[args.feature_set]
        run_single(args.feature_set, feature_list, df, target=args.target)
