"""
Direction model training (4h horizon) with walk-forward validation.

Usage:
    python -m research.dual_model.train_direction_model_4h
    python -m research.dual_model.train_direction_model_4h --feature-set full_expanded
    python -m research.dual_model.train_direction_model_4h --ablation

Outputs saved to research/results/dual_model/:
  - direction_oos_predictions_{feature_set}.parquet
  - direction_metrics_{feature_set}.csv
  - direction_feature_importance_{feature_set}.csv
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
from research.dual_model.build_direction_labels import build_direction_labels
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, FULL_DIRECTION, filter_available,
)
from research.dual_model.evaluate_direction_4h import (
    evaluate_direction, print_direction_report,
)

logger = logging.getLogger(__name__)

# XGBoost hyperparams (conservative for small datasets)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
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


def train_direction_walk_forward(
    df: pd.DataFrame,
    feature_names: list[str],
    label_col: str = "y_dir",
    k: float = 0.5,
    initial_train: int = 288,
    test_size: int = 48,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Walk-forward training for direction classifier.

    Parameters
    ----------
    df : Feature DataFrame with 'close' column.
    feature_names : Feature columns to use.
    label_col : Direction label column name.
    k : Deadzone multiplier for label construction.
    initial_train, test_size : Walk-forward params.

    Returns
    -------
    oos_preds : DataFrame with OOS predictions (index, prob_up, y_true, fold)
    metrics : Dict with aggregated OOS metrics.
    importance : DataFrame with feature importance.
    """
    # Build labels
    labels = build_direction_labels(df, k=k)
    df = df.copy()
    df[label_col] = labels["y_dir"]
    df["return_4h"] = labels["return_4h"]

    # Filter features to those actually present
    features = filter_available(feature_names, list(df.columns))
    logger.info("Direction training: %d features requested, %d available",
                len(feature_names), len(features))

    # Walk-forward splits
    splits = walk_forward_splits(len(df), initial_train=initial_train,
                                  test_size=test_size, step=test_size)
    if not splits:
        raise ValueError("Not enough data for walk-forward validation")

    all_oos = []
    all_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Get train/test data, filter NaN labels (deadzone excluded)
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_mask = train_df[label_col].notna()
        test_mask = test_df[label_col].notna()

        X_train = train_df.loc[train_mask, features].fillna(0)
        y_train = train_df.loc[train_mask, label_col].values.astype(int)
        X_test = test_df.loc[test_mask, features].fillna(0)
        y_test = test_df.loc[test_mask, label_col].values.astype(int)

        if len(y_train) < 50 or len(y_test) < 5:
            logger.warning("Fold %d: skipping (train=%d, test=%d)",
                           fold_i, len(y_train), len(y_test))
            continue

        # Handle class imbalance
        up_rate = y_train.mean()
        scale = (1 - up_rate) / up_rate if up_rate > 0 else 1.0

        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = scale

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        prob_up = model.predict_proba(X_test)[:, 1]

        # Collect OOS predictions
        oos_df = pd.DataFrame({
            "prob_up": prob_up,
            "y_true": y_test,
            "return_4h": test_df.loc[test_mask, "return_4h"].values,
            "fold": fold_i,
        }, index=test_df.loc[test_mask].index)
        all_oos.append(oos_df)

        # Feature importance
        imp = pd.Series(model.feature_importances_, index=features, name=f"fold_{fold_i}")
        all_importances.append(imp)

        fold_auc = evaluate_direction(y_test, prob_up).get("roc_auc", 0)
        logger.info("Fold %d: train=%d test=%d AUC=%.4f",
                     fold_i, len(y_train), len(y_test), fold_auc)

    if not all_oos:
        raise ValueError("No valid folds produced")

    # Combine OOS predictions
    oos_preds = pd.concat(all_oos)

    # Aggregate metrics
    metrics = evaluate_direction(oos_preds["y_true"].values, oos_preds["prob_up"].values)
    metrics["n_folds"] = len(all_oos)
    metrics["n_features"] = len(features)

    # Average feature importance
    importance = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    importance = importance.reset_index()
    importance.columns = ["feature", "importance"]

    return oos_preds, metrics, importance


def run_single(feature_set_name: str, feature_list: list[str], df: pd.DataFrame):
    """Train, evaluate, and save results for one feature set."""
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("Direction model — feature set: %s (%d features)",
                feature_set_name, len(feature_list))

    oos, metrics, importance = train_direction_walk_forward(df, feature_list)

    # Save
    tag = feature_set_name.replace(" ", "_").replace("+", "plus")
    oos.to_parquet(RESULTS_DIR / f"direction_oos_{tag}.parquet")
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f"direction_metrics_{tag}.csv", index=False)
    importance.to_csv(RESULTS_DIR / f"direction_importance_{tag}.csv", index=False)

    print_direction_report(metrics, label=f"Direction [{feature_set_name}]")
    return metrics


def run_ablation(df: pd.DataFrame):
    """Run all ablation groups and compare."""
    all_results = []

    for name, features in ABLATION_GROUPS.items():
        try:
            metrics = run_single(name, features, df)
            metrics["feature_set"] = name
            all_results.append(metrics)
        except Exception as e:
            logger.error("Ablation '%s' failed: %s", name, e)

    if all_results:
        comparison = pd.DataFrame(all_results)
        cols = ["feature_set", "n_features", "roc_auc", "pr_auc", "accuracy",
                "top_decile_precision", "bot_decile_down_rate", "f1"]
        available_cols = [c for c in cols if c in comparison.columns]
        comparison = comparison[available_cols].sort_values("roc_auc", ascending=False)
        comparison.to_csv(RESULTS_DIR / "direction_ablation_comparison.csv", index=False)

        print("\n" + "=" * 70)
        print("  DIRECTION ABLATION COMPARISON")
        print("=" * 70)
        print(comparison.to_string(index=False))
        print()

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train direction model (4h)")
    parser.add_argument("--feature-set", default="full_expanded",
                        choices=list(ABLATION_GROUPS.keys()),
                        help="Feature set to use (default: full_expanded)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation experiment")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data")
    args = parser.parse_args()

    df = load_and_cache_data(force_refresh=args.refresh)

    if args.ablation:
        run_ablation(df)
    else:
        feature_list = ABLATION_GROUPS[args.feature_set]
        run_single(args.feature_set, feature_list, df)
