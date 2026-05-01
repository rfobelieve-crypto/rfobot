"""
Export MAGNITUDE production model (regression on |return_4h|).

⚠ This script does NOT export the direction model by default. Production
direction is the REGRESSION model (rolling-percentile decoder), exported
by `export_direction_reg_model.py`. Running direction export here would
overwrite the regression artifacts with the legacy binary classifier and
silently break the runtime decoder (config still claims regression while
the model file is binary — same artifact name, different schema).

Usage:
    python -m research.dual_model.export_production_models
        # magnitude-only (default — safe to run after direction-reg export)

    python -m research.dual_model.export_production_models --refresh
        # also force re-fetch raw data before training

    python -m research.dual_model.export_production_models --include-direction-binary
        # legacy: also re-export the binary direction classifier.
        # Use ONLY if rolling back to the binary architecture; otherwise this
        # corrupts the regression direction setup.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, RESULTS_DIR, ensure_dirs,
)
from research.dual_model.build_direction_labels import build_direction_labels
from research.dual_model.build_magnitude_labels import build_magnitude_labels
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from research.dual_model.magnitude_features_v2 import MAGNITUDE_GROUPS

logger = logging.getLogger(__name__)

EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"

# Best feature sets from experiment results
DIRECTION_FEATURE_SET = "+ key_4_only"   # AUC 0.6004 (best)
MAGNITUDE_FEATURE_SET = "expanded"     # IC 0.3904, ICIR 1.098

# XGBoost params (same as research)
DIR_PARAMS = {
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
}

MAG_PARAMS = {
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
}


def export_direction_model(df: pd.DataFrame):
    """Train direction model on full data and export."""
    logger.info("=== Exporting Direction Model ===")

    feature_names = ABLATION_GROUPS[DIRECTION_FEATURE_SET]
    features = filter_available(feature_names, list(df.columns))

    labels = build_direction_labels(df, k=0.5)
    df = df.copy()
    df["y_dir"] = labels["y_dir"]

    mask = df["y_dir"].notna()
    X = df.loc[mask, features].fillna(0)
    y = df.loc[mask, "y_dir"].values.astype(int)

    logger.info("Direction: %d samples, %d features, UP rate=%.1f%%",
                len(y), len(features), y.mean() * 100)

    # Handle class imbalance
    up_rate = y.mean()
    scale = (1 - up_rate) / up_rate if up_rate > 0 else 1.0

    params = DIR_PARAMS.copy()
    params["scale_pos_weight"] = scale

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    # Export
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(EXPORT_DIR / "direction_xgb.json"))

    with open(EXPORT_DIR / "direction_feature_cols.json", "w") as f:
        json.dump(features, f, indent=2)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=features)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(EXPORT_DIR / "direction_importance.csv")

    # Config
    config = {
        "feature_set": DIRECTION_FEATURE_SET,
        "n_features": len(features),
        "n_samples": len(y),
        "up_rate": float(up_rate),
        "scale_pos_weight": float(scale),
        "params": DIR_PARAMS,
    }
    with open(EXPORT_DIR / "direction_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Direction model exported: %d features → %s",
                len(features), EXPORT_DIR / "direction_xgb.json")
    return model, features


def export_magnitude_model(df: pd.DataFrame):
    """Train magnitude model on full data and export."""
    logger.info("=== Exporting Magnitude Model ===")

    feature_names = MAGNITUDE_GROUPS[MAGNITUDE_FEATURE_SET]
    features = filter_available(feature_names, list(df.columns))

    labels = build_magnitude_labels(df)
    df = df.copy()
    for col in labels.columns:
        df[col] = labels[col]

    # Target = y_vol_adj_abs (|ret_4h| / realized_vol).
    # Inference layer (indicator/inference.py ~L347-363) already multiplies
    # model output by realized_vol, so the sigma-scale target matches.
    # Walk-forward comparison 2026-04-14: vol_adj lifts Apr IC 0.13 → 0.25,
    # ICIR 1.11 → 1.54 (see research/results/mag_target_voladj_test.json).
    target = "y_vol_adj_abs"
    mask = df[target].notna()
    X = df.loc[mask, features].fillna(0)
    y = df.loc[mask, target].values

    logger.info("Magnitude: %d samples, %d features, target=%s, y_mean=%.4f",
                len(y), len(features), target, y.mean())

    model = xgb.XGBRegressor(**MAG_PARAMS)
    model.fit(X, y, verbose=False)

    # Export
    model.save_model(str(EXPORT_DIR / "magnitude_xgb.json"))

    with open(EXPORT_DIR / "magnitude_feature_cols.json", "w") as f:
        json.dump(features, f, indent=2)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=features)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(EXPORT_DIR / "magnitude_importance.csv")

    # Config
    config = {
        "feature_set": MAGNITUDE_FEATURE_SET,
        "target": target,
        "n_features": len(features),
        "n_samples": len(y),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "params": MAG_PARAMS,
    }
    with open(EXPORT_DIR / "magnitude_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Magnitude model exported: %d features → %s",
                len(features), EXPORT_DIR / "magnitude_xgb.json")
    return model, features


def build_pred_history(df: pd.DataFrame, mag_model, mag_features):
    """Build magnitude prediction history for warmup.

    Reads existing training_stats.json and updates ONLY the magnitude
    `pred_history` key — preserves `dir_pred_history` written by
    export_direction_reg_model.py (mistake.md 2026-04-19: shared-file write
    must be read-then-update).
    """
    X_mag = df[mag_features].fillna(0)
    mag_pred = mag_model.predict(X_mag)

    # Save last 300 bars of |mag_pred| for expanding percentile warmup
    history = [float(abs(m)) for m in mag_pred[-300:] if not np.isnan(m)]

    stats_path = EXPORT_DIR / "training_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["pred_history"] = history
    stats["n_bars"] = len(df)
    stats["date_range"] = f"{df.index[0]} ~ {df.index[-1]}"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Pred history saved: %d bars for warmup", len(history))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(
        description="Export magnitude (and optionally legacy binary direction) "
                    "production models. See module docstring for usage notes."
    )
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch raw data before training.")
    parser.add_argument(
        "--include-direction-binary", action="store_true",
        help="Also re-export the legacy BINARY direction classifier. "
             "WARNING: this overwrites the regression direction artifacts "
             "produced by export_direction_reg_model.py and breaks the "
             "rolling-percentile decoder. Only use when intentionally "
             "rolling back to the binary architecture.",
    )
    args = parser.parse_args()

    ensure_dirs()
    df = load_and_cache_data(force_refresh=args.refresh)

    print(f"\nData: {len(df)} bars x {len(df.columns)} cols")
    print(f"Range: {df.index[0]} → {df.index[-1]}")

    if args.include_direction_binary:
        logger.warning(
            "EXPORTING LEGACY BINARY DIRECTION CLASSIFIER — this overwrites "
            "the regression artifacts. Direction config (regression schema) "
            "and the binary model file will become inconsistent until "
            "export_direction_reg_model.py is rerun. This flag should only "
            "be used during a deliberate rollback.")
        export_direction_model(df)
    else:
        logger.info("Skipping direction export "
                    "(production direction = regression model, owned by "
                    "export_direction_reg_model.py). Pass "
                    "--include-direction-binary to override.")

    mag_model, mag_features = export_magnitude_model(df)
    build_pred_history(df, mag_model, mag_features)

    print(f"\n{'=' * 60}")
    print(f"  MAGNITUDE ARTIFACTS EXPORTED")
    print(f"  Directory: {EXPORT_DIR}")
    print(f"  Magnitude: {MAGNITUDE_FEATURE_SET} ({len(mag_features)} features)")
    if args.include_direction_binary:
        print(f"  Direction (BINARY, legacy): {DIRECTION_FEATURE_SET}  "
              f"⚠ rerun export_direction_reg_model.py to restore regression")
    else:
        print(f"  Direction: UNTOUCHED  (managed by export_direction_reg_model.py)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
