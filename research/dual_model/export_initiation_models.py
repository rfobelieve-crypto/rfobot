"""
Export initiation-model artifacts for production deployment.

Trains long_init + short_init XGBClassifiers on ALL available data using
first-touch labels (k=0.008, horizon=4h, trailing-breakout confirm) and
exports them to indicator/model_artifacts/dual_model/ alongside
thresholds calibrated to the winning operating point from walk-forward:

    Strong   : init_prob top-5%  AND  breakout  AND  mag_percentile >= 0.80
    Moderate : init_prob top-10% AND  breakout  AND  mag_percentile >= 0.65

Walk-forward (k=0.008, first-touch) achieved (6 months OOS):
    Long  Strong  71.4%  (42 fires)    Moderate  55.9%  (118 fires)
    Short Strong  71.4%  (42 fires)    Moderate  54.3%  (164 fires)

Usage:
    python -m research.dual_model.export_initiation_models
    python -m research.dual_model.export_initiation_models --refresh
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, ensure_dirs
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
MAG_MODEL_PATH = EXPORT_DIR / "magnitude_xgb.json"
MAG_FEATS_PATH = EXPORT_DIR / "magnitude_feature_cols.json"

BASE_FS = "+ key_4_only"
K_PCT = 0.008
HORIZON_BARS = 4
BREAKOUT_LOOKBACK = 20
MAG_ROLLING_WINDOW = 720

# Same XGB params as research/initiation_train_v2.py (scale_pos_weight REMOVED
# — first-touch labels are dense enough to stay calibrated without it).
INIT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

# Operating points locked from walk-forward experiment on 2026-04-14
STRONG_TOP_FRAC = 0.05
MODERATE_TOP_FRAC = 0.10
STRONG_MAG_PCT = 0.80
MODERATE_MAG_PCT = 0.65


def _score_mag_percentile(df: pd.DataFrame) -> pd.Series:
    """Score production MAG model and convert to rolling percentile."""
    feats = json.loads(MAG_FEATS_PATH.read_text())
    feats = [f for f in feats if f in df.columns]
    booster = xgb.Booster()
    booster.load_model(str(MAG_MODEL_PATH))
    X = df[feats].fillna(0).values
    dm = xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index)
    pct = mag.rolling(MAG_ROLLING_WINDOW, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )
    return pct


def _train_side(df: pd.DataFrame, features: list[str], label_col: str,
                side: str) -> tuple[xgb.XGBClassifier, np.ndarray]:
    mask = df[label_col].notna()
    X = df.loc[mask, features].fillna(0)
    y = df.loc[mask, label_col].values.astype(int)
    logger.info("%s: %d samples, %d features, pos_rate=%.2f%%",
                side, len(y), len(features), y.mean() * 100)

    model = xgb.XGBClassifier(**INIT_PARAMS)
    model.fit(X, y, verbose=False)
    probs = model.predict_proba(X)[:, 1]
    return model, probs


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    df = load_and_cache_data(force_refresh=args.refresh)
    df = add_initiation_features(df)

    print(f"\nData: {len(df)} bars x {len(df.columns)} cols")
    print(f"Range: {df.index[0]} → {df.index[-1]}")

    # --- Labels (first-touch) -----------------------------------------------
    cfg = InitiationLabelConfig(
        k_pct=K_PCT,
        breakout_lookback=BREAKOUT_LOOKBACK,
        use_breakout_confirm=True,
        horizon_bars=HORIZON_BARS,
    )
    labels = build_initiation_labels(df, cfg)
    df = df.copy()
    df["y_long_touch"] = labels["y_long_touch"]
    df["y_short_touch"] = labels["y_short_touch"]

    # --- Feature set --------------------------------------------------------
    base_feats = filter_available(ABLATION_GROUPS[BASE_FS], list(df.columns))
    init_feats = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    features = sorted(set(base_feats) | set(init_feats))
    logger.info("Feature set: %d base + %d init = %d total",
                len(base_feats), len(init_feats), len(features))

    # --- Train both sides on full data --------------------------------------
    long_model, long_probs = _train_side(df, features, "y_long_touch", "LONG")
    short_model, short_probs = _train_side(df, features, "y_short_touch", "SHORT")

    # --- Score MAG percentile (for threshold metadata only) -----------------
    try:
        mag_pct = _score_mag_percentile(df)
        mag_pct_summary = {
            "mag_percentile_coverage": float(mag_pct.notna().mean()),
            "mag_percentile_mean": float(mag_pct.mean(skipna=True)),
        }
    except Exception as e:
        logger.warning("MAG percentile scoring failed: %s", e)
        mag_pct_summary = {"error": str(e)}

    # --- Threshold quantiles from in-sample training preds ------------------
    # NOTE: these are upper bounds — live walk-forward thresholds ran slightly
    # lower (~0.16-0.17 for top-5%). Production uses these as starting
    # cutoffs; the indicator gate can recalibrate on rolling OOS later.
    long_mask = df["y_long_touch"].notna().values
    short_mask = df["y_short_touch"].notna().values
    long_valid = long_probs
    short_valid = short_probs

    strong_long_thr = float(np.quantile(long_valid, 1 - STRONG_TOP_FRAC))
    strong_short_thr = float(np.quantile(short_valid, 1 - STRONG_TOP_FRAC))
    mod_long_thr = float(np.quantile(long_valid, 1 - MODERATE_TOP_FRAC))
    mod_short_thr = float(np.quantile(short_valid, 1 - MODERATE_TOP_FRAC))

    # --- Export models ------------------------------------------------------
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    long_model.save_model(str(EXPORT_DIR / "initiation_long_xgb.json"))
    short_model.save_model(str(EXPORT_DIR / "initiation_short_xgb.json"))

    with open(EXPORT_DIR / "initiation_feature_cols.json", "w") as f:
        json.dump(features, f, indent=2)

    pd.Series(long_model.feature_importances_, index=features).sort_values(
        ascending=False).to_csv(EXPORT_DIR / "initiation_long_importance.csv")
    pd.Series(short_model.feature_importances_, index=features).sort_values(
        ascending=False).to_csv(EXPORT_DIR / "initiation_short_importance.csv")

    # --- initiation_config.json ---------------------------------------------
    config = {
        "label": {
            "type": "first_touch",
            "k_pct": K_PCT,
            "horizon_bars": HORIZON_BARS,
            "breakout_lookback": BREAKOUT_LOOKBACK,
            "use_breakout_confirm": True,
        },
        "features": {
            "base_set": BASE_FS,
            "n_base": len(base_feats),
            "n_init": len(init_feats),
            "n_total": len(features),
        },
        "training": {
            "n_bars": int(len(df)),
            "n_long_valid": int(long_mask.sum()),
            "n_short_valid": int(short_mask.sum()),
            "long_pos_rate": float(df.loc[long_mask, "y_long_touch"].mean()),
            "short_pos_rate": float(df.loc[short_mask, "y_short_touch"].mean()),
            "date_range": f"{df.index[0]} ~ {df.index[-1]}",
            "scale_pos_weight_used": False,
        },
        "params": INIT_PARAMS,
        "mag_summary": mag_pct_summary,
    }
    with open(EXPORT_DIR / "initiation_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # --- initiation_thresholds.json -----------------------------------------
    thresholds = {
        "strong": {
            "prob_long_threshold": strong_long_thr,
            "prob_short_threshold": strong_short_thr,
            "mag_percentile_min": STRONG_MAG_PCT,
            "require_breakout": True,
        },
        "moderate": {
            "prob_long_threshold": mod_long_thr,
            "prob_short_threshold": mod_short_thr,
            "mag_percentile_min": MODERATE_MAG_PCT,
            "require_breakout": True,
        },
        "metadata": {
            "k_pct": K_PCT,
            "horizon_bars": HORIZON_BARS,
            "breakout_lookback": BREAKOUT_LOOKBACK,
            "label_type": "first_touch",
            "strong_top_frac": STRONG_TOP_FRAC,
            "moderate_top_frac": MODERATE_TOP_FRAC,
            "expected_strong_wr_long": 0.714,
            "expected_strong_wr_short": 0.714,
            "expected_moderate_wr_long": 0.559,
            "expected_moderate_wr_short": 0.543,
            "walk_forward_source": "research/initiation_train_v2.py @ 2026-04-14",
            "trained_on": f"{df.index[0]} ~ {df.index[-1]}",
            "note": (
                "Thresholds are in-sample quantiles from full-data training. "
                "Live gate should monitor rolling OOS precision and recalibrate "
                "if drift is detected."
            ),
        },
    }
    with open(EXPORT_DIR / "initiation_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  INITIATION ARTIFACTS EXPORTED")
    print(f"  Dir: {EXPORT_DIR}")
    print(f"  Features: {len(features)}  Bars: {len(df)}")
    print(f"  Long  pos_rate={df.loc[long_mask, 'y_long_touch'].mean()*100:.2f}%")
    print(f"  Short pos_rate={df.loc[short_mask, 'y_short_touch'].mean()*100:.2f}%")
    print(f"  Strong  thr: long={strong_long_thr:.3f}  short={strong_short_thr:.3f}  mag>={STRONG_MAG_PCT}")
    print(f"  Moderate thr: long={mod_long_thr:.3f}  short={mod_short_thr:.3f}  mag>={MODERATE_MAG_PCT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
