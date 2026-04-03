"""
Direction model training pipeline with walk-forward validation.

Trains binary classifiers (XGBoost / LightGBM) to predict price direction,
using direction-specific labels and features.

Key differences from magnitude model (train_1h_regime.py):
  - Binary classification (not regression)
  - Direction-specific label strategies (deadzone, triple-barrier)
  - Direction-specific feature stack (asymmetric features)
  - AUC as primary metric (not IC/ICIR)
  - No vol-adjustment on targets
  - Train-only normalization for z-score features

Usage:
    python research/train_direction_model.py
    python research/train_direction_model.py --horizon 1 --method triple_barrier --k 0.8
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from research.direction_features import build_direction_feature_set
from research.direction_labels import build_direction_labels
from research.evaluate_direction import evaluate_direction_model, format_summary

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH = Path("research/ml_data/BTC_USD_1h_enhanced.parquet")
RESULTS_DIR = Path("research/results")
MODEL_DIR = Path("research/direction_models")

# ── XGBoost classifier parameters ───────────────────────────────────────
XGB_CLS_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.2,
    scale_pos_weight=1.0,  # adjusted per fold if imbalanced
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    verbosity=0,
    early_stopping_rounds=30,
)

# ── Columns to exclude from features ────────────────────────────────────
EXCLUDE = {
    "open", "high", "low", "close", "volume",
    "log_return", "price_change_pct",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "volume_ma_4h", "volume_ma_24h",
    "taker_delta_ma_4h", "taker_delta_std_4h",
    "return_skew",
    # magnitude targets
    "y_return_4h",
    "up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj",
    "future_high_4h", "future_low_4h",
    "up_move_1h_vol_adj", "down_move_1h_vol_adj", "strength_1h_vol_adj",
    "future_high_1h", "future_low_1h",
    "regime_name",
}


def load_data() -> pd.DataFrame:
    """Load the 1h enhanced training data."""
    df = pd.read_parquet(DATA_PATH)
    logger.info("Loaded %d bars, %d columns from %s", len(df), len(df.columns), DATA_PATH)
    return df


def prepare_features(
    df: pd.DataFrame,
    use_direction_features: bool = True,
    use_existing_features: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix.

    Parameters
    ----------
    use_direction_features : include direction-specific features
    use_existing_features : include existing magnitude features (baseline)
    """
    parts = []

    if use_existing_features:
        # Use existing features (excluding EXCLUDE set and high-NaN columns)
        existing_cols = [c for c in df.columns if c not in EXCLUDE]
        existing = df[existing_cols].copy()
        # Drop columns with >15% NaN
        nan_rate = existing.isna().mean()
        keep = nan_rate[nan_rate < 0.15].index.tolist()
        parts.append(existing[keep])

    if use_direction_features:
        dir_feats = build_direction_feature_set(df)
        parts.append(dir_feats)

    if not parts:
        raise ValueError("No features selected")

    X = pd.concat(parts, axis=1)

    # Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]

    # Remove any remaining target-like columns
    target_patterns = ["dir_", "label_", "y_return", "future_high", "future_low",
                       "up_move", "down_move", "strength_"]
    drop_cols = [c for c in X.columns if any(c.startswith(p) for p in target_patterns)]
    X = X.drop(columns=drop_cols, errors="ignore")

    logger.info("Feature matrix: %d bars × %d features", *X.shape)
    return X


def assign_regime(df: pd.DataFrame) -> np.ndarray:
    """Trailing-only regime classification (same as train_1h_regime.py)."""
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    ret_24h = close.pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

    regime = np.full(len(df), "CHOPPY", dtype=object)
    regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
    regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"
    regime[:168] = "WARMUP"
    return regime


def walk_forward_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    regime: np.ndarray | None = None,
    n_folds: int = 5,
    model_type: str = "xgboost",
) -> dict:
    """
    Walk-forward cross-validation for binary classification.

    Returns dict with OOS predictions, fold metrics, feature importance.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)

    all_oos_idx = []
    all_oos_prob = []
    all_oos_true = []
    all_oos_regime = []
    fold_metrics = []
    feature_importance = np.zeros(X.shape[1])

    feature_cols = X.columns.tolist()

    for k in range(1, n_folds + 1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr = X.iloc[:tr_end].values
        y_tr = y[:tr_end]
        X_te = X.iloc[tr_end:te_end].values
        y_te = y[tr_end:te_end]

        # Drop NaN labels
        tr_mask = ~np.isnan(y_tr)
        te_mask = ~np.isnan(y_te)

        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_te_clean, y_te_clean = X_te[te_mask], y_te[te_mask]

        if len(y_tr) < 100 or len(y_te_clean) < 30:
            logger.warning("Fold %d: insufficient data (train=%d, test=%d), skip",
                           k, len(y_tr), len(y_te_clean))
            continue

        # Adjust scale_pos_weight for class imbalance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        spw = n_neg / max(n_pos, 1)
        spw = np.clip(spw, 0.5, 2.0)  # don't over-correct

        # Fill NaN features with 0 (after train/test split — no leakage)
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te_clean = np.nan_to_num(X_te_clean, nan=0.0)

        if model_type == "xgboost":
            params = {**XGB_CLS_PARAMS, "scale_pos_weight": spw}
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te_clean, y_te_clean)], verbose=False)
            prob = model.predict_proba(X_te_clean)[:, 1]

            # Feature importance (gain)
            try:
                imp = model.feature_importances_
                feature_importance += imp
            except Exception:
                pass

        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                logger.error("LightGBM not installed, falling back to XGBoost")
                return walk_forward_cv(X, y, regime, n_folds, "xgboost")

            lgb_params = {
                "n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
                "subsample": 0.8, "colsample_bytree": 0.6,
                "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 1.0,
                "scale_pos_weight": spw, "objective": "binary",
                "metric": "auc", "random_state": 42, "verbosity": -1,
                "n_jobs": 1,
            }
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_te_clean, y_te_clean)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            prob = model.predict_proba(X_te_clean)[:, 1]

            try:
                imp = model.feature_importances_.astype(float)
                feature_importance += imp
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Metrics
        if len(np.unique(y_te_clean)) >= 2:
            auc = roc_auc_score(y_te_clean, prob)
        else:
            auc = np.nan

        acc = (prob >= 0.5).astype(int)
        accuracy = (acc == y_te_clean).mean()

        fold_metrics.append({
            "fold": k,
            "train_size": len(y_tr),
            "test_size": len(y_te_clean),
            "train_up_rate": float(y_tr.mean()),
            "test_up_rate": float(y_te_clean.mean()),
            "auc": float(auc),
            "accuracy": float(accuracy),
        })

        logger.info("Fold %d: train=%d, test=%d, AUC=%.4f, acc=%.4f, up_rate=%.3f",
                     k, len(y_tr), len(y_te_clean), auc, accuracy, y_te_clean.mean())

        # Collect OOS
        te_global_idx = np.arange(tr_end, te_end)[te_mask]
        all_oos_idx.extend(te_global_idx)
        all_oos_prob.extend(prob)
        all_oos_true.extend(y_te_clean)
        if regime is not None:
            all_oos_regime.extend(regime[tr_end:te_end][te_mask])

    # Aggregate
    oos_prob = np.array(all_oos_prob)
    oos_true = np.array(all_oos_true)
    oos_regime = np.array(all_oos_regime) if all_oos_regime else None

    # Overall OOS AUC
    if len(np.unique(oos_true)) >= 2:
        overall_auc = roc_auc_score(oos_true, oos_prob)
    else:
        overall_auc = np.nan

    # Feature importance ranking
    feature_importance /= max(len(fold_metrics), 1)
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "fold_metrics": fold_metrics,
        "overall_auc": overall_auc,
        "overall_accuracy": float(((oos_prob >= 0.5).astype(int) == oos_true).mean())
            if len(oos_true) > 0 else np.nan,
        "oos_prob": oos_prob,
        "oos_true": oos_true,
        "oos_regime": oos_regime,
        "oos_idx": np.array(all_oos_idx),
        "feature_importance": fi_df,
        "n_features": len(feature_cols),
    }


def run_experiment(
    method: str = "deadzone",
    horizon: int = 4,
    k: float = 0.5,
    use_direction_features: bool = True,
    use_existing_features: bool = True,
    model_type: str = "xgboost",
    n_folds: int = 5,
) -> dict:
    """
    Run a single direction prediction experiment.

    Returns complete results dict with metrics, OOS predictions, etc.
    """
    # Load data
    df = load_data()

    # Build labels
    labels = build_direction_labels(
        df, method=method, horizon_bars=horizon,
        vol_col="realized_vol_20b", k=k,
    )
    y = labels.values

    # Build features
    X = prepare_features(df, use_direction_features, use_existing_features)

    # Regime
    regime = assign_regime(df)

    # Filter warmup
    warmup_mask = regime != "WARMUP"
    X = X[warmup_mask]
    y = y[warmup_mask]
    regime = regime[warmup_mask]

    # Label stats
    valid = ~np.isnan(y)
    n_valid = valid.sum()
    n_up = (y[valid] == 1).sum()
    n_down = (y[valid] == 0).sum()
    n_neutral = len(y) - n_valid

    logger.info("Labels: %d valid (UP=%d, DOWN=%d), %d neutral/dropped, balance=%.1f%%",
                n_valid, n_up, n_down, n_neutral, n_up / max(n_valid, 1) * 100)

    # Walk-forward CV
    cv_results = walk_forward_cv(X, y, regime, n_folds, model_type)

    # Full evaluation
    summary, detail_df = evaluate_direction_model(
        cv_results["oos_true"], cv_results["oos_prob"],
        regime=cv_results["oos_regime"],
    )

    # Combine everything
    result = {
        "config": {
            "method": method,
            "horizon": horizon,
            "k": k,
            "use_direction_features": use_direction_features,
            "use_existing_features": use_existing_features,
            "model_type": model_type,
        },
        "label_stats": {
            "n_valid": int(n_valid),
            "n_up": int(n_up),
            "n_down": int(n_down),
            "n_neutral": int(n_neutral),
            "up_rate": float(n_up / max(n_valid, 1)),
        },
        "cv_results": cv_results,
        "evaluation": summary,
        "detail_df": detail_df,
    }

    return result


def main():
    ap = argparse.ArgumentParser(description="Train direction prediction model")
    ap.add_argument("--method", default="deadzone",
                    choices=["raw_sign", "deadzone", "triple_barrier"])
    ap.add_argument("--horizon", type=int, default=4, help="Horizon in bars (1h bars)")
    ap.add_argument("--k", type=float, default=0.5, help="Vol multiplier for threshold/barrier")
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    ap.add_argument("--dir-features", action="store_true", default=True,
                    help="Include direction-specific features")
    ap.add_argument("--no-dir-features", dest="dir_features", action="store_false")
    ap.add_argument("--no-existing", action="store_true",
                    help="Exclude existing magnitude features")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("Direction Model Training")
    logger.info("  method=%s, horizon=%d, k=%.2f, model=%s",
                args.method, args.horizon, args.k, args.model)
    logger.info("  dir_features=%s, existing_features=%s",
                args.dir_features, not args.no_existing)
    logger.info("=" * 60)

    result = run_experiment(
        method=args.method,
        horizon=args.horizon,
        k=args.k,
        use_direction_features=args.dir_features,
        use_existing_features=not args.no_existing,
        model_type=args.model,
        n_folds=args.folds,
    )

    # Print summary
    print(format_summary(result["evaluation"]))

    # Print fold details
    print("\nFold Details:")
    for fm in result["cv_results"]["fold_metrics"]:
        print(f"  Fold {fm['fold']}: AUC={fm['auc']:.4f}, acc={fm['accuracy']:.4f}, "
              f"test={fm['test_size']}")

    # Print top features
    fi = result["cv_results"]["feature_importance"]
    print(f"\nTop 20 Features ({result['cv_results']['n_features']} total):")
    for _, row in fi.head(20).iterrows():
        print(f"  {row['importance']:8.1f}  {row['feature']}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_h{args.horizon}_k{args.k}"

    # Save OOS predictions
    oos_df = pd.DataFrame({
        "y_true": result["cv_results"]["oos_true"],
        "y_prob": result["cv_results"]["oos_prob"],
    })
    if result["cv_results"]["oos_regime"] is not None:
        oos_df["regime"] = result["cv_results"]["oos_regime"]
    oos_df.to_parquet(RESULTS_DIR / f"direction_oos_{tag}.parquet")

    # Save feature importance
    fi.to_csv(RESULTS_DIR / f"direction_fi_{tag}.csv", index=False)

    # Save summary
    summary = {**result["config"], **result["label_stats"], **result["evaluation"]}
    with open(RESULTS_DIR / f"direction_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to %s/direction_*_%s.*", RESULTS_DIR, tag)


if __name__ == "__main__":
    main()
