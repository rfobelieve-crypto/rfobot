"""
Meta-Model: second-layer binary classifier for direction confidence.

Instead of predicting direction from raw features, this model learns
"when does the base model's strength prediction correlate with actual direction?"

Architecture:
  1. Walk-forward CV generates OOS base model predictions (up/down 4h/1h)
  2. Meta-features = base predictions + select raw features (BBP, absorption, regime)
  3. LightGBM binary classifier: P(actual_4h_return > 0)
  4. Only output UP/DOWN when meta_prob > threshold (else NEUTRAL)

Usage:
    python research/train_meta_model.py           # CV only
    python research/train_meta_model.py --save     # CV + save model
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

PARQUET = Path(__file__).parent / "ml_data" / "BTC_USD_1h_enhanced.parquet"
ARTIFACT_DIR = Path(__file__).parent.parent / "indicator" / "model_artifacts"
REGIME_DIR = ARTIFACT_DIR / "regime_models"

N_FOLDS = 5
HORIZON_4H = 4
HORIZON_1H = 1

# Base model params (same as train_1h_regime.py)
XGB_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.05, reg_lambda=0.8, gamma=0.1,
    random_state=42, verbosity=0, early_stopping_rounds=40,
)

# Meta-model params (lightweight — avoid overfitting on ~3000 samples)
META_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    num_leaves=8,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

# Raw features to include in meta-model (contextual signals)
META_RAW_FEATURES = [
    "bull_bear_power",       # Coinglass consensus
    "absorption_score",      # delta-price divergence
    "absorption_zscore",
    "realized_vol_20b",      # volatility regime
    "cg_taker_delta_zscore", # taker flow direction
    "cg_oi_delta_zscore",    # OI change direction
    "cg_funding_close_zscore",  # funding sentiment
    "cg_liq_imbalance",      # liquidation asymmetry
    "cg_liq_surge",          # liquidation spike
    "taker_delta_ratio",     # raw taker imbalance
    "vol_regime",            # vol vs 24h mean
]

# Direction thresholds for meta-model output
META_THRESHOLD_UP = 0.58    # P(up) > this → UP
META_THRESHOLD_DOWN = 0.42  # P(up) < this → DOWN


def load_and_prepare():
    """Load data, create targets, assign regimes."""
    df = pd.read_parquet(PARQUET).sort_index()

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    rvol = df["realized_vol_20b"].values
    rvol_safe = np.where(rvol > 1e-6, rvol, np.nan)

    # 4h targets
    fh_4h = np.full(len(df), np.nan)
    fl_4h = np.full(len(df), np.nan)
    for i in range(len(df) - HORIZON_4H):
        fh_4h[i] = np.max(high[i+1:i+1+HORIZON_4H])
        fl_4h[i] = np.min(low[i+1:i+1+HORIZON_4H])
    df["up_move_vol_adj"] = np.maximum(fh_4h / close - 1, 0) / rvol_safe
    df["down_move_vol_adj"] = np.maximum(1 - fl_4h / close, 0) / rvol_safe

    # 1h targets
    fh_1h = np.full(len(df), np.nan)
    fl_1h = np.full(len(df), np.nan)
    for i in range(len(df) - HORIZON_1H):
        fh_1h[i] = high[i+1]
        fl_1h[i] = low[i+1]
    df["up_move_1h_vol_adj"] = np.maximum(fh_1h / close - 1, 0) / rvol_safe
    df["down_move_1h_vol_adj"] = np.maximum(1 - fl_1h / close, 0) / rvol_safe

    # Binary direction target: actual 4h return > 0?
    y_ret_4h = np.full(len(df), np.nan)
    for i in range(len(df) - HORIZON_4H):
        y_ret_4h[i] = close[i + HORIZON_4H] / close[i] - 1
    df["y_direction"] = (y_ret_4h > 0).astype(float)
    df.loc[np.isnan(y_ret_4h), "y_direction"] = np.nan

    # Clip targets
    for t in ["up_move_vol_adj", "down_move_vol_adj",
              "up_move_1h_vol_adj", "down_move_1h_vol_adj"]:
        p01, p99 = df[t].quantile(0.01), df[t].quantile(0.99)
        df[t] = df[t].clip(p01, p99)

    # Regime
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime.iloc[:168] = "WARMUP"
    df["regime_name"] = regime

    # BBP (compute inline for meta-features)
    bbp_components = []
    for col, sign in [("cg_oi_delta_zscore", 1), ("cg_funding_close_zscore", -1),
                       ("cg_taker_delta_zscore", 1), ("cg_ls_ratio_zscore", -1)]:
        if col in df.columns:
            bbp_components.append(sign * df[col].clip(-3, 3) / 3)
    if bbp_components:
        df["bull_bear_power"] = pd.concat(bbp_components, axis=1).mean(axis=1).clip(-1, 1).fillna(0)
    else:
        df["bull_bear_power"] = 0

    return df


def get_base_feature_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    """Get per-target feature columns (same logic as train_1h_regime.py)."""
    # Inline the constants to avoid import issues
    EXCLUDE = {
        "open", "high", "low", "close", "volume",
        "log_return", "price_change_pct",
        "taker_buy_vol", "taker_buy_quote", "trade_count",
        "volume_ma_4h", "volume_ma_24h",
        "taker_delta_ma_4h", "taker_delta_std_4h",
        "return_skew",
        "y_return_4h",
        "up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj",
        "future_high_4h", "future_low_4h",
        "up_move_1h_vol_adj", "down_move_1h_vol_adj", "strength_1h_vol_adj",
        "future_high_1h", "future_low_1h",
        "regime_name", "y_direction", "bull_bear_power",
    }
    NEW_FEATURES = {
        "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
        "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
        "cg_oi_range", "cg_oi_range_zscore", "cg_oi_range_pct",
        "cg_oi_upper_shadow", "cg_oi_binance_share_zscore",
        "quote_vol_zscore", "quote_vol_ratio",
    }
    TARGET_EXTRA_FEATURES = {
        "up_move_vol_adj":      ["cg_oi_close_pctchg_8h", "cg_oi_range_zscore"],
        "down_move_vol_adj":    ["cg_oi_range_pct"],
        "up_move_1h_vol_adj":   [],
        "down_move_1h_vol_adj": [],
    }

    all_cols = sorted([c for c in df.columns if c not in EXCLUDE])
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    all_cols = [c for c in all_cols if c not in drop]

    base_cols = sorted([c for c in all_cols if c not in NEW_FEATURES])
    target_feat_cols = {}
    for target in ["up_move_vol_adj", "down_move_vol_adj",
                    "up_move_1h_vol_adj", "down_move_1h_vol_adj"]:
        extras = [f for f in TARGET_EXTRA_FEATURES.get(target, []) if f in all_cols]
        target_feat_cols[target] = sorted(base_cols + extras)

    return target_feat_cols


def generate_oos_predictions(df: pd.DataFrame,
                              target_feat_cols: dict) -> pd.DataFrame:
    """
    Walk-forward CV to generate OOS base model predictions.
    Returns DataFrame with OOS predictions aligned to original index.
    """
    targets_to_predict = [
        "up_move_vol_adj", "down_move_vol_adj",
        "up_move_1h_vol_adj", "down_move_1h_vol_adj",
    ]

    # Prepare clean data
    all_feature_cols = sorted(set().union(*(target_feat_cols[t] for t in targets_to_predict)))
    mask = df[targets_to_predict + ["y_direction"]].notna().all(axis=1)
    mask &= df["regime_name"] != "WARMUP"
    mask &= df[all_feature_cols].notna().all(axis=1)
    clean = df[mask].copy()

    n = len(clean)
    fold_size = n // (N_FOLDS + 1)

    # Collect OOS predictions
    oos_rows = []

    for k in range(1, N_FOLDS + 1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k + 1), n)
        if te_end <= tr_end:
            break

        train_data = clean.iloc[:tr_end]
        test_data = clean.iloc[tr_end:te_end]

        preds = {"idx": test_data.index}

        for target in targets_to_predict:
            fc = target_feat_cols[target]
            X_tr = train_data[fc].fillna(0).values
            y_tr = train_data[target].values
            X_te = test_data[fc].fillna(0).values

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_te, test_data[target].values)],
                      verbose=False)
            pred = model.predict(X_te)
            preds[f"pred_{target}"] = np.maximum(pred, 0)

        # Compute derived meta-features
        up4 = preds["pred_up_move_vol_adj"]
        dn4 = preds["pred_down_move_vol_adj"]
        up1 = preds["pred_up_move_1h_vol_adj"]
        dn1 = preds["pred_down_move_1h_vol_adj"]

        preds["strength_4h"] = up4 - dn4
        preds["strength_1h"] = up1 - dn1
        preds["strength_blend"] = 0.65 * (up4 - dn4) + 0.35 * (up1 - dn1)
        preds["up_asymmetry"] = up4 / (dn4 + 1e-6)
        preds["magnitude_4h"] = up4 + dn4
        preds["magnitude_1h"] = up1 + dn1

        # Actual direction
        preds["y_direction"] = test_data["y_direction"].values

        # Raw features for context
        for col in META_RAW_FEATURES:
            if col in test_data.columns:
                preds[col] = test_data[col].values
            else:
                preds[col] = 0.0

        # Regime encoding
        preds["is_trending_bull"] = (test_data["regime_name"] == "TRENDING_BULL").astype(float).values
        preds["is_trending_bear"] = (test_data["regime_name"] == "TRENDING_BEAR").astype(float).values
        preds["is_choppy"] = (test_data["regime_name"] == "CHOPPY").astype(float).values

        oos_rows.append(pd.DataFrame(preds).set_index("idx"))
        print(f"  Fold {k}/{N_FOLDS}: {len(test_data)} OOS bars")

    oos_df = pd.concat(oos_rows)
    print(f"  Total OOS: {len(oos_df)} bars")
    return oos_df


def get_meta_feature_cols(oos_df: pd.DataFrame) -> list[str]:
    """Return the meta-model feature column names."""
    exclude = {"y_direction"}
    return sorted([c for c in oos_df.columns if c not in exclude and oos_df[c].dtype != object])


def train_and_evaluate(oos_df: pd.DataFrame):
    """Train meta-model with walk-forward on OOS predictions."""
    meta_cols = get_meta_feature_cols(oos_df)
    print(f"\nMeta-model features ({len(meta_cols)}):")
    for c in meta_cols:
        print(f"  {c}")

    X = oos_df[meta_cols].fillna(0).values
    y = oos_df["y_direction"].values

    # Remove NaN targets
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    n = len(X)

    print(f"\nMeta-model training: {n} samples, {y.mean():.1%} positive (UP)")

    # Walk-forward CV for meta-model (3-fold since data is smaller)
    meta_folds = 3
    fold_size = n // (meta_folds + 1)
    fold_results = []

    for k in range(1, meta_folds + 1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        model = lgb.LGBMClassifier(**META_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])

        prob = model.predict_proba(X_te)[:, 1]

        # Metrics at various thresholds
        for thr_name, up_thr, dn_thr in [
            ("0.50", 0.50, 0.50),
            ("0.55", 0.55, 0.45),
            ("0.58", 0.58, 0.42),
            ("0.60", 0.60, 0.40),
            ("0.65", 0.65, 0.35),
        ]:
            pred_dir = np.where(prob > up_thr, 1,
                       np.where(prob < dn_thr, 0, -1))  # -1 = NEUTRAL
            active = pred_dir != -1
            if active.sum() > 0:
                dir_acc = accuracy_score(y_te[active], pred_dir[active])
                coverage = active.mean()
            else:
                dir_acc = 0
                coverage = 0

            if k == 1:  # Only print once per threshold
                pass  # Will print in summary

            fold_results.append({
                "fold": k,
                "threshold": thr_name,
                "dir_acc": dir_acc,
                "coverage": coverage,
                "n_active": int(active.sum()),
                "n_total": len(y_te),
                "auc": roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else 0.5,
            })

        # Feature importance
        if k == meta_folds:
            imp = pd.DataFrame({
                "feature": meta_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            print(f"\n  Top 10 meta-features (fold {k}):")
            for _, row in imp.head(10).iterrows():
                print(f"    {row['feature']:30s} imp={row['importance']:.0f}")

    # Summary
    results_df = pd.DataFrame(fold_results)
    print(f"\n{'='*70}")
    print("META-MODEL CV RESULTS")
    print(f"{'='*70}")

    for thr in ["0.50", "0.55", "0.58", "0.60", "0.65"]:
        sub = results_df[results_df["threshold"] == thr]
        mean_acc = sub["dir_acc"].mean()
        mean_cov = sub["coverage"].mean()
        mean_auc = sub["auc"].mean()
        print(f"  Threshold {thr}: dir_acc={mean_acc:.1%}  "
              f"coverage={mean_cov:.1%}  AUC={mean_auc:.3f}  "
              f"(avg {sub['n_active'].mean():.0f}/{sub['n_total'].mean():.0f} bars)")

    return results_df


def save_meta_model(oos_df: pd.DataFrame):
    """Train final meta-model on ALL OOS data and save."""
    meta_cols = get_meta_feature_cols(oos_df)

    X = oos_df[meta_cols].fillna(0).values
    y = oos_df["y_direction"].values
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    # Train on all data (no early stopping for final model)
    params = {k: v for k, v in META_PARAMS.items()}
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    # Save
    out_dir = REGIME_DIR / "meta_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.booster_.save_model(str(out_dir / "meta_lgbm.txt"))

    with open(out_dir / "meta_feature_cols.json", "w") as f:
        json.dump(meta_cols, f, indent=2)

    meta_config = {
        "threshold_up": META_THRESHOLD_UP,
        "threshold_down": META_THRESHOLD_DOWN,
        "n_training_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "meta_raw_features": META_RAW_FEATURES,
    }
    with open(out_dir / "meta_config.json", "w") as f:
        json.dump(meta_config, f, indent=2)

    print(f"\n  Meta-model saved to {out_dir}")
    print(f"  Features: {len(meta_cols)}")
    print(f"  Thresholds: UP>{META_THRESHOLD_UP}, DOWN<{META_THRESHOLD_DOWN}")

    # Print calibration check
    prob = model.predict_proba(X)[:, 1]
    for thr in [0.55, 0.58, 0.60, 0.65]:
        up_mask = prob > thr
        dn_mask = prob < (1 - thr)
        if up_mask.sum() > 0:
            up_acc = y[up_mask].mean()
            print(f"  Calibration check: P>{thr:.2f} → actual UP rate {up_acc:.1%} ({up_mask.sum()} bars)")
        if dn_mask.sum() > 0:
            dn_acc = 1 - y[dn_mask].mean()
            print(f"  Calibration check: P<{1-thr:.2f} → actual DN rate {dn_acc:.1%} ({dn_mask.sum()} bars)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print("Loading data...")
    df = load_and_prepare()

    print("\nGenerating OOS base model predictions (walk-forward)...")
    target_feat_cols = get_base_feature_cols(df)
    oos_df = generate_oos_predictions(df, target_feat_cols)

    print("\nTraining meta-model...")
    train_and_evaluate(oos_df)

    if args.save:
        print("\nSaving final meta-model...")
        save_meta_model(oos_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
