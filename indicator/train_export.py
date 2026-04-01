"""
Train model on API-compatible features and export artifacts.

Uses only features available from Binance klines + Coinglass API,
so the exported model can run on Railway without aggTrades.

Usage:
    python -m indicator.train_export
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import json as json_mod
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from indicator.feature_config import ALL_FEATURES, EXCLUDE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PARQUET = ROOT / "research" / "ml_data" / "BTC_USD_1h_enhanced.parquet"
ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"

HORIZON_BARS = 4  # 4h = 4 × 1h
TARGET = "y_return_4h"
N_FOLDS = 5

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.6,
    min_child_weight=3, reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=40,
)


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df = df.set_index("dt")
        elif "ts_open" in df.columns:
            df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
            df = df.set_index("dt")
    df = df.sort_index()

    if TARGET not in df.columns:
        df[TARGET] = df["close"].shift(-HORIZON_BARS) / df["close"] - 1
    df = df.iloc[:-HORIZON_BARS].copy()
    df = df.dropna(subset=[TARGET])

    # All features are already computed in the 1h enhanced parquet
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE and df[c].dtype in ("float64", "float32", "int64")]
    df[feat_cols] = df[feat_cols].ffill()

    nan_rate = df[feat_cols].isnull().mean()
    feat_cols = [c for c in feat_cols if nan_rate[c] <= 0.10]
    df = df.dropna(subset=feat_cols)

    print(f"Data: {len(df)} bars x {len(feat_cols)} API-compatible features")
    print(f"Target: mean={df[TARGET].mean():.5f}  std={df[TARGET].std():.4f}")
    return df, feat_cols


def select_features(df: pd.DataFrame, feat_cols: list[str],
                    top_k: int = 40) -> list[str]:
    if top_k >= len(feat_cols):
        return feat_cols

    n_init = int(len(df) * 0.6)
    X_init = df[feat_cols].fillna(0).values[:n_init]
    y_init = df[TARGET].values[:n_init]

    m = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbosity=0,
    )
    m.fit(X_init, y_init)

    imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
    selected = list(imp.head(top_k).index)

    print(f"\nFeature selection: {len(feat_cols)} -> {top_k}")
    for i, (name, score) in enumerate(imp.head(10).items()):
        print(f"  {i+1:2d}. {name:35s}  imp={score:.4f}")

    return selected


def train_and_export(df: pd.DataFrame, feat_cols: list[str]):
    """Walk-forward train, evaluate, export final fold's XGBoost model."""
    n = len(df)
    fold = n // (N_FOLDS + 1)
    oos = np.full(n, np.nan)

    final_xgb = None

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr = df[feat_cols].fillna(0).values[:tr_end]
        y_tr = df[TARGET].values[:tr_end]
        X_te = df[feat_cols].fillna(0).values[tr_end:te_end]
        y_te = df[TARGET].values[tr_end:te_end]

        m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = m_xgb.predict(X_te)
        oos[tr_end:te_end] = pred

        ic, _ = spearmanr(y_te, pred)
        dir_acc = (np.sign(pred) == np.sign(y_te)).mean()
        print(f"  fold {k}/{N_FOLDS}  IC={ic:+.4f}  dir={dir_acc:.1%}  n={len(y_te)}")

        final_xgb = m_xgb

    # Overall OOS evaluation
    valid = ~np.isnan(oos)
    ic_all, _ = spearmanr(df[TARGET].values[valid], oos[valid])
    dir_all = (np.sign(oos[valid]) == np.sign(df[TARGET].values[valid])).mean()
    print(f"\nOverall OOS: IC={ic_all:+.4f}  dir={dir_all:.1%}")

    # Export artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    final_xgb.save_model(str(ARTIFACT_DIR / "xgb_model.json"))

    with open(ARTIFACT_DIR / "feature_cols.json", "w") as f:
        json_mod.dump(feat_cols, f, indent=2)

    # Save rolling stats for z-score warmup (last 200 bars of OOS predictions)
    last_200 = oos[valid][-200:]
    stats = {
        "pred_mean": float(np.mean(last_200)),
        "pred_std": float(np.std(last_200)),
        "pred_history": [float(x) for x in last_200],
    }
    with open(ARTIFACT_DIR / "training_stats.json", "w") as f:
        json_mod.dump(stats, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")
    print(f"  xgb_model.json")
    print(f"  feature_cols.json ({len(feat_cols)} features)")
    print(f"  training_stats.json (warmup: {len(last_200)} bars)")


def main():
    df, feat_cols = load_data()
    feat_cols = select_features(df, feat_cols, top_k=40)
    train_and_export(df, feat_cols)


if __name__ == "__main__":
    main()
