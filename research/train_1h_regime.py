"""
Regime-Conditional XGBoost — 1h Multi-Target Training.

Trains on BTC_USD_1h_enhanced.parquet (Coinglass + kline features).
Produces regime-routed models for production inference.

Targets:
  - up_move_vol_adj   : upside potential (>= 0)
  - down_move_vol_adj : downside potential (>= 0)
  - strength_vol_adj  : net direction (up - down)

Architecture per target:
  - 1 Global model (fallback)
  - N Regime-specific models (TRENDING_BULL / TRENDING_BEAR / CHOPPY)

Exports to: indicator/model_artifacts/regime_models/

Usage:
    python research/train_1h_regime.py           # CV only
    python research/train_1h_regime.py --save     # CV + save models
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PARQUET = Path(__file__).parent / "ml_data" / "BTC_USD_1h_enhanced.parquet"
ARTIFACT_DIR = Path(__file__).parent.parent / "indicator" / "model_artifacts"
N_FOLDS = 5
HORIZON = 4  # 4h = 4 × 1h bars

TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

# ── Per-target feature selection (validated by target_feature_selection.py) ──
# Only add features that IMPROVE the specific target's ICIR across all folds.
# up_move:  +pctchg_8h (ΔICIR +1.65) +oi_range_zscore (ΔICIR +1.63)  → combo ICIR 4.04
# down_move: +oi_range_pct only (ΔICIR +0.15) — all others HARMFUL
# strength:  NO new features — all 11 candidates HARMFUL
NEW_FEATURES = {
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range", "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_upper_shadow",
    "cg_oi_binance_share_zscore",
    "quote_vol_zscore", "quote_vol_ratio",
}

TARGET_EXTRA_FEATURES: dict[str, list[str]] = {
    "up_move_vol_adj":   ["cg_oi_close_pctchg_8h", "cg_oi_range_zscore"],
    "down_move_vol_adj": ["cg_oi_range_pct"],
    "strength_vol_adj":  [],  # baseline only — all new features harmful
}

# Features to exclude (targets, OHLC, intermediate columns)
EXCLUDE = {
    "open", "high", "low", "close",
    "volume",  # raw volume — use derived features instead
    "log_return", "price_change_pct",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "volume_ma_4h", "volume_ma_24h",
    "taker_delta_ma_4h", "taker_delta_std_4h",
    "return_skew",  # high NaN rate at edges
    # targets
    "y_return_4h",
    "up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj",
    "future_high_4h", "future_low_4h",
    # regime (routing only)
    "regime_name",
}

# Relaxed hyperparameters — allow more signal through
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.05,
    reg_lambda=0.8,
    gamma=0.1,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=40,
)


# ─── Data Preparation ────────────────────────────────────────────────────────

def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """Load 1h data, create targets, assign regimes, return feature cols."""
    df = pd.read_parquet(PARQUET)
    df = df.sort_index()

    # ── Create multi-target labels ────────────────────────────────────────
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    future_high = np.full(len(df), np.nan)
    future_low = np.full(len(df), np.nan)
    for i in range(len(df) - HORIZON):
        future_high[i] = np.max(high[i + 1: i + 1 + HORIZON])
        future_low[i] = np.min(low[i + 1: i + 1 + HORIZON])

    df["future_high_4h"] = future_high
    df["future_low_4h"] = future_low

    up_move = np.maximum(future_high / close - 1, 0)
    down_move = np.maximum(1 - future_low / close, 0)

    # Volatility-adjusted
    rvol = df["realized_vol_20b"].values
    rvol_safe = np.where(rvol > 1e-6, rvol, np.nan)

    df["up_move_vol_adj"] = up_move / rvol_safe
    df["down_move_vol_adj"] = down_move / rvol_safe
    df["strength_vol_adj"] = df["up_move_vol_adj"] - df["down_move_vol_adj"]

    # Clip extreme outliers (vol-adj can spike on low-vol bars)
    for t in TARGETS:
        p01, p99 = df[t].quantile(0.01), df[t].quantile(0.99)
        df[t] = df[t].clip(p01, p99)

    # ── Regime detection (trailing-only, same as inference.py) ────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime.iloc[:168] = "WARMUP"
    df["regime_name"] = regime

    # ── Feature columns ───────────────────────────────────────────────────
    all_cols = sorted([c for c in df.columns if c not in EXCLUDE])
    df = df.dropna(subset=TARGETS)

    # Drop features with >10% NaN
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    if drop:
        print(f"  Dropping high-NaN features: {drop}")
        all_cols = [c for c in all_cols if c not in drop]

    df = df.dropna(subset=all_cols)
    df = df[df["regime_name"] != "WARMUP"]  # don't train on warmup

    # Build per-target feature sets: base (no new features) + target-specific extras
    base_cols = sorted([c for c in all_cols if c not in NEW_FEATURES])
    target_feat_cols: dict[str, list[str]] = {}
    for target in TARGETS:
        extras = [f for f in TARGET_EXTRA_FEATURES.get(target, []) if f in all_cols]
        target_feat_cols[target] = sorted(base_cols + extras)

    print(f"  Data: {len(df)} rows")
    print(f"  Base features: {len(base_cols)}")
    for t in TARGETS:
        extras = TARGET_EXTRA_FEATURES.get(t, [])
        print(f"  {t}: {len(target_feat_cols[t])} features (+{len(extras)}: {extras})")
    print(f"  Regime distribution:")
    print(f"    {df['regime_name'].value_counts().to_string()}")
    print(f"  Target stats:")
    for t in TARGETS:
        print(f"    {t}: mean={df[t].mean():.4f}, std={df[t].std():.4f}")

    return df, target_feat_cols


# ─── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    r2 = r2_score(y_true, y_pred)
    ic, _ = spearmanr(y_true, y_pred)
    pred_std = np.std(y_pred)

    dir_acc = np.nan
    if target == "strength_vol_adj":
        mask = np.abs(y_true) > 0.1
        if mask.sum() > 10:
            dir_acc = (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()

    return {"r2": r2, "ic": ic, "dir_acc": dir_acc,
            "pred_std": pred_std, "n": len(y_true)}


# ─── Walk-Forward CV ──────────────────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, target: str,
                    n_folds: int = N_FOLDS, label: str = "global") -> dict:
    n = len(X)
    fold = n // (n_folds + 1)
    fold_metrics = []
    all_oos_pred = []
    all_oos_true = []

    for k in range(1, n_folds + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        pred = model.predict(X_te)
        m = calc_metrics(y_te, pred, target)
        fold_metrics.append(m)
        all_oos_pred.extend(pred.tolist())
        all_oos_true.extend(y_te.tolist())

        dir_str = f"dir={m['dir_acc']:.1%}" if not np.isnan(m["dir_acc"]) else "dir=N/A"
        print(f"    [{label}] fold {k}/{n_folds}  "
              f"R2={m['r2']:+.4f}  IC={m['ic']:+.4f}  "
              f"pred_std={m['pred_std']:.6f}  {dir_str}  n={m['n']}")

    ics = [m["ic"] for m in fold_metrics]
    dirs = [m["dir_acc"] for m in fold_metrics if not np.isnan(m["dir_acc"])]

    oos_pred = np.array(all_oos_pred)
    oos_true = np.array(all_oos_true)

    return {
        "fold_metrics": fold_metrics,
        "mean_ic": np.mean(ics),
        "std_ic": np.std(ics),
        "icir": np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0.0,
        "mean_r2": np.mean([m["r2"] for m in fold_metrics]),
        "mean_dir_acc": np.mean(dirs) if dirs else np.nan,
        "mean_pred_std": np.mean([m["pred_std"] for m in fold_metrics]),
        "oos_pred": oos_pred,
        "oos_true": oos_true,
    }


# ─── Train final model ───────────────────────────────────────────────────────

def train_final(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    params = {k: v for k, v in XGB_PARAMS.items()
              if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


# ─── Run one target ───────────────────────────────────────────────────────────

def run_target(df: pd.DataFrame, feat_cols: list[str], target: str,
               target_name: str = "") -> dict:
    X_all = df[feat_cols].fillna(0).values
    y_all = df[target].values
    results = {}

    print(f"\n  [global]  ({len(X_all)} bars)")
    results["global"] = walk_forward_cv(X_all, y_all, target, label="global")

    for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        mask = df["regime_name"] == regime
        n = mask.sum()
        if n < 300:
            print(f"\n  [{regime}]: only {n} bars, skipping (need 300+)")
            continue
        print(f"\n  [{regime}]  ({n} bars)")
        X_r = df.loc[mask, feat_cols].fillna(0).values
        y_r = df.loc[mask, target].values
        results[regime] = walk_forward_cv(X_r, y_r, target, label=regime)

    return results


# ─── Save models ──────────────────────────────────────────────────────────────

def save_models(df: pd.DataFrame, target_feat_cols: dict[str, list[str]]):
    """Save all regime-conditional models + per-target feature_cols."""
    out_dir = ARTIFACT_DIR / "regime_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_manifest = {"targets": {}}

    for target in TARGETS:
        feat_cols = target_feat_cols[target]
        tdir = out_dir / target
        tdir.mkdir(parents=True, exist_ok=True)

        target_info = {"models": {}, "n_features": len(feat_cols)}

        # Global model
        X_all = df[feat_cols].fillna(0).values
        y_all = df[target].values
        m = train_final(X_all, y_all)
        m.save_model(str(tdir / "global_xgb.json"))
        target_info["models"]["global"] = "global_xgb.json"
        print(f"  [{target}] saved global_xgb.json ({len(feat_cols)} features)")

        # Regime models
        for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
            mask = df["regime_name"] == regime
            if mask.sum() < 300:
                continue
            X_r = df.loc[mask, feat_cols].fillna(0).values
            y_r = df.loc[mask, target].values
            mr = train_final(X_r, y_r)
            fname = regime.lower() + "_xgb.json"
            mr.save_model(str(tdir / fname))
            target_info["models"][regime] = fname
            print(f"  [{target}] saved {fname}")

        # Per-target feature_cols
        with open(tdir / "feature_cols.json", "w") as f:
            json.dump(feat_cols, f, indent=2)
        print(f"  [{target}] saved feature_cols.json ({len(feat_cols)} cols)")

        model_manifest["targets"][target] = target_info

    # Save superset feature_cols (union of all targets — for feature builder)
    all_feats = sorted(set().union(*(target_feat_cols[t] for t in TARGETS)))
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(all_feats, f, indent=2)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(model_manifest, f, indent=2)

    # Save training stats for confidence warmup
    # Use up_move - down_move OOS predictions (synthetic strength)
    up_cols = target_feat_cols["up_move_vol_adj"]
    down_cols = target_feat_cols["down_move_vol_adj"]
    X_up = df[up_cols].fillna(0).values
    X_down = df[down_cols].fillna(0).values
    y_up = df["up_move_vol_adj"].values
    y_down = df["down_move_vol_adj"].values
    n = len(df)
    fold = n // (N_FOLDS + 1)
    warmup_preds = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break
        m_up = xgb.XGBRegressor(**XGB_PARAMS)
        m_up.fit(X_up[:tr_end], y_up[:tr_end],
                 eval_set=[(X_up[tr_end:te_end], y_up[tr_end:te_end])], verbose=False)
        m_down = xgb.XGBRegressor(**XGB_PARAMS)
        m_down.fit(X_down[:tr_end], y_down[:tr_end],
                   eval_set=[(X_down[tr_end:te_end], y_down[tr_end:te_end])], verbose=False)
        up_p = m_up.predict(X_up[tr_end:te_end])
        down_p = m_down.predict(X_down[tr_end:te_end])
        strength = np.maximum(up_p, 0) - np.maximum(down_p, 0)
        warmup_preds.extend(strength.tolist())

    stats = {
        "pred_mean": float(np.mean(warmup_preds)),
        "pred_std": float(np.std(warmup_preds)),
        "pred_history": warmup_preds[-200:],
    }
    with open(out_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved training_stats.json (pred_std={stats['pred_std']:.6f})")

    print(f"\n  All artifacts saved to {out_dir}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, dict]):
    print(f"\n{'='*75}")
    print("SUMMARY — IC / ICIR / Dir-Acc / Pred-Std")
    print(f"{'='*75}")

    rows = []
    for target, res in all_results.items():
        for model_key, r in res.items():
            rows.append({
                "target": target,
                "model": model_key,
                "mean_IC": round(r["mean_ic"], 4),
                "ICIR": round(r["icir"], 4),
                "Dir_Acc": round(r["mean_dir_acc"], 4)
                           if not np.isnan(r["mean_dir_acc"]) else "—",
                "Pred_Std": round(r["mean_pred_std"], 6),
            })
    df_sum = pd.DataFrame(rows).set_index(["target", "model"])
    print(df_sum.to_string())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Save models to artifacts")
    args = ap.parse_args()

    print("Loading 1h enhanced data...")
    df, target_feat_cols = load_and_prepare()

    all_results: dict[str, dict] = {}

    for target in TARGETS:
        feat_cols = target_feat_cols[target]
        extras = TARGET_EXTRA_FEATURES.get(target, [])
        print(f"\n{'='*65}")
        print(f"TARGET: {target}  ({len(feat_cols)} features, +{extras})")
        print(f"{'='*65}")
        results = run_target(df, feat_cols, target)
        all_results[target] = results

    print_summary(all_results)

    if args.save:
        print("\nSaving models...")
        save_models(df, target_feat_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
