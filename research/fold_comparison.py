"""
Per-fold comparison: old features (69) vs new features (80).
Uses the SAME dataset and fold splits for fair comparison.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PARQUET = Path("research/ml_data/BTC_USD_1h_enhanced.parquet")
N_FOLDS = 5
HORIZON = 4

TARGETS = ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]

# Old feature columns to exclude from ALL features
NEW_FEATURES = {
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range", "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_upper_shadow",
    "cg_oi_binance_share_zscore",
    "quote_vol_zscore", "quote_vol_ratio",
}

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
    "regime_name",
}

XGB_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.05, reg_lambda=0.8, gamma=0.1,
    random_state=42, verbosity=0, early_stopping_rounds=40,
)


def load_data():
    df = pd.read_parquet(PARQUET)
    df = df.sort_index()

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
    rvol = df["realized_vol_20b"].values
    rvol_safe = np.where(rvol > 1e-6, rvol, np.nan)

    df["up_move_vol_adj"] = up_move / rvol_safe
    df["down_move_vol_adj"] = down_move / rvol_safe
    df["strength_vol_adj"] = df["up_move_vol_adj"] - df["down_move_vol_adj"]

    for t in TARGETS:
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

    # ALL feature cols (new model)
    all_cols = sorted([c for c in df.columns if c not in EXCLUDE])
    # OLD feature cols (exclude new ones)
    old_cols = [c for c in all_cols if c not in NEW_FEATURES]

    df = df.dropna(subset=TARGETS)

    # Drop high NaN
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    all_cols = [c for c in all_cols if c not in drop]
    old_cols = [c for c in old_cols if c not in drop]

    df = df.dropna(subset=all_cols)
    df = df[df["regime_name"] != "WARMUP"]

    return df, old_cols, all_cols


def run_cv(X, y, target):
    """Run walk-forward CV, return per-fold IC list."""
    n = len(X)
    fold = n // (N_FOLDS + 1)
    fold_ics = []
    fold_dir_accs = []

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        pred = model.predict(X_te)
        ic, _ = spearmanr(y_te, pred)
        fold_ics.append(ic)

        if target == "strength_vol_adj":
            mask = np.abs(y_te) > 0.1
            if mask.sum() > 10:
                fold_dir_accs.append(
                    (np.sign(pred[mask]) == np.sign(y_te[mask])).mean()
                )
            else:
                fold_dir_accs.append(np.nan)
        else:
            fold_dir_accs.append(np.nan)

    return fold_ics, fold_dir_accs


def main():
    df, old_cols, new_cols = load_data()
    print(f"Data: {len(df)} rows")
    print(f"Old features: {len(old_cols)}")
    print(f"New features: {len(new_cols)}")
    print(f"Added: {sorted(set(new_cols) - set(old_cols))}")
    print()

    for target in TARGETS:
        y = df[target].values

        X_old = df[old_cols].fillna(0).values
        X_new = df[new_cols].fillna(0).values

        old_ics, old_dirs = run_cv(X_old, y, target)
        new_ics, new_dirs = run_cv(X_new, y, target)

        print(f"{'='*75}")
        print(f"TARGET: {target}")
        print(f"{'='*75}")
        print(f"{'Fold':>6s}  {'Old IC':>8s}  {'New IC':>8s}  {'Delta':>8s}  {'Better?':>8s}")
        print(f"{'-'*50}")

        improvements = 0
        for k in range(len(old_ics)):
            delta = new_ics[k] - old_ics[k]
            better = "YES" if delta > 0 else "no"
            if delta > 0:
                improvements += 1
            print(f"  {k+1:>3d}   {old_ics[k]:+.4f}   {new_ics[k]:+.4f}   {delta:+.4f}   {better:>6s}")

        old_mean = np.mean(old_ics)
        new_mean = np.mean(new_ics)
        old_std = np.std(old_ics)
        new_std = np.std(new_ics)
        old_icir = old_mean / old_std if old_std > 0 else 0
        new_icir = new_mean / new_std if new_std > 0 else 0

        print(f"{'-'*50}")
        print(f"  Mean  {old_mean:+.4f}   {new_mean:+.4f}   {new_mean-old_mean:+.4f}")
        print(f"  Std   {old_std:.4f}   {new_std:.4f}")
        print(f"  ICIR  {old_icir:+.4f}   {new_icir:+.4f}   {new_icir-old_icir:+.4f}")
        print(f"  Folds improved: {improvements}/{len(old_ics)}")

        # Direction accuracy for strength
        if target == "strength_vol_adj":
            print(f"\n  Direction Accuracy:")
            print(f"  {'Fold':>6s}  {'Old Dir':>8s}  {'New Dir':>8s}  {'Delta':>8s}")
            for k in range(len(old_dirs)):
                if not np.isnan(old_dirs[k]):
                    d = new_dirs[k] - old_dirs[k]
                    print(f"    {k+1:>3d}   {old_dirs[k]:.1%}   {new_dirs[k]:.1%}   {d:+.1%}")

        # Positive IC fold count
        old_pos = sum(1 for ic in old_ics if ic > 0)
        new_pos = sum(1 for ic in new_ics if ic > 0)
        print(f"\n  Positive IC folds: {old_pos} → {new_pos}")
        print()


if __name__ == "__main__":
    main()
