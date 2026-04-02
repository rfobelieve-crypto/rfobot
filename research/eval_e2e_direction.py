"""
End-to-end evaluation: per-target models → direction accuracy + calibration.

Simulates the actual inference pipeline:
  1. Predict up_move with up_move features (71)
  2. Predict down_move with down_move features (70)
  3. direction = UP if (up - down) > deadzone, DOWN if < -deadzone, else NEUTRAL
  4. Measure: direction accuracy, calibration monotonicity, regime breakdown
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
STRENGTH_DEADZONE = 0.15

TARGETS = ["up_move_vol_adj", "down_move_vol_adj"]

NEW_FEATURES = {
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range", "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_upper_shadow",
    "cg_oi_binance_share_zscore",
    "quote_vol_zscore", "quote_vol_ratio",
}

TARGET_EXTRA_FEATURES = {
    "up_move_vol_adj":   ["cg_oi_close_pctchg_8h", "cg_oi_range_zscore"],
    "down_move_vol_adj": ["cg_oi_range_pct"],
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
    df = pd.read_parquet(PARQUET).sort_index()

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

    for t in ["up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"]:
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

    # Actual 4h return for direction accuracy
    df["y_return_4h"] = df["close"].shift(-HORIZON) / df["close"] - 1

    all_cols = sorted([c for c in df.columns if c not in EXCLUDE and c != "y_return_4h"])
    df = df.dropna(subset=["up_move_vol_adj", "down_move_vol_adj"])
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    all_cols = [c for c in all_cols if c not in drop]
    df = df.dropna(subset=all_cols)
    df = df[df["regime_name"] != "WARMUP"]

    base_cols = sorted([c for c in all_cols if c not in NEW_FEATURES])
    target_feat_cols = {}
    for t in TARGETS:
        extras = [f for f in TARGET_EXTRA_FEATURES.get(t, []) if f in all_cols]
        target_feat_cols[t] = sorted(base_cols + extras)

    return df, target_feat_cols


def main():
    df, target_feat_cols = load_data()
    print(f"Data: {len(df)} rows")

    n = len(df)
    fold = n // (N_FOLDS + 1)

    # Collect OOS predictions for both targets
    up_oos = np.full(n, np.nan)
    down_oos = np.full(n, np.nan)

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        for target in TARGETS:
            fc = target_feat_cols[target]
            X = df[fc].fillna(0).values
            y = df[target].values

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X[:tr_end], y[:tr_end],
                      eval_set=[(X[tr_end:te_end], y[tr_end:te_end])],
                      verbose=False)
            pred = model.predict(X[tr_end:te_end])

            if target == "up_move_vol_adj":
                up_oos[tr_end:te_end] = np.maximum(pred, 0)
            else:
                down_oos[tr_end:te_end] = np.maximum(pred, 0)

    # Filter to bars with both predictions
    mask = ~np.isnan(up_oos) & ~np.isnan(down_oos) & ~np.isnan(df["y_return_4h"].values)
    idx = np.where(mask)[0]

    up = up_oos[idx]
    down = down_oos[idx]
    strength = up - down
    y_ret = df["y_return_4h"].values[idx]
    regimes = df["regime_name"].values[idx]

    # Direction
    pred_dir = np.where(strength > STRENGTH_DEADZONE, "UP",
               np.where(strength < -STRENGTH_DEADZONE, "DOWN", "NEUTRAL"))
    actual_dir = np.where(y_ret > 0, "UP", "DOWN")

    # ══════════════════════════════════════════════════════════════════
    # 1. OVERALL DIRECTION ACCURACY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. DIRECTION ACCURACY")
    print(f"{'='*70}")

    # Excluding NEUTRAL
    non_neutral = pred_dir != "NEUTRAL"
    dir_acc = (pred_dir[non_neutral] == actual_dir[non_neutral]).mean()
    n_total = len(pred_dir)
    n_signal = non_neutral.sum()
    n_neutral = n_total - n_signal

    print(f"  Total bars: {n_total}")
    print(f"  Signal bars (UP/DOWN): {n_signal} ({n_signal/n_total:.1%})")
    print(f"  NEUTRAL bars: {n_neutral} ({n_neutral/n_total:.1%})")
    print(f"  Direction accuracy (signal only): {dir_acc:.1%}")

    for d in ["UP", "DOWN"]:
        m = pred_dir == d
        if m.sum() > 0:
            acc = (actual_dir[m] == d).mean()
            avg_ret = y_ret[m].mean() * 100
            print(f"  {d}: {m.sum()} bars, accuracy={acc:.1%}, avg actual return={avg_ret:+.3f}%")

    # ══════════════════════════════════════════════════════════════════
    # 2. PER-REGIME DIRECTION ACCURACY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("2. PER-REGIME BREAKDOWN")
    print(f"{'='*70}")

    for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        rm = regimes == regime
        if rm.sum() == 0:
            continue
        r_pred = pred_dir[rm]
        r_actual = actual_dir[rm]
        r_nn = r_pred != "NEUTRAL"
        if r_nn.sum() > 0:
            r_acc = (r_pred[r_nn] == r_actual[r_nn]).mean()
        else:
            r_acc = float('nan')
        r_neutral_pct = (~r_nn).sum() / rm.sum()
        print(f"  {regime:20s}: {rm.sum():4d} bars, "
              f"dir_acc={r_acc:.1%}, neutral_rate={r_neutral_pct:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # 3. STRENGTH IC (composite up-down as predictor)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. STRENGTH IC (up_pred - down_pred vs actual return)")
    print(f"{'='*70}")

    ic, _ = spearmanr(strength, y_ret)
    print(f"  Overall Spearman IC: {ic:+.4f}")

    for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        rm = regimes == regime
        if rm.sum() > 30:
            r_ic, _ = spearmanr(strength[rm], y_ret[rm])
            print(f"  {regime:20s}: IC={r_ic:+.4f} ({rm.sum()} bars)")

    # ══════════════════════════════════════════════════════════════════
    # 4. CALIBRATION MONOTONICITY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("4. CALIBRATION — strength quintiles vs actual return")
    print(f"{'='*70}")

    # Quintile bins
    quantiles = np.percentile(strength, [0, 20, 40, 60, 80, 100])
    quantiles[-1] += 0.01  # include max
    labels = ["Q1 (most bearish)", "Q2", "Q3 (neutral)", "Q4", "Q5 (most bullish)"]

    print(f"\n  {'Quintile':22s} {'N':>5s} {'Avg Strength':>14s} {'Avg Return':>12s} {'Dir Acc':>9s}")
    print(f"  {'-'*65}")

    monotonic = True
    prev_ret = None
    for i in range(5):
        qm = (strength >= quantiles[i]) & (strength < quantiles[i+1])
        if qm.sum() == 0:
            continue
        avg_s = strength[qm].mean()
        avg_r = y_ret[qm].mean() * 100
        # Direction accuracy for this quintile
        qp = pred_dir[qm]
        qa = actual_dir[qm]
        qnn = qp != "NEUTRAL"
        q_acc = (qp[qnn] == qa[qnn]).mean() if qnn.sum() > 0 else float('nan')
        print(f"  {labels[i]:22s} {qm.sum():5d} {avg_s:+14.4f} {avg_r:+11.4f}% {q_acc:8.1%}")

        if prev_ret is not None and avg_r < prev_ret:
            monotonic = False
        prev_ret = avg_r

    print(f"\n  Calibration monotonic: {'YES ✓' if monotonic else 'NO ✗'}")

    # ══════════════════════════════════════════════════════════════════
    # 5. CONFIDENCE TIERS (by |strength| percentile)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. CONFIDENCE TIERS — stronger signal → better accuracy?")
    print(f"{'='*70}")

    abs_s = np.abs(strength)
    tiers = [
        ("Weak (bottom 65%)", np.percentile(abs_s, 0), np.percentile(abs_s, 65)),
        ("Moderate (65-90%)", np.percentile(abs_s, 65), np.percentile(abs_s, 90)),
        ("Strong (top 10%)", np.percentile(abs_s, 90), np.percentile(abs_s, 100) + 0.01),
    ]

    print(f"\n  {'Tier':25s} {'N':>5s} {'Signal%':>8s} {'Dir Acc':>9s} {'Avg |Ret|':>11s}")
    print(f"  {'-'*62}")

    for name, lo, hi in tiers:
        tm = (abs_s >= lo) & (abs_s < hi)
        if tm.sum() == 0:
            continue
        t_pred = pred_dir[tm]
        t_actual = actual_dir[tm]
        t_nn = t_pred != "NEUTRAL"
        signal_pct = t_nn.sum() / tm.sum()
        t_acc = (t_pred[t_nn] == t_actual[t_nn]).mean() if t_nn.sum() > 0 else float('nan')
        avg_abs_ret = np.abs(y_ret[tm]).mean() * 100
        print(f"  {name:25s} {tm.sum():5d} {signal_pct:7.1%} {t_acc:8.1%} {avg_abs_ret:10.4f}%")

    # ══════════════════════════════════════════════════════════════════
    # 6. PER-FOLD STABILITY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("6. PER-FOLD DIRECTION ACCURACY STABILITY")
    print(f"{'='*70}")

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        fm = np.zeros(n, dtype=bool)
        fm[tr_end:te_end] = True
        fm = fm[idx]
        if fm.sum() == 0:
            continue

        f_pred = pred_dir[fm]
        f_actual = actual_dir[fm]
        f_nn = f_pred != "NEUTRAL"
        f_acc = (f_pred[f_nn] == f_actual[f_nn]).mean() if f_nn.sum() > 0 else float('nan')
        f_ic, _ = spearmanr(strength[fm], y_ret[fm])
        neutral_pct = (~f_nn).sum() / fm.sum()
        print(f"  Fold {k}: dir_acc={f_acc:.1%}, strength_IC={f_ic:+.4f}, "
              f"neutral={neutral_pct:.1%}, n={fm.sum()}")


if __name__ == "__main__":
    main()
