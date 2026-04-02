"""
Test direction derivation fixes for the up-down architecture.

Problem: raw (up_pred - down_pred) is biased because targets have different means.
Tests multiple normalization approaches to fix direction accuracy.
"""
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

    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime.iloc[:168] = "WARMUP"
    df["regime_name"] = regime
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
    for t in ["up_move_vol_adj", "down_move_vol_adj"]:
        extras = [f for f in TARGET_EXTRA_FEATURES.get(t, []) if f in all_cols]
        target_feat_cols[t] = sorted(base_cols + extras)

    return df, target_feat_cols


def get_oos_predictions(df, target_feat_cols):
    """Walk-forward OOS predictions for up and down models."""
    n = len(df)
    fold = n // (N_FOLDS + 1)

    up_oos = np.full(n, np.nan)
    down_oos = np.full(n, np.nan)

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        for target in ["up_move_vol_adj", "down_move_vol_adj"]:
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

    return up_oos, down_oos


def eval_direction(strength, y_ret, regimes, label, deadzone=0.15):
    """Evaluate direction accuracy for a strength signal."""
    pred_dir = np.where(strength > deadzone, "UP",
               np.where(strength < -deadzone, "DOWN", "NEUTRAL"))
    actual_dir = np.where(y_ret > 0, "UP", "DOWN")

    nn = pred_dir != "NEUTRAL"
    if nn.sum() == 0:
        return

    dir_acc = (pred_dir[nn] == actual_dir[nn]).mean()
    ic, _ = spearmanr(strength, y_ret)
    n_up = (pred_dir == "UP").sum()
    n_down = (pred_dir == "DOWN").sum()
    n_neut = (pred_dir == "NEUTRAL").sum()
    n_total = len(strength)

    print(f"\n  [{label}]  deadzone={deadzone:.2f}")
    print(f"    DIR ACC: {dir_acc:.1%}  |  IC: {ic:+.4f}")
    print(f"    UP: {n_up} ({n_up/n_total:.0%})  DOWN: {n_down} ({n_down/n_total:.0%})  "
          f"NEUTRAL: {n_neut} ({n_neut/n_total:.0%})")

    # Per-regime
    for regime in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
        rm = regimes == regime
        if rm.sum() < 30:
            continue
        rnn = (pred_dir[rm] != "NEUTRAL")
        if rnn.sum() > 0:
            racc = (pred_dir[rm][rnn] == actual_dir[rm][rnn]).mean()
        else:
            racc = float('nan')
        ric, _ = spearmanr(strength[rm], y_ret[rm])
        print(f"    {regime:20s}: dir_acc={racc:.1%}, IC={ric:+.4f}")

    # Calibration: quintile monotonicity
    quantiles = np.percentile(strength, [0, 20, 40, 60, 80, 100])
    quantiles[-1] += 0.01
    rets = []
    for i in range(5):
        qm = (strength >= quantiles[i]) & (strength < quantiles[i+1])
        if qm.sum() > 0:
            rets.append(y_ret[qm].mean())
    monotonic = all(rets[i] <= rets[i+1] for i in range(len(rets)-1))
    print(f"    Calibration monotonic: {'YES' if monotonic else 'NO'}")
    print(f"    Quintile returns: {['%.4f%%' % (r*100) for r in rets]}")

    # Strong tier accuracy
    abs_s = np.abs(strength)
    top10 = abs_s >= np.percentile(abs_s, 90)
    t_pred = pred_dir[top10]
    t_actual = actual_dir[top10]
    t_nn = t_pred != "NEUTRAL"
    if t_nn.sum() > 0:
        t_acc = (t_pred[t_nn] == t_actual[t_nn]).mean()
        print(f"    Strong tier (top 10%): dir_acc={t_acc:.1%}, n={t_nn.sum()}")


def main():
    df, target_feat_cols = load_data()
    up_oos, down_oos = get_oos_predictions(df, target_feat_cols)

    mask = ~np.isnan(up_oos) & ~np.isnan(down_oos) & ~np.isnan(df["y_return_4h"].values)
    idx = np.where(mask)[0]

    up = up_oos[idx]
    down = down_oos[idx]
    y_ret = df["y_return_4h"].values[idx]
    regimes = df["regime_name"].values[idx]

    print(f"OOS bars: {len(idx)}")
    print(f"up_pred:   mean={up.mean():.4f}, std={up.std():.4f}")
    print(f"down_pred: mean={down.mean():.4f}, std={down.std():.4f}")
    print(f"actual:    up_target_mean={df['up_move_vol_adj'].values[idx].mean():.4f}, "
          f"down_target_mean={df['down_move_vol_adj'].values[idx].mean():.4f}")

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 0: RAW SUBTRACTION (current)")
    print(f"{'='*70}")
    raw_strength = up - down
    eval_direction(raw_strength, y_ret, regimes, "raw", deadzone=0.15)

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 1: EXPANDING Z-SCORE NORMALIZATION")
    print(f"{'='*70}")
    # Use expanding mean/std from training data (no lookahead)
    n = len(up)
    up_z = np.full(n, np.nan)
    down_z = np.full(n, np.nan)
    min_warmup = 100

    for i in range(min_warmup, n):
        up_z[i] = (up[i] - np.mean(up[:i])) / max(np.std(up[:i]), 1e-6)
        down_z[i] = (down[i] - np.mean(down[:i])) / max(np.std(down[:i]), 1e-6)

    valid = ~np.isnan(up_z)
    strength_z = up_z[valid] - down_z[valid]
    eval_direction(strength_z, y_ret[valid], regimes[valid], "z-score", deadzone=0.0)

    # Test different deadzones
    for dz in [0.1, 0.2, 0.3, 0.5]:
        eval_direction(strength_z, y_ret[valid], regimes[valid], f"z-score dz={dz}", deadzone=dz)

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 2: EXPANDING RANK NORMALIZATION")
    print(f"{'='*70}")
    # Rank within expanding window → percentile
    up_rank = np.full(n, np.nan)
    down_rank = np.full(n, np.nan)
    for i in range(min_warmup, n):
        up_rank[i] = (up[:i] < up[i]).mean()     # percentile rank
        down_rank[i] = (down[:i] < down[i]).mean()

    valid2 = ~np.isnan(up_rank)
    strength_rank = up_rank[valid2] - down_rank[valid2]
    eval_direction(strength_rank, y_ret[valid2], regimes[valid2], "rank", deadzone=0.0)
    for dz in [0.05, 0.10, 0.15, 0.20]:
        eval_direction(strength_rank, y_ret[valid2], regimes[valid2], f"rank dz={dz}", deadzone=dz)

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 3: RATIO-BASED (up / (up + down) - 0.5)")
    print(f"{'='*70}")
    total = up + down
    total_safe = np.where(total > 1e-6, total, np.nan)
    ratio_strength = up / total_safe - 0.5  # centered at 0
    valid3 = ~np.isnan(ratio_strength)
    eval_direction(ratio_strength[valid3], y_ret[valid3], regimes[valid3],
                   "ratio", deadzone=0.0)
    for dz in [0.01, 0.02, 0.05, 0.10]:
        eval_direction(ratio_strength[valid3], y_ret[valid3], regimes[valid3],
                       f"ratio dz={dz}", deadzone=dz)

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 4: EXPANDING MEAN-CENTERED SUBTRACTION")
    print(f"{'='*70}")
    # Simple: subtract expanding mean from each before comparing
    up_centered = np.full(n, np.nan)
    down_centered = np.full(n, np.nan)
    for i in range(min_warmup, n):
        up_centered[i] = up[i] - np.mean(up[:i])
        down_centered[i] = down[i] - np.mean(down[:i])

    valid4 = ~np.isnan(up_centered)
    strength_centered = up_centered[valid4] - down_centered[valid4]
    eval_direction(strength_centered, y_ret[valid4], regimes[valid4],
                   "mean-centered", deadzone=0.0)
    for dz in [0.05, 0.10, 0.15, 0.20]:
        eval_direction(strength_centered, y_ret[valid4], regimes[valid4],
                       f"mean-centered dz={dz}", deadzone=dz)

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("METHOD 5: LOG RATIO (ln(up/down))")
    print(f"{'='*70}")
    up_safe = np.maximum(up, 1e-6)
    down_safe = np.maximum(down, 1e-6)
    log_ratio = np.log(up_safe / down_safe)
    eval_direction(log_ratio, y_ret, regimes, "log-ratio", deadzone=0.0)
    for dz in [0.05, 0.10, 0.20, 0.30]:
        eval_direction(log_ratio, y_ret, regimes, f"log-ratio dz={dz}", deadzone=dz)


if __name__ == "__main__":
    main()
