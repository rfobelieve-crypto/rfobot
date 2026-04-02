"""
Compare direction accuracy: dual-model (up-down) vs direct return model.

Tests whether training directly on y_return_4h gives better direction signal.
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

TARGET_EXTRA = {
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

    # Direct return target
    df["y_return_4h"] = df["close"].shift(-HORIZON) / df["close"] - 1

    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    regime = pd.Series("CHOPPY", index=df.index)
    regime[(vol_pct > 0.6) & (ret_24h > 0.005)] = "TRENDING_BULL"
    regime[(vol_pct > 0.6) & (ret_24h < -0.005)] = "TRENDING_BEAR"
    regime.iloc[:168] = "WARMUP"
    df["regime_name"] = regime

    all_cols = sorted([c for c in df.columns if c not in EXCLUDE and c != "y_return_4h"])
    df = df.dropna(subset=["up_move_vol_adj", "down_move_vol_adj", "y_return_4h"])
    nan_rate = df[all_cols].isnull().mean()
    drop = list(nan_rate[nan_rate > 0.10].index)
    all_cols = [c for c in all_cols if c not in drop]
    df = df.dropna(subset=all_cols)
    df = df[df["regime_name"] != "WARMUP"]

    base_cols = sorted([c for c in all_cols if c not in NEW_FEATURES])
    target_feat_cols = {}
    for t in ["up_move_vol_adj", "down_move_vol_adj"]:
        extras = [f for f in TARGET_EXTRA.get(t, []) if f in all_cols]
        target_feat_cols[t] = sorted(base_cols + extras)

    return df, base_cols, all_cols, target_feat_cols


def eval_signal(pred, y_ret, regimes, label, deadzones=[0.0]):
    """Evaluate prediction as direction signal."""
    ic, _ = spearmanr(pred, y_ret)
    actual_dir = np.where(y_ret > 0, "UP", "DOWN")

    print(f"\n  [{label}] IC={ic:+.4f}")

    for dz in deadzones:
        if dz == 0:
            pred_dir = np.where(pred > 0, "UP", "DOWN")
        else:
            pred_dir = np.where(pred > dz, "UP",
                       np.where(pred < -dz, "DOWN", "NEUTRAL"))

        nn = pred_dir != "NEUTRAL"
        n_total = len(pred)
        if nn.sum() == 0:
            print(f"    dz={dz}: all NEUTRAL")
            continue

        dir_acc = (pred_dir[nn] == actual_dir[nn]).mean()
        signal_pct = nn.sum() / n_total

        # Per-regime
        regime_str = []
        for r in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
            rm = regimes == r
            rnn = (pred_dir[rm] != "NEUTRAL")
            if rnn.sum() > 0:
                racc = (pred_dir[rm][rnn] == actual_dir[rm][rnn]).mean()
                regime_str.append(f"{r[:4]}={racc:.1%}")

        # Quantile calibration
        qs = np.percentile(pred, [0, 20, 40, 60, 80, 100])
        qs[-1] += 0.001
        q_rets = []
        for i in range(5):
            qm = (pred >= qs[i]) & (pred < qs[i+1])
            if qm.sum() > 0:
                q_rets.append(y_ret[qm].mean() * 100)
        mono = all(q_rets[i] <= q_rets[i+1] for i in range(len(q_rets)-1))

        # Strong tier
        abs_p = np.abs(pred)
        top10 = abs_p >= np.percentile(abs_p, 90)
        tnn = pred_dir[top10] != "NEUTRAL"
        t_acc = (pred_dir[top10][tnn] == actual_dir[top10][tnn]).mean() if tnn.sum() > 0 else float('nan')

        print(f"    dz={dz:.3f}: acc={dir_acc:.1%} ({signal_pct:.0%} signaled) "
              f"| {' '.join(regime_str)} | mono={'Y' if mono else 'N'} "
              f"| strong={t_acc:.1%}")
        print(f"    Quintile rets: {['%.4f%%' % r for r in q_rets]}")


def main():
    df, base_cols, all_cols, target_feat_cols = load_data()
    n = len(df)
    fold = n // (N_FOLDS + 1)

    y_ret = df["y_return_4h"].values
    regimes = df["regime_name"].values

    # ── Model A: Dual up-down (current) ──────────────────────────────
    up_oos = np.full(n, np.nan)
    down_oos = np.full(n, np.nan)
    ret_oos_base = np.full(n, np.nan)    # direct return on base features
    ret_oos_all = np.full(n, np.nan)     # direct return on all features
    strength_oos = np.full(n, np.nan)    # strength_vol_adj direct model

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        # Up-down models (per-target features)
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

        # Direct return model (base features, no new OI features)
        X_base = df[base_cols].fillna(0).values
        y_r = y_ret
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_base[:tr_end], y_r[:tr_end],
                  eval_set=[(X_base[tr_end:te_end], y_r[tr_end:te_end])],
                  verbose=False)
        ret_oos_base[tr_end:te_end] = model.predict(X_base[tr_end:te_end])

        # Direct return model (all features)
        X_all = df[all_cols].fillna(0).values
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_all[:tr_end], y_r[:tr_end],
                  eval_set=[(X_all[tr_end:te_end], y_r[tr_end:te_end])],
                  verbose=False)
        ret_oos_all[tr_end:te_end] = model.predict(X_all[tr_end:te_end])

        # Strength model (base features only — validated as best for this target)
        y_str = df["strength_vol_adj"].values
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_base[:tr_end], y_str[:tr_end],
                  eval_set=[(X_base[tr_end:te_end], y_str[tr_end:te_end])],
                  verbose=False)
        strength_oos[tr_end:te_end] = model.predict(X_base[tr_end:te_end])

    # Filter valid OOS
    mask = ~np.isnan(up_oos) & ~np.isnan(down_oos) & ~np.isnan(ret_oos_base) & ~np.isnan(y_ret)
    idx = np.where(mask)[0]

    up = up_oos[idx]
    down = down_oos[idx]
    y = y_ret[idx]
    reg = regimes[idx]

    print(f"OOS bars: {len(idx)}")
    print(f"Actual return: mean={y.mean()*100:.4f}%, std={y.std()*100:.4f}%")

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("A. DUAL MODEL: up_pred - down_pred (current)")
    print(f"{'='*70}")
    raw_str = up - down
    eval_signal(raw_str, y, reg, "dual raw", [0.15])

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("B. DIRECT RETURN MODEL (base 69 features)")
    print(f"{'='*70}")
    ret_b = ret_oos_base[idx]
    eval_signal(ret_b, y, reg, "direct return (base)",
                [0.0, 0.0003, 0.0006, 0.001])

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("C. DIRECT RETURN MODEL (all 80 features)")
    print(f"{'='*70}")
    ret_a = ret_oos_all[idx]
    eval_signal(ret_a, y, reg, "direct return (all)",
                [0.0, 0.0003, 0.0006, 0.001])

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("D. STRENGTH_VOL_ADJ DIRECT MODEL (base features)")
    print(f"{'='*70}")
    str_d = strength_oos[idx]
    eval_signal(str_d, y, reg, "strength direct",
                [0.0, 0.05, 0.10, 0.15])

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("E. ENSEMBLE: 0.5 * direct_return + 0.5 * (up-down) scaled")
    print(f"{'='*70}")
    # Scale up-down to return scale
    raw_str_scaled = raw_str * 0.003  # same as inference.py
    ensemble = 0.5 * ret_b + 0.5 * raw_str_scaled
    eval_signal(ensemble, y, reg, "ensemble",
                [0.0, 0.0003, 0.0006, 0.001])

    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("F. PER-FOLD IC COMPARISON")
    print(f"{'='*70}")

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        fm = np.zeros(n, dtype=bool)
        fm[tr_end:te_end] = True
        fm = fm[idx]
        if fm.sum() == 0:
            continue

        ic_dual, _ = spearmanr(raw_str[fm], y[fm])
        ic_ret_b, _ = spearmanr(ret_b[fm], y[fm])
        ic_ret_a, _ = spearmanr(ret_a[fm], y[fm])
        ic_str, _ = spearmanr(str_d[fm], y[fm])

        print(f"  Fold {k}: dual={ic_dual:+.4f}  direct_base={ic_ret_b:+.4f}  "
              f"direct_all={ic_ret_a:+.4f}  strength={ic_str:+.4f}")


if __name__ == "__main__":
    main()
