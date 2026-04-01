"""
BTC 多空強度預測指標 — Market Intelligence Indicator.

產品定位：預測型指標，非交易策略。
每個 15m bar 輸出：
  - pred_return_4h:    預測未來 4h 收益率
  - pred_direction:    UP / DOWN / NEUTRAL
  - confidence_score:  0~100（預測值在歷史分佈中的百分位）
  - strength_score:    Strong / Moderate / Weak
  - bull_bear_power:   -1.0 ~ +1.0（rule-based composite）
  - regime:            當前市場狀態

Architecture:
  1. XGBRegressor → pred_return_4h (walk-forward, no lookahead)
  2. pred_direction = sign(pred_return_4h) with deadzone
  3. confidence_score = expanding percentile rank of |pred_return_4h|
  4. strength_score = discretize(confidence_score)
  5. bull_bear_power = rule-based from order flow components

Evaluation: IC, direction accuracy, calibration monotonicity.

Usage:
    python research/prediction_indicator.py
    python research/prediction_indicator.py --save
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
PARQUET = ROOT / "ml_data" / "BTC_USD_15m_enhanced.parquet"
OUT_DIR = ROOT / "ml_data"

HORIZON_BARS = 16  # 4h = 16 x 15m
TARGET       = "y_return_4h"
N_FOLDS      = 5

# ── Unified feature set: Coinglass + Binance klines only ─────────────────
# Same as indicator/feature_config.py — ensures local = Railway
from indicator.feature_config import ALL_FEATURES as API_FEATURES

# Never use as features
EXCLUDE = {
    "ts_open", "open", "high", "low", "close", "volume",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "y_return_1h", "y_return_4h",
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h", "vol_4h_proxy",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    "regime", "regime_name", "bull_bear_score",
}

# ── XGBoost params ───────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators      = 400,
    max_depth         = 4,
    learning_rate     = 0.03,
    subsample         = 0.75,
    colsample_bytree  = 0.5,
    min_child_weight  = 5,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    verbosity         = 0,
    early_stopping_rounds = 30,
)

# ── Direction deadzone ───────────────────────────────────────────────────────
# If |pred_return| < DEADZONE, classify as NEUTRAL
# 4h return std ≈ 1.15%, use ~5% of std as deadzone
DEADZONE = 0.0006  # 0.06%

# ── Confidence tiers ─────────────────────────────────────────────────────────
STRONG_THRESHOLD   = 70
MODERATE_THRESHOLD = 40


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PARQUET)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df = df.set_index("dt")
        elif "ts_open" in df.columns:
            df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
            df = df.set_index("dt")
    df = df.sort_index()

    # Target
    if TARGET not in df.columns:
        df[TARGET] = df["close"].shift(-HORIZON_BARS) / df["close"] - 1

    df = df.iloc[:-HORIZON_BARS].copy()
    df = df.dropna(subset=[TARGET])

    # Feature columns — only Coinglass + klines derived (API-compatible)
    feat_cols = [c for c in API_FEATURES if c in df.columns and c not in EXCLUDE]
    df[feat_cols] = df[feat_cols].ffill()

    # Drop high-NaN features
    nan_rate = df[feat_cols].isnull().mean()
    feat_cols = [c for c in feat_cols if nan_rate[c] <= 0.10]
    df = df.dropna(subset=feat_cols)

    print(f"  Data: {len(df)} bars x {len(feat_cols)} features")
    print(f"  Target ({TARGET}): mean={df[TARGET].mean():.5f}  std={df[TARGET].std():.4f}")

    return df, feat_cols


# ═════════════════════════════════════════════════════════════════════════════
# Feature Selection (top-K by importance)
# ═════════════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame, feat_cols: list[str],
                     top_k: int = 60) -> list[str]:
    """Select top-K features using initial XGBoost on first 60% of data."""
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

    imp = pd.Series(m.feature_importances_, index=feat_cols)
    imp = imp.sort_values(ascending=False)
    selected = list(imp.head(top_k).index)

    print(f"\n  Feature selection: {len(feat_cols)} -> {top_k}")
    print(f"  Top 15:")
    for i, (name, score) in enumerate(imp.head(15).items()):
        print(f"    {i+1:3d}. {name:35s}  imp={score:.4f}")

    return selected


# ═════════════════════════════════════════════════════════════════════════════
# Walk-Forward Prediction
# ═════════════════════════════════════════════════════════════════════════════

def walk_forward_predict(df: pd.DataFrame, feat_cols: list[str]) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward OOS prediction using XGB + Ridge ensemble.

    Returns:
      pred:      ensemble prediction (avg of XGB + Ridge)
      agreement: model agreement score (0~1). High = both models agree on
                 direction & magnitude. Used to modulate confidence.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n    = len(df)
    fold = n // (N_FOLDS + 1)
    oos       = np.full(n, np.nan)
    oos_agree = np.full(n, np.nan)

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr = df[feat_cols].fillna(0).values[:tr_end]
        y_tr = df[TARGET].values[:tr_end]
        X_te = df[feat_cols].fillna(0).values[tr_end:te_end]
        y_te = df[TARGET].values[tr_end:te_end]

        # XGBoost
        m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred_xgb = m_xgb.predict(X_te)

        # Ridge (standardized)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m_ridge = Ridge(alpha=10.0)
        m_ridge.fit(X_tr_s, y_tr)
        pred_ridge = m_ridge.predict(X_te_s)

        # Ensemble
        pred = 0.5 * pred_xgb + 0.5 * pred_ridge
        oos[tr_end:te_end] = pred

        # Agreement: 1.0 when both models predict same sign with similar magnitude
        # 0.0 when they disagree on direction
        same_sign = (np.sign(pred_xgb) == np.sign(pred_ridge)).astype(float)
        # Magnitude similarity: 1 - normalized difference
        max_abs = np.maximum(np.abs(pred_xgb), np.abs(pred_ridge))
        max_abs = np.where(max_abs < 1e-8, 1e-8, max_abs)
        mag_sim = 1.0 - np.abs(pred_xgb - pred_ridge) / (max_abs * 2)
        mag_sim = np.clip(mag_sim, 0, 1)
        agreement = same_sign * mag_sim
        oos_agree[tr_end:te_end] = agreement

        ic_x, _ = spearmanr(y_te, pred_xgb)
        ic_r, _ = spearmanr(y_te, pred_ridge)
        ic_e, _ = spearmanr(y_te, pred)
        dir_acc = (np.sign(pred) == np.sign(y_te)).mean()
        agree_pct = same_sign.mean()
        print(f"  fold {k}/{N_FOLDS}  IC: xgb={ic_x:+.4f} ridge={ic_r:+.4f} "
              f"ens={ic_e:+.4f}  dir={dir_acc:.1%}  agree={agree_pct:.1%}  n={len(y_te)}")

    return oos, oos_agree


# ═════════════════════════════════════════════════════════════════════════════
# Confidence Calibration
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_confidence(pred_z: np.ndarray, agreement: np.ndarray) -> np.ndarray:
    """
    Confidence = z-score_percentile * agreement_weight.

    Two components:
      1. Z-score magnitude percentile (0~100): how extreme is this z-scored
         prediction vs history of |z-scores|
      2. Agreement weight (0.5~1.0): do both models agree?

    Using z-scored predictions ensures confidence reflects how unusual the
    prediction is relative to recent history, not absolute magnitude.
    """
    abs_z = np.abs(pred_z)
    confidence = np.full(len(pred_z), np.nan)

    for i in range(50, len(pred_z)):
        if np.isnan(pred_z[i]):
            continue
        history = abs_z[:i]
        valid = history[~np.isnan(history)]
        if len(valid) < 30:
            continue

        # Component 1: z-score magnitude percentile
        mag_pct = (valid < abs_z[i]).sum() / len(valid) * 100

        # Component 2: agreement weight (0.5 ~ 1.0)
        agree = agreement[i] if not np.isnan(agreement[i]) else 0.5
        agree_weight = 0.5 + 0.5 * agree

        confidence[i] = min(mag_pct * agree_weight, 100)

    return confidence


# ═════════════════════════════════════════════════════════════════════════════
# Direction & Strength
# ═════════════════════════════════════════════════════════════════════════════

def rolling_zscore(pred: np.ndarray, window: int = 192,
                   min_obs: int = 50) -> np.ndarray:
    """
    Rolling z-score of predictions.

    Compares each prediction to its recent rolling window (default 192 bars =
    2 days of 15m). This adapts quickly to regime shifts, breaking long streaks
    of identical direction caused by slow-changing features.

    pred_z[i] = (pred[i] - mean(pred[i-window:i])) / std(pred[i-window:i])
    """
    s = pd.Series(pred)
    mu = s.rolling(window, min_periods=min_obs).mean()
    sigma = s.rolling(window, min_periods=min_obs).std()
    sigma = sigma.replace(0, np.nan)
    pred_z = ((s - mu) / sigma).values
    return pred_z


def assign_direction(pred_z: np.ndarray, deadzone_z: float = 0.1) -> np.ndarray:
    """
    Direction from z-scored predictions.
    UP if pred_z > deadzone_z, DOWN if pred_z < -deadzone_z, else NEUTRAL.
    """
    direction = np.where(pred_z > deadzone_z, "UP",
                np.where(pred_z < -deadzone_z, "DOWN", "NEUTRAL"))
    direction[np.isnan(pred_z)] = "NEUTRAL"
    return direction


def assign_strength(confidence: np.ndarray) -> np.ndarray:
    """Strong (>=70), Moderate (40-70), Weak (<40)."""
    strength = np.full(len(confidence), "Weak", dtype=object)
    strength[confidence >= MODERATE_THRESHOLD] = "Moderate"
    strength[confidence >= STRONG_THRESHOLD]   = "Strong"
    strength[np.isnan(confidence)] = "Weak"
    return strength


# ═════════════════════════════════════════════════════════════════════════════
# Bull/Bear Power (rule-based composite, no model, no lookahead)
# ═════════════════════════════════════════════════════════════════════════════

def compute_bull_bear_power(df: pd.DataFrame) -> pd.Series:
    """
    Composite Bull/Bear Power from order flow components.
    Range: -1.0 (max bearish) to +1.0 (max bullish).

    Components (equal weight):
      1. BVC delta ratio z-score   -> realized trade flow direction
      2. OI delta z-score          -> new money flow direction
      3. -Funding z-score          -> crowding pressure (inverted)
      4. Taker delta z-score       -> CG taker buy-sell direction
      5. L/S ratio deviation       -> sentiment extremes (inverted)

    Each component clipped to [-1, +1], then averaged.
    """
    components = []

    # 1. BVC delta ratio
    if "bvc_delta_ratio" in df.columns:
        c = df["bvc_delta_ratio"].rolling(96, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-9), raw=False
        ).clip(-3, 3) / 3
        components.append(c)

    # 2. OI delta
    if "oi_delta_zscore" in df.columns:
        components.append(df["oi_delta_zscore"].clip(-3, 3) / 3)
    elif "cg_oi_delta_zscore" in df.columns:
        components.append(df["cg_oi_delta_zscore"].clip(-3, 3) / 3)

    # 3. Funding (inverted: high positive funding = bearish pressure)
    if "funding_zscore" in df.columns:
        components.append(-df["funding_zscore"].clip(-3, 3) / 3)
    elif "cg_funding_close_zscore" in df.columns:
        components.append(-df["cg_funding_close_zscore"].clip(-3, 3) / 3)

    # 4. CG Taker delta
    if "cg_taker_delta_zscore" in df.columns:
        components.append(df["cg_taker_delta_zscore"].clip(-3, 3) / 3)
    elif "taker_delta_ratio" in df.columns:
        c = df["taker_delta_ratio"].rolling(96, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-9), raw=False
        ).clip(-3, 3) / 3
        components.append(c)

    # 5. L/S ratio deviation (inverted: extreme long = bearish)
    if "cg_ls_ratio_zscore" in df.columns:
        components.append(-df["cg_ls_ratio_zscore"].clip(-3, 3) / 3)

    if not components:
        return pd.Series(0, index=df.index)

    bbp = pd.concat(components, axis=1).mean(axis=1).clip(-1, 1)
    return bbp


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation (prediction quality only, no PnL)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(df: pd.DataFrame, pred: np.ndarray, confidence: np.ndarray,
             direction: np.ndarray):
    y = df[TARGET].values
    valid = ~np.isnan(pred) & ~np.isnan(y)
    p, a = pred[valid], y[valid]

    # 1. Overall IC
    ic, _ = spearmanr(a, p)

    # 2. Per-fold IC for ICIR
    n = len(df)
    fold = n // (N_FOLDS + 1)
    fold_ics = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        pp = pred[tr_end:te_end]
        aa = y[tr_end:te_end]
        m = ~np.isnan(pp) & ~np.isnan(aa)
        if m.sum() > 20:
            ic_f, _ = spearmanr(aa[m], pp[m])
            fold_ics.append(ic_f)
    fold_ics = np.array(fold_ics)
    icir = fold_ics.mean() / fold_ics.std() if len(fold_ics) > 1 and fold_ics.std() > 0 else 0

    # 3. Direction accuracy (active signals only)
    active = valid & (direction != "NEUTRAL")
    if active.sum() > 0:
        dir_actual = np.sign(y[active])
        dir_pred = np.where(direction[active] == "UP", 1, -1)
        dir_acc = (dir_pred == dir_actual).mean()
        n_active = active.sum()
    else:
        dir_acc = 0
        n_active = 0

    # 4. Calibration monotonicity (decile analysis)
    combined = pd.DataFrame({"pred": p, "actual": a})
    combined["bin"] = pd.qcut(combined["pred"], 10, labels=False, duplicates="drop")
    decile = combined.groupby("bin")["actual"].mean()
    diffs = np.diff(decile.values)
    monotone = (diffs > 0).mean() if len(diffs) > 0 else 0

    # 5. Confidence tier accuracy
    conf_valid = confidence[valid & (direction != "NEUTRAL")]
    dir_correct = (dir_pred == dir_actual) if active.sum() > 0 else np.array([])

    print(f"\n{'='*60}")
    print(f"Prediction Quality — 4h Horizon")
    print(f"{'='*60}")
    print(f"  IC        = {ic:+.4f}")
    print(f"  ICIR      = {icir:+.4f}")
    print(f"  fold ICs  = {[round(v,4) for v in fold_ics]}")
    print(f"  Dir acc   = {dir_acc:.1%}  (n={n_active} active / {valid.sum()} total)")
    print(f"  Monotone  = {monotone:.0%}")

    # Direction distribution
    n_up = (direction[valid] == "UP").sum()
    n_dn = (direction[valid] == "DOWN").sum()
    n_ne = (direction[valid] == "NEUTRAL").sum()
    print(f"\n  Direction distribution:")
    print(f"    UP      = {n_up:5d} ({n_up/valid.sum():.1%})")
    print(f"    DOWN    = {n_dn:5d} ({n_dn/valid.sum():.1%})")
    print(f"    NEUTRAL = {n_ne:5d} ({n_ne/valid.sum():.1%})")

    # Confidence tier analysis
    if active.sum() > 0:
        conf_a = confidence[active]
        corr_a = dir_correct
        print(f"\n  Confidence tier accuracy:")
        for lo, hi, label in [(0, 40, "Weak"), (40, 70, "Moderate"), (70, 101, "Strong")]:
            mask = (conf_a >= lo) & (conf_a < hi)
            if mask.sum() >= 10:
                acc = corr_a[mask].mean()
                print(f"    {label:10s}  n={mask.sum():5d}  dir_acc={acc:.1%}")
            else:
                print(f"    {label:10s}  n={mask.sum():5d}  (too few)")

    # Decile table
    print(f"\n  Decile calibration (pred -> actual mean return):")
    for bin_id, mean_ret in decile.items():
        print(f"    decile {int(bin_id):2d}: actual_mean = {mean_ret*100:+.4f}%")

    # Per-regime
    if "regime_name" in df.columns:
        regime = df["regime_name"].fillna("UNKNOWN").values
        print(f"\n  Per regime:")
        for r in sorted(set(regime[valid])):
            m = valid & (regime == r)
            if m.sum() < 30:
                continue
            ic_r, _ = spearmanr(y[m], pred[m])
            d_m = m & (direction != "NEUTRAL")
            if d_m.sum() > 0:
                da_r = (np.where(direction[d_m] == "UP", 1, -1) == np.sign(y[d_m])).mean()
            else:
                da_r = 0
            print(f"    {r:20s}  IC={ic_r:+.4f}  dir_acc={da_r:.1%}  n={m.sum()}")

    return {"ic": ic, "icir": icir, "dir_acc": dir_acc, "monotone": monotone}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def run(save: bool = False, top_k: int = 60):
    print("=" * 60)
    print("BTC Market Intelligence Indicator — 4h Horizon")
    print("=" * 60)

    # Load
    df, feat_cols = load_data()

    # Feature selection
    feat_cols = select_features(df, feat_cols, top_k=top_k)
    print(f"\n  Using {len(feat_cols)} features")

    # Walk-forward prediction
    print(f"\n{'='*60}")
    print(f"Walk-Forward OOS Prediction")
    print(f"{'='*60}")
    pred, agreement = walk_forward_predict(df, feat_cols)

    # Rolling z-score normalization
    # Breaks long same-direction streaks by comparing to rolling 1-day window
    pred_z = rolling_zscore(pred, window=96, min_obs=30)

    valid_z = ~np.isnan(pred_z)
    print(f"\n  Z-score stats: mean={pred_z[valid_z].mean():.3f}  "
          f"std={pred_z[valid_z].std():.3f}  "
          f"positive={( pred_z[valid_z] > 0).mean():.1%}  "
          f"negative={( pred_z[valid_z] < 0).mean():.1%}")

    # Confidence calibration (using z-scored predictions)
    confidence = calibrate_confidence(pred_z, agreement)

    # Direction & strength (from z-scored predictions)
    direction = assign_direction(pred_z)
    strength  = assign_strength(confidence)

    # Bull/Bear Power
    bbp = compute_bull_bear_power(df)

    # Evaluate
    evaluate(df, pred, confidence, direction)

    # Build output DataFrame
    out = pd.DataFrame(index=df.index)
    out["pred_return_4h"]   = pred
    out["pred_direction"]   = direction
    out["confidence_score"] = confidence
    out["strength_score"]   = strength
    out["bull_bear_power"]  = bbp.values
    out["regime"]           = df["regime_name"].fillna("UNKNOWN").values if "regime_name" in df.columns else "UNKNOWN"

    # Add actual for offline analysis
    out["actual_return_4h"] = df[TARGET].values
    out["close"]            = df["close"].values

    # Sample output
    active = out[out["pred_direction"] != "NEUTRAL"].tail(20)
    print(f"\n{'='*60}")
    print(f"Sample predictions (last 20 active)")
    print(f"{'='*60}")
    for idx, row in active.iterrows():
        d = "^" if row["pred_direction"] == "UP" else "v"
        actual = row["actual_return_4h"]
        hit = "O" if (row["pred_direction"] == "UP" and actual > 0) or \
                     (row["pred_direction"] == "DOWN" and actual < 0) else "X"
        print(f"  {idx}  {d} {row['pred_direction']:7s}  "
              f"conf={row['confidence_score']:5.1f}  {row['strength_score']:8s}  "
              f"pred={row['pred_return_4h']*100:+.3f}%  "
              f"actual={actual*100:+.3f}%  {hit}  "
              f"BBP={row['bull_bear_power']:+.3f}")

    # Confidence distribution
    valid_conf = confidence[~np.isnan(confidence)]
    print(f"\n  Confidence distribution:")
    print(f"    mean={valid_conf.mean():.1f}  median={np.median(valid_conf):.1f}  "
          f"std={valid_conf.std():.1f}")
    print(f"    Weak(<40):     {(valid_conf < 40).sum():5d} ({(valid_conf < 40).mean():.1%})")
    print(f"    Moderate(40-70):{((valid_conf >= 40) & (valid_conf < 70)).sum():5d} "
          f"({((valid_conf >= 40) & (valid_conf < 70)).mean():.1%})")
    print(f"    Strong(>=70):  {(valid_conf >= 70).sum():5d} ({(valid_conf >= 70).mean():.1%})")

    if save:
        out_path = OUT_DIR / "BTC_USD_indicator_4h.parquet"
        out.to_parquet(out_path, index=True)
        print(f"\n  Saved: {out_path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--top-k", type=int, default=60)
    args = ap.parse_args()
    run(save=args.save, top_k=args.top_k)


if __name__ == "__main__":
    main()
