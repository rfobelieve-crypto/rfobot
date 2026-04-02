"""
BTC 多空強度預測指標 v2 — Market Intelligence Indicator.

v2 改動（E1+E2）：
  - Feature selection: 固定特徵集（burn-in 前 1000 bars 選定，鎖死不變）
  - Direction: 直接用 sign(raw_pred) + deadzone，移除 rolling z-score
  - Confidence: 基於 training distribution percentile，非 rolling z-score
  - 所有評估以 raw pred 為主

Usage:
    python -m research.prediction_indicator_v2
    python -m research.prediction_indicator_v2 --save
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
PARQUET = ROOT / "ml_data" / "BTC_USD_1h_enhanced.parquet"
OUT_DIR = ROOT / "ml_data"

HORIZON_BARS = 4
TARGET       = "y_return_4h"

# ── Burn-in for feature selection (no OOS predictions in this window) ────
BURN_IN_BARS = 1000  # ~42 days

# ── Walk-forward config ─────────────────────────────────────────────────
N_FOLDS = 5  # applied to post-burn-in data

# ── EXCLUDE (single source of truth) ────────────────────────────────────
from indicator.feature_config import EXCLUDE as _BASE_EXCLUDE
EXCLUDE = _BASE_EXCLUDE | {
    "y_return_1h",
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h", "vol_4h_proxy",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    "regime", "regime_name", "bull_bear_score",
}

# ── XGBoost params ──────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.6,
    min_child_weight=3, reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, verbosity=0, early_stopping_rounds=40,
)

# ── Direction deadzone (raw pred magnitude) ─────────────────────────────
DEADZONE = 0.0006  # 0.06% — predictions smaller than this → NEUTRAL

# ── Confidence tiers ────────────────────────────────────────────────────
STRONG_THRESHOLD   = 98.8
MODERATE_THRESHOLD = 90.2


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

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

    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE and df[c].dtype in ("float64", "float32", "int64")]
    df[feat_cols] = df[feat_cols].ffill()

    nan_rate = df[feat_cols].isnull().mean()
    feat_cols = [c for c in feat_cols if nan_rate[c] <= 0.10]
    df = df.dropna(subset=feat_cols)

    print(f"  Data: {len(df)} bars x {len(feat_cols)} features")
    print(f"  Target ({TARGET}): mean={df[TARGET].mean():.5f}  std={df[TARGET].std():.4f}")
    print(f"  Burn-in: first {BURN_IN_BARS} bars (feature selection only, no OOS)")
    print(f"  OOS window: bar {BURN_IN_BARS} ~ {len(df)} ({len(df) - BURN_IN_BARS} bars)")

    return df, feat_cols


# ═══════════════════════════════════════════════════════════════════════════
# E1: Fixed Feature Selection (burn-in only, locked after)
# ═══════════════════════════════════════════════════════════════════════════

def select_features_burnin(df: pd.DataFrame, feat_cols: list[str],
                           top_k: int = 40) -> list[str]:
    """Select top-K features using ONLY burn-in data. Locked after selection."""
    if top_k >= len(feat_cols):
        return feat_cols

    X_burn = df[feat_cols].fillna(0).values[:BURN_IN_BARS]
    y_burn = df[TARGET].values[:BURN_IN_BARS]

    m = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbosity=0,
    )
    m.fit(X_burn, y_burn)

    imp = pd.Series(m.feature_importances_, index=feat_cols)
    imp = imp.sort_values(ascending=False)
    selected = list(imp.head(top_k).index)

    print(f"\n  Feature selection (burn-in {BURN_IN_BARS} bars): {len(feat_cols)} -> {top_k}")
    print(f"  Burn-in range: {df.index[0]} ~ {df.index[BURN_IN_BARS-1]}")
    print(f"  Top 15:")
    for i, (name, score) in enumerate(imp.head(15).items()):
        print(f"    {i+1:3d}. {name:35s}  imp={score:.4f}")

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Prediction (post-burn-in only)
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_predict(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """
    Walk-forward OOS prediction on post-burn-in data.

    Training always starts from bar 0 (includes burn-in as training data).
    OOS predictions only for bars >= BURN_IN_BARS.

    Returns: oos_pred array (NaN for burn-in bars, filled for OOS bars)
    """
    n = len(df)
    oos_data = n - BURN_IN_BARS
    fold = oos_data // N_FOLDS
    oos_pred = np.full(n, np.nan)

    print(f"\n  Walk-forward: {N_FOLDS} folds x ~{fold} bars each")
    print(f"  Training starts from bar 0 (expanding window)")

    for k in range(N_FOLDS):
        # Training: bar 0 to train_end (expanding)
        train_end = BURN_IN_BARS + fold * k
        # Test: next fold
        test_start = BURN_IN_BARS + fold * k
        test_end = min(BURN_IN_BARS + fold * (k + 1), n)

        if k == 0:
            # First fold: train on burn-in only
            train_end = BURN_IN_BARS

        if test_end <= test_start:
            break

        X_tr = df[feat_cols].fillna(0).values[:train_end]
        y_tr = df[TARGET].values[:train_end]
        X_te = df[feat_cols].fillna(0).values[test_start:test_end]
        y_te = df[TARGET].values[test_start:test_end]

        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = m.predict(X_te)

        oos_pred[test_start:test_end] = pred

        ic, _ = spearmanr(y_te, pred)
        dir_acc = (np.sign(pred) == np.sign(y_te)).mean()
        print(f"  fold {k+1}/{N_FOLDS}  train=[0:{train_end}]  "
              f"test=[{test_start}:{test_end}]  "
              f"IC={ic:+.4f}  dir={dir_acc:.1%}  n={len(y_te)}")

    return oos_pred


# ═══════════════════════════════════════════════════════════════════════════
# E2: Direction from raw pred (no z-score)
# ═══════════════════════════════════════════════════════════════════════════

def assign_direction_raw(pred: np.ndarray, deadzone: float = DEADZONE) -> np.ndarray:
    """Direction from raw prediction magnitude. No z-score."""
    direction = np.where(pred > deadzone, "UP",
                np.where(pred < -deadzone, "DOWN", "NEUTRAL"))
    direction[np.isnan(pred)] = "NEUTRAL"
    return direction


# ═══════════════════════════════════════════════════════════════════════════
# E2: Confidence from training distribution percentile (no z-score)
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_confidence_v2(pred: np.ndarray,
                            burn_in_end: int) -> np.ndarray:
    """
    Confidence v2: percentile of |pred| in burn-in training distribution.

    This is stable (reference distribution doesn't change) and directly
    measures how "opinionated" the model is vs its historical baseline.
    """
    # Build reference distribution from burn-in predictions
    # We don't have burn-in OOS predictions, so use expanding reference
    # from OOS predictions as they accumulate
    abs_pred = np.abs(pred)
    confidence = np.full(len(pred), np.nan)

    # Collect OOS predictions as they come in, use expanding percentile
    oos_history = []
    for i in range(len(pred)):
        if np.isnan(pred[i]):
            continue

        oos_history.append(abs_pred[i])

        if len(oos_history) < 30:
            continue

        # Percentile rank of current |pred| in all OOS history so far
        hist_arr = np.array(oos_history[:-1])  # exclude current
        pct = (hist_arr < abs_pred[i]).sum() / len(hist_arr) * 100
        confidence[i] = min(pct, 100)

    return confidence


def assign_strength(confidence: np.ndarray) -> np.ndarray:
    strength = np.full(len(confidence), "Weak", dtype=object)
    strength[confidence >= MODERATE_THRESHOLD] = "Moderate"
    strength[confidence >= STRONG_THRESHOLD] = "Strong"
    strength[np.isnan(confidence)] = "Weak"
    return strength


# ═══════════════════════════════════════════════════════════════════════════
# Bull/Bear Power (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def compute_bull_bear_power(df: pd.DataFrame) -> pd.Series:
    components = []
    if "cg_oi_delta_zscore" in df.columns:
        components.append(df["cg_oi_delta_zscore"].clip(-3, 3) / 3)
    if "cg_funding_close_zscore" in df.columns:
        components.append(-df["cg_funding_close_zscore"].clip(-3, 3) / 3)
    if "cg_taker_delta_zscore" in df.columns:
        components.append(df["cg_taker_delta_zscore"].clip(-3, 3) / 3)
    if "cg_ls_ratio_zscore" in df.columns:
        components.append(-df["cg_ls_ratio_zscore"].clip(-3, 3) / 3)
    if "cg_ls_divergence_zscore" in df.columns:
        components.append(df["cg_ls_divergence_zscore"].clip(-3, 3) / 3)
    if not components:
        return pd.Series(0, index=df.index)
    return pd.concat(components, axis=1).mean(axis=1).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation (comprehensive, raw-pred focused)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(df, pred, confidence, direction):
    y = df[TARGET].values
    valid = ~np.isnan(pred) & ~np.isnan(y)
    n_oos = valid.sum()
    p, a = pred[valid], y[valid]

    # ── 1. Overall OOS IC ──
    ic, pval = spearmanr(a, p)

    # ── 2. Per-fold IC ──
    n = len(df)
    oos_data = n - BURN_IN_BARS
    fold_size = oos_data // N_FOLDS
    fold_ics = []
    fold_details = []

    for k in range(N_FOLDS):
        fs = BURN_IN_BARS + fold_size * k
        fe = min(BURN_IN_BARS + fold_size * (k + 1), n)
        mask = np.zeros(n, dtype=bool)
        mask[fs:fe] = True
        mask = mask & valid
        if mask.sum() < 20:
            continue
        ic_k, p_k = spearmanr(y[mask], pred[mask])
        dir_k = (np.sign(pred[mask]) == np.sign(y[mask])).mean()
        fold_ics.append(ic_k)
        dates = df.index[mask]
        fold_details.append((k+1, ic_k, p_k, dir_k, mask.sum(),
                             dates[0].strftime('%m/%d'), dates[-1].strftime('%m/%d')))

    fold_ics_arr = np.array(fold_ics)
    icir = fold_ics_arr.mean() / fold_ics_arr.std() if fold_ics_arr.std() > 0 else 0

    # ── 3. Direction accuracy ──
    active = valid & (direction != "NEUTRAL")
    if active.sum() > 0:
        dir_actual = np.sign(y[active])
        dir_pred = np.where(direction[active] == "UP", 1, -1)
        dir_acc = (dir_pred == dir_actual).mean()
    else:
        dir_acc = 0

    # ── 4. Decile calibration ──
    combined = pd.DataFrame({"pred": p, "actual": a})
    combined["bin"] = pd.qcut(combined["pred"], 10, labels=False, duplicates="drop")
    decile = combined.groupby("bin")["actual"].mean()
    diffs = np.diff(decile.values)
    monotone = (diffs > 0).mean() if len(diffs) > 0 else 0

    # ═══════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TRUE BASELINE REPORT (v2: fixed features + raw pred)")
    print(f"{'='*70}")

    print(f"\n  [A] Overall OOS (raw pred, no z-score)")
    print(f"      IC        = {ic:+.4f}  (p={pval:.2e})")
    print(f"      ICIR      = {icir:+.4f}")
    print(f"      Dir acc   = {dir_acc:.1%}  (active={active.sum()}, total={n_oos})")
    print(f"      Monotone  = {monotone:.0%}")
    print(f"      OOS bars  = {n_oos}")

    print(f"\n  [B] Fold-by-fold OOS IC")
    for k, ic_k, p_k, dir_k, nk, d0, d1 in fold_details:
        sig = "***" if p_k < 0.01 else "* " if p_k < 0.05 else "  "
        sign = "+" if ic_k > 0 else " "
        print(f"      Fold {k}  IC={ic_k:+.4f} {sig}  dir={dir_k:.1%}  "
              f"n={nk}  ({d0} ~ {d1})")
    neg = (fold_ics_arr < 0).sum()
    print(f"      Negative folds: {neg}/{len(fold_ics_arr)}")
    print(f"      Fold IC std:    {fold_ics_arr.std():.4f}")

    # ── [C] Direction distribution ──
    n_up = (direction[valid] == "UP").sum()
    n_dn = (direction[valid] == "DOWN").sum()
    n_ne = (direction[valid] == "NEUTRAL").sum()
    print(f"\n  [C] Direction distribution (raw pred)")
    print(f"      UP      = {n_up:5d} ({n_up/n_oos:.1%})")
    print(f"      DOWN    = {n_dn:5d} ({n_dn/n_oos:.1%})")
    print(f"      NEUTRAL = {n_ne:5d} ({n_ne/n_oos:.1%})")

    # ── [D] Confidence tier (training-percentile based) ──
    print(f"\n  [D] Confidence tier accuracy (percentile-based, no z-score)")
    if active.sum() > 0:
        conf_a = confidence[active]
        dir_correct = (dir_pred == dir_actual)
        for lo, hi, label in [(0, 40, "Weak"), (40, 70, "Moderate"), (70, 101, "Strong")]:
            mask = (conf_a >= lo) & (conf_a < hi) & ~np.isnan(conf_a)
            if mask.sum() >= 10:
                acc = dir_correct[mask].mean()
                ic_b, _ = spearmanr(y[active][mask], pred[active][mask])
                print(f"      {label:10s}  n={mask.sum():5d}  dir_acc={acc:.1%}  IC={ic_b:+.4f}")
            else:
                print(f"      {label:10s}  n={mask.sum():5d}  (too few)")

    # ── [E] Decile calibration ──
    print(f"\n  [E] Decile calibration (raw pred → actual mean return)")
    for bid, mr in decile.items():
        bar = "+" * max(0, int(mr * 5000)) + "-" * max(0, int(-mr * 5000))
        print(f"      decile {int(bid):2d}: {mr*100:+.4f}%  {bar}")

    # ── [F] Per-regime ──
    if "regime_name" in df.columns:
        regime = df["regime_name"].fillna("UNKNOWN").values
        print(f"\n  [F] Per-regime OOS performance")
        for r in sorted(set(regime[valid])):
            m = valid & (regime == r)
            if m.sum() < 30:
                continue
            ic_r, p_r = spearmanr(y[m], pred[m])
            dir_m = m & (direction != "NEUTRAL")
            if dir_m.sum() > 0:
                da_r = (np.where(direction[dir_m] == "UP", 1, -1) == np.sign(y[dir_m])).mean()
            else:
                da_r = 0
            sig = "***" if p_r < 0.01 else "* " if p_r < 0.05 else "  "
            print(f"      {r:20s}  IC={ic_r:+.4f} {sig}  dir={da_r:.1%}  n={m.sum()}")

    # ── [G] Monthly IC ──
    print(f"\n  [G] Monthly OOS IC")
    df_oos = df[valid].copy()
    df_oos["pred"] = pred[valid]
    df_oos["month"] = df_oos.index.strftime("%Y-%m")
    for mo, grp in df_oos.groupby("month"):
        if len(grp) < 50:
            continue
        ic_m, p_m = spearmanr(grp[TARGET], grp["pred"])
        dir_m = (np.sign(grp["pred"]) == np.sign(grp[TARGET])).mean()
        sig = "***" if p_m < 0.01 else "* " if p_m < 0.05 else "  "
        print(f"      {mo}  IC={ic_m:+.4f} {sig}  dir={dir_m:.1%}  n={len(grp)}")

    # ── [H] Weekly IC distribution ──
    df_oos["year_week"] = df_oos.index.strftime("%Y-W%U")
    weekly_ics = []
    for yw, grp in df_oos.groupby("year_week"):
        if len(grp) < 20:
            continue
        ic_w, _ = spearmanr(grp[TARGET], grp["pred"])
        weekly_ics.append(ic_w)
    weekly_ics = np.array(weekly_ics)
    print(f"\n  [H] Weekly IC distribution")
    print(f"      Weeks:    {len(weekly_ics)}")
    print(f"      Mean:     {weekly_ics.mean():+.4f}")
    print(f"      Median:   {np.median(weekly_ics):+.4f}")
    print(f"      Std:      {weekly_ics.std():.4f}")
    print(f"      Positive: {(weekly_ics > 0).sum()}/{len(weekly_ics)} "
          f"({(weekly_ics > 0).mean():.0%})")

    return {"ic": ic, "icir": icir, "dir_acc": dir_acc, "monotone": monotone}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(save: bool = False, top_k: int = 40):
    print("=" * 70)
    print("  BTC Market Intelligence Indicator v2")
    print("  E1: Fixed feature set (burn-in 1000 bars)")
    print("  E2: Raw pred direction + training-percentile confidence")
    print("=" * 70)

    df, feat_cols = load_data()

    # E1: Fixed feature selection on burn-in only
    feat_cols = select_features_burnin(df, feat_cols, top_k=top_k)

    # Walk-forward OOS prediction (post-burn-in)
    print(f"\n{'='*70}")
    print(f"  Walk-Forward OOS Prediction (post burn-in)")
    print(f"{'='*70}")
    oos_pred = walk_forward_predict(df, feat_cols)

    # E2: Direction from raw pred (no z-score)
    direction = assign_direction_raw(oos_pred)

    # E2: Confidence from expanding OOS percentile
    confidence = calibrate_confidence_v2(oos_pred, BURN_IN_BARS)
    strength = assign_strength(confidence)

    # Bull/Bear Power (unchanged)
    bbp = compute_bull_bear_power(df)

    # Evaluate
    evaluate(df, oos_pred, confidence, direction)

    # Build output
    out = pd.DataFrame(index=df.index)
    out["pred_return_4h"] = oos_pred
    out["pred_direction"] = direction
    out["confidence_score"] = confidence
    out["strength_score"] = strength
    out["bull_bear_power"] = bbp.values
    out["regime"] = df["regime_name"].fillna("UNKNOWN").values if "regime_name" in df.columns else "UNKNOWN"
    out["actual_return_4h"] = df[TARGET].values
    out["close"] = df["close"].values

    if save:
        out_path = OUT_DIR / "BTC_USD_indicator_4h_v2.parquet"
        out.to_parquet(out_path, index=True)
        print(f"\n  Saved: {out_path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    run(save=args.save, top_k=args.top_k)


if __name__ == "__main__":
    main()
