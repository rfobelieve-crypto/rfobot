"""
AI Trading Signal Generator — BTC 15-min Order Flow Model.

Outputs per 15m bar:
  - direction:  "UP" / "DOWN" / "NEUTRAL"
  - confidence: 0–100 (model conviction)
  - range_low, range_high: estimated price range over horizon
  - bull_bear_power: -1.0 to +1.0 composite indicator

Architecture:
  1. XGBoost Classifier  → P(up) probability → direction + confidence
  2. XGBoost Regressor   → E[return] → range estimate ± uncertainty
  3. Bull/Bear Power      → composite from BVC, OI, funding, VPIN

Walk-forward training: expanding window, no lookahead.

Usage:
    python research/signal_generator.py
    python research/signal_generator.py --save
    python research/signal_generator.py --horizon 4h
    python research/signal_generator.py --horizon 1h
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
PARQUET = ROOT / "ml_data" / "BTC_USD_15m_enhanced.parquet"
OUT_DIR = ROOT / "eda_charts"

HORIZONS = {"1h": 4, "2h": 8, "4h": 16}
N_FOLDS  = 5

# Never use as features
EXCLUDE = {
    "ts_open", "open", "high", "low", "close",
    "y_return_1h", "y_return_4h",
    "future_high_4h", "future_low_4h",
    "up_move_4h", "down_move_4h", "vol_4h_proxy",
    "up_move_vol_adj", "down_move_vol_adj",
    "strength_raw", "strength_vol_adj",
    "future_return_5m", "future_return_15m", "future_return_1h",
    "label_5m", "label_15m", "label_1h",
    "regime", "regime_name", "bull_bear_score", "volume",
}

# ─── Classifier params (direction + confidence) ─────────────────────────────
CLF_PARAMS = dict(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    eval_metric="logloss", early_stopping_rounds=20,
)

# ─── Regressor params (return magnitude) ─────────────────────────────────────
REG_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    early_stopping_rounds=20,
)

# ─── Confidence thresholds ───────────────────────────────────────────────────
# prob in [0.5 ± NEUTRAL_BAND] → NEUTRAL
NEUTRAL_BAND = 0.02   # prob 0.48~0.52 = neutral


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_data(horizon: str = "4h") -> tuple[pd.DataFrame, list[str], int]:
    bars = HORIZONS[horizon]

    df = pd.read_parquet(PARQUET)
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
        df = df.set_index("dt")
    df = df.sort_index()

    # Target: forward return
    target_col = f"y_return_{horizon}"
    df[target_col] = df["close"].shift(-bars) / df["close"] - 1

    # Direction label: 1 = up, 0 = down
    df["y_dir"] = (df[target_col] > 0).astype(int)

    # Drop incomplete
    df = df.iloc[:-bars].copy()
    df = df.dropna(subset=[target_col])

    # Feature columns
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE
                 and c != target_col and c != "y_dir"]

    df[feat_cols] = df[feat_cols].ffill()
    nan_rate = df[feat_cols].isnull().mean()
    feat_cols = [c for c in feat_cols if nan_rate[c] <= 0.10]
    df = df.dropna(subset=feat_cols)

    print(f"  Data: {len(df)} bars, {len(feat_cols)} features, horizon={horizon} ({bars} bars)")
    return df, feat_cols, bars


# ═════════════════════════════════════════════════════════════════════════════
# Bull/Bear Power (rule-based composite, no lookahead)
# ═════════════════════════════════════════════════════════════════════════════

def compute_bull_bear_power(df: pd.DataFrame) -> pd.Series:
    """
    Composite Bull/Bear Power from order flow components.
    Range: -1.0 (max bearish) to +1.0 (max bullish).

    Components (equal weight):
      1. BVC delta ratio z-score → direction of trade flow
      2. OI delta z-score        → new money flow direction
      3. Funding z-score         → crowding pressure (inverted: high funding = bearish)
      4. VPIN deviation          → informed trading intensity (sign from delta)

    Each component is clipped to [-1, +1] before averaging.
    """
    components = []

    # 1. BVC delta ratio z-score (taker flow direction)
    if "bvc_delta_ratio" in df.columns:
        c1 = df["bvc_delta_ratio"].rolling(96, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-9), raw=False
        ).clip(-3, 3) / 3
    elif "taker_delta_ratio" in df.columns:
        c1 = df["taker_delta_ratio"].rolling(96, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-9), raw=False
        ).clip(-3, 3) / 3
    else:
        c1 = pd.Series(0, index=df.index)
    components.append(c1)

    # 2. OI delta z-score (position flow)
    if "oi_delta_zscore" in df.columns:
        c2 = df["oi_delta_zscore"].clip(-3, 3) / 3
    elif "oi_delta" in df.columns:
        c2 = df["oi_delta"].rolling(96, min_periods=4).apply(
            lambda x: (x.iloc[-1] - x.mean()) / max(x.std(), 1e-9), raw=False
        ).clip(-3, 3) / 3
    else:
        c2 = pd.Series(0, index=df.index)
    components.append(c2)

    # 3. Funding z-score (inverted: extreme positive funding = bearish pressure)
    if "funding_zscore" in df.columns:
        c3 = (-df["funding_zscore"]).clip(-3, 3) / 3
    else:
        c3 = pd.Series(0, index=df.index)
    components.append(c3)

    # 4. VPIN-weighted delta direction
    if "vpin_deviation" in df.columns and "bvc_delta_ratio" in df.columns:
        # High VPIN deviation + positive delta = strong bullish informed flow
        c4 = (df["vpin_deviation"] * np.sign(df["bvc_delta_ratio"])).clip(-0.3, 0.3) / 0.3
    else:
        c4 = pd.Series(0, index=df.index)
    components.append(c4)

    # Average and clip to [-1, 1]
    power = pd.concat(components, axis=1).mean(axis=1).clip(-1, 1)
    return power


# ═════════════════════════════════════════════════════════════════════════════
# Walk-Forward Signal Generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, feat_cols: list[str],
                      bars: int, horizon: str,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Walk-forward OOS signal generation.

    Returns DataFrame with columns:
      direction, confidence, pred_return, range_low, range_high,
      bull_bear_power, strength
    """
    n    = len(df)
    fold = n // (N_FOLDS + 1)

    target_col = f"y_return_{horizon}"
    y_ret = df[target_col].values
    y_dir = df["y_dir"].values
    X     = df[feat_cols].fillna(0).values

    # Output arrays
    prob_up    = np.full(n, np.nan)
    pred_ret   = np.full(n, np.nan)

    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_end = min(fold * (k + 1), n)
        if te_end <= tr_end:
            break

        X_tr, X_te = X[:tr_end], X[tr_end:te_end]
        y_dir_tr   = y_dir[:tr_end]
        y_ret_tr   = y_ret[:tr_end]
        y_dir_te   = y_dir[tr_end:te_end]
        y_ret_te   = y_ret[tr_end:te_end]

        # ── Classifier: P(up) ────────────────────────────────────────────
        clf = xgb.XGBClassifier(**CLF_PARAMS)
        clf.fit(X_tr, y_dir_tr,
                eval_set=[(X_te, y_dir_te)], verbose=False)
        p = clf.predict_proba(X_te)[:, 1]
        prob_up[tr_end:te_end] = p

        # ── Regressor: E[return] ─────────────────────────────────────────
        reg = xgb.XGBRegressor(**REG_PARAMS)
        reg.fit(X_tr, y_ret_tr,
                eval_set=[(X_te, y_ret_te)], verbose=False)
        r = reg.predict(X_te)
        pred_ret[tr_end:te_end] = r

        if verbose:
            acc = ((p > 0.5) == y_dir_te).mean()
            print(f"  fold {k}/{N_FOLDS}  acc={acc:.1%}  "
                  f"prob_range=[{p.min():.3f}, {p.max():.3f}]  "
                  f"pred_ret_std={r.std()*100:.4f}%")

    # ── Bull/Bear Power ──────────────────────────────────────────────────
    bbp = compute_bull_bear_power(df)

    # ── Assemble signals ─────────────────────────────────────────────────
    # Confidence: 0 at prob=0.5, 100 at prob=0 or 1
    raw_conf = np.abs(prob_up - 0.5) * 200  # 0~100

    # Direction
    direction = np.where(
        prob_up > 0.5 + NEUTRAL_BAND, "UP",
        np.where(prob_up < 0.5 - NEUTRAL_BAND, "DOWN", "NEUTRAL")
    )

    # Strength: Strong (conf > 70), Moderate (40-70), Weak (< 40)
    strength = np.where(
        raw_conf > 70, "Strong",
        np.where(raw_conf > 40, "Moderate", "Weak")
    )

    # Range estimate: pred_return ± historical prediction error
    # Use expanding std of past prediction errors as uncertainty band
    actual_ret = y_ret
    pred_err   = np.abs(pred_ret - actual_ret)
    # Expanding std of errors (no lookahead)
    err_series = pd.Series(pred_err)
    err_std    = err_series.expanding(min_periods=20).std().values

    close = df["close"].values
    range_low  = close * (1 + pred_ret - err_std)
    range_high = close * (1 + pred_ret + err_std)

    # Also compute return range in %
    range_ret_low  = pred_ret - err_std
    range_ret_high = pred_ret + err_std

    signals = pd.DataFrame({
        "close":           close,
        "direction":       direction,
        "confidence":      np.round(raw_conf, 1),
        "strength":        strength,
        "prob_up":         np.round(prob_up, 4),
        "pred_return":     np.round(pred_ret * 100, 4),   # in %
        "range_ret_low":   np.round(range_ret_low * 100, 4),
        "range_ret_high":  np.round(range_ret_high * 100, 4),
        "range_low":       np.round(range_low, 2),
        "range_high":      np.round(range_high, 2),
        "bull_bear_power": np.round(bbp.values, 4),
        "actual_return":   np.round(actual_ret * 100, 4),
        "actual_dir":      np.where(actual_ret > 0, "UP", "DOWN"),
    }, index=df.index)

    # Mark bars with no OOS prediction
    signals.loc[np.isnan(prob_up), "direction"]  = None
    signals.loc[np.isnan(prob_up), "confidence"] = np.nan

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_signals(signals: pd.DataFrame, horizon: str):
    """Evaluate signal quality."""
    s = signals.dropna(subset=["confidence"])

    print(f"\n{'='*60}")
    print(f"Signal Evaluation — {horizon}")
    print(f"{'='*60}")
    print(f"  Total signals: {len(s)}")

    # Direction distribution
    for d in ["UP", "DOWN", "NEUTRAL"]:
        n = (s["direction"] == d).sum()
        print(f"  {d:8s}: {n:5d} ({n/len(s):.1%})")

    # Direction accuracy (exclude NEUTRAL)
    active = s[s["direction"].isin(["UP", "DOWN"])]
    if len(active) > 0:
        correct = (active["direction"] == active["actual_dir"]).mean()
        print(f"\n  Direction accuracy (active): {correct:.1%}  ({len(active)} signals)")

    # Accuracy by confidence tier
    print("\n  Accuracy by confidence tier:")
    for tier, lo, hi in [("Weak (0-40)", 0, 40), ("Moderate (40-70)", 40, 70), ("Strong (70+)", 70, 101)]:
        mask = (s["confidence"] >= lo) & (s["confidence"] < hi) & s["direction"].isin(["UP", "DOWN"])
        grp = s[mask]
        if len(grp) < 10:
            print(f"    {tier:20s}  n={len(grp):5d}  (too few)")
            continue
        acc = (grp["direction"] == grp["actual_dir"]).mean()
        mean_ret = grp["actual_return"].mean()
        print(f"    {tier:20s}  n={len(grp):5d}  acc={acc:.1%}  mean_actual={mean_ret:+.4f}%")

    # Accuracy by regime
    if "regime_name" in signals.columns or True:
        # Load regime from enhanced parquet
        try:
            tab = pd.read_parquet(PARQUET)
            tab["dt"] = pd.to_datetime(tab["ts_open"], unit="ms", utc=True)
            tab = tab.set_index("dt")
            if "regime_name" in tab.columns:
                s = s.join(tab[["regime_name"]], how="left")
                print("\n  Accuracy by regime:")
                for r in sorted(s["regime_name"].dropna().unique()):
                    m = (s["regime_name"] == r) & s["direction"].isin(["UP", "DOWN"])
                    grp = s[m]
                    if len(grp) < 20:
                        continue
                    acc = (grp["direction"] == grp["actual_dir"]).mean()
                    print(f"    {r:20s}  n={len(grp):5d}  acc={acc:.1%}")
        except Exception:
            pass

    # Range calibration: what % of actual returns fall within predicted range?
    active = s[s["direction"].isin(["UP", "DOWN"])]
    if len(active) > 0:
        in_range = (
            (active["actual_return"] >= active["range_ret_low"])
            & (active["actual_return"] <= active["range_ret_high"])
        ).mean()
        print(f"\n  Range calibration: {in_range:.1%} of returns within predicted range")

    # Bull/Bear Power correlation with future return
    from scipy.stats import spearmanr
    valid = s["bull_bear_power"].notna() & s["actual_return"].notna()
    if valid.sum() > 50:
        ic, _ = spearmanr(s.loc[valid, "bull_bear_power"],
                          s.loc[valid, "actual_return"])
        print(f"  Bull/Bear Power IC: {ic:+.4f}")

    # P&L simulation (maker fee scenario)
    print(f"\n  P&L simulation (maker fee 0.02% x 2):")
    active = s[s["direction"].isin(["UP", "DOWN"])].copy()
    if len(active) > 0:
        pos = np.where(active["direction"] == "UP", 1, -1)
        ret = pos * active["actual_return"].values / 100
        fee = 0.0004

        for tier_name, conf_min in [("All signals", 0), ("Moderate+", 40), ("Strong only", 70)]:
            mask = active["confidence"].values >= conf_min
            if mask.sum() < 10:
                continue
            r = ret[mask] - fee
            eq = np.cumprod(1 + r)
            wr = (r > 0).mean()
            mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
            print(f"    {tier_name:20s}  n={mask.sum():5d}  "
                  f"Ret={eq[-1]-1:+.1%}  WR={wr:.1%}  MDD={mdd:.1%}")

    return s


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="4h", choices=["1h", "2h", "4h"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(f"AI Trading Signal Generator — BTC 15m — {args.horizon}")
    print("=" * 60)

    df, feat_cols, bars = load_data(args.horizon)

    print(f"\nGenerating walk-forward signals...")
    signals = generate_signals(df, feat_cols, bars, args.horizon)

    # Evaluate
    evaluated = evaluate_signals(signals, args.horizon)

    # Save
    if args.save:
        out = ROOT / "ml_data" / f"BTC_USD_signals_{args.horizon}.parquet"
        signals.to_parquet(out)
        print(f"\n  Saved signals to {out}")

    # Print sample signals
    print(f"\n{'='*60}")
    print("Sample signals (last 20 active):")
    print("=" * 60)
    active = signals[signals["direction"].isin(["UP", "DOWN"])].tail(20)
    for dt, row in active.iterrows():
        arrow = "^" if row["direction"] == "UP" else "v"
        check = "O" if row["direction"] == row["actual_dir"] else "X"
        print(f"  {dt}  {arrow} {row['direction']:5s}  "
              f"conf={row['confidence']:5.1f}  "
              f"strength={row['strength']:8s}  "
              f"pred={row['pred_return']:+.3f}%  "
              f"range=[{row['range_ret_low']:+.3f}%, {row['range_ret_high']:+.3f}%]  "
              f"actual={row['actual_return']:+.3f}%  {check}  "
              f"BBP={row['bull_bear_power']:+.3f}")


if __name__ == "__main__":
    main()
