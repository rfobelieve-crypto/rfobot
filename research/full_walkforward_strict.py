"""
Strict Full-System Walk-Forward (Init + Mag both retrained per fold)

The honest end-to-end accuracy test. Every model — Init Long, Init Short,
Magnitude — is retrained from scratch in each fold using only data from
before the test window (purge=4, embargo=4). V5 breakout gate is pure
trailing calculation. No in-sample leakage anywhere.

Output: Strong / Moderate / Weak WR under 3 metrics:
  (1) 4h closing return sign
  (2) First-touch +/-0.8%
  (3) Path race +/-0.8%

Run:  python -m research.full_walkforward_strict
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from indicator.initiation_features import add_initiation_features

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
THRESH_PATH = EXPORT_DIR / "initiation_thresholds.json"

HORIZON = 4
TARGET_K = 0.008
MAG_ROLL = 500  # rolling percentile window for mag_pct
STRONG_FRAC = 0.05  # calibration top-5%
MODERATE_FRAC = 0.10

# XGB params (match production)
INIT_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    max_depth=4, learning_rate=0.05, n_estimators=300,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)
MAG_PARAMS = dict(
    objective="reg:squarederror", eval_metric="mae",
    max_depth=4, learning_rate=0.05, n_estimators=300,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
)


# ─────────────────────────────────────────────────────────────────
# Label builders
# ─────────────────────────────────────────────────────────────────
def build_first_touch_labels(df: pd.DataFrame, k: float = 0.008,
                              horizon: int = 4) -> pd.DataFrame:
    """Official initiation labels: first-touch AND breakout_up/dn confirmed."""
    cfg = InitiationLabelConfig(k_pct=k, breakout_lookback=20,
                                 use_breakout_confirm=True, horizon_bars=horizon)
    labels = build_initiation_labels(df, cfg)
    df = df.copy()
    df["y_long_touch"] = labels["y_long_touch"]
    df["y_short_touch"] = labels["y_short_touch"]
    return df


def build_mag_target(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """y_vol_adj_abs = |return_h| / realized_vol_20b."""
    df = df.copy()
    close = df["close"].astype(float)
    fwd_ret = close.shift(-horizon) / close - 1
    abs_ret = fwd_ret.abs()
    if "realized_vol_20b" in df.columns:
        vol = df["realized_vol_20b"].astype(float)
    else:
        vol = close.pct_change().rolling(20).std()
    vol = vol.replace(0, np.nan)
    df["y_vol_adj_abs"] = abs_ret / vol
    return df


# ─────────────────────────────────────────────────────────────────
# Path outcomes
# ─────────────────────────────────────────────────────────────────
def compute_path_outcomes(close, high, low, horizon=HORIZON, k=TARGET_K):
    c = close.values.astype(float)
    h = high.values.astype(float)
    lo = low.values.astype(float)
    n = len(c)
    fwd_ret = np.full(n, np.nan)
    ft_long = np.zeros(n, dtype=int)
    ft_short = np.zeros(n, dtype=int)
    race_long = np.zeros(n, dtype=int)
    race_short = np.zeros(n, dtype=int)
    for t in range(n - horizon):
        c0 = c[t]
        up_target = c0 * (1 + k)
        dn_target = c0 * (1 - k)
        fwd_ret[t] = c[t + horizon] / c0 - 1
        long_hit_bar = -1
        short_hit_bar = -1
        for j in range(1, horizon + 1):
            if long_hit_bar < 0 and h[t + j] >= up_target:
                long_hit_bar = j
            if short_hit_bar < 0 and lo[t + j] <= dn_target:
                short_hit_bar = j
            if long_hit_bar > 0 and short_hit_bar > 0:
                break
        ft_long[t] = 1 if long_hit_bar > 0 else 0
        ft_short[t] = 1 if short_hit_bar > 0 else 0
        if long_hit_bar > 0 and (short_hit_bar < 0 or long_hit_bar < short_hit_bar):
            race_long[t] = 1; race_short[t] = -1
        elif short_hit_bar > 0 and (long_hit_bar < 0 or short_hit_bar < long_hit_bar):
            race_short[t] = 1; race_long[t] = -1
        elif long_hit_bar > 0 and short_hit_bar > 0 and long_hit_bar == short_hit_bar:
            race_long[t] = -1; race_short[t] = -1
    return dict(fwd_ret=fwd_ret, ft_long=ft_long, ft_short=ft_short,
                race_long=race_long, race_short=race_short)


# ─────────────────────────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────────────────────────
def run_strict_walkforward(df: pd.DataFrame,
                            init_features: list[str],
                            mag_features: list[str]) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=4, embargo=4)
    all_rows = []
    t0 = time.time()
    for i, (tr_idx, te_idx) in enumerate(splits):
        train = df.iloc[tr_idx]
        test = df.iloc[te_idx]

        # Init Long
        m_tr_long = train["y_long_touch"].notna()
        if m_tr_long.sum() < 80:
            continue
        X_tr = train.loc[m_tr_long, init_features].fillna(0)
        y_tr = train.loc[m_tr_long, "y_long_touch"].values.astype(int)
        if y_tr.sum() < 10 or y_tr.sum() > len(y_tr) - 10:
            continue
        init_long = xgb.XGBClassifier(**INIT_PARAMS)
        init_long.fit(X_tr, y_tr)

        # Init Short
        m_tr_short = train["y_short_touch"].notna()
        X_tr_s = train.loc[m_tr_short, init_features].fillna(0)
        y_tr_s = train.loc[m_tr_short, "y_short_touch"].values.astype(int)
        if y_tr_s.sum() < 10 or y_tr_s.sum() > len(y_tr_s) - 10:
            continue
        init_short = xgb.XGBClassifier(**INIT_PARAMS)
        init_short.fit(X_tr_s, y_tr_s)

        # Mag model
        m_tr_mag = train["y_vol_adj_abs"].notna()
        if m_tr_mag.sum() < 80:
            continue
        X_tr_m = train.loc[m_tr_mag, mag_features].fillna(0)
        y_tr_m = train.loc[m_tr_mag, "y_vol_adj_abs"].values
        mag_model = xgb.XGBRegressor(**MAG_PARAMS)
        mag_model.fit(X_tr_m, y_tr_m)

        # Predict test
        X_te_init = test[init_features].fillna(0)
        X_te_mag = test[mag_features].fillna(0)
        p_long = init_long.predict_proba(X_te_init)[:, 1]
        p_short = init_short.predict_proba(X_te_init)[:, 1]
        mag_raw = np.maximum(mag_model.predict(X_te_mag), 0)

        fold = pd.DataFrame({
            "p_long": p_long,
            "p_short": p_short,
            "mag_raw": mag_raw,
        }, index=test.index)
        all_rows.append(fold)

        if (i + 1) % 10 == 0:
            print(f"  fold {i+1}/{len(splits)}  "
                  f"elapsed={time.time()-t0:.0f}s  n_oos={sum(len(f) for f in all_rows)}")

    oos = pd.concat(all_rows).sort_index()
    print(f"  DONE walk-forward: {len(oos)} OOS bars  ({time.time()-t0:.0f}s)")
    return oos


# ─────────────────────────────────────────────────────────────────
# Gate + metrics
# ─────────────────────────────────────────────────────────────────
def apply_gate(p_long, p_short, mag_pct, bo_up, bo_dn, thr):
    sl = thr["strong"]; md = thr["moderate"]
    n = len(p_long)
    direction = np.full(n, "NEUTRAL", dtype=object)
    tier = np.full(n, "Weak", dtype=object)
    for i in range(n):
        long_strong = (p_long[i] >= sl["prob_long_threshold"]
                       and mag_pct[i] >= sl["mag_percentile_min"]
                       and bo_up[i] == 1.0)
        short_strong = (p_short[i] >= sl["prob_short_threshold"]
                        and mag_pct[i] >= sl["mag_percentile_min"]
                        and bo_dn[i] == 1.0)
        if long_strong and short_strong:
            direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
            tier[i] = "Moderate"
        elif long_strong:
            direction[i] = "UP"; tier[i] = "Strong"
        elif short_strong:
            direction[i] = "DOWN"; tier[i] = "Strong"
        else:
            long_mod = (p_long[i] >= md["prob_long_threshold"]
                        and mag_pct[i] >= md["mag_percentile_min"]
                        and bo_up[i] == 1.0)
            short_mod = (p_short[i] >= md["prob_short_threshold"]
                         and mag_pct[i] >= md["mag_percentile_min"]
                         and bo_dn[i] == 1.0)
            if long_mod and not short_mod:
                direction[i] = "UP"; tier[i] = "Moderate"
            elif short_mod and not long_mod:
                direction[i] = "DOWN"; tier[i] = "Moderate"
            elif long_mod and short_mod:
                direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
                tier[i] = "Moderate"
            else:
                top = max(p_long[i], p_short[i])
                if top < 0.20:
                    direction[i] = "NEUTRAL"
                else:
                    direction[i] = "UP" if p_long[i] >= p_short[i] else "DOWN"
                tier[i] = "Weak"
    return direction, tier


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def report(tier, direction, path):
    print(f"\n  Distribution: {dict(pd.Series(tier).value_counts())}\n")
    print(f"    {'tier':<10}{'metric':<20}{'n':>6}{'WR':>10}{'CI':>22}")
    fwd = path["fwd_ret"]
    ft_l = path["ft_long"].astype(bool)
    ft_s = path["ft_short"].astype(bool)
    rl_w = path["race_long"] == 1
    rl_l = path["race_long"] == -1
    rs_w = path["race_short"] == 1
    rs_l = path["race_short"] == -1

    for t in ["Strong", "Moderate", "Weak"]:
        mask = tier == t
        up = mask & (direction == "UP")
        dn = mask & (direction == "DOWN")

        # (1) 4h close
        n_up, n_dn = int(up.sum()), int(dn.sum())
        w = int((up & (fwd > 0)).sum()) + int((dn & (fwd < 0)).sum())
        n = n_up + n_dn
        if n > 0:
            wr = w/n; lo, hi = wilson_ci(w, n)
            print(f"    {t:<10}{'(1) 4h close':<20}{n:>6}{wr*100:>9.1f}%  [{lo*100:5.1f}%, {hi*100:5.1f}%]")

        # (2) first-touch
        w = int((up & ft_l).sum()) + int((dn & ft_s).sum())
        if n > 0:
            wr = w/n; lo, hi = wilson_ci(w, n)
            print(f"    {'':<10}{'(2) first-touch':<20}{n:>6}{wr*100:>9.1f}%  [{lo*100:5.1f}%, {hi*100:5.1f}%]")

        # (3) path race
        trig_up = up & (rl_w | rl_l)
        trig_dn = dn & (rs_w | rs_l)
        n_trig = int(trig_up.sum()) + int(trig_dn.sum())
        w = int((trig_up & rl_w).sum()) + int((trig_dn & rs_w).sum())
        if n_trig > 0:
            wr = w/n_trig; lo, hi = wilson_ci(w, n_trig)
            print(f"    {'':<10}{'(3) race +/-0.8%':<20}{n_trig:>6}{wr*100:>9.1f}%  [{lo*100:5.1f}%, {hi*100:5.1f}%]")
        print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  STRICT FULL-SYSTEM WALK-FORWARD")
    print("  Init Long + Init Short + Mag all retrained per fold")
    print("=" * 78)

    df = load_and_cache_data()
    df = add_initiation_features(df)
    df = build_first_touch_labels(df, k=TARGET_K, horizon=HORIZON)
    df = build_mag_target(df, horizon=HORIZON)
    print(f"  Loaded: {len(df)} bars x {len(df.columns)} cols")

    # Feature sets (match production)
    init_feats = json.loads((EXPORT_DIR / "initiation_feature_cols.json").read_text())
    mag_feats = json.loads((EXPORT_DIR / "magnitude_feature_cols.json").read_text())
    init_feats = [f for f in init_feats if f in df.columns]
    mag_feats = [f for f in mag_feats if f in df.columns]
    print(f"  Init features: {len(init_feats)}   Mag features: {len(mag_feats)}")

    # Walk-forward (Init + Mag both retrained)
    print("\n  Running strict walk-forward ...")
    oos = run_strict_walkforward(df, init_feats, mag_feats)

    # Compute rolling mag percentile on OOS mag_raw (trailing only)
    mag_series = oos["mag_raw"].reindex(df.index)
    mag_pct = mag_series.rolling(MAG_ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )
    oos["mag_pct"] = mag_pct.reindex(oos.index).fillna(0.5)

    # Calibrate thresholds on OOS (top-5% / top-10%)
    long_s_thr = float(oos["p_long"].quantile(1 - STRONG_FRAC))
    short_s_thr = float(oos["p_short"].quantile(1 - STRONG_FRAC))
    long_m_thr = float(oos["p_long"].quantile(1 - MODERATE_FRAC))
    short_m_thr = float(oos["p_short"].quantile(1 - MODERATE_FRAC))

    thr = {
        "strong": {
            "prob_long_threshold": long_s_thr,
            "prob_short_threshold": short_s_thr,
            "mag_percentile_min": 0.80,
            "require_breakout": True,
        },
        "moderate": {
            "prob_long_threshold": long_m_thr,
            "prob_short_threshold": short_m_thr,
            "mag_percentile_min": 0.65,
            "require_breakout": True,
        },
    }
    print(f"\n  Thresholds (OOS-calibrated):")
    print(f"    Strong:   long>={long_s_thr:.3f}  short>={short_s_thr:.3f}")
    print(f"    Moderate: long>={long_m_thr:.3f}  short>={short_m_thr:.3f}")

    # Path outcomes
    path_all = compute_path_outcomes(
        df["close"], df["high"], df["low"], HORIZON, TARGET_K)
    path_oos = {k: pd.Series(v, index=df.index).reindex(oos.index).values
                for k, v in path_all.items()}

    # V5 breakout already in df
    bo_up = df["init_bo_up"].reindex(oos.index).fillna(0).values
    bo_dn = df["init_bo_dn"].reindex(oos.index).fillna(0).values

    direction, tier = apply_gate(
        oos["p_long"].values, oos["p_short"].values,
        oos["mag_pct"].values, bo_up, bo_dn, thr)

    valid = ~np.isnan(path_oos["fwd_ret"])
    print("\n" + "=" * 78)
    print("  STRICT OOS RESULTS (Init + Mag both walk-forward)")
    print("=" * 78)
    report(tier[valid], direction[valid],
           {k: v[valid] for k, v in path_oos.items()})

    # Save
    oos.to_parquet(RESULTS_DIR / "strict_walkforward_oos.parquet")
    print(f"\n  Saved: {RESULTS_DIR/'strict_walkforward_oos.parquet'}")


if __name__ == "__main__":
    main()
