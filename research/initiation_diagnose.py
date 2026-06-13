"""
Diagnose the initiation model's Strong-gate fires.

Goal: split Strong-gate predictions into TP / FP / FN and show what
distinguishes them, so we can make an informed decision on label redesign.

For each bar we compute at prediction-time (trailing only) + forward (label-time):
    ret_4h        : realized forward return
    max_up_4h     : max(close[t+1..t+4]/close[t]-1)   (best reached during window)
    max_dn_4h     : min(close[t+1..t+4]/close[t]-1)   (worst reached)
    touched_long  : max_up_4h >= k
    touched_short : max_dn_4h <= -k

Strong gate (long side, k=0.008):
    p_long_init >= top-5% threshold AND close > rolling_high_20

Split of gate fires:
    TP                 : y_long_init == 1
    FP_touched_reverse : touched_long True but closed below k  (reached TP then reversed)
    FP_slow_drift      : 0 < ret_4h < k                         (right direction, not enough)
    FP_reversal        : ret_4h <= 0                            (wrong direction)

Also: FN = bars with y_long_init==1 but NOT gate-fired. Shows what we miss.

Runs k=0.008 only (where real signal is strongest).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import add_initiation_features, INITIATION_FEATURE_COLS

K = 0.008
BASE_FS = "+ key_4_only"

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 400,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 40,
}


def add_path_stats(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Add forward path diagnostics (max_up / max_dn over t+1..t+horizon)."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float) if "high" in df.columns else close
    low = df["low"].values.astype(float) if "low" in df.columns else close
    n = len(close)
    mx_up = np.full(n, np.nan)
    mx_dn = np.full(n, np.nan)
    for i in range(n - horizon):
        window_high = high[i + 1 : i + 1 + horizon].max()
        window_low = low[i + 1 : i + 1 + horizon].min()
        mx_up[i] = window_high / close[i] - 1
        mx_dn[i] = window_low / close[i] - 1
    out = df.copy()
    out["path_max_up_4h"] = mx_up
    out["path_max_dn_4h"] = mx_dn
    return out


def run_walk_forward(df: pd.DataFrame, features: list[str], label_col: str) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48,
                                  purge=4, embargo=4)
    all_oos = []
    for tr, te in splits:
        train_df = df.iloc[tr]
        test_df = df.iloc[te]
        tr_m = train_df[label_col].notna()
        te_m = test_df[label_col].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train_df.loc[tr_m, features].fillna(0)
        y_tr = train_df.loc[tr_m, label_col].values.astype(int)
        X_te = test_df.loc[te_m, features].fillna(0)
        y_te = test_df.loc[te_m, label_col].values.astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = (len(y_tr) - y_tr.sum()) / y_tr.sum()
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p = m.predict_proba(X_te)[:, 1]
        sub = test_df.loc[te_m].copy()
        sub["y_prob"] = p
        sub["y_true"] = y_te
        all_oos.append(sub)
    return pd.concat(all_oos)


def diagnose_side(oos: pd.DataFrame, side: str, k: float) -> None:
    """side = 'long' or 'short'."""
    confirm_col = "init_bo_up" if side == "long" else "init_bo_dn"
    # top-5% threshold (same as Strong gate sim)
    thr = oos["y_prob"].quantile(0.95)
    gate = (oos["y_prob"] >= thr) & (oos[confirm_col] == 1.0)
    fired = oos.loc[gate].copy()

    print("=" * 92)
    print(f"SIDE = {side.upper()}   k = {k*100:.1f}%   gate fires: {len(fired)}")
    print(f"  threshold prob = {thr:.3f}   baseline pos rate = {oos['y_true'].mean()*100:.2f}%")
    print("=" * 92)

    if len(fired) == 0:
        return

    # Classify fires
    ret = fired["ret_4h"] if "ret_4h" in fired.columns else None
    mx_up = fired["path_max_up_4h"]
    mx_dn = fired["path_max_dn_4h"]
    y = fired["y_true"]

    if side == "long":
        touched_tp = mx_up >= k
        wrong_dir = ret <= 0
        slow_drift = (ret > 0) & (ret < k) & ~touched_tp
        touched_reverse = touched_tp & (y == 0)
    else:  # short
        touched_tp = mx_dn <= -k
        wrong_dir = ret >= 0
        slow_drift = (ret < 0) & (ret > -k) & ~touched_tp
        touched_reverse = touched_tp & (y == 0)

    tp = y == 1
    fp = y == 0

    n = len(fired)
    print(f"  TP                    {tp.sum():>4}  ({tp.mean()*100:5.1f}%)")
    print(f"  FP total              {fp.sum():>4}  ({fp.mean()*100:5.1f}%)")
    print(f"    touched_tp_reverse  {touched_reverse.sum():>4}  ({touched_reverse.mean()*100:5.1f}%)")
    print(f"    slow_drift          {(slow_drift & fp).sum():>4}  ({(slow_drift & fp).mean()*100:5.1f}%)")
    print(f"    wrong_direction     {(wrong_dir & fp).sum():>4}  ({(wrong_dir & fp).mean()*100:5.1f}%)")

    print()
    print("  Return path stats (gate fires only):")
    print(f"                         mean    median   p25     p75     min     max")
    for name, series in [("ret_4h", ret), ("max_up_4h", mx_up), ("max_dn_4h", mx_dn)]:
        if series is None:
            continue
        print(f"    {name:<18} {series.mean()*100:+6.2f}% "
              f"{series.median()*100:+6.2f}% "
              f"{series.quantile(0.25)*100:+6.2f}% "
              f"{series.quantile(0.75)*100:+6.2f}% "
              f"{series.min()*100:+6.2f}% "
              f"{series.max()*100:+6.2f}%")

    # Compare TP vs FP distributions
    print()
    print("  TP vs FP path comparison (what separates winners from losers?):")
    if tp.sum() > 0 and fp.sum() > 0:
        for name, series in [("ret_4h", ret), ("max_up_4h", mx_up),
                              ("max_dn_4h", mx_dn)]:
            if series is None:
                continue
            tp_med = series[tp].median() * 100
            fp_med = series[fp].median() * 100
            tp_mean = series[tp].mean() * 100
            fp_mean = series[fp].mean() * 100
            print(f"    {name:<18} TP median {tp_med:+6.2f}%  FP median {fp_med:+6.2f}%   "
                  f"TP mean {tp_mean:+6.2f}%  FP mean {fp_mean:+6.2f}%")

    # Monthly breakdown (check stability)
    print()
    print("  Monthly breakdown:")
    fired["month"] = fired.index.strftime("%Y-%m")
    monthly = fired.groupby("month").agg(
        n=("y_true", "size"),
        wr=("y_true", "mean"),
    )
    for m, row in monthly.iterrows():
        print(f"    {m}  n={int(row['n']):>3}  win_rate={row['wr']*100:5.1f}%")

    # FN analysis: positives that gate MISSED
    pos = oos[oos["y_true"] == 1]
    missed = pos[~((pos["y_prob"] >= thr) & (pos[confirm_col] == 1.0))]
    print()
    print(f"  False negatives ({len(missed)} missed positives out of {len(pos)} total):")
    if len(missed) > 0:
        # Why missed?
        miss_low_prob = (missed["y_prob"] < thr).sum()
        miss_no_breakout = (missed[confirm_col] != 1.0).sum()
        miss_both = ((missed["y_prob"] < thr) & (missed[confirm_col] != 1.0)).sum()
        print(f"    low prob only      : {miss_low_prob - miss_both}")
        print(f"    no breakout only   : {miss_no_breakout - miss_both}")
        print(f"    both               : {miss_both}")
        print(f"    missed prob median : {missed['y_prob'].median():.3f}")


def main():
    df = load_and_cache_data()
    df = add_initiation_features(df)
    df = add_path_stats(df, horizon=4)

    labels = build_initiation_labels(df, InitiationLabelConfig(k_pct=K))
    for c in ["ret_4h", "y_long_init", "y_short_init"]:
        df[c] = labels[c]

    base_feats = filter_available(ABLATION_GROUPS[BASE_FS], list(df.columns))
    extra = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    features = sorted(set(base_feats) | set(extra))
    print(f"Bars: {len(df)}  Features: {len(features)}  k: {K*100:.1f}%\n")

    print(">>> training long side ...")
    oos_long = run_walk_forward(df, features, "y_long_init")
    print(">>> training short side ...\n")
    oos_short = run_walk_forward(df, features, "y_short_init")

    diagnose_side(oos_long, "long", K)
    print()
    diagnose_side(oos_short, "short", K)


if __name__ == "__main__":
    main()
