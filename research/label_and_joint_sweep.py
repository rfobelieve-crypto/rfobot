"""
Breakthrough experiments for Initiation model Strong-tier WR:
    Phase 1 — label sweep: test (k, horizon) combinations to find cleaner
              positive-class definition. Hypothesis: raising k from 0.8% to
              1.2-1.5% and extending horizon to 6-8 bars removes noise events
              so the remaining positives are "real strong signals".

    Phase 2 — 3-class joint training: train a single XGB multi:softprob
              classifier over UP / DOWN / NEUTRAL instead of two independent
              binary classifiers. Removes long/short conflict whipsaw and
              lets the model learn joint probability structure.

Both use the current DFP_all feature set (init_* + 20 DFP).
Walk-forward purge = max(4, horizon), embargo = 4.
Strong gate = top-5% probability quantile x mag_pct>=0.80 x breakout-required.

Run:  python -m research.label_and_joint_sweep
"""
from __future__ import annotations

import json
import logging
import sys
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
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

STRONG_FRAC = 0.05
STRONG_MAG_PCT = 0.80
MAG_ROLL = 720
EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Label combinations to test (k_pct, horizon_bars)
LABEL_COMBOS = [
    (0.008, 4),   # baseline
    (0.012, 4),
    (0.015, 4),
    (0.012, 6),
    (0.015, 6),
    (0.012, 8),
    (0.015, 8),
]

XGB_BINARY = dict(
    objective="binary:logistic", eval_metric="auc",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    early_stopping_rounds=40,
)

XGB_3CLASS = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    early_stopping_rounds=40,
)


# ═══════════════════════════════════════════════════════════════════════
#   Helpers
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_mag_percentile(df: pd.DataFrame) -> pd.Series:
    feats = json.loads((EXPORT_DIR / "magnitude_feature_cols.json").read_text())
    feats = [f for f in feats if f in df.columns]
    booster = xgb.Booster()
    booster.load_model(str(EXPORT_DIR / "magnitude_xgb.json"))
    X = df[feats].fillna(0).values
    dm = xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index)
    return mag.rolling(MAG_ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )


def run_wf_binary(df: pd.DataFrame, features: list[str], label: str,
                  purge: int) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=purge, embargo=4)
    oos = []
    for tr, te in splits:
        train = df.iloc[tr]; test = df.iloc[te]
        tr_m = train[label].notna()
        te_m = test[label].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train.loc[tr_m, features].fillna(0)
        y_tr = train.loc[tr_m, label].values.astype(int)
        X_te = test.loc[te_m, features].fillna(0)
        y_te = test.loc[te_m, label].values.astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        model = xgb.XGBClassifier(**XGB_BINARY)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p = model.predict_proba(X_te)[:, 1]
        sub = test.loc[te_m, []].copy()
        sub["prob"] = p
        sub["y_true"] = y_te
        oos.append(sub)
    return pd.concat(oos).sort_index() if oos else pd.DataFrame()


def run_wf_3class(df: pd.DataFrame, features: list[str], label: str,
                  purge: int) -> pd.DataFrame:
    """3-class walk-forward. label values: 0=NEUTRAL, 1=UP, 2=DOWN."""
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=purge, embargo=4)
    oos = []
    for tr, te in splits:
        train = df.iloc[tr]; test = df.iloc[te]
        tr_m = train[label].notna()
        te_m = test[label].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train.loc[tr_m, features].fillna(0)
        y_tr = train.loc[tr_m, label].values.astype(int)
        X_te = test.loc[te_m, features].fillna(0)
        y_te = test.loc[te_m, label].values.astype(int)
        # Need all 3 classes present for multiclass training
        if len(np.unique(y_tr)) < 2:
            continue
        # Remap classes to 0..K-1 for training if one is missing
        classes_present = sorted(np.unique(y_tr))
        class_map = {c: i for i, c in enumerate(classes_present)}
        y_tr_mapped = np.array([class_map[c] for c in y_tr])
        params = dict(XGB_3CLASS)
        params["num_class"] = len(classes_present)

        model = xgb.XGBClassifier(**params)
        y_te_mapped = np.array([class_map.get(c, 0) for c in y_te])
        model.fit(X_tr, y_tr_mapped, eval_set=[(X_te, y_te_mapped)], verbose=False)
        proba = model.predict_proba(X_te)

        # Reconstruct probs for all 3 classes (fill missing with 0)
        p_neu = np.zeros(len(X_te))
        p_up = np.zeros(len(X_te))
        p_dn = np.zeros(len(X_te))
        for i, c in enumerate(classes_present):
            col = proba[:, i]
            if c == 0:
                p_neu = col
            elif c == 1:
                p_up = col
            elif c == 2:
                p_dn = col

        sub = test.loc[te_m, []].copy()
        sub["p_up"] = p_up
        sub["p_dn"] = p_dn
        sub["p_neu"] = p_neu
        sub["y_true"] = y_te
        oos.append(sub)
    return pd.concat(oos).sort_index() if oos else pd.DataFrame()


def strong_gate_binary(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
                        df: pd.DataFrame, mag_pct: pd.Series) -> dict:
    idx = oos_long.index.intersection(oos_short.index)
    if len(idx) == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0)
    pl = oos_long.loc[idx, "prob"]
    ps = oos_short.loc[idx, "prob"]
    yl = oos_long.loc[idx, "y_true"].values
    ys = oos_short.loc[idx, "y_true"].values
    bo_up = df.loc[idx, "init_bo_up"].fillna(0).values
    bo_dn = df.loc[idx, "init_bo_dn"].fillna(0).values
    mp = mag_pct.reindex(idx).fillna(0.5).values

    long_thr = float(pl.quantile(1 - STRONG_FRAC))
    short_thr = float(ps.quantile(1 - STRONG_FRAC))

    long_strong = (pl.values >= long_thr) & (mp >= STRONG_MAG_PCT) & (bo_up == 1.0)
    short_strong = (ps.values >= short_thr) & (mp >= STRONG_MAG_PCT) & (bo_dn == 1.0)

    wins = fires = up = dn = 0
    for i in range(len(idx)):
        ls, ss = long_strong[i], short_strong[i]
        if ls and ss:
            continue  # conflict: exclude
        if ls:
            fires += 1; up += 1; wins += int(yl[i])
        elif ss:
            fires += 1; dn += 1; wins += int(ys[i])
    if fires == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0, up=0, dn=0,
                    long_thr=long_thr, short_thr=short_thr)
    wr = wins / fires
    lo, hi = wilson_ci(wins, fires)
    return dict(n=fires, wins=wins, wr=wr, ci_lo=lo, ci_hi=hi, up=up, dn=dn,
                long_thr=long_thr, short_thr=short_thr)


def strong_gate_3class(oos: pd.DataFrame, df: pd.DataFrame,
                       mag_pct: pd.Series) -> dict:
    """For 3-class: UP fire if p_up top-5% AND mag AND bo_up. Same for DOWN."""
    idx = oos.index
    if len(idx) == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0)
    p_up = oos["p_up"].values
    p_dn = oos["p_dn"].values
    y = oos["y_true"].values.astype(int)  # 0=NEU 1=UP 2=DN
    bo_up = df.loc[idx, "init_bo_up"].fillna(0).values
    bo_dn = df.loc[idx, "init_bo_dn"].fillna(0).values
    mp = mag_pct.reindex(idx).fillna(0.5).values

    up_thr = float(pd.Series(p_up).quantile(1 - STRONG_FRAC))
    dn_thr = float(pd.Series(p_dn).quantile(1 - STRONG_FRAC))

    up_strong = (p_up >= up_thr) & (mp >= STRONG_MAG_PCT) & (bo_up == 1.0)
    dn_strong = (p_dn >= dn_thr) & (mp >= STRONG_MAG_PCT) & (bo_dn == 1.0)

    wins = fires = up = dn = 0
    for i in range(len(idx)):
        us, ds = up_strong[i], dn_strong[i]
        if us and ds:
            continue
        if us:
            fires += 1; up += 1
            if y[i] == 1:
                wins += 1
        elif ds:
            fires += 1; dn += 1
            if y[i] == 2:
                wins += 1
    if fires == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0, up=0, dn=0,
                    up_thr=up_thr, dn_thr=dn_thr)
    wr = wins / fires
    lo, hi = wilson_ci(wins, fires)
    return dict(n=fires, wins=wins, wr=wr, ci_lo=lo, ci_hi=hi, up=up, dn=dn,
                up_thr=up_thr, dn_thr=dn_thr)


# ═══════════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    df = add_initiation_features(df)
    print(f"Data: {len(df)} bars x {len(df.columns)} cols")

    # Feature set: base key_4_only + all init_* + DFP_all
    base = filter_available(ABLATION_GROUPS["+ key_4_only"], list(df.columns))
    init = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    features = sorted(set(base) | set(init))
    print(f"Features: {len(features)} (base={len(base)} init={len(init)})")

    print("Scoring MAG percentile (reused across all runs)...")
    mag_pct = score_mag_percentile(df)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — Label sweep
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PHASE 1 — Label sweep  (k_pct, horizon_bars)")
    print("=" * 80)

    phase1_results = []
    for k, H in LABEL_COMBOS:
        cfg = InitiationLabelConfig(k_pct=k, breakout_lookback=20,
                                     use_breakout_confirm=True, horizon_bars=H)
        lbl = build_initiation_labels(df, cfg)
        df_run = df.copy()
        df_run["y_long_touch"] = lbl["y_long_touch"]
        df_run["y_short_touch"] = lbl["y_short_touch"]
        valid = lbl["y_long_touch"].notna()
        long_pos = lbl.loc[valid, "y_long_touch"].mean()
        short_pos = lbl.loc[valid, "y_short_touch"].mean()

        purge = max(4, H)
        print(f"\n--- k={k*100:.1f}% H={H}  "
              f"long_pos={long_pos*100:.2f}% short_pos={short_pos*100:.2f}% "
              f"purge={purge}")

        if long_pos * valid.sum() < 30 or short_pos * valid.sum() < 30:
            print("  SKIP: too few positives")
            continue

        oos_l = run_wf_binary(df_run, features, "y_long_touch", purge)
        oos_s = run_wf_binary(df_run, features, "y_short_touch", purge)
        res = strong_gate_binary(oos_l, oos_s, df, mag_pct)
        res["k_pct"] = k; res["horizon"] = H
        res["long_pos"] = long_pos; res["short_pos"] = short_pos
        phase1_results.append(res)
        print(f"  Strong: n={res['n']}  WR={res['wr']*100:.1f}%  "
              f"CI=[{res['ci_lo']*100:.1f}, {res['ci_hi']*100:.1f}]  "
              f"UP={res.get('up',0)}  DN={res.get('dn',0)}")

    print("\n" + "=" * 80)
    print("  PHASE 1 SUMMARY")
    print("=" * 80)
    print(f"{'k':>7}{'H':>5}{'long+%':>9}{'short+%':>10}{'n':>6}"
          f"{'WR':>9}{'CI_lo':>9}{'CI_hi':>9}")
    print("-" * 68)
    for r in phase1_results:
        print(f"{r['k_pct']*100:>6.1f}%{r['horizon']:>5}"
              f"{r['long_pos']*100:>8.2f}%{r['short_pos']*100:>9.2f}%"
              f"{r['n']:>6}{r['wr']*100:>8.1f}%"
              f"{r['ci_lo']*100:>8.1f}%{r['ci_hi']*100:>8.1f}%")

    # Pick phase 1 winner by CI_lo (requires n>=30)
    eligible = [r for r in phase1_results if r["n"] >= 30]
    if not eligible:
        print("\nPhase 1: no config has enough Strong fires.")
        return
    p1_winner = max(eligible, key=lambda r: r["ci_lo"])
    print(f"\nPhase 1 winner: k={p1_winner['k_pct']*100:.1f}% "
          f"H={p1_winner['horizon']}  "
          f"WR={p1_winner['wr']*100:.1f}%  "
          f"CI_lo={p1_winner['ci_lo']*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — 3-class joint training
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"  PHASE 2 — 3-class joint (using k={p1_winner['k_pct']*100:.1f}% "
          f"H={p1_winner['horizon']})")
    print("=" * 80)

    cfg_joint = InitiationLabelConfig(
        k_pct=p1_winner["k_pct"], breakout_lookback=20,
        use_breakout_confirm=True, horizon_bars=p1_winner["horizon"],
    )
    lbl_joint = build_initiation_labels(df, cfg_joint)
    df_joint = df.copy()

    # Build 3-class target: 0=NEU, 1=UP, 2=DN
    # - both touched -> NEU (whipsaw, ambiguous)
    yl = lbl_joint["y_long_touch"].values
    ys = lbl_joint["y_short_touch"].values
    y3 = np.full(len(df_joint), np.nan)
    for i in range(len(df_joint)):
        if np.isnan(yl[i]) or np.isnan(ys[i]):
            continue
        if yl[i] == 1 and ys[i] == 1:
            y3[i] = 0  # whipsaw -> neutral
        elif yl[i] == 1:
            y3[i] = 1  # UP
        elif ys[i] == 1:
            y3[i] = 2  # DOWN
        else:
            y3[i] = 0  # neutral
    df_joint["y_3class"] = y3
    valid = ~np.isnan(y3)
    n_neu = int((y3[valid] == 0).sum())
    n_up = int((y3[valid] == 1).sum())
    n_dn = int((y3[valid] == 2).sum())
    print(f"3-class distribution: NEU={n_neu} ({n_neu/valid.sum()*100:.1f}%)  "
          f"UP={n_up} ({n_up/valid.sum()*100:.1f}%)  "
          f"DN={n_dn} ({n_dn/valid.sum()*100:.1f}%)")

    purge = max(4, p1_winner["horizon"])
    oos_joint = run_wf_3class(df_joint, features, "y_3class", purge)
    print(f"Joint OOS: {len(oos_joint)} bars")

    p2_res = strong_gate_3class(oos_joint, df, mag_pct)
    print(f"\nPhase 2 Strong: n={p2_res['n']}  WR={p2_res['wr']*100:.1f}%  "
          f"CI=[{p2_res['ci_lo']*100:.1f}, {p2_res['ci_hi']*100:.1f}]  "
          f"UP={p2_res.get('up',0)}  DN={p2_res.get('dn',0)}")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  BREAKTHROUGH EXPERIMENT SUMMARY")
    print("=" * 80)
    baseline = next((r for r in phase1_results
                     if r["k_pct"] == 0.008 and r["horizon"] == 4), None)
    if baseline:
        print(f"  Baseline       (k=0.8% H=4):   "
              f"n={baseline['n']:>3}  WR={baseline['wr']*100:.1f}%  "
              f"CI_lo={baseline['ci_lo']*100:.1f}%")
    print(f"  Phase1 winner  (k={p1_winner['k_pct']*100:.1f}% "
          f"H={p1_winner['horizon']}):   "
          f"n={p1_winner['n']:>3}  WR={p1_winner['wr']*100:.1f}%  "
          f"CI_lo={p1_winner['ci_lo']*100:.1f}%")
    print(f"  Phase2 3-class (k={p1_winner['k_pct']*100:.1f}% "
          f"H={p1_winner['horizon']}):   "
          f"n={p2_res['n']:>3}  WR={p2_res['wr']*100:.1f}%  "
          f"CI_lo={p2_res['ci_lo']*100:.1f}%")

    # Save
    out = dict(
        phase1=[{k: (v if not isinstance(v, float) else float(v))
                 for k, v in r.items()} for r in phase1_results],
        phase1_winner=p1_winner,
        phase2=p2_res,
    )
    with open(RESULTS_DIR / "label_and_joint_sweep.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {RESULTS_DIR/'label_and_joint_sweep.json'}")


if __name__ == "__main__":
    main()
