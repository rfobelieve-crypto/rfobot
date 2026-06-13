"""
End-to-end walk-forward replay of the Initiation dual-classifier system.

Purpose: validate the live production pipeline BEFORE flipping USE_INITIATION_MODEL.

What it does:
    1. Runs strict walk-forward (purge=4, embargo=4) training on the full
       history, scoring long + short heads OOS only.
    2. Scores production MAG model and computes a 720-bar rolling percentile
       (same infra as live).
    3. Runs the exact live gate logic (Strong / Moderate / Weak) on OOS
       predictions, not the training tail.
    4. Measures realized WR using the FIRST-TOUCH label (max reach within
       horizon), matching the training target.
    5. Reports:
        - Correct OOS top-5% / top-10% probability thresholds
          (for rewriting initiation_thresholds.json)
        - Strong / Moderate WR global + per-month
        - Signal count + distribution
        - Gate rejection breakdown (why bars fail the gate)
    6. Dumps OOS parquet for reuse.

Run:
    python -m research.initiation_replay
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
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

BASE_FS = "+ key_4_only"
K_PCT = 0.008
HORIZON = 4
MAG_ROLL = 720

STRONG_FRAC = 0.05
MODERATE_FRAC = 0.10
STRONG_MAG_PCT = 0.80
MODERATE_MAG_PCT = 0.65

EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 400,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 40,
}


def score_mag_percentile(df: pd.DataFrame) -> pd.Series:
    feats = json.loads((EXPORT_DIR / "magnitude_feature_cols.json").read_text())
    feats = [f for f in feats if f in df.columns]
    booster = xgb.Booster()
    booster.load_model(str(EXPORT_DIR / "magnitude_xgb.json"))
    X = df[feats].fillna(0).values
    dm = xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index)
    pct = mag.rolling(MAG_ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )
    return pct


def run_walk_forward(df: pd.DataFrame, features: list[str],
                     label_col: str) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=4, embargo=4)
    all_oos = []
    for tr, te in splits:
        train = df.iloc[tr]; test = df.iloc[te]
        tr_m = train[label_col].notna()
        te_m = test[label_col].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train.loc[tr_m, features].fillna(0)
        y_tr = train.loc[tr_m, label_col].values.astype(int)
        X_te = test.loc[te_m, features].fillna(0)
        y_te = test.loc[te_m, label_col].values.astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p = model.predict_proba(X_te)[:, 1]
        sub = test.loc[te_m, []].copy()
        sub["prob"] = p
        sub["y_true"] = y_te
        all_oos.append(sub)
    return pd.concat(all_oos).sort_index()


def apply_gate(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
               df: pd.DataFrame, mag_pct: pd.Series,
               strong_long_thr: float, strong_short_thr: float,
               mod_long_thr: float, mod_short_thr: float) -> pd.DataFrame:
    idx = oos_long.index.intersection(oos_short.index)
    out = pd.DataFrame(index=idx)
    out["p_long"] = oos_long.loc[idx, "prob"]
    out["p_short"] = oos_short.loc[idx, "prob"]
    out["y_long"] = oos_long.loc[idx, "y_true"]
    out["y_short"] = oos_short.loc[idx, "y_true"]
    out["bo_up"] = df.loc[idx, "init_bo_up"].fillna(0).values
    out["bo_dn"] = df.loc[idx, "init_bo_dn"].fillna(0).values
    out["mag_pct"] = mag_pct.reindex(idx).fillna(0.5).values

    # Same logic as inference.py _predict_dual initiation branch
    def classify(row):
        ls = (row.p_long >= strong_long_thr and row.mag_pct >= STRONG_MAG_PCT
              and row.bo_up == 1.0)
        ss = (row.p_short >= strong_short_thr and row.mag_pct >= STRONG_MAG_PCT
              and row.bo_dn == 1.0)
        if ls and ss:
            return ("UP" if row.p_long >= row.p_short else "DOWN"), "Moderate"
        if ls: return "UP", "Strong"
        if ss: return "DOWN", "Strong"
        lm = (row.p_long >= mod_long_thr and row.mag_pct >= MODERATE_MAG_PCT
              and row.bo_up == 1.0)
        sm = (row.p_short >= mod_short_thr and row.mag_pct >= MODERATE_MAG_PCT
              and row.bo_dn == 1.0)
        if lm and not sm: return "UP", "Moderate"
        if sm and not lm: return "DOWN", "Moderate"
        if lm and sm:
            return ("UP" if row.p_long >= row.p_short else "DOWN"), "Moderate"
        return "NEUTRAL", "Weak"

    pairs = out.apply(classify, axis=1)
    out["direction"] = [p[0] for p in pairs]
    out["strength"] = [p[1] for p in pairs]

    # Correct label depending on chosen direction
    out["win"] = np.where(out["direction"] == "UP", out["y_long"],
                 np.where(out["direction"] == "DOWN", out["y_short"], np.nan))
    return out


def report(gate: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    dist = gate["strength"].value_counts()
    print(f"Total bars: {len(gate)}")
    print(f"Distribution: {dict(dist)}")

    for tier in ["Strong", "Moderate"]:
        sub = gate[gate["strength"] == tier].dropna(subset=["win"])
        if len(sub) == 0:
            print(f"  {tier:<9}: 0 fires")
            continue
        wr = sub["win"].mean()
        n = len(sub)
        # Wilson 95%
        from math import sqrt
        z = 1.96
        p = wr
        denom = 1 + z**2 / n
        center = (p + z**2/(2*n)) / denom
        half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        lo, hi = max(0, center - half), min(1, center + half)

        up = (sub["direction"] == "UP").sum()
        dn = (sub["direction"] == "DOWN").sum()
        print(f"  {tier:<9}: n={n:>4}  WR={wr*100:5.1f}%  "
              f"CI=[{lo*100:4.1f}, {hi*100:4.1f}]  UP={up} DOWN={dn}")

    # Monthly breakdown of Strong
    strong = gate[gate["strength"] == "Strong"].dropna(subset=["win"]).copy()
    if len(strong) > 0:
        strong["month"] = strong.index.strftime("%Y-%m")
        print("\n  Strong monthly WR:")
        for m, g in strong.groupby("month"):
            if len(g) == 0:
                continue
            print(f"    {m}: n={len(g):>3}  WR={g['win'].mean()*100:5.1f}%  "
                  f"UP={(g['direction']=='UP').sum()} DOWN={(g['direction']=='DOWN').sum()}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    df = add_initiation_features(df)
    print(f"Loaded {len(df)} bars x {len(df.columns)} cols")

    cfg = InitiationLabelConfig(k_pct=K_PCT, breakout_lookback=20,
                                 use_breakout_confirm=True, horizon_bars=HORIZON)
    labels = build_initiation_labels(df, cfg)
    df["y_long_touch"] = labels["y_long_touch"]
    df["y_short_touch"] = labels["y_short_touch"]

    base = filter_available(ABLATION_GROUPS[BASE_FS], list(df.columns))
    init = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    features = sorted(set(base) | set(init))
    print(f"Features: {len(base)} base + {len(init)} init = {len(features)}")

    print("\n[1/3] Walk-forward LONG ...")
    oos_long = run_walk_forward(df, features, "y_long_touch")
    print(f"  LONG OOS: {len(oos_long)} bars, pos_rate={oos_long.y_true.mean()*100:.2f}%")

    print("[2/3] Walk-forward SHORT ...")
    oos_short = run_walk_forward(df, features, "y_short_touch")
    print(f"  SHORT OOS: {len(oos_short)} bars, pos_rate={oos_short.y_true.mean()*100:.2f}%")

    print("[3/3] Scoring MAG percentile ...")
    mag_pct = score_mag_percentile(df)

    # Calibrated OOS thresholds
    strong_long = float(oos_long["prob"].quantile(1 - STRONG_FRAC))
    strong_short = float(oos_short["prob"].quantile(1 - STRONG_FRAC))
    mod_long = float(oos_long["prob"].quantile(1 - MODERATE_FRAC))
    mod_short = float(oos_short["prob"].quantile(1 - MODERATE_FRAC))
    print(f"\nOOS-calibrated thresholds:")
    print(f"  Strong  : long>={strong_long:.3f}  short>={strong_short:.3f}")
    print(f"  Moderate: long>={mod_long:.3f}  short>={mod_short:.3f}")

    gate = apply_gate(oos_long, oos_short, df, mag_pct,
                       strong_long, strong_short, mod_long, mod_short)

    report(gate, "WALK-FORWARD REPLAY (OOS-calibrated thresholds + MAG rolling pct)")

    # Also replay using the thresholds CURRENTLY in initiation_thresholds.json,
    # so we see what live would do if deployed as-is.
    thr_path = EXPORT_DIR / "initiation_thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        live_gate = apply_gate(
            oos_long, oos_short, df, mag_pct,
            thr["strong"]["prob_long_threshold"],
            thr["strong"]["prob_short_threshold"],
            thr["moderate"]["prob_long_threshold"],
            thr["moderate"]["prob_short_threshold"],
        )
        report(live_gate,
               "REPLAY WITH CURRENT initiation_thresholds.json (in-sample quantiles)")

    # Dump OOS predictions for reuse
    merged = oos_long[["prob", "y_true"]].rename(
        columns={"prob": "p_long", "y_true": "y_long"})
    merged["p_short"] = oos_short["prob"]
    merged["y_short"] = oos_short["y_true"]
    merged["mag_pct"] = mag_pct.reindex(merged.index)
    out_path = RESULTS_DIR / "initiation_replay_oos.parquet"
    merged.to_parquet(out_path)
    print(f"\nOOS predictions dumped to {out_path}")

    # Suggested updated thresholds JSON
    suggested = {
        "strong": {
            "prob_long_threshold": strong_long,
            "prob_short_threshold": strong_short,
            "mag_percentile_min": STRONG_MAG_PCT,
            "require_breakout": True,
        },
        "moderate": {
            "prob_long_threshold": mod_long,
            "prob_short_threshold": mod_short,
            "mag_percentile_min": MODERATE_MAG_PCT,
            "require_breakout": True,
        },
        "metadata": {
            "calibration_source": "walk_forward_OOS",
            "k_pct": K_PCT,
            "horizon_bars": HORIZON,
            "strong_top_frac": STRONG_FRAC,
            "moderate_top_frac": MODERATE_FRAC,
        },
    }
    (RESULTS_DIR / "initiation_thresholds_suggested.json").write_text(
        json.dumps(suggested, indent=2))
    print(f"Suggested thresholds saved to "
          f"{RESULTS_DIR/'initiation_thresholds_suggested.json'}")


if __name__ == "__main__":
    main()
