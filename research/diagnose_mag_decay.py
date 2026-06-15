"""
Diagnose why Magnitude IC decayed from 0.44 (Dec) → 0.21 (Apr).

Hypotheses:
    H1. Target drift — |ret_4h| distribution changed (vol regime shift)
    H2. Feature drift — top feature distributions shifted from training window
    H3. Model staleness — no retrain, using stale model on new regime
    H4. Data pollution — known cg_bfx_margin_ratio gap 2026-04-07..04-12

Method:
    1. Load full cached feature matrix (≈4000 bars, ~6 months)
    2. Split into monthly windows (Dec / Jan / Feb / Mar / Apr)
    3. Per month:
        (a) Target |ret_4h| mean/std/vol regime
        (b) Top-10 Mag features mean/std
        (c) KS-test each top feature vs training window (Dec)
    4. Load production Magnitude model, predict OOS on ALL months
       → per-month IC of STATIC model on each window
       → distinguishes "model stale" from "signal dead"
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp, spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data

MODEL_PATH = Path("indicator/model_artifacts/dual_model/magnitude_xgb.json")
FEATS_PATH = Path("indicator/model_artifacts/dual_model/magnitude_feature_cols.json")
IMP_PATH = Path("indicator/model_artifacts/dual_model/magnitude_importance.csv")
OUT = Path("research/results/mag_decay_diagnosis.json")

HORIZON = 4


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Rows: {len(df)}, range: {df.index[0]} → {df.index[-1]}")

    # Target
    ret_4h = (df["close"].shift(-HORIZON) / df["close"] - 1)
    abs_ret = ret_4h.abs()

    # Load production Mag model
    mag_feats = json.loads(FEATS_PATH.read_text())
    mag_feats = [f for f in mag_feats if f in df.columns]
    print(f"Mag features available: {len(mag_feats)}")

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    X = df[mag_feats].fillna(0).values
    pred = model.predict(X)
    df = df.copy()
    df["_abs_ret"] = abs_ret
    df["_mag_pred"] = pred

    # Top features by importance
    imp = pd.read_csv(IMP_PATH, header=0, names=["feature", "imp"])
    top_feats = imp.nlargest(10, "imp")["feature"].tolist()
    top_feats = [f for f in top_feats if f in df.columns]
    print(f"Top features: {top_feats}")

    # Monthly split
    df["_month"] = df.index.to_period("M").astype(str)
    months = sorted(df["_month"].unique())

    # Training window = earliest full month (proxy for "in-sample" distribution)
    train_month = months[0]
    df_train = df[df["_month"] == train_month]
    print(f"\nReference (train proxy) month: {train_month} (n={len(df_train)})")

    # Per-month diagnostics
    print("\n" + "=" * 85)
    print(f"{'month':<10}{'n':>6}{'|ret| μ':>10}{'|ret| σ':>10}{'ret σ':>9}{'mag_ic':>9}{'mag_icir_proxy':>16}")
    print("-" * 85)

    month_stats = {}
    for m in months:
        sub = df[df["_month"] == m]
        valid = sub[["_abs_ret", "_mag_pred"]].dropna()
        if len(valid) < 30:
            continue
        abs_mu = float(valid["_abs_ret"].mean())
        abs_sd = float(valid["_abs_ret"].std())
        ret_sd = float(sub["_abs_ret"].std())  # proxy
        ic, _ = spearmanr(valid["_mag_pred"], valid["_abs_ret"])
        # ICIR proxy: split in two halves, compute IC per half
        half = len(valid) // 2
        ic1, _ = spearmanr(valid.iloc[:half]["_mag_pred"], valid.iloc[:half]["_abs_ret"])
        ic2, _ = spearmanr(valid.iloc[half:]["_mag_pred"], valid.iloc[half:]["_abs_ret"])
        icir = float(np.mean([ic1, ic2]) / np.std([ic1, ic2])) if np.std([ic1, ic2]) > 0 else float("nan")
        print(f"{m:<10}{len(valid):>6}{abs_mu:>10.5f}{abs_sd:>10.5f}{ret_sd:>9.5f}{ic:>9.4f}{icir:>+16.3f}")
        month_stats[m] = {
            "n": int(len(valid)),
            "abs_ret_mean": abs_mu,
            "abs_ret_std": abs_sd,
            "mag_ic": float(ic),
            "mag_icir_halves": icir,
        }

    print("\n" + "=" * 85)
    print("FEATURE DRIFT — KS test vs training month (" + train_month + ")")
    print("=" * 85)
    print(f"{'feature':<35}{'month':<10}{'KS stat':>10}{'p-val':>12}{'μ drift %':>12}")
    print("-" * 85)

    drift_records = []
    ref_vals = {f: df_train[f].dropna().values for f in top_feats}
    ref_means = {f: float(df_train[f].mean()) if df_train[f].notna().any() else 0.0 for f in top_feats}

    for f in top_feats:
        for m in months[1:]:  # skip reference month
            sub = df[df["_month"] == m]
            cur_vals = sub[f].dropna().values
            if len(cur_vals) < 30 or len(ref_vals[f]) < 30:
                continue
            ks, p = ks_2samp(ref_vals[f], cur_vals)
            cur_mu = float(np.mean(cur_vals))
            ref_mu = ref_means[f]
            drift_pct = ((cur_mu - ref_mu) / (abs(ref_mu) + 1e-9)) * 100 if ref_mu != 0 else 0.0
            flag = " <<<" if ks > 0.3 and p < 0.01 else ""
            print(f"{f:<35}{m:<10}{ks:>10.4f}{p:>12.2e}{drift_pct:>+11.1f}%{flag}")
            drift_records.append({"feature": f, "month": m, "ks": float(ks),
                                  "p_val": float(p), "drift_pct": drift_pct})

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "train_month": train_month,
        "month_stats": month_stats,
        "feature_drift": drift_records,
        "top_features": top_feats,
    }, indent=2, default=str))
    print(f"\nSaved: {OUT}")

    # Final interpretation
    print("\n" + "=" * 85)
    print("INTERPRETATION")
    print("=" * 85)
    # Did target |ret| drop materially?
    ics = [month_stats[m]["mag_ic"] for m in months if m in month_stats]
    abs_mus = [month_stats[m]["abs_ret_mean"] for m in months if m in month_stats]
    if len(ics) >= 3:
        ic_drop = ics[0] - ics[-1]
        vol_drop = (abs_mus[0] - abs_mus[-1]) / abs_mus[0] * 100
        print(f"  IC drop (first → last):  {ic_drop:+.4f}  ({ics[0]:.3f} → {ics[-1]:.3f})")
        print(f"  |ret| mean change:       {vol_drop:+.1f}%  ({abs_mus[0]:.5f} → {abs_mus[-1]:.5f})")
        if abs(vol_drop) > 20:
            print("  → Significant vol regime change. Target-side drift contributes to IC decay.")
        severe_drift = [r for r in drift_records if r["ks"] > 0.3 and r["p_val"] < 0.01]
        if severe_drift:
            feats_hit = sorted({r["feature"] for r in severe_drift})
            print(f"  → {len(severe_drift)} severe feature drifts (KS > 0.3, p < 0.01)")
            print(f"    Affected features: {feats_hit}")
        else:
            print("  → No severe feature drift detected (top-10 features all stable by KS).")


if __name__ == "__main__":
    run()
