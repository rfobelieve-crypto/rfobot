"""
Per-feature monthly IC timeline for the magnitude model.

Goal: pinpoint WHICH features lost predictive power for |ret_4h| in Mar/Apr
      compared to their Nov-Jan baseline. This is concept drift localization.

Method:
    For each feature in magnitude_feature_cols.json, compute monthly spearman
    rank correlation with |ret_4h|. Rank features by (early_avg - late_avg)
    IC drop.

    early = Nov 2025 + Dec 2025 + Jan 2026
    late  = Mar 2026 + Apr 2026

Output:
    research/results/mag_feature_ic_timeline.json
    research/results/mag_feature_ic_timeline.csv  (sorted by drop)

Usage:
    python research/mag_feature_ic_timeline.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data
from research.dual_model.build_magnitude_labels import build_magnitude_labels

FEATS_PATH = Path("indicator/model_artifacts/dual_model/magnitude_feature_cols.json")
OUT_JSON = Path("research/results/mag_feature_ic_timeline.json")
OUT_CSV = Path("research/results/mag_feature_ic_timeline.csv")

EARLY_MONTHS = ["2025-11", "2025-12", "2026-01"]
LATE_MONTHS = ["2026-03", "2026-04"]
ALL_MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


def monthly_ic(values: np.ndarray, target: np.ndarray, months: np.ndarray,
               month_label: str) -> tuple[float, int]:
    mask = months == month_label
    if mask.sum() < 30:
        return np.nan, int(mask.sum())
    v = values[mask]
    t = target[mask]
    ok = np.isfinite(v) & np.isfinite(t)
    if ok.sum() < 30:
        return np.nan, int(ok.sum())
    if np.nanstd(v[ok]) == 0:
        return np.nan, int(ok.sum())
    ic, _ = spearmanr(v[ok], t[ok])
    return float(ic), int(ok.sum())


def main():
    feats = json.loads(FEATS_PATH.read_text())
    print(f"Loaded {len(feats)} magnitude features from {FEATS_PATH.name}")

    df = load_and_cache_data()
    labels = build_magnitude_labels(df)
    target = labels["y_abs_return"].values

    # Align
    df = df.copy()
    df["_tgt"] = target
    df = df.dropna(subset=["_tgt"])
    months = df.index.strftime("%Y-%m").values
    target = df["_tgt"].values

    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"[warn] {len(missing)} features missing from cache: {missing[:5]}...")
    feats = [f for f in feats if f in df.columns]

    print(f"Evaluating {len(feats)} features x {len(ALL_MONTHS)} months...")
    rows = []
    for fname in feats:
        vals = df[fname].values
        rec = {"feature": fname}
        for m in ALL_MONTHS:
            ic, n = monthly_ic(vals, target, months, m)
            rec[f"ic_{m}"] = ic
            rec[f"n_{m}"] = n
        early = [rec[f"ic_{m}"] for m in EARLY_MONTHS if not np.isnan(rec[f"ic_{m}"])]
        late = [rec[f"ic_{m}"] for m in LATE_MONTHS if not np.isnan(rec[f"ic_{m}"])]
        rec["early_avg"] = float(np.mean([abs(x) for x in early])) if early else np.nan
        rec["late_avg"] = float(np.mean([abs(x) for x in late])) if late else np.nan
        rec["ic_drop"] = rec["early_avg"] - rec["late_avg"] if early and late else np.nan
        # Sign flip detection: was early_avg_signed positive then late negative, or vice versa?
        rec["early_signed"] = float(np.mean(early)) if early else np.nan
        rec["late_signed"] = float(np.mean(late)) if late else np.nan
        rec["sign_flip"] = bool(
            not np.isnan(rec["early_signed"]) and not np.isnan(rec["late_signed"])
            and np.sign(rec["early_signed"]) != np.sign(rec["late_signed"])
            and abs(rec["early_signed"]) > 0.03 and abs(rec["late_signed"]) > 0.03
        )
        rows.append(rec)

    ranked = sorted(rows, key=lambda r: -(r["ic_drop"] if not np.isnan(r["ic_drop"]) else -999))

    # Print top-15 decayed
    print()
    print("=" * 96)
    print("TOP 15 FEATURES BY IC DECAY (|early| - |late|)")
    print("=" * 96)
    print(f"{'feature':<38} {'early':>7} {'late':>7} {'drop':>7} "
          f"{'Nov':>6} {'Dec':>6} {'Jan':>6} {'Feb':>6} {'Mar':>6} {'Apr':>6} {'flip':>5}")
    print("-" * 96)
    for r in ranked[:15]:
        fmt = lambda x: f"{x:+.3f}" if not np.isnan(x) else "  nan"
        print(f"{r['feature']:<38} "
              f"{r['early_avg']:>+7.3f} {r['late_avg']:>+7.3f} {r['ic_drop']:>+7.3f} "
              f"{fmt(r['ic_2025-11']):>6} {fmt(r['ic_2025-12']):>6} "
              f"{fmt(r['ic_2026-01']):>6} {fmt(r['ic_2026-02']):>6} "
              f"{fmt(r['ic_2026-03']):>6} {fmt(r['ic_2026-04']):>6} "
              f"{'YES' if r['sign_flip'] else '':>5}")

    # Features that FLIPPED sign (potentially worse than just decayed)
    flipped = [r for r in rows if r["sign_flip"]]
    if flipped:
        print()
        print("=" * 96)
        print(f"SIGN-FLIPPED FEATURES ({len(flipped)}) — relationship reversed")
        print("=" * 96)
        for r in sorted(flipped, key=lambda x: -abs(x["early_signed"] - x["late_signed"]))[:10]:
            print(f"  {r['feature']:<38} "
                  f"early {r['early_signed']:+.3f} → late {r['late_signed']:+.3f}  "
                  f"(Δ={r['late_signed']-r['early_signed']:+.3f})")

    # Features that STAYED strong (survivors)
    survivors = [r for r in rows
                 if not np.isnan(r["early_avg"]) and not np.isnan(r["late_avg"])
                 and r["early_avg"] > 0.08 and r["late_avg"] > 0.08]
    if survivors:
        print()
        print("=" * 96)
        print(f"SURVIVORS ({len(survivors)}) — IC > 0.08 in both early and late")
        print("=" * 96)
        for r in sorted(survivors, key=lambda x: -x["late_avg"])[:10]:
            print(f"  {r['feature']:<38} "
                  f"early {r['early_avg']:.3f}  late {r['late_avg']:.3f}")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_features": len(feats),
        "early_months": EARLY_MONTHS,
        "late_months": LATE_MONTHS,
        "rows": rows,
        "ranked_top15_decay": [r["feature"] for r in ranked[:15]],
        "flipped": [r["feature"] for r in flipped],
        "survivors": [r["feature"] for r in survivors],
    }, indent=2, default=str))

    pd.DataFrame(ranked).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
