"""
Empirical evidence for steps 2/3/4 of the confidence-WR fix.

Reads:
  research/results/dual_model/direction_concept_drift_oos.parquet  (OOS preds)
  research/dual_model/.cache/features_all.parquet                  (features)

Reports for step 2 (|pred| absolute floor):
  - |pred| percentile boundaries
  - Cumulative sign-acc / IC if we threshold at each candidate floor
  - How often production rolling-percentile cutoffs would dip below the floor

Reports for step 3 (isotonic confidence):
  - Fit isotonic regression: sign_correct ~ |pred|
  - Output knot points (X_thresholds_ / y_thresholds_) for inference-side use

Reports for step 4 (realized-vol gate):
  - Sign-acc by realized_vol_20b percentile bucket
  - Sign-acc when filtering out high-rvol bars at various cutoffs
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OOS_PATH = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_concept_drift_oos.parquet"
FEATURES_PATH = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"


def main():
    oos = pd.read_parquet(OOS_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded OOS: n={len(oos)}, features: n={len(features)}")

    oos["abs_pred"] = oos["pred"].abs()
    oos["sign_correct"] = (np.sign(oos["pred"]) == np.sign(oos["y"])).astype(int)
    # |pred| > 0 only — sign undefined at 0
    oos = oos[oos["abs_pred"] > 0].copy()

    # ── STEP 2: |pred| absolute floor ─────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 2: |pred| absolute floor candidates")
    print("=" * 90)
    print("|pred| percentile boundaries (use to anchor floors):")
    for p in [50, 60, 70, 75, 80, 85, 90, 95]:
        print(f"  p{p:>3}: {np.percentile(oos['abs_pred'], p):.5f}")

    print("\nIf we filter to bars with |pred| >= floor:")
    print(f"{'floor':>10} {'kept_pct':>9} {'n':>5} {'sign_acc':>9} {'IC':>9}")
    print("-" * 50)
    for floor in [0.00010, 0.00020, 0.00030, 0.00040, 0.00050,
                  0.00060, 0.00070, 0.00080, 0.00100, 0.00150]:
        mask = oos["abs_pred"] >= floor
        sub = oos[mask]
        if len(sub) < 30:
            print(f"  {floor:>.5f} {len(sub) / len(oos) * 100:>8.1f}% "
                  f"{len(sub):>5d}     <30 — skip")
            continue
        ic, _ = spearmanr(sub["pred"], sub["y"])
        print(f"  {floor:>.5f} {len(sub) / len(oos) * 100:>8.1f}% "
              f"{len(sub):>5d} {sub['sign_correct'].mean():>9.3f} {ic:>+9.4f}")

    print("\nSimulated rolling top-15% cutoffs (production Moderate gate):")
    print("If buffer's recent 500 |pred| has small dispersion, cutoff drops below floor.")
    abs_pred_series = oos["abs_pred"].values
    rolling_top15 = pd.Series(abs_pred_series).rolling(500, min_periods=100).quantile(0.85)
    rolling_top5 = pd.Series(abs_pred_series).rolling(500, min_periods=100).quantile(0.95)
    valid = rolling_top15.dropna()
    print(f"  rolling top-15 cutoff: min={valid.min():.5f} max={valid.max():.5f} "
          f"median={valid.median():.5f}")
    valid5 = rolling_top5.dropna()
    print(f"  rolling top-5  cutoff: min={valid5.min():.5f} max={valid5.max():.5f} "
          f"median={valid5.median():.5f}")
    for floor in [0.00040, 0.00060, 0.00080]:
        below = (valid < floor).mean() * 100
        print(f"  rolling top-15 below {floor:.5f}: {below:.1f}% of time")

    # ── STEP 4: Realized vol gate ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 4: realized_vol_20b gate")
    print("=" * 90)
    if "realized_vol_20b" not in features.columns:
        print("realized_vol_20b not in features — skipping step 4 empirics")
    else:
        rvol = features["realized_vol_20b"].copy()
        oos2 = oos.set_index("ts").copy()
        oos2["rvol"] = rvol.reindex(oos2.index).values
        # Time-series percentile rank (trailing only — production-realistic)
        rvol_full = rvol.dropna()
        rvol_pct = rvol_full.rolling(500, min_periods=100).rank(pct=True)
        oos2["rvol_pct"] = rvol_pct.reindex(oos2.index).values
        oos2 = oos2.dropna(subset=["rvol_pct"])
        print(f"\nUsing trailing 500-bar rvol percentile, n={len(oos2)}")

        print(f"\n{'rvol_pct bucket':>16} {'n':>5} {'sign_acc':>9} {'IC':>9} {'mean|pred|':>12}")
        print("-" * 60)
        for low, high in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            m = (oos2["rvol_pct"] >= low / 100) & (oos2["rvol_pct"] < high / 100 + 0.001)
            sub = oos2[m]
            if len(sub) >= 30:
                ic, _ = spearmanr(sub["pred"], sub["y"])
                print(f"  p{low:>3}-{high:>3}    {len(sub):>5d} {sub['sign_correct'].mean():>9.3f} "
                      f"{ic:>+9.4f} {sub['abs_pred'].mean():>12.5f}")

        print("\nFiltering out HIGH rvol bars (vol gate kicks in above cutoff):")
        print(f"{'cutoff_pct':>10} {'n_kept':>7} {'kept%':>7} {'sign_acc':>9} {'IC':>9}")
        print("-" * 50)
        for cut in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
            sub = oos2[oos2["rvol_pct"] < cut]
            if len(sub) < 30:
                continue
            ic, _ = spearmanr(sub["pred"], sub["y"])
            print(f"     <p{int(cut*100)}  {len(sub):>7d} "
                  f"{len(sub) / len(oos2) * 100:>6.1f}% "
                  f"{sub['sign_correct'].mean():>9.3f} {ic:>+9.4f}")

        # Combined: vol gate + |pred| floor
        print("\nCombined: |pred| >= 0.0004 AND rvol_pct < cutoff:")
        print(f"{'rvol_cut':>10} {'n':>5} {'sign_acc':>9} {'IC':>9}")
        print("-" * 45)
        for cut in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75]:
            mask = (oos2["abs_pred"] >= 0.0004) & (oos2["rvol_pct"] < cut)
            sub = oos2[mask]
            if len(sub) < 30:
                continue
            ic, _ = spearmanr(sub["pred"], sub["y"])
            print(f"   <p{int(cut*100):>3}    {len(sub):>5d} "
                  f"{sub['sign_correct'].mean():>9.3f} {ic:>+9.4f}")

    # ── STEP 3: Isotonic calibration of confidence ────────────────────────
    print("\n" + "=" * 90)
    print("STEP 3: isotonic regression (sign_correct ~ |pred|)")
    print("=" * 90)

    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(oos["abs_pred"].values, oos["sign_correct"].values)
    print(f"Increasing: {iso.increasing_}")
    print(f"X_min..X_max: {iso.X_min_:.5f} .. {iso.X_max_:.5f}")

    print("\nKnot points (used as confidence lookup table in inference):")
    print(f"{'|pred|':>10} {'P(correct)':>11}")
    print("-" * 25)
    for x, y in zip(iso.X_thresholds_, iso.y_thresholds_):
        print(f"  {x:.5f} {y:>10.4f}")

    # Save isotonic knots for inference
    iso_path = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model" / "confidence_isotonic.json"
    iso_path.parent.mkdir(parents=True, exist_ok=True)
    iso_data = {
        "increasing": bool(iso.increasing_),
        "x_thresholds": [float(x) for x in iso.X_thresholds_],
        "y_thresholds": [float(y) for y in iso.y_thresholds_],
        "x_min": float(iso.X_min_),
        "x_max": float(iso.X_max_),
        "n_train_oos_samples": len(oos),
        "fit_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open(iso_path, "w") as f:
        json.dump(iso_data, f, indent=2)
    print(f"\nSaved isotonic knots → {iso_path}")

    # Sanity: confidence at example |pred| values
    print("\nIsotonic confidence at example |pred|:")
    for x in [0.0001, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]:
        p = float(iso.predict([x])[0])
        print(f"  |pred|={x:.5f}  →  P(correct)={p:.3f}  →  conf~{p*100:.1f}")


if __name__ == "__main__":
    main()
