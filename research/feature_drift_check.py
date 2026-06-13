"""
Feature drift detection for the deployed direction + magnitude models.

Checks two questions:
  Q1. Within the training window, did the feature distributions shift
      between earlier months (reference) and the most recent month
      (test)? If yes, the model trained on a non-stationary signal
      and predictions in the recent regime are extrapolative.
  Q2. Vs feature importance — are the most drifted features also the
      most important to the model? A drifted #1 feature is far worse
      than a drifted #50 feature.

Metrics:
  - PSI (Population Stability Index) on 10 quantile bins:
      < 0.10 = stable, 0.10–0.25 = moderate drift,
      > 0.25 = significant drift, > 0.50 = severe.
  - z-score of test mean: |Δμ| / ref_σ. > 0.5 noteworthy, > 1.0 strong.

Outputs a ranked table of (feature, importance_rank, PSI, |z|) sorted
by PSI × importance — i.e. worst-drifted-times-most-important first.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
ART = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
DIR_FEATS = ART / "direction_feature_cols.json"
DIR_IMP = ART / "direction_importance.csv"
MAG_FEATS = ART / "magnitude_feature_cols.json"
MAG_IMP = ART / "magnitude_importance.csv"


def psi(reference: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on quantile bins of `reference`.

    Bins are defined by `reference` quantiles, then both arrays are
    histogrammed into those bins. PSI = Σ (test_p - ref_p) × ln(test_p / ref_p).
    epsilon-floor on bin probabilities to avoid log(0).
    """
    reference = reference[~np.isnan(reference)]
    test = test[~np.isnan(test)]
    if len(reference) < 50 or len(test) < 20:
        return float("nan")
    # If feature is near-constant, PSI is undefined / near-zero
    if reference.std() < 1e-12:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    test_hist, _ = np.histogram(test, bins=edges)
    ref_p = ref_hist / ref_hist.sum()
    test_p = test_hist / test_hist.sum()
    eps = 1e-6
    ref_p = np.where(ref_p == 0, eps, ref_p)
    test_p = np.where(test_p == 0, eps, test_p)
    return float(np.sum((test_p - ref_p) * np.log(test_p / ref_p)))


def z_drift(reference: np.ndarray, test: np.ndarray) -> float:
    """|Δμ| / ref_σ — units of training std the test mean shifted."""
    reference = reference[~np.isnan(reference)]
    test = test[~np.isnan(test)]
    if len(reference) < 50 or len(test) < 20:
        return float("nan")
    ref_sigma = reference.std(ddof=0)
    if ref_sigma < 1e-12:
        return 0.0
    return float(abs(test.mean() - reference.mean()) / ref_sigma)


def load_importance(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    imp = pd.read_csv(csv_path, index_col=0)
    if "importance" in imp.columns:
        return imp["importance"].to_dict()
    if "gain" in imp.columns:
        return imp["gain"].to_dict()
    # First numeric col
    for c in imp.columns:
        if pd.api.types.is_numeric_dtype(imp[c]):
            return imp[c].to_dict()
    return {}


def evaluate(df: pd.DataFrame, features: list[str], importance: dict[str, float],
             ref_mask: pd.Series, test_mask: pd.Series, label: str):
    """Compute drift table for one model's feature set."""
    rows = []
    ref_n = int(ref_mask.sum())
    test_n = int(test_mask.sum())
    for f in features:
        if f not in df.columns:
            rows.append({"feature": f, "psi": np.nan, "z": np.nan,
                         "importance": importance.get(f, 0.0),
                         "missing": True})
            continue
        ref = df.loc[ref_mask, f].values.astype(float)
        tst = df.loc[test_mask, f].values.astype(float)
        rows.append({
            "feature": f,
            "psi": psi(ref, tst),
            "z": z_drift(ref, tst),
            "importance": importance.get(f, 0.0),
            "missing": False,
        })
    drift = pd.DataFrame(rows)
    drift["score"] = drift["psi"].fillna(0) * drift["importance"]
    drift = drift.sort_values("score", ascending=False)

    # Importance ranking: 1 = highest importance
    # For features with 0 importance (not in importance dict), rank = NaN
    imp_sorted = sorted(importance.items(), key=lambda x: -x[1])
    rank_lookup = {f: i + 1 for i, (f, _) in enumerate(imp_sorted)}
    drift["imp_rank"] = drift["feature"].map(rank_lookup)

    print(f"\n{'=' * 78}")
    print(f"  {label}  ref n={ref_n}, test n={test_n}")
    print(f"{'=' * 78}")
    sig = drift[drift["psi"] >= 0.10].copy()
    sev = drift[drift["psi"] >= 0.25].copy()
    print(f"  Stable      (PSI < 0.10): {(drift['psi'] < 0.10).sum()}")
    print(f"  Moderate     (0.10-0.25): {((drift['psi'] >= 0.10) & (drift['psi'] < 0.25)).sum()}")
    print(f"  Significant   (>= 0.25): {(drift['psi'] >= 0.25).sum()}")
    print(f"  Severe         (>= 0.50): {(drift['psi'] >= 0.50).sum()}")
    print()

    # Top 20 by PSI × importance
    top = drift.head(20)
    print(f"  Top 20 drifters (PSI × importance):")
    print(f"  {'feature':<35s}  {'imp_rank':>8s}  {'imp':>7s}  {'PSI':>6s}  {'|z|':>5s}  flag")
    for _, r in top.iterrows():
        if pd.isna(r["psi"]):
            continue
        flag = ""
        if r["psi"] >= 0.50:
            flag = "🚨 SEVERE"
        elif r["psi"] >= 0.25:
            flag = "⚠ significant"
        elif r["psi"] >= 0.10:
            flag = "moderate"
        rank_s = f"{int(r['imp_rank']):>8d}" if pd.notna(r["imp_rank"]) else "       -"
        print(f"  {r['feature']:<35s}  {rank_s}  {r['importance']:>7.3f}  "
              f"{r['psi']:>6.3f}  {r['z']:>5.2f}  {flag}")

    # Top-20 most important features regardless of drift
    if importance:
        top_imp = drift.copy()
        top_imp = top_imp.dropna(subset=["imp_rank"])
        top_imp = top_imp.sort_values("imp_rank").head(20)
        print(f"\n  Top 20 by importance — drift status:")
        print(f"  {'feature':<35s}  {'imp_rank':>8s}  {'imp':>7s}  {'PSI':>6s}  {'|z|':>5s}  flag")
        for _, r in top_imp.iterrows():
            flag = ""
            if pd.notna(r["psi"]):
                if r["psi"] >= 0.50:
                    flag = "🚨 SEVERE"
                elif r["psi"] >= 0.25:
                    flag = "⚠ significant"
                elif r["psi"] >= 0.10:
                    flag = "moderate"
                else:
                    flag = "stable"
            print(f"  {r['feature']:<35s}  {int(r['imp_rank']):>8d}  {r['importance']:>7.3f}  "
                  f"{r['psi']:>6.3f}  {r['z']:>5.2f}  {flag}")

    return drift


def main():
    df = pd.read_parquet(CACHE).sort_index()
    print(f"Loaded {len(df)} bars × {len(df.columns)} cols")
    print(f"Range: {df.index.min()} ~ {df.index.max()}")

    dir_feats = json.load(open(DIR_FEATS))
    mag_feats = json.load(open(MAG_FEATS))
    dir_imp = load_importance(DIR_IMP)
    mag_imp = load_importance(MAG_IMP)
    print(f"Direction model: {len(dir_feats)} features, {len(dir_imp)} have importance")
    print(f"Magnitude model: {len(mag_feats)} features, {len(mag_imp)} have importance")

    # Question 1: Within-training-window drift
    # Reference: 2025-12-01 to 2026-03-31  (4 months, ~2900 bars)
    # Test: 2026-04-01 to end
    ref_mask_q1 = (df.index >= pd.Timestamp("2025-12-01", tz="UTC")) & \
                  (df.index < pd.Timestamp("2026-04-01", tz="UTC"))
    test_mask_q1 = df.index >= pd.Timestamp("2026-04-01", tz="UTC")
    print(f"\n{'#' * 78}")
    print(f"  Q1: Within-training-window drift  "
          f"(ref: 2025-12 to 2026-03, test: 2026-04)")
    print(f"{'#' * 78}")
    evaluate(df, dir_feats, dir_imp, ref_mask_q1, test_mask_q1,
             "DIRECTION model — within-training drift")
    evaluate(df, mag_feats, mag_imp, ref_mask_q1, test_mask_q1,
             "MAGNITUDE model — within-training drift")

    # Question 2: Recent (last 7 days) vs full training distribution
    # Useful as a leading indicator for "is the model still in-distribution
    # right now?"
    ref_mask_q2 = df.index < df.index.max() - pd.Timedelta(days=7)
    test_mask_q2 = df.index >= df.index.max() - pd.Timedelta(days=7)
    print(f"\n{'#' * 78}")
    print(f"  Q2: Last 7 days vs prior training data  "
          f"(test: last 7 days, ref: everything earlier)")
    print(f"{'#' * 78}")
    evaluate(df, dir_feats, dir_imp, ref_mask_q2, test_mask_q2,
             "DIRECTION model — recent 7d drift")
    evaluate(df, mag_feats, mag_imp, ref_mask_q2, test_mask_q2,
             "MAGNITUDE model — recent 7d drift")


if __name__ == "__main__":
    main()
