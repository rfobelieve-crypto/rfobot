"""Half-life check for V7 features (AR(1) persistence).

For each feature in FULL_DIRECTION (or the union of available numeric
columns), fit an AR(1) on the series and compute the half-life:

    rho = corr(x_{t-1}, x_t) via OLS (slope)
    t_half = -ln(2) / ln(rho)         (in bars)

Interpretation:
    rho >= 1   → non-stationary / drifting          [excluded]
    rho <= 0   → no persistence (white noise / anti-persistent) [excluded]
    rho ~ 0.99 → t_half ~ 69 bars → slow, tradable feature
    rho ~ 0.95 → t_half ~ 14 bars → medium persistence
    rho ~ 0.80 → t_half ~ 3 bars  → fast-decaying, costs probably eat the edge
    rho ~ 0.50 → t_half ~ 1 bar   → basically noise

Why this matters: a feature whose half-life is much shorter than your
retraining cadence is paying execution costs to chase noise.  When you
re-train monthly but a feature's half-life is 2 bars (= 2h on 1h target),
that feature contributes mostly variance, not signal.

Surfaces a ranked table and a "prune candidates" list for the next
re-train cycle.  Read-only; does not modify any production state.

Usage:
    python scripts/feature_halflife_check.py
    python scripts/feature_halflife_check.py --top 30
    python scripts/feature_halflife_check.py --filter cg_     # prefix filter

Read-only.  Requires research/dual_model/.cache/features_all.parquet
(gitignored — regenerate locally via research.dual_model.shared_data
if missing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"


def ar1_halflife_bars(series: pd.Series) -> tuple[float, float, int]:
    """Compute AR(1) slope (rho) and resulting half-life in bars.

    Returns (rho, halflife_bars, n_observations).  halflife is NaN when
    rho is not in (0, 1) — that is, the feature is non-stationary,
    drifting, or has no positive persistence.
    """
    s = series.dropna().astype(float)
    if len(s) < 100:
        return (np.nan, np.nan, len(s))
    x_lag = s.iloc[:-1].values
    x_now = s.iloc[1:].values
    # Demean both — AR(1) without intercept can flip sign on biased series
    x_lag_c = x_lag - x_lag.mean()
    x_now_c = x_now - x_now.mean()
    denom = (x_lag_c ** 2).sum()
    if denom <= 0:
        return (np.nan, np.nan, len(s))
    rho = float((x_lag_c * x_now_c).sum() / denom)
    if not (0 < rho < 1):
        return (rho, np.nan, len(s))
    halflife = -np.log(2) / np.log(rho)
    return (rho, float(halflife), len(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50,
                    help="Show the slowest-decaying N features (default 50)")
    ap.add_argument("--filter", type=str, default=None,
                    help="Prefix filter on feature names (e.g. 'cg_')")
    ap.add_argument("--short-cutoff", type=float, default=5.0,
                    help="Half-life below this (bars) flags a prune candidate")
    args = ap.parse_args()

    if not FEATURE_CACHE.exists():
        print(f"ERROR: feature cache not found at {FEATURE_CACHE}")
        print("Regenerate locally via research.dual_model.shared_data.")
        return 1

    print(f"Loading {FEATURE_CACHE} …")
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"  shape: {df.shape}")

    # Pick numeric columns only; exclude obvious non-feature columns
    skip_prefixes = ("y_", "endpoint_", "abs_path", "ret_", "regime_label")
    skip_exact = {"open", "high", "low", "close", "volume", "dt", "ts"}
    candidates = [
        c for c in df.columns
        if df[c].dtype.kind in "fc"
        and c not in skip_exact
        and not any(c.startswith(p) for p in skip_prefixes)
    ]
    if args.filter:
        candidates = [c for c in candidates if c.startswith(args.filter)]
    print(f"  scanning {len(candidates)} feature columns")

    rows = []
    for col in candidates:
        rho, halflife, n = ar1_halflife_bars(df[col])
        rows.append({"feature": col, "rho": rho, "halflife_bars": halflife,
                     "n": n})
    result = pd.DataFrame(rows).sort_values(
        "halflife_bars", ascending=False, na_position="last")

    # Buckets
    stationary = result[result["halflife_bars"].notna()]
    nonstation = result[result["halflife_bars"].isna()]
    short = stationary[stationary["halflife_bars"] < args.short_cutoff]

    print()
    print(f"=== HALF-LIFE SUMMARY ===")
    print(f"  Stationary (rho ∈ (0,1)) : {len(stationary)}")
    print(f"  Non-stationary / random  : {len(nonstation)}")
    print(f"  Short half-life (< {args.short_cutoff} bars): {len(short)} "
          f"← prune candidates")

    # Top-N slowest (most tradable)
    print()
    print(f"=== TOP {args.top} SLOWEST-DECAYING FEATURES (most tradable) ===")
    print(f"{'feature':<48} {'rho':>7} {'half-life (bars)':>18} {'n':>7}")
    print("-" * 84)
    for _, r in stationary.head(args.top).iterrows():
        print(f"{r['feature'][:48]:<48} "
              f"{r['rho']:>7.4f} {r['halflife_bars']:>18.1f} "
              f"{r['n']:>7}")

    # Bottom-N fastest (prune candidates)
    if len(short):
        print()
        print(f"=== SHORT HALF-LIFE FEATURES (< {args.short_cutoff} bars) ===")
        print("These contribute mostly variance.  Prune candidates for the "
              "next re-train.")
        print(f"{'feature':<48} {'rho':>7} {'half-life (bars)':>18} {'n':>7}")
        print("-" * 84)
        for _, r in short.iterrows():
            print(f"{r['feature'][:48]:<48} "
                  f"{r['rho']:>7.4f} {r['halflife_bars']:>18.1f} "
                  f"{r['n']:>7}")

    # Non-stationary features
    if len(nonstation):
        print()
        print(f"=== NON-STATIONARY / NON-PERSISTENT ({len(nonstation)}) ===")
        print("(rho not in (0,1) — either drifting, anti-persistent, "
              "or pure noise)")
        print(f"{'feature':<48} {'rho':>7} {'n':>7}")
        print("-" * 65)
        for _, r in nonstation.head(20).iterrows():
            rho_str = f"{r['rho']:.4f}" if pd.notna(r["rho"]) else "NaN"
            print(f"{r['feature'][:48]:<48} {rho_str:>7} {r['n']:>7}")
        if len(nonstation) > 20:
            print(f"  … {len(nonstation) - 20} more")

    # Write CSV for follow-up analysis
    out_csv = PROJECT_ROOT / "research" / "results" / "feature_halflife.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print()
    print(f"Wrote full table → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
