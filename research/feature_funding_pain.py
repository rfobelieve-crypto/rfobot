"""
IC validation for Funding Pain Accumulation features.

Intuition: When funding stays positive for many hours AND OI keeps growing,
longs are paying more and more to hold. This pressure eventually reverses.

Features:
  funding_pain_24h      — 24h rolling sum of funding*sign(funding) * OI change
  funding_pain_48h      — 48h window variant
  funding_cum_stress    — cumulative funding deviation from 0 over 72h
  funding_squeeze       — abs(funding) * OI pct change 8h (extreme + growth)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.build_direction_labels import build_direction_labels

CACHE = Path(__file__).resolve().parent / "dual_model" / ".cache" / "features_all.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, needed: dict[str, list[str]]) -> dict[str, str]:
    """Map logical names to actual column names. Returns mapping or raises."""
    available_cg = sorted(c for c in df.columns if c.startswith("cg_"))
    mapping = {}
    missing = []
    for logical, candidates in needed.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            missing.append((logical, candidates))
        else:
            mapping[logical] = found

    if missing:
        print("\n=== Available cg_* columns ===")
        for c in available_cg:
            print(f"  {c}")
        print()
        for logical, candidates in missing:
            print(f"ERROR: No column found for '{logical}'. Tried: {candidates}")
        print("\nPlease update the candidate lists above to match your data.")
        sys.exit(1)

    return mapping


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all funding pain features. Returns DataFrame aligned to df.index."""
    col_map = _require_columns(df, {
        "funding": ["cg_funding_close", "cg_funding_rate", "cg_funding_agg_close"],
        "oi": ["cg_oi_close", "cg_oi_agg_close"],
    })

    funding = df[col_map["funding"]].fillna(0).astype(float)
    oi = df[col_map["oi"]].fillna(method="ffill").astype(float)

    # OI hourly change
    oi_change_1h = oi.diff(1)
    # OI 8h pct change
    oi_pct_8h = oi.pct_change(8)

    out = pd.DataFrame(index=df.index)

    # --- funding_pain_24h ---
    # rolling sum of funding*sign(funding) captures one-sided accumulation
    funding_signed = funding * np.sign(funding)  # always positive = accumulated cost
    pain_component = funding_signed.rolling(24, min_periods=12).sum()
    oi_growth_24h = oi_change_1h.rolling(24, min_periods=12).mean()
    out["funding_pain_24h"] = pain_component * oi_growth_24h

    # --- funding_pain_48h ---
    pain_48 = funding_signed.rolling(48, min_periods=24).sum()
    oi_growth_48h = oi_change_1h.rolling(48, min_periods=24).mean()
    out["funding_pain_48h"] = pain_48 * oi_growth_48h

    # --- funding_cum_stress ---
    # How far has funding deviated from neutral over 72h?
    # Large positive = longs paying a lot cumulatively = bearish pressure
    out["funding_cum_stress"] = funding.rolling(72, min_periods=36).sum()

    # --- funding_squeeze ---
    # Extreme funding + OI growth = squeeze setup
    out["funding_squeeze"] = funding.abs() * oi_pct_8h

    # Normalize features to avoid scale issues
    for col in out.columns:
        s = out[col]
        mu = s.rolling(168, min_periods=84).mean()
        sd = s.rolling(168, min_periods=84).std().replace(0, np.nan)
        out[col] = (s - mu) / sd

    logger.info("Built %d funding pain features", len(out.columns))
    return out


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def spearman_ic(feat: pd.Series, target: pd.Series) -> tuple[float, float, int]:
    """Compute Spearman IC. Returns (rho, pval, n)."""
    mask = feat.notna() & target.notna()
    n = int(mask.sum())
    if n < 100:
        return np.nan, np.nan, n
    rho, p = spearmanr(feat[mask], target[mask])
    return rho, p, n


def bootstrap_ic(feat: pd.Series, target: pd.Series, n_boot: int = 1000,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for Spearman IC. Returns (mean, ci_lo, ci_hi)."""
    mask = feat.notna() & target.notna()
    f = feat[mask].values
    t = target[mask].values
    n = len(f)
    if n < 100:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = spearmanr(f[idx], t[idx])
        ics.append(rho)
    ics = np.array(ics)
    return float(np.mean(ics)), float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))


def monthly_ic(feat: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Compute IC per calendar month."""
    combined = pd.DataFrame({"feat": feat, "target": target}).dropna()
    if len(combined) == 0:
        return pd.DataFrame()
    combined["month"] = combined.index.to_period("M")
    rows = []
    for m, grp in combined.groupby("month"):
        if len(grp) < 50:
            continue
        rho, p = spearmanr(grp["feat"], grp["target"])
        rows.append({"month": str(m), "ic": rho, "p": p, "n": len(grp)})
    return pd.DataFrame(rows)


def correlation_with_existing(new_feat: pd.Series, df: pd.DataFrame,
                               top_features: list[str]) -> dict[str, float]:
    """Compute correlation of new feature with top existing features."""
    corrs = {}
    for ref in top_features:
        if ref in df.columns:
            mask = new_feat.notna() & df[ref].notna()
            if mask.sum() > 100:
                corrs[ref] = float(new_feat[mask].corr(df[ref][mask]))
    return corrs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOP_EXISTING = [
    "cg_oi_close_zscore", "cg_funding_close_zscore", "cg_bfx_margin_ratio",
    "vol_acceleration", "cg_liq_imbalance_zscore", "oi_price_divergence",
    "impact_asymmetry", "realized_vol_20b",
]


def main():
    print("=" * 70)
    print("Funding Pain Accumulation — Feature IC Validation")
    print("=" * 70)

    # Load data
    if not CACHE.exists():
        print(f"ERROR: Cache not found at {CACHE}. Run shared_data.py first.")
        sys.exit(1)
    df = pd.read_parquet(CACHE)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()
    print(f"\nData: {len(df)} bars x {len(df.columns)} cols")
    print(f"Range: {df.index.min()} -> {df.index.max()}")

    # Print available cg_* columns for reference
    cg_cols = sorted(c for c in df.columns if c.startswith("cg_"))
    print(f"\nAvailable cg_* columns ({len(cg_cols)}):")
    for c in cg_cols:
        print(f"  {c}")

    # Build labels
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    target = labels["return_4h"]
    y_path = labels.get("path_return_4h", target)  # prefer TWAP if available

    # Build features
    new_feats = build_features(df)
    print(f"\nNew features: {list(new_feats.columns)}")
    for c in new_feats.columns:
        s = new_feats[c]
        print(f"  {c:28s}  nan={s.isna().mean():.1%}  "
              f"mean={s.mean():+.3f}  std={s.std():.3f}")

    # --- Full-sample IC ---
    print("\n" + "=" * 70)
    print("Full-sample Spearman IC vs y_path_ret_4h")
    print("-" * 70)
    results = {}
    for col in new_feats.columns:
        rho, p, n = spearman_ic(new_feats[col], y_path)
        print(f"  {col:28s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
        results[col] = {"ic": rho, "p": p, "n": n}

    # --- Bootstrap CI ---
    print("\n" + "=" * 70)
    print("Bootstrap 95% CI (1000 resamples)")
    print("-" * 70)
    for col in new_feats.columns:
        mean_ic, lo, hi = bootstrap_ic(new_feats[col], y_path)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {col:28s}  IC={mean_ic:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
        results[col]["bootstrap_mean"] = mean_ic
        results[col]["ci_lo"] = lo
        results[col]["ci_hi"] = hi
        results[col]["significant"] = sig == "SIG"

    # --- Monthly stability ---
    print("\n" + "=" * 70)
    print("Monthly IC Stability")
    print("-" * 70)
    for col in new_feats.columns:
        mic = monthly_ic(new_feats[col], y_path)
        if mic.empty:
            print(f"  {col}: insufficient data")
            continue
        print(f"\n  {col}:")
        sign_consistent = 0
        dominant_sign = np.sign(mic["ic"].mean())
        for _, row in mic.iterrows():
            marker = "+" if np.sign(row["ic"]) == dominant_sign else "FLIP"
            print(f"    {row['month']:8s}  IC={row['ic']:+.4f}  n={row['n']:4d}  {marker}")
            if np.sign(row["ic"]) == dominant_sign:
                sign_consistent += 1
        consistency = sign_consistent / len(mic) if len(mic) > 0 else 0
        print(f"    Sign consistency: {consistency:.0%} ({sign_consistent}/{len(mic)})")
        results[col]["monthly_consistency"] = consistency
        results[col]["monthly_ics"] = mic.to_dict(orient="records")

    # --- Correlation with existing top features ---
    print("\n" + "=" * 70)
    print("Correlation with existing top features (|corr| > 0.8 = redundant)")
    print("-" * 70)
    for col in new_feats.columns:
        corrs = correlation_with_existing(new_feats[col], df, TOP_EXISTING)
        results[col]["correlations"] = corrs
        if not corrs:
            print(f"  {col}: no overlap with reference features")
            continue
        max_corr = max(abs(v) for v in corrs.values())
        flag = "REDUNDANT" if max_corr > 0.8 else "OK"
        print(f"  {col} (max |corr|={max_corr:.3f} {flag}):")
        for ref, c in sorted(corrs.items(), key=lambda x: -abs(x[1])):
            print(f"    vs {ref:35s}  corr={c:+.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  {'Feature':28s}  {'IC':>8s}  {'CI':>22s}  {'Sig':>4s}  {'Consist':>8s}")
    for col in new_feats.columns:
        r = results[col]
        ic_str = f"{r['ic']:+.4f}" if not np.isnan(r.get('ic', np.nan)) else "  N/A"
        ci_str = f"[{r.get('ci_lo', np.nan):+.4f}, {r.get('ci_hi', np.nan):+.4f}]"
        sig_str = "YES" if r.get("significant", False) else " no"
        cons_str = f"{r.get('monthly_consistency', 0):.0%}"
        print(f"  {col:28s}  {ic_str}  {ci_str}  {sig_str}  {cons_str:>8s}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_funding_pain_ic.json"

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return obj

    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: _convert(vv) if not isinstance(vv, (dict, list)) else vv
                           for kk, vv in v.items()}

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
