"""
IC validation for Participant Divergence features.

Intuition: When different market participants disagree (e.g., LS ratio says
bullish but funding is negative), the resulting tension often resolves in a
directional move. Divergence = potential energy for price discovery.

Features:
  ls_funding_divergence     — LS ratio vs funding rate disagreement
  taker_oi_divergence       — takers buying but OI decreasing (closing, not opening)
  cross_exchange_divergence — Coinbase premium vs Bitfinex margin agreement/divergence
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

# Column candidates: (logical_name, [candidate_col_names], required)
COLUMN_SPEC = {
    "ls_ratio":       (["cg_ls_ratio", "cg_ls_agg_ratio"], True),
    "funding":        (["cg_funding_close", "cg_funding_rate", "cg_funding_agg_close"], True),
    "taker_delta":    (["cg_taker_delta", "cg_taker_agg_delta", "cg_taker_buy_sell_delta"], True),
    "oi":             (["cg_oi_close", "cg_oi_agg_close"], True),
    "cb_premium_z":   (["cg_cb_premium_zscore", "cg_coinbase_premium_zscore",
                         "cg_coinbase_premium_index_zscore"], False),
    "bfx_margin_z":   (["cg_bfx_margin_zscore", "cg_bfx_margin_ratio_zscore",
                         "cg_bitfinex_margin_zscore", "cg_bfx_margin_ratio"], False),
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Resolve logical column names to actual columns. None if not found."""
    available_cg = sorted(c for c in df.columns if c.startswith("cg_"))
    mapping = {}
    has_error = False

    for logical, (candidates, required) in COLUMN_SPEC.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        mapping[logical] = found
        if found is None and required:
            has_error = True
            logger.error("Required column '%s' not found. Tried: %s", logical, candidates)

    if has_error:
        print("\n=== Available cg_* columns ===")
        for c in available_cg:
            print(f"  {c}")
        print("\nPlease update COLUMN_SPEC candidate lists to match your data.")
        sys.exit(1)

    return mapping


def _zscore_rolling(s: pd.Series, window: int = 168) -> pd.Series:
    """Rolling z-score normalization."""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (s - mu) / sd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all divergence features. Returns DataFrame aligned to df.index."""
    col_map = _resolve_columns(df)
    out = pd.DataFrame(index=df.index)

    # --- ls_funding_divergence ---
    # When LS ratio > 1 (more longs) but funding negative (shorts paying),
    # or LS < 1 (more shorts) but funding positive (longs paying) = divergence
    ls = df[col_map["ls_ratio"]].fillna(method="ffill").astype(float)
    funding = df[col_map["funding"]].fillna(0).astype(float)
    raw_div = (ls - 1.0) * (-funding) * 1000
    out["ls_funding_divergence"] = _zscore_rolling(raw_div)

    # --- taker_oi_divergence ---
    # Takers buying (positive delta) but OI decreasing = closing shorts, not new longs
    # This is weaker than it looks — true buying should also open new positions
    taker = df[col_map["taker_delta"]].fillna(0).astype(float)
    oi = df[col_map["oi"]].fillna(method="ffill").astype(float)
    oi_pct_1h = oi.pct_change(1).clip(-0.1, 0.1)  # clip extremes
    raw_taker_oi = taker * (-oi_pct_1h)
    out["taker_oi_divergence"] = _zscore_rolling(raw_taker_oi)

    # --- cross_exchange_divergence ---
    cb_col = col_map.get("cb_premium_z")
    bfx_col = col_map.get("bfx_margin_z")
    if cb_col is not None and bfx_col is not None:
        cb = df[cb_col].fillna(0).astype(float)
        bfx = df[bfx_col].fillna(0).astype(float)
        # Same sign = agreement (positive product), opposite = divergence (negative)
        raw_cross = cb * bfx
        out["cross_exchange_divergence"] = _zscore_rolling(raw_cross)
        logger.info("cross_exchange_divergence built using %s x %s", cb_col, bfx_col)
    else:
        skipped = []
        if cb_col is None:
            skipped.append("cb_premium_z")
        if bfx_col is None:
            skipped.append("bfx_margin_z")
        logger.warning("Skipping cross_exchange_divergence: missing %s", skipped)

    logger.info("Built %d divergence features", len(out.columns))
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
    "cg_ls_ratio", "cg_taker_delta", "vol_acceleration",
    "cg_liq_imbalance_zscore", "oi_price_divergence",
    "impact_asymmetry", "realized_vol_20b",
]


def main():
    print("=" * 70)
    print("Participant Divergence — Feature IC Validation")
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
        print(f"  {c:30s}  nan={s.isna().mean():.1%}  "
              f"mean={s.mean():+.3f}  std={s.std():.3f}")

    # --- Full-sample IC ---
    print("\n" + "=" * 70)
    print("Full-sample Spearman IC vs y_path_ret_4h")
    print("-" * 70)
    results = {}
    for col in new_feats.columns:
        rho, p, n = spearman_ic(new_feats[col], y_path)
        print(f"  {col:30s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
        results[col] = {"ic": rho, "p": p, "n": n}

    # --- Bootstrap CI ---
    print("\n" + "=" * 70)
    print("Bootstrap 95% CI (1000 resamples)")
    print("-" * 70)
    for col in new_feats.columns:
        mean_ic, lo, hi = bootstrap_ic(new_feats[col], y_path)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {col:30s}  IC={mean_ic:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
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

    # --- Inter-feature correlation (check new features aren't redundant with each other) ---
    if len(new_feats.columns) > 1:
        print("\n" + "=" * 70)
        print("Inter-feature correlation (new features vs each other)")
        print("-" * 70)
        for i, c1 in enumerate(new_feats.columns):
            for c2 in new_feats.columns[i + 1:]:
                mask = new_feats[c1].notna() & new_feats[c2].notna()
                if mask.sum() > 100:
                    corr = new_feats[c1][mask].corr(new_feats[c2][mask])
                    flag = "HIGH" if abs(corr) > 0.7 else ""
                    print(f"  {c1:30s} vs {c2:30s}  corr={corr:+.3f}  {flag}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  {'Feature':30s}  {'IC':>8s}  {'CI':>22s}  {'Sig':>4s}  {'Consist':>8s}")
    for col in new_feats.columns:
        r = results[col]
        ic_str = f"{r['ic']:+.4f}" if not np.isnan(r.get('ic', np.nan)) else "  N/A"
        ci_str = f"[{r.get('ci_lo', np.nan):+.4f}, {r.get('ci_hi', np.nan):+.4f}]"
        sig_str = "YES" if r.get("significant", False) else " no"
        cons_str = f"{r.get('monthly_consistency', 0):.0%}"
        print(f"  {col:30s}  {ic_str}  {ci_str}  {sig_str}  {cons_str:>8s}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_divergence_ic.json"

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
