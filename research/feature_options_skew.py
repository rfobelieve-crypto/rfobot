"""
IC validation for Options-Derived / DVOL features.

Intuition: Implied volatility (DVOL) encodes market expectations about future
moves. When IV is elevated relative to realised vol, options are "overpriced"
and a mean-reversion in vol often follows. Interactions between DVOL regime
and funding/OI capture positioning under extreme vol expectations.

Features (from existing DVOL data):
  dvol_zscore_24h          — 24h rolling z-score of DVOL
  dvol_regime              — high/normal/low vol regime (-1/0/1)
  dvol_acceleration        — rate of change of 4h DVOL moving average
  dvol_vs_realized         — IV premium: DVOL / realized_vol (annualised)
  dvol_mean_reversion      — signed z-score clipped for extreme IV events
  dvol_funding_interaction — dvol_zscore × funding_zscore
  dvol_oi_interaction      — dvol_zscore × oi_change_zscore

Note on put/call ratio and 25-delta skew:
  Historical options chain data (strike-level IV, OI by strike) is not
  available via free Deribit API. The current Deribit integration only
  provides DVOL (aggregate IV index). For future live use, these features
  could be computed from:
    put_call_oi_ratio = sum(put_OI) / sum(call_OI)
      via GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
    skew_25d = IV(25d put) - IV(25d call)
      requires interpolating the vol surface, needs per-strike data
  These are snapshot-only and cannot be backtested with current data.
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

REQUIRED_DVOL_COLS = ["dvol_close"]
OPTIONAL_DVOL_COLS = [
    "dvol_open", "dvol_high", "dvol_low",
    "dvol_ma_24h", "dvol_change_4h", "dvol_change_24h",
    "dvol_zscore_72h", "dvol_rv_spread",
]


def _check_dvol_available(df: pd.DataFrame) -> bool:
    """Check that minimal DVOL data exists."""
    dvol_cols = [c for c in df.columns if "dvol" in c.lower()]
    bvol_cols = [c for c in df.columns if "bvol" in c.lower()]
    option_cols = [c for c in df.columns if "option" in c.lower()]

    print("\n=== Columns containing 'dvol' ===")
    for c in sorted(dvol_cols):
        nan_pct = df[c].isna().mean()
        print(f"  {c:30s}  nan={nan_pct:.1%}  mean={df[c].mean():.4f}")

    if bvol_cols:
        print("\n=== Columns containing 'bvol' ===")
        for c in sorted(bvol_cols):
            print(f"  {c}")

    if option_cols:
        print("\n=== Columns containing 'option' ===")
        for c in sorted(option_cols):
            print(f"  {c}")

    if not dvol_cols:
        print("\nERROR: No DVOL columns found in parquet. Cannot proceed.")
        print("Ensure Deribit DVOL is included in the backfill pipeline.")
        return False

    for req in REQUIRED_DVOL_COLS:
        if req not in df.columns:
            print(f"\nERROR: Required column '{req}' not found.")
            return False

    return True


def _zscore_rolling(s: pd.Series, window: int = 168) -> pd.Series:
    """Rolling z-score (trailing only, no look-ahead)."""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (s - mu) / sd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all options-derived / DVOL features. Returns DataFrame aligned to df.index."""
    out = pd.DataFrame(index=df.index)

    dvol = df["dvol_close"].astype(float)

    # --- dvol_zscore_24h: 24h rolling z-score ---
    out["dvol_zscore_24h"] = _zscore_rolling(dvol, window=24)

    # --- dvol_regime: high/normal/low classification ---
    # -1 = low vol (<30), 0 = normal, 1 = high vol (>60)
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[dvol > 60] = 1
    regime[dvol < 30] = -1
    out["dvol_regime"] = regime

    # --- dvol_acceleration: rate of change of 4h moving average ---
    dvol_ma4 = dvol.rolling(4, min_periods=2).mean()
    out["dvol_acceleration"] = dvol_ma4.diff()

    # --- dvol_vs_realized: IV premium ---
    # DVOL is annualised IV (%). Realised vol is per-bar std of log returns.
    # Annualise realised vol: realized_vol_20b * sqrt(8760) (hourly bars, ~8760h/yr)
    if "realized_vol_20b" in df.columns:
        rv_annual = df["realized_vol_20b"].astype(float) * np.sqrt(8760) * 100  # to percentage
        rv_annual = rv_annual.replace(0, np.nan)
        out["dvol_vs_realized"] = dvol / rv_annual
        logger.info("dvol_vs_realized built (DVOL / annualised RV)")
    else:
        # Fallback: compute from close prices
        if "close" in df.columns:
            log_ret = np.log(df["close"].astype(float)).diff()
            rv_annual = log_ret.rolling(20, min_periods=10).std() * np.sqrt(8760) * 100
            rv_annual = rv_annual.replace(0, np.nan)
            out["dvol_vs_realized"] = dvol / rv_annual
            logger.info("dvol_vs_realized built from close prices (fallback)")
        else:
            logger.warning("Skipping dvol_vs_realized: no realized_vol_20b or close column")

    # --- dvol_mean_reversion: signed extreme signal ---
    # When IV is extremely high (zscore > 2), vol tends to drop → bearish for vol
    # When IV is extremely low (zscore < -2), vol tends to spike → bullish for vol
    # Sign: positive = expect vol increase, negative = expect vol decrease
    z72 = df["dvol_zscore_72h"].astype(float) if "dvol_zscore_72h" in df.columns else _zscore_rolling(dvol, 72)
    out["dvol_mean_reversion"] = -z72.clip(-3, 3)

    # --- dvol_funding_interaction: high IV + extreme funding = big move ---
    dvol_z = out["dvol_zscore_24h"]
    funding_z_col = None
    for cand in ["funding_zscore", "cg_funding_close_zscore"]:
        if cand in df.columns:
            funding_z_col = cand
            break
    if funding_z_col is not None:
        funding_z = df[funding_z_col].astype(float)
        out["dvol_funding_interaction"] = dvol_z * funding_z
        logger.info("dvol_funding_interaction built using %s", funding_z_col)
    else:
        logger.warning("Skipping dvol_funding_interaction: no funding zscore column found")

    # --- dvol_oi_interaction: high IV + OI change = positioning for big move ---
    oi_z_col = None
    for cand in ["cg_oi_close_pctchg_4h", "cg_oi_delta"]:
        if cand in df.columns:
            oi_z_col = cand
            break
    if oi_z_col is not None:
        oi_change = df[oi_z_col].astype(float)
        oi_z = _zscore_rolling(oi_change, window=72)
        out["dvol_oi_interaction"] = dvol_z * oi_z
        logger.info("dvol_oi_interaction built using %s", oi_z_col)
    else:
        logger.warning("Skipping dvol_oi_interaction: no OI change column found")

    logger.info("Built %d options-derived features", len(out.columns))
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
    "dvol_zscore_72h", "dvol_rv_spread", "dvol_change_4h",
]


def main():
    print("=" * 70)
    print("Options-Derived / DVOL Features — IC Validation")
    print("=" * 70)

    # Load data
    if not CACHE.exists():
        print(f"ERROR: Cache not found at {CACHE}. Run shared_data.py first.")
        sys.exit(1)
    df = pd.read_parquet(CACHE)

    # Ensure datetime index
    if df.index.name != "dt":
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.set_index("dt").sort_index()
        else:
            print("WARNING: No 'dt' column or index found, using integer index.")
    df = df.sort_index()

    print(f"\nData: {len(df)} bars x {len(df.columns)} cols")
    print(f"Range: {df.index.min()} -> {df.index.max()}")

    # Check DVOL availability
    if not _check_dvol_available(df):
        print("\nExiting: no DVOL data available for backtesting.")
        print("For live use, Deribit DVOL must be added to the backfill pipeline.")
        sys.exit(0)

    # Build labels
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    target = labels["return_4h"]
    y_path = labels.get("path_return_4h", target)  # prefer TWAP path return

    # Build features
    new_feats = build_features(df)
    print(f"\nNew features: {list(new_feats.columns)}")
    for c in new_feats.columns:
        s = new_feats[c]
        print(f"  {c:35s}  nan={s.isna().mean():.1%}  "
              f"mean={s.mean():+.4f}  std={s.std():.4f}")

    # --- Note about unavailable features ---
    print("\n" + "=" * 70)
    print("NOTE: Features requiring historical options chain data")
    print("-" * 70)
    print("  The following features cannot be backtested with current data")
    print("  (Deribit public API only provides current snapshots):\n")
    print("  put_call_oi_ratio = sum(put_OI) / sum(call_OI)")
    print("    Source: GET /api/v2/public/get_book_summary_by_currency"
          "?currency=BTC&kind=option")
    print("    Aggregate open_interest by option type (put vs call)\n")
    print("  skew_25d = IV(25-delta put) - IV(25-delta call)")
    print("    Requires per-strike IV data and delta interpolation")
    print("    Not feasible without historical vol surface data\n")
    print("  These are noted for future live feature integration.")

    # --- Full-sample IC ---
    print("\n" + "=" * 70)
    print("Full-sample Spearman IC vs y_path_ret_4h")
    print("-" * 70)
    results = {}
    for col in new_feats.columns:
        rho, p, n = spearman_ic(new_feats[col], y_path)
        print(f"  {col:35s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
        results[col] = {"ic": rho, "p": p, "n": n}

    # --- Bootstrap CI ---
    print("\n" + "=" * 70)
    print("Bootstrap 95% CI (1000 resamples)")
    print("-" * 70)
    for col in new_feats.columns:
        mean_ic, lo, hi = bootstrap_ic(new_feats[col], y_path)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {col:35s}  IC={mean_ic:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
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

    # --- Inter-feature correlation ---
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
                    print(f"  {c1:35s} vs {c2:35s}  corr={corr:+.3f}  {flag}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  {'Feature':35s}  {'IC':>8s}  {'CI':>22s}  {'Sig':>4s}  {'Consist':>8s}")
    for col in new_feats.columns:
        r = results[col]
        ic_str = f"{r['ic']:+.4f}" if not np.isnan(r.get('ic', np.nan)) else "  N/A"
        ci_str = f"[{r.get('ci_lo', np.nan):+.4f}, {r.get('ci_hi', np.nan):+.4f}]"
        sig_str = "YES" if r.get("significant", False) else " no"
        cons_str = f"{r.get('monthly_consistency', 0):.0%}"
        print(f"  {col:35s}  {ic_str}  {ci_str}  {sig_str}  {cons_str:>8s}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_options_skew.json"

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
