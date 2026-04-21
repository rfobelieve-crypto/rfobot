"""
IC validation for Cross-Market features (S&P500, DXY, Gold, US10Y).

Intuition: BTC does not trade in a vacuum. Macro risk-on/off shifts, dollar
strength, and bond yields create cross-asset currents that precede or amplify
BTC moves. Daily macro signals forward-filled to hourly capture slow-moving
regime context that on-chain/derivative features miss.

Features per market (SPX, DXY, GOLD, US10Y):
  {mkt}_return_1d          - 1-day return
  {mkt}_return_5d          - 5-day return
  {mkt}_zscore_20d         - 20-day rolling z-score of price
  {mkt}_momentum_diff      - 5d return minus 20d return (momentum acceleration)

Cross features:
  spx_btc_corr_20d         - 20-day rolling correlation SPX vs BTC daily returns
  dxy_btc_divergence       - DXY 1d ret * BTC 1d ret (positive = same direction)
  risk_on_composite        - SPX_ret + GOLD_ret - DXY_ret (risk appetite)
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

CACHE = Path(__file__).resolve().parent / "dual_model" / ".cache" / "features_all.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results"

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
TICKERS = {
    "SPX":   "^GSPC",       # S&P 500
    "DXY":   "DX-Y.NYB",    # US Dollar Index
    "GOLD":  "GC=F",        # Gold futures
    "US10Y": "^TNX",        # 10Y Treasury yield
}


# ---------------------------------------------------------------------------
# Cross-market data download + alignment
# ---------------------------------------------------------------------------

def download_cross_market(start: str, end: str, btc_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Download daily OHLC for macro tickers, forward-fill to hourly, align to BTC index."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    all_series = {}

    for name, ticker in TICKERS.items():
        logger.info("Downloading %s (%s) ...", name, ticker)
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data is None or len(data) == 0:
                logger.warning("No data returned for %s — skipping", name)
                continue
            # yfinance may return MultiIndex columns when downloading single ticker
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Close"].dropna()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            all_series[name] = close
            logger.info("  %s: %d daily bars (%s -> %s)",
                        name, len(close), close.index.min().date(), close.index.max().date())
        except Exception as e:
            logger.warning("Failed to download %s: %s — skipping", name, e)

    if not all_series:
        print("ERROR: No cross-market data downloaded. Check network / yfinance.")
        sys.exit(1)

    # Combine into daily DataFrame
    daily = pd.DataFrame(all_series)
    daily = daily.sort_index()

    # Align timezone: daily from yfinance is tz-naive, BTC index may be tz-aware
    if daily.index.tz is None and btc_index.tz is not None:
        daily.index = daily.index.tz_localize("UTC")
    elif daily.index.tz is not None and btc_index.tz is None:
        daily.index = daily.index.tz_localize(None)

    # Forward-fill to hourly: reindex to BTC hourly index, then ffill
    hourly = daily.reindex(btc_index, method="ffill")
    # Also backfill the first few hours before first daily bar
    hourly = hourly.fillna(method="bfill", limit=24)

    logger.info("Hourly aligned: %d rows, columns: %s, NaN%%: %s",
                len(hourly), list(hourly.columns),
                {c: f"{hourly[c].isna().mean():.1%}" for c in hourly.columns})
    return hourly


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cross_hourly: pd.DataFrame) -> pd.DataFrame:
    """Build all cross-market features. Returns DataFrame aligned to df.index."""
    out = pd.DataFrame(index=df.index)

    # BTC daily return (for correlation features)
    btc_close = df["close"] if "close" in df.columns else None
    if btc_close is not None:
        btc_ret_1d = btc_close.pct_change(24)  # 24 hourly bars = 1 day
    else:
        logger.warning("No 'close' column in BTC data — skipping BTC-dependent cross features")
        btc_ret_1d = None

    available_mkts = [m for m in TICKERS if m in cross_hourly.columns]

    for mkt in available_mkts:
        price = cross_hourly[mkt]

        # {mkt}_return_1d: 1-day return (24h)
        out[f"{mkt}_return_1d"] = price.pct_change(24)

        # {mkt}_return_5d: 5-day return (120h)
        out[f"{mkt}_return_5d"] = price.pct_change(120)

        # {mkt}_zscore_20d: 20-day rolling z-score of price (480h)
        mu = price.rolling(480, min_periods=240).mean()
        sd = price.rolling(480, min_periods=240).std().replace(0, np.nan)
        out[f"{mkt}_zscore_20d"] = (price - mu) / sd

        # {mkt}_momentum_diff: 5d return - 20d return
        ret_5d = price.pct_change(120)
        ret_20d = price.pct_change(480)
        out[f"{mkt}_momentum_diff"] = ret_5d - ret_20d

    # --- Cross features ---

    # spx_btc_corr_20d: 20-day rolling correlation of SPX vs BTC daily returns
    if "SPX" in cross_hourly.columns and btc_ret_1d is not None:
        spx_ret_1d = cross_hourly["SPX"].pct_change(24)
        out["spx_btc_corr_20d"] = spx_ret_1d.rolling(480, min_periods=240).corr(btc_ret_1d)
    else:
        logger.warning("Skipping spx_btc_corr_20d: missing SPX or BTC close")

    # dxy_btc_divergence: DXY 1d return * BTC 1d return
    # positive = same direction (unusual, since BTC and DXY are normally inverse)
    if "DXY" in cross_hourly.columns and btc_ret_1d is not None:
        dxy_ret_1d = cross_hourly["DXY"].pct_change(24)
        out["dxy_btc_divergence"] = dxy_ret_1d * btc_ret_1d
    else:
        logger.warning("Skipping dxy_btc_divergence: missing DXY or BTC close")

    # risk_on_composite: SPX_ret + GOLD_ret - DXY_ret
    has_all = all(m in cross_hourly.columns for m in ["SPX", "GOLD", "DXY"])
    if has_all:
        spx_r = cross_hourly["SPX"].pct_change(24)
        gold_r = cross_hourly["GOLD"].pct_change(24)
        dxy_r = cross_hourly["DXY"].pct_change(24)
        out["risk_on_composite"] = spx_r + gold_r - dxy_r
    else:
        logger.warning("Skipping risk_on_composite: missing one of SPX/GOLD/DXY")

    logger.info("Built %d cross-market features", len(out.columns))
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
    print("Cross-Market Features — IC Validation")
    print("=" * 70)

    # Load BTC data
    if not CACHE.exists():
        print(f"ERROR: Cache not found at {CACHE}. Run shared_data.py first.")
        sys.exit(1)
    df = pd.read_parquet(CACHE)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()
    print(f"\nBTC data: {len(df)} bars x {len(df.columns)} cols")
    print(f"Range: {df.index.min()} -> {df.index.max()}")

    # Print available columns for reference
    col_groups = {}
    for c in sorted(df.columns):
        prefix = c.split("_")[0] if "_" in c else c
        col_groups.setdefault(prefix, []).append(c)
    print(f"\nAvailable columns ({len(df.columns)} total):")
    for prefix in sorted(col_groups):
        print(f"  {prefix}_*: {len(col_groups[prefix])} cols")

    # Build target
    try:
        from research.dual_model.build_direction_labels import build_direction_labels
        labels = build_direction_labels(df, horizon_bars=4, k=0.5)
        y_path = labels.get("path_return_4h", labels["return_4h"])
        print(f"\nTarget (build_direction_labels): {y_path.notna().sum()} valid values")
    except Exception as e:
        logger.warning("build_direction_labels failed (%s), computing manually", e)
        y_path = df["close"].shift(-4).rolling(4).mean() / df["close"] - 1
        y_path.name = "y_path_ret_4h"
        print(f"\nTarget (manual TWAP 4h): {y_path.notna().sum()} valid values")

    # Download cross-market data
    start_date = (df.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cross_hourly = download_cross_market(start_date, end_date, df.index)

    # Build features
    new_feats = build_features(df, cross_hourly)
    print(f"\nCross-market features ({len(new_feats.columns)}):")
    for c in new_feats.columns:
        s = new_feats[c]
        valid = s.notna().sum()
        print(f"  {c:30s}  valid={valid:5d} ({valid/len(s):.0%})  "
              f"mean={s.mean():+.4f}  std={s.std():.4f}")

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
            results[col]["monthly_consistency"] = 0.0
            results[col]["monthly_ics"] = []
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
        feat_cols = list(new_feats.columns)
        for i, c1 in enumerate(feat_cols):
            for c2 in feat_cols[i + 1:]:
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
        ic_val = r.get("ic", np.nan)
        ic_str = f"{ic_val:+.4f}" if not (isinstance(ic_val, float) and np.isnan(ic_val)) else "  N/A"
        lo = r.get("ci_lo", np.nan)
        hi = r.get("ci_hi", np.nan)
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]" if not (isinstance(lo, float) and np.isnan(lo)) else "         N/A"
        sig_str = "YES" if r.get("significant", False) else " no"
        cons_str = f"{r.get('monthly_consistency', 0):.0%}"
        print(f"  {col:30s}  {ic_str}  {ci_str}  {sig_str}  {cons_str:>8s}")

    # Count significant features
    n_sig = sum(1 for r in results.values() if r.get("significant", False))
    print(f"\n  Significant features: {n_sig}/{len(results)}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_cross_market.json"

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
        serializable[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, (dict, list)):
                serializable[k][kk] = vv
            else:
                serializable[k][kk] = _convert(vv)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=_convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
