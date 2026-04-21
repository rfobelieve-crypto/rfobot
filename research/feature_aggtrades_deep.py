"""
Deep aggTrades / trade-flow feature extraction for BTC 4h direction prediction.

The system already collects aggTrades but only uses basic metrics.  This script
mines more sophisticated patterns from volume, trade_count, taker flow, and
delta columns already present in the parquet cache.

Features (~12):
  Trade Intensity
    trade_intensity_24h_zscore  — trade_count z-scored over 24h rolling window
    volume_per_trade            — volume / trade_count (institutional proxy)
    volume_per_trade_zscore     — z-score of above over 168h

  Taker Imbalance Dynamics
    taker_imbalance             — net buy fraction of total volume
    taker_imbalance_mom_4h      — 4h change in taker_imbalance
    taker_cumulative_8h         — 8h rolling sum of taker_imbalance
    taker_reversal_signal       — extreme cumulative + momentum flip

  Volume Profile
    volume_high_low_ratio       — close>open proxy for directional volume
    volume_climax               — big volume + big move flag
    volume_dry_up               — low volume flag (breakout setup)

  Flow Persistence
    delta_persistence_8h        — sign consistency of hourly delta over 8h
    delta_acceleration_zscore   — delta[t] - delta[t-4], z-scored
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

# Top existing features for redundancy check
TOP_EXISTING = [
    "cg_oi_close_zscore", "cg_funding_close_zscore", "cg_bfx_margin_ratio",
    "vol_acceleration", "cg_liq_imbalance_zscore", "oi_price_divergence",
    "impact_asymmetry", "realized_vol_20b", "taker_delta_ratio",
    "trade_intensity_zscore", "cg_taker_delta_zscore", "quote_vol_zscore",
]


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def print_relevant_columns(df: pd.DataFrame) -> None:
    """Print all columns matching aggTrade-related keywords."""
    keywords = ["volume", "trade", "taker", "delta", "flow", "cvd"]
    print("\n" + "=" * 70)
    print("Available columns matching aggTrade keywords")
    print("=" * 70)
    for kw in keywords:
        matches = sorted(c for c in df.columns if kw.lower() in c.lower())
        if matches:
            print(f"\n  --- {kw} ({len(matches)}) ---")
            for m in matches:
                print(f"    {m}")


# ---------------------------------------------------------------------------
# Helper: trailing z-score
# ---------------------------------------------------------------------------

def _trailing_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing-only z-score (no look-ahead)."""
    if min_periods is None:
        min_periods = window // 2
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sd


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggTrades-deep features. Skips features whose source cols are missing."""
    out = pd.DataFrame(index=df.index)
    built = []
    skipped = []

    # ── Trade Intensity ──────────────────────────────────────────────────
    if "trade_count" in df.columns:
        tc = df["trade_count"].astype(float)

        # trade_intensity_24h_zscore: trade_count z-scored over 24h
        # (Note: the parquet already has trade_intensity_zscore but with
        #  unknown window; we build our own with explicit 24h window.)
        out["trade_intensity_24h_zscore"] = _trailing_zscore(tc, 24)
        built.append("trade_intensity_24h_zscore")

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            # volume_per_trade: average trade size (institutional proxy)
            vpt = vol / tc.replace(0, np.nan)
            out["volume_per_trade"] = vpt
            out["volume_per_trade_zscore"] = _trailing_zscore(vpt, 168)
            built.extend(["volume_per_trade", "volume_per_trade_zscore"])
    else:
        skipped.extend(["trade_intensity_24h_zscore", "volume_per_trade",
                         "volume_per_trade_zscore"])

    # ── Taker Imbalance Dynamics ─────────────────────────────────────────
    has_taker = "taker_buy_vol" in df.columns and "volume" in df.columns
    has_cg_taker = "cg_taker_buy" in df.columns and "cg_taker_sell" in df.columns

    if has_taker:
        vol = df["volume"].astype(float)
        tbv = df["taker_buy_vol"].astype(float)
        tsv = vol - tbv  # taker sell vol

        # taker_imbalance: net buy fraction
        imb = (tbv - tsv) / vol.replace(0, np.nan)
        out["taker_imbalance"] = imb
        built.append("taker_imbalance")

        # taker_imbalance_mom_4h: 4h change in imbalance (acceleration)
        out["taker_imbalance_mom_4h"] = imb.diff(4)
        built.append("taker_imbalance_mom_4h")

        # taker_cumulative_8h: 8h rolling sum of imbalance
        cum_8h = imb.rolling(8, min_periods=4).sum()
        out["taker_cumulative_8h"] = cum_8h
        built.append("taker_cumulative_8h")

        # taker_reversal_signal: extreme cumulative + momentum sign flip
        cum_z = _trailing_zscore(cum_8h, 168)
        mom_4h = imb.diff(4)
        # extreme = |z| > 1.5 AND momentum flips vs cumulative sign
        extreme = cum_z.abs() > 1.5
        sign_flip = (np.sign(cum_8h) * np.sign(mom_4h)) < 0
        out["taker_reversal_signal"] = (extreme & sign_flip).astype(float)
        built.append("taker_reversal_signal")

    elif has_cg_taker:
        # Fallback: use Coinglass taker data
        cg_buy = df["cg_taker_buy"].astype(float)
        cg_sell = df["cg_taker_sell"].astype(float)
        cg_total = (cg_buy + cg_sell).replace(0, np.nan)

        imb = (cg_buy - cg_sell) / cg_total
        out["taker_imbalance"] = imb
        out["taker_imbalance_mom_4h"] = imb.diff(4)

        cum_8h = imb.rolling(8, min_periods=4).sum()
        out["taker_cumulative_8h"] = cum_8h

        cum_z = _trailing_zscore(cum_8h, 168)
        mom_4h = imb.diff(4)
        extreme = cum_z.abs() > 1.5
        sign_flip = (np.sign(cum_8h) * np.sign(mom_4h)) < 0
        out["taker_reversal_signal"] = (extreme & sign_flip).astype(float)
        built.extend(["taker_imbalance", "taker_imbalance_mom_4h",
                       "taker_cumulative_8h", "taker_reversal_signal"])
    else:
        skipped.extend(["taker_imbalance", "taker_imbalance_mom_4h",
                         "taker_cumulative_8h", "taker_reversal_signal"])

    # ── Volume Profile ───────────────────────────────────────────────────
    has_ohlcv = all(c in df.columns for c in ["open", "close", "high", "low", "volume"])

    if has_ohlcv:
        o = df["open"].astype(float)
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        lo = df["low"].astype(float)
        vol = df["volume"].astype(float)

        # volume_high_low_ratio: proxy for directional volume attribution
        # If close > open (bullish bar), more volume attributed to buying
        bar_range = (h - lo).replace(0, np.nan)
        close_position = (c - lo) / bar_range  # 0=closed at low, 1=closed at high
        out["volume_high_low_ratio"] = close_position
        built.append("volume_high_low_ratio")

        # volume_climax: volume_zscore > 2 AND |return| > realized_vol
        vol_z = _trailing_zscore(vol, 24)
        ret = (c / o) - 1
        if "realized_vol_20b" in df.columns:
            rv = df["realized_vol_20b"].astype(float)
        else:
            rv = ret.abs().rolling(20, min_periods=10).mean()
        out["volume_climax"] = ((vol_z > 2) & (ret.abs() > rv)).astype(float)
        built.append("volume_climax")

        # volume_dry_up: volume_zscore < -1.5
        out["volume_dry_up"] = (vol_z < -1.5).astype(float)
        built.append("volume_dry_up")
    else:
        skipped.extend(["volume_high_low_ratio", "volume_climax", "volume_dry_up"])

    # ── Flow Persistence ─────────────────────────────────────────────────
    # Use taker_delta_ratio (Binance) or cg_taker_delta (Coinglass) as delta
    delta_col = None
    if "taker_delta_ratio" in df.columns:
        delta_col = "taker_delta_ratio"
    elif "cg_taker_delta" in df.columns:
        delta_col = "cg_taker_delta"

    if delta_col is not None:
        delta = df[delta_col].astype(float)

        # delta_persistence_8h: fraction of last 8 bars with same sign delta
        delta_sign = np.sign(delta)
        # Rolling: count how many of last 8 bars share the same sign as the
        # majority sign in that window.  Simpler: fraction with sign == sign[t].
        def _sign_consistency(window: int) -> pd.Series:
            """For each bar, fraction of last `window` bars with same sign as current."""
            pos_count = (delta_sign == 1).rolling(window, min_periods=window).sum()
            neg_count = (delta_sign == -1).rolling(window, min_periods=window).sum()
            # Persistence = max(pos_frac, neg_frac) — doesn't matter which direction
            return pd.concat([pos_count, neg_count], axis=1).max(axis=1) / window

        out["delta_persistence_8h"] = _sign_consistency(8)
        built.append("delta_persistence_8h")

        # delta_acceleration: delta[t] - delta[t-4], z-scored
        delta_accel = delta.diff(4)
        out["delta_acceleration_zscore"] = _trailing_zscore(delta_accel, 168)
        built.append("delta_acceleration_zscore")
    else:
        skipped.extend(["delta_persistence_8h", "delta_acceleration_zscore"])

    logger.info("Built %d features, skipped %d", len(built), len(skipped))
    if skipped:
        logger.info("Skipped (missing source cols): %s", skipped)
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
# JSON serialization helper
# ---------------------------------------------------------------------------

def _convert(obj):
    """Convert numpy types for JSON serialization."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AggTrades Deep Feature Extraction — IC Validation")
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

    # Print relevant columns for reference
    print_relevant_columns(df)

    # Build labels
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    y_path = labels.get("path_return_4h", labels["return_4h"])

    # Build features
    new_feats = build_features(df)
    print(f"\nNew features ({len(new_feats.columns)}):")
    for c in new_feats.columns:
        s = new_feats[c]
        print(f"  {c:32s}  nan={s.isna().mean():.1%}  "
              f"mean={s.mean():+.4f}  std={s.std():.4f}")

    if new_feats.empty:
        print("\nERROR: No features could be built. Check source columns.")
        sys.exit(1)

    # --- Full-sample IC ---
    print("\n" + "=" * 70)
    print("Full-sample Spearman IC vs y_path_ret_4h")
    print("-" * 70)
    results = {}
    for col in new_feats.columns:
        rho, p, n = spearman_ic(new_feats[col], y_path)
        print(f"  {col:32s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
        results[col] = {"ic": rho, "p": p, "n": n}

    # --- Bootstrap CI ---
    print("\n" + "=" * 70)
    print("Bootstrap 95% CI (1000 resamples)")
    print("-" * 70)
    for col in new_feats.columns:
        mean_ic, lo, hi = bootstrap_ic(new_feats[col], y_path)
        sig = "SIG" if not np.isnan(lo) and (lo > 0 or hi < 0) else "n.s."
        print(f"  {col:32s}  IC={mean_ic:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")
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
            results[col]["monthly_consistency"] = None
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

    # --- Inter-feature correlation ---
    print("\n" + "=" * 70)
    print("Inter-feature correlation (new features among themselves)")
    print("-" * 70)
    if len(new_feats.columns) > 1:
        corr_matrix = new_feats.corr(method="spearman")
        for i, c1 in enumerate(new_feats.columns):
            for c2 in new_feats.columns[i + 1:]:
                r = corr_matrix.loc[c1, c2]
                flag = " <-- HIGH" if abs(r) > 0.8 else ""
                print(f"  {c1:32s} x {c2:32s}  corr={r:+.3f}{flag}")

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
        for ref, cv in sorted(corrs.items(), key=lambda x: -abs(x[1])):
            print(f"    vs {ref:35s}  corr={cv:+.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  {'Feature':32s}  {'IC':>8s}  {'CI':>22s}  {'Sig':>4s}  {'Consist':>8s}")
    for col in new_feats.columns:
        r = results[col]
        ic_val = r.get("ic", np.nan)
        ic_str = f"{ic_val:+.4f}" if not np.isnan(ic_val) else "  N/A"
        lo = r.get("ci_lo", np.nan)
        hi = r.get("ci_hi", np.nan)
        if np.isnan(lo) or np.isnan(hi):
            ci_str = "         N/A          "
        else:
            ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        sig_str = "YES" if r.get("significant", False) else " no"
        cons = r.get("monthly_consistency")
        cons_str = f"{cons:.0%}" if cons is not None else "N/A"
        print(f"  {col:32s}  {ic_str}  {ci_str}  {sig_str}  {cons_str:>8s}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_aggtrades_deep.json"

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
