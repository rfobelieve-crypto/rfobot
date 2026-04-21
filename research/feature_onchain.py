"""
IC validation for synthetic on-chain proxy features + Fear & Greed sentiment.

NOTE: Features #1-#6 are PROXIES derived from exchange data (OI, volume,
funding, price), NOT real on-chain data.  They approximate on-chain concepts
(exchange flows, whale activity, SOPR, MVRV) using perpetual-swap observables.
Treat IC results accordingly -- these capture derivative-market echoes of
on-chain dynamics, not the dynamics themselves.

Feature #7 (Fear & Greed) is actual sentiment data from alternative.me,
already present in the parquet cache.

All features are trailing-only (no look-ahead).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.build_direction_labels import build_direction_labels

CACHE = Path(__file__).resolve().parent / "dual_model" / ".cache" / "features_all.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Column candidates (handle naming variations across parquet versions)
# ---------------------------------------------------------------------------
OI_CANDIDATES = ["cg_oi_close", "cg_oi_agg_close", "cg_oi_total"]
FUNDING_CANDIDATES = ["cg_funding_close", "funding_rate"]
VOLUME_CANDIDATES = ["volume"]
TRADE_COUNT_CANDIDATES = ["trade_count"]
FNG_CANDIDATES = ["fear_greed_value", "fng_value"]


def _find_col(df: pd.DataFrame, candidates: list[str], label: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            print(f"  Using '{c}' for {label}")
            return c
    print(f"  WARNING: No column found for {label}. Tried: {candidates}")
    return None


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ~10 synthetic on-chain proxy features."""
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    vol_col = _find_col(df, VOLUME_CANDIDATES, "volume")
    oi_col = _find_col(df, OI_CANDIDATES, "OI")
    fund_col = _find_col(df, FUNDING_CANDIDATES, "funding")
    tc_col = _find_col(df, TRADE_COUNT_CANDIDATES, "trade_count")
    fng_col = _find_col(df, FNG_CANDIDATES, "fear_greed")

    volume = df[vol_col].astype(float) if vol_col else None
    oi = df[oi_col].astype(float) if oi_col else None
    funding = df[fund_col].astype(float) if fund_col else None
    tc = df[tc_col].astype(float) if tc_col else None

    # ---- #1  Exchange Flow Proxy (OI + volume) ----
    if oi is not None and volume is not None:
        oi_change = oi.pct_change()
        vol_mean = volume.rolling(24, min_periods=12).mean()
        vol_std = volume.rolling(24, min_periods=12).std()
        vol_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)

        # sign(oi_change) * volume_zscore
        out["exchange_flow_proxy"] = np.sign(oi_change) * vol_zscore

        # 8h rolling sum
        out["exchange_flow_8h"] = out["exchange_flow_proxy"].rolling(8, min_periods=4).sum()

    # ---- #2  Whale Activity Proxy (volume per trade) ----
    if volume is not None and tc is not None:
        vol_per_trade = volume / tc.replace(0, np.nan)
        vpt_mean = vol_per_trade.rolling(24, min_periods=12).mean()
        vpt_std = vol_per_trade.rolling(24, min_periods=12).std()
        whale_proxy = (vol_per_trade - vpt_mean) / vpt_std.replace(0, np.nan)
        out["whale_proxy"] = whale_proxy

        # Whale accumulation: high whale_proxy AND price is flat
        ret_4h = close.pct_change(4).abs()
        rvol = close.pct_change().rolling(24, min_periods=12).std()
        # ratio of |return| to realized vol -- low = price is flat
        price_flatness = 1.0 - (ret_4h / rvol.replace(0, np.nan)).clip(0, 1)
        whale_accum = (whale_proxy * price_flatness).clip(0, 5)
        out["whale_accumulation"] = whale_accum

    # ---- #3  SOPR Proxy (price below SMA + funding negative = capitulation) ----
    if funding is not None:
        sma_20d = close.rolling(20 * 24, min_periods=120).mean()
        price_vs_sma = close / sma_20d - 1  # negative when below SMA

        # Only when close < SMA: deviation * sign(-funding)
        # Capitulation = price below SMA AND funding negative (underwater positions closing)
        below_sma = close < sma_20d
        sopr_proxy = price_vs_sma * np.sign(-funding)
        out["sopr_proxy"] = np.where(below_sma, sopr_proxy, 0.0)
        out["sopr_proxy"] = pd.Series(out["sopr_proxy"].values, index=df.index, dtype=float)

    # ---- #4  OI / Volume Ratio (conviction vs hot money) ----
    if oi is not None and volume is not None:
        oi_vol_ratio = oi / (volume * close).replace(0, np.nan)  # OI / notional volume
        out["oi_to_volume_ratio"] = oi_vol_ratio

        oiv_mean = oi_vol_ratio.rolling(48, min_periods=24).mean()
        oiv_std = oi_vol_ratio.rolling(48, min_periods=24).std()
        out["oi_to_volume_zscore"] = (oi_vol_ratio - oiv_mean) / oiv_std.replace(0, np.nan)

    # ---- #5  OI Buildup Signal (sustained OI increase + moderate funding) ----
    if oi is not None and funding is not None:
        oi_change_sign = np.sign(oi.pct_change())
        # Rolling count of positive OI changes in last 8 bars
        oi_up_count = oi_change_sign.clip(lower=0).rolling(8, min_periods=8).sum()

        fund_mean = funding.rolling(24, min_periods=12).mean()
        fund_std = funding.rolling(24, min_periods=12).std()
        fund_zscore = (funding - fund_mean) / fund_std.replace(0, np.nan)

        # OI increasing 6+ out of 8 bars AND funding not extreme
        buildup = ((oi_up_count >= 6) & (fund_zscore.abs() < 1.5)).astype(float)
        out["oi_buildup_signal"] = buildup

    # ---- #6  MVRV Proxy (close / 20d VWAP) ----
    if volume is not None:
        window = 20 * 24  # 20 days in hourly bars
        close_x_vol = close * volume
        rolling_vwap = close_x_vol.rolling(window, min_periods=120).sum() / \
                       volume.rolling(window, min_periods=120).sum().replace(0, np.nan)
        out["mvrv_proxy"] = close / rolling_vwap

    # ---- #7  Fear & Greed (actual sentiment data) ----
    if fng_col is not None:
        fng = df[fng_col].astype(float)
        # Raw value (already forward-filled to hourly in the parquet)
        out["fng_value"] = fng

        # 30-day rolling z-score (30d = 720 hourly bars)
        fng_mean = fng.rolling(720, min_periods=168).mean()
        fng_std = fng.rolling(720, min_periods=168).std()
        out["fng_zscore"] = (fng - fng_mean) / fng_std.replace(0, np.nan)

        # Extreme fear (<20) or extreme greed (>80)
        out["fng_extreme"] = ((fng < 20) | (fng > 80)).astype(float)

    return out


# ---------------------------------------------------------------------------
# Fetch Fear & Greed from API (if not already in parquet)
# ---------------------------------------------------------------------------

def fetch_fng_from_api() -> pd.DataFrame | None:
    """Try to fetch Fear & Greed Index from alternative.me (free, daily)."""
    try:
        import requests
        url = "https://api.alternative.me/fng/?limit=365&format=json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            print("  FNG API returned empty data")
            return None

        rows = []
        for item in data:
            ts = pd.Timestamp(int(item["timestamp"]), unit="s", tz="UTC")
            rows.append({"date": ts.normalize(), "fng_api": int(item["value"])})

        fng_df = pd.DataFrame(rows).set_index("date").sort_index()
        fng_df = fng_df[~fng_df.index.duplicated(keep="first")]
        print(f"  Fetched {len(fng_df)} days of FNG data from API "
              f"({fng_df.index.min().date()} -> {fng_df.index.max().date()})")
        return fng_df
    except Exception as e:
        print(f"  FNG API fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Validation helpers (same pattern as other feature research scripts)
# ---------------------------------------------------------------------------

def ic_report(feat: pd.Series, target: pd.Series, name: str, target_name: str):
    mask = feat.notna() & target.notna()
    if mask.sum() < 100:
        print(f"  {name} vs {target_name}: insufficient samples ({mask.sum()})")
        return None
    rho, p = spearmanr(feat[mask], target[mask])
    n = int(mask.sum())
    print(f"  {name:34s} vs {target_name:14s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
    return rho


def bootstrap_ic(feat: pd.Series, target: pd.Series, n_boot: int = 1000,
                 seed: int = 42) -> dict:
    mask = feat.notna() & target.notna()
    f = feat[mask].values
    t = target[mask].values
    n = len(f)
    if n < 100:
        return {"mean": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    rng = np.random.RandomState(seed)
    ics = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rho, _ = spearmanr(f[idx], t[idx])
        ics.append(rho)
    ics = np.array(ics)
    return {
        "mean": float(np.mean(ics)),
        "std": float(np.std(ics)),
        "ci_lo": float(np.percentile(ics, 2.5)),
        "ci_hi": float(np.percentile(ics, 97.5)),
        "n": n,
    }


def monthly_ic(feat: pd.Series, target: pd.Series) -> dict:
    mask = feat.notna() & target.notna()
    combined = pd.DataFrame({"f": feat[mask], "t": target[mask]})
    combined["month"] = combined.index.to_period("M")
    result = {}
    for m, grp in combined.groupby("month"):
        if len(grp) < 30:
            continue
        rho, _ = spearmanr(grp["f"], grp["t"])
        result[str(m)] = round(float(rho), 4)
    return result


def rolling_ic(feat: pd.Series, target: pd.Series, window: int = 168) -> pd.Series:
    df_tmp = pd.concat([feat, target], axis=1).dropna()
    if len(df_tmp) < window:
        return pd.Series(dtype=float)
    out = []
    idx = []
    for i in range(window, len(df_tmp) + 1):
        chunk = df_tmp.iloc[i - window:i]
        rho, _ = spearmanr(chunk.iloc[:, 0], chunk.iloc[:, 1])
        out.append(rho)
        idx.append(df_tmp.index[i - 1])
    return pd.Series(out, index=idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading features from {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  shape={df.shape}  range={df.index.min()} -> {df.index.max()}")

    # Print available columns for reference
    print(f"\nAvailable columns ({len(df.columns)}):")
    for c in sorted(df.columns):
        print(f"  {c}")

    # Check required columns
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        return

    # ----- Optionally fetch FNG from API if not in parquet -----
    fng_col_existing = _find_col(df, FNG_CANDIDATES, "fear_greed (check)")
    if fng_col_existing is None:
        print("\n--- Fear & Greed not in parquet, trying API ---")
        fng_api = fetch_fng_from_api()
        if fng_api is not None:
            # Forward-fill daily FNG to hourly index
            df["fear_greed_value"] = np.nan
            df_dates = df.index.normalize()
            for dt, row in fng_api.iterrows():
                mask = df_dates == dt
                df.loc[mask, "fear_greed_value"] = row["fng_api"]
            df["fear_greed_value"] = df["fear_greed_value"].ffill()
            filled = df["fear_greed_value"].notna().sum()
            print(f"  Merged FNG into dataframe: {filled}/{len(df)} rows filled")
    else:
        print(f"\n--- Fear & Greed already in parquet as '{fng_col_existing}' ---")

    # ----- Build labels -----
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_return_4h"] = labels["return_4h"]

    # Build TWAP path return target
    close_arr = df["close"].values.astype(float)
    n = len(close_arr)
    path_ret = np.full(n, np.nan)
    for i in range(n - 4):
        path_ret[i] = (sum(close_arr[i + k] for k in range(1, 5)) / 4.0) / close_arr[i] - 1
    df["y_path_ret_4h"] = path_ret

    # ----- Build features -----
    print("\n" + "=" * 70)
    print("NOTE: Features #1-#6 are PROXIES from exchange data, NOT real on-chain.")
    print("      Feature #7 (FNG) is actual sentiment data from alternative.me.")
    print("=" * 70)

    new_feats = build_features(df)
    print(f"\nNew features built: {list(new_feats.columns)}")
    for c in new_feats.columns:
        non_zero = (new_feats[c] != 0).sum()
        nan_pct = new_feats[c].isna().mean()
        print(f"  {c:34s}  nan={nan_pct:.2%}  "
              f"non_zero={non_zero} ({non_zero / len(new_feats):.1%})  "
              f"std={new_feats[c].std():.6f}")

    results = {"_meta": {
        "note": "Features 1-6 are PROXIES from exchange data, not real on-chain. "
                "FNG features are actual sentiment data.",
        "data_range": f"{df.index.min()} -> {df.index.max()}",
        "n_rows": len(df),
    }}
    target_col = "y_path_ret_4h"

    # ---- Inter-feature correlation ----
    print("\n=== Inter-feature correlation matrix ===")
    feat_names = list(new_feats.columns)
    if len(feat_names) > 1:
        corr_matrix = new_feats[feat_names].corr()
        for i in range(len(feat_names)):
            for j in range(i + 1, len(feat_names)):
                c = corr_matrix.iloc[i, j]
                flag = " *** HIGH" if abs(c) > 0.8 else ""
                print(f"  {feat_names[i]:30s} x {feat_names[j]:30s}  corr={c:+.3f}{flag}")

    # ---- Full-sample IC ----
    print("\n=== Full-sample IC ===")
    for col in new_feats.columns:
        rho = ic_report(new_feats[col], df[target_col], col, target_col)
        results[col] = {"full_sample_ic": rho}

    # ---- Bootstrap IC ----
    print("\n=== Bootstrap IC (1000x) ===")
    for col in new_feats.columns:
        bs = bootstrap_ic(new_feats[col], df[target_col])
        results[col]["bootstrap"] = bs
        print(f"  {col:34s}  mean={bs['mean']:+.4f}  "
              f"95% CI=[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  n={bs['n']}")

    # ---- Monthly IC stability ----
    print("\n=== Monthly IC ===")
    for col in new_feats.columns:
        mic = monthly_ic(new_feats[col], df[target_col])
        results[col]["monthly_ic"] = mic
        vals = list(mic.values())
        if vals:
            sign_consistent = sum(1 for v in vals if v > 0) / len(vals) if vals else 0
            print(f"  {col:34s}  months={len(vals)}  "
                  f"mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
                  f"sign_consistency={sign_consistent:.0%}")
            for m, v in mic.items():
                print(f"    {m}: IC={v:+.4f}")

    # ---- Train/Test split IC (50/50) ----
    print("\n=== Train/Test split IC (50/50) ===")
    mid = len(df) // 2
    for col in new_feats.columns:
        f = new_feats[col]
        rt = ic_report(f.iloc[:mid], df[target_col].iloc[:mid], col, f"{target_col}[train]")
        rs = ic_report(f.iloc[mid:], df[target_col].iloc[mid:], col, f"{target_col}[test]")
        if rt is not None and rs is not None:
            stable = "STABLE" if np.sign(rt) == np.sign(rs) else "FLIP"
            print(f"    -> {stable}")
            results[col]["train_ic"] = rt
            results[col]["test_ic"] = rs
            results[col]["train_test_stability"] = stable

    # ---- Rolling IC stats ----
    print("\n=== Rolling IC (168h window) stats ===")
    for col in new_feats.columns:
        ric = rolling_ic(new_feats[col], df[target_col], window=168)
        if len(ric) == 0:
            continue
        stats = {
            "mean": float(ric.mean()),
            "std": float(ric.std()),
            "min": float(ric.min()),
            "max": float(ric.max()),
            "frac_abs_gt_005": float((ric.abs() > 0.05).mean()),
        }
        results[col]["rolling_ic"] = stats
        print(f"  {col:34s}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}  "
              f"|IC|>0.05 frac={stats['frac_abs_gt_005']:.2%}")

    # ---- Correlation with top existing features ----
    print("\n=== Correlation with top existing features ===")
    top_existing = [
        "cg_oi_close_zscore", "cg_oi_agg_close", "cg_bfx_margin_ratio",
        "vol_acceleration", "cg_funding_close_zscore", "realized_vol_20b",
        "cg_funding_close", "cg_ls_ratio", "impact_asymmetry",
    ]
    for col in new_feats.columns:
        print(f"\n  {col}:")
        corr_map = {}
        for ref in top_existing:
            if ref in df.columns:
                corr = new_feats[col].corr(df[ref])
                flag = " *** HIGH" if abs(corr) > 0.7 else ""
                corr_map[ref] = round(float(corr), 3)
                print(f"    vs {ref:35s}  corr={corr:+.3f}{flag}")
        results[col]["correlation_with_existing"] = corr_map

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_onchain.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nResults saved to {out_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for col in new_feats.columns:
        r = results.get(col, {})
        ic = r.get("full_sample_ic")
        bs = r.get("bootstrap", {})
        stab = r.get("train_test_stability", "N/A")
        ci_lo = bs.get("ci_lo", np.nan)
        ci_hi = bs.get("ci_hi", np.nan)

        # Verdict
        if ic is None or np.isnan(ic):
            verdict = "SKIP (insufficient data)"
        elif ci_lo is not None and ci_hi is not None and not np.isnan(ci_lo):
            ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
            if ci_excludes_zero and stab == "STABLE":
                verdict = "CANDIDATE"
            elif ci_excludes_zero:
                verdict = "MARGINAL (CI excludes 0, but train/test flip)"
            else:
                verdict = "WEAK (CI includes 0)"
        else:
            verdict = "UNKNOWN"

        ic_str = f"{ic:+.4f}" if ic is not None and not np.isnan(ic) else "  N/A "
        print(f"  {col:34s}  IC={ic_str}  "
              f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"stability={stab:8s}  -> {verdict}")

    print(f"\nDone. Results at {out_path}")


if __name__ == "__main__":
    main()
