"""
IC validation for Liquidity Migration features:
  #1 volume_session_ratio       — volume last 8h / previous 8h (activity shift)
  #2 volume_concentration       — max(hourly vol in 8h) / mean(hourly vol in 8h)
  #3 volume_trend_8h            — linreg slope of hourly volume over 8 bars, normalized
  #4 night_day_ratio            — Asia session vol / US session vol over trailing 24h

All features are trailing-only (no look-ahead).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.build_direction_labels import build_direction_labels

CACHE = Path("research/dual_model/.cache/features_all.parquet")
RESULTS_DIR = Path("research/results")


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    vol = df["volume"].copy()

    # ---- #1: volume_session_ratio ----
    # volume in last 8h / volume in previous 8h
    vol_8h = vol.rolling(8, min_periods=4).sum()
    vol_prev_8h = vol.shift(8).rolling(8, min_periods=4).sum()
    out["volume_session_ratio"] = vol_8h / vol_prev_8h.replace(0, np.nan)

    # ---- #2: volume_concentration ----
    # max(hourly vol in last 8h) / mean(hourly vol in last 8h)
    vol_max_8h = vol.rolling(8, min_periods=4).max()
    vol_mean_8h = vol.rolling(8, min_periods=4).mean()
    out["volume_concentration"] = vol_max_8h / vol_mean_8h.replace(0, np.nan)

    # ---- #3: volume_trend_8h ----
    # Linear regression slope of hourly volume over 8 bars, normalized by mean
    def _vol_trend(window_vals):
        if np.isnan(window_vals).sum() > len(window_vals) // 2:
            return np.nan
        y = window_vals
        x = np.arange(len(y))
        valid = ~np.isnan(y)
        if valid.sum() < 4:
            return np.nan
        slope, _, _, _, _ = linregress(x[valid], y[valid])
        mean_val = np.nanmean(y)
        if mean_val == 0:
            return np.nan
        return slope / mean_val

    out["volume_trend_8h"] = vol.rolling(8, min_periods=4).apply(
        _vol_trend, raw=True
    )

    # ---- #4: night_day_ratio ----
    # Asia session (00-08 UTC) vs US session (13-21 UTC) over trailing 24h
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour

    asia_mask = (hour >= 0) & (hour < 8)
    us_mask = (hour >= 13) & (hour < 21)

    asia_vol = vol.where(asia_mask, 0.0)
    us_vol = vol.where(us_mask, 0.0)

    asia_24h = asia_vol.rolling(24, min_periods=8).sum()
    us_24h = us_vol.rolling(24, min_periods=8).sum()
    out["night_day_ratio"] = asia_24h / us_24h.replace(0, np.nan)

    return out


def ic_report(feat: pd.Series, target: pd.Series, name: str, target_name: str):
    mask = feat.notna() & target.notna()
    if mask.sum() < 100:
        print(f"  {name} vs {target_name}: insufficient samples ({mask.sum()})")
        return None
    rho, p = spearmanr(feat[mask], target[mask])
    n = int(mask.sum())
    print(f"  {name:30s} vs {target_name:14s}  IC={rho:+.4f}  p={p:.2e}  n={n}")
    return rho


def bootstrap_ic(feat: pd.Series, target: pd.Series, n_boot: int = 1000,
                 seed: int = 42) -> dict:
    """Bootstrap 1000x Spearman IC, return mean, std, CI."""
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
    """Compute Spearman IC per calendar month."""
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


def main():
    print(f"Loading features from {CACHE}...")
    df = pd.read_parquet(CACHE)
    print(f"  shape={df.shape}  range={df.index.min()} -> {df.index.max()}")

    # Print available columns for reference
    print(f"\nAvailable columns ({len(df.columns)}):")
    for c in sorted(df.columns):
        print(f"  {c}")

    # Check required columns
    required = ["volume", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        print("Cannot proceed. Exiting.")
        return

    # Build labels
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_return_4h"] = labels["return_4h"]

    # Build TWAP path return target
    close = df["close"].values.astype(float)
    n = len(close)
    path_ret = np.full(n, np.nan)
    for i in range(n - 4):
        path_ret[i] = (sum(close[i + k] for k in range(1, 5)) / 4.0) / close[i] - 1
    df["y_path_ret_4h"] = path_ret

    # Build features
    new_feats = build_features(df)
    print(f"\nNew features built: {list(new_feats.columns)}")
    for c in new_feats.columns:
        print(f"  {c:30s}  nan={new_feats[c].isna().mean():.2%}  "
              f"std={new_feats[c].std():.4f}")

    results = {}
    target_col = "y_path_ret_4h"

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
        print(f"  {col:30s}  mean={bs['mean']:+.4f}  95% CI=[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  n={bs['n']}")

    # ---- Monthly IC stability ----
    print("\n=== Monthly IC ===")
    for col in new_feats.columns:
        mic = monthly_ic(new_feats[col], df[target_col])
        results[col]["monthly_ic"] = mic
        vals = list(mic.values())
        if vals:
            sign_consistent = sum(1 for v in vals if v > 0) / len(vals) if vals else 0
            print(f"  {col:30s}  months={len(vals)}  "
                  f"mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
                  f"sign_consistency={sign_consistent:.0%}")
            for m, v in mic.items():
                print(f"    {m}: IC={v:+.4f}")

    # ---- Train/Test split IC ----
    print("\n=== Train/Test split IC (50/50) ===")
    mid = len(df) // 2
    for col in new_feats.columns:
        f = new_feats[col]
        rt = ic_report(f.iloc[:mid], df[target_col].iloc[:mid], col, f"{target_col}[train]")
        rs = ic_report(f.iloc[mid:], df[target_col].iloc[mid:], col, f"{target_col}[test]")
        if rt is not None and rs is not None:
            stable = "STABLE" if np.sign(rt) == np.sign(rs) else "FLIP"
            print(f"    -> {stable}")
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
        print(f"  {col:30s}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}  "
              f"|IC|>0.05 frac={stats['frac_abs_gt_005']:.2%}")

    # ---- Correlation with top existing features ----
    print("\n=== Correlation with top existing features ===")
    top_existing = ["cg_oi_close_zscore", "cg_oi_agg_close", "cg_bfx_margin_ratio",
                    "vol_acceleration", "cg_funding_close_zscore", "realized_vol_20b",
                    "volume_zscore", "volume"]
    for col in new_feats.columns:
        print(f"\n  {col}:")
        corr_map = {}
        for ref in top_existing:
            if ref in df.columns:
                corr = new_feats[col].corr(df[ref])
                corr_map[ref] = round(float(corr), 3)
                print(f"    vs {ref:35s}  corr={corr:+.3f}")
        results[col]["correlation_with_existing"] = corr_map

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_liquidity_migration.json"
    # Convert any non-serializable values
    for col_data in results.values():
        for k, v in col_data.items():
            if isinstance(v, (np.floating, np.integer)):
                col_data[k] = float(v) if v is not None else None
            elif v is None:
                col_data[k] = None

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nResults saved to {out_path}")

    # Save features parquet
    feat_path = RESULTS_DIR / "feature_liquidity_migration.parquet"
    new_feats.to_parquet(feat_path)
    print(f"Features saved to {feat_path}")


if __name__ == "__main__":
    main()
