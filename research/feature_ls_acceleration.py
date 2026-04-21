"""
IC validation for Long/Short Ratio Acceleration features:
  #1 ls_velocity_4h       — 4h change in LS ratio
  #2 ls_acceleration_8h   — change in velocity (second derivative)
  #3 ls_extreme_reversion — mean reversion signal at LS extremes
  #4 ls_oi_alignment      — alignment between LS velocity and OI change

Uses Coinglass LS ratio and OI columns.
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

CACHE = Path("research/dual_model/.cache/features_all.parquet")
RESULTS_DIR = Path("research/results")

# Possible column names for LS ratio and OI
LS_CANDIDATES = ["cg_ls_ratio", "cg_ls_agg_ratio", "cg_global_ls_ratio"]
OI_CANDIDATES = ["cg_oi_close", "cg_oi_agg_close", "cg_oi_total"]


def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str | None:
    """Find first available column from candidates list."""
    for c in candidates:
        if c in df.columns:
            print(f"  Using '{c}' for {label}")
            return c
    print(f"  WARNING: No column found for {label}. Tried: {candidates}")
    return None


def build_features(df: pd.DataFrame, ls_col: str, oi_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ls = df[ls_col].copy()

    # ---- #1: ls_velocity_4h ----
    # 4h change in LS ratio
    out["ls_velocity_4h"] = ls - ls.shift(4)

    # ---- #2: ls_acceleration_8h ----
    # Change in velocity = second derivative of LS ratio
    velocity = out["ls_velocity_4h"]
    out["ls_acceleration_8h"] = velocity - velocity.shift(4)

    # ---- #3: ls_extreme_reversion ----
    # When LS ratio is at rolling extreme, predict mean reversion
    # z-score using 24h rolling stats
    rolling_med = ls.rolling(24, min_periods=12).median()
    rolling_std = ls.rolling(24, min_periods=12).std()
    z = (ls - rolling_med) / rolling_std.replace(0, np.nan)

    # Only fire when |z| > 1.5, direction = -z (mean reversion)
    out["ls_extreme_reversion"] = np.where(
        z.abs() > 1.5,
        -1.0 * z,
        0.0
    )

    # ---- #4: ls_oi_alignment ----
    if oi_col is not None:
        oi = df[oi_col]
        oi_pct_4h = oi.pct_change(4)
        # sign(ls_velocity) * sign(oi_pct_change_4h)
        # +1 = aligned (both increasing or both decreasing)
        # -1 = divergent
        out["ls_oi_alignment"] = np.sign(velocity) * np.sign(oi_pct_4h)
        # Replace NaN from sign(0)=0 cases
        out["ls_oi_alignment"] = out["ls_oi_alignment"].replace(0, np.nan)
    else:
        print("  SKIPPING ls_oi_alignment (no OI column available)")

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

    # Find LS ratio and OI columns
    print("\nColumn detection:")
    ls_col = _find_column(df, LS_CANDIDATES, "LS ratio")
    oi_col = _find_column(df, OI_CANDIDATES, "OI")

    if ls_col is None:
        print("\nERROR: No LS ratio column found. Cannot proceed.")
        print("Available cg_ columns:")
        for c in sorted(df.columns):
            if c.startswith("cg_"):
                print(f"  {c}")
        return

    # Check required columns
    if "close" not in df.columns:
        print("\nERROR: Missing 'close' column. Cannot proceed.")
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
    new_feats = build_features(df, ls_col, oi_col)
    print(f"\nNew features built: {list(new_feats.columns)}")
    for c in new_feats.columns:
        non_zero = (new_feats[c] != 0).sum() if c == "ls_extreme_reversion" else len(new_feats[c].dropna())
        print(f"  {c:30s}  nan={new_feats[c].isna().mean():.2%}  "
              f"non_zero/valid={non_zero}  std={new_feats[c].std():.6f}")

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
                    "vol_acceleration", "cg_funding_close_zscore", "cg_ls_ratio",
                    "cg_ls_agg_ratio", "cg_oi_close_pctchg_8h"]
    for col in new_feats.columns:
        print(f"\n  {col}:")
        corr_map = {}
        for ref in top_existing:
            if ref in df.columns:
                corr = new_feats[col].corr(df[ref])
                corr_map[ref] = round(float(corr), 3)
                print(f"    vs {ref:35s}  corr={corr:+.3f}")
        results[col]["correlation_with_existing"] = corr_map

    # ---- Cross-correlation between new features ----
    print("\n=== Cross-correlation between new features ===")
    cross_corr = {}
    cols = list(new_feats.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = new_feats[cols[i]].corr(new_feats[cols[j]])
            pair = f"{cols[i]} x {cols[j]}"
            cross_corr[pair] = round(float(c), 3) if not np.isnan(c) else None
            print(f"  {pair:55s}  corr={c:+.3f}" if not np.isnan(c) else f"  {pair:55s}  corr=NaN")

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "feature_ls_acceleration.json"
    results["_cross_correlation"] = cross_corr
    results["_columns_used"] = {"ls_ratio": ls_col, "oi": oi_col}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nResults saved to {out_path}")

    feat_path = RESULTS_DIR / "feature_ls_acceleration.parquet"
    new_feats.to_parquet(feat_path)
    print(f"Features saved to {feat_path}")


if __name__ == "__main__":
    main()
