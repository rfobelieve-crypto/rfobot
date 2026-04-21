"""
IC validation for Funding Settlement Anomaly features:
  #1 pre_settlement_momentum      — price change in 2h before funding settlement
  #2 pre_settlement_volume_spike  — volume ratio for bars within 2h of settlement
  #3 post_settlement_reversal     — price reversal after settlement
  #4 settlement_proximity         — hours until next funding settlement (cyclical)

BTC funding settles every 8h at 00:00, 08:00, 16:00 UTC.
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    vol = df["volume"]

    # Extract hour from datetime index
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour

    # Settlement hours: 0, 8, 16 UTC
    # Hours until next settlement: hour % 8 gives position within cycle
    # hours_until = 8 - (hour % 8), with 8 -> 0
    hours_in_cycle = hour % 8  # 0=at settlement, 1=1h after, ..., 7=1h before

    # ---- #1: pre_settlement_momentum ----
    # Price change in the 2h before settlement (hours_in_cycle >= 6)
    # i.e., when we are 1-2h before next settlement
    is_pre_settlement = (hours_in_cycle >= 6)  # hour%8 == 6 or 7
    price_change_2h = close / close.shift(2) - 1
    out["pre_settlement_momentum"] = np.where(
        is_pre_settlement, price_change_2h, 0.0
    )

    # ---- #2: pre_settlement_volume_spike ----
    # Volume ratio (current bar / 8h avg) but only for pre-settlement bars
    vol_avg_8h = vol.rolling(8, min_periods=4).mean()
    vol_ratio = vol / vol_avg_8h.replace(0, np.nan)
    out["pre_settlement_volume_spike"] = np.where(
        is_pre_settlement, vol_ratio, 0.0
    )

    # ---- #3: post_settlement_reversal ----
    # After settlement, does price reverse vs the pre-settlement move?
    # For bars 1-2h after settlement (hours_in_cycle == 1 or 2)
    is_post_settlement = (hours_in_cycle >= 1) & (hours_in_cycle <= 2)

    # Find the settlement bar close (the bar where hours_in_cycle == 0)
    # For hours_in_cycle=1: settlement was 1h ago, for hours_in_cycle=2: 2h ago
    settlement_close = close.copy()
    settlement_close[hours_in_cycle != 0] = np.nan
    settlement_close = settlement_close.ffill()  # forward fill to post-settlement bars

    post_reversal = close / settlement_close - 1
    out["post_settlement_reversal"] = np.where(
        is_post_settlement, post_reversal, 0.0
    )

    # ---- #4: settlement_proximity ----
    # Continuous feature: hours until next settlement (0-7 range)
    # hours_in_cycle=0 means at settlement, hours until next = 8 (or 0 if just settled)
    # hours_in_cycle=7 means 1h until next settlement
    hours_until = (8 - hours_in_cycle) % 8  # 0 at settlement, 7 one hour after
    out["settlement_proximity"] = hours_until.astype(float)

    return out


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
    required = ["close", "volume"]
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
        non_zero = (new_feats[c] != 0).sum()
        print(f"  {c:34s}  nan={new_feats[c].isna().mean():.2%}  "
              f"non_zero={non_zero} ({non_zero/len(new_feats):.1%})  "
              f"std={new_feats[c].std():.6f}")

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
        print(f"  {col:34s}  mean={bs['mean']:+.4f}  95% CI=[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  n={bs['n']}")

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
        print(f"  {col:34s}  mean={stats['mean']:+.4f}  std={stats['std']:.4f}  "
              f"|IC|>0.05 frac={stats['frac_abs_gt_005']:.2%}")

    # ---- Correlation with top existing features ----
    print("\n=== Correlation with top existing features ===")
    top_existing = ["cg_oi_close_zscore", "cg_oi_agg_close", "cg_bfx_margin_ratio",
                    "vol_acceleration", "cg_funding_close_zscore", "realized_vol_20b",
                    "cg_funding_close", "cg_ls_ratio"]
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
    out_path = RESULTS_DIR / "feature_funding_settlement.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    print(f"\nResults saved to {out_path}")

    feat_path = RESULTS_DIR / "feature_funding_settlement.parquet"
    new_feats.to_parquet(feat_path)
    print(f"Features saved to {feat_path}")


if __name__ == "__main__":
    main()
