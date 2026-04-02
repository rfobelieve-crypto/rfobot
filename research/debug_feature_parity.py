"""
Train-Serve Skew Detector: compare live features vs training parquet.

Fetches live data, builds features, then compares against the training
parquet for overlapping bars. Any mismatch > threshold indicates skew.

Usage:
    python research/debug_feature_parity.py
"""
import sys
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
from indicator.feature_builder_live import build_live_features

PARQUET_PATH = "research/ml_data/BTC_USD_1h_enhanced.parquet"
THRESHOLD = 1e-4  # Features with diff > this are flagged
CRITICAL_THRESHOLD = 0.01  # Diffs > this are serious

print("=" * 70)
print("TRAIN-SERVE SKEW DETECTOR")
print("=" * 70)

# ── Load training parquet ──────────────────────────────────────────────
print("\n1. Loading training parquet...")
train_df = pd.read_parquet(PARQUET_PATH)
print(f"   Training data: {train_df.index[0]} ~ {train_df.index[-1]} ({len(train_df)} bars)")

# ── Build live features ────────────────────────────────────────────────
print("\n2. Fetching live data & building features...")
klines = fetch_binance_klines(limit=500)
cg = fetch_coinglass(interval="1h", limit=500)
live_df = build_live_features(klines, cg)
print(f"   Live features: {live_df.index[0]} ~ {live_df.index[-1]} ({len(live_df)} bars)")

# ── Find overlap ───────────────────────────────────────────────────────
overlap_start = max(train_df.index[0], live_df.index[0])
overlap_end = min(train_df.index[-1], live_df.index[-1])

train_overlap = train_df[(train_df.index >= overlap_start) & (train_df.index <= overlap_end)]
live_overlap = live_df[(live_df.index >= overlap_start) & (live_df.index <= overlap_end)]

# Align indices
common_idx = train_overlap.index.intersection(live_overlap.index)
print(f"\n3. Overlap: {overlap_start} ~ {overlap_end}")
print(f"   Common bars: {len(common_idx)}")

if len(common_idx) < 10:
    print("\n   WARNING: Very few overlapping bars. Training parquet may be stale.")
    print("   Consider running rebuild_1h_enhanced.py first.")
    sys.exit(0)

# Skip first 50 bars (rolling windows need warmup — expect diffs there)
# Focus on bars where both live and training have full window history
skip_warmup = 50
if len(common_idx) > skip_warmup + 10:
    common_idx = common_idx[skip_warmup:]
    print(f"   After skipping {skip_warmup} warmup bars: {len(common_idx)} bars for comparison")

train_sub = train_overlap.loc[common_idx]
live_sub = live_overlap.loc[common_idx]

# ── Compare shared features ───────────────────────────────────────────
shared_cols = sorted(set(train_sub.columns) & set(live_sub.columns))
# Exclude non-feature columns
exclude = {"open", "high", "low", "close", "volume", "taker_buy_vol",
           "taker_buy_quote", "trade_count", "quote_vol",
           "y_return_4h", "y_return_1h", "regime",
           "up_move", "down_move", "strength",
           "up_move_vol_adj", "down_move_vol_adj", "strength_vol_adj"}
feature_cols = [c for c in shared_cols if c not in exclude]

print(f"\n4. Comparing {len(feature_cols)} shared feature columns...")
print()

# ── Per-feature analysis ──────────────────────────────────────────────
results = []
for col in feature_cols:
    t = train_sub[col].astype(float).values
    l = live_sub[col].astype(float).values

    # Skip if both are all NaN
    t_na = np.isnan(t)
    l_na = np.isnan(l)
    valid = (~t_na) & (~l_na)
    n_valid = valid.sum()
    if n_valid < 5:
        results.append({
            "feature": col, "max_diff": np.nan, "mean_diff": np.nan,
            "nan_mismatch": 0, "n_valid": n_valid, "status": "SKIP (few valid)"
        })
        continue

    abs_diff = np.abs(t[valid] - l[valid])

    # NaN mismatch: one has value, other is NaN
    nan_mismatch = int((t_na != l_na).sum())

    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    median_diff = float(np.median(abs_diff))

    # Relative diff (normalized by feature std)
    feature_std = float(np.std(t[valid]))
    rel_max = max_diff / feature_std if feature_std > 0 else 0

    if max_diff > CRITICAL_THRESHOLD:
        status = "CRITICAL"
    elif max_diff > THRESHOLD:
        status = "WARNING"
    elif nan_mismatch > 0:
        status = "NaN MISMATCH"
    else:
        status = "OK"

    results.append({
        "feature": col,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "rel_max": rel_max,
        "nan_mismatch": nan_mismatch,
        "n_valid": n_valid,
        "status": status,
    })

results_df = pd.DataFrame(results).sort_values("max_diff", ascending=False)

# ── Report ─────────────────────────────────────────────────────────────
critical = results_df[results_df["status"] == "CRITICAL"]
warning = results_df[results_df["status"] == "WARNING"]
nan_mm = results_df[results_df["status"] == "NaN MISMATCH"]
ok = results_df[results_df["status"] == "OK"]

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"  CRITICAL (diff > {CRITICAL_THRESHOLD}):  {len(critical)}")
print(f"  WARNING  (diff > {THRESHOLD}):  {len(warning)}")
print(f"  NaN MISMATCH:                    {len(nan_mm)}")
print(f"  OK:                              {len(ok)}")
print()

if len(critical) > 0:
    print("=== CRITICAL FEATURES (train-serve skew!) ===")
    for _, row in critical.iterrows():
        print(f"  {row['feature']:40s}  max_diff={row['max_diff']:.6f}  "
              f"mean={row['mean_diff']:.6f}  rel={row['rel_max']:.2f}σ  "
              f"nan_mm={row['nan_mismatch']}")
    print()

if len(warning) > 0:
    print("=== WARNING FEATURES ===")
    for _, row in warning.iterrows():
        print(f"  {row['feature']:40s}  max_diff={row['max_diff']:.6f}  "
              f"mean={row['mean_diff']:.6f}  rel={row['rel_max']:.2f}σ")
    print()

if len(nan_mm) > 0:
    print("=== NaN MISMATCH FEATURES ===")
    for _, row in nan_mm.iterrows():
        print(f"  {row['feature']:40s}  nan_mismatch={row['nan_mismatch']}")
    print()

# ── Detailed spot-check on worst features ──────────────────────────────
if len(critical) > 0 or len(warning) > 0:
    worst = results_df[results_df["status"].isin(["CRITICAL", "WARNING"])].head(5)
    print("=== SPOT CHECK (last 5 bars, worst features) ===")
    check_idx = common_idx[-5:]
    for _, row in worst.iterrows():
        col = row["feature"]
        print(f"\n  {col}:")
        print(f"  {'Bar':20s}  {'Train':>12s}  {'Live':>12s}  {'Diff':>12s}")
        for idx in check_idx:
            tv = float(train_sub.loc[idx, col]) if not pd.isna(train_sub.loc[idx, col]) else np.nan
            lv = float(live_sub.loc[idx, col]) if not pd.isna(live_sub.loc[idx, col]) else np.nan
            d = abs(tv - lv) if not (np.isnan(tv) or np.isnan(lv)) else np.nan
            print(f"  {str(idx)[:19]}  {tv:12.6f}  {lv:12.6f}  {d:12.6f}")

# ── Feature presence check ─────────────────────────────────────────────
print()
print("=== FEATURE PRESENCE CHECK ===")
train_only = set(train_sub.columns) - set(live_sub.columns) - exclude
live_only = set(live_sub.columns) - set(train_sub.columns) - exclude
if train_only:
    print(f"  In training but NOT in live: {sorted(train_only)}")
if live_only:
    print(f"  In live but NOT in training: {sorted(live_only)}")
if not train_only and not live_only:
    print(f"  All feature columns match ({len(feature_cols)} features)")

# ── Impact estimate ────────────────────────────────────────────────────
print()
print("=== IMPACT ON STRENGTH SIGNAL ===")
# Check if the skewed features are in the model's active feature sets
import json
from pathlib import Path

regime_dir = Path("indicator/model_artifacts/regime_models")
for target in ["up_move_vol_adj", "down_move_vol_adj"]:
    fc_path = regime_dir / target / "feature_cols.json"
    if fc_path.exists():
        with open(fc_path) as f:
            model_features = json.load(f)
        skewed = [r["feature"] for _, r in critical.iterrows()
                  if r["feature"] in model_features]
        if skewed:
            print(f"  {target}: {len(skewed)} CRITICAL features used by model: {skewed}")
        else:
            print(f"  {target}: No critical features used by model")

print()
if len(critical) == 0 and len(warning) == 0:
    print("VERDICT: No significant train-serve skew detected.")
else:
    print(f"VERDICT: {len(critical)} critical + {len(warning)} warning features with skew.")
    print("         Fix these before trusting live signals.")
