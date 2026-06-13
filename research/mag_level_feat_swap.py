"""
Test whether replacing non-stationary level features with zscore/pctchg
versions recovers Magnitude walk-forward OOS IC.

Swaps:
    cg_oi_close       → cg_oi_close_zscore
    cg_pos_ls_ratio   → cg_pos_ls_ratio_zscore

Both stationary alternatives already exist in the feature cache.

Walk-forward config matches production:
    initial_train=288, test_size=48, step=48, purge=4, embargo=4

Reports monthly IC + ICIR + top/bot ratio for:
    - baseline (current production feature set)
    - swapped  (same set with 2 features replaced)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits

FEATS_PATH = Path("indicator/model_artifacts/dual_model/magnitude_feature_cols.json")
OUT = Path("research/results/mag_level_feat_swap.json")

HORIZON = 4
PURGE = 4
EMBARGO = 4

SWAPS = {
    "cg_oi_close": "cg_oi_close_zscore",
    "cg_pos_ls_ratio": "cg_pos_ls_ratio_zscore",
}

MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}


def walk_forward_oos(df: pd.DataFrame, feats: list[str], y: np.ndarray) -> pd.DataFrame:
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )
    records = []
    for tr_idx, te_idx in splits:
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask]
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask]
        idx_te = np.array(te_idx)[te_mask]

        m = xgb.XGBRegressor(**MAG_PARAMS)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict(X_te)
        for i, p, actual in zip(idx_te, pred, y_te):
            records.append({"idx": int(i), "pred": float(p), "actual": float(actual)})
    return pd.DataFrame(records)


def monthly_metrics(oos: pd.DataFrame, df: pd.DataFrame, name: str) -> dict:
    oos = oos.copy()
    oos["ts"] = df.index[oos["idx"].values]
    oos["month"] = oos["ts"].dt.to_period("M").astype(str)
    rows = []
    for m, sub in oos.groupby("month"):
        if len(sub) < 30:
            continue
        ic, _ = spearmanr(sub["pred"], sub["actual"])
        # ICIR proxy: split halves
        half = len(sub) // 2
        ic1, _ = spearmanr(sub.iloc[:half]["pred"], sub.iloc[:half]["actual"])
        ic2, _ = spearmanr(sub.iloc[half:]["pred"], sub.iloc[half:]["actual"])
        icir = float(np.mean([ic1, ic2]) / np.std([ic1, ic2])) if np.std([ic1, ic2]) > 0 else float("nan")
        # top/bot ratio
        q = sub["pred"].quantile([0.2, 0.8])
        top_mean = sub[sub["pred"] >= q.iloc[1]]["actual"].mean()
        bot_mean = sub[sub["pred"] <= q.iloc[0]]["actual"].mean()
        ratio = top_mean / bot_mean if bot_mean > 0 else float("nan")
        rows.append({"month": m, "n": len(sub), "ic": ic, "icir_halves": icir, "top_bot": ratio})
    return {"name": name, "n_total": len(oos), "overall_ic": float(spearmanr(oos["pred"], oos["actual"])[0]),
            "monthly": rows}


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Rows: {len(df)}")

    prod_feats = json.loads(FEATS_PATH.read_text())
    prod_feats = [f for f in prod_feats if f in df.columns]
    print(f"Production Mag features: {len(prod_feats)}")

    # Verify swaps
    for old, new in SWAPS.items():
        if old not in prod_feats:
            print(f"  WARN: {old} not in prod feats, skipping swap")
        if new not in df.columns:
            print(f"  WARN: {new} not in df columns!")
    swapped_feats = [SWAPS.get(f, f) for f in prod_feats]
    swapped_feats = [f for f in swapped_feats if f in df.columns]
    # Drop duplicates preserving order
    seen, swapped_feats_u = set(), []
    for f in swapped_feats:
        if f not in seen:
            swapped_feats_u.append(f)
            seen.add(f)
    print(f"After swap: {len(swapped_feats_u)} features")

    # Target: |ret_4h|
    ret_4h = (df["close"].shift(-HORIZON) / df["close"] - 1).values
    y = np.abs(ret_4h)

    print("\n[1/2] Walk-forward baseline (production feature set)...")
    oos_base = walk_forward_oos(df, prod_feats, y)
    print(f"  n OOS = {len(oos_base)}")

    print("\n[2/2] Walk-forward SWAPPED feature set...")
    oos_swap = walk_forward_oos(df, swapped_feats_u, y)
    print(f"  n OOS = {len(oos_swap)}")

    base = monthly_metrics(oos_base, df, "baseline")
    swap = monthly_metrics(oos_swap, df, "swapped")

    print("\n" + "=" * 78)
    print(f"{'month':<10}{'n':>6}{'base IC':>11}{'swap IC':>11}{'Δ':>9}{'base TB':>10}{'swap TB':>10}")
    print("-" * 78)
    base_by_m = {r["month"]: r for r in base["monthly"]}
    swap_by_m = {r["month"]: r for r in swap["monthly"]}
    months = sorted(set(base_by_m) | set(swap_by_m))
    for m in months:
        b = base_by_m.get(m, {})
        s = swap_by_m.get(m, {})
        if not b or not s:
            continue
        d = s["ic"] - b["ic"]
        print(f"{m:<10}{b['n']:>6}{b['ic']:>+11.4f}{s['ic']:>+11.4f}{d:>+9.4f}"
              f"{b.get('top_bot', float('nan')):>+10.2f}{s.get('top_bot', float('nan')):>+10.2f}")
    print("-" * 78)
    print(f"{'ALL':<10}{base['n_total']:>6}{base['overall_ic']:>+11.4f}{swap['overall_ic']:>+11.4f}"
          f"{(swap['overall_ic'] - base['overall_ic']):>+9.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "swaps": SWAPS,
        "baseline": base,
        "swapped": swap,
        "n_feats_base": len(prod_feats),
        "n_feats_swapped": len(swapped_feats_u),
    }, indent=2, default=str))
    print(f"\nSaved: {OUT}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    delta = swap["overall_ic"] - base["overall_ic"]
    if delta > 0.02:
        print(f"  Swap improves overall IC by {delta:+.4f}. → Migrate features.")
    elif delta > 0:
        print(f"  Swap marginally better ({delta:+.4f}). Check per-month — may still help late months.")
    else:
        print(f"  Swap worsens overall IC by {delta:+.4f}. Level features may encode useful info.")


if __name__ == "__main__":
    run()
