"""
Rolling training window experiment for Magnitude model.

Hypothesis:
    If the IC decay is driven by stale training data (model fits old regime
    and can't adapt), then a rolling window that drops old data should
    recover late-month OOS IC.

Compares:
    - expanding (current walk-forward behavior): train on all bars before te
    - rolling_500 / rolling_1000 / rolling_1500 / rolling_2000

All use the same purge=4, embargo=4, test_size=48, step=48, initial=288.

Metric: per-month OOS Spearman IC of (mag_pred, |ret_4h|), especially
2026-03 / 2026-04 where the concept drift was observed.
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
OUT = Path("research/results/mag_rolling_window.json")

HORIZON = 4
PURGE = 4
EMBARGO = 4

MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}

WINDOW_SIZES = {
    "expanding": None,
    "roll_2000": 2000,
    "roll_1500": 1500,
    "roll_1000": 1000,
    "roll_500": 500,
}


def walk_forward_rolling(df: pd.DataFrame, feats: list[str], y: np.ndarray,
                         window_cap: int | None) -> pd.DataFrame:
    """If window_cap is None → expanding. Otherwise use last N bars as train."""
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )
    records = []
    for tr_idx, te_idx in splits:
        tr_idx = list(tr_idx)
        if window_cap is not None and len(tr_idx) > window_cap:
            tr_idx = tr_idx[-window_cap:]
        tr_arr = np.array(tr_idx)
        tr_mask = ~np.isnan(y[tr_arr])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue
        X_tr = df.iloc[tr_arr][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_arr][tr_mask]
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask]
        idx_te = np.array(te_idx)[te_mask]

        m = xgb.XGBRegressor(**MAG_PARAMS)
        m.fit(X_tr, y_tr, verbose=False)
        pred = m.predict(X_te)
        for i, p, a in zip(idx_te, pred, y_te):
            records.append({"idx": int(i), "pred": float(p), "actual": float(a)})
    return pd.DataFrame(records)


def monthly_ic(oos: pd.DataFrame, df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    if oos.empty:
        return out
    oos = oos.copy()
    oos["ts"] = df.index[oos["idx"].values]
    oos["month"] = oos["ts"].dt.to_period("M").astype(str)
    for m, sub in oos.groupby("month"):
        if len(sub) < 30:
            continue
        ic, _ = spearmanr(sub["pred"], sub["actual"])
        out[m] = {"n": int(len(sub)), "ic": float(ic)}
    return out


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Rows: {len(df)}")

    prod_feats = json.loads(FEATS_PATH.read_text())
    feats = [f for f in prod_feats if f in df.columns]
    print(f"Mag features: {len(feats)}")

    ret_4h = (df["close"].shift(-HORIZON) / df["close"] - 1).values
    y = np.abs(ret_4h)

    results = {}
    for name, cap in WINDOW_SIZES.items():
        label = f"{name}" + (f" (cap={cap})" if cap else " (all history)")
        print(f"\n[{label}] walk-forward...")
        oos = walk_forward_rolling(df, feats, y, cap)
        results[name] = monthly_ic(oos, df)
        overall_ic = float(spearmanr(oos["pred"], oos["actual"])[0]) if len(oos) else float("nan")
        results[name]["_overall"] = {"n": int(len(oos)), "ic": overall_ic}

    # Print comparison table
    months = sorted({
        m for name in results for m in results[name]
        if not m.startswith("_")
    })
    print("\n" + "=" * 90)
    header = f"{'month':<12}" + "".join(f"{name:>14}" for name in WINDOW_SIZES)
    print(header)
    print("-" * 90)
    for m in months:
        row = f"{m:<12}"
        for name in WINDOW_SIZES:
            r = results[name].get(m)
            row += f"{(r['ic'] if r else float('nan')):>+14.4f}"
        print(row)
    print("-" * 90)
    row = f"{'ALL':<12}"
    for name in WINDOW_SIZES:
        row += f"{results[name]['_overall']['ic']:>+14.4f}"
    print(row)
    row_n = f"{'(n OOS)':<12}"
    for name in WINDOW_SIZES:
        row_n += f"{results[name]['_overall']['n']:>14d}"
    print(row_n)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {OUT}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT (focus on 2026-03 / 2026-04)")
    print("=" * 90)
    for target_month in ("2026-03", "2026-04"):
        print(f"\n  {target_month}:")
        for name in WINDOW_SIZES:
            r = results[name].get(target_month)
            if r:
                print(f"    {name:<12} IC={r['ic']:+.4f}  n={r['n']}")
    best_apr = max(
        ((n, results[n].get("2026-04", {}).get("ic", float("-inf")))
         for n in WINDOW_SIZES),
        key=lambda x: x[1],
    )
    print(f"\n  Best for 2026-04: {best_apr[0]}  IC={best_apr[1]:+.4f}")
    if best_apr[1] > 0.20:
        print("  → Short window helps. Recommend rolling retrain.")
    elif best_apr[1] > results["expanding"].get("2026-04", {}).get("ic", 0) + 0.03:
        print("  → Short window marginally better. Worth considering.")
    else:
        print("  → No window size recovers 2026-04 IC. Concept drift is real, not stale data.")


if __name__ == "__main__":
    run()
