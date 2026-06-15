"""
Ensemble experiment: mix expanding-window Mag model with rolling-500 model.

Rationale (from mag_rolling_window.py):
    - expanding IC is stable across all months but weak in recent (0.07 Apr)
    - roll_500 recovers Apr to 0.13 but only has 20 days of training data
      (overfit risk, single-event sensitivity)
    - Hypothesis: a weighted average gets the best of both — recent adaptation
      plus long-run stability.

Compares:
    - expanding alone (baseline)
    - roll_500 alone
    - ensemble rank-averaged and score-averaged with weights
      {0.2, 0.3, 0.5, 0.7} on roll_500

Metric: per-month Spearman IC of (ensemble_pred, |ret_4h|).
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
OUT = Path("research/results/mag_ensemble.json")

HORIZON = 4
PURGE = 4
EMBARGO = 4

MAG_PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "mae",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42, "verbosity": 0,
}


def walk_forward_both(df: pd.DataFrame, feats: list[str], y: np.ndarray,
                      roll_cap: int = 500) -> pd.DataFrame:
    """Single pass: produces both expanding and rolling predictions per OOS bar."""
    splits = walk_forward_splits(
        len(df), initial_train=288, test_size=48, step=48,
        purge=PURGE, embargo=EMBARGO,
    )
    records = []
    for tr_idx, te_idx in splits:
        tr_arr_exp = np.array(tr_idx)
        tr_mask_exp = ~np.isnan(y[tr_arr_exp])
        if tr_mask_exp.sum() < 50:
            continue
        # Rolling subset
        if len(tr_arr_exp) > roll_cap:
            tr_arr_roll = tr_arr_exp[-roll_cap:]
        else:
            tr_arr_roll = tr_arr_exp
        tr_mask_roll = ~np.isnan(y[tr_arr_roll])
        if tr_mask_roll.sum() < 50:
            continue

        te_mask = ~np.isnan(y[te_idx])
        if te_mask.sum() < 5:
            continue
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        y_te = y[te_idx][te_mask]
        idx_te = np.array(te_idx)[te_mask]

        # Expanding model
        X_tr_e = df.iloc[tr_arr_exp][feats].fillna(0).values[tr_mask_exp]
        y_tr_e = y[tr_arr_exp][tr_mask_exp]
        m_exp = xgb.XGBRegressor(**MAG_PARAMS)
        m_exp.fit(X_tr_e, y_tr_e, verbose=False)
        pred_exp = m_exp.predict(X_te)

        # Rolling model
        X_tr_r = df.iloc[tr_arr_roll][feats].fillna(0).values[tr_mask_roll]
        y_tr_r = y[tr_arr_roll][tr_mask_roll]
        m_roll = xgb.XGBRegressor(**MAG_PARAMS)
        m_roll.fit(X_tr_r, y_tr_r, verbose=False)
        pred_roll = m_roll.predict(X_te)

        for i, pe, pr, a in zip(idx_te, pred_exp, pred_roll, y_te):
            records.append({"idx": int(i), "pred_exp": float(pe),
                            "pred_roll": float(pr), "actual": float(a)})
    return pd.DataFrame(records)


def eval_series(oos: pd.DataFrame, col: str, df: pd.DataFrame) -> dict:
    oos = oos.copy()
    oos["ts"] = df.index[oos["idx"].values]
    oos["month"] = oos["ts"].dt.to_period("M").astype(str)
    out = {}
    for m, sub in oos.groupby("month"):
        if len(sub) < 30:
            continue
        ic, _ = spearmanr(sub[col], sub["actual"])
        out[m] = float(ic)
    ic_all, _ = spearmanr(oos[col], oos["actual"])
    out["_ALL"] = float(ic_all)
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def run() -> None:
    print("Loading data...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    print(f"Rows: {len(df)}")

    prod_feats = json.loads(FEATS_PATH.read_text())
    feats = [f for f in prod_feats if f in df.columns]
    print(f"Mag features: {len(feats)}")

    ret_4h = (df["close"].shift(-HORIZON) / df["close"] - 1).values
    y = np.abs(ret_4h)

    print("\nRunning dual walk-forward (expanding + roll_500)...")
    oos = walk_forward_both(df, feats, y, roll_cap=500)
    print(f"n OOS = {len(oos)}")

    # Build ensembles: score-average at various weights (w = roll weight)
    weights = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    # Score-level ensemble (raw XGB output is already regression target space)
    for w in weights:
        oos[f"pred_mix_{int(w*100)}"] = (1 - w) * oos["pred_exp"] + w * oos["pred_roll"]
    # Rank-level ensemble (robust to scale differences)
    rank_exp = oos["pred_exp"].rank(pct=True).values
    rank_roll = oos["pred_roll"].rank(pct=True).values
    for w in weights:
        oos[f"pred_rank_{int(w*100)}"] = (1 - w) * rank_exp + w * rank_roll

    # Evaluate per-month
    results = {}
    for w in weights:
        results[f"score_{int(w*100)}"] = eval_series(oos, f"pred_mix_{int(w*100)}", df)
        results[f"rank_{int(w*100)}"] = eval_series(oos, f"pred_rank_{int(w*100)}", df)

    # Print comparison focused on Apr
    print("\n" + "=" * 92)
    print("Per-month IC (score-average ensemble; w = roll_500 weight)")
    print("=" * 92)
    header = f"{'month':<10}" + "".join(f"{'w=' + str(w):>11}" for w in weights)
    print(header)
    print("-" * 92)
    months = sorted({
        m for r in results.values() for m in r if not m.startswith("_")
    })
    for m in months:
        row = f"{m:<10}"
        for w in weights:
            ic = results[f"score_{int(w*100)}"].get(m, float("nan"))
            row += f"{ic:>+11.4f}"
        print(row)
    print("-" * 92)
    row = f"{'ALL':<10}"
    for w in weights:
        row += f"{results[f'score_{int(w*100)}']['_ALL']:>+11.4f}"
    print(row)

    print("\n" + "=" * 92)
    print("Per-month IC (rank-average ensemble)")
    print("=" * 92)
    print(header)
    print("-" * 92)
    for m in months:
        row = f"{m:<10}"
        for w in weights:
            ic = results[f"rank_{int(w*100)}"].get(m, float("nan"))
            row += f"{ic:>+11.4f}"
        print(row)
    print("-" * 92)
    row = f"{'ALL':<10}"
    for w in weights:
        row += f"{results[f'rank_{int(w*100)}']['_ALL']:>+11.4f}"
    print(row)

    # Verdict
    print("\n" + "=" * 92)
    print("VERDICT — focus on 2026-04 (the drift zone)")
    print("=" * 92)
    apr_results = []
    for w in weights:
        ic_s = results[f"score_{int(w*100)}"].get("2026-04", float("nan"))
        ic_r = results[f"rank_{int(w*100)}"].get("2026-04", float("nan"))
        apr_results.append((w, ic_s, ic_r))
        print(f"  w={w:<4}  score_avg IC={ic_s:+.4f}  rank_avg IC={ic_r:+.4f}")

    best_score = max(apr_results, key=lambda x: x[1])
    best_rank = max(apr_results, key=lambda x: x[2])
    print(f"\n  Best score-avg: w={best_score[0]}  Apr IC={best_score[1]:+.4f}")
    print(f"  Best rank-avg:  w={best_rank[0]}  Apr IC={best_rank[2]:+.4f}")

    # Also check Mar
    print("\n  Mar cross-check:")
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        ic_s = results[f"score_{int(w*100)}"].get("2026-03", float("nan"))
        print(f"    w={w:<4}  score IC={ic_s:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    run()
