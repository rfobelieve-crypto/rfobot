"""
v9 statistical robustness tests (Part A of validation).

Four tests to rule out luck/lookahead/overfit before forward window:

  A1. Permutation test — shuffle predictions, recompute "filtered cohort" WR.
       If our 56.6% > 95th percentile of permuted distribution, signal is
       not from random alignment of predictions to outcomes.

  A2. Out-of-time per-month WR — show forward-out-of-fold WR per month
       in walk-forward.  Look for consistency across months / regimes.
       If WR concentrated in one month -> likely regime artifact.

  A3. Hyperparameter sensitivity — retrain v9 with different random seeds,
       see if same horizon/threshold still yields >51% WR.  If 5 seeds
       agree on edge -> robust to seed.  If only one lucky seed -> noise.

  A4. Random selection baseline — randomly pick 53 SHORT bars (matching
       cohort size), compute OHLC trade outcomes, report distribution.
       Our actual 56.6% should be in the upper tail.

Inputs:
  research/results/dual_model/direction_v9_winrate_H8_oos.parquet
  research/dual_model/.cache/features_all.parquet
  research/dual_model/.cache/labels_winrate_TP50_SL30_H8.parquet
  market_data/raw_data/binance_klines_1h.parquet
"""
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import xgboost as xgb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OOS_PATH = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_v9_winrate_H8_oos.parquet"
FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
LABEL_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "labels_winrate_TP50_SL30_H8.parquet"
FEATURE_COLS_FILE = (
    PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
    / "direction_feature_cols.json"
)
KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

THRESHOLD = 0.65
TP_DIST = 0.005
SL_DIST = 0.003
BE_WR_NO_COST = 0.375
BE_WR_TAKER = 0.51
BE_WR_MAKER = 0.414


def _load_oos():
    return pd.read_parquet(OOS_PATH)


def _load_klines():
    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
    return klines


# ─── Test A1: Permutation test ─────────────────────────────────────────────

def test_a1_permutation(oos: pd.DataFrame, n_iterations: int = 2000):
    print("\n" + "="*90)
    print("A1: Permutation test — shuffle p_short_win, recompute cohort WR")
    print("="*90)

    actual_p = oos["p_short_win"].values
    actual_y = oos["y_short_win"].values

    # Actual WR
    sel = actual_p >= THRESHOLD
    actual_wr = actual_y[sel].mean()
    actual_n = sel.sum()
    print(f"  Actual: n={actual_n} cohort, WR={actual_wr*100:.2f}%")

    # Permute predictions: pair shuffled p with actual y
    rng = np.random.default_rng(42)
    permuted_wrs = []
    for _ in range(n_iterations):
        perm_p = rng.permutation(actual_p)
        sel_p = perm_p >= THRESHOLD
        if sel_p.sum() < 5:
            continue
        permuted_wrs.append(actual_y[sel_p].mean())
    permuted_wrs = np.array(permuted_wrs)

    p_value = (permuted_wrs >= actual_wr).mean()
    pct_95 = np.percentile(permuted_wrs, 95)
    pct_99 = np.percentile(permuted_wrs, 99)
    print(f"  Permuted (n={len(permuted_wrs)} iterations):")
    print(f"    mean WR:           {permuted_wrs.mean()*100:.2f}%")
    print(f"    95th pct WR:       {pct_95*100:.2f}%")
    print(f"    99th pct WR:       {pct_99*100:.2f}%")
    print(f"    P(permuted >= actual {actual_wr*100:.2f}%) = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  PASS p < 0.05 — predictions ARE informative (significant)")
    else:
        print(f"  FAIL p >= 0.05 — predictions NOT distinguishable from random shuffle")
    return {"p_value": float(p_value), "actual_wr": float(actual_wr),
            "permuted_mean_wr": float(permuted_wrs.mean()),
            "permuted_95th": float(pct_95)}


# ─── Test A2: Per-month / time split WR ─────────────────────────────────────

def test_a2_temporal(oos: pd.DataFrame):
    print("\n" + "="*90)
    print("A2: Per-month WR breakdown — is edge consistent or concentrated?")
    print("="*90)
    df = oos[oos["p_short_win"] >= THRESHOLD].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["month"] = df["ts"].dt.to_period("M")

    print(f"  {'month':<10} {'n':>4} {'WR':>7} {'vs 51% BE':>10} {'95% CI':>20}")
    print("-" * 60)
    results = {}
    for month, g in df.groupby("month"):
        n = len(g)
        wins = int(g["y_short_win"].sum())
        wr = wins / n if n > 0 else 0
        if n >= 5:
            ci = binomtest(wins, n).proportion_ci(confidence_level=0.95)
            ci_str = f"[{ci.low*100:.1f}, {ci.high*100:.1f}]"
        else:
            ci_str = "(n<5)"
        edge_pp = (wr - 0.51) * 100
        print(f"  {str(month):<10} {n:>4} {wr*100:>5.1f}%  {edge_pp:>+8.1f}pp  {ci_str:>20}")
        results[str(month)] = {"n": n, "wr": float(wr), "ci_str": ci_str}

    above_be = sum(1 for r in results.values() if r["wr"] > 0.51 and r["n"] >= 5)
    n_months = sum(1 for r in results.values() if r["n"] >= 5)
    print(f"\n  Months above 51% BE: {above_be}/{n_months}")
    if above_be == n_months:
        print(f"  PASS Consistent edge across all sampled months")
    elif above_be >= n_months * 0.6:
        print(f"  o Majority of months above BE — moderate consistency")
    else:
        print(f"  FAIL Edge concentrated in <60% of months — likely regime-conditional")
    return results


# ─── Test A3: Hyperparameter sensitivity ────────────────────────────────────

BASE_PARAMS = dict(
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    early_stopping_rounds=30,
    objective="binary:logistic", eval_metric="auc",
)


def _train_one_seed(seed: int) -> np.ndarray:
    """Walk-forward train v9 SHORT side with given random_state, return OOS P."""
    features_df = pd.read_parquet(FEATURES_CACHE)
    labels_df = pd.read_parquet(LABEL_CACHE)
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_convert("UTC").tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_convert("UTC").tz_localize(None)
    df = features_df.join(
        labels_df[["y_short_win"]], how="inner"
    ).dropna(subset=["y_short_win"])

    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_short_win"].values.astype(int)

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                   step=48, purge=4, embargo=4)
    oos_p = np.full(len(df), np.nan)
    params = dict(BASE_PARAMS, random_state=seed)

    for tr_idx, te_idx in splits:
        tr_arr = np.array(tr_idx)
        cut = int(len(tr_arr) * 0.85)
        tr_in, tr_val = tr_arr[:cut], tr_arr[cut:]
        n_pos = int(y[tr_in].sum())
        n_neg = len(tr_in) - n_pos
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_in], y[tr_in], eval_set=[(X[tr_val], y[tr_val])], verbose=False)
        oos_p[te_idx] = model.predict_proba(X[te_idx])[:, 1]
    return oos_p, y


def test_a3_seed_sensitivity():
    print("\n" + "="*90)
    print("A3: Hyperparameter sensitivity — retrain v9 SHORT with 5 seeds")
    print("="*90)
    seeds = [42, 1, 7, 123, 2026]
    cohort_results = []
    seed_predictions = {}
    for seed in seeds:
        logger.info("  Training with seed=%d...", seed)
        p, y = _train_one_seed(seed)
        mask = np.isfinite(p)
        p_v = p[mask]
        y_v = y[mask]
        sel = p_v >= THRESHOLD
        n = sel.sum()
        wr = y_v[sel].mean() if n > 0 else 0
        auc = roc_auc_score(y_v, p_v)
        cohort_results.append({"seed": seed, "n": int(n), "wr": float(wr),
                                "auc": float(auc), "p_v": p_v, "y_v": y_v})
        seed_predictions[seed] = (p_v, mask)
        print(f"  seed={seed:>5}  AUC={auc:.4f}  n>=0.65: {n:>3}  WR={wr*100:>5.1f}%")

    wrs = np.array([r["wr"] for r in cohort_results])
    n_above_be = sum(r["wr"] >= BE_WR_TAKER for r in cohort_results)
    print(f"\n  WR across seeds: mean={wrs.mean()*100:.2f}%  std={wrs.std()*100:.2f}%  "
          f"min={wrs.min()*100:.1f}%  max={wrs.max()*100:.1f}%")
    print(f"  Seeds above 51% BE: {n_above_be}/5")

    # Cohort overlap (Jaccard)
    cohorts = [set(np.where(seed_predictions[s][0] >= THRESHOLD)[0])
               for s in seeds]
    jaccards = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = cohorts[i], cohorts[j]
            j_idx = len(a & b) / max(len(a | b), 1)
            jaccards.append(j_idx)
    print(f"  Cohort Jaccard (pairwise): mean={np.mean(jaccards):.3f}  "
          f"min={min(jaccards):.3f}  max={max(jaccards):.3f}")
    if np.mean(jaccards) > 0.7:
        print(f"  PASS High cohort overlap — seeds agree on which bars are 'high P'")
    elif np.mean(jaccards) > 0.5:
        print(f"  o Moderate cohort overlap — partial consensus")
    else:
        print(f"  FAIL Low cohort overlap — model picks differ widely between seeds")
    return cohort_results


# ─── Test A4: Random selection baseline ─────────────────────────────────────

def test_a4_random_baseline(oos: pd.DataFrame, klines: pd.DataFrame,
                              actual_cohort_size: int = 53,
                              n_iterations: int = 2000):
    print("\n" + "="*90)
    print(f"A4: Random selection baseline — pick {actual_cohort_size} random SHORT bars")
    print("="*90)
    # Build OHLC outcomes for every bar's potential short trade
    # (we already have these via y_short_win column)
    oos = oos.copy()
    oos["ts"] = pd.to_datetime(oos["ts"])
    if oos["ts"].dt.tz is not None:
        oos["ts"] = oos["ts"].dt.tz_convert("UTC").dt.tz_localize(None)

    all_y = oos["y_short_win"].values
    n = len(all_y)
    rng = np.random.default_rng(42)

    random_wrs = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=actual_cohort_size, replace=False)
        random_wrs.append(all_y[idx].mean())
    random_wrs = np.array(random_wrs)

    actual_wr = all_y[oos["p_short_win"].values >= THRESHOLD].mean()
    p_value = (random_wrs >= actual_wr).mean()
    pct_95 = np.percentile(random_wrs, 95)
    pct_99 = np.percentile(random_wrs, 99)

    print(f"  Actual filtered cohort WR (P>=0.65): {actual_wr*100:.2f}%")
    print(f"  Random cohort (n={actual_cohort_size}) WR distribution over {n_iterations} iters:")
    print(f"    mean:    {random_wrs.mean()*100:.2f}%")
    print(f"    95th:    {pct_95*100:.2f}%")
    print(f"    99th:    {pct_99*100:.2f}%")
    print(f"    P(random >= actual) = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  PASS Actual WR significantly above random selection")
    else:
        print(f"  FAIL Actual WR not significantly above random")
    return {"actual_wr": float(actual_wr), "random_mean": float(random_wrs.mean()),
            "p_value": float(p_value)}


def main():
    print("="*90)
    print(f"v9 H=8 SHORT @ P>={THRESHOLD} — Statistical robustness tests")
    print("="*90)
    print(f"  BE WR (no cost): {BE_WR_NO_COST*100:.1f}%")
    print(f"  BE WR (taker 13bp):  {BE_WR_TAKER*100:.1f}%")
    print(f"  BE WR (maker 5bp):   {BE_WR_MAKER*100:.1f}%")

    oos = _load_oos()
    klines = _load_klines()
    logger.info("OOS loaded: n=%d", len(oos))

    a1 = test_a1_permutation(oos)
    a2 = test_a2_temporal(oos)
    a3 = test_a3_seed_sensitivity()
    a4 = test_a4_random_baseline(oos, klines)

    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"  A1 permutation test p-value: {a1['p_value']:.4f}  "
          f"({'PASS' if a1['p_value'] < 0.05 else 'FAIL'})")
    n_consistent = sum(1 for r in a2.values() if r["n"] >= 5 and r["wr"] > 0.51)
    n_total_months = sum(1 for r in a2.values() if r["n"] >= 5)
    print(f"  A2 per-month consistency: {n_consistent}/{n_total_months} above 51% BE  "
          f"({'PASS' if n_consistent >= n_total_months * 0.6 else 'WARN'})")
    seeds_above = sum(1 for r in a3 if r["wr"] >= BE_WR_TAKER)
    print(f"  A3 seed sensitivity: {seeds_above}/5 seeds above 51% BE  "
          f"({'PASS' if seeds_above >= 3 else 'FAIL'})")
    print(f"  A4 random baseline p-value: {a4['p_value']:.4f}  "
          f"({'PASS' if a4['p_value'] < 0.05 else 'FAIL'})")


if __name__ == "__main__":
    main()
