"""
Initiation model v1 — two binary classifiers (long_init / short_init)
with vol-adjusted labels.

Label definition (a) — vol-adjusted:
    long_init(t)  = 1 if  ret_4h(t) / realized_vol_20b(t) >=  +k
    short_init(t) = 1 if  ret_4h(t) / realized_vol_20b(t) <=  -k
    k is auto-tuned so that each class ≈ 5% (total ≈ 10%).

Features: production direction set (+ key_4_only, 98 features).
Walk-forward: initial_train=288, test=48, step=48, purge+embargo=4.

Evaluation:
    - per-fold precision@top {1,2,5,10}%
    - monthly precision@top5% with Wilson CI
    - death test: P(long|is_init) AUC should be > 0.55 — otherwise the
      model only finds "something is happening" but can't pick direction
      (same failure mode as triple-barrier Stage 2 in mistake log)
    - comparison vs current direction baseline (top-5% precision = 0.676)

Kill criteria:
    - top-1% precision >= 0.70
    - monthly CI_lo (top 5%) >= 0.55

Usage:
    python research/initiation_model_v1.py
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
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.direction_features_v2 import ABLATION_GROUPS, filter_available

OUT = Path("research/results/initiation_model_v1.json")

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": 0,
    "early_stopping_rounds": 30,
}

TARGET_PCT_EACH = 0.05    # aim ~5% per class → ~10% total
HORIZON_BARS = 4
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_labels(df: pd.DataFrame, k: float) -> pd.DataFrame:
    close = df["close"].values.astype(float)
    n = len(close)
    ret4 = np.full(n, np.nan)
    for i in range(n - HORIZON_BARS):
        ret4[i] = close[i + HORIZON_BARS] / close[i] - 1
    vol = df["realized_vol_20b"].values.astype(float)
    vol_safe = np.where(vol > 0, vol, np.nan)
    ret_vol_adj = ret4 / vol_safe
    long_init = (ret_vol_adj >= k).astype(float)
    short_init = (ret_vol_adj <= -k).astype(float)
    long_init[np.isnan(ret_vol_adj)] = np.nan
    short_init[np.isnan(ret_vol_adj)] = np.nan
    return pd.DataFrame({
        "ret_4h": ret4,
        "ret_vol_adj": ret_vol_adj,
        "y_long_init": long_init,
        "y_short_init": short_init,
    }, index=df.index)


def find_k_for_target(df: pd.DataFrame, target_each: float) -> float:
    """Binary search k so P(y_long_init=1) ≈ target_each."""
    close = df["close"].values.astype(float)
    n = len(close)
    ret4 = np.full(n, np.nan)
    for i in range(n - HORIZON_BARS):
        ret4[i] = close[i + HORIZON_BARS] / close[i] - 1
    vol = df["realized_vol_20b"].values.astype(float)
    vol_safe = np.where(vol > 0, vol, np.nan)
    ret_vol_adj = ret4 / vol_safe
    valid = ~np.isnan(ret_vol_adj)
    adj = ret_vol_adj[valid]
    # symmetric: pick k so both tails ≈ target_each
    # use upper tail quantile
    k = np.quantile(adj, 1 - target_each)
    k_neg = -np.quantile(adj, target_each)
    return float((k + k_neg) / 2)


def run_binary(df: pd.DataFrame, features: list[str], label_col: str,
                initial_train=288, test_size=48) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=initial_train,
                                   test_size=test_size, step=test_size)
    all_oos = []
    for fold_i, (tr, te) in enumerate(splits):
        train_df = df.iloc[tr]; test_df = df.iloc[te]
        tr_mask = train_df[label_col].notna()
        te_mask = test_df[label_col].notna()
        X_tr = train_df.loc[tr_mask, features].fillna(0)
        y_tr = train_df.loc[tr_mask, label_col].values.astype(int)
        X_te = test_df.loc[te_mask, features].fillna(0)
        y_te = test_df.loc[te_mask, label_col].values.astype(int)
        if len(y_tr) < 50 or len(y_te) < 5:
            continue
        pos_rate = y_tr.mean()
        if pos_rate <= 0 or pos_rate >= 1:
            continue
        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = (1 - pos_rate) / pos_rate
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        prob = model.predict_proba(X_te)[:, 1]
        all_oos.append(pd.DataFrame({
            "prob": prob, "y": y_te,
            "ret_4h": test_df.loc[te_mask, "ret_4h"].values,
            "fold": fold_i,
        }, index=test_df.loc[te_mask].index))
    return pd.concat(all_oos) if all_oos else pd.DataFrame()


def precision_at_k(oos: pd.DataFrame, k_frac: float) -> dict:
    k = max(1, int(round(len(oos) * k_frac)))
    top = oos.nlargest(k, "prob")
    pos = int(top["y"].sum())
    prec = pos / k
    ci_lo, ci_hi = wilson_ci(pos, k)
    # Frequency
    span = (oos.index.max() - oos.index.min()).total_seconds() / 86400
    per_month = k / span * 30 if span > 0 else 0
    # Avg return (for signal validation — top-k should have big directional moves)
    return {
        "k_frac": k_frac, "k_count": k, "n_total": len(oos),
        "precision": prec, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "signals_per_month": per_month,
        "min_prob": float(top["prob"].min()),
        "avg_ret_in_top": float(top["ret_4h"].mean()),
    }


def monthly_precision(oos: pd.DataFrame, k_frac: float) -> dict:
    out = {}
    months = oos.index.strftime("%Y-%m")
    for m in MONTHS:
        mask = months == m
        sub = oos[mask]
        if len(sub) < 50:
            out[m] = {"n": len(sub), "precision": None}
            continue
        k = max(1, int(round(len(sub) * k_frac)))
        top = sub.nlargest(k, "prob")
        pos = int(top["y"].sum())
        lo, hi = wilson_ci(pos, k)
        out[m] = {"n": len(sub), "k": k, "precision": pos / k, "ci_lo": lo, "ci_hi": hi}
    return out


def main():
    print("Loading data...")
    df = load_and_cache_data()
    print(f"  {len(df)} bars, range {df.index[0]} ~ {df.index[-1]}")

    # --- 1. Find k for ~5% each ---
    k = find_k_for_target(df, TARGET_PCT_EACH)
    print(f"\nTuned k = {k:.3f}  (target {TARGET_PCT_EACH:.0%} each class)")

    # --- 2. Build labels ---
    labels = build_labels(df, k)
    df = df.copy()
    for c in labels.columns:
        df[c] = labels[c]
    valid = df["y_long_init"].notna()
    print(f"Label rates: long={df.loc[valid,'y_long_init'].mean():.2%}  "
          f"short={df.loc[valid,'y_short_init'].mean():.2%}  "
          f"total={((df['y_long_init']==1)|(df['y_short_init']==1))[valid].mean():.2%}")

    # Monthly distribution
    print("\nMonthly label rates:")
    months_all = df.index.strftime("%Y-%m")
    for m in MONTHS:
        sub = df[(months_all == m) & valid]
        if len(sub) < 30:
            continue
        print(f"  {m}: n={len(sub):4d}  long={sub['y_long_init'].mean():.2%}  short={sub['y_short_init'].mean():.2%}")

    # --- 3. Features ---
    feats = ABLATION_GROUPS["+ key_4_only"]
    features = filter_available(feats, list(df.columns))
    print(f"\nFeatures: {len(features)} / {len(feats)} requested")

    # --- 4. Train both binary models ---
    print("\n[A] Training long_init classifier (walk-forward)...")
    oos_long = run_binary(df, features, "y_long_init")
    print(f"    OOS preds: {len(oos_long)}  positives: {int(oos_long['y'].sum())}")

    print("[B] Training short_init classifier (walk-forward)...")
    oos_short = run_binary(df, features, "y_short_init")
    print(f"    OOS preds: {len(oos_short)}  positives: {int(oos_short['y'].sum())}")

    # --- 5. Evaluate precision@k for each ---
    print("\n" + "=" * 78)
    print("PRECISION @ TOP-K  (long_init)")
    print("=" * 78)
    print(f"{'k%':>5} {'count':>6} {'prec':>7} {'CI_lo':>7} {'CI_hi':>7} {'/month':>8} {'avg_ret':>9}")
    res_long = []
    for kf in [0.01, 0.02, 0.05, 0.10, 0.20]:
        r = precision_at_k(oos_long, kf)
        res_long.append(r)
        print(f"{kf*100:>4.0f}% {r['k_count']:>6d} {r['precision']:>7.3f} "
              f"{r['ci_lo']:>7.3f} {r['ci_hi']:>7.3f} {r['signals_per_month']:>8.1f} "
              f"{r['avg_ret_in_top']:>+9.4f}")

    print("\n" + "=" * 78)
    print("PRECISION @ TOP-K  (short_init)")
    print("=" * 78)
    print(f"{'k%':>5} {'count':>6} {'prec':>7} {'CI_lo':>7} {'CI_hi':>7} {'/month':>8} {'avg_ret':>9}")
    res_short = []
    for kf in [0.01, 0.02, 0.05, 0.10, 0.20]:
        r = precision_at_k(oos_short, kf)
        res_short.append(r)
        print(f"{kf*100:>4.0f}% {r['k_count']:>6d} {r['precision']:>7.3f} "
              f"{r['ci_lo']:>7.3f} {r['ci_hi']:>7.3f} {r['signals_per_month']:>8.1f} "
              f"{r['avg_ret_in_top']:>+9.4f}")

    # --- 6. PR-AUC + ROC-AUC ---
    y_l = oos_long["y"].values; p_l = oos_long["prob"].values
    y_s = oos_short["y"].values; p_s = oos_short["prob"].values
    print()
    print(f"long_init   ROC-AUC = {roc_auc_score(y_l, p_l):.4f}  "
          f"PR-AUC = {average_precision_score(y_l, p_l):.4f}  "
          f"base rate = {y_l.mean():.4f}")
    print(f"short_init  ROC-AUC = {roc_auc_score(y_s, p_s):.4f}  "
          f"PR-AUC = {average_precision_score(y_s, p_s):.4f}  "
          f"base rate = {y_s.mean():.4f}")

    # --- 7. Monthly precision @ top 5% for both ---
    print("\n" + "=" * 78)
    print("MONTHLY PRECISION @ TOP 5%")
    print("=" * 78)
    mpl = monthly_precision(oos_long, 0.05)
    mps = monthly_precision(oos_short, 0.05)
    print(f"{'month':<9} {'long_prec':>10} {'long_CI':>20} {'short_prec':>12} {'short_CI':>20}")
    for m in MONTHS:
        l = mpl.get(m, {}); s = mps.get(m, {})
        lp = f"{l['precision']:.3f}" if l.get('precision') is not None else "  —  "
        sp = f"{s['precision']:.3f}" if s.get('precision') is not None else "  —  "
        lci = f"[{l['ci_lo']:.3f},{l['ci_hi']:.3f}]" if l.get('precision') is not None else ""
        sci = f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]" if s.get('precision') is not None else ""
        print(f"{m:<9} {lp:>10} {lci:>20} {sp:>12} {sci:>20}")

    # --- 8. Death test: P(long|is_init) ---
    # For each bar where EITHER model fires high, can we distinguish direction?
    # Approach: take top-10% by max(prob_long, prob_short), check accuracy of
    # picking long when prob_long > prob_short vs actual.
    print("\n" + "=" * 78)
    print("DEATH TEST — direction recoverability when both models fire")
    print("=" * 78)
    # Merge on index
    merged = oos_long[["prob", "y", "ret_4h"]].rename(columns={"prob": "p_long", "y": "y_long"})
    merged["p_short"] = oos_short["prob"].reindex(merged.index)
    merged["y_short"] = oos_short["y"].reindex(merged.index)
    merged = merged.dropna()
    merged["max_prob"] = merged[["p_long", "p_short"]].max(axis=1)
    merged["pred_long"] = (merged["p_long"] > merged["p_short"]).astype(int)
    merged["true_long"] = (merged["ret_4h"] > 0).astype(int)

    for kf in [0.02, 0.05, 0.10]:
        k = max(1, int(round(len(merged) * kf)))
        top = merged.nlargest(k, "max_prob")
        dir_acc = (top["pred_long"] == top["true_long"]).mean()
        lo, hi = wilson_ci(int((top["pred_long"] == top["true_long"]).sum()), k)
        # Among top, what fraction actually moved big (hit initiation threshold)
        is_init = ((top["y_long"] == 1) | (top["y_short"] == 1)).mean()
        print(f"  top {kf*100:>3.0f}% ({k:>4d} signals): "
              f"dir_acc={dir_acc:.3f} CI=[{lo:.3f},{hi:.3f}]  "
              f"is_init_rate={is_init:.3f}")

    # --- 9. Kill criteria check ---
    print("\n" + "=" * 78)
    print("KILL CRITERIA CHECK")
    print("=" * 78)
    top1_long = res_long[0]["precision"]
    top1_short = res_short[0]["precision"]
    top5_long_monthly_ci = [mpl[m]["ci_lo"] for m in MONTHS if mpl.get(m, {}).get("ci_lo") is not None]
    top5_short_monthly_ci = [mps[m]["ci_lo"] for m in MONTHS if mps.get(m, {}).get("ci_lo") is not None]
    print(f"  top-1% precision   long={top1_long:.3f}   short={top1_short:.3f}   (target &gt;= 0.70)")
    print(f"  monthly CI_lo@5%   long min={min(top5_long_monthly_ci):.3f}   "
          f"short min={min(top5_short_monthly_ci):.3f}   (target &gt;= 0.55)")
    print()
    print(f"  baseline direction top-5% precision = 0.676 (from topk_precision_sweep.py)")

    # --- 10. Save ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "k_threshold": k,
        "n_features": len(features),
        "n_samples_long": len(oos_long),
        "n_samples_short": len(oos_short),
        "long": {
            "roc_auc": float(roc_auc_score(y_l, p_l)),
            "pr_auc": float(average_precision_score(y_l, p_l)),
            "precision_at_k": res_long,
            "monthly_top5": mpl,
        },
        "short": {
            "roc_auc": float(roc_auc_score(y_s, p_s)),
            "pr_auc": float(average_precision_score(y_s, p_s)),
            "precision_at_k": res_short,
            "monthly_top5": mps,
        },
    }, indent=2, default=str))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
