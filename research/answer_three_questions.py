"""
Compute the three credibility-test answers for the BTC 4h direction system.
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
DIR_FEATS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")


def main():
    df = pd.read_parquet(CACHE)
    labels = build_direction_labels(df, horizon_bars=4, k=0.5)
    df["y_return_4h"] = labels["return_4h"]
    df["y_dir"] = labels["y_dir"]

    with open(DIR_FEATS) as f:
        feats = json.load(f)
    feats = [c for c in feats if c in df.columns]
    print(f"Direction model pruned features: {len(feats)}")
    print(f"Sample range: {df.index.min()} -> {df.index.max()}  (n={len(df)})")

    # ── Q1: strongest single feature OOS IC, train vs test ──────────────
    print("\n" + "=" * 60)
    print("Q1: Strongest single feature, train vs test IC")
    print("=" * 60)
    mid = len(df) // 2
    rows = []
    for c in feats:
        f, y = df[c], df["y_return_4h"]
        m_tr = f.iloc[:mid].notna() & y.iloc[:mid].notna()
        m_te = f.iloc[mid:].notna() & y.iloc[mid:].notna()
        if m_tr.sum() < 100 or m_te.sum() < 100:
            continue
        ic_tr, _ = spearmanr(f.iloc[:mid][m_tr], y.iloc[:mid][m_tr])
        ic_te, _ = spearmanr(f.iloc[mid:][m_te], y.iloc[mid:][m_te])
        m_full = f.notna() & y.notna()
        ic_full, p_full = spearmanr(f[m_full], y[m_full])
        rows.append((c, ic_full, p_full, ic_tr, ic_te))

    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    print(f"\n{'Feature':32s}  {'Full IC':>9s}  {'p':>9s}  {'Train IC':>9s}  {'Test IC':>9s}")
    print("-" * 76)
    for r in rows[:10]:
        print(f"{r[0]:32s}  {r[1]:+9.4f}  {r[2]:9.1e}  {r[3]:+9.4f}  {r[4]:+9.4f}")

    top = rows[0]
    print(f"\n→ Strongest: {top[0]}")
    print(f"  Full-sample IC = {top[1]:+.4f} (p={top[2]:.1e}, n={int((df[top[0]].notna() & df['y_return_4h'].notna()).sum())})")
    print(f"  Train (first half)  IC = {top[3]:+.4f}")
    print(f"  Test  (second half) IC = {top[4]:+.4f}")
    print(f"  Direction stability: {'STABLE' if np.sign(top[3]) == np.sign(top[4]) else 'FLIP'}")

    # ── Q2: walk-forward folds + per-fold IC std ─────────────────────────
    print("\n" + "=" * 60)
    print("Q2: Walk-forward folds + per-fold IC stability")
    print("=" * 60)
    initial_train, test_size, step = 288, 48, 48
    n = len(df)
    folds = []
    start = initial_train
    while start + test_size <= n:
        folds.append((0, start, start, start + test_size))
        start += step
    print(f"\nWalk-forward config: initial_train={initial_train}, test={test_size}, step={step}")
    print(f"Total folds: {len(folds)}")

    target_feat = top[0]
    fold_ics = []
    for tr_s, tr_e, te_s, te_e in folds:
        f = df[target_feat].iloc[te_s:te_e]
        y = df["y_return_4h"].iloc[te_s:te_e]
        m = f.notna() & y.notna()
        if m.sum() < 20:
            continue
        rho, _ = spearmanr(f[m], y[m])
        if not np.isnan(rho):
            fold_ics.append(rho)

    fold_ics = np.array(fold_ics)
    print(f"\nPer-fold IC for {target_feat}:")
    print(f"  Folds with valid IC: {len(fold_ics)}")
    print(f"  Mean IC : {fold_ics.mean():+.4f}")
    print(f"  Std IC  : {fold_ics.std():.4f}")
    print(f"  Min IC  : {fold_ics.min():+.4f}")
    print(f"  Max IC  : {fold_ics.max():+.4f}")
    print(f"  Sign-consistent folds: {(np.sign(fold_ics) == np.sign(fold_ics.mean())).mean():.1%}")
    print(f"  ICIR (mean/std)      : {fold_ics.mean() / fold_ics.std():+.3f}")

    # ── Q3: a simple verifiable formula ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Q3: A simple verifiable feature formula")
    print("=" * 60)
    print("""
Pick: long_liq_exhaustion_4h  (just added, IC=+0.0667 vs y_return_4h)

Formula:
  cg_liq_long = aggregated long liquidations USD per 1h bar
                (Coinglass /futures/liquidation/aggregated-history, exchange_list=Binance)

  cum_4h(t)   = sum(cg_liq_long[t-3 : t])             # 4h rolling sum
  mu_168h(t)  = mean(cum_4h[t-167 : t])                # 168h rolling mean
  sd_168h(t)  = std(cum_4h[t-167 : t])                 # 168h rolling std
  long_liq_exhaustion_4h(t) = (cum_4h(t) - mu_168h(t)) / sd_168h(t)

Target:
  y_return_4h(t) = close(t+4) / close(t) - 1     # 4h forward return

Validate:
  spearmanr(long_liq_exhaustion_4h, y_return_4h)
  Expected on this dataset: IC = +0.0667, p < 1e-4, n ~3900
""")

    # Quick sanity verification
    feat = df["long_liq_exhaustion_4h"] if "long_liq_exhaustion_4h" in df.columns else None
    if feat is not None:
        m = feat.notna() & df["y_return_4h"].notna()
        rho, p = spearmanr(feat[m], df["y_return_4h"][m])
        print(f"Live verify: IC={rho:+.4f}  p={p:.2e}  n={int(m.sum())}")


if __name__ == "__main__":
    main()
