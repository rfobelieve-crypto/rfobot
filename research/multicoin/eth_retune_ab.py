"""
V7 multicoin — ETH hyperparameter retune A/B (2026-07-23, follow-up to
eth_direction_gate_a.py's NO-GO).

Question: was ETH's 0.5057 clean AUC (vs BTC 0.5412) because the 136-feature
mechanism genuinely doesn't transfer, or because we blindly reused BTC-tuned
hyperparameters + kept 4 mostly-NaN columns on a thinner/noisier asset?

Tests TWO a-priori (NOT searched/grid-swept) variants against the cached
baseline, using the exact same 4-condition sanity gate from mistake.md
2026-06-02 / feature_search_ab.py (agg lift, per-fold mean lift, frac
positive folds, bootstrap CI) — so an "improvement" has to survive the same
scrutiny that would catch an outlier-driven mirage, not just look better on
one pooled number:

  A. "regularized": max_depth 4->3, min_child_weight 10->20, reg_lambda
     1.0->2.0 — hypothesis: thinner-liquidity ETH has a noisier signal, more
     regularization should reduce overfit-to-noise per fold.
  B. "drop_thin_oi_cm": remove the 4 oi_coin_margin_* columns that were 75%
     NaN (constant-zero after fillna for most of history) — hypothesis:
     these are net noise, not signal, for ETH specifically.
  C. A + B combined.

This is deliberately NOT a hyperparameter grid search (that would just be
the threshold-sweep mistake in a new costume) — three specific, justified
variants, each judged pass/fail against the baseline by the same 4-condition
gate used for BTC's own feature A/B tests.

Run: python research/multicoin/eth_retune_ab.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available
from research.feature_search_ab import BASE_PARAMS, _pooled, _fold_auc, _verdict_block

CACHE = PROJECT_ROOT / "research" / "multicoin" / ".cache" / "eth_features_all.parquet"

THIN_OI_CM_COLS = ["cg_oi_cm_close", "cg_oi_cm_delta", "cg_oi_cm_vs_usd",
                   "cg_oi_cm_delta_zscore"]

VARIANTS = {
    "regularized": dict(BASE_PARAMS, max_depth=3, min_child_weight=20, reg_lambda=2.0),
    "drop_thin_oi_cm": dict(BASE_PARAMS),
    "regularized+drop": dict(BASE_PARAMS, max_depth=3, min_child_weight=20, reg_lambda=2.0),
}
DROP_COLS = {
    "regularized": False,
    "drop_thin_oi_cm": True,
    "regularized+drop": True,
}


def _per_fold_oos_custom(df, features, params):
    """Like feature_search_ab._per_fold_oos(leaky=False) but with a caller-
    supplied params dict instead of the hardcoded module-level BASE_PARAMS."""
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    out = {}
    for fi, (tr, te) in enumerate(splits):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        m_tr = tr_df["y_path_ret_4h"].notna()
        m_te = te_df["y_path_ret_4h"].notna()
        Xtr = tr_df.loc[m_tr, features].fillna(0)
        ytr = tr_df.loc[m_tr, "y_path_ret_4h"].values.astype(float)
        Xte = te_df.loc[m_te, features].fillna(0)
        yte = te_df.loc[m_te, "y_path_ret_4h"].values.astype(float)
        if len(ytr) < 50 or len(yte) < 5:
            continue
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr)
        out[fi] = pd.DataFrame({"pred": m.predict(Xte), "y": yte},
                               index=te_df.loc[m_te].index)
    return out


def main():
    print("=" * 72)
    print("  ETH RETUNE A/B — hyperparameters + thin-coverage column drop")
    print("=" * 72)

    df = pd.read_parquet(CACHE)
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    deployed = filter_available(FULL_DIRECTION, list(df.columns))
    print(f"  ETH bars={len(df)}  features={len(deployed)}")

    print("\n  Building BASELINE (identical to eth_direction_gate_a.py's NO-GO run)...")
    t0 = time.time()
    baseline_folds = _per_fold_oos_custom(df, deployed, dict(BASE_PARAMS))
    b_auc, b_ic = _pooled(baseline_folds)
    print(f"  baseline: pooled AUC={b_auc:.4f} IC={b_ic:+.4f} ({time.time()-t0:.0f}s)")

    for label, params in VARIANTS.items():
        feats = [f for f in deployed if not (DROP_COLS[label] and f in THIN_OI_CM_COLS)]
        n_dropped = len(deployed) - len(feats)
        t0 = time.time()
        new_folds = _per_fold_oos_custom(df, feats, params)
        n_auc, n_ic = _pooled(new_folds)
        verdict = _verdict_block(baseline_folds, new_folds, "auc")
        cond = verdict["conditions"]
        print(f"\n  --- {label} (dropped {n_dropped} cols, "
              f"max_depth={params['max_depth']} min_child_weight={params['min_child_weight']} "
              f"reg_lambda={params['reg_lambda']}) ---")
        print(f"    pooled AUC {b_auc:.4f} -> {n_auc:.4f}  (agg_lift={verdict['agg_lift']:+.4f})  "
              f"IC {b_ic:+.4f} -> {n_ic:+.4f}")
        print(f"    mean_fold_lift={verdict['mean_fold_lift']:+.5f}  frac_pos={verdict['frac_pos']:.2f}  "
              f"boot_p_le0={verdict['boot_p_le0']:.4f}  ({time.time()-t0:.0f}s)")
        print(f"    4-cond: agg>0.005={cond['c1_agg']} mean>0.001={cond['c2_mean']} "
              f"fpos>0.55={cond['c3_frac']} bootp<0.05={cond['c4_boot']}  "
              f"=> {'REAL IMPROVEMENT' if verdict['DEPLOY'] else 'no significant lift'}")
        print(f"    absolute vs BTC 0.5412 gate(>=0.54): "
              f"{'PASS' if n_auc >= 0.54 else 'still FAIL'} ({n_auc:.4f})")


if __name__ == "__main__":
    main()
