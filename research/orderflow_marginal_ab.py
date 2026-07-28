# -*- coding: utf-8 -*-
"""Marginal value of the 30 hourly order-flow features OVER positioning.

Keep-only established positioning (65) as the only group that stands alone
(66% of baseline edge) and orderflow alone as nothing (AUC 0.4995). The open
question: does orderflow add anything ON TOP of positioning, or is its
drop-one contribution (1.33x random) just ensemble padding?

Configs (clean WF, identical harness as feature_search_ab):
  P         positioning only (65)                       — base
  P+OF      positioning + orderflow (95)                — the question
  P+CTRL30  positioning + 30 features drawn from the
            OTHER groups (liquidation/return_lags/
            volatility/time pool, n=41)                 — size-matched control,
            clean by construction (keep-only's RAND_30 was positioning-
            contaminated, ~14/30 expected overlap)
  ALL       all 136                                     — reference

Decision numbers:
  lift(P+OF vs P)        — does adding orderflow help at all?
  lift(P+CTRL30 vs P)    — does adding ANY 30 features help the same way?
  paired (P+OF − P+CTRL30) per fold — the verdict: is orderflow better
                                       than 30 junk features?

Run: python research/orderflow_marginal_ab.py
Out: research/results/orderflow_marginal_ab.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.feature_search_ab import _boot_p, _fold_auc, _per_fold_oos, _pooled  # noqa: E402
from research.dual_model.shared_data import load_and_cache_data  # noqa: E402
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels  # noqa: E402
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available  # noqa: E402
from research.feature_group_ablation import group_of  # noqa: E402

OUT = ROOT / "research/results/orderflow_marginal_ab.json"


def main() -> int:
    print("=" * 74)
    print("  ORDERFLOW MARGINAL A/B — clean WF, base = positioning-only")
    print("=" * 74)

    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    deployed = filter_available(FULL_DIRECTION, list(df.columns))

    groups: dict[str, list[str]] = {}
    for f in deployed:
        groups.setdefault(group_of(f), []).append(f)
    P, OF = groups["positioning"], groups["orderflow"]
    other_pool = [f for f in deployed
                  if group_of(f) not in ("positioning", "orderflow")]
    rng = np.random.default_rng(11)
    ctrl30 = list(rng.choice(other_pool, size=30, replace=False))
    print(f"  bars={len(df)}  P={len(P)}  OF={len(OF)}  "
          f"other_pool={len(other_pool)} -> ctrl30={len(ctrl30)}")

    t0 = time.time()
    base_folds = {k: _fold_auc(v) for k, v in _per_fold_oos(df, P, leaky=False).items()}
    print(f"  P baseline done ({time.time() - t0:.0f}s/config)")

    per_fold: dict[str, dict[int, float]] = {"P": base_folds}
    pooled: dict[str, tuple[float, float]] = {}
    for label, feats in [("P+OF", P + OF), ("P+CTRL30", P + ctrl30),
                         ("ALL", deployed)]:
        folds = _per_fold_oos(df, feats, leaky=False)
        per_fold[label] = {k: _fold_auc(v) for k, v in folds.items()}
        pooled[label] = _pooled(folds)

    def lift(a: str, b: str):
        ks = [k for k in per_fold[a] if k in per_fold[b]
              and np.isfinite(per_fold[a][k]) and np.isfinite(per_fold[b][k])]
        ls = [per_fold[a][k] - per_fold[b][k] for k in ks]
        lo, hi, p = _boot_p(ls)
        return float(np.mean(ls)), lo, hi, p, len(ls)

    base_auc = float(np.nanmean(list(base_folds.values())))
    print(f"\n  P per-fold mean AUC = {base_auc:.4f}")
    results = {"per_fold_mean_auc_P": base_auc, "ctrl30": ctrl30, "comparisons": {}}
    for a, b in [("P+OF", "P"), ("P+CTRL30", "P"), ("P+OF", "P+CTRL30"),
                 ("ALL", "P")]:
        m, lo, hi, p, nf = lift(a, b)
        results["comparisons"][f"{a} vs {b}"] = {
            "per_fold_mean_lift": m, "boot_ci_lo": lo, "boot_ci_hi": hi,
            "boot_p_le_0": p, "n_folds": nf}
        print(f"  {a:>9} vs {b:<9}  per-fold {m:+.5f}  CI [{lo:+.4f},{hi:+.4f}]"
              f"  p(<=0)={p:.3f}")
    for label, (auc, ic) in pooled.items():
        results.setdefault("pooled", {})[label] = {"auc": float(auc), "ic": float(ic)}
        print(f"  pooled {label:<9} AUC {auc:.4f}  IC {ic:+.4f}")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
