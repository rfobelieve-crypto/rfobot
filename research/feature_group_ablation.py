# -*- coding: utf-8 -*-
"""Where is the edge? — clean walk-forward GROUP ABLATION on the 136 direction features.

Motivation: XGBoost gain importance on the production model is almost flat
(top-5 features = 7.5% of total; 43 features needed to reach 50%). That tells
us where the model spends its splits, NOT where the predictive edge lives —
gain is in-sample and splits credit arbitrarily among correlated features.

The only way to answer "which features carry the edge" is to remove a group and
see whether CLEAN out-of-sample AUC actually falls. This runs that test.

Design notes:
  - CLEAN walk-forward only (no early-stop-on-test). The leaky path inflates
    AUC 0.541 -> 0.592 (mistake.md 2026-06-19) and would drown the ablation
    deltas we are trying to read.
  - Reuses `_per_fold_oos` / `_fold_auc` / `_pooled` / `_boot_p` from
    feature_search_ab.py so the split scheme, params and bootstrap are
    byte-identical to the harness the project already trusts.
  - Sign convention: `lift = ablated - baseline`. A group that CARRIES edge
    produces a NEGATIVE lift when dropped. A group that is pure dilution
    produces zero or positive lift.
  - Groups are semantic, not prefix-based: `return_lag_*` are price momentum
    (not order flow) and `cg_taker_*` are order flow (not "other"), so a
    naive prefix split misattributes both.
  - RANDOM_CTRL is a size-matched negative control (30 random features, fixed
    seed). If dropping 30 random features hurts as much as dropping the 30
    order-flow features, no group-level claim from this run means anything.

Run: python research/feature_group_ablation.py
Out: research/results/feature_group_ablation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.feature_search_ab import _boot_p, _fold_auc, _per_fold_oos, _pooled  # noqa: E402
from research.dual_model.shared_data import load_and_cache_data  # noqa: E402
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels  # noqa: E402
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available  # noqa: E402

OUT = ROOT / "research/results/feature_group_ablation.json"

# ── Semantic groups ────────────────────────────────────────────────────────
ORDERFLOW_EXPLICIT = {
    "taker_delta_std_24h", "taker_delta_ma_24h", "cvd_persistence_12h",
    "taker_delta_ratio", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
    "cg_taker_delta_slope_4h", "cg_taker_buy", "cg_taker_delta_accel",
    "cg_taker_delta_mom_1h", "cg_taker_delta_zscore",
    "cg_spot_futures_cvd_divergence", "impact_asymmetry",
    "impact_asymmetry_zscore", "post_absorb_breakout", "post_absorb_breakout_z",
    "abs_completion", "abs_completion_z", "flow_trend_score",
}
LIQUIDATION_EXPLICIT = {"long_liq_exhaustion_4h", "squeeze_proxy"}


def group_of(f: str) -> str:
    if f in ORDERFLOW_EXPLICIT or f.startswith(("cg_fcvd", "cg_scvd")):
        return "orderflow"
    if f.startswith("return_lag") or f == "return_kurtosis":
        return "return_lags"
    if f.startswith(("vol_", "realized", "atr", "quote_vol")):
        return "volatility"
    if f.startswith(("hour", "weekday", "dow")):
        return "time"
    if f in LIQUIDATION_EXPLICIT or f.startswith("cg_liq") or f.startswith("fragility"):
        return "liquidation"
    return "positioning"


def main() -> int:
    print("=" * 74)
    print("  FEATURE GROUP ABLATION — clean WF, lift = ablated - baseline")
    print("  (negative lift => that group CARRIES edge)")
    print("=" * 74)

    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    deployed = filter_available(FULL_DIRECTION, list(df.columns))

    groups: dict[str, list[str]] = {}
    for f in deployed:
        groups.setdefault(group_of(f), []).append(f)
    rng = np.random.default_rng(42)
    groups["RANDOM_CTRL"] = list(
        rng.choice(deployed, size=len(groups["orderflow"]), replace=False))

    print(f"  bars={len(df)}  deployed={len(deployed)}")
    for g, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    {g:<14} n={len(fs)}")

    t0 = time.time()
    base_folds = _per_fold_oos(df, deployed, leaky=False)
    base_auc, base_ic = _pooled(base_folds)
    per_fold_base = {k: _fold_auc(v) for k, v in base_folds.items()}
    dt = time.time() - t0
    print(f"\n  BASELINE  clean AUC={base_auc:.4f}  IC={base_ic:.4f}  "
          f"folds={len(base_folds)}  ({dt:.0f}s)")
    print(f"  estimated total runtime ~{dt * (len(groups) + 1) / 60:.0f} min\n")

    results = []
    for g in sorted(groups, key=lambda k: -len(groups[k])):
        drop = set(groups[g])
        feats = [f for f in deployed if f not in drop]
        folds = _per_fold_oos(df, feats, leaky=False)
        auc, ic = _pooled(folds)
        lifts = [_fold_auc(folds[k]) - per_fold_base[k]
                 for k in folds if k in per_fold_base]
        lifts = [x for x in lifts if np.isfinite(x)]
        lo, hi, p_le0 = _boot_p(lifts)
        rec = {
            "group": g, "n_dropped": len(drop), "n_remaining": len(feats),
            "pooled_auc": float(auc), "pooled_auc_lift": float(auc - base_auc),
            "pooled_ic": float(ic), "pooled_ic_lift": float(ic - base_ic),
            "per_fold_mean_lift": float(np.mean(lifts)),
            "frac_folds_worse": float(np.mean([x < 0 for x in lifts])),
            "boot_ci_lo": lo, "boot_ci_hi": hi,
            # p(lift<=0) high => dropping reliably HURTS => group carries edge
            "boot_p_lift_le_0": p_le0,
            "carries_edge": bool(hi < 0),
        }
        results.append(rec)
        print(f"  drop {g:<14} (n={len(drop):>3})  AUC {auc:.4f} "
              f"({auc - base_auc:+.4f})  per-fold {np.mean(lifts):+.5f}  "
              f"CI [{lo:+.4f},{hi:+.4f}]  "
              f"{'<< CARRIES EDGE' if hi < 0 else ''}")

    payload = {
        "baseline": {"clean_auc": float(base_auc), "clean_ic": float(base_ic),
                     "n_features": len(deployed), "n_folds": len(base_folds)},
        "ablations": results,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("\n  READ: a group only 'carries edge' if its bootstrap CI is entirely")
    print("  below 0. Anything overlapping 0 is indistinguishable from dropping")
    print("  RANDOM_CTRL — i.e. that group is not demonstrably doing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
