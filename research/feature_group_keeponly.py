# -*- coding: utf-8 -*-
"""Keep-only counterpart to feature_group_ablation.py — "where IS the edge".

Drop-one-group answers "what breaks if this leaves". It cannot separate a group
that carries signal from a group that merely pads the ensemble, because dropping
30 RANDOM features already costs -0.012 per-fold AUC. Keep-only asks the direct
question instead: train on ONLY this group and see how much of the baseline edge
survives.

Size-matched random controls are the whole point. A 12-feature model built from
volatility+time must be compared against a 12-feature model built from 12 random
features drawn from the same pool — otherwise "small model does okay" is just a
statement about model size, not about which features matter.

Edge is measured as AUC - 0.5 (coin flip), so "retained %" is the fraction of
the baseline's actual edge that the subset reproduces.

Run: python research/feature_group_keeponly.py
Out: research/results/feature_group_keeponly.json
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

OUT = ROOT / "research/results/feature_group_keeponly.json"


def main() -> int:
    print("=" * 74)
    print("  KEEP-ONLY — clean WF. How much baseline edge does each group alone hold?")
    print("=" * 74)

    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    deployed = filter_available(FULL_DIRECTION, list(df.columns))

    groups: dict[str, list[str]] = {}
    for f in deployed:
        groups.setdefault(group_of(f), []).append(f)

    rng = np.random.default_rng(7)
    sets = {
        "ALL (baseline)": deployed,
        "volatility+time": groups["volatility"] + groups["time"],
        "volatility": groups["volatility"],
        "orderflow": groups["orderflow"],
        "positioning": groups["positioning"],
    }
    # size-matched random controls for the two small subsets that matter
    for name, k in [("RAND_12", 12), ("RAND_8", 8), ("RAND_30", 30)]:
        sets[name] = list(rng.choice(deployed, size=k, replace=False))

    t0 = time.time()
    base_folds = _per_fold_oos(df, deployed, leaky=False)
    base_auc, _ = _pooled(base_folds)
    per_fold_base = {k: _fold_auc(v) for k, v in base_folds.items()}
    base_edge = base_auc - 0.5
    print(f"  baseline clean AUC={base_auc:.4f}  edge={base_edge:+.4f}  "
          f"({time.time() - t0:.0f}s/config, ~{time.time() - t0} * {len(sets)} total)\n")

    results = []
    for label, feats in sets.items():
        folds = _per_fold_oos(df, feats, leaky=False)
        auc, ic = _pooled(folds)
        lifts = [_fold_auc(folds[k]) - per_fold_base[k]
                 for k in folds if k in per_fold_base]
        lifts = [x for x in lifts if np.isfinite(x)]
        lo, hi, _ = _boot_p(lifts)
        rec = {"set": label, "n": len(feats), "pooled_auc": float(auc),
               "pooled_ic": float(ic), "edge": float(auc - 0.5),
               "edge_retained_pct": float(100 * (auc - 0.5) / base_edge),
               "per_fold_lift_vs_all": float(np.mean(lifts)),
               "boot_ci_lo": lo, "boot_ci_hi": hi}
        results.append(rec)
        print(f"  {label:<16} n={len(feats):>3}  AUC {auc:.4f}  "
              f"edge {auc - 0.5:+.4f}  保留 {rec['edge_retained_pct']:5.1f}%  "
              f"vs-ALL per-fold {np.mean(lifts):+.5f} CI[{lo:+.4f},{hi:+.4f}]")

    OUT.write_text(json.dumps({"baseline_auc": float(base_auc),
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
