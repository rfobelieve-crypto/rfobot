"""Re-run V7 vs V7+liq A/B using the PRODUCTION training function.

The previous A/B (train_with_liq_features.py) used a custom early-stopping
setup with eval_set = last 10% of train; that triggered early stopping
too aggressively and produced degenerate baselines (IC ≈ 0, AUC ≈ 0.52)
vs the canonical V7 OOS (IC = 0.17, AUC = 0.59).

This script imports `train_direction_reg_walk_forward` directly from
`train_direction_reg_4h.py` so the comparison uses identical hyperparams,
identical fold splits, identical early-stopping (eval_set = test).
Only feature_names differs.

Output:
  research/results/dual_model/liq_ab_v2_metrics.csv
  research/results/dual_model/liq_ab_v2_oos_base.parquet
  research/results/dual_model/liq_ab_v2_oos_new.parquet
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, RESULTS_DIR
from research.dual_model.direction_features_v2 import FULL_DIRECTION
from research.dual_model.train_direction_reg_4h import (
    train_direction_reg_walk_forward, _compute_metrics,
)
from research.dual_model.train_with_liq_features import (
    add_liquidity_features, LIQ_FEATURES,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("Loading features…")
    df = load_and_cache_data(limit=4000)
    logger.info("Loaded: %d bars × %d cols", *df.shape)

    logger.info("Adding 7 liquidity features…")
    df = add_liquidity_features(df)
    cov = {c: int(df[c].notna().sum()) for c in LIQ_FEATURES}
    logger.info("liq feature coverage: %s", cov)

    logger.info("=" * 70)
    logger.info("Run 1: BASELINE V7 (FULL_DIRECTION only)")
    logger.info("=" * 70)
    oos_base, metrics_base, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective="mse",
    )
    oos_base.to_parquet(RESULTS_DIR / "liq_ab_v2_oos_base.parquet")

    logger.info("=" * 70)
    logger.info("Run 2: NEW V7 + 9 liq features")
    logger.info("=" * 70)
    oos_new, metrics_new, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION + LIQ_FEATURES, objective="mse",
    )
    oos_new.to_parquet(RESULTS_DIR / "liq_ab_v2_oos_new.parquet")

    # Compare
    rows = []
    keys = list(metrics_base.keys())
    print()
    print("=" * 84)
    print(f"{'Metric':28s}  {'BASELINE':>15s}  {'NEW':>15s}  {'Δ':>14s}")
    print("=" * 84)
    for k in keys:
        a, b = metrics_base.get(k), metrics_new.get(k)
        if a is None or b is None:
            continue
        try:
            if isinstance(a, (int, np.integer)):
                a_s, b_s = f"{int(a)}", f"{int(b)}"
                d_s = f"{int(b) - int(a):+d}"
            elif isinstance(a, str):
                a_s, b_s, d_s = a, b, "—"
            else:
                a_s, b_s = f"{a:.5f}", f"{b:.5f}"
                d_s = f"{b - a:+.5f}"
        except Exception:
            a_s, b_s, d_s = str(a), str(b), "—"
        print(f"{k:28s}  {a_s:>15s}  {b_s:>15s}  {d_s:>14s}")
        rows.append({"metric": k, "baseline": a, "new": b})
    print("=" * 84)

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "liq_ab_v2_metrics.csv", index=False)
    logger.info("Wrote metrics → %s", RESULTS_DIR / "liq_ab_v2_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
