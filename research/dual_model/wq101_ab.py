"""V7 baseline vs V7 + WQ101 adapted candidates — ensemble A/B retrain.

6 strong candidates from `research/wq101_adapted.py` (cond_IC > 0.03 AND
frac_pos > 65% across 50+ folds):
    alpha008  +0.076  68.5%   (open×returns 5/10d reversal)
    alpha047  +0.060  72.7%   (vwap momentum × volume)
    alpha005  +0.045  72.7%   (vwap deviation)
    alpha020  +0.042  72.7%   (gap analysis open vs prior H/L/C)
    alpha024  +0.037  87.7%   (100d SMA trend — most stable)
    alpha084  +0.033  73.2%   (vwap_max delta × close delta)

Discipline (mistake.md 2026-06-01):
  - Use PROD train_direction_reg_walk_forward, identical hyperparams
  - Threshold: +0.005 sign_AUC OR +0.005 IC lift to ship
  - Anything below threshold → confirms V7 has saturated for BTC alpha
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
    train_direction_reg_walk_forward,
)
from research.wq101_adapted import WQ101

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


CANDIDATE_ALPHAS = ("alpha008", "alpha047", "alpha005",
                    "alpha020", "alpha024", "alpha084")


def add_wq_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 candidate alphas from OHLCV and join to df."""
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
    wq = WQ101(ohlcv)
    for name in CANDIDATE_ALPHAS:
        series = getattr(wq, name)()
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        # Robustify: inf → nan, then quantile clip + hard clip to ±1e6
        # (alpha084 can produce 10^150 from negative exponents — guard hard)
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() > 100:
            s = series.dropna()
            lo, hi = s.quantile(0.001), s.quantile(0.999)
            series = series.clip(lo, hi)
        series = series.clip(-1e6, 1e6)  # hard guard against any remaining blow-ups
        df[name] = series.reindex(df.index)
        cov = df[name].notna().sum()
        logger.info("%s coverage: %d/%d  range=[%.3g, %.3g]",
                    name, cov,
                    len(df),
                    df[name].min() if cov else float('nan'),
                    df[name].max() if cov else float('nan'))
    return df


def main() -> int:
    logger.info("Loading features…")
    df = load_and_cache_data(limit=4000)
    logger.info("Loaded: %d bars × %d cols", *df.shape)

    logger.info("Adding %d WQ101 candidate alphas…", len(CANDIDATE_ALPHAS))
    df = add_wq_alphas(df)

    logger.info("=" * 70)
    logger.info("Run 1: BASELINE V7 (FULL_DIRECTION only)")
    logger.info("=" * 70)
    oos_base, metrics_base, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective="mse",
    )
    oos_base.to_parquet(RESULTS_DIR / "wq101_ab_oos_base.parquet")

    logger.info("=" * 70)
    logger.info("Run 2: NEW V7 + %d WQ101 candidates", len(CANDIDATE_ALPHAS))
    logger.info("=" * 70)
    oos_new, metrics_new, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION + list(CANDIDATE_ALPHAS), objective="mse",
    )
    oos_new.to_parquet(RESULTS_DIR / "wq101_ab_oos_new.parquet")

    rows = []
    print()
    print("=" * 84)
    print(f"{'Metric':28s}  {'BASELINE':>15s}  {'NEW':>15s}  {'Δ':>14s}")
    print("=" * 84)
    for k in metrics_base.keys():
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

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "wq101_ab_metrics.csv", index=False)
    logger.info("Wrote metrics → %s", RESULTS_DIR / "wq101_ab_metrics.csv")

    # Final verdict
    auc_lift = (metrics_new.get("sign_auc", 0) - metrics_base.get("sign_auc", 0))
    ic_lift = (metrics_new.get("spearman_ic_mean", 0) - metrics_base.get("spearman_ic_mean", 0))

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"sign_AUC lift: {auc_lift:+.5f}  (threshold +0.005)")
    print(f"IC lift:       {ic_lift:+.5f}  (threshold +0.005)")
    if auc_lift > 0.005 or ic_lift > 0.005:
        print("→ DEPLOY: WQ101 candidates bring measurable lift to V7 ensemble")
    elif auc_lift > 0.001 and ic_lift > 0.001:
        print("→ MARGINAL: small but consistent lift — consider larger sample")
    else:
        print("→ NO-GO: V7 has saturated BTC alpha. Need exotic data source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
