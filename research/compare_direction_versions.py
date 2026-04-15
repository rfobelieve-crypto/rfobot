"""
Head-to-head WR comparison across direction-model generations.

Every candidate is evaluated on its own WF OOS parquet (all strictly true OOS,
77-fold walk-forward with purge+embargo where applicable). Metrics:
  - Spearman IC
  - Sign AUC
  - Top-5% rolling symmetric WR (production-style decoding)
  - Wilson CI on the top-5% WR

Candidates (as available on disk):
  A. Binary baseline (direction_oos_baseline_old)
  B. Binary full_expanded (direction_oos_full_expanded) — current production
  C. Binary + bfx_margin
  D. Binary + cb_premium
  E. Binary + spot_fut_cvd
  F. Binary + liq_fragility
  G. Init head (strict walk-forward) — strict_walkforward_oos
  H. Direction REG mse (direction_reg_oos_mse)
  I. Direction REG huber (direction_reg_oos_huber)

For binary OOS files, prob_up is converted to "pred_ret" as (prob_up - 0.5) so
the same top-5% rolling rule applies symmetrically across all candidates.
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "research" / "results"
DUAL_DIR = RESULTS_DIR / "dual_model"

ROLL = 500
STRONG_FRAC = 0.05


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rolling_top5(score, y):
    """Symmetric rolling top-5% fires on `score` (signed)."""
    q_up = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        1 - STRONG_FRAC / 2).values
    q_dn = pd.Series(score).rolling(ROLL, min_periods=100).quantile(
        STRONG_FRAC / 2).values
    valid = ~np.isnan(q_up)
    long_fire = valid & (score >= q_up)
    short_fire = valid & (score <= q_dn)
    fire = long_fire | short_fire
    win = (long_fire & (y > 0)) | (short_fire & (y < 0))
    return fire, win


def evaluate(name, oos, score_col, y_col):
    if not len(oos):
        return None
    score = oos[score_col].values.astype(float)
    y = oos[y_col].values.astype(float)

    mask = np.isfinite(score) & np.isfinite(y)
    score = score[mask]
    y = y[mask]
    if len(score) < 500:
        return None

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    ic = float(spearmanr(score, y).correlation)
    auc_mask = y != 0
    auc = (float(roc_auc_score((y[auc_mask] > 0).astype(int), score[auc_mask]))
           if auc_mask.sum() > 50 else float("nan"))

    fire, win = rolling_top5(score, y)
    n_fire = int(fire.sum())
    n_win = int(win.sum())
    wr = n_win / n_fire if n_fire else float("nan")
    lo, hi = wilson_ci(n_win, n_fire)

    return dict(
        name=name, n_oos=len(score),
        ic=ic, auc=auc,
        n_fire=n_fire, wr=wr, ci_lo=lo, ci_hi=hi,
    )


def load_binary(path: Path, name: str):
    """Binary OOS has columns: prob_up, y_true, return_4h, fold."""
    if not path.exists():
        return None
    oos = pd.read_parquet(path)
    # Convert prob_up → signed score centered at 0
    oos = oos.copy()
    oos["signed_score"] = oos["prob_up"] - 0.5
    # Use return_4h (endpoint) as outcome — binary models were trained with
    # path-return labels but endpoint return is what the WR rule evaluates.
    # If the column is absent fall back to y_true as ±1.
    y_col = "return_4h" if "return_4h" in oos.columns else "y_true"
    return evaluate(name, oos, "signed_score", y_col)


def load_reg(path: Path, name: str):
    if not path.exists():
        return None
    oos = pd.read_parquet(path)
    return evaluate(name, oos, "pred_ret", "y_path_ret_4h")


def load_strict_init(path: Path, name: str):
    """strict_walkforward_oos has p_long/p_short/mag_pct. Build a signed score
    as (p_long - p_short). fwd_ret is endpoint 4h return used for WR."""
    if not path.exists():
        return None
    oos = pd.read_parquet(path)
    # Compute forward return from close if present; else cannot eval
    if "fwd_ret" in oos.columns:
        y_col = "fwd_ret"
    else:
        # Derive endpoint fwd ret from close if stored
        if "close" in oos.columns:
            oos = oos.copy()
            oos["fwd_ret_calc"] = (oos["close"].shift(-4) / oos["close"] - 1)
            y_col = "fwd_ret_calc"
        else:
            return None
    oos = oos.copy()
    oos["signed_score"] = oos["p_long"] - oos["p_short"]
    return evaluate(name, oos, "signed_score", y_col)


def main():
    rows = []

    candidates = [
        ("Binary baseline", DUAL_DIR / "direction_oos_baseline_old.parquet", "binary"),
        ("Binary full_expanded (prev prod)", DUAL_DIR / "direction_oos_full_expanded.parquet", "binary"),
        ("Binary + bfx_margin", DUAL_DIR / "direction_oos_plus_bfx_margin.parquet", "binary"),
        ("Binary + cb_premium", DUAL_DIR / "direction_oos_plus_cb_premium.parquet", "binary"),
        ("Binary + spot_fut_cvd", DUAL_DIR / "direction_oos_plus_spot_fut_cvd.parquet", "binary"),
        ("Binary + liq_fragility", DUAL_DIR / "direction_oos_plus_liq_fragility.parquet", "binary"),
        ("Init strict WF", RESULTS_DIR / "strict_walkforward_oos.parquet", "strict_init"),
        ("REG mse (NEW)", DUAL_DIR / "direction_reg_oos_mse.parquet", "reg"),
        ("REG huber (NEW)", DUAL_DIR / "direction_reg_oos_huber.parquet", "reg"),
    ]

    for name, path, kind in candidates:
        if kind == "binary":
            r = load_binary(path, name)
        elif kind == "reg":
            r = load_reg(path, name)
        elif kind == "strict_init":
            r = load_strict_init(path, name)
        else:
            r = None
        if r is None:
            print(f"  [skip] {name}: missing or incompatible ({path.name})")
            continue
        rows.append(r)

    if not rows:
        print("No candidates evaluated.")
        return

    print("\n" + "=" * 104)
    print("  DIRECTION VERSION HEAD-TO-HEAD  (top-5% rolling symmetric, window=500)")
    print("=" * 104)
    hdr = f"{'model':<38}{'n_oos':>8}{'IC':>9}{'AUC':>8}{'fires':>8}" \
          f"{'WR':>9}{'Wilson CI':>22}"
    print(hdr)
    print("-" * 104)
    # Sort by WR desc
    rows_sorted = sorted(rows, key=lambda r: r["wr"] if r["wr"] == r["wr"] else 0,
                         reverse=True)
    for r in rows_sorted:
        print(f"{r['name']:<38}{r['n_oos']:>8d}"
              f"{r['ic']:>+9.4f}{r['auc']:>8.4f}"
              f"{r['n_fire']:>8d}{r['wr']*100:>8.1f}%"
              f"  [{r['ci_lo']*100:5.1f}%, {r['ci_hi']*100:5.1f}%]")
    print()


if __name__ == "__main__":
    main()
