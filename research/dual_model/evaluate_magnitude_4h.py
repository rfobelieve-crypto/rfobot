"""
Magnitude model evaluation suite (4h horizon).

Metrics:
  - IC (Spearman rank correlation) and ICIR
  - RMSE, MAE
  - Decile monotonicity (predicted quantile vs realized move)
  - Top-decile realized move
  - Bucket analysis
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def evaluate_magnitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 5,
) -> dict:
    """
    Full magnitude evaluation suite.

    Parameters
    ----------
    y_true : Actual magnitude values (abs_return or vol_adj_abs_return).
    y_pred : Model predicted magnitude.
    n_deciles : Number of buckets for monotonicity analysis.

    Returns
    -------
    Dict with all metrics.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y = y_true[mask]
    p = y_pred[mask]

    if len(y) < 20:
        logger.warning("Too few samples for evaluation: %d", len(y))
        return {"n_samples": len(y), "error": "insufficient_data"}

    # IC and ICIR
    ic, ic_pval = spearmanr(y, p)

    # Rolling IC for ICIR (use 48-bar windows if enough data)
    icir = _compute_icir(y, p, window=48)

    # Regression metrics
    residuals = y - p
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    result = {
        "n_samples": len(y),
        "ic": float(ic),
        "ic_pval": float(ic_pval),
        "icir": icir,
        "rmse": rmse,
        "mae": mae,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std()),
    }

    # Decile monotonicity
    decile_result = _decile_analysis(y, p, n_deciles)
    result["deciles"] = decile_result["deciles"]
    result["monotonicity_score"] = decile_result["monotonicity_score"]
    result["top_decile_mean"] = decile_result["top_decile_mean"]
    result["bot_decile_mean"] = decile_result["bot_decile_mean"]
    result["top_bot_ratio"] = decile_result["top_bot_ratio"]

    return result


def _compute_icir(y: np.ndarray, p: np.ndarray, window: int = 48) -> float:
    """Compute ICIR = mean(rolling IC) / std(rolling IC)."""
    if len(y) < window * 2:
        ic, _ = spearmanr(y, p)
        return float(ic)  # Not enough data for rolling

    ics = []
    for i in range(0, len(y) - window, window // 2):
        chunk_y = y[i:i + window]
        chunk_p = p[i:i + window]
        if len(chunk_y) >= 10:
            ic_i, _ = spearmanr(chunk_y, chunk_p)
            if not np.isnan(ic_i):
                ics.append(ic_i)

    if len(ics) < 2:
        return float(np.mean(ics)) if ics else 0.0

    ic_mean = np.mean(ics)
    ic_std = np.std(ics)
    return float(ic_mean / ic_std) if ic_std > 0 else float(ic_mean)


def _decile_analysis(y: np.ndarray, p: np.ndarray, n_deciles: int) -> dict:
    """Analyze realized magnitude by predicted quantile."""
    quantiles = np.quantile(p, np.linspace(0, 1, n_deciles + 1))
    quantiles[-1] += 1e-10  # include right edge

    deciles = []
    means = []
    for i in range(n_deciles):
        mask = (p >= quantiles[i]) & (p < quantiles[i + 1])
        n = mask.sum()
        if n > 0:
            m = float(y[mask].mean())
            deciles.append({
                "decile": i + 1,
                "pred_range": f"{quantiles[i]:.4f}-{quantiles[i+1]:.4f}",
                "n": int(n),
                "realized_mean": m,
                "realized_std": float(y[mask].std()),
            })
            means.append(m)
        else:
            deciles.append({"decile": i + 1, "n": 0, "realized_mean": np.nan})
            means.append(np.nan)

    # Monotonicity score: how often does next decile have higher realized move?
    valid_means = [m for m in means if not np.isnan(m)]
    if len(valid_means) >= 2:
        monotonic_pairs = sum(
            1 for i in range(len(valid_means) - 1)
            if valid_means[i + 1] > valid_means[i]
        )
        mono_score = monotonic_pairs / (len(valid_means) - 1)
    else:
        mono_score = 0.0

    top_mean = valid_means[-1] if valid_means else 0
    bot_mean = valid_means[0] if valid_means else 0

    return {
        "deciles": deciles,
        "monotonicity_score": float(mono_score),
        "top_decile_mean": float(top_mean),
        "bot_decile_mean": float(bot_mean),
        "top_bot_ratio": float(top_mean / bot_mean) if bot_mean > 0 else 0,
    }


def print_magnitude_report(metrics: dict, label: str = "Magnitude Model"):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  IC (Spearman):  {metrics.get('ic', 0):.4f}  (p={metrics.get('ic_pval', 1):.4f})")
    print(f"  ICIR:           {metrics.get('icir', 0):.4f}")
    print(f"  RMSE:           {metrics.get('rmse', 0):.6f}")
    print(f"  MAE:            {metrics.get('mae', 0):.6f}")
    print(f"  Monotonicity:   {metrics.get('monotonicity_score', 0):.2f}")
    print(f"  Top/Bot ratio:  {metrics.get('top_bot_ratio', 0):.2f}x")

    deciles = metrics.get("deciles", [])
    if deciles:
        print(f"\n  Decile Analysis:")
        print(f"  {'D':>3s}  {'N':>5s}  {'Realized Mean':>14s}  {'Realized Std':>12s}")
        for d in deciles:
            if d["n"] > 0:
                print(f"  {d['decile']:>3d}  {d['n']:>5d}  "
                      f"{d['realized_mean']:>14.6f}  {d.get('realized_std', 0):>12.6f}")
    print()
