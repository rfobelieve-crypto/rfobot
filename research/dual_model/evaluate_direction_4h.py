"""
Direction model evaluation suite (4h horizon).

Metrics:
  - ROC AUC, PR AUC
  - Accuracy, Precision, Recall, F1
  - Top-decile precision (strongest predictions)
  - Calibration table (predicted prob vs actual up rate)
  - Regime-wise AUC (if regime column available)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, brier_score_loss,
)

logger = logging.getLogger(__name__)


def evaluate_direction(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    threshold: float = 0.5,
    regime: np.ndarray | None = None,
) -> dict:
    """
    Full direction evaluation suite.

    Parameters
    ----------
    y_true : Binary labels (1=UP, 0=DOWN). NaN rows should be pre-filtered.
    prob_up : Model predicted P(UP) ∈ [0, 1].
    threshold : Classification threshold for binary metrics.
    regime : Optional regime labels for regime-wise AUC.

    Returns
    -------
    Dict with all metrics.
    """
    # Filter valid
    mask = ~(np.isnan(y_true) | np.isnan(prob_up))
    y = y_true[mask].astype(int)
    p = prob_up[mask]

    if len(y) < 20:
        logger.warning("Too few samples for evaluation: %d", len(y))
        return {"n_samples": len(y), "error": "insufficient_data"}

    y_pred = (p >= threshold).astype(int)

    result = {
        "n_samples": len(y),
        "up_rate": float(y.mean()),
        "pred_up_rate": float(y_pred.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    # Top-decile precision (top 10% most confident UP predictions)
    n_top = max(1, len(p) // 10)
    top_idx = np.argsort(p)[-n_top:]
    result["top_decile_precision"] = float(y[top_idx].mean())
    result["top_decile_n"] = n_top

    # Bottom-decile precision (most confident DOWN predictions)
    bot_idx = np.argsort(p)[:n_top]
    result["bot_decile_down_rate"] = float(1 - y[bot_idx].mean())
    result["bot_decile_n"] = n_top

    # Calibration table (5 bins)
    result["calibration"] = _calibration_table(y, p, n_bins=5)

    # Regime-wise AUC
    if regime is not None:
        regime_filtered = regime[mask]
        result["regime_auc"] = _regime_auc(y, p, regime_filtered)

    return result


def _calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Bin predictions and compute actual up rate per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    table = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        n = mask.sum()
        if n > 0:
            table.append({
                "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                "n": int(n),
                "mean_pred": float(p[mask].mean()),
                "actual_up_rate": float(y[mask].mean()),
                "gap": float(y[mask].mean() - p[mask].mean()),
            })
    return table


def _regime_auc(y: np.ndarray, p: np.ndarray, regime: np.ndarray) -> dict:
    """Compute AUC per regime."""
    result = {}
    for r in np.unique(regime):
        if pd.isna(r):
            continue
        mask = regime == r
        n = mask.sum()
        if n < 10 or len(np.unique(y[mask])) < 2:
            result[str(r)] = {"n": int(n), "auc": None}
        else:
            result[str(r)] = {
                "n": int(n),
                "auc": float(roc_auc_score(y[mask], p[mask])),
            }
    return result


def print_direction_report(metrics: dict, label: str = "Direction Model"):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Samples: {metrics['n_samples']}  (UP rate: {metrics.get('up_rate', 0):.1%})")
    print(f"  ROC AUC:       {metrics.get('roc_auc', 0):.4f}")
    print(f"  PR AUC:        {metrics.get('pr_auc', 0):.4f}")
    print(f"  Brier Score:   {metrics.get('brier_score', 0):.4f}")
    print(f"  Accuracy:      {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:     {metrics.get('precision', 0):.4f}")
    print(f"  Recall:        {metrics.get('recall', 0):.4f}")
    print(f"  F1:            {metrics.get('f1', 0):.4f}")
    print(f"  Top-decile P:  {metrics.get('top_decile_precision', 0):.4f} (n={metrics.get('top_decile_n', 0)})")
    print(f"  Bot-decile DR: {metrics.get('bot_decile_down_rate', 0):.4f}")

    cal = metrics.get("calibration", [])
    if cal:
        print(f"\n  Calibration:")
        print(f"  {'Bin':>12s}  {'N':>5s}  {'Pred':>6s}  {'Actual':>6s}  {'Gap':>6s}")
        for row in cal:
            print(f"  {row['bin']:>12s}  {row['n']:>5d}  "
                  f"{row['mean_pred']:>6.3f}  {row['actual_up_rate']:>6.3f}  "
                  f"{row['gap']:>+6.3f}")

    regime = metrics.get("regime_auc", {})
    if regime:
        print(f"\n  Regime AUC:")
        for r, v in regime.items():
            auc_str = f"{v['auc']:.4f}" if v.get("auc") else "N/A"
            print(f"    {r:20s}  n={v['n']:>4d}  AUC={auc_str}")
    print()
