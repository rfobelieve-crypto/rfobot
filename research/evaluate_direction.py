"""
Direction model evaluation module.

Provides comprehensive evaluation of binary direction classifiers:
  - ROC AUC, PR AUC
  - Accuracy, Precision, Recall, F1
  - Top-decile precision
  - Calibration table by probability decile
  - Regime-wise AUC breakdown

Usage:
    from research.evaluate_direction import evaluate_direction_model
    summary, detail_df = evaluate_direction_model(y_true, y_prob, regime=regime)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute area under the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # Trapezoidal integration (recall is decreasing)
    return -np.trapz(precision, recall)


def compute_top_decile_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Precision when only taking top/bottom 10% of predictions.

    Returns dict with:
        top10_precision: precision among highest 10% prob_up
        bot10_precision: precision of DOWN among lowest 10% prob_up
    """
    n = len(y_true)
    k = max(1, n // 10)

    # Top 10% (highest prob_up → predict UP)
    top_idx = np.argsort(y_prob)[-k:]
    top_prec = y_true[top_idx].mean()

    # Bottom 10% (lowest prob_up → predict DOWN)
    bot_idx = np.argsort(y_prob)[:k]
    bot_prec = 1 - y_true[bot_idx].mean()  # fraction that are actually DOWN

    return {
        "top10_precision": float(top_prec),
        "bot10_precision": float(bot_prec),
        "top10_count": int(k),
    }


def compute_calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calibration table: group by predicted probability decile,
    compute actual positive rate per bin.

    Perfect calibration: predicted ≈ actual in each bin.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], n_bins, labels=False, duplicates="drop")

    table = df.groupby("decile").agg(
        count=("y_true", "count"),
        mean_pred_prob=("y_prob", "mean"),
        actual_up_rate=("y_true", "mean"),
        min_prob=("y_prob", "min"),
        max_prob=("y_prob", "max"),
    ).reset_index()

    # Calibration error per bin
    table["calibration_error"] = (table["actual_up_rate"] - table["mean_pred_prob"]).abs()

    return table


def compute_regime_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    regime: np.ndarray,
) -> pd.DataFrame:
    """
    AUC breakdown by regime.

    Returns DataFrame with AUC and sample count per regime.
    """
    rows = []
    for r in np.unique(regime):
        mask = regime == r
        if mask.sum() < 20:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        # Need both classes present
        if len(np.unique(yt)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(yt, yp)
        rows.append({
            "regime": r,
            "auc": auc,
            "count": int(mask.sum()),
            "up_rate": float(yt.mean()),
        })
    return pd.DataFrame(rows)


def evaluate_direction_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    regime: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    """
    Comprehensive evaluation of a binary direction model.

    Parameters
    ----------
    y_true : actual labels (0 or 1)
    y_prob : predicted probability of UP (0 to 1)
    regime : optional regime labels for regime-wise breakdown
    threshold : classification threshold (default 0.5)

    Returns
    -------
    summary : dict with all scalar metrics
    detail_df : DataFrame with calibration + regime tables
    """
    # Clean NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]

    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    n = len(y_true)
    n_up = int(y_true.sum())
    n_down = n - n_up

    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = compute_pr_auc(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    top_decile = compute_top_decile_precision(y_true, y_prob)

    summary = {
        "n_samples": n,
        "n_up": n_up,
        "n_down": n_down,
        "class_balance": float(n_up / n) if n > 0 else np.nan,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        **top_decile,
    }

    # Calibration table
    cal_table = compute_calibration_table(y_true, y_prob)
    summary["mean_calibration_error"] = float(cal_table["calibration_error"].mean())

    # Monotonicity check: does actual_up_rate increase with predicted prob?
    if len(cal_table) >= 3:
        diffs = cal_table["actual_up_rate"].diff().dropna()
        summary["calibration_monotonic_frac"] = float((diffs > 0).mean())
    else:
        summary["calibration_monotonic_frac"] = np.nan

    # Regime breakdown
    detail_parts = [cal_table.assign(table="calibration")]
    if regime is not None:
        regime_clean = regime[mask]
        regime_df = compute_regime_auc(y_true, y_prob, regime_clean)
        if len(regime_df) > 0:
            summary["regime_auc_mean"] = float(regime_df["auc"].mean())
            summary["regime_auc_std"] = float(regime_df["auc"].std())
            detail_parts.append(
                regime_df.assign(table="regime").rename(
                    columns={"count": "count", "auc": "actual_up_rate"}
                )
            )

    detail_df = pd.concat(detail_parts, ignore_index=True)

    return summary, detail_df


def format_summary(summary: dict) -> str:
    """Pretty-print evaluation summary."""
    lines = [
        "=" * 55,
        "Direction Model Evaluation",
        "=" * 55,
        f"  Samples: {summary['n_samples']}  "
        f"(UP={summary['n_up']}, DOWN={summary['n_down']}, "
        f"balance={summary['class_balance']:.1%})",
        "",
        f"  ROC AUC:     {summary['roc_auc']:.4f}",
        f"  PR AUC:      {summary['pr_auc']:.4f}",
        f"  Accuracy:    {summary['accuracy']:.4f}",
        f"  Precision:   {summary['precision']:.4f}",
        f"  Recall:      {summary['recall']:.4f}",
        f"  F1:          {summary['f1']:.4f}",
        "",
        f"  Top-10% precision (UP):   {summary['top10_precision']:.4f}",
        f"  Bot-10% precision (DOWN): {summary['bot10_precision']:.4f}",
        "",
        f"  Calibration error (mean): {summary['mean_calibration_error']:.4f}",
        f"  Calibration monotonic:    {summary.get('calibration_monotonic_frac', 'N/A')}",
        "=" * 55,
    ]
    return "\n".join(lines)
