"""
Master experiment runner for the dual-model research framework.

Runs all 4 experiments in sequence:
  1. Direction: Baseline vs Expanded (AUC uplift)
  2. Direction: Feature Ablation (per-group contribution)
  3. Magnitude: Baseline vs Expanded (IC uplift)
  4. Dual-Model Joint Analysis (signal regime combinations)

Usage:
    python -m research.dual_model.run_dual_model_experiments
    python -m research.dual_model.run_dual_model_experiments --experiment 1
    python -m research.dual_model.run_dual_model_experiments --experiment 4
    python -m research.dual_model.run_dual_model_experiments --refresh

All results saved to research/results/dual_model/
"""
from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import (
    load_and_cache_data, RESULTS_DIR, ensure_dirs,
)

logger = logging.getLogger(__name__)


def experiment_1(df: pd.DataFrame):
    """
    Experiment 1: Direction Baseline vs Expanded.

    Compare old direction features against full expanded set.
    Report AUC uplift, top-decile precision uplift.
    """
    print("\n" + "#" * 70)
    print("  EXPERIMENT 1: Direction Baseline vs Expanded")
    print("#" * 70)

    from research.dual_model.train_direction_model_4h import run_single
    from research.dual_model.direction_features_v2 import ABLATION_GROUPS

    old_metrics = run_single("baseline_old", ABLATION_GROUPS["baseline_old"], df)
    new_metrics = run_single("full_expanded", ABLATION_GROUPS["full_expanded"], df)

    print("\n  === UPLIFT SUMMARY ===")
    for metric in ["roc_auc", "pr_auc", "accuracy", "top_decile_precision", "f1"]:
        old_v = old_metrics.get(metric, 0)
        new_v = new_metrics.get(metric, 0)
        delta = new_v - old_v
        print(f"  {metric:25s}  old={old_v:.4f}  new={new_v:.4f}  "
              f"delta={delta:+.4f}  ({delta/max(abs(old_v), 1e-6):+.1%})")

    # Save comparison
    comparison = pd.DataFrame([
        {"set": "baseline_old", **old_metrics},
        {"set": "full_expanded", **new_metrics},
    ])
    comparison.to_csv(RESULTS_DIR / "exp1_direction_baseline_vs_expanded.csv", index=False)

    return old_metrics, new_metrics


def experiment_2(df: pd.DataFrame):
    """
    Experiment 2: Direction Feature Ablation.

    Add each new feature group incrementally to measure OOS uplift.
    """
    print("\n" + "#" * 70)
    print("  EXPERIMENT 2: Direction Feature Ablation")
    print("#" * 70)

    from research.dual_model.train_direction_model_4h import run_ablation
    results = run_ablation(df)
    return results


def experiment_3(df: pd.DataFrame):
    """
    Experiment 3: Magnitude Baseline vs Expanded.

    Compare old magnitude features against expanded set.
    """
    print("\n" + "#" * 70)
    print("  EXPERIMENT 3: Magnitude Baseline vs Expanded")
    print("#" * 70)

    from research.dual_model.train_magnitude_model_4h import run_single
    from research.dual_model.magnitude_features_v2 import MAGNITUDE_GROUPS

    old_metrics = run_single("baseline_old", MAGNITUDE_GROUPS["baseline_old"], df)
    new_metrics = run_single("expanded", MAGNITUDE_GROUPS["expanded"], df)

    print("\n  === UPLIFT SUMMARY ===")
    for metric in ["ic", "icir", "rmse", "mae", "monotonicity_score", "top_bot_ratio"]:
        old_v = old_metrics.get(metric, 0)
        new_v = new_metrics.get(metric, 0)
        delta = new_v - old_v
        better = "better" if (metric in ["ic", "icir", "monotonicity_score", "top_bot_ratio"]
                              and delta > 0) else ("better" if delta < 0 else "worse")
        print(f"  {metric:25s}  old={old_v:.4f}  new={new_v:.4f}  "
              f"delta={delta:+.4f}")

    comparison = pd.DataFrame([
        {"set": "baseline_old", **old_metrics},
        {"set": "expanded", **new_metrics},
    ])
    comparison.to_csv(RESULTS_DIR / "exp3_magnitude_baseline_vs_expanded.csv", index=False)

    return old_metrics, new_metrics


def experiment_4(df: pd.DataFrame):
    """
    Experiment 4: Dual-Model Joint Analysis.

    Load OOS predictions from both models and analyze signal regimes:
      - high prob_up + high magnitude → strongest signal
      - uncertain direction + high magnitude → vol event, no directional bet
      - low magnitude → skip regardless of direction
    """
    print("\n" + "#" * 70)
    print("  EXPERIMENT 4: Dual-Model Joint Analysis")
    print("#" * 70)

    # Load or generate OOS predictions
    dir_path = RESULTS_DIR / "direction_oos_full_expanded.parquet"
    mag_path = RESULTS_DIR / "magnitude_oos_expanded.parquet"

    if not dir_path.exists():
        logger.info("Direction OOS not found, running experiment 1 first...")
        experiment_1(df)
    if not mag_path.exists():
        logger.info("Magnitude OOS not found, running experiment 3 first...")
        experiment_3(df)

    dir_oos = pd.read_parquet(dir_path)
    mag_oos = pd.read_parquet(mag_path)

    # Align on common timestamps
    common = dir_oos.index.intersection(mag_oos.index)
    if len(common) < 10:
        logger.error("Not enough overlapping predictions: %d", len(common))
        return

    joint = pd.DataFrame({
        "prob_up": dir_oos.loc[common, "prob_up"],
        "dir_true": dir_oos.loc[common, "y_true"],
        "mag_pred": mag_oos.loc[common, "y_pred"],
        "mag_true": mag_oos.loc[common, "y_true"],
        "return_4h": dir_oos.loc[common, "return_4h"],
    })

    # Define signal regimes
    prob_high = joint["prob_up"] > 0.6
    prob_low = joint["prob_up"] < 0.4
    prob_uncertain = (~prob_high) & (~prob_low)

    mag_median = joint["mag_pred"].median()
    mag_high = joint["mag_pred"] > mag_median
    mag_low = ~mag_high

    regimes = {
        "high_up + high_mag":  prob_high & mag_high,
        "high_up + low_mag":   prob_high & mag_low,
        "high_down + high_mag": prob_low & mag_high,
        "high_down + low_mag": prob_low & mag_low,
        "uncertain + high_mag": prob_uncertain & mag_high,
        "uncertain + low_mag": prob_uncertain & mag_low,
    }

    print(f"\n  Joint predictions: {len(joint)} bars")
    print(f"  Direction OOS: prob_up mean={joint['prob_up'].mean():.3f}")
    print(f"  Magnitude OOS: mag_pred mean={joint['mag_pred'].mean():.6f}")
    print()

    regime_stats = []
    print(f"  {'Regime':30s}  {'N':>5s}  {'Ret Mean':>10s}  {'Ret Std':>10s}  "
          f"{'Dir Acc':>8s}  {'Mag Mean':>10s}")
    print("  " + "-" * 85)

    for name, mask in regimes.items():
        n = mask.sum()
        if n < 3:
            continue

        sub = joint[mask]
        ret_mean = sub["return_4h"].mean()
        ret_std = sub["return_4h"].std()

        # Direction accuracy for this regime
        if "high_up" in name:
            dir_correct = (sub["return_4h"] > 0).mean()
        elif "high_down" in name:
            dir_correct = (sub["return_4h"] < 0).mean()
        else:
            dir_correct = np.nan

        mag_actual = sub["mag_true"].mean()

        print(f"  {name:30s}  {n:>5d}  {ret_mean:>+10.4f}  {ret_std:>10.4f}  "
              f"{dir_correct:>8.3f}  {mag_actual:>10.6f}")

        regime_stats.append({
            "regime": name, "n": n,
            "return_mean": ret_mean, "return_std": ret_std,
            "direction_accuracy": dir_correct,
            "actual_magnitude_mean": mag_actual,
        })

    # Save
    if regime_stats:
        regime_df = pd.DataFrame(regime_stats)
        regime_df.to_csv(RESULTS_DIR / "exp4_joint_regime_analysis.csv", index=False)

    joint.to_parquet(RESULTS_DIR / "exp4_joint_predictions.parquet")

    # Key insight
    print("\n  === KEY INSIGHT ===")
    if "high_up + high_mag" in regimes and regimes["high_up + high_mag"].sum() > 0:
        strong_up = joint[regimes["high_up + high_mag"]]
        print(f"  Strong UP signal: {len(strong_up)} bars, "
              f"mean return = {strong_up['return_4h'].mean():+.4f}, "
              f"win rate = {(strong_up['return_4h'] > 0).mean():.1%}")
    if "high_down + high_mag" in regimes and regimes["high_down + high_mag"].sum() > 0:
        strong_down = joint[regimes["high_down + high_mag"]]
        print(f"  Strong DN signal: {len(strong_down)} bars, "
              f"mean return = {strong_down['return_4h'].mean():+.4f}, "
              f"win rate = {(strong_down['return_4h'] < 0).mean():.1%}")

    return regime_stats


def main():
    """Run selected or all experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Dual-model experiments")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4],
                        help="Run specific experiment (default: all)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-fetch data")
    args = parser.parse_args()

    ensure_dirs()
    df = load_and_cache_data(force_refresh=args.refresh)

    experiments = {
        1: experiment_1,
        2: experiment_2,
        3: experiment_3,
        4: experiment_4,
    }

    if args.experiment:
        experiments[args.experiment](df)
    else:
        # Run all in order
        print("\n" + "=" * 70)
        print("  DUAL-MODEL EXPERIMENT SUITE")
        print(f"  Data: {len(df)} bars, {len(df.columns)} features")
        print(f"  Range: {df.index[0]} → {df.index[-1]}")
        print("=" * 70)

        for exp_num in [1, 2, 3, 4]:
            try:
                experiments[exp_num](df)
            except Exception as e:
                logger.error("Experiment %d failed: %s", exp_num, e, exc_info=True)

        print("\n" + "=" * 70)
        print("  ALL EXPERIMENTS COMPLETE")
        print(f"  Results saved to: {RESULTS_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
