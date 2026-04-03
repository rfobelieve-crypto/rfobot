"""
Multi-horizon direction experiment runner.

Runs a grid of experiments varying:
  - Horizon: 1, 2, 4, 8 bars (= 1h, 2h, 4h, 8h on 1h data)
  - Label method: raw_sign, deadzone, triple_barrier
  - Feature set: existing only, direction only, both
  - Vol multiplier k: 0.3, 0.5, 0.8, 1.0

Outputs a summary CSV comparing all experiments.

Usage:
    python research/run_direction_experiments.py
    python research/run_direction_experiments.py --phase 1
    python research/run_direction_experiments.py --phase 2
    python research/run_direction_experiments.py --phase 3
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.train_direction_model import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("research/results")


def run_grid(experiments: list[dict], tag: str = "grid") -> pd.DataFrame:
    """
    Run a list of experiment configs and collect results.

    Parameters
    ----------
    experiments : list of dicts, each with keys matching run_experiment() params
    tag : identifier for this grid run

    Returns
    -------
    DataFrame with one row per experiment and all metrics
    """
    results = []

    for i, exp in enumerate(experiments):
        logger.info("=" * 60)
        logger.info("Experiment %d/%d: %s", i + 1, len(experiments), exp)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            result = run_experiment(**exp)
            elapsed = time.time() - t0

            row = {
                **exp,
                "n_valid": result["label_stats"]["n_valid"],
                "n_neutral_dropped": result["label_stats"]["n_neutral"],
                "up_rate": result["label_stats"]["up_rate"],
                "roc_auc": result["evaluation"]["roc_auc"],
                "pr_auc": result["evaluation"]["pr_auc"],
                "accuracy": result["evaluation"]["accuracy"],
                "precision": result["evaluation"]["precision"],
                "recall": result["evaluation"]["recall"],
                "f1": result["evaluation"]["f1"],
                "top10_precision": result["evaluation"]["top10_precision"],
                "bot10_precision": result["evaluation"]["bot10_precision"],
                "mean_calibration_error": result["evaluation"]["mean_calibration_error"],
                "calibration_monotonic": result["evaluation"].get("calibration_monotonic_frac", np.nan),
                "n_features": result["cv_results"]["n_features"],
                "elapsed_s": elapsed,
            }

            # Per-fold AUC
            for fm in result["cv_results"]["fold_metrics"]:
                row[f"auc_fold{fm['fold']}"] = fm["auc"]

        except Exception as e:
            logger.error("Experiment %d FAILED: %s", i + 1, e)
            row = {**exp, "roc_auc": np.nan, "error": str(e)}
            elapsed = time.time() - t0
            row["elapsed_s"] = elapsed

        results.append(row)

    df = pd.DataFrame(results)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"direction_experiment_{tag}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d experiment results to %s", len(df), out_path)

    return df


def phase1_label_experiments() -> list[dict]:
    """
    Phase 1: Only change labels, keep features fixed (existing only).

    Goal: is label noise the bottleneck?
    """
    experiments = []
    horizon = 4  # fixed at 4h

    # Baseline: raw sign
    experiments.append({
        "method": "raw_sign", "horizon": horizon, "k": 0.0,
        "use_direction_features": False, "use_existing_features": True,
    })

    # Deadzone with different k values
    for k in [0.3, 0.5, 0.8, 1.0]:
        experiments.append({
            "method": "deadzone", "horizon": horizon, "k": k,
            "use_direction_features": False, "use_existing_features": True,
        })

    # Triple barrier with different k values
    for k in [0.3, 0.5, 0.8, 1.0]:
        experiments.append({
            "method": "triple_barrier", "horizon": horizon, "k": k,
            "use_direction_features": False, "use_existing_features": True,
        })

    return experiments


def phase2_horizon_experiments() -> list[dict]:
    """
    Phase 2: Vary horizon with best label method from Phase 1.

    Goal: find the timescale where direction alpha exists.
    Tests both deadzone and triple_barrier at each horizon.
    """
    experiments = []

    for horizon in [1, 2, 4, 8]:
        # Raw sign baseline
        experiments.append({
            "method": "raw_sign", "horizon": horizon, "k": 0.0,
            "use_direction_features": False, "use_existing_features": True,
        })
        # Deadzone (k=0.5 as reasonable default)
        experiments.append({
            "method": "deadzone", "horizon": horizon, "k": 0.5,
            "use_direction_features": False, "use_existing_features": True,
        })
        # Triple barrier
        experiments.append({
            "method": "triple_barrier", "horizon": horizon, "k": 0.5,
            "use_direction_features": False, "use_existing_features": True,
        })

    return experiments


def phase3_feature_experiments() -> list[dict]:
    """
    Phase 3: Add direction-specific features.

    Uses best horizon and label from Phase 1+2.
    Tests direction features alone, existing alone, and combined.
    """
    experiments = []

    # Test at multiple horizons with direction features
    for horizon in [1, 2, 4]:
        for method in ["deadzone", "triple_barrier"]:
            # Existing features only (baseline)
            experiments.append({
                "method": method, "horizon": horizon, "k": 0.5,
                "use_direction_features": False, "use_existing_features": True,
            })
            # Direction features only
            experiments.append({
                "method": method, "horizon": horizon, "k": 0.5,
                "use_direction_features": True, "use_existing_features": False,
            })
            # Combined
            experiments.append({
                "method": method, "horizon": horizon, "k": 0.5,
                "use_direction_features": True, "use_existing_features": True,
            })

    return experiments


def print_summary(df: pd.DataFrame):
    """Pretty-print experiment results table."""
    cols = ["method", "horizon", "k", "use_direction_features",
            "n_valid", "up_rate", "roc_auc", "accuracy",
            "top10_precision", "bot10_precision"]
    available = [c for c in cols if c in df.columns]
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    display = df[available].copy()
    if "roc_auc" in display.columns:
        display = display.sort_values("roc_auc", ascending=False)

    print(display.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    if "roc_auc" in df.columns and df["roc_auc"].notna().any():
        best = df.loc[df["roc_auc"].idxmax()]
        print(f"\nBest AUC: {best['roc_auc']:.4f}")
        print(f"  method={best['method']}, horizon={best['horizon']}, k={best.get('k', 'N/A')}")
        if "use_direction_features" in best:
            print(f"  dir_features={best['use_direction_features']}")


def main():
    ap = argparse.ArgumentParser(description="Run direction prediction experiments")
    ap.add_argument("--phase", type=int, default=0,
                    help="Phase to run: 1=labels, 2=horizons, 3=features, 0=all")
    args = ap.parse_args()

    if args.phase == 0:
        # Run all phases sequentially
        print("\n" + "#" * 60)
        print("# PHASE 1: Label experiments")
        print("#" * 60)
        df1 = run_grid(phase1_label_experiments(), "phase1_labels")
        print_summary(df1)

        print("\n" + "#" * 60)
        print("# PHASE 2: Horizon experiments")
        print("#" * 60)
        df2 = run_grid(phase2_horizon_experiments(), "phase2_horizons")
        print_summary(df2)

        print("\n" + "#" * 60)
        print("# PHASE 3: Feature experiments")
        print("#" * 60)
        df3 = run_grid(phase3_feature_experiments(), "phase3_features")
        print_summary(df3)

        # Combined summary
        all_df = pd.concat([df1, df2, df3], ignore_index=True)
        all_df.to_csv(RESULTS_DIR / "direction_experiment_summary.csv", index=False)
        print("\nAll results saved to research/results/direction_experiment_summary.csv")

    elif args.phase == 1:
        df = run_grid(phase1_label_experiments(), "phase1_labels")
        print_summary(df)

    elif args.phase == 2:
        df = run_grid(phase2_horizon_experiments(), "phase2_horizons")
        print_summary(df)

    elif args.phase == 3:
        df = run_grid(phase3_feature_experiments(), "phase3_features")
        print_summary(df)

    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
