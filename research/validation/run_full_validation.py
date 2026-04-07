"""
Full validation pipeline for the dual-model system.

Runs all 5 validation layers sequentially and produces a final summary.

Usage:
    python -m research.validation.run_full_validation
    python -m research.validation.run_full_validation --layer 3   (single layer)

Layers:
    1. Time Stability — model consistent across time periods?
    2. Regime Analysis — model works in all market conditions?
    3. Threshold Sweep — thresholds robust or overfit?
    4. Signal Decay — how long does signal persist?
    5. Strategy Backtest — can it make money?

All results saved to: research/results/validation/
"""
from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DUAL_RESULTS = PROJECT_ROOT / "research" / "results" / "dual_model"
FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
OUTPUT_DIR = PROJECT_ROOT / "research" / "results" / "validation"


def load_oos_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load OOS prediction data from walk-forward experiments."""
    dir_path = DUAL_RESULTS / "direction_oos_plus_key_4_only.parquet"
    mag_path = DUAL_RESULTS / "magnitude_oos_expanded.parquet"
    joint_path = DUAL_RESULTS / "exp4_joint_predictions.parquet"

    dir_oos = pd.read_parquet(dir_path)
    mag_oos = pd.read_parquet(mag_path)
    joint = pd.read_parquet(joint_path)

    logger.info("Loaded OOS data: dir=%d, mag=%d, joint=%d",
                len(dir_oos), len(mag_oos), len(joint))
    logger.info("Date range: %s → %s", joint.index[0], joint.index[-1])

    return dir_oos, mag_oos, joint


def layer_1_time_stability(dir_oos, mag_oos, joint):
    """Layer 1: Time Stability."""
    from research.validation.time_stability import run_time_stability, print_time_stability_report

    print("\n" + "=" * 70)
    print("  LAYER 1: TIME STABILITY")
    print("=" * 70)

    df = run_time_stability(dir_oos, mag_oos, joint, OUTPUT_DIR)
    print_time_stability_report(df)
    return df


def layer_2_regime_analysis(joint):
    """Layer 2: Regime Analysis."""
    from research.validation.regime_analysis import run_regime_analysis, print_regime_report

    print("\n" + "=" * 70)
    print("  LAYER 2: REGIME ANALYSIS")
    print("=" * 70)

    df = run_regime_analysis(joint, FEATURES_CACHE, OUTPUT_DIR)
    print_regime_report(df)
    return df


def layer_3_threshold_sweep(joint):
    """Layer 3: Threshold Sweep."""
    from research.validation.threshold_sweep import run_threshold_sweep, print_threshold_report

    print("\n" + "=" * 70)
    print("  LAYER 3: THRESHOLD SWEEP")
    print("=" * 70)

    df = run_threshold_sweep(joint, OUTPUT_DIR)
    print_threshold_report(df)
    return df


def layer_4_signal_decay(joint):
    """Layer 4: Signal Decay."""
    from research.validation.signal_decay import run_signal_decay, print_signal_decay_report

    print("\n" + "=" * 70)
    print("  LAYER 4: SIGNAL DECAY")
    print("=" * 70)

    df = run_signal_decay(joint, FEATURES_CACHE, OUTPUT_DIR)
    print_signal_decay_report(df)
    return df


def layer_5_strategy_backtest(joint):
    """Layer 5: Strategy Backtest."""
    from research.validation.strategy_backtest import run_strategy_backtest, print_strategy_report

    print("\n" + "=" * 70)
    print("  LAYER 5: STRATEGY BACKTEST")
    print("=" * 70)

    results = run_strategy_backtest(joint, OUTPUT_DIR)
    print_strategy_report(results)
    return results


def print_final_verdict(time_df, regime_df, threshold_df, decay_df, strategy_results):
    """Print final go/no-go assessment based on all validation layers."""
    print("\n" + "#" * 70)
    print("  FINAL VALIDATION VERDICT")
    print("#" * 70)

    issues = []
    passes = []

    # Layer 1: Time stability
    if time_df is not None and "dir_auc" in time_df.columns:
        monthly = time_df[time_df["period"].str.startswith("month")]
        if len(monthly) > 0:
            auc_vals = monthly["dir_auc"].dropna()
            if len(auc_vals) > 0:
                min_auc = auc_vals.min()
                below_05 = (auc_vals < 0.50).sum()
                if below_05 > 0:
                    issues.append(f"TIME: {below_05}/{len(auc_vals)} months with AUC < 0.50 (worst: {min_auc:.3f})")
                else:
                    passes.append(f"TIME: All months AUC > 0.50 (min: {min_auc:.3f})")

    # Layer 2: Regime
    if regime_df is not None and "dir_auc" in regime_df.columns:
        weak_regimes = regime_df[regime_df["dir_auc"].fillna(0) < 0.50]
        if len(weak_regimes) > 0:
            names = ", ".join(weak_regimes["regime_value"].tolist()[:3])
            issues.append(f"REGIME: AUC < 0.50 in: {names}")
        else:
            passes.append("REGIME: AUC > 0.50 in all regimes")

    # Layer 3: Threshold
    if threshold_df is not None and "is_stable" in threshold_df.columns:
        stable_count = threshold_df["is_stable"].sum()
        total = len(threshold_df)
        if stable_count < 3:
            issues.append(f"THRESHOLD: Only {stable_count}/{total} combos stable (< 3 = fragile)")
        else:
            passes.append(f"THRESHOLD: {stable_count}/{total} stable combos (robust)")

    # Layer 4: Signal decay
    if decay_df is not None and "dir_auc" in decay_df.columns:
        baseline_4h = decay_df[decay_df["horizon_hours"] == 4]
        h1 = decay_df[decay_df["horizon_hours"] == 1]
        if len(baseline_4h) > 0 and len(h1) > 0:
            auc_4h = baseline_4h["dir_auc"].iloc[0]
            auc_1h = h1["dir_auc"].iloc[0]
            if auc_4h > auc_1h:
                passes.append(f"DECAY: Signal peaks at 4h (AUC 4h={auc_4h:.3f} > 1h={auc_1h:.3f})")
            else:
                issues.append(f"DECAY: Signal peaks at 1h, not 4h — horizon mismatch?")

    # Layer 5: Strategy
    if strategy_results is not None:
        perf = strategy_results.get("performance")
        if perf is not None and len(perf) > 0:
            row = perf.iloc[0]
            total_ret = row.get("total_return", 0)
            win_rate = row.get("win_rate", 0)
            max_dd = row.get("max_drawdown", 0)
            sharpe = row.get("sharpe_ratio", 0)

            if total_ret <= 0:
                issues.append(f"STRATEGY: Negative total return ({total_ret:.2%})")
            else:
                passes.append(f"STRATEGY: Positive return ({total_ret:.2%})")

            if win_rate < 0.52:
                issues.append(f"STRATEGY: Win rate too low ({win_rate:.1%})")
            if max_dd < -0.15:
                issues.append(f"STRATEGY: Max drawdown too deep ({max_dd:.1%})")
            if sharpe < 0.5:
                issues.append(f"STRATEGY: Sharpe < 0.5 ({sharpe:.2f})")
            elif sharpe > 0.5:
                passes.append(f"STRATEGY: Sharpe = {sharpe:.2f}")

    # Verdict
    print()
    if passes:
        print("  PASSES:")
        for p in passes:
            print(f"    [+] {p}")
    if issues:
        print("\n  ISSUES:")
        for i in issues:
            print(f"    [-] {i}")

    print()
    if len(issues) == 0:
        print("  >>> VERDICT: ALL CLEAR — model is validated for live use")
    elif len(issues) <= 2 and all("STRATEGY" not in i for i in issues):
        print("  >>> VERDICT: CONDITIONAL PASS — minor issues, monitor closely")
    elif any("STRATEGY: Negative" in i for i in issues):
        print("  >>> VERDICT: FAIL — strategy loses money on OOS data")
    else:
        print("  >>> VERDICT: CAUTION — multiple issues found, investigate before going live")

    print()
    print("  Results saved to: research/results/validation/")
    print()


def main():
    parser = argparse.ArgumentParser(description="Full validation pipeline")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run specific layer (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dir_oos, mag_oos, joint = load_oos_data()

    print("\n" + "#" * 70)
    print("  DUAL-MODEL VALIDATION PIPELINE")
    print(f"  OOS bars: {len(joint)}")
    print(f"  Date range: {joint.index[0]} → {joint.index[-1]}")
    print(f"  Direction OOS: {len(dir_oos)} bars")
    print(f"  Magnitude OOS: {len(mag_oos)} bars")
    print("#" * 70)

    time_df = regime_df = threshold_df = decay_df = strategy_results = None

    layers = {
        1: lambda: layer_1_time_stability(dir_oos, mag_oos, joint),
        2: lambda: layer_2_regime_analysis(joint),
        3: lambda: layer_3_threshold_sweep(joint),
        4: lambda: layer_4_signal_decay(joint),
        5: lambda: layer_5_strategy_backtest(joint),
    }

    if args.layer:
        result = layers[args.layer]()
        if args.layer == 1: time_df = result
        elif args.layer == 2: regime_df = result
        elif args.layer == 3: threshold_df = result
        elif args.layer == 4: decay_df = result
        elif args.layer == 5: strategy_results = result
    else:
        for layer_num in [1, 2, 3, 4, 5]:
            try:
                result = layers[layer_num]()
                if layer_num == 1: time_df = result
                elif layer_num == 2: regime_df = result
                elif layer_num == 3: threshold_df = result
                elif layer_num == 4: decay_df = result
                elif layer_num == 5: strategy_results = result
            except Exception as e:
                logger.error("Layer %d failed: %s", layer_num, e, exc_info=True)

    # Final verdict (only when running all layers)
    if not args.layer:
        print_final_verdict(time_df, regime_df, threshold_df, decay_df, strategy_results)


if __name__ == "__main__":
    main()
