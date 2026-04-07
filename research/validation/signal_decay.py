"""
Layer 4: Signal Decay Analysis.

Determine how long the dual-model signal remains predictive by evaluating
OOS predictions against forward returns at multiple horizons (1h to 24h).

A healthy predictive signal should:
  - Peak around the target horizon (4h)
  - Decay gradually at longer horizons
  - If signal persists too long -> possible look-ahead or data leakage
  - If signal dies before 4h -> model may be fitting noise

Metrics per horizon:
  - Direction AUC: prob_up vs (return > 0)
  - Direction Spearman: rank correlation of prob_up vs return
  - Magnitude IC: Spearman correlation of mag_pred vs abs(return)

Usage:
    python -m research.validation.signal_decay
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Default horizons: bars ahead (1 bar = 1h for hourly data)
HORIZONS = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "8h": 8,
    "12h": 12,
    "24h": 24,
}

DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "research" / "results" / "validation"


def _compute_forward_returns(
    close: pd.Series,
    horizons: dict[str, int],
) -> pd.DataFrame:
    """
    Compute forward returns at multiple horizons from a close price series.

    Parameters
    ----------
    close : pd.Series with DatetimeIndex, containing close prices.
    horizons : Dict mapping horizon label to number of bars ahead.

    Returns
    -------
    DataFrame with columns like return_1h, return_2h, etc.
    Index matches the close series.
    """
    returns = pd.DataFrame(index=close.index)
    for label, bars in horizons.items():
        future_close = close.shift(-bars)
        returns[f"return_{label}"] = future_close / close - 1
    return returns


def _evaluate_horizon(
    prob_up: np.ndarray,
    mag_pred: np.ndarray,
    forward_return: np.ndarray,
) -> dict:
    """
    Evaluate direction and magnitude predictions against a single horizon.

    Parameters
    ----------
    prob_up : Model P(UP) predictions, shape (n,).
    mag_pred : Model magnitude predictions, shape (n,).
    forward_return : Actual forward returns at this horizon, shape (n,).

    Returns
    -------
    Dict with dir_auc, dir_spearman, mag_ic, n_samples.
    """
    # Filter NaN
    mask = ~(np.isnan(prob_up) | np.isnan(mag_pred) | np.isnan(forward_return))
    p = prob_up[mask]
    m = mag_pred[mask]
    r = forward_return[mask]
    n = len(r)

    result = {"n_samples": n}

    if n < 30:
        logger.warning("Horizon has only %d valid samples, skipping", n)
        result.update({"dir_auc": np.nan, "dir_spearman": np.nan, "mag_ic": np.nan})
        return result

    # Direction AUC: prob_up vs binary (return > 0)
    y_binary = (r > 0).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == n:
        # All same class, AUC undefined
        result["dir_auc"] = np.nan
    else:
        result["dir_auc"] = float(roc_auc_score(y_binary, p))

    # Direction Spearman: prob_up vs actual return (continuous)
    sp_dir, _ = stats.spearmanr(p, r)
    result["dir_spearman"] = float(sp_dir)

    # Magnitude IC: mag_pred vs abs(return)
    sp_mag, _ = stats.spearmanr(m, np.abs(r))
    result["mag_ic"] = float(sp_mag)

    return result


def _estimate_half_life(
    decay_df: pd.DataFrame,
    metric: str = "dir_auc",
    baseline: float = 0.5,
) -> float | None:
    """
    Estimate the half-life: horizon (in hours) where metric drops to
    midpoint between peak and baseline.

    Uses linear interpolation between measured horizons.

    Parameters
    ----------
    decay_df : Output of run_signal_decay with horizon_hours column.
    metric : Column to use (default: dir_auc).
    baseline : The floor value (0.5 for AUC).

    Returns
    -------
    Estimated half-life in hours, or None if cannot be determined.
    """
    df = decay_df[["horizon_hours", metric]].dropna().sort_values("horizon_hours")
    if len(df) < 2:
        return None

    hours = df["horizon_hours"].values
    values = df[metric].values

    peak_val = values.max()
    if peak_val <= baseline:
        return None  # Signal never exceeds baseline

    midpoint = (peak_val + baseline) / 2

    # Find where values cross below midpoint (after peak)
    peak_idx = int(np.argmax(values))

    for i in range(peak_idx, len(values) - 1):
        if values[i] >= midpoint and values[i + 1] < midpoint:
            # Linear interpolation
            frac = (midpoint - values[i]) / (values[i + 1] - values[i])
            half_life = hours[i] + frac * (hours[i + 1] - hours[i])
            return float(half_life)

    # If never crosses midpoint, signal persists beyond measured range
    return None


def run_signal_decay(
    joint: pd.DataFrame,
    features_path: Path = DEFAULT_FEATURES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Run signal decay analysis across multiple horizons.

    Parameters
    ----------
    joint : DataFrame with OOS predictions. Required columns:
            [prob_up, dir_true, mag_pred, mag_true, return_4h].
            Index must be DatetimeIndex (UTC), named "dt".
    features_path : Path to features parquet containing "close" column.
    output_dir : Directory for output CSV.

    Returns
    -------
    DataFrame with columns:
        [horizon_hours, n_samples, dir_auc, dir_spearman, mag_ic]
    Sorted by horizon_hours ascending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load close prices ---
    logger.info("Loading close prices from %s", features_path)
    features = pd.read_parquet(features_path)

    if "close" not in features.columns:
        raise ValueError(
            f"features parquet must contain 'close' column. "
            f"Found: {list(features.columns[:10])}..."
        )

    # Ensure datetime index
    if not isinstance(features.index, pd.DatetimeIndex):
        if "dt" in features.columns:
            features = features.set_index("dt")
        else:
            raise ValueError("Features must have DatetimeIndex or 'dt' column")

    close = features["close"].sort_index()
    logger.info(
        "Close prices: %d rows, %s to %s",
        len(close), close.index.min(), close.index.max(),
    )

    # --- Compute forward returns at all horizons ---
    forward_returns = _compute_forward_returns(close, HORIZONS)

    # --- Align joint predictions with forward returns ---
    # joint index = prediction timestamps, forward_returns index = all timestamps
    common_idx = joint.index.intersection(forward_returns.index)
    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping timestamps between OOS predictions and close prices. "
            f"Joint range: {joint.index.min()} to {joint.index.max()}, "
            f"Close range: {close.index.min()} to {close.index.max()}"
        )

    logger.info(
        "Aligned %d / %d OOS predictions with close prices",
        len(common_idx), len(joint),
    )

    joint_aligned = joint.loc[common_idx]
    returns_aligned = forward_returns.loc[common_idx]

    prob_up = joint_aligned["prob_up"].values
    mag_pred = joint_aligned["mag_pred"].values

    # --- Evaluate each horizon ---
    rows = []
    for label, bars in HORIZONS.items():
        col = f"return_{label}"
        fwd_ret = returns_aligned[col].values

        metrics = _evaluate_horizon(prob_up, mag_pred, fwd_ret)
        metrics["horizon"] = label
        metrics["horizon_hours"] = bars  # 1 bar = 1 hour
        rows.append(metrics)

    decay_df = pd.DataFrame(rows)
    decay_df = decay_df[
        ["horizon", "horizon_hours", "n_samples", "dir_auc", "dir_spearman", "mag_ic"]
    ].sort_values("horizon_hours").reset_index(drop=True)

    # --- Estimate half-life ---
    half_life = _estimate_half_life(decay_df, metric="dir_auc", baseline=0.5)
    if half_life is not None:
        logger.info("Estimated AUC half-life: %.1f hours", half_life)
    else:
        logger.info("Could not estimate half-life (signal may persist beyond 24h)")

    # --- Save ---
    out_path = output_dir / "validation_signal_decay.csv"
    decay_df.to_csv(out_path, index=False)
    logger.info("Saved signal decay results to %s", out_path)

    # --- Print report ---
    print_signal_decay_report(decay_df, half_life=half_life)

    return decay_df


def print_signal_decay_report(
    df: pd.DataFrame,
    half_life: float | None = None,
) -> None:
    """
    Print a formatted signal decay report to stdout.

    Parameters
    ----------
    df : Output of run_signal_decay.
    half_life : Optional pre-computed half-life in hours.
    """
    print("\n" + "=" * 70)
    print("  LAYER 4: SIGNAL DECAY ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Horizon':<10s}  {'N':>6s}  {'AUC':>7s}  {'Spearman':>9s}  {'Mag IC':>7s}")
    print("  " + "-" * 46)

    for _, row in df.iterrows():
        auc_str = f"{row['dir_auc']:.4f}" if not np.isnan(row["dir_auc"]) else "   N/A"
        sp_str = f"{row['dir_spearman']:+.4f}" if not np.isnan(row["dir_spearman"]) else "    N/A"
        ic_str = f"{row['mag_ic']:+.4f}" if not np.isnan(row["mag_ic"]) else "   N/A"
        print(
            f"  {row['horizon']:<10s}  {row['n_samples']:>6.0f}  "
            f"{auc_str:>7s}  {sp_str:>9s}  {ic_str:>7s}"
        )

    # Identify peak
    valid_auc = df.dropna(subset=["dir_auc"])
    if len(valid_auc) > 0:
        peak_row = valid_auc.loc[valid_auc["dir_auc"].idxmax()]
        print(f"\n  Peak AUC: {peak_row['dir_auc']:.4f} at {peak_row['horizon']}")

    # Half-life
    if half_life is None and len(valid_auc) > 0:
        half_life = _estimate_half_life(df, metric="dir_auc", baseline=0.5)

    if half_life is not None:
        print(f"  AUC half-life: {half_life:.1f}h")
    else:
        print("  AUC half-life: could not estimate (signal persists or insufficient data)")

    # Interpretation
    print("\n  Interpretation:")
    if len(valid_auc) > 0:
        peak_h = int(peak_row["horizon_hours"])
        auc_4h = df.loc[df["horizon_hours"] == 4, "dir_auc"]
        auc_24h = df.loc[df["horizon_hours"] == 24, "dir_auc"]

        if peak_h <= 4:
            print("  [OK] Signal peaks at or before target horizon (4h)")
        else:
            print(f"  [WARN] Signal peaks at {peak_h}h, later than target 4h")

        if not auc_24h.empty and not np.isnan(auc_24h.values[0]):
            if auc_24h.values[0] > 0.55:
                print("  [WARN] Signal still strong at 24h -- possible data leakage?")
            elif auc_24h.values[0] < 0.52:
                print("  [OK] Signal decays to near-random at 24h (healthy)")
            else:
                print("  [INFO] Moderate signal remaining at 24h")

        if half_life is not None:
            if half_life < 2:
                print(f"  [WARN] Very short half-life ({half_life:.1f}h) -- signal may be noise")
            elif half_life <= 8:
                print(f"  [OK] Half-life {half_life:.1f}h is reasonable for 4h predictions")
            else:
                print(f"  [INFO] Long half-life ({half_life:.1f}h) -- investigate persistence")

    print("=" * 70 + "\n")


# ── CLI entrypoint ──────────────────────────────────────────────────────

def main():
    """Run signal decay analysis from saved OOS predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    results_dir = PROJECT_ROOT / "research" / "results" / "dual_model"

    # Look for joint OOS results from experiment 4
    joint_path = results_dir / "exp4_joint_oos.csv"
    if not joint_path.exists():
        print(f"Joint OOS file not found: {joint_path}")
        print("Run experiment 4 first: python -m research.dual_model.run_dual_model_experiments --experiment 4")
        sys.exit(1)

    print(f"Loading joint OOS from {joint_path}")
    joint = pd.read_csv(joint_path, parse_dates=["dt"])
    joint = joint.set_index("dt")

    if not DEFAULT_FEATURES_PATH.exists():
        print(f"Features cache not found: {DEFAULT_FEATURES_PATH}")
        print("Run shared_data.load_and_cache_data() first.")
        sys.exit(1)

    decay_df = run_signal_decay(
        joint=joint,
        features_path=DEFAULT_FEATURES_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    return decay_df


if __name__ == "__main__":
    main()
