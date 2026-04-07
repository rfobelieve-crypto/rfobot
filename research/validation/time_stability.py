"""
Layer 1 Validation: Time Stability Analysis

Confirms the dual-model (Direction + Magnitude) prediction system does not
only work in certain time periods. Detects regime collapse by splitting
OOS predictions into monthly and rolling 2-week windows, then computing
per-window metrics for direction, magnitude, and joint signal quality.

No look-ahead bias: all inputs are out-of-sample predictions from
walk-forward validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: per-window metric calculations
# ---------------------------------------------------------------------------

def _direction_metrics(df: pd.DataFrame) -> dict:
    """Compute direction model metrics for a single time window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: prob_up, y_true.
        y_true is binary (1 = up, 0 = down).

    Returns
    -------
    dict with keys: dir_auc, dir_accuracy, dir_top_decile
    """
    n = len(df)
    if n < 10:
        return {"dir_auc": np.nan, "dir_accuracy": np.nan, "dir_top_decile": np.nan}

    y_true = df["y_true"].values
    prob_up = df["prob_up"].values

    # AUC — needs both classes present
    try:
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, prob_up)
    except Exception:
        auc = np.nan

    # Accuracy (threshold = 0.5)
    pred_label = (prob_up >= 0.5).astype(int)
    accuracy = float(np.mean(pred_label == y_true))

    # Top-decile precision: among the top 10% highest prob_up, what
    # fraction are actually up?
    top_k = max(1, int(np.ceil(n * 0.1)))
    top_idx = np.argsort(prob_up)[-top_k:]
    if len(top_idx) > 0:
        top_decile_precision = float(np.mean(y_true[top_idx] == 1))
    else:
        top_decile_precision = np.nan

    return {
        "dir_auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "dir_accuracy": round(accuracy, 4),
        "dir_top_decile": round(top_decile_precision, 4),
    }


def _magnitude_metrics(df: pd.DataFrame) -> dict:
    """Compute magnitude model metrics for a single time window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: y_pred, y_true, return_4h.

    Returns
    -------
    dict with keys: mag_ic, mag_icir
    """
    n = len(df)
    if n < 10:
        return {"mag_ic": np.nan, "mag_icir": np.nan}

    y_pred = df["y_pred"].values
    y_true = df["y_true"].values

    # Overall Spearman IC
    try:
        ic, _ = spearmanr(y_pred, y_true)
        if np.isnan(ic):
            ic = np.nan
    except Exception:
        ic = np.nan

    # ICIR: split window into ~weekly sub-windows, compute IC per sub,
    # then IC_mean / IC_std
    icir = _compute_icir(df, col_pred="y_pred", col_true="y_true")

    return {
        "mag_ic": round(ic, 4) if not np.isnan(ic) else np.nan,
        "mag_icir": round(icir, 4) if not np.isnan(icir) else np.nan,
    }


def _compute_icir(
    df: pd.DataFrame,
    col_pred: str,
    col_true: str,
    sub_window: str = "7D",
    min_sub_samples: int = 5,
    min_sub_windows: int = 2,
) -> float:
    """Compute ICIR (IC mean / IC std) over rolling sub-windows.

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex
    col_pred, col_true : column names
    sub_window : pandas offset string for sub-window grouping
    min_sub_samples : minimum samples per sub-window to compute IC
    min_sub_windows : minimum sub-windows needed for ICIR

    Returns
    -------
    float : ICIR value, or NaN if insufficient data
    """
    ics = []
    grouper = pd.Grouper(freq=sub_window)
    for _, sub_df in df.groupby(grouper):
        if len(sub_df) < min_sub_samples:
            continue
        try:
            ic_val, _ = spearmanr(sub_df[col_pred].values, sub_df[col_true].values)
            if not np.isnan(ic_val):
                ics.append(ic_val)
        except Exception:
            continue

    if len(ics) < min_sub_windows:
        return np.nan

    ic_mean = np.mean(ics)
    ic_std = np.std(ics, ddof=1)
    if ic_std == 0:
        return np.nan
    return ic_mean / ic_std


def _joint_metrics(df: pd.DataFrame) -> dict:
    """Compute joint signal metrics for a single time window.

    Active signal: prob_up > 0.6 AND mag_pred > median(mag_pred).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: prob_up, mag_pred, return_4h.

    Returns
    -------
    dict with keys: joint_return, joint_winrate
    """
    n = len(df)
    if n < 10:
        return {"joint_return": np.nan, "joint_winrate": np.nan}

    mag_median = df["mag_pred"].median()
    active = df[(df["prob_up"] > 0.6) & (df["mag_pred"] > mag_median)]

    if len(active) < 3:
        return {"joint_return": np.nan, "joint_winrate": np.nan}

    returns = active["return_4h"].values
    mean_ret = float(np.mean(returns))
    win_rate = float(np.mean(returns > 0))

    return {
        "joint_return": round(mean_ret, 6),
        "joint_winrate": round(win_rate, 4),
    }


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def _generate_monthly_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Generate monthly windows covering the data range."""
    windows = []
    periods = pd.date_range(start=start.normalize().replace(day=1), end=end, freq="MS")
    for ps in periods:
        pe = ps + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        pe = min(pe, end + pd.Timedelta(days=1))
        windows.append({
            "period": ps.strftime("%Y-%m"),
            "period_start": ps,
            "period_end": pe,
        })
    return windows


def _generate_rolling_windows(
    start: pd.Timestamp, end: pd.Timestamp, window_days: int = 14, step_days: int = 7
) -> list[dict]:
    """Generate rolling 2-week windows with 1-week step."""
    windows = []
    ws = start.normalize()
    idx = 0
    while ws < end:
        we = ws + pd.Timedelta(days=window_days)
        label = f"roll_{idx:02d}_{ws.strftime('%m%d')}-{min(we, end).strftime('%m%d')}"
        windows.append({
            "period": label,
            "period_start": ws,
            "period_end": min(we, end + pd.Timedelta(days=1)),
        })
        ws += pd.Timedelta(days=step_days)
        idx += 1
    return windows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_time_stability(
    dir_oos: pd.DataFrame,
    mag_oos: pd.DataFrame,
    joint: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run Layer 1 time stability validation.

    Splits OOS predictions by monthly and rolling 2-week windows, computes
    direction, magnitude, and joint metrics per window, marks best/worst
    periods, and saves results to CSV.

    Parameters
    ----------
    dir_oos : pd.DataFrame
        Direction model OOS predictions with DatetimeIndex (UTC).
        Required columns: prob_up, y_true, return_4h, fold.
    mag_oos : pd.DataFrame
        Magnitude model OOS predictions with DatetimeIndex (UTC).
        Required columns: y_pred, y_true, return_4h, fold.
    joint : pd.DataFrame
        Joint predictions with DatetimeIndex (UTC).
        Required columns: prob_up, dir_true, mag_pred, mag_true, return_4h.
    output_dir : Path
        Directory to save validation_time_stability.csv.

    Returns
    -------
    pd.DataFrame
        Validation results with one row per time window.
    """
    # Validate inputs
    for name, df, required_cols in [
        ("dir_oos", dir_oos, ["prob_up", "y_true"]),
        ("mag_oos", mag_oos, ["y_pred", "y_true", "return_4h"]),
        ("joint", joint, ["prob_up", "mag_pred", "return_4h"]),
    ]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex, got {type(df.index)}")

    # Determine overall data range
    all_starts = [df.index.min() for df in [dir_oos, mag_oos, joint]]
    all_ends = [df.index.max() for df in [dir_oos, mag_oos, joint]]
    data_start = min(all_starts)
    data_end = max(all_ends)

    logger.info(
        "Time stability analysis: %s to %s",
        data_start.strftime("%Y-%m-%d"),
        data_end.strftime("%Y-%m-%d"),
    )

    # Generate windows
    monthly = _generate_monthly_windows(data_start, data_end)
    rolling = _generate_rolling_windows(data_start, data_end)
    all_windows = monthly + rolling

    rows = []
    for win in all_windows:
        ps, pe = win["period_start"], win["period_end"]

        dir_slice = dir_oos[(dir_oos.index >= ps) & (dir_oos.index < pe)]
        mag_slice = mag_oos[(mag_oos.index >= ps) & (mag_oos.index < pe)]
        jnt_slice = joint[(joint.index >= ps) & (joint.index < pe)]

        n_samples = max(len(dir_slice), len(mag_slice), len(jnt_slice))
        if n_samples == 0:
            continue

        d_metrics = _direction_metrics(dir_slice)
        m_metrics = _magnitude_metrics(mag_slice)
        j_metrics = _joint_metrics(jnt_slice)

        row = {
            "period": win["period"],
            "period_start": ps.strftime("%Y-%m-%d"),
            "period_end": pe.strftime("%Y-%m-%d"),
            "n_samples": n_samples,
            **d_metrics,
            **m_metrics,
            **j_metrics,
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        logger.warning("No windows had any data. Returning empty DataFrame.")
        return result

    # Mark best / worst periods based on composite score
    # Composite: normalise each metric to [0,1] within its column, average
    metric_cols = ["dir_auc", "dir_accuracy", "mag_ic", "joint_return"]
    available = [c for c in metric_cols if c in result.columns]

    if available:
        normed = pd.DataFrame(index=result.index)
        for col in available:
            vals = result[col]
            vmin, vmax = vals.min(), vals.max()
            if vmax != vmin:
                normed[col] = (vals - vmin) / (vmax - vmin)
            else:
                normed[col] = 0.5
        result["_composite"] = normed.mean(axis=1)

        best_idx = result["_composite"].idxmax()
        worst_idx = result["_composite"].idxmin()
        result["mark"] = ""
        result.loc[best_idx, "mark"] = "BEST"
        result.loc[worst_idx, "mark"] = "WORST"
        result.drop(columns=["_composite"], inplace=True)
    else:
        result["mark"] = ""

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "validation_time_stability.csv"
    result.to_csv(out_path, index=False)
    logger.info("Saved time stability results to %s", out_path)

    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_time_stability_report(df: pd.DataFrame) -> None:
    """Print a formatted table of time stability validation results.

    Parameters
    ----------
    df : pd.DataFrame
        Output from run_time_stability().
    """
    if df.empty:
        print("No time stability data to display.")
        return

    print()
    print("=" * 110)
    print("  LAYER 1: TIME STABILITY VALIDATION")
    print("=" * 110)

    # Separate monthly and rolling windows for clarity
    monthly = df[~df["period"].str.startswith("roll_")]
    rolling = df[df["period"].str.startswith("roll_")]

    for label, subset in [("MONTHLY WINDOWS", monthly), ("ROLLING 2-WEEK WINDOWS", rolling)]:
        if subset.empty:
            continue

        print(f"\n--- {label} ---")
        header = (
            f"{'Period':<22} {'N':>5} "
            f"{'AUC':>6} {'Acc':>6} {'Top10':>6} "
            f"{'IC':>7} {'ICIR':>7} "
            f"{'JntRet':>9} {'WinR':>6} "
            f"{'':>5}"
        )
        print(header)
        print("-" * len(header))

        for _, row in subset.iterrows():
            mark = row.get("mark", "")
            mark_str = f" <{mark}>" if mark else ""

            def fmt(val, width=6, pct=False):
                if pd.isna(val):
                    return f"{'---':>{width}}"
                if pct:
                    return f"{val * 100:>{width}.1f}%"
                return f"{val:>{width}.4f}"

            line = (
                f"{row['period']:<22} {int(row['n_samples']):>5} "
                f"{fmt(row.get('dir_auc'), 6)} "
                f"{fmt(row.get('dir_accuracy'), 6)} "
                f"{fmt(row.get('dir_top_decile'), 6)} "
                f"{fmt(row.get('mag_ic'), 7)} "
                f"{fmt(row.get('mag_icir'), 7)} "
                f"{fmt(row.get('joint_return'), 9)} "
                f"{fmt(row.get('joint_winrate'), 6)} "
                f"{mark_str}"
            )
            print(line)

    # Summary statistics
    print(f"\n--- SUMMARY ---")
    for col, label in [
        ("dir_auc", "Direction AUC"),
        ("dir_accuracy", "Direction Accuracy"),
        ("mag_ic", "Magnitude IC"),
        ("mag_icir", "Magnitude ICIR"),
        ("joint_return", "Joint Mean Return"),
        ("joint_winrate", "Joint Win Rate"),
    ]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(
                    f"  {label:<22}: "
                    f"mean={vals.mean():.4f}  "
                    f"std={vals.std():.4f}  "
                    f"min={vals.min():.4f}  "
                    f"max={vals.max():.4f}"
                )

    # Flag periods of concern
    if "dir_auc" in df.columns:
        bad_auc = df[df["dir_auc"] < 0.5].dropna(subset=["dir_auc"])
        if not bad_auc.empty:
            periods = ", ".join(bad_auc["period"].tolist())
            print(f"\n  WARNING: Direction AUC < 0.5 in: {periods}")

    if "mag_ic" in df.columns:
        negative_ic = df[df["mag_ic"] < 0].dropna(subset=["mag_ic"])
        if not negative_ic.empty:
            periods = ", ".join(negative_ic["period"].tolist())
            print(f"  WARNING: Negative Magnitude IC in: {periods}")

    print("=" * 110)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 1: Time Stability Validation")
    parser.add_argument("--dir-oos", type=str, required=True, help="Path to direction OOS parquet")
    parser.add_argument("--mag-oos", type=str, required=True, help="Path to magnitude OOS parquet")
    parser.add_argument("--joint", type=str, required=True, help="Path to joint parquet")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/results/validation",
        help="Output directory for CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dir_oos = pd.read_parquet(args.dir_oos)
    mag_oos = pd.read_parquet(args.mag_oos)
    joint_df = pd.read_parquet(args.joint)

    # Ensure DatetimeIndex named "dt" if stored as column
    for name, df in [("dir_oos", dir_oos), ("mag_oos", mag_oos), ("joint", joint_df)]:
        if "dt" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("dt", inplace=True)

    result = run_time_stability(dir_oos, mag_oos, joint_df, Path(args.output_dir))
    print_time_stability_report(result)
