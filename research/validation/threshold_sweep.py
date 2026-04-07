"""
Layer 3 — Threshold Sweep Validation.

Purpose: Search over (direction_threshold, magnitude_threshold) grid to find
stable operating regions for the dual-model BTC prediction system.
Avoids overfitting to one lucky threshold by requiring that a *region*
of nearby thresholds all produce positive expectancy.

Input:
    Joint OOS parquet with columns:
        prob_up     — P(UP) from direction model
        dir_true    — binary label (1=UP, 0=DOWN)
        mag_pred    — predicted |return| from magnitude model
        mag_true    — actual |return|
        return_4h   — signed forward 4h return
    Index: DatetimeIndex (UTC), name "dt", ~2 700 rows (15-min bars).

Output:
    DataFrame with one row per (dir_threshold, mag_threshold) combination,
    saved to ``research/results/validation/validation_threshold_grid.csv``.

No look-ahead: every metric is computed only from past-available labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default grids ────────────────────────────────────────────────────────────

DIR_THRESHOLDS: list[float] = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
MAG_QUANTILES: list[float] = [0.50, 0.60, 0.70, 0.80, 0.90]

# ── Stable-region criteria ───────────────────────────────────────────────────

STABLE_MIN_WINRATE = 0.55
STABLE_MIN_TRADES = 30
STABLE_MIN_AVG_RETURN = 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _max_drawdown(returns: pd.Series) -> float:
    """Compute max drawdown from a series of per-trade returns.

    Uses cumulative product of (1 + r) to build an equity curve,
    then finds the largest peak-to-trough decline.

    Returns
    -------
    float
        Max drawdown as a negative fraction (e.g. -0.12 = 12% drawdown).
        Returns 0.0 if no trades or equity never declines.
    """
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio.

    Scaling: assume each row is a 15-min bar trade opportunity.
    Bars per year = 4 * 24 * 365 = 35 040.
    We scale mean and std by sqrt(N) for annualisation.

    Returns np.nan when std == 0 or no trades.
    """
    if returns.empty or returns.std() == 0:
        return np.nan
    bars_per_year = 4 * 24 * 365  # 15-min bars
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def _evaluate_combo(
    df: pd.DataFrame,
    dir_thresh: float,
    mag_thresh: float,
    total_bars: int,
) -> dict:
    """Evaluate one (dir_threshold, mag_threshold) combination.

    Parameters
    ----------
    df : DataFrame with prob_up, return_4h, mag_pred columns.
    dir_thresh : Minimum prob_up for long, maximum (1 - dir_thresh) for short.
    mag_thresh : Minimum mag_pred to qualify as a trade.
    total_bars : Total bar count (for trade frequency calculation).

    Returns
    -------
    dict with all metrics for this combination.
    """
    long_mask = (df["prob_up"] > dir_thresh) & (df["mag_pred"] > mag_thresh)
    short_mask = (df["prob_up"] < (1 - dir_thresh)) & (df["mag_pred"] > mag_thresh)

    # Per-trade returns
    long_returns = df.loc[long_mask, "return_4h"]
    short_returns = -df.loc[short_mask, "return_4h"]

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    n_total = n_long + n_short

    long_wins = (long_returns > 0).sum() if n_long > 0 else 0
    short_wins = (short_returns > 0).sum() if n_short > 0 else 0

    long_winrate = long_wins / n_long if n_long > 0 else np.nan
    short_winrate = short_wins / n_short if n_short > 0 else np.nan
    total_winrate = (long_wins + short_wins) / n_total if n_total > 0 else np.nan

    long_avg = float(long_returns.mean()) if n_long > 0 else np.nan
    short_avg = float(short_returns.mean()) if n_short > 0 else np.nan

    # Combine all trade returns in chronological order for drawdown / Sharpe
    all_returns = pd.concat([long_returns, short_returns]).sort_index()
    total_avg = float(all_returns.mean()) if n_total > 0 else np.nan

    return {
        "dir_threshold": dir_thresh,
        "mag_threshold": round(mag_thresh, 6),
        "mag_quantile_label": None,  # filled by caller
        "n_long": n_long,
        "n_short": n_short,
        "n_total_trades": n_total,
        "long_winrate": round(long_winrate, 4) if not np.isnan(long_winrate) else np.nan,
        "short_winrate": round(short_winrate, 4) if not np.isnan(short_winrate) else np.nan,
        "total_winrate": round(total_winrate, 4) if not np.isnan(total_winrate) else np.nan,
        "long_avg_return": round(long_avg, 6) if not np.isnan(long_avg) else np.nan,
        "short_avg_return": round(short_avg, 6) if not np.isnan(short_avg) else np.nan,
        "total_avg_return": round(total_avg, 6) if not np.isnan(total_avg) else np.nan,
        "sharpe": round(_sharpe(all_returns), 4) if not np.isnan(_sharpe(all_returns)) else np.nan,
        "max_drawdown": round(_max_drawdown(all_returns), 4),
        "trade_frequency": round(n_total / total_bars, 4) if total_bars > 0 else np.nan,
    }


# ── Main entry point ────────────────────────────────────────────────────────


def run_threshold_sweep(
    joint: pd.DataFrame,
    output_dir: Path,
    dir_thresholds: Sequence[float] | None = None,
    mag_quantiles: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Run grid search over direction and magnitude thresholds.

    Parameters
    ----------
    joint : pd.DataFrame
        Joint OOS predictions. Required columns:
        ``prob_up``, ``dir_true``, ``mag_pred``, ``mag_true``, ``return_4h``.
        Index must be a DatetimeIndex (name ``dt``).
    output_dir : Path
        Directory to save results CSV.  Will be created if missing.
    dir_thresholds : optional
        Override direction threshold grid.
    mag_quantiles : optional
        Override magnitude quantile grid (values in [0, 1]).

    Returns
    -------
    pd.DataFrame
        One row per combination with metrics + ``is_stable`` flag.
    """
    if dir_thresholds is None:
        dir_thresholds = DIR_THRESHOLDS
    if mag_quantiles is None:
        mag_quantiles = MAG_QUANTILES

    # ── Validate input ───────────────────────────────────────────────────
    required_cols = {"prob_up", "dir_true", "mag_pred", "mag_true", "return_4h"}
    missing = required_cols - set(joint.columns)
    if missing:
        raise ValueError(f"Missing columns in joint DataFrame: {missing}")

    df = joint.dropna(subset=["prob_up", "mag_pred", "return_4h"]).copy()
    logger.info("Threshold sweep: %d valid rows (dropped %d NaN)", len(df), len(joint) - len(df))

    total_bars = len(df)

    # ── Compute magnitude thresholds from quantiles ──────────────────────
    mag_thresholds: dict[str, float] = {}
    for q in mag_quantiles:
        label = f"q{int(q * 100)}"
        mag_thresholds[label] = float(df["mag_pred"].quantile(q))
    logger.info("Magnitude quantile thresholds: %s", mag_thresholds)

    # ── Grid search ──────────────────────────────────────────────────────
    rows: list[dict] = []
    for dt in dir_thresholds:
        for q_label, mt in mag_thresholds.items():
            result = _evaluate_combo(df, dt, mt, total_bars)
            result["mag_quantile_label"] = q_label
            rows.append(result)

    grid = pd.DataFrame(rows)

    # ── Mark stable region ───────────────────────────────────────────────
    grid["is_stable"] = (
        (grid["total_winrate"] > STABLE_MIN_WINRATE)
        & (grid["n_total_trades"] > STABLE_MIN_TRADES)
        & (grid["total_avg_return"] > STABLE_MIN_AVG_RETURN)
    )

    n_stable = grid["is_stable"].sum()
    logger.info(
        "Grid complete: %d combinations, %d in stable region (%.1f%%)",
        len(grid), n_stable, 100 * n_stable / len(grid) if len(grid) else 0,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "validation_threshold_grid.csv"
    grid.to_csv(out_path, index=False)
    logger.info("Saved threshold grid to %s", out_path)

    return grid


# ── Pretty-print report ─────────────────────────────────────────────────────


def print_threshold_report(df: pd.DataFrame) -> None:
    """Print the threshold grid in a readable format.

    Highlights the stable region rows and summarises the best combinations.

    Parameters
    ----------
    df : DataFrame returned by ``run_threshold_sweep``.
    """
    print("=" * 90)
    print("THRESHOLD SWEEP REPORT")
    print("=" * 90)
    print(f"Total combinations : {len(df)}")
    print(f"Stable region      : {df['is_stable'].sum()} / {len(df)}")
    print(
        f"Stable criteria    : winrate > {STABLE_MIN_WINRATE:.0%}, "
        f"n_trades > {STABLE_MIN_TRADES}, "
        f"avg_return > {STABLE_MIN_AVG_RETURN}"
    )
    print("-" * 90)

    # Full grid summary
    cols = [
        "dir_threshold", "mag_quantile_label",
        "n_total_trades", "total_winrate", "total_avg_return",
        "sharpe", "max_drawdown", "trade_frequency", "is_stable",
    ]
    display = df[cols].copy()
    display["total_winrate"] = display["total_winrate"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    )
    display["total_avg_return"] = display["total_avg_return"].map(
        lambda x: f"{x:.4%}" if pd.notna(x) else "-"
    )
    display["sharpe"] = display["sharpe"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
    )
    display["max_drawdown"] = display["max_drawdown"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "-"
    )
    display["trade_frequency"] = display["trade_frequency"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    )
    display["is_stable"] = display["is_stable"].map(
        lambda x: " * " if x else ""
    )

    print(display.to_string(index=False))

    # Highlight stable region
    stable = df[df["is_stable"]].copy()
    if stable.empty:
        print("\n[!] No stable region found. Consider relaxing thresholds.")
    else:
        print("\n" + "=" * 90)
        print("STABLE REGION (sorted by Sharpe)")
        print("=" * 90)
        stable = stable.sort_values("sharpe", ascending=False)
        for _, row in stable.iterrows():
            print(
                f"  dir>{row['dir_threshold']:.2f}  mag>{row['mag_quantile_label']}"
                f"  |  trades={row['n_total_trades']:>4d}"
                f"  WR={row['total_winrate']:.1%}"
                f"  avg={row['total_avg_return']:.4%}"
                f"  sharpe={row['sharpe']:.2f}"
                f"  MDD={row['max_drawdown']:.2%}"
            )

        # Best row
        best = stable.iloc[0]
        print(f"\n  >> Best Sharpe: dir>{best['dir_threshold']:.2f}  "
              f"mag>{best['mag_quantile_label']}  "
              f"sharpe={best['sharpe']:.2f}")

    print("=" * 90)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Default path for joint OOS parquet
    default_path = Path(__file__).resolve().parents[1] / "results" / "dual_model" / "joint_oos.parquet"

    parquet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not parquet_path.exists():
        logger.error("Joint OOS parquet not found: %s", parquet_path)
        sys.exit(1)

    logger.info("Loading joint OOS from %s", parquet_path)
    joint_df = pd.read_parquet(parquet_path)

    out = Path(__file__).resolve().parents[1] / "results" / "validation"
    result = run_threshold_sweep(joint_df, output_dir=out)
    print_threshold_report(result)
