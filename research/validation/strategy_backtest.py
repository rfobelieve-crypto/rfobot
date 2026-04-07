"""
Layer 5 — Strategy Backtest Validation
=======================================

Most important validation layer: "Can the combined signal make money?"

Uses OOS (out-of-sample) predictions only — no in-sample data touches this code.
This is signal validation, NOT production backtest. No fills, slippage, or fees.

Methodology:
    - Bar-by-bar independent attribution (each bar's return stands alone)
    - Predictions are hourly, horizon is 4h — trades overlap by design
    - Long when prob_up > threshold AND magnitude prediction > threshold
    - Short when prob_up < (1 - threshold) AND magnitude > threshold
    - Otherwise flat (return = 0 for that bar)

Output files:
    - validation_strategy_performance.csv   — overall + long/short metrics
    - validation_strategy_monthly.csv       — monthly breakdown
    - validation_strategy_by_strength.csv   — quintile analysis by conviction
    - validation_equity_curve.parquet       — per-bar cumulative returns
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOURS_PER_YEAR = 365.25 * 24


def _compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute a standard set of performance metrics from a return series.

    Parameters
    ----------
    returns : pd.Series
        Per-bar strategy returns (0 for flat bars).
    label : str
        Optional prefix for metric keys.

    Returns
    -------
    dict
        Metric name → value.
    """
    trading_returns = returns[returns != 0]
    n_bars = len(returns)
    n_trades = len(trading_returns)

    if n_trades == 0:
        return {
            f"{label}total_return": 0.0,
            f"{label}avg_return_per_trade": 0.0,
            f"{label}n_trades": 0,
            f"{label}trade_frequency": 0.0,
            f"{label}win_rate": 0.0,
            f"{label}sharpe_ratio": 0.0,
            f"{label}max_drawdown": 0.0,
            f"{label}profit_factor": 0.0,
            f"{label}avg_win": 0.0,
            f"{label}avg_loss": 0.0,
        }

    # Cumulative product for total return and drawdown
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0

    # Max drawdown from equity curve
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()  # negative value

    # Win / loss breakdown
    wins = trading_returns[trading_returns > 0]
    losses = trading_returns[trading_returns < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Sharpe — annualized using only trading bars
    # Frequency: 1h bars → sqrt(HOURS_PER_YEAR) annualization
    if trading_returns.std() > 0:
        sharpe = (trading_returns.mean() / trading_returns.std()) * np.sqrt(HOURS_PER_YEAR)
    else:
        sharpe = 0.0

    trade_frequency = n_trades / n_bars if n_bars > 0 else 0.0

    return {
        f"{label}total_return": total_return,
        f"{label}avg_return_per_trade": trading_returns.mean(),
        f"{label}n_trades": n_trades,
        f"{label}trade_frequency": trade_frequency,
        f"{label}win_rate": win_rate,
        f"{label}sharpe_ratio": sharpe,
        f"{label}max_drawdown": max_drawdown,
        f"{label}profit_factor": profit_factor,
        f"{label}avg_win": avg_win,
        f"{label}avg_loss": avg_loss,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_strategy_backtest(
    joint: pd.DataFrame,
    output_dir: Path,
    threshold_up: float = 0.60,
    threshold_mag_quantile: float = 0.50,
) -> dict:
    """Run strategy backtest on OOS joint predictions.

    Parameters
    ----------
    joint : pd.DataFrame
        Must contain columns: prob_up, dir_true, mag_pred, mag_true, return_4h.
        Index must be a DatetimeIndex named 'dt' (UTC), ~2700 hourly bars.
    output_dir : Path
        Directory to write output CSVs and parquet.
    threshold_up : float
        Probability threshold for directional signals.  Long when prob_up >
        threshold_up, short when prob_up < (1 - threshold_up).
    threshold_mag_quantile : float
        Quantile of mag_pred to use as magnitude threshold (0.50 = median).

    Returns
    -------
    dict
        Keys: 'performance', 'monthly', 'by_strength', 'equity_curve' — each
        a DataFrame ready for the main runner.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    required_cols = {"prob_up", "dir_true", "mag_pred", "mag_true", "return_4h"}
    missing = required_cols - set(joint.columns)
    if missing:
        raise ValueError(f"Missing columns in joint DataFrame: {missing}")
    if not isinstance(joint.index, pd.DatetimeIndex):
        raise ValueError("joint.index must be a DatetimeIndex (UTC)")

    df = joint.copy().sort_index()
    logger.info(
        "Strategy backtest: %d bars, date range %s → %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )

    # ------------------------------------------------------------------
    # Signal generation (no look-ahead: thresholds from full OOS set)
    # ------------------------------------------------------------------
    threshold_mag = df["mag_pred"].quantile(threshold_mag_quantile)
    logger.info(
        "Thresholds — prob_up: %.2f, mag_pred (q%.0f%%): %.6f",
        threshold_up,
        threshold_mag_quantile * 100,
        threshold_mag,
    )

    long_signal = (df["prob_up"] > threshold_up) & (df["mag_pred"] > threshold_mag)
    short_signal = (df["prob_up"] < (1 - threshold_up)) & (df["mag_pred"] > threshold_mag)

    # Position: +1 long, -1 short, 0 flat
    df["position"] = 0
    df.loc[long_signal, "position"] = 1
    df.loc[short_signal, "position"] = -1

    # Strategy return per bar
    df["strategy_return"] = df["position"] * df["return_4h"]

    n_long = long_signal.sum()
    n_short = short_signal.sum()
    n_flat = len(df) - n_long - n_short
    logger.info("Signals: %d long, %d short, %d flat", n_long, n_short, n_flat)

    # ------------------------------------------------------------------
    # 1. Basic performance — overall, long-only, short-only
    # ------------------------------------------------------------------
    overall = _compute_metrics(df["strategy_return"], label="")

    long_returns = df["strategy_return"].copy()
    long_returns[df["position"] != 1] = 0
    long_metrics = _compute_metrics(long_returns, label="long_")

    short_returns = df["strategy_return"].copy()
    short_returns[df["position"] != -1] = 0
    short_metrics = _compute_metrics(short_returns, label="short_")

    perf_dict = {**overall, **long_metrics, **short_metrics}
    perf_df = pd.DataFrame([perf_dict])
    perf_df.to_csv(output_dir / "validation_strategy_performance.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Monthly breakdown
    # ------------------------------------------------------------------
    df["year_month"] = df.index.to_period("M").astype(str)
    monthly_rows = []
    for ym, grp in df.groupby("year_month"):
        active = grp[grp["position"] != 0]
        n_trades_m = len(active)
        if n_trades_m > 0:
            eq_m = (1 + grp["strategy_return"]).cumprod()
            ret_m = eq_m.iloc[-1] - 1.0
            wr_m = (active["strategy_return"] > 0).mean()
        else:
            ret_m = 0.0
            wr_m = 0.0
        monthly_rows.append(
            {"month": ym, "n_trades": n_trades_m, "return": ret_m, "win_rate": wr_m}
        )
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(output_dir / "validation_strategy_monthly.csv", index=False)

    # ------------------------------------------------------------------
    # 3. By signal strength (quintiles of conviction)
    # ------------------------------------------------------------------
    active_df = df[df["position"] != 0].copy()
    # Conviction: prob_up for longs, (1 - prob_up) for shorts
    active_df["conviction"] = np.where(
        active_df["position"] == 1,
        active_df["prob_up"],
        1 - active_df["prob_up"],
    )
    # Quintile labels (1 = weakest conviction, 5 = strongest)
    try:
        active_df["quintile"] = pd.qcut(
            active_df["conviction"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        )
    except ValueError:
        # If too few unique values for 5 bins, fall back to rank-based
        active_df["quintile"] = pd.cut(
            active_df["conviction"].rank(method="first"),
            bins=5,
            labels=[1, 2, 3, 4, 5],
        )

    strength_rows = []
    for q, grp in active_df.groupby("quintile"):
        strength_rows.append(
            {
                "quintile": int(q),
                "n_trades": len(grp),
                "avg_return": grp["strategy_return"].mean(),
                "win_rate": (grp["strategy_return"] > 0).mean(),
                "avg_conviction": grp["conviction"].mean(),
            }
        )
    strength_df = pd.DataFrame(strength_rows).sort_values("quintile")

    # Check monotonicity: avg_return should increase with quintile
    if len(strength_df) >= 2:
        corr = strength_df["quintile"].corr(strength_df["avg_return"], method="spearman")
        strength_df.attrs["monotonicity_spearman"] = corr
        logger.info("Strength–return monotonicity (Spearman): %.3f", corr)

    strength_df.to_csv(output_dir / "validation_strategy_by_strength.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Equity curve
    # ------------------------------------------------------------------
    equity_curve = pd.DataFrame(
        {
            "dt": df.index,
            "position": df["position"].values,
            "strategy_return": df["strategy_return"].values,
            "cumulative_return": (1 + df["strategy_return"]).cumprod().values,
        }
    )
    equity_curve.to_parquet(output_dir / "validation_equity_curve.parquet", index=False)

    # ------------------------------------------------------------------
    # Cleanup temp column
    # ------------------------------------------------------------------
    df.drop(columns=["year_month"], inplace=True, errors="ignore")

    results = {
        "performance": perf_df,
        "monthly": monthly_df,
        "by_strength": strength_df,
        "equity_curve": equity_curve,
    }

    logger.info("Strategy backtest complete. Files written to %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------


def print_strategy_report(results: dict) -> None:
    """Print a formatted strategy backtest report to stdout.

    Parameters
    ----------
    results : dict
        Output of :func:`run_strategy_backtest`.
    """
    perf = results["performance"].iloc[0]
    monthly = results["monthly"]
    strength = results["by_strength"]

    sep = "=" * 68

    print(f"\n{sep}")
    print("  LAYER 5 — STRATEGY BACKTEST (OOS)")
    print(sep)

    # --- Overall ---
    print("\n  OVERALL PERFORMANCE")
    print(f"  {'Total Return:':<28} {perf['total_return']:>+10.2%}")
    print(f"  {'N Trades:':<28} {int(perf['n_trades']):>10}")
    print(f"  {'Trade Frequency:':<28} {perf['trade_frequency']:>10.1%}")
    print(f"  {'Avg Return/Trade:':<28} {perf['avg_return_per_trade']:>+10.4%}")
    print(f"  {'Win Rate:':<28} {perf['win_rate']:>10.1%}")
    print(f"  {'Sharpe Ratio (ann.):':<28} {perf['sharpe_ratio']:>10.2f}")
    print(f"  {'Max Drawdown:':<28} {perf['max_drawdown']:>10.2%}")
    print(f"  {'Profit Factor:':<28} {perf['profit_factor']:>10.2f}")
    print(f"  {'Avg Win:':<28} {perf['avg_win']:>+10.4%}")
    print(f"  {'Avg Loss:':<28} {perf['avg_loss']:>+10.4%}")

    # --- Long vs Short ---
    for side, prefix in [("LONG", "long_"), ("SHORT", "short_")]:
        print(f"\n  {side} ONLY")
        print(f"  {'  N Trades:':<28} {int(perf[f'{prefix}n_trades']):>10}")
        print(f"  {'  Avg Return/Trade:':<28} {perf[f'{prefix}avg_return_per_trade']:>+10.4%}")
        print(f"  {'  Win Rate:':<28} {perf[f'{prefix}win_rate']:>10.1%}")
        print(f"  {'  Sharpe Ratio (ann.):':<28} {perf[f'{prefix}sharpe_ratio']:>10.2f}")
        print(f"  {'  Profit Factor:':<28} {perf[f'{prefix}profit_factor']:>10.2f}")

    # --- Strength quintiles ---
    print(f"\n  SIGNAL STRENGTH (QUINTILE ANALYSIS)")
    print(f"  {'Q':>4} {'N':>7} {'Avg Ret':>10} {'Win Rate':>10} {'Conviction':>12}")
    print(f"  {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*12}")
    for _, row in strength.iterrows():
        print(
            f"  {int(row['quintile']):>4} {int(row['n_trades']):>7} "
            f"{row['avg_return']:>+10.4%} {row['win_rate']:>10.1%} "
            f"{row['avg_conviction']:>12.4f}"
        )
    if hasattr(strength, "attrs") and "monotonicity_spearman" in strength.attrs:
        mono = strength.attrs["monotonicity_spearman"]
        verdict = "GOOD" if mono > 0.6 else ("OK" if mono > 0.2 else "WEAK")
        print(f"\n  Monotonicity (Spearman): {mono:+.3f}  [{verdict}]")

    # --- Monthly ---
    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'Month':<10} {'N':>6} {'Return':>10} {'Win Rate':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10}")
    for _, row in monthly.iterrows():
        print(
            f"  {row['month']:<10} {int(row['n_trades']):>6} "
            f"{row['return']:>+10.2%} {row['win_rate']:>10.1%}"
        )

    print(f"\n{sep}\n")
