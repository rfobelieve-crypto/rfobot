"""
Paper-trading baseline (Stage 1 of indicator → quant migration).

NOT execution code.  NOT live trading.  This is a pure offline simulation
that asks one question:

    "If I had blindly opened a 4h-hold position at every Strong/Moderate
     signal, taking taker fees + realistic slippage + funding, what would
     my net PnL look like — and which sub-segments are net positive?"

The answer dictates whether walking the indicator → trading path is
viable, and if so, what filters / regimes / tiers to focus on.

Reads:
    MySQL tracked_signals — filled=1 only
    (entry_price + exit_price + actual_return_4h are pre-computed by
     signal_tracker.backfill_outcomes; we don't recompute them here.)

Slices reported (per direction × per tier × per regime × per conf-quintile):
    n, WR, avg_endpoint_pnl_bps, avg_net_pnl_bps, sharpe, MDD, profit_factor

Two PnL definitions reported side-by-side:
    endpoint = (exit_price / entry_price - 1) × dir_sign  — real "hold 4h"
    twap     = actual_return_4h × dir_sign               — model training target

Cost model (round-trip, in bps):
    taker fee  : 10 bps   (Binance perp 5 bps × 2 sides)
    slippage   :  2 bps   (1 bp per side at typical sizes)
    funding    :  1 bps   (4h hold ≈ half of 8h funding cycle, avg funding ~10bp/yr)
    TOTAL      : 13 bps   ← hard for any signal to overcome

Outputs:
    research/results/paper_trading_baseline.csv   # one row per slice
    Console table sorted by net_pnl_per_trade.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.db import get_db_conn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cost model — round-trip total in bps (1 bp = 0.0001).
COST_BPS = 13.0
COST = COST_BPS / 10000.0


def fetch_signals() -> pd.DataFrame:
    """Pull all settled tracked_signals into a DataFrame."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, strength, confidence,
                       entry_price, exit_price, actual_return_4h, regime,
                       mag_pct_200, model_version
                FROM tracked_signals
                WHERE filled = 1
                  AND entry_price IS NOT NULL
                  AND exit_price  IS NOT NULL
                  AND actual_return_4h IS NOT NULL
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No filled signals returned from tracked_signals")
    df["entry_price"] = df["entry_price"].astype(float)
    df["exit_price"] = df["exit_price"].astype(float)
    df["actual_return_4h"] = df["actual_return_4h"].astype(float)
    df["confidence"] = df["confidence"].astype(float)
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    return df


def annotate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signed PnL and net PnL after cost for both endpoint and TWAP."""
    sign = np.where(df["direction"] == "UP", 1.0, -1.0)
    df["endpoint_ret"] = (df["exit_price"] / df["entry_price"] - 1.0) * sign
    df["twap_ret"] = df["actual_return_4h"] * sign
    df["endpoint_net"] = df["endpoint_ret"] - COST
    df["twap_net"] = df["twap_ret"] - COST
    df["win_endpoint"] = (df["endpoint_ret"] > 0).astype(int)
    return df


def slice_metrics(group: pd.DataFrame, label: str) -> dict:
    n = len(group)
    if n == 0:
        return None
    e_net = group["endpoint_net"].values
    e_gross = group["endpoint_ret"].values
    cum = np.cumsum(e_net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max  # in return space (additive)
    mdd_pct = float(drawdown.min())  # most-negative cumulative drawdown
    wins = e_net[e_net > 0]
    losses = e_net[e_net < 0]
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    return {
        "slice": label,
        "n": n,
        "wr_endpoint": float(group["win_endpoint"].mean()),
        "avg_endpoint_bps": float(group["endpoint_ret"].mean()) * 10000,
        "avg_twap_bps": float(group["twap_ret"].mean()) * 10000,
        "avg_net_bps": float(e_net.mean()) * 10000,
        "median_net_bps": float(np.median(e_net)) * 10000,
        "sharpe_per_trade": float(e_net.mean() / e_net.std()) if e_net.std() > 0 else np.nan,
        "cum_net_pct": float(cum[-1]) * 100,
        "mdd_pct": mdd_pct * 100,
        "profit_factor": float(profit_factor),
    }


def report_slices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(slice_metrics(df, "ALL"))

    for tier in ["Strong", "Moderate"]:
        sub = df[df["strength"] == tier]
        if len(sub) >= 20:
            rows.append(slice_metrics(sub, f"{tier}"))
            for direction in ["UP", "DOWN"]:
                ssd = sub[sub["direction"] == direction]
                if len(ssd) >= 10:
                    rows.append(slice_metrics(ssd, f"{tier} / {direction}"))

    # Regime slice (Strong only — too noisy on Moderate)
    strong = df[df["strength"] == "Strong"]
    for regime, ssr in strong.groupby("regime"):
        if len(ssr) >= 20:
            rows.append(slice_metrics(ssr, f"Strong / regime={regime or 'NULL'}"))

    # Confidence quintile slice (use full df, quintiles per tier)
    for tier in ["Strong", "Moderate"]:
        sub = df[df["strength"] == tier].copy()
        if len(sub) < 50:
            continue
        sub["conf_q"] = pd.qcut(
            sub["confidence"], q=5, labels=["Q1_lo", "Q2", "Q3", "Q4", "Q5_hi"],
            duplicates="drop",
        )
        for q, ssq in sub.groupby("conf_q", observed=True):
            if len(ssq) >= 10:
                rows.append(slice_metrics(ssq, f"{tier} / conf={q}"))

    # mag_pct_200 quintile (Strong only — Moderate has many NULLs)
    sub = df[(df["strength"] == "Strong") & df["mag_pct_200"].notna()].copy()
    if len(sub) >= 50:
        sub["mag_q"] = pd.qcut(
            sub["mag_pct_200"], q=5, labels=["Q1_lo", "Q2", "Q3", "Q4", "Q5_hi"],
            duplicates="drop",
        )
        for q, ssq in sub.groupby("mag_q", observed=True):
            if len(ssq) >= 10:
                rows.append(slice_metrics(ssq, f"Strong / magp200={q}"))

    return pd.DataFrame([r for r in rows if r is not None])


def print_table(metrics: pd.DataFrame):
    print()
    print(f"{'slice':<32} {'n':>5} {'WR':>6} "
          f"{'avg_ep':>8} {'avg_twap':>9} {'net_bps':>8} "
          f"{'sharpe':>7} {'cum_pct':>8} {'MDD%':>7} {'PF':>6}")
    print("-" * 110)
    for _, r in metrics.iterrows():
        print(f"{r['slice']:<32} {int(r['n']):>5} "
              f"{r['wr_endpoint']*100:>5.1f}% "
              f"{r['avg_endpoint_bps']:>+7.1f} "
              f"{r['avg_twap_bps']:>+8.1f} "
              f"{r['avg_net_bps']:>+7.1f} "
              f"{r['sharpe_per_trade']:>+7.3f} "
              f"{r['cum_net_pct']:>+7.1f}% "
              f"{r['mdd_pct']:>+6.1f}% "
              f"{r['profit_factor']:>5.2f}")


def main():
    logger.info("Fetching tracked_signals...")
    df = fetch_signals()
    logger.info("Loaded %d filled signals (%s ~ %s)",
                len(df), df["signal_time"].min(), df["signal_time"].max())

    df = annotate_pnl(df)

    overall_n = len(df)
    overall_avg_ep = df["endpoint_ret"].mean() * 10000
    overall_avg_twap = df["twap_ret"].mean() * 10000
    print(f"\n=== Cost model ===")
    print(f"  Round-trip cost: {COST_BPS:.1f} bps "
          f"(taker 10 + slippage 2 + funding 1)")
    print(f"  Hurdle: every signal must average > +{COST_BPS:.1f} bps "
          f"endpoint return to be net-positive.")

    print(f"\n=== Sanity ===")
    print(f"  n = {overall_n}")
    print(f"  avg endpoint = {overall_avg_ep:+.2f} bps  "
          f"(should be near production /signal-perf avg × 10000)")
    print(f"  avg TWAP     = {overall_avg_twap:+.2f} bps")
    print(f"  endpoint vs twap diff = {overall_avg_ep - overall_avg_twap:+.2f} bps "
          f"(small ~ TWAP smoothing of endpoint)")

    metrics = report_slices(df)
    metrics = metrics.sort_values("avg_net_bps", ascending=False)
    print_table(metrics)

    out_path = PROJECT_ROOT / "research" / "results" / "paper_trading_baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)
    logger.info("Saved metrics → %s", out_path)


if __name__ == "__main__":
    main()
