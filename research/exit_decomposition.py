"""Exit reason decomposition + regret distribution.

Step 1: group closed backtest trades by exit_reason → WR / avg_net_bps /
        count / share.

Step 2: for each trade, compute "regret" = bps that *would have* been earned
        if the position were held an extra h hours past its actual exit.
        Aggregate regret by exit_reason × horizon, then flag the worst
        regret quintile and inspect what's common about those trades.

This is the systematic Step 1 + Step 2 of the exit-optimization framework
(see exit_oracle_analysis.py for the absolute oracle gap).  Step 3
(predictive features at exit time) is left as a follow-up.

Why this matters: optimization of a deterministic exit rule has saturated
on every variant tested in exit_variants_backtest.py.  The remaining
lever is CONDITIONAL exit rules — different behaviour depending on
context.  This script tells you which exit_reason has the highest
regret cluster, which is where conditional logic has the most ROI.

Usage:
    python research/exit_decomposition.py
    python research/exit_decomposition.py --horizons 4,12,24
    python research/exit_decomposition.py --entry next_open

Read-only.  Writes a CSV summary to
research/results/exit_decomposition_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import research.v71_v7_sizing_1x as bt


def compute_regret(trades: pd.DataFrame, df: pd.DataFrame,
                   horizons_h: list[int]) -> pd.DataFrame:
    """For each trade, compute the bps you would have earned if held h
    additional hours past the actual exit.  Positive = exited too early.
    """
    ts_to_idx = {ts: i for i, ts in enumerate(df.index)}
    closes = df["close"].values
    n = len(df)

    rows = []
    for _, tr in trades.iterrows():
        exit_ts = tr["exit_ts"]
        exit_idx = ts_to_idx.get(exit_ts)
        if exit_idx is None:
            continue
        side = tr["side"]
        exit_px = float(tr["exit_price"])

        regret = {}
        for h in horizons_h:
            future_idx = exit_idx + h
            if future_idx >= n:
                regret[f"regret_{h}h_bps"] = np.nan
                continue
            future_px = float(closes[future_idx])
            if side == "LONG":
                ret = (future_px - exit_px) / exit_px
            else:  # SHORT — flip sign
                ret = (exit_px - future_px) / exit_px
            regret[f"regret_{h}h_bps"] = ret * 10000

        rows.append({
            "exit_ts": exit_ts,
            "entry_ts": tr["entry_ts"],
            "side": side,
            "tier": tr["tier"],
            "exit_reason": tr["exit_reason"],
            "bars_held": tr["bars_held"],
            "net_bps": float(tr["net_pct"]) * 10000,
            "gross_bps": float(tr["gross_pct"]) * 10000,
            "win": bool(tr["win"]),
            **regret,
        })
    return pd.DataFrame(rows)


def step1_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """One row per exit_reason: n / share / WR / avg_net_bps."""
    g = df.groupby("exit_reason")
    out = g.agg(
        n=("net_bps", "size"),
        wr=("win", "mean"),
        avg_net_bps=("net_bps", "mean"),
        median_net_bps=("net_bps", "median"),
        cum_net_pct=("net_bps", lambda s: s.sum() / 100),
        avg_bars_held=("bars_held", "mean"),
    ).round(2)
    out["share_pct"] = (out["n"] / out["n"].sum() * 100).round(1)
    out["wr_pct"] = (out["wr"] * 100).round(1)
    return out.drop(columns=["wr"]).sort_values("n", ascending=False)


def step2_regret(df: pd.DataFrame, horizons_h: list[int]) -> pd.DataFrame:
    """Regret stats per exit_reason × horizon."""
    rows = []
    for reason, grp in df.groupby("exit_reason"):
        row = {"exit_reason": reason, "n": len(grp)}
        for h in horizons_h:
            col = f"regret_{h}h_bps"
            vals = grp[col].dropna()
            if len(vals) == 0:
                row[f"r{h}h_mean"] = np.nan
                row[f"r{h}h_median"] = np.nan
                row[f"r{h}h_p75"] = np.nan
                continue
            row[f"r{h}h_mean"] = round(float(vals.mean()), 1)
            row[f"r{h}h_median"] = round(float(vals.median()), 1)
            row[f"r{h}h_p75"] = round(float(vals.quantile(0.75)), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def worst_regret_metadata(df: pd.DataFrame, h: int = 4,
                          quintile: float = 0.8) -> pd.DataFrame:
    """For the worst-regret quintile (regret >= 80th percentile), show
    what's common — exit_reason / side / tier / hour-of-day mix."""
    col = f"regret_{h}h_bps"
    if col not in df.columns:
        return pd.DataFrame()
    vals = df[col].dropna()
    if len(vals) < 10:
        return pd.DataFrame()
    threshold = float(vals.quantile(quintile))
    worst = df[df[col] >= threshold].copy()
    worst["hour_of_day"] = pd.to_datetime(worst["exit_ts"]).dt.hour

    summary = {
        "n_worst": len(worst),
        "threshold_bps": round(threshold, 1),
        "exit_reason_mix": worst["exit_reason"].value_counts().to_dict(),
        "side_mix": worst["side"].value_counts().to_dict(),
        "tier_mix": worst["tier"].value_counts().to_dict(),
        "hour_top3": worst["hour_of_day"].value_counts().head(3).to_dict(),
        "avg_bars_held": round(float(worst["bars_held"].mean()), 1),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="4,12,24",
                    help="Comma-separated hours to compute regret over")
    ap.add_argument("--entry", choices=["signal_close", "next_open"],
                    default="signal_close")
    args = ap.parse_args()

    horizons_h = [int(x) for x in args.horizons.split(",") if x.strip()]

    print(f"Loading data + decoding signals (entry={args.entry}) …")
    df = bt.load_data()
    direction, tier, _ = bt.decode_signals(df)

    print("Running backtest to generate trade list …")
    trades = bt.simulate(df, direction, tier, entry_mode=args.entry)
    if len(trades) == 0:
        print("No trades generated. Abort.")
        return 1
    print(f"  {len(trades)} trades generated over the OOS window")

    print(f"Computing regret over horizons {horizons_h}h …")
    enriched = compute_regret(trades, df, horizons_h)
    print(f"  {len(enriched)} trades enriched with regret")

    # ── Step 1
    print("\n" + "=" * 80)
    print("STEP 1 — Exit reason decomposition")
    print("=" * 80)
    s1 = step1_decomposition(enriched)
    print(s1.to_string())

    # ── Step 2
    print("\n" + "=" * 80)
    print("STEP 2 — Regret distribution per exit_reason")
    print(f"         (regret = bps you'd have earned if held h hours longer)")
    print("=" * 80)
    s2 = step2_regret(enriched, horizons_h)
    print(s2.to_string(index=False))

    # ── Worst-regret cluster
    for h in horizons_h:
        print(f"\n--- Worst-regret quintile @ +{h}h horizon ---")
        meta = worst_regret_metadata(enriched, h=h)
        if not meta:
            print("  insufficient data")
            continue
        print(f"  n_worst         = {meta['n_worst']}")
        print(f"  threshold (bps) = {meta['threshold_bps']}")
        print(f"  exit_reason mix = {meta['exit_reason_mix']}")
        print(f"  side mix        = {meta['side_mix']}")
        print(f"  tier mix        = {meta['tier_mix']}")
        print(f"  hour top-3      = {meta['hour_top3']}")
        print(f"  avg bars held   = {meta['avg_bars_held']}")

    # Save CSV
    out_csv = PROJECT_ROOT / "research" / "results" / "exit_decomposition_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_csv, index=False)
    print(f"\nWrote per-trade enriched table → {out_csv}")
    print("\n--- INTERPRETATION HINTS ---")
    print("  · High r4h_mean by exit_reason → that reason is exiting too early")
    print("  · Trail vs opp_signal mean comparison → which mechanism leaks more")
    print("  · Side mix in worst quintile asymmetric → directional exit logic flaw")
    print("  · Hour clustering → time-of-day conditional exit candidate")
    print("  · Tier clustering → strength-conditional exit candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
