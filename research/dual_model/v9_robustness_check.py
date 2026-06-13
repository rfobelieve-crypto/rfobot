"""
v9 robustness check on the H=8 SHORT @ P>=0.65 finding.

Three tests:
    1. Bootstrap 95% CI on actual WR at P>=0.65 — is 56.6% > 51% BE statistically robust?
    2. Time split (front half vs back half) — is the edge present in both halves
       or concentrated in one period (regime artifact)?
    3. OHLC backtest — when we apply real 1h high/low to compute trade outcomes
       (instead of just using clean label), does WR stay above BE?  Noise / ambiguous
       bars may pull it down 1-2 pp.

Inputs:
    direction_v9_winrate_H{horizon}_oos.parquet  (default H=8)
    binance_klines_1h.parquet  (for OHLC simulation)
"""
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TP_DIST = 0.005
SL_DIST = 0.003
COST_BPS = 13.0
COST = COST_BPS / 10000.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 5:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ohlc_backtest_signals(signals: pd.DataFrame, klines: pd.DataFrame,
                          direction: str, horizon_bars: int) -> pd.DataFrame:
    """For each signal timestamp, simulate trade with real OHLC TP/SL rule.
    Returns trade-level results."""
    rows = []
    klines_idx = klines.index
    for ts, _ in signals.iterrows():
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        entry_price = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry_price,
            direction=direction,
            sl_dist=SL_DIST,
            rr=TP_DIST / SL_DIST,
            bars=future,
            timeout_bars=horizon_bars,
        )
        sign = 1.0 if direction == "UP" else -1.0
        gross = (exit_price / entry_price - 1.0) * sign
        net = gross - COST
        rows.append({
            "ts": ts, "exit_price": exit_price, "bars_held": bars_held,
            "reason": reason, "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def report_block(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"  {label}: n=0")
        return
    n = len(df)
    wr = df["win"].mean()
    net = df["net_ret"].values
    avg_net = net.mean() * 10000
    sharpe = net.mean() / net.std() if net.std() > 0 else 0.0
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    # bootstrap CI on WR
    wins = df["win"].values
    lo, hi = bootstrap_ci(wins.astype(float))
    print(f"  {label:<32} n={n:>4} WR={wr*100:>5.1f}% CI=[{lo*100:>4.1f}, {hi*100:>4.1f}]  "
          f"net={avg_net:>+6.1f} bps  Sharpe={sharpe:>+6.3f}  cum={cum[-1]*100:>+5.1f}%  MDD={mdd:>+5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--side", type=str, default="short", choices=["long", "short"])
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    oos_path = RESULTS_DIR / f"direction_v9_winrate_H{args.horizon}_oos.parquet"
    if not oos_path.exists():
        logger.error("OOS file not found: %s", oos_path)
        sys.exit(1)

    oos = pd.read_parquet(oos_path)
    logger.info("Loaded v9 OOS H=%d: n=%d", args.horizon, len(oos))

    klines = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    pred_col = f"p_{args.side}_win"
    label_col = f"y_{args.side}_win"
    direction = "UP" if args.side == "long" else "DOWN"

    selected = oos[oos[pred_col] >= args.threshold].copy()
    selected["ts"] = pd.to_datetime(selected["ts"])
    if selected["ts"].dt.tz is not None:
        selected["ts"] = selected["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    selected = selected.set_index("ts").sort_index()
    logger.info("Selected signals at P>=%.2f: n=%d", args.threshold, len(selected))

    print(f"\n{'='*100}")
    print(f"v9 Robustness Check — H={args.horizon}h, side={args.side.upper()}, P>={args.threshold}")
    print(f"TP={TP_DIST*100:.1f}% SL={SL_DIST*100:.1f}% cost={COST_BPS:.0f}bps  BE WR={51.0:.0f}%")
    print(f"{'='*100}")

    # ── Test 1: Clean label WR + bootstrap CI ──
    print(f"\n--- Test 1: Clean-label WR with bootstrap CI ---")
    if len(selected) > 0:
        y = selected[label_col].astype(float).values
        wr = y.mean()
        lo, hi = bootstrap_ci(y)
        print(f"  WR = {wr*100:.2f}%  95% CI = [{lo*100:.2f}, {hi*100:.2f}]")
        if lo > 0.51:
            print(f"  ✓ CI lower bound {lo*100:.1f}% > 51% BE — STATISTICALLY DEFENSIBLE EDGE")
        elif wr > 0.51:
            print(f"  o WR {wr*100:.1f}% > 51% BE but CI lower {lo*100:.1f}% < 51% — EDGE PLAUSIBLE BUT NOT CONFIRMED")
        else:
            print(f"  x WR < 51% — no edge")
    else:
        print("  (n=0)")

    # ── Test 2: Time split ──
    print(f"\n--- Test 2: Time split (front half vs back half) ---")
    if len(selected) >= 10:
        mid = len(selected) // 2
        front = selected.iloc[:mid]
        back = selected.iloc[mid:]
        for label, sub in [("first half", front), ("second half", back)]:
            y = sub[label_col].astype(float).values
            wr = y.mean()
            lo, hi = bootstrap_ci(y)
            print(f"  {label:<14} n={len(sub):>3}  WR={wr*100:>5.1f}%  CI=[{lo*100:>4.1f}, {hi*100:>4.1f}]  "
                  f"range {sub.index.min().date()} ~ {sub.index.max().date()}")
        front_wr = front[label_col].astype(float).mean()
        back_wr = back[label_col].astype(float).mean()
        if abs(front_wr - back_wr) < 0.10:
            print(f"  ✓ Front {front_wr*100:.1f}% vs Back {back_wr*100:.1f}% — within 10pp, consistent")
        else:
            print(f"  x Front {front_wr*100:.1f}% vs Back {back_wr*100:.1f}% — > 10pp gap, CONCENTRATED edge (regime artifact?)")
    else:
        print("  (sample too small)")

    # ── Test 3: OHLC backtest with real path simulation ──
    print(f"\n--- Test 3: OHLC backtest (real 1h high/low TP/SL simulation) ---")
    trades = ohlc_backtest_signals(selected, klines, direction, args.horizon)
    if not trades.empty:
        report_block(trades, f"all signals (n={len(trades)})")
        # breakdown by exit reason
        for reason in trades["reason"].unique():
            sub = trades[trades["reason"] == reason]
            report_block(sub, f"  reason={reason}")
    else:
        print("  (no trades)")

    # ── Compare to baseline: random entry at all signals (no filter) ──
    print(f"\n--- Test 4: Sanity — compare to all-bar baseline (any P) ---")
    all_oos = oos.copy()
    all_oos["ts"] = pd.to_datetime(all_oos["ts"])
    if all_oos["ts"].dt.tz is not None:
        all_oos["ts"] = all_oos["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    all_oos = all_oos.set_index("ts")
    baseline_trades = ohlc_backtest_signals(all_oos, klines, direction, args.horizon)
    report_block(baseline_trades, f"baseline {args.side} (every bar)")
    report_block(trades, f"v9 filtered (P>={args.threshold})")


if __name__ == "__main__":
    main()
