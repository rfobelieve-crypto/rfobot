"""
Hybrid strategy: v12 regime filter + v9 direction prediction.

Hypothesis:
    v12 (3-class regime) shows AUC 0.64 for "no_trend" detection — much
    stronger than direction prediction (0.56).  Use it as a TRADING FILTER:
    only act when v12 says "NOT no_trend" (i.e., regime is trending).

    For direction within those trending bars, fall back to v9 P(short_win).

Trade rule (for SHORT side only, since v9 LONG AUC ~0.50):
    if v12_p_no_trend < T_no_trend         (regime says trending)
        AND v9_p_short_win >= T_short      (direction says short edge)
        → take SHORT trade

Comparison:
    Pure v9 baseline   : v9 P(short_win) >= 0.65, no regime filter
    Hybrid v12+v9      : both filters active
    Pure v12 SHORT     : v12 argmax == dn AND P(dn) >= 0.5
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_tpsl import _find_exit_for_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
KLINES_PATH = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

V12_OOS = RESULTS_DIR / "direction_v12_regime_T120_H8_oos.parquet"
V9_OOS = RESULTS_DIR / "direction_v9_winrate_H8_oos.parquet"

TP_DIST = 0.005   # 0.5%
SL_DIST = 0.003   # 0.3%
MAKER_COST = 0.0005   # 5 bps
TAKER_COST = 0.0013   # 13 bps


def load_oos():
    v12 = pd.read_parquet(V12_OOS)
    v9 = pd.read_parquet(V9_OOS)
    v12["ts"] = pd.to_datetime(v12["ts"])
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v12["ts"].dt.tz is not None:
        v12["ts"] = v12["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    # Join on ts
    merged = pd.merge(v9, v12, on="ts", how="inner", suffixes=("_v9", "_v12"))
    return merged.sort_values("ts").reset_index(drop=True)


def load_klines():
    k = pd.read_parquet(KLINES_PATH)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    return k


def ohlc_backtest_short(df: pd.DataFrame, klines: pd.DataFrame,
                          tp: float, sl: float, horizon: int,
                          cost: float) -> pd.DataFrame:
    """Run OHLC TP/SL backtest on the SHORT signals in df.

    df should have columns: ts (filtered), 1 row = 1 signal.
    Returns per-trade results.
    """
    rows = []
    klines_idx = klines.index
    for ts in df["ts"]:
        if ts not in klines_idx:
            continue
        entry = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry, direction="DOWN",
            sl_dist=sl, rr=tp / sl,
            bars=future, timeout_bars=horizon,
        )
        gross = -(exit_price / entry - 1.0)
        net = gross - cost
        rows.append({
            "ts": ts, "exit_price": exit_price, "bars_held": bars_held,
            "reason": reason, "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str):
    n = len(df)
    if n == 0:
        print(f"  {label:<48} (n=0)")
        return
    wr = df["win"].mean()
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    sharpe = net.mean() / net.std() if net.std() > 0 else 0.0
    avg_net = net.mean() * 10000
    avg_gross = df["gross_ret"].mean() * 10000
    # Bootstrap WR 95% CI
    wins_int = int(df["win"].sum())
    ci = binomtest(wins_int, n).proportion_ci(0.95) if n >= 5 else None
    ci_str = f"[{ci.low*100:.1f}, {ci.high*100:.1f}]" if ci else ""
    print(f"  {label:<48} n={n:>4} WR={wr*100:>5.1f}% {ci_str:<18}  "
          f"gross={avg_gross:>+6.1f}bp net={avg_net:>+6.1f}bp  "
          f"cum={cum[-1]*100:>+5.1f}% MDD={mdd:>+5.1f}% PF={pf:>4.2f}")


def main():
    df = load_oos()
    klines = load_klines()
    logger.info("Joined OOS: n=%d, range %s ~ %s",
                len(df), df["ts"].min(), df["ts"].max())

    # ── Sanity: distributions ──
    print(f"\n{'='*100}")
    print("v12 + v9 OOS overlap analysis")
    print(f"{'='*100}")
    print(f"  v9 P(short_win) stats:   mean={df['p_short_win'].mean():.3f}  "
          f"std={df['p_short_win'].std():.3f}  "
          f">=0.65: {(df['p_short_win']>=0.65).sum()} bars")
    print(f"  v12 P(no_trend) stats:    mean={df['p_no_trend'].mean():.3f}  "
          f"std={df['p_no_trend'].std():.3f}  "
          f"<0.4: {(df['p_no_trend']<0.4).sum()} bars")
    print(f"  v12 P(down_trend) stats:  mean={df['p_dn_trend'].mean():.3f}  "
          f"std={df['p_dn_trend'].std():.3f}  "
          f">=0.4: {(df['p_dn_trend']>=0.4).sum()} bars")

    # ── Baselines ──
    print(f"\n{'='*100}")
    print(f"Strategy backtest (SHORT only, TP=0.5%/SL=0.3%/H=8h, cost=5bp maker)")
    print(f"{'='*100}")
    print(f"  {'strategy':<48} {'n':>5} {'WR':>7} {'CI 95%':<18}  "
          f"{'gross':>8} {'net':>8} {'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 130)

    # Baseline 1: pure v9 P>=0.65
    pure_v9 = df[df["p_short_win"] >= 0.65]
    trades_v9 = ohlc_backtest_short(pure_v9, klines, TP_DIST, SL_DIST, 8, MAKER_COST)
    report(trades_v9, "Pure v9 P_short>=0.65 (baseline)")

    # Strategy 2: pure v12 (argmax == down AND P>=0.5)
    pure_v12 = df[(df["p_dn_trend"] >= df["p_up_trend"]) &
                   (df["p_dn_trend"] >= df["p_no_trend"]) &
                   (df["p_dn_trend"] >= 0.5)]
    trades_v12 = ohlc_backtest_short(pure_v12, klines, TP_DIST, SL_DIST, 8, MAKER_COST)
    report(trades_v12, "Pure v12 argmax=dn & P_dn>=0.5")

    # Strategy 3-7: hybrid v9 + v12 filter at various v12 thresholds
    print()
    for t_nt in [0.30, 0.35, 0.40, 0.45, 0.50]:
        hybrid = df[(df["p_short_win"] >= 0.65) & (df["p_no_trend"] < t_nt)]
        trades = ohlc_backtest_short(hybrid, klines, TP_DIST, SL_DIST, 8, MAKER_COST)
        report(trades, f"Hybrid v9 P>=0.65 & v12 P_no_trend<{t_nt}")

    print()
    for t_dn in [0.30, 0.35, 0.40, 0.45]:
        hybrid = df[(df["p_short_win"] >= 0.65) & (df["p_dn_trend"] >= t_dn)]
        trades = ohlc_backtest_short(hybrid, klines, TP_DIST, SL_DIST, 8, MAKER_COST)
        report(trades, f"Hybrid v9 P>=0.65 & v12 P_dn>={t_dn}")

    print()
    for t_short in [0.55, 0.60, 0.65, 0.70]:
        for t_nt in [0.35, 0.40, 0.45]:
            hybrid = df[(df["p_short_win"] >= t_short) & (df["p_no_trend"] < t_nt)]
            trades = ohlc_backtest_short(hybrid, klines, TP_DIST, SL_DIST, 8, MAKER_COST)
            report(trades, f"v9 P>={t_short} & v12 P_no_trend<{t_nt}")


if __name__ == "__main__":
    main()
