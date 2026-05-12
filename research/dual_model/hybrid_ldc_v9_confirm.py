"""
Hybrid LDC + v9 with "MUST AGREE" rule (user, 2026-05-12).

Critical distinction from previous attempts:
    Before:  v9_signal AND (LDC_not_opposite) → "veto" layer (marginal)
    NOW   :  v9_signal AND LDC_same_direction → "must agree" filter (stricter)

Models stay independent — LDC stays as standalone Python port (no retrain
with v9 features), v9 stays as binary win classifier.  Combination is
purely inference-time: at each bar, check both, trade if both agree.

Test grid:
    v9 P(short_win) thresholds: 0.45, 0.50, 0.55, 0.60, 0.65
    LDC alignment: signal carry == -1 (short) at same bar
    direction: SHORT only (LONG has no v9 alpha)

Compare to:
    Pure v9 P>=0.65 baseline   (53 trades, WR 56.6%)
    Pure LDC short entries     (44 trades, WR ~55%)

OHLC backtest with TP=0.5% / SL=0.3% / H=8h / maker cost 5bp.
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

V9_OOS = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_v9_winrate_H8_oos.parquet"
LDC_FULL = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"

TP_DIST = 0.005
SL_DIST = 0.003
MAKER_COST = 0.0005   # 5 bps
HORIZON_HOURS = 8


def load_aligned():
    """Load v9 OOS + LDC signal carry, joined on timestamp."""
    v9 = pd.read_parquet(V9_OOS)
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v9 = v9.set_index("ts").sort_index()

    ldc = pd.read_parquet(LDC_FULL)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    ldc = ldc.sort_index()

    # Join: take LDC signal at same ts (LDC is signal carry, always defined)
    v9["ldc_signal"] = ldc["signal"].reindex(v9.index).fillna(0).astype(int)
    v9["ldc_kernel_bear"] = ldc["is_bearish_kernel"].reindex(v9.index).fillna(False)
    v9["ldc_ema_down"] = ldc["ema_downtrend"].reindex(v9.index).fillna(False)
    logger.info("Aligned: n=%d (v9=%d, LDC join=%d non-null)",
                 len(v9), len(v9), int((v9["ldc_signal"] != 0).sum()))
    return v9


def load_klines():
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    return k


def ohlc_backtest_short(ts_list, klines: pd.DataFrame, cost: float) -> pd.DataFrame:
    rows = []
    klines_idx = klines.index
    for ts in ts_list:
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        entry = float(klines.loc[ts, "close"])
        future_idx = klines_idx[klines_idx > ts]
        if len(future_idx) < 1:
            continue
        future = klines.loc[future_idx]
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry, direction="DOWN",
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=future, timeout_bars=HORIZON_HOURS,
        )
        gross = -(exit_price / entry - 1.0)
        net = gross - cost
        rows.append({
            "ts": ts, "exit_price": exit_price,
            "bars_held": bars_held, "reason": reason,
            "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str):
    n = len(df)
    if n == 0:
        print(f"  {label:<55} n=0")
        return None
    wr = df["win"].mean()
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    avg_net = net.mean() * 10000
    avg_gross = df["gross_ret"].mean() * 10000
    wins_int = int(df["win"].sum())
    ci = binomtest(wins_int, n).proportion_ci(0.95) if n >= 5 else None
    ci_str = f"[{ci.low*100:.1f}, {ci.high*100:.1f}]" if ci else ""
    print(f"  {label:<55} n={n:>3} WR={wr*100:>5.1f}% {ci_str:<16}  "
          f"gross={avg_gross:>+6.1f}bp net={avg_net:>+6.1f}bp  "
          f"cum={cum[-1]*100:>+5.1f}% MDD={mdd:>+5.1f}% PF={pf:>4.2f}")
    return {"n": n, "wr": wr, "net_bps": avg_net, "cum": cum[-1] * 100, "mdd": mdd}


def main():
    df = load_aligned()
    klines = load_klines()

    print(f"\n{'='*125}")
    print(f"LDC + v9 'MUST AGREE' hybrid backtest (SHORT only)")
    print(f"TP={TP_DIST*100:.1f}% / SL={SL_DIST*100:.1f}% / H={HORIZON_HOURS}h / maker cost={MAKER_COST*10000:.0f}bp")
    print(f"{'='*125}")
    print(f"  {'strategy':<55} {'n':>5} {'WR':>7} {'CI 95%':<18}  "
          f"{'gross':>8} {'net':>8} {'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 130)

    # ── Baselines ──
    # Pure v9 P>=0.65
    pure_v9 = df[df["p_short_win"] >= 0.65].index
    report(ohlc_backtest_short(pure_v9, klines, MAKER_COST),
           "BASELINE: Pure v9 P_short>=0.65")

    # Pure LDC short (carry == -1)
    pure_ldc = df[df["ldc_signal"] == -1].index
    report(ohlc_backtest_short(pure_ldc, klines, MAKER_COST),
           "BASELINE: Pure LDC short carry (-1)")

    print()
    print(">> HYBRID: v9 + LDC must agree on SHORT direction:")
    print()
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
        hybrid = df[(df["p_short_win"] >= thr) & (df["ldc_signal"] == -1)].index
        report(ohlc_backtest_short(hybrid, klines, MAKER_COST),
               f"v9 P>={thr:.2f} AND LDC carry==short")

    print()
    print(">> Reference: v9 + LDC NOT in conflict (LDC != opposite):")
    print()
    for thr in [0.55, 0.60, 0.65]:
        nonconflict = df[(df["p_short_win"] >= thr) & (df["ldc_signal"] != +1)].index
        report(ohlc_backtest_short(nonconflict, klines, MAKER_COST),
               f"v9 P>={thr:.2f} AND LDC != long (allow short or neutral)")

    print()
    print(">> Stricter: v9 + LDC short + EMA200 down + Kernel bearish (LDC '4 stars'):")
    print()
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
        stricter = df[
            (df["p_short_win"] >= thr)
            & (df["ldc_signal"] == -1)
            & df["ldc_ema_down"]
            & df["ldc_kernel_bear"]
        ].index
        report(ohlc_backtest_short(stricter, klines, MAKER_COST),
               f"v9 P>={thr:.2f} + LDC SHORT + EMA_dn + Kernel_bear")


if __name__ == "__main__":
    main()
