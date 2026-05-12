"""
Hybrid backtest with LDC PRIMARY + v9 VETO (Option A, user 2026-05-12).

New logic (vs must-agree):
    Must-agree (current):  v9_P(side) >= 0.65 AND LDC == side
    Option A (this script): LDC == side AND v9_P(opposite) < veto_threshold

LDC fires the entry; v9 only blocks if it STRONGLY disagrees with LDC.
This is more permissive — keeps weaker but valid LDC signals.

Test grid:
    SHORT: LDC signal == -1 AND v9 P(long_win) < threshold
    LONG : LDC signal == +1 AND v9 P(short_win) < threshold

    thresholds: 0.50 (basically no veto), 0.55, 0.60, 0.65 (mild veto), 0.70

Compare to baselines:
    Pure LDC (no v9 input)
    Must-agree (current production)

OHLC backtest, TP=0.5% / SL=0.3% / H=8h / maker cost 5bps.
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
MAKER_COST = 0.0005
HORIZON_HOURS = 8


def load_aligned():
    v9 = pd.read_parquet(V9_OOS)
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v9 = v9.set_index("ts").sort_index()
    ldc = pd.read_parquet(LDC_FULL)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    # ldc_signal: persistent carry signal (-1/0/+1 each bar)
    # start_*_trade: True only at the bar where signal flips into that side
    v9["ldc_signal"] = ldc["signal"].reindex(v9.index).fillna(0).astype(int)
    v9["ldc_start_short"] = ldc["start_short_trade"].reindex(v9.index).fillna(False).astype(bool)
    v9["ldc_start_long"] = ldc["start_long_trade"].reindex(v9.index).fillna(False).astype(bool)
    return v9


def load_klines():
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    return k


def backtest(ts_list, klines, direction: str, cost: float) -> pd.DataFrame:
    """direction: 'LONG' or 'SHORT'"""
    rows = []
    klines_idx = klines.index
    dir_for_exit = "UP" if direction == "LONG" else "DOWN"
    for ts in ts_list:
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in klines_idx:
            continue
        entry = float(klines.loc[ts, "close"])
        future = klines.loc[klines_idx > ts]
        if len(future) < 1:
            continue
        exit_price, bars_held, reason = _find_exit_for_signal(
            entry_price=entry, direction=dir_for_exit,
            sl_dist=SL_DIST, rr=TP_DIST / SL_DIST,
            bars=future, timeout_bars=HORIZON_HOURS,
        )
        if direction == "LONG":
            gross = exit_price / entry - 1.0
        else:
            gross = -(exit_price / entry - 1.0)
        net = gross - cost
        rows.append({
            "ts": ts, "direction": direction,
            "gross_ret": gross, "net_ret": net,
            "win": int(gross > 0),
        })
    return pd.DataFrame(rows)


def report(df, label):
    n = len(df)
    if n == 0:
        print(f"  {label:<60} n=0")
        return None
    wr = df["win"].mean()
    net = df["net_ret"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    mdd = (cum - rmax).min() * 100
    wins = net[net > 0]
    losses = net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
    avg_net_bps = net.mean() * 10000
    avg_gross_bps = df["gross_ret"].mean() * 10000
    wins_int = int(df["win"].sum())
    ci = binomtest(wins_int, n).proportion_ci(0.95) if n >= 5 else None
    ci_str = f"[{ci.low*100:.1f},{ci.high*100:.1f}]" if ci else ""
    print(f"  {label:<60} n={n:>3} WR={wr*100:>5.1f}% {ci_str:<14}  "
          f"net={avg_net_bps:>+6.1f}bp gross={avg_gross_bps:>+6.1f}bp  "
          f"cum={cum[-1]*100:>+5.1f}% MDD={mdd:>+5.1f}% PF={pf:>4.2f}")
    return {"n": n, "wr": wr, "net_bps": avg_net_bps,
            "cum_pct": cum[-1] * 100, "mdd_pct": mdd, "pf": pf}


def main():
    df = load_aligned()
    klines = load_klines()

    print(f"\n{'='*135}")
    print(f"Hybrid: LDC PRIMARY + v9 VETO layer (Option A)")
    print(f"TP={TP_DIST*100:.1f}% / SL={SL_DIST*100:.1f}% / H={HORIZON_HOURS}h / maker cost={MAKER_COST*10000:.0f}bp")
    print(f"{'='*135}")
    print(f"  {'strategy':<60} {'n':>5} {'WR':>8} {'CI 95%':<14}  "
          f"{'net':>9} {'gross':>9} {'cum%':>7} {'MDD%':>7} {'PF':>5}")
    print("-" * 135)

    # ── BASELINES (using ENTRY-TRIGGER bar, not carry filter) ──
    # LDC carry filter = many overlapping false entries (3696 in 5.5mo, all lose).
    # Real LDC entries fire only on signal transitions (start_short_trade /
    # start_long_trade), which matches what TradingView arrows show.
    print(">> Baselines (LDC entry-trigger only):")
    pure_ldc_short = df[df["ldc_start_short"]].index
    report(backtest(pure_ldc_short, klines, "SHORT", MAKER_COST),
           "Pure LDC SHORT (start_short_trade)")

    pure_ldc_long = df[df["ldc_start_long"]].index
    report(backtest(pure_ldc_long, klines, "LONG", MAKER_COST),
           "Pure LDC LONG  (start_long_trade)")

    short_df_base = backtest(pure_ldc_short, klines, "SHORT", MAKER_COST)
    long_df_base = backtest(pure_ldc_long, klines, "LONG", MAKER_COST)
    pure_ldc_all = pd.concat([short_df_base, long_df_base], ignore_index=True).sort_values("ts")
    report(pure_ldc_all, "Pure LDC ALL (long+short, entry-trigger)")

    # Must-agree (current production — uses CARRY signal because v9 P>=0.65
    # is rare, want to know "during LDC's bias period, did v9 also agree")
    print()
    print(">> Current production (must-agree, uses carry signal):")
    mustagree_short = df[(df["p_short_win"] >= 0.65) & (df["ldc_signal"] == -1)].index
    report(backtest(mustagree_short, klines, "SHORT", MAKER_COST),
           "Must-agree SHORT: v9 P_short>=0.65 AND ldc_signal==-1")
    mustagree_long = df[(df["p_long_win"] >= 0.65) & (df["ldc_signal"] == +1)].index
    report(backtest(mustagree_long, klines, "LONG", MAKER_COST),
           "Must-agree LONG : v9 P_long>=0.65 AND ldc_signal==+1")

    # ── OPTION A: LDC PRIMARY (entry-trigger only) + v9 VETO ──
    print()
    print(">> Option A: LDC PRIMARY (start_*_trade) + v9 VETO if strongly disagrees:")
    print()
    for veto_thr in [0.50, 0.55, 0.60, 0.65, 0.70, 1.0]:
        # veto_thr=1.0 means no veto at all (pure LDC entry)
        a_short_idx = df[df["ldc_start_short"] & (df["p_long_win"] < veto_thr)].index
        a_long_idx = df[df["ldc_start_long"] & (df["p_short_win"] < veto_thr)].index
        s = backtest(a_short_idx, klines, "SHORT", MAKER_COST)
        l = backtest(a_long_idx, klines, "LONG", MAKER_COST)

        if veto_thr == 1.0:
            print(f"  -- veto_threshold = 1.00 (no veto, sanity check vs pure LDC) --")
        else:
            print(f"  -- veto_threshold = {veto_thr:.2f} (block if v9 P(opp) >= {veto_thr:.2f}) --")
        report(s, f"  SHORT  (LDC start_short, v9 P_long < {veto_thr:.2f})")
        report(l, f"  LONG   (LDC start_long,  v9 P_short < {veto_thr:.2f})")
        combined = pd.concat([s, l], ignore_index=True).sort_values("ts")
        report(combined, f"  COMBINED")
        print()


if __name__ == "__main__":
    main()
