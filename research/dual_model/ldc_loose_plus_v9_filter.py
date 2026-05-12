"""
Loosen LDC entry conditions + use v9 order flow as filter (user 2026-05-12).

Production LDC swing (current) uses jdehorty's chained filters:
    start_long_trade  = is_new_buy  AND is_bullish_rate AND ema_uptrend
    start_short_trade = is_new_sell AND is_bearish_rate AND ema_downtrend
Where:
    is_new_buy  = (signal == 1)  AND (signal != prev_signal) AND ema_uptrend
    is_new_sell = (signal == -1) AND (signal != prev_signal) AND ema_downtrend

This gives only 37 SHORT + 30 LONG = 67 entries over 5.5 months.

User idea: peel off filters one at a time to get more entries, then use
v9 P(side_win) as confirmation filter.  Goal — find a config where more
entries × v9 filter = better cum than pure LDC.

Test grid:
    Loose levels (entry candidates):
      L0: signal flip ONLY (no filters)               — most entries
      L1: signal flip + ema_filter
      L2: signal flip + ema + kernel_rate (jdehorty's start_*_trade)  - baseline production

    v9 filter modes (applied to each loose level):
      F0: no v9 filter (pure LDC at that loose level)
      F1: v9 P(side_win) >= 0.55
      F2: v9 P(side_win) >= 0.60
      F3: v9 P(side_win) >= 0.65

Exit: dynamic_cross + min hold 4h (production setting).
Cost: maker 5 bps.
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

from research.dual_model.ldc_swing_backtest import simulate_swing, load_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_loose_entries(ldc: pd.DataFrame, level: str) -> pd.DataFrame:
    """Build a frame with custom start_long_trade / start_short_trade columns
    according to the chosen loose level.

    Returns a COPY of ldc with overridden start_*_trade columns.
    """
    out = ldc.copy()
    signal = out["signal"].astype(int)
    prev_signal = signal.shift(1).fillna(0).astype(int)
    flip = signal != prev_signal

    if level == "L0":
        # Pure signal flip (any direction change), no other filters
        out["start_long_trade"] = (signal == 1) & flip
        out["start_short_trade"] = (signal == -1) & flip
    elif level == "L1":
        # signal flip + EMA filter only
        out["start_long_trade"] = (signal == 1) & flip & out["ema_uptrend"].astype(bool)
        out["start_short_trade"] = (signal == -1) & flip & out["ema_downtrend"].astype(bool)
    elif level == "L2":
        # signal flip + EMA + kernel rate (jdehorty's production filter)
        out["start_long_trade"] = (
            (signal == 1) & flip
            & out["ema_uptrend"].astype(bool)
            & out["is_bullish_kernel"].astype(bool)
        )
        out["start_short_trade"] = (
            (signal == -1) & flip
            & out["ema_downtrend"].astype(bool)
            & out["is_bearish_kernel"].astype(bool)
        )
    else:
        raise ValueError(f"unknown level {level}")
    return out


def main():
    ldc, klines, v9 = load_data()

    # Align v9 P columns to ldc index
    ldc["p_long_win"] = v9["p_long_win"].reindex(ldc.index)
    ldc["p_short_win"] = v9["p_short_win"].reindex(ldc.index)

    print(f"\n{'='*135}")
    print(f"LDC LOOSE-ENTRY + v9 FILTER backtest")
    print(f"Exit: dynamic_cross + min hold 4h | Cost: 5 bps maker")
    print(f"{'='*135}")

    def report_row(label, t):
        if t.empty:
            print(f"  {label:<60} n=0")
            return None
        n = len(t)
        wr = t["win"].mean()
        net = t["net_pct"].values
        cum = np.cumsum(net)
        rmax = np.maximum.accumulate(cum)
        mdd = (cum - rmax).min() * 100
        wins = net[net > 0]
        losses = net[net < 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf
        avg_bp = net.mean() * 10000
        cum_3x = cum[-1] * 100 * 3
        mdd_3x = mdd * 3
        wins_int = int(t["win"].sum())
        ci = binomtest(wins_int, n).proportion_ci(0.95) if n >= 5 else None
        ci_str = f"[{ci.low*100:.0f},{ci.high*100:.0f}]" if ci else ""
        print(f"  {label:<60} n={n:>4} WR={wr*100:>5.1f}% {ci_str:<11}  "
              f"net={avg_bp:>+6.1f}bp cum={cum[-1]*100:>+5.1f}%(1x)/{cum_3x:>+5.1f}%(3x) "
              f"MDD={mdd:>+5.1f}%/{mdd_3x:>+6.1f}%(3x) PF={pf:>4.2f}")
        return n, wr, avg_bp, cum[-1] * 100

    for level in ["L0", "L1", "L2"]:
        loose_ldc = build_loose_entries(ldc, level)
        # how many candidates each level offers?
        n_long_cand = int(loose_ldc["start_long_trade"].sum())
        n_short_cand = int(loose_ldc["start_short_trade"].sum())
        print()
        print(f">> LEVEL {level} — {n_long_cand} LONG candidates, {n_short_cand} SHORT candidates")
        print()

        # Filter modes
        for f_label, long_thr, short_thr in [
            ("F0 (no v9 filter)", None, None),
            ("F1 (v9 P >= 0.55)", 0.55, 0.55),
            ("F2 (v9 P >= 0.60)", 0.60, 0.60),
            ("F3 (v9 P >= 0.65)", 0.65, 0.65),
        ]:
            if long_thr is not None:
                def filt(ts, row, d, long_thr=long_thr, short_thr=short_thr):
                    if d == "LONG":
                        pl = row.get("p_long_win", np.nan)
                        if pd.isna(pl) or pl < long_thr:
                            return False
                    else:
                        ps = row.get("p_short_win", np.nan)
                        if pd.isna(ps) or ps < short_thr:
                            return False
                    return True
            else:
                filt = None
            t = simulate_swing(
                loose_ldc, klines,
                entry_filter=filt,
                exit_mode="dynamic_cross",
                min_hold_bars=4,
                cost=0.0005,
            )
            report_row(f"  {f_label}", t)
            # SHORT-only subline (more focused since SHORT is real edge)
            if not t.empty:
                t_short = t[t["direction"] == "SHORT"]
                report_row(f"    └ SHORT only", t_short)
                t_long = t[t["direction"] == "LONG"]
                report_row(f"    └ LONG  only", t_long)


if __name__ == "__main__":
    main()
