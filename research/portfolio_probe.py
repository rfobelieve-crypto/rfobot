# -*- coding: utf-8 -*-
"""Portfolio probe — empirically test the K=2 portfolio claims (axes 1-3)
on the overlap window of the two trade streams we actually have.

Streams:
  A = V7 baseline backtest (3xATR trail / opp_signal / 72h cap) on the
      WF-OOS window — regenerated via exit_variants_backtest's simulator.
      CAVEAT: level is optimistic (OOS preds carry the production
      early-stop leak; mistake.md 2026-06-19 family). Correlation and
      utilization structure are far less level-sensitive than the mean —
      we read rho/overlap here, NOT absolute Sharpe.
  B = strategy #3 sweep-failure, corrected scenario-A costs (frozen rules).

Measured (2026-07-29, before any portfolio engineering exists):
  1. axis-1: daily-PnL correlation rho(A,B); each stream standardized to
     unit daily vol; combined 50/50 Sharpe vs the sqrt(2/(1+rho))
     prediction from the same individual Sharpes.
  2. axis-2: capital utilization — fraction of days each stream holds a
     position; joint-idle fraction; pooled trades/month.
  3. shared-beta check: the 5 worst BTC days in the window — do A and B
     lose together on tail days?

This is a measurement, not a build. No production surface touched.
Run: python research/portfolio_probe.py
Out: research/results/portfolio_probe.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research" / "results" / "portfolio_probe.json"


def v7_daily() -> tuple[pd.Series, pd.Series, int]:
    """V7 baseline backtest trades -> (daily PnL %, daily in-position flag, n)."""
    from research.exit_variants_backtest import (
        variants_catalog, simulate_with_policy, _atr_wilder, ATR_PERIOD)
    from research.v71_v7_sizing_1x import load_data, decode_signals
    df = load_data()
    df["atr"] = _atr_wilder(df, ATR_PERIOD)
    direction, tier, warmup_n = decode_signals(df)
    df = df.iloc[warmup_n:].copy()
    trades = simulate_with_policy(df, direction[warmup_n:], tier[warmup_n:],
                                  variants_catalog()["baseline"])
    pnl = defaultdict(float)
    inpos = set()
    for _, t in trades.iterrows():
        d = pd.Timestamp(t["exit_ts"]).normalize()
        pnl[d] += float(t["net_pct"]) * 100.0          # % per trade, exit-booked
        span = pd.date_range(pd.Timestamp(t["entry_ts"]).normalize(),
                             pd.Timestamp(t["exit_ts"]).normalize(), freq="D")
        inpos.update(span)
    days = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(), freq="D")
    ser = pd.Series({d: pnl.get(d, 0.0) for d in days}).sort_index()
    pos = pd.Series({d: (1.0 if d in inpos else 0.0) for d in days}).sort_index()
    return ser, pos, len(trades)


def s3_daily() -> tuple[pd.Series, pd.Series, int]:
    """#3 scenario-A trades -> (daily PnL in R*risk %, in-position flag, n)."""
    os.environ["SLIP"] = "0"
    import sweep_core as SC
    from sweep_forward import SYMS, CACHE, SCEN
    RISK = 0.5   # % per trade — scale is irrelevant for rho; kept for readability
    pnl = defaultdict(float)
    inpos = set()
    n = 0
    lo = hi = None
    for s in SYMS:
        for fill_ts, exit_ts, r, lvl, atr, stopped in SC.backtest_symbol(
                SC.load_csv(str(CACHE / f"{s}USDT_1h.csv"))):
            legs = SCEN["A"]["entry"] + (SCEN["A"]["sexit"] if stopped
                                         else SCEN["A"]["texit"])
            r_net = r - legs / 1e4 * lvl / (SC.DIS * atr)
            d0 = pd.Timestamp(fill_ts, unit="s").normalize()
            d1 = pd.Timestamp(exit_ts, unit="s").normalize()
            pnl[d1] += RISK * r_net
            inpos.update(pd.date_range(d0, d1, freq="D"))
            n += 1
            lo = d0 if lo is None or d0 < lo else lo
            hi = d1 if hi is None or d1 > hi else hi
    days = pd.date_range(lo, hi, freq="D")
    ser = pd.Series({d: pnl.get(d, 0.0) for d in days}).sort_index()
    pos = pd.Series({d: (1.0 if d in inpos else 0.0) for d in days}).sort_index()
    return ser, pos, n


def sharpe(x: pd.Series) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(365.0)) if s > 0 else float("nan")


def main() -> int:
    a, a_pos, n_a = v7_daily()
    b, b_pos, n_b = s3_daily()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    a_pos, b_pos = a_pos.loc[idx], b_pos.loc[idx]
    days = len(idx)
    months = days / 30.44
    print(f"overlap: {idx[0].date()} -> {idx[-1].date()}  ({days} days)")
    print(f"trades in window: V7={int((a != 0).sum())} bookings "
          f"(total {n_a} in its run)  #3={int((b != 0).sum())} booking-days")

    # axis 1 — correlation & combined Sharpe
    rho_p = float(a.corr(b))
    rho_s = float(a.corr(b, method="spearman"))
    az = a / a.std(ddof=1)
    bz = b / b.std(ddof=1)
    combo = (az + bz) / 2.0
    sh_a, sh_b, sh_c = sharpe(a), sharpe(b), sharpe(combo)
    pred = math.sqrt(2.0 / (1.0 + rho_p)) * (sh_a + sh_b) / 2.0
    print(f"\n[axis1] daily rho: pearson {rho_p:+.3f}  spearman {rho_s:+.3f}")
    print(f"        Sharpe(ann): V7 {sh_a:+.2f}  #3 {sh_b:+.2f}  "
          f"combined50/50 {sh_c:+.2f}  (formula predicts {pred:+.2f})")

    # axis 2 — utilization / n-stacking
    u_a = float(a_pos.mean())
    u_b = float(b_pos.mean())
    idle_joint = float(((a_pos == 0) & (b_pos == 0)).mean())
    print(f"\n[axis2] days-in-position: V7 {u_a:.0%}  #3 {u_b:.0%}  "
          f"both idle {idle_joint:.0%}  (single-strategy idle was "
          f"{1-u_a:.0%} / {1-u_b:.0%})")
    print(f"        pooled bookings/month: {((a != 0).sum() + (b != 0).sum()) / months:.0f}")

    # axis 3 proxy — shared beta on tail days
    btc = pd.read_csv(ROOT / "research/sweep_failure/.cache/BTCUSDT_1h.csv",
                      encoding="utf-8-sig")
    btc["d"] = pd.to_datetime(btc.iloc[:, 0], unit="s").dt.normalize()
    dret = btc.groupby("d")["close"].last().pct_change().dropna()
    dret = dret.loc[dret.index.intersection(idx)]
    worst = dret.nsmallest(5)
    print("\n[beta ] 5 worst BTC days in window — same-day PnL:")
    both_neg = 0
    for d, r in worst.items():
        pa, pb = float(a.get(d, 0.0)), float(b.get(d, 0.0))
        both_neg += int(pa < 0 and pb < 0)
        print(f"        {d.date()}  BTC {r:+.2%}   V7 {pa:+.3f}%   #3 {pb:+.3f}%")
    print(f"        both negative on {both_neg}/5 tail days")

    OUT.write_text(json.dumps({
        "overlap_days": days, "rho_pearson": rho_p, "rho_spearman": rho_s,
        "sharpe_v7": sh_a, "sharpe_s3": sh_b, "sharpe_combined": sh_c,
        "sharpe_predicted": pred, "util_v7": u_a, "util_s3": u_b,
        "joint_idle": idle_joint, "tail_both_neg": both_neg,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
