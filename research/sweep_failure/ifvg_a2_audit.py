"""F8 (a2) leakage audit — is the surviving cell real or intra-bar look-ahead?

The frozen (a2) filter reads the RETEST bar's total volume and taker share,
but those finalise at bar CLOSE while the limit fill happens intra-bar.
Conditioning a fill on information completed after the fill is the
shadow-exec class of artifact (mistake.md 2026-07-28).  For a SHORT at the
zone's upper edge, "taker share <= 0.45" selects bars that closed sell-heavy
— i.e. bars that touched our edge and then fell — which manufactures
favourable same-bar markout by construction.

Audit: identical IFVG set, identical filter, but the CLEAN variant fills at
the NEXT bar's open (the first price at which the filter's inputs are fully
known).  If (a2)'s edge survives the move, it was real; if it collapses,
the +0.05 was leakage.  This is an instrument audit, not a tuning pass —
the frozen definitions in ifvg_backtest.py are untouched.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ifvg_backtest as IB


def a2_trades_both(bars, atr, sweeps):
    """(frozen_edge_fill, clean_next_open_fill) for the same (a2) signals."""
    trades = IB.find_trades(bars, atr, sweeps)
    frozen, clean = [], []
    for t in trades:
        if not t.get("cohort_a2"):
            continue
        frozen.append(t)
        j = t["entry_i"] + 1                    # first bar after filter known
        if j + 1 >= len(bars) or atr[j] <= 0:
            continue
        sim = IB._simulate(bars, j, bars[j][IB.O], t["dir"], atr[j])
        if sim:
            clean.append(sim)
    return frozen, clean


def halves(trades):
    if len(trades) < 4:
        return 0.0, 0.0
    ts = sorted(t["ts"] for t in trades)
    cut = ts[len(ts) // 2]
    h1 = [t["net"] for t in trades if t["ts"] < cut]
    h2 = [t["net"] for t in trades if t["ts"] >= cut]
    return (sum(h1) / len(h1) if h1 else 0.0,
            sum(h2) / len(h2) if h2 else 0.0)


def run(tf: str, universe: list[str], label: str) -> None:
    print(f"\n════ (a2) audit — {tf} · {label} ════")
    pool_f, pool_c = [], []
    pos_f = pos_c = n_sym = 0
    for sym in universe:
        bars = IB.fetch_bars(sym, tf, IB.MONTHS)
        if len(bars) < 1000:
            print(f"  {sym}: only {len(bars)} bars — skipped")
            continue
        atr = IB.atr14(bars)
        sweeps = IB.sweep_events(bars, atr)
        f, c = a2_trades_both(bars, atr, sweeps)
        pool_f += f
        pool_c += c
        n_sym += 1
        mf = sum(t["net"] for t in f) / len(f) if f else 0.0
        mc = sum(t["net"] for t in c) / len(c) if c else 0.0
        pos_f += 1 if f and mf > 0 else 0
        pos_c += 1 if c and mc > 0 else 0
        print(f"  {sym:<5} n={len(f):>4}  frozen={mf:+.4f}  clean={mc:+.4f}")
    for name, pool, pos in (("frozen(edge fill)", pool_f, pos_f),
                            ("CLEAN(next open)", pool_c, pos_c)):
        if not pool:
            print(f"  {name:<18} n=0")
            continue
        mean, lo, hi = IB.clustered_ci(pool)
        g = sum(t["gross"] for t in pool) / len(pool)
        h1, h2 = halves(pool)
        print(f"  {name:<18} n={len(pool):>5}  grossR={g:+.4f}  netR={mean:+.4f}"
              f"  CI95[{lo:+.4f},{hi:+.4f}]  halves {h1:+.4f}/{h2:+.4f}"
              f"  pos {pos}/{n_sym}")


if __name__ == "__main__":
    run("15m", IB.CORE9, "core9 (the surviving cell)")
    run("5m", IB.CORE9, "core9 (P4-dead cell, for completeness)")
