"""Sweep-failure reversal — per-symbol backtest + split-half report.

Usage:
    python research/sweep_failure/run_backtest.py                 # uses .cache/
    python research/sweep_failure/run_backtest.py --data-dir PATH # existing CSVs

Env overrides: PIVOT / W / HOLD / DIS / SLIP (see sweep_core.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sweep_core as SC

SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / ".cache"))
    ap.add_argument("--risk-pct", type=float, default=1.0)
    args = ap.parse_args()
    data = Path(args.data_dir)

    print(f"sweep-failure backtest  PIVOT={SC.PIVOT} W={SC.W} HOLD={SC.HOLD} "
          f"DIS={SC.DIS} SLIP={SC.SLIP} risk={args.risk_pct}%")
    hdr = f"{'sym':<6}{'n':>5}{'net%':>8}{'PF':>6}{'WR%':>6}{'expR':>8}{'MDD%':>7}{'t':>7}  halves(exp)"
    print(hdr)
    print("-" * len(hdr))
    pool = []
    pos = 0
    for s in SYMS:
        p = data / f"{s}USDT_1h.csv"
        if not p.exists():
            print(f"{s:<6} missing {p}")
            continue
        bars = SC.load_csv(str(p))
        trs = SC.backtest_symbol(bars)
        rs = [t[2] for t in trs]
        pool += rs
        m = SC.metrics(rs, args.risk_pct)
        if not m:
            continue
        half = len(rs) // 2
        e1 = sum(rs[:half]) / max(half, 1)
        e2 = sum(rs[half:]) / max(len(rs) - half, 1)
        if m["net"] > 0:
            pos += 1
        print(f"{s:<6}{m['n']:>5}{m['net']:>+8.1f}{m['pf']:>6.2f}{m['wr']:>5.0f}%"
              f"{m['exp']:>+8.4f}{m['mdd']:>6.1f}%{m['t']:>+7.2f}  {e1:+.4f}/{e2:+.4f}")
    m = SC.metrics(pool, args.risk_pct)
    print("-" * len(hdr))
    print(f"{'pool':<6}{m['n']:>5}{'':>8}{m['pf']:>6.2f}{m['wr']:>5.0f}%"
          f"{m['exp']:>+8.4f}{'':>7}{m['t']:>+7.2f}   positive {pos}/9")


if __name__ == "__main__":
    main()
