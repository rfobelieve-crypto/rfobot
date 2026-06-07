"""Live signal-layer vs WF-OOS baseline diagnostic.

Answers the roadmap question: backtest exit logic is healthy (+90 bps in
exit_variants_backtest), but live shows negative P/L. Is the gap an EXIT
problem, an EXECUTION problem, or an ALPHA-DECAY problem?

This script isolates the SIGNAL layer (not the exit/execution layer):
for every filled tracked signal it has direction-correctness and the 4h
TWAP forward return (`actual_return_4h`) — the exact training target
y_path_ret_4h. No exit mechanics, no slippage, no leverage. Pure signal edge.

If live signal-layer WR / avg-return has decayed vs the WF-OOS baseline,
the live P/L problem is ALPHA DECAY, not exit mechanics — and changing
exit logic (oracle / ML exit) cannot fix it.

Splits:
  - by tier (Strong / Moderate)
  - by direction (UP / DOWN)
  - by month (decay trend)
  - LIVE cutover (>= 2026-04-17) vs pre-cutover

Usage:
    python -m research.live_vs_baseline_diag
    python -m research.live_vs_baseline_diag --live-since 2026-04-17
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from shared.db import get_db_conn

# WF-OOS baselines from the /perf dashboard (roadmap §訊號層數據)
BASELINE_STRONG_WR = 69.2  # %


def fetch():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT signal_time, direction, strength, confidence, "
        "       actual_return_4h, correct, regime "
        "FROM tracked_signals WHERE filled=1 AND actual_return_4h IS NOT NULL "
        "ORDER BY signal_time"
    )
    rows = cur.fetchall()  # DictCursor -> list[dict]
    cur.close()
    conn.close()
    return rows


def _stats(rows):
    """Return (n, wr%, avg_ret_bps, sum_ret_bps) for a row subset."""
    if not rows:
        return 0, float("nan"), float("nan"), float("nan")
    corr = np.array([int(r["correct"]) for r in rows], dtype=float)
    ret = np.array([float(r["actual_return_4h"]) for r in rows], dtype=float)
    return (len(rows),
            corr.mean() * 100.0,
            ret.mean() * 10000.0,
            ret.sum() * 10000.0)


def _line(label, rows):
    n, wr, avg, tot = _stats(rows)
    if n == 0:
        return f"  {label:<22} n={n:>4}   —"
    return (f"  {label:<22} n={n:>4}  WR={wr:>5.1f}%  "
            f"avg={avg:>+7.1f}bps  sum={tot:>+9.1f}bps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-since", default="2026-04-17")
    args = ap.parse_args()
    live_since = datetime.fromisoformat(args.live_since)

    rows = fetch()
    print(f"Total filled+scored tracked signals: {len(rows)}")
    if not rows:
        return
    print(f"Range: {rows[0]['signal_time']} -> {rows[-1]['signal_time']}")
    print(f"LIVE cutover: {live_since.date()}")
    print("=" * 70)

    def sub(rs, tier=None, direc=None):
        out = rs
        if tier:
            out = [r for r in out if r["strength"] == tier]
        if direc:
            out = [r for r in out if r["direction"] == direc]
        return out

    pre = [r for r in rows if r["signal_time"] < live_since]
    live = [r for r in rows if r["signal_time"] >= live_since]

    for label, rs in [("PRE-cutover", pre), ("LIVE (post-cutover)", live)]:
        print(f"\n### {label}")
        print(_line("ALL", rs))
        for tier in ("Strong", "Moderate"):
            print(_line(f"{tier}", sub(rs, tier)))
            for direc in ("UP", "DOWN"):
                print(_line(f"  {tier} {direc}", sub(rs, tier, direc)))

    print("\n" + "=" * 70)
    print(f"WF-OOS baseline Strong WR = {BASELINE_STRONG_WR}%")
    n, wr, avg, tot = _stats(sub(live, "Strong"))
    if n > 0:
        print(f"LIVE Strong WR          = {wr:.1f}%  "
              f"(Δ {wr - BASELINE_STRONG_WR:+.1f}pp vs baseline)")
        print(f"LIVE Strong avg 4h ret  = {avg:+.1f} bps  "
              f"({'NEGATIVE — alpha decay' if avg < 0 else 'positive'})")

    # ── Monthly decay trend ──
    print("\n### Monthly trend (ALL tiers, signal-layer)")
    months = sorted({str(r["signal_time"])[:7] for r in rows})
    for m in months:
        mr = [r for r in rows if str(r["signal_time"])[:7] == m]
        print(_line(m, mr))

    # ── Verdict ──
    print("\n" + "=" * 70)
    _, live_wr, live_avg, _ = _stats(live)
    _, _, pre_avg, _ = _stats(pre)
    print("VERDICT:")
    if live_avg < 0 <= pre_avg:
        print("  Signal-layer avg return flipped NEGATIVE post-cutover while")
        print("  pre-cutover was positive. The live P/L problem is ALPHA DECAY")
        print("  at the SIGNAL layer, NOT exit mechanics. Exit ML/oracle work")
        print("  would target the wrong layer — prioritise retrain / regime gate.")
    elif live_avg >= 0:
        print("  Signal-layer avg return is still POSITIVE post-cutover.")
        print("  The live negative P/L must come from EXIT or EXECUTION layer")
        print("  (slippage, trail give-back, fees) — exit analysis IS warranted.")
    else:
        print("  Both pre and post are negative — signal edge was never strong")
        print("  in this window; re-examine the tier thresholds / target.")


if __name__ == "__main__":
    main()
