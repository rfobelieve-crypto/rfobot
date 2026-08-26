# -*- coding: utf-8 -*-
"""Are DOWN signals starved in an uptrend? — TODO §0.63.

Operator's question: "上升段的做空訊號是不是更嚴格，這樣多單不就更難出場"
— i.e. if Strong DOWN fires less often while the tape rises, then a LONG
held through an uptrend loses its `opp_signal` exit exactly when it might
need one.

THE CONCERN HAS PRECEDENT IN THIS SYSTEM, which is why it gets measured
rather than reasoned about. mistake.md 2026-08-08: the rolling-percentile
decode compares today's prediction against the trailing window of its own
predictions. That guarantees ~2.5% per tail ONLY IF the prediction
distribution is stationary. Under sustained drift the trailing window is
stale, today's value sits near one edge of it, and the far tail starves —
July fired 14 UP : 1 DOWN, August 20 : 1. So "one side can starve" is not
hypothetical here; it happened for three months and nobody noticed,
because rank metrics are blind to a level shift.

The 2026-08-11 fix (rebuild the buffer from live DB predictions, window
500 -> 200) shortens the memory so it adapts faster, but a drift that
persists WITHIN 200 bars would still skew it. So the question is open.

What this measures, per frozen ADX x direction cell:
  * bars in the cell
  * Strong UP / Strong DOWN counts
  * FIRING RATE PER BAR for each side — the count alone is confounded by
    how many bars each cell has, and the cells are very unequal
  * the DOWN share, which is the number the operator's concern is about

Pre-committed reading:
  * DOWN rate in TREND_UP materially below its rate elsewhere
        -> starvation is real, opp_signal exit degrades for LONGs in
           uptrends, and that is an exit-side problem needing its own fix
  * DOWN rate in TREND_UP at or above the others
        -> the decode is adapting as designed; the concern does not
           reproduce and the LONG exit story must be elsewhere
        (trail_stop / conviction_decay carrying the load)

Signal layer only. Read-only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from research.crowd_battery2 import adx_state              # noqa: E402
from shared.db import get_db_conn                          # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "v7_down_rate_by_regime.json"
CELLS = ("RANGING", "TREND_UP", "TREND_DOWN", "NEUTRAL")
LB = 24


def main() -> int:
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    c = [b[SC.C] for b in bars]
    adx = adx_state(bars)
    cell_of_h = {}
    for i in range(LB, len(bars)):
        ts = bars[i][0]
        lab = adx.get(ts // 3600 * 3600)
        if lab is None:
            continue
        if lab == "RANGING":
            cell_of_h[ts] = "RANGING"
        elif lab != "TRENDING":
            cell_of_h[ts] = "NEUTRAL"
        else:
            cell_of_h[ts] = ("TREND_UP" if c[i] / c[i - LB] - 1 > 0
                             else "TREND_DOWN")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction FROM tracked_signals "
                "WHERE strength='Strong' ORDER BY signal_time")
            rows = cur.fetchall()
    finally:
        conn.close()

    lo = min(cell_of_h)
    hi = max(cell_of_h)
    sig = defaultdict(lambda: {"UP": 0, "DOWN": 0})
    used = 0
    for r in rows:
        ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        h = ts // 3600 * 3600
        cell = cell_of_h.get(h)
        if cell is None or r["direction"] not in ("UP", "DOWN"):
            continue
        sig[cell][r["direction"]] += 1
        used += 1

    # bar counts must be restricted to the SAME span the signals cover,
    # otherwise a cell that is common in a period with no signal history
    # gets a fake-low rate.
    sig_hours = [int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
                 // 3600 * 3600 for r in rows]
    s_lo, s_hi = min(sig_hours), max(sig_hours)
    bar_n = defaultdict(int)
    for h, cell in cell_of_h.items():
        if s_lo <= h <= s_hi:
            bar_n[cell] += 1
    tot_bars = sum(bar_n.values())

    print("§0.63 上升段的 DOWN 訊號有沒有被餓死")
    print(f"  訊號樣本 n={used}（Strong，含未結算）｜"
          f"對齊的 bar 數 {tot_bars}\n")
    print(f"{'cell':12} {'bar 數':>7} {'UP':>5} {'DOWN':>6} "
          f"{'UP/bar':>8} {'DOWN/bar':>9} {'DOWN 佔比':>10}")
    res = {}
    for cell in CELLS:
        nb = bar_n.get(cell, 0)
        u, d = sig[cell]["UP"], sig[cell]["DOWN"]
        if not nb:
            continue
        ur, dr = 100 * u / nb, 100 * d / nb
        share = 100 * d / (u + d) if (u + d) else 0
        mark = "  ← 問題所在的格" if cell == "TREND_UP" else ""
        print(f"{cell:12} {nb:7d} {u:5d} {d:6d} {ur:7.2f}% {dr:8.2f}% "
              f"{share:9.1f}%{mark}")
        res[cell] = {"bars": nb, "up": u, "down": d,
                     "up_rate_pct": round(ur, 3),
                     "down_rate_pct": round(dr, 3),
                     "down_share_pct": round(share, 1)}

    tu = res.get("TREND_UP")
    others = [res[k]["down_rate_pct"] for k in res if k != "TREND_UP"]
    if tu and others:
        med = sorted(others)[len(others) // 2]
        rel = tu["down_rate_pct"] / med if med else float("nan")
        print(f"\n  TREND_UP 的 DOWN 開火率 {tu['down_rate_pct']:.2f}%/bar "
              f"vs 其他格中位 {med:.2f}%/bar  →  {rel:.2f}×")
        starved = rel < 0.7
        verdict = ("餓死成立：上升段的 DOWN 開火率明顯偏低，多單的 "
                   "opp_signal 出場在最需要的時候變稀薄"
                   if starved else
                   "餓死**不成立**：上升段的 DOWN 開火率沒有偏低，"
                   "解碼如設計般在適應。多單的出場問題若存在，成因在別處")
        print(f"\n判讀：{verdict}")
        res["verdict"] = verdict
        res["trend_up_down_rate_vs_median"] = round(rel, 3)

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
