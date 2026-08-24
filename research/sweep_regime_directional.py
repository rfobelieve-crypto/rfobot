# -*- coding: utf-8 -*-
"""Directional regime split for the Gate F attachment — TODO §0.54b.

Product side reported (jarvis CLAUDE.md 2026-08-23), on variant C over the
shadow log: RANGING +0.083R (n=218), DOWN-trend -0.118R (n=10), UP-trend
-0.330R (n=68) — "one-way markets do maul it, and UP hurts most (a
mean-reversion book run over by a trend)". Their ask: Gate F must be judged
per regime, or the sample window's regime mix decides the verdict.

§0.54 already registers a regime attachment, but it splits on ADX only —
TRENDING vs RANGING, no direction. If UP and DOWN trends really differ by
2.8x, an undirected split averages two different populations and hides it.

This tests the directional claim where it matters: variant B (the Gate F
track, not C), full history (not the 4-week shadow window), core9. Regime
is the ADX label already frozen in §0.49d, subdivided by the sign of the
concurrent 24h return — sign of realised move, no new threshold invented.

Pre-committed reading (written before running):
  - if UP-trend is materially worse than DOWN-trend on variant B too, the
    §0.54 attachment gains a directional axis and Gate F's verdict must
    report all three cells
  - if the gap does not reproduce, the product-side finding is variant-C
    and/or short-window specific, and the undirected split stays

Read-only. Nothing here can pass or fail Gate F; it decides how the
verdict is READ.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                   # noqa: E402
from research.crowd_battery2 import adx_state             # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
OUT = ROOT / "research" / "results" / "sweep_regime_directional.json"
random.seed(7)


def cell_of(adx_lab: str, ret24: float) -> str:
    """ADX label (frozen §0.49d) x sign of the concurrent 24h move."""
    if adx_lab == "RANGING":
        return "RANGING"
    if adx_lab != "TRENDING":
        return "NEUTRAL"
    return "TREND_UP" if ret24 > 0 else "TREND_DOWN"


def day_ci(pairs, n=2000):
    by = defaultdict(list)
    for ts, r in pairs:
        by[ts // 86400].append(r)
    days = list(by.values())
    if len(days) < 8:
        return None
    ms = []
    for _ in range(n):
        s = [v for _ in range(len(days))
             for v in days[random.randrange(len(days))]]
        ms.append(sum(s) / len(s))
    ms.sort()
    return ms[int(.025 * n)], ms[int(.975 * n)]


def main() -> int:
    cells = defaultdict(list)          # cell -> [(ts, R)]
    per_coin = defaultdict(lambda: defaultdict(list))
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        c = [b[SC.C] for b in bars]
        idx = {b[0]: i for i, b in enumerate(bars)}
        adx = adx_state(bars)
        for fill_ts, _x, R, _lvl, _A, _st, pierce, _side in \
                SC.backtest_symbol(bars):
            if pierce > PIERCE_B:
                continue               # variant B = the Gate F track
            hour = int(fill_ts) // 3600 * 3600
            lab = adx.get(hour)
            i = idx.get(int(fill_ts))
            if lab is None or i is None or i < 24:
                continue
            ret24 = c[i] / c[i - 24] - 1
            cell = cell_of(lab, ret24)
            cells[cell].append((int(fill_ts), R))
            per_coin[cell][sym].append(R)

    print("Gate F attachment — directional regime split")
    print("variant B (pierce<=0.25), core9, full history, ADX(14) 25/20 "
          "x sign(24h return)\n")
    print(f"{'cell':12} {'n':>6} {'share':>7} {'meanR':>9} {'sumR':>9} "
          f"{'day-CI95':>22} {'+coins':>8}")
    res = {}
    total = sum(len(v) for v in cells.values())
    for cell in ("RANGING", "NEUTRAL", "TREND_UP", "TREND_DOWN"):
        v = cells.get(cell, [])
        if not v:
            continue
        m = st.mean(r for _, r in v)
        ci = day_ci(v)
        pos = sum(1 for s in per_coin[cell].values()
                  if s and st.mean(s) > 0)
        cis = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "—"
        print(f"{cell:12} {len(v):6d} {100*len(v)/total:6.1f}% {m:+9.4f} "
              f"{sum(r for _, r in v):+9.1f} {cis:>22} "
              f"{pos:3d}/{len(per_coin[cell]):<4d}")
        res[cell] = {"n": len(v), "meanR": round(m, 4),
                     "sumR": round(sum(r for _, r in v), 1),
                     "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                     "breadth": f"{pos}/{len(per_coin[cell])}"}

    up = res.get("TREND_UP", {}).get("meanR")
    dn = res.get("TREND_DOWN", {}).get("meanR")
    rg = res.get("RANGING", {}).get("meanR")
    print()
    if up is not None and dn is not None:
        print(f"UP vs DOWN trend: {up:+.4f} vs {dn:+.4f}  "
              f"(gap {up - dn:+.4f})")
        # pre-committed reading
        if abs(up - dn) >= 0.03 and up < dn:
            verdict = ("directional axis CONFIRMED on variant B — the "
                       "attachment must report TREND_UP / TREND_DOWN "
                       "separately, and Gate F's verdict cites all three")
        elif abs(up - dn) < 0.03:
            verdict = ("no material directional gap on variant B — product "
                       "side's 2.8x is variant-C and/or short-window "
                       "specific; undirected ADX split stays")
        else:
            verdict = ("directional gap exists but INVERTED vs the product "
                       "report (DOWN worse than UP) — investigate before "
                       "wiring anything")
        print(f"reading: {verdict}")
        res["reading"] = verdict
    if rg is not None:
        print(f"RANGING (mechanism home) {rg:+.4f} — B-P9's premise")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
