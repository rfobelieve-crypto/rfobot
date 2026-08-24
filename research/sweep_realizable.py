# -*- coding: utf-8 -*-
"""Gate F reachability recompute — TODO §0.57.

The frozen backtest fills at the LEVEL (plus adverse slippage) because the
retest touch happens intrabar. A batch-published signal cannot be acted on
until that bar CLOSES and the hourly train republishes — measured lag floor
65 minutes, observed median 342. So the frozen fill price is not a price
the follower can actually get.

This recomputes the same frozen rules with ONE substitution — entry = the
fill bar's close, the earliest price a batch consumer could truly transact
at — and reports both side by side. Everything else (sweep detection, W,
HOLD, DIS, pierce filter, exit logic, cost model) is imported unchanged
from sweep_core; the stop is re-anchored to the new entry because a real
stop is placed off the real fill.

This does NOT modify the frozen rules and is NOT a new variant: it is a
measurement of an execution constraint that already exists in production.
Exits are unaffected — once a position is open the follower watches price
live (60s poll), so stop and time exit fire on schedule; only the entry
is gated by publication.

    python research/sweep_realizable.py
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

import sweep_core as SC  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
OUT = ROOT / "research" / "results" / "sweep_realizable.json"
random.seed(7)


def backtest_realizable(bars, entry_mode: str):
    """Frozen rules; entry_mode 'level' = as frozen, 'close' = fill bar close."""
    n = len(bars)
    h = [b[SC.H] for b in bars]
    lo = [b[SC.L] for b in bars]
    c = [b[SC.C] for b in bars]
    a = SC.atr14(bars)
    trades, last_exit = [], -1
    for e in SC.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        if a[j] is None or a[j] == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd
        fill = None
        for f in range(j + 1, min(j + 1 + SC.W, n)):
            if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                fill = f
                break
        if fill is None or fill <= last_exit or fill + 1 >= n:
            continue
        A = a[j]
        if entry_mode == "level":
            entry = lvl + d * SC.SLIP * A
        else:
            # earliest actionable price for a batch consumer
            entry = c[fill] + d * SC.SLIP * A
        risk = SC.DIS * A
        stop = entry - d * risk
        R, exitbar = None, min(fill + SC.HOLD, n - 1)
        for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
            if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                R, exitbar = -1.0 - SC.SLIP / SC.DIS, k
                break
        if R is None:
            ex = c[exitbar] - d * SC.SLIP * A
            R = d * (ex - entry) / risk
        pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / A
        trades.append((bars[fill][0], R, pierce))
        last_exit = exitbar
    return trades


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
    res = {}
    for mode in ("level", "close"):
        pooled, breadth = [], 0
        per_coin = {}
        for sym in CORE9:
            fp = CACHE / f"{sym}USDT_1h.csv"
            if not fp.exists():
                continue
            tr = [(ts, R) for ts, R, p in backtest_realizable(
                SC.load_csv(str(fp)), mode) if p <= PIERCE_B]
            if not tr:
                continue
            m = st.mean(r for _, r in tr)
            per_coin[sym] = {"n": len(tr), "meanR": round(m, 4)}
            breadth += 1 if m > 0 else 0
            pooled += tr
        ci = day_ci(pooled)
        res[mode] = {"n": len(pooled),
                     "meanR": round(st.mean(r for _, r in pooled), 4),
                     "sumR": round(sum(r for _, r in pooled), 1),
                     "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                     "breadth": f"{breadth}/{len(per_coin)}",
                     "per_coin": per_coin}

    print("Gate F reachability — variant B (pierce<=0.25), core9, full history")
    print(f"{'entry':8} {'n':>6} {'meanR':>9} {'sumR':>9} "
          f"{'day-CI95':>22} {'breadth':>9}")
    for mode, lab in (("level", "frozen"), ("close", "realizable")):
        r = res[mode]
        ci = f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]" if r["ci95"] else "—"
        print(f"{lab:8} {r['n']:6d} {r['meanR']:+9.4f} {r['sumR']:+9.1f} "
              f"{ci:>22} {r['breadth']:>9}")
    gap = res["level"]["meanR"] - res["close"]["meanR"]
    print(f"\ncost of the publication gate: {gap:+.4f} R/trade "
          f"({100 * gap / res['level']['meanR']:.0f}% of the frozen edge)")
    lo = res["close"]["ci95"][0] if res["close"]["ci95"] else None
    print(f"realizable edge {'SURVIVES' if lo and lo > 0 else 'does NOT clear zero'}"
          f" on the day-clustered CI low bound")
    print("\nper-coin (realizable):")
    for s, v in res["close"]["per_coin"].items():
        f = res["level"]["per_coin"].get(s, {})
        print(f"  {s:5} n={v['n']:4d}  frozen {f.get('meanR', 0):+.4f}"
              f"  ->  realizable {v['meanR']:+.4f}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
