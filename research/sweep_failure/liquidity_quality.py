# -*- coding: utf-8 -*-
"""Is the liquidity DEFINITION load-bearing? Two tests the user's critique demands.

The critique (2026-07-29): #3 defines liquidity purely by position — a pivot
high/low — and the pivot strength (10 bars each side) is an arbitrary number.
The LMSR Pine map the user shared draws far richer pools (equal levels, session
extremes, PDH/PDL, PWH/PWL, ATR-scaled depth bands) but its own header states
they are "drawn but never fire" — nobody has tested whether raiding those
behaves the same way.

Two of those can be tested right now on the full 29-coin, 30-month history
(22k events), with no dependence on the order-flow tables:

  1 EQUAL LEVELS. Two or more pivots printing the same price to within a
    tolerance are the densest stop cluster on the chart — that is where the
    orders actually are. If liquidity depth matters at all, sweeping a cluster
    should behave differently from sweeping a lone pivot. Bucketed by how many
    same-side pivots sit within tol x ATR of the level. Categorical, no fitted
    threshold; the tolerance is swept and all values reported.

  2 PIVOT STRENGTH. If 10 is special, the original research tuned it and the
    edge is partly a fit. If the result is a PLATEAU across 3..20, the number
    is not load-bearing and the effect is structural. This is the healthier
    outcome and the one worth hoping for; a peak exactly at 10 is a red flag.

Everything is measured on net R under the scenario-A cost model. Nothing here
changes the frozen rules — a survivor becomes a pre-registered variant, and
only one of them, because each extra variant costs multiple-testing burden
against every other candidate (the Deflated Sharpe N).

Run: python research/sweep_failure/liquidity_quality.py
Out: research/results/sweep_liquidity_quality.json
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_liquidity_quality.json"
CACHE = HERE / ".cache"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX",
         "TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]
TAKER = 5.0
EQ_TOLS = [0.05, 0.10, 0.20]      # x ATR, all reported


def net(R, lvl, atr):
    return R - 2 * TAKER / 1e4 * lvl / (SC.DIS * atr)


def stat(rs):
    n = len(rs)
    if n < 40:
        return None
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    return {"n": n, "mean": m, "t": m / (sd / math.sqrt(n))}


def fmt(s):
    return f"{s['mean']:+.4f} (t{s['t']:+.1f}, n={s['n']})" if s else "thin"


def equal_level_test():
    """For each sweep, count same-side pivots within tol x ATR of its level."""
    print("=" * 78)
    print("  [1] EQUAL LEVELS — does stop-cluster density matter?")
    print("=" * 78)
    res = {}
    for tol in EQ_TOLS:
        buckets = {"1 (lone)": [], "2": [], "3+": []}
        for sym in COINS:
            p = CACHE / f"{sym}USDT_1h.csv"
            if not p.exists():
                continue
            bars = SC.load_csv(str(p))
            a = SC.atr14(bars)
            evs = SC.detect_sweeps(bars)
            # every confirmed pivot level, by side, for the density count
            highs = [e["level"] for e in evs if e["kind"] == "buy"]
            lows = [e["level"] for e in evs if e["kind"] == "sell"]
            trades = SC.backtest_symbol(bars)
            # map fill_ts -> level so the trade can be matched to its pivot
            lvl_by_ts = {t[0]: (t[3], t[4]) for t in trades}
            side_by_lvl = {}
            for e in evs:
                side_by_lvl.setdefault(e["level"], e["kind"])
            for (fill_ts, _x, R, lvl, atr, _s, _p) in trades:
                pool = highs if side_by_lvl.get(lvl) == "buy" else lows
                d = tol * atr
                k = sum(1 for q in pool if abs(q - lvl) <= d)
                key = "1 (lone)" if k <= 1 else ("2" if k == 2 else "3+")
                buckets[key].append(net(R, lvl, atr))
        line = "  ".join(f"{k}: {fmt(stat(v))}" for k, v in buckets.items())
        print(f"  tol={tol:.2f} ATR   {line}")
        res[f"tol_{tol}"] = {k: stat(v) for k, v in buckets.items()}
    return res


def pivot_strength_test():
    """Re-run the whole engine at several pivot strengths."""
    print("\n" + "=" * 78)
    print("  [2] PIVOT STRENGTH — is 10 special (fitted) or a plateau (structural)?")
    print("=" * 78)
    print(f"  {'PIVOT':>6}{'n':>8}{'meanR':>10}{'t':>8}   (validated value = 10)")
    res = {}
    for pv in (3, 5, 7, 10, 15, 20):
        os.environ["PIVOT"] = str(pv)
        importlib.reload(SC)
        rs = []
        for sym in COINS:
            p = CACHE / f"{sym}USDT_1h.csv"
            if not p.exists():
                continue
            for (_t, _x, R, lvl, atr, _s, _pc) in SC.backtest_symbol(
                    SC.load_csv(str(p))):
                rs.append(net(R, lvl, atr))
        s = stat(rs)
        res[pv] = s
        mark = "  <- validated" if pv == 10 else ""
        print(f"  {pv:>6}{s['n']:>8}{s['mean']:>+10.4f}{s['t']:>+8.2f}{mark}")
    os.environ["PIVOT"] = "10"
    importlib.reload(SC)
    return res


def main() -> int:
    eq = equal_level_test()
    pv = pivot_strength_test()
    OUT.write_text(json.dumps({"equal_levels": eq, "pivot_strength": pv},
                              indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: a PLATEAU in [2] is the good outcome — it means the arbitrary "
          "10 is not carrying the result. A peak at 10 would mean it was fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
