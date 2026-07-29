# -*- coding: utf-8 -*-
"""Where do #3's profits actually come from? Anatomy of the winning tail.

The fill diagnostics showed the top 1% of trades carry ~69% of total R.
That raises one question worth answering and one trap worth avoiding.

The question: do those trades share an EX-ANTE property? If they do, it is a
categorical filter and the edge per trade could rise by an order of
magnitude. If they do not, #3 is confirmed as a strategy that can only win
on volume, which is itself a decision-grade answer.

The trap: hunting for "what predicts a big winner" in the same data that
produced the winners is the WQ101 machine (mistake.md 2026-06-01/02). So
this is deliberately constrained:
  * every property is CATEGORICAL and pre-existing (sweep depth in ATR,
    pivot age, retest speed, volatility regime, sweep volume, level position,
    coin tier, direction) — no thresholds are fitted, buckets are terciles or
    natural classes;
  * ALL properties are reported, none dropped;
  * a property only counts as interesting if the bucket means are MONOTONIC
    and the pattern holds in core9 and added20 SEPARATELY;
  * nothing here changes the frozen rules — a survivor becomes a new
    pre-registered forward variant, never a retrofit.

Plus the control that decides whether the question is even meaningful:
RANDOM entries with the same 3.5xATR stop and 8-bar hold. A stop-loss
strategy has bounded losses and unbounded wins, so SOME outlier
concentration is structural. If random entries concentrate the same way,
the "top 1% carries 69%" observation says nothing about the signal.

Run: python research/sweep_failure/winner_anatomy.py
Out: research/results/sweep_winner_anatomy.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_winner_anatomy.json"
CACHE = HERE / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ADDED = ["TRX", "DOT", "LTC", "UNI", "ATOM", "ETC", "NEAR", "APT", "FIL",
         "ARB", "OP", "INJ", "SUI", "AAVE", "ICP", "ALGO", "VET", "HBAR",
         "SAND", "AXS"]
TIER1, TIER2 = {"BTC", "ETH"}, set(CORE9) - {"BTC", "ETH"}
TAKER = 5.0
RNG = random.Random(11)


def net_of(R, lvl, atr):
    return R - 2 * TAKER / 1e4 * lvl / (SC.DIS * atr)


def collect(sym: str) -> list[dict]:
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return []
    bars = SC.load_csv(str(p))
    n = len(bars)
    H, L, C, V = SC.H, SC.L, SC.C, SC.V
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    vol = [b[V] for b in bars]
    a = SC.atr14(bars)
    out, last_exit = [], -1
    for e in SC.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        if a[j] is None or a[j] == 0 or j < 200:
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
        risk = SC.DIS * A
        stop = lvl - d * risk
        R, exitbar = None, min(fill + SC.HOLD, n - 1)
        for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
            if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                R, exitbar = -1.0, k
                break
        if R is None:
            R = d * (cl[exitbar] - lvl) / risk
        last_exit = exitbar

        # ── ex-ante properties, all known at or before the fill bar ──
        pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / A     # ATR units
        # pivot age: the pivot extreme sits PIVOT bars before its confirmation
        age = None
        for i in range(j - 1, max(0, j - 400), -1):
            if (kd == 1 and h[i] == lvl) or (kd == -1 and lo[i] == lvl):
                age = j - i
                break
        atr_win = [x for x in a[max(0, j - 200):j] if x]
        atr_pct = (sum(1 for x in atr_win if x < A) / len(atr_win)
                   if atr_win else 0.5)
        vwin = vol[max(0, j - 100):j]
        vmed = sorted(vwin)[len(vwin) // 2] if vwin else 0
        vshock = (vol[j] / vmed) if vmed > 0 else 1.0
        rng_win = bars[max(0, j - 168):j]
        rhi = max(b[H] for b in rng_win)
        rlo = min(b[L] for b in rng_win)
        pos = (lvl - rlo) / (rhi - rlo) if rhi > rlo else 0.5
        out.append({
            "sym": sym, "netR": net_of(R, lvl, A), "side": "SHORT" if d < 0 else "LONG",
            "pierce_atr": pierce, "pivot_age": age if age else 0,
            "retest_bars": fill - j, "atr_pct": atr_pct, "vshock": vshock,
            "range_pos": pos,
            "tier": "T1" if sym in TIER1 else ("T2" if sym in TIER2 else "T3"),
        })
    return out


def random_control(sym: str, k: int) -> list[float]:
    """Same stop/hold machinery, random entry bars and sides."""
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return []
    bars = SC.load_csv(str(p))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    a = SC.atr14(bars)
    out = []
    for _ in range(k):
        f = RNG.randrange(220, n - SC.HOLD - 2)
        A = a[f]
        if not A:
            continue
        d = RNG.choice((1, -1))
        entry = bars[f][C]
        risk = SC.DIS * A
        stop = entry - d * risk
        R, exitbar = None, min(f + SC.HOLD, n - 1)
        for k2 in range(f + 1, min(f + SC.HOLD + 1, n)):
            if (d == 1 and bars[k2][L] <= stop) or (d == -1 and bars[k2][H] >= stop):
                R, exitbar = -1.0, k2
                break
        if R is None:
            R = d * (bars[exitbar][C] - entry) / risk
        out.append(net_of(R, entry, A))
    return out


def conc(rs: list[float]) -> dict:
    s = sorted(rs, reverse=True)
    tot = sum(s)
    n = len(s)
    k = max(1, n // 100)
    return {"n": n, "mean": tot / n, "median": sorted(rs)[n // 2],
            "top1pct_share": 100 * sum(s[:k]) / tot if tot else float("nan")}


def buckets(trades, key, labels=None):
    """Terciles for numeric keys, natural classes for string keys."""
    if isinstance(trades[0][key], str):
        g = defaultdict(list)
        for t in trades:
            g[t[key]].append(t["netR"])
        return {k: g[k] for k in sorted(g)}
    vals = sorted(t[key] for t in trades)
    q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    g = {"low": [], "mid": [], "high": []}
    for t in trades:
        v = t[key]
        g["low" if v <= q1 else ("mid" if v <= q2 else "high")].append(t["netR"])
    return g


def show(name, g, tag=""):
    parts = []
    for k in g:
        rs = g[k]
        if len(rs) < 50:
            parts.append(f"{k}: n={len(rs)} (thin)")
            continue
        m = sum(rs) / len(rs)
        sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1))
        t = m / (sd / math.sqrt(len(rs)))
        parts.append(f"{k}: {m:+.4f} (t{t:+.1f}, n={len(rs)})")
    print(f"  {name:<14}{tag} " + " | ".join(parts))


def main() -> int:
    core = [t for s in CORE9 for t in collect(s)]
    added = [t for s in ADDED for t in collect(s)]
    allt = core + added
    print("=" * 78)
    print("  #3 WINNER ANATOMY — do the big winners share an ex-ante property?")
    print("=" * 78)

    ctrl = [r for s in CORE9 + ADDED for r in random_control(s, 800)]
    cs, ct = conc([t["netR"] for t in allt]), conc(ctrl)
    print(f"\n  [control] is outlier concentration structural to the exit?")
    print(f"    signal  n={cs['n']:<6} mean {cs['mean']:+.4f}  "
          f"median {cs['median']:+.4f}  top1% share {cs['top1pct_share']:.0f}%")
    print(f"    random  n={ct['n']:<6} mean {ct['mean']:+.4f}  "
          f"median {ct['median']:+.4f}  top1% share {ct['top1pct_share']:.0f}%")
    print(f"    -> concentration is {'STRUCTURAL (same as random)' if abs(cs['top1pct_share']-ct['top1pct_share'])<15 else 'signal-specific'}")

    print(f"\n  [properties] mean netR by bucket — ALL reported, none dropped")
    props = ["pierce_atr", "pivot_age", "retest_bars", "atr_pct", "vshock",
             "range_pos", "tier", "side"]
    res = {}
    for p in props:
        g = buckets(allt, p)
        show(p, g)
        res[p] = {k: {"n": len(v), "mean": (sum(v) / len(v)) if v else None}
                  for k, v in g.items()}
    print(f"\n  [split check] same buckets on core9 vs added20 separately")
    for p in props:
        show(p, buckets(core, p), "core9 ")
        show(p, buckets(added, p), "added ")

    OUT.write_text(json.dumps({"concentration": {"signal": cs, "random": ct},
                               "properties": res}, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: 8 properties x 3 buckets — expect ~1 spurious at p<0.05. "
          "Only MONOTONIC + core/added CONSISTENT counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
