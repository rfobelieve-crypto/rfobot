# -*- coding: utf-8 -*-
"""CVD divergence at the raid — the last untested cell of the A-L menu
(third paste, 2026-08-02). Combo B's divergence leg: price takes the pool
(a new local extreme by construction) while cumulative delta FAILS to make
a corresponding extreme.

Event-anchored, causal definition (computed on the sweep bar's close):
  hourly delta = sum of minute taker deltas (flow_bars), window = trailing
  24 completed hours ending AT the sweep bar. CD = running cumsum inside
  the window, signed INTO the break direction (s x delta), so "CVD made
  its window high together with price" reads the same for both sides.
    div_gap  = (max CD over window - CD at sweep) / std(hourly deltas)
               0 = CVD confirmed the extreme; larger = bigger divergence
    div_flag = CD at sweep < max CD over the PRIOR bars of the window

Named predictions (combo B): divergence -> reversal (better netR, fewer
breakouts); CVD confirming -> continuation. Bar: monotone + both symbols
+ halves. BTC+ETH, ~100d. 2 looks.

Run: python research/sweep_raid_divergence.py
Out: research/results/sweep_raid_divergence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_divergence.json"
LOOK = 24


def hourly_delta(flow, hh):
    m0 = hh * 60
    win = [flow[m] for m in range(m0, m0 + 60) if m in flow]
    if len(win) < 40:
        return None
    return sum(w[1] for w in win)


def attach_div(rows, flow):
    out = []
    for r in rows:
        hh = r["ts"] // 3600
        s = r["side"]
        ds = []
        ok = True
        for k in range(LOOK, -1, -1):
            d = hourly_delta(flow, hh - k)
            if d is None:
                ok = False
                break
            ds.append(s * d)
        if not ok:
            continue
        cd = np.cumsum(ds)
        sd = float(np.std(ds))
        if sd <= 0:
            continue
        x = dict(r)
        x["div_gap"] = float((cd.max() - cd[-1]) / sd)
        x["div_flag"] = int(cd[-1] < cd[:-1].max())
        out.append(x)
    return out


def terc(rows, key, target, label=""):
    vals = sorted(r[key] for r in rows)
    if len(vals) < 90:
        return None, f"  {label}{key:<9} thin (n={len(vals)})"
    lo_c, hi_c = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    parts, rec = [], {}
    for nm, pr in (("low", lambda v: v <= lo_c), ("mid", lambda v: lo_c < v < hi_c),
                   ("high", lambda v: v >= hi_c)):
        g = [r for r in rows if pr(r[key])]
        if not g:
            continue
        if target == "netR":
            xs = [r["netR"] for r in g if r["netR"] is not None]
            m = sum(xs) / len(xs)
            wr = 100 * sum(1 for x in xs if x > 0) / len(xs)
            rec[nm] = {"n": len(g), "netR": round(m, 3), "wr": round(wr, 1)}
            parts.append(f"{nm} {m:+.3f}/{wr:.0f}% (n={len(g)})")
        else:
            br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
            rec[nm] = {"n": len(g), "breakout_pct": round(br, 1)}
            parts.append(f"{nm} 突破{br:.0f}% (n={len(g)})")
    return rec, f"  {label}{key:<9}" + " | ".join(parts)


def main() -> int:
    print("=" * 78)
    print("  CVD DIVERGENCE at raid — A-L 菜單最後一格（2 looks；單調+雙幣+兩半）")
    print("=" * 78)
    allr = []
    for sym, canon in A.SYMS.items():
        flow, _i, _c = A.load_flow(canon)
        rr = attach_div(raids_with_fill(sym), flow)
        print(f"  {sym}: {len(rr)} raids with 24h flow window")
        allr += rr
    allr.sort(key=lambda r: r["ts"])
    res = {}
    br = 100 * sum(1 for r in allr if r["cls"] == "BREAKOUT") / len(allr)
    print(f"  pooled n={len(allr)}, 基準突破 {br:.0f}%\n")
    for tgt in ("cls", "netR"):
        rec, line = terc(allr, "div_gap", tgt)
        res[f"pool_{tgt}"] = rec
        print(line)
    g0 = [r for r in allr if r["div_flag"] == 0]
    g1 = [r for r in allr if r["div_flag"] == 1]
    for nm, g in (("CVD確認(無背離)", g0), ("有背離", g1)):
        b = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
        xs = [r["netR"] for r in g if r["netR"] is not None]
        print(f"  {nm:<12} n={len(g):>4}  突破{b:.0f}%  netR {sum(xs)/len(xs):+.3f}")
        res[f"flag_{nm}"] = {"n": len(g), "breakout_pct": round(b, 1),
                             "netR": round(sum(xs) / len(xs), 3)}
    print("\n  [split] 雙幣 + 兩半 (div_gap)")
    half = len(allr) // 2
    for tag, seg in (("BTC", [r for r in allr if r["sym"] == "BTC"]),
                     ("ETH", [r for r in allr if r["sym"] == "ETH"]),
                     ("H1", allr[:half]), ("H2", allr[half:])):
        for tgt in ("cls", "netR"):
            _, line = terc(seg, "div_gap", tgt, label=f"{tag:<4}")
            print(line)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  具名預測: 背離大 → 反轉(netR高·突破少); CVD確認 → 延續。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
