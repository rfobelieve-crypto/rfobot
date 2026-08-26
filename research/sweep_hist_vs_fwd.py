# -*- coding: utf-8 -*-
"""H: why is the forward log 0.09R below the backtest? — TODO §0.58.

Product side (產品端回報_獵取執行管線_20260826.md, item H) flagged the gap
and correctly refused to optimise exits until it is explained: shadow log
carries NO execution cost (it places no orders), so slippage cannot be the
answer, and a +0.0078R exit tweak is meaningless on an unaligned baseline.

Same rules, same fill price (level), two samples:
    backtest  SC.backtest_symbol over full history
    forward   sweep_shadow_log.csv, post-FREEZE (2026-07-28+)

Two hypotheses, and this file is built to separate them:
  (1) REGIME MIX — the forward window drew an unusual regime composition
      (§0.54b already showed TREND_UP at 30% vs a 22% base rate, and it is
      the only losing cell). If so, re-weighting the forward cells to the
      historical base rate closes the gap.
  (2) WITHIN-CELL DECAY — the same regime cell now pays less than it did
      historically. Re-weighting would NOT close the gap, and the edge is
      genuinely weaker rather than unlucky.

Pre-committed reading (before running):
  - reweighted forward >= backtest - 0.02R  -> mix explains it (unlucky)
  - reweighted still >= 0.05R below         -> within-cell decay dominates
  - between                                 -> both, report the split

Read-only. Cannot pass or fail Gate F; it decides what the verdict MEANS.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
OUT = ROOT / "research" / "results" / "sweep_hist_vs_fwd.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
FREEZE = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
LB = 24
CELLS = ("TREND_UP", "TREND_DOWN", "NEUTRAL", "RANGING")


def cell_of(lab, ret24):
    if lab == "RANGING":
        return "RANGING"
    if lab != "TRENDING":
        return "NEUTRAL"
    return "TREND_UP" if ret24 > 0 else "TREND_DOWN"


def main() -> int:
    hist = defaultdict(list)      # cell -> [R]   backtest, pre-freeze only
    fwd = defaultdict(list)       # cell -> [R]   forward log
    labs, rets = {}, {}

    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        c = [b[SC.C] for b in bars]
        idx = {b[0]: i for i, b in enumerate(bars)}
        adx = adx_state(bars)
        labs[sym], rets[sym] = adx, {
            bars[i][0]: c[i] / c[i - LB] - 1 for i in range(LB, len(bars))}
        for fill_ts, _x, R, _l, _A, _s, pierce, _sd in SC.backtest_symbol(bars):
            if pierce > PIERCE_B or int(fill_ts) >= FREEZE:
                continue                      # backtest = pre-freeze only,
                #                               so the two samples never overlap
            h = int(fill_ts) // 3600 * 3600
            lab = adx.get(h)
            if lab is None or h not in rets[sym]:
                continue
            hist[cell_of(lab, rets[sym][h])].append(R)

    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "CLOSED" or r.get("variant_b") != "1":
                continue
            if r.get("universe") != "core9":
                continue
            sym = r["symbol"]
            if sym not in labs:
                continue
            try:
                ts, R = int(float(r["fill_ts"])), float(r["net_r"])
            except (ValueError, TypeError):
                continue
            h = ts // 3600 * 3600
            lab = labs[sym].get(h)
            if lab is None or h not in rets[sym]:
                continue
            fwd[cell_of(lab, rets[sym][h])].append(R)

    nh = sum(len(v) for v in hist.values())
    nf = sum(len(v) for v in fwd.values())
    mh = st.mean([x for v in hist.values() for x in v])
    mf = st.mean([x for v in fwd.values() for x in v])

    print("H — backtest (pre-freeze) vs forward (shadow log), variant B core9")
    print(f"   backtest n={nh}  meanR {mh:+.4f}")
    print(f"   forward  n={nf}  meanR {mf:+.4f}     gap {mf - mh:+.4f}\n")
    print(f"{'cell':11} {'hist n':>7} {'hist R':>9} {'fwd n':>6} {'fwd R':>9} "
          f"{'within-gap':>11} {'hist mix':>9} {'fwd mix':>8}")
    rows = {}
    for cell in CELLS:
        hv, fv = hist.get(cell, []), fwd.get(cell, [])
        if not hv or not fv:
            continue
        h_m, f_m = st.mean(hv), st.mean(fv)
        rows[cell] = {"hist_n": len(hv), "hist_meanR": round(h_m, 4),
                      "fwd_n": len(fv), "fwd_meanR": round(f_m, 4),
                      "within_gap": round(f_m - h_m, 4),
                      "hist_share": round(len(hv) / nh, 4),
                      "fwd_share": round(len(fv) / nf, 4)}
        print(f"{cell:11} {len(hv):7d} {h_m:+9.4f} {len(fv):6d} {f_m:+9.4f} "
              f"{f_m - h_m:+11.4f} {100*len(hv)/nh:8.1f}% {100*len(fv)/nf:7.1f}%")

    # counterfactual: forward cell means, historical mix
    rw = sum(rows[c]["fwd_meanR"] * rows[c]["hist_share"] for c in rows)
    rw /= sum(rows[c]["hist_share"] for c in rows)
    # and the mirror: historical cell means under the forward mix
    rw2 = sum(rows[c]["hist_meanR"] * rows[c]["fwd_share"] for c in rows)
    rw2 /= sum(rows[c]["fwd_share"] for c in rows)

    print(f"\n  forward cells @ historical mix : {rw:+.4f}   "
          f"(actual forward {mf:+.4f})")
    print(f"  historical cells @ forward mix : {rw2:+.4f}   "
          f"(actual backtest {mh:+.4f})")
    mix_effect = rw2 - mh
    within_effect = rw - mh - mix_effect if False else mf - rw
    print(f"\n  decomposition of the {mf - mh:+.4f} gap:")
    print(f"    regime-mix component   {mix_effect:+.4f}"
          f"   ({100*mix_effect/(mf-mh):.0f}% of gap)" if mf != mh else "")
    print(f"    within-cell component  {rw - mh:+.4f}"
          f"   ({100*(rw-mh)/(mf-mh):.0f}% of gap)" if mf != mh else "")

    # pre-committed reading
    if rw >= mh - 0.02:
        verdict = ("MIX explains it — forward cells at the historical mix "
                   "recover the backtest level; the window was unlucky, "
                   "not the edge decaying")
    elif rw <= mh - 0.05:
        verdict = ("WITHIN-CELL DECAY dominates — the same regime cells pay "
                   "materially less than they did historically; re-weighting "
                   "does not rescue it")
    else:
        verdict = "BOTH contribute; neither dominates"
    print(f"\n  reading: {verdict}")

    OUT.write_text(json.dumps(
        {"backtest": {"n": nh, "meanR": round(mh, 4)},
         "forward": {"n": nf, "meanR": round(mf, 4)},
         "gap": round(mf - mh, 4),
         "cells": rows,
         "forward_at_hist_mix": round(rw, 4),
         "hist_at_forward_mix": round(rw2, 4),
         "reading": verdict}, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
