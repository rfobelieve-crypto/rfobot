# -*- coding: utf-8 -*-
"""Exit variants × regime cell — EXPLORATORY, feeds the §0.59 verdict list.

Why this exists (2026-08-26): the nine pre-registered exit variants
(exit_variants.py) were scored on FULL history, i.e. on a world where the
non-home regimes still paid. §0.58 then showed the forward sample decayed
almost entirely INSIDE the non-home cells (TREND_UP -0.2475, NEUTRAL
-0.1502) while RANGING held (-0.0050). So the headline "+0.0078R for
fail_fast" was measured in a world that no longer exists in those cells,
and its sign there is unknown.

The question this answers: are the exit variants and the §0.59 regime
filter COMPETING (both try to remove the same bad trades) or COMPLEMENTARY
(the filter stops entry, fail_fast rescues the ones it lets through)?

Method: entries and exit rules are IMPORTED unchanged from
exit_variants.py, so this is the same paired comparison — identical fills,
only the exit differs — sliced by the frozen ADX×direction cell of the
fill bar. No new parameter is introduced anywhere.

STATUS AND LIMITS, stated before the numbers:
  * EXPLORATORY. 9 variants x 4 cells = 36 comparisons; at p<0.05 roughly
    two cells pass by chance alone. Nothing here may be wired to anything.
  * The forward sample was already spent proposing §0.59, so this cannot
    serve as that rule's evidence either. Its ONLY output is a decision:
    which cells the §0.59 verdict day should score exits in.
  * A cell result is only interesting if it survives the same shape in
    both halves of the sample; single-half wins are noise by default.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                   # noqa: E402
from exit_variants import VARIANTS, entries, run_exit     # noqa: E402
from research.crowd_battery2 import adx_state             # noqa: E402

CACHE = HERE / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
LB = 24
CELLS = ("RANGING", "TREND_UP", "TREND_DOWN", "NEUTRAL")
OUT = ROOT / "research" / "results" / "exit_by_regime.json"


def cell_of(lab, ret24):
    if lab == "RANGING":
        return "RANGING"
    if lab != "TRENDING":
        return "NEUTRAL"
    return "TREND_UP" if ret24 > 0 else "TREND_DOWN"


def main() -> int:
    # cell -> variant -> [R]; paired, so index i is the same fill everywhere
    data = {c: defaultdict(list) for c in CELLS}
    halves = {c: {v: ([], []) for v in VARIANTS} for c in CELLS}
    per_coin = {c: defaultdict(lambda: defaultdict(list)) for c in CELLS}

    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if not fp.exists():
            continue
        bars = SC.load_csv(str(fp))
        c = [b[SC.C] for b in bars]
        adx = adx_state(bars)
        r24 = {bars[i][0]: c[i] / c[i - LB] - 1 for i in range(LB, len(bars))}
        es = entries(bars)
        mid = len(bars) // 2
        for e in es:
            # variant B only: the Gate F track
            pierce = ((bars[e["j"]][SC.H] - e["lvl"]) if e["d"] == -1
                      else (e["lvl"] - bars[e["j"]][SC.L])) / e["A"]
            if pierce > PIERCE_B:
                continue
            ts = bars[e["fill"]][0]
            lab = adx.get(ts // 3600 * 3600)
            if lab is None or ts not in r24:
                continue
            cell = cell_of(lab, r24[ts])
            for v in VARIANTS:
                got = run_exit(bars, e, v)
                if got is None:
                    continue
                R = got[0]
                data[cell][v].append(R)
                halves[cell][v][0 if e["fill"] < mid else 1].append(R)
                per_coin[cell][v][sym].append(R)

    print("Exit variants x regime cell — variant B, core9, full history")
    print("EXPLORATORY: 9x4 comparisons, ~2 pass by chance. Wire nothing.\n")

    res = {}
    for cell in CELLS:
        base = data[cell].get("baseline", [])
        if len(base) < 60:
            print(f"{cell}: n={len(base)} too thin, skipped\n")
            continue
        bm = st.mean(base)
        print(f"── {cell}  n={len(base)}  baseline meanR {bm:+.4f} ──")
        print(f"{'variant':12} {'ΔR':>9} {'+coins':>8} {'h1 Δ':>9} "
              f"{'h2 Δ':>9} {'兩半同號':>9}")
        cell_res = {"n": len(base), "baseline": round(bm, 4), "variants": {}}
        rows = []
        for v in VARIANTS:
            if v == "baseline":
                continue
            vv = data[cell][v]
            if len(vv) != len(base):
                continue
            d = st.mean(vv) - bm
            b1, b2 = halves[cell]["baseline"]
            v1, v2 = halves[cell][v]
            d1 = (st.mean(v1) - st.mean(b1)) if len(b1) > 20 and v1 else float("nan")
            d2 = (st.mean(v2) - st.mean(b2)) if len(b2) > 20 and v2 else float("nan")
            same = (d1 > 0) == (d2 > 0) if d1 == d1 and d2 == d2 else False
            pos = sum(1 for s in per_coin[cell][v]
                      if per_coin[cell][v][s]
                      and st.mean(per_coin[cell][v][s])
                      > st.mean(per_coin[cell]["baseline"][s]))
            nco = len(per_coin[cell][v])
            rows.append((d, v, pos, nco, d1, d2, same))
            cell_res["variants"][v] = {
                "delta": round(d, 4), "coins_pos": f"{pos}/{nco}",
                "h1": round(d1, 4) if d1 == d1 else None,
                "h2": round(d2, 4) if d2 == d2 else None,
                "both_halves_same_sign": same}
        for d, v, pos, nco, d1, d2, same in sorted(rows, reverse=True):
            print(f"{v:12} {d:+9.4f} {pos:4d}/{nco:<3d} {d1:+9.4f} {d2:+9.4f} "
                  f"{'✓' if same else '✗':>8}")
        res[cell] = cell_res
        print()

    # the decision this file exists to make
    print("── 給 §0.59 判決日的清單 ──")
    for cell in CELLS:
        if cell not in res:
            continue
        good = [(x["delta"], v) for v, x in res[cell]["variants"].items()
                if x["both_halves_same_sign"] and x["delta"] > 0
                and int(x["coins_pos"].split("/")[0]) >= 6]
        good.sort(reverse=True)
        if good:
            print(f"  {cell:11} 值得在判決日重測: "
                  + ", ".join(f"{v}({d:+.4f})" for d, v in good[:3]))
        else:
            print(f"  {cell:11} 無變體同時滿足兩半同號 ∧ 廣度≥6/9")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
