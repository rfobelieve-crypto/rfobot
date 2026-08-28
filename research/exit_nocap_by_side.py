# -*- coding: utf-8 -*-
"""no_cap's gain, split by side — the beta check. Companion to §0.77.

The stop-only long-hold arm showed +0.1006R over baseline at 9/9 breadth.
Average hold is 220 bars (~9 days) and the sample era is a bull market, so
the cheap explanation is BETA: longs drift up while held, shorts bleed —
the gain is the market's, not the exit's (the same anatomy the mill audit
found: money from the market, not from skill).

If the delta is positive on BOTH sides, the long-hold effect is real.
If it is a LONG-only story, §0.77's control arm is drift and dies too.
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

import sweep_core as SC                                    # noqa: E402
from exit_variants import entries, run_exit                # noqa: E402
from research.exit_opp_raid import (                        # noqa: E402
    CACHE, CORE9, PIERCE_B, clustered_ci, run_long_exit,
)

OUT = ROOT / "research" / "results" / "exit_nocap_by_side.json"
random.seed(233)


def main() -> int:
    rows = []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        for e in entries(bars):
            pierce = ((bars[e["j"]][SC.H] - e["lvl"]) if e["d"] == -1
                      else (e["lvl"] - bars[e["j"]][SC.L])) / e["A"]
            if pierce > PIERCE_B:
                continue
            got = run_exit(bars, e, "baseline")
            if got is None:
                continue
            nR, _ = run_long_exit(bars, e, [], use_signal=False)
            rows.append({"ts": bars[e["fill"]][0], "sym": sym,
                         "d": e["d"], "dif": nR - got[0]})

    print("§0.77b no_cap − baseline，分方向（beta 檢查）")
    print(f"  n={len(rows)}\n")
    mid = sorted(r["ts"] for r in rows)[len(rows) // 2]
    res = {}
    for d, lab in ((1, "LONG（掃低點後做多）"), (-1, "SHORT（掃高點後做空）")):
        v = [r for r in rows if r["d"] == d]
        m = st.mean(r["dif"] for r in v)
        ci = clustered_ci([(r["ts"] // 86400, r["dif"]) for r in v])
        per = defaultdict(list)
        for r in v:
            if r["sym"] in CORE9:
                per[r["sym"]].append(r["dif"])
        br = sum(1 for s in per if st.mean(per[s]) > 0)
        h1 = st.mean(r["dif"] for r in v if r["ts"] < mid)
        h2 = st.mean(r["dif"] for r in v if r["ts"] >= mid)
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"  {lab:24} n={len(v):<6} Δ {m:+.4f}  CI {cis:<20} "
              f"廣度 {br}/9  前半 {h1:+.4f} 後半 {h2:+.4f}")
        res[lab] = {"n": len(v), "delta": round(m, 4),
                    "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                    "breadth": f"{br}/9",
                    "h1": round(h1, 4), "h2": round(h2, 4)}

    L = res["LONG（掃低點後做多）"]
    S = res["SHORT（掃高點後做空）"]
    if L["delta"] > 0.03 and S["delta"] <= 0:
        v = ("**beta**：加值全在多單、空單不賺甚至倒貼 —— 長持有吃的是"
             "多頭市的漂移,不是出場的功勞。§0.77 的對照臂死於此。")
    elif L["delta"] > 0 and S["delta"] > 0:
        v = ("兩側皆正 —— 不是純 beta。但仍需過吞吐帳"
             "（220 根持有 ≈ 年化機會 -73%）與後半衰退那兩關。")
    else:
        v = "形狀不典型，逐格判讀"
    print(f"\n判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
