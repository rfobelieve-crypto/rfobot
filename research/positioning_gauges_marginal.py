# -*- coding: utf-8 -*-
"""Do the positioning gauges add anything BEYOND ADX? — TODO §0.65b.

The first run of positioning_gauges.py passed all five gauges at tier 2.
Five for five is the shape mistake.md 2026-08-02 says to distrust:
"跟先驗矛盾**或**完全符合先驗的漂亮結果，都要先查產生它的程式碼".
Two defects were found, and this file is the correction.

DEFECT 1 — MY TIER-2 CRITERION WAS NEARLY AUTOMATIC.
It asked whether the BEST bucket's CI low is above zero. The best bucket
was the LARGEST in 4 of 5 gauges; the largest bucket sits closest to the
population mean and carries the tightest interval, and the SF backtest
population mean is positive (+0.0856). So the test mostly asked "is the
overall edge positive", which is already known. This is the same class of
error as the "CI 下緣低於全體" false threshold caught earlier the same day:
a criterion that the current data satisfies by construction is not a
criterion. THE GAP between buckets is the claim, so the interval must be
on the GAP.

DEFECT 2 — THE GAUGES MAY ALL BE ONE GAUGE.
G1 MID / G3 normal funding / G4 quiet liquidations are plausibly the same
bars: a calm tape has mid OI, normal funding and no forced selling. And
"calm is good for SF" is ALREADY the layer's finding via ADX (§0.49:
RANGING +0.075 vs TRENDING +0.016). A gauge that reproduces ADX through a
different sensor adds nothing — that is exactly why trend_z was retired
(same effect, CI twice as wide).

So this file runs the two tests that actually decide it:

  T1 GAP CI     day-clustered bootstrap on (best bucket - worst bucket),
                resampling days ONCE so both arms move together
  T2 MARGINAL   the same gap computed INSIDE each ADX stratum. If the gap
                collapses within RANGING and within TRENDING, the gauge is
                an ADX proxy and carries no independent information
  T3 OVERLAP    how much each gauge's "good" state coincides with ADX
                RANGING — a plain contingency number, so the reader can
                see the proxy directly rather than infer it

Pre-committed reading:
  * T1 CI spans zero                    -> the gauge never had a gap
  * T1 excludes zero but T2 collapses   -> ADX proxy, no marginal value,
                                           retire it like trend_z
  * T1 excludes zero and T2 survives in
    BOTH strata with consistent sign    -> genuine independent gauge,
                                           promote to the board
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
from research.crowd_battery2 import adx_state              # noqa: E402
from research.positioning_gauges import (                  # noqa: E402
    build_gauges, sf_fills, CORE9, CACHE,
)

OUT = ROOT / "research" / "results" / "positioning_gauges_marginal.json"
random.seed(37)


def gap_ci(rows_best, rows_worst, n_boot=4000):
    """Day-clustered CI of (best - worst), days resampled once for both."""
    by = defaultdict(lambda: ([], []))
    for r in rows_best:
        by[r["ts"] // 86400][0].append(r["R"])
    for r in rows_worst:
        by[r["ts"] // 86400][1].append(r["R"])
    days = list(by)
    if len(days) < 5:
        return None
    d = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        a = [x for k in pick for x in by[k][0]]
        b = [x for k in pick for x in by[k][1]]
        if a and b:
            d.append(st.mean(a) - st.mean(b))
    if len(d) < n_boot // 2:
        return None
    d.sort()
    return d[int(.025 * len(d))], d[int(.975 * len(d))]


def main() -> int:
    gauges = build_gauges()
    sf = sf_fills()

    # ADX stratum per fill, from that coin's own bars (the frozen instrument)
    adx_by_sym = {}
    for sym in CORE9:
        fp = CACHE / f"{sym}USDT_1h.csv"
        if fp.exists():
            adx_by_sym[sym] = adx_state(SC.load_csv(str(fp)))
    for r in sf:
        lab = adx_by_sym.get(r["sym"], {}).get(r["ts"])
        r["adx"] = lab if lab in ("RANGING", "TRENDING") else "OTHER"

    print("§0.65b 持倉儀表在 ADX 之外還剩多少 —— 兩個缺陷的修正\n")
    print("  缺陷1：原判準問「最佳桶 CI 是否離零」，而最佳桶多半是最大桶，")
    print("         最大桶貼近母體均值且 CI 最窄，母體均值又是正的 → 幾乎必過。")
    print("         claim 是「桶間有差」，所以 CI 要放在**差**上。")
    print("  缺陷2：G1/G3/G4 可能是同一批 bar（平靜盤），而「平靜對 SF 好」")
    print("         已經是 ADX 的結論。重現 ADX 的儀表沒有價值（trend_z 前例）。\n")

    res = {}
    print(f"{'儀表':<18} {'差':>9} {'差的日聚類CI':>22} {'RANGING內':>11} "
          f"{'TRENDING內':>11} {'與RANGING重疊':>13}")
    for gname, gmap in gauges.items():
        buck = defaultdict(list)
        for r in sf:
            s = gmap.get(r["ts"])
            if s:
                buck[s].append(r)
        if len(buck) < 2:
            continue
        means = {k: st.mean(x["R"] for x in v) for k, v in buck.items()}
        best = max(means, key=means.get)
        worst = min(means, key=means.get)
        gap = means[best] - means[worst]
        ci = gap_ci(buck[best], buck[worst])

        # T2 — the same gap inside each ADX stratum
        strat = {}
        for lab in ("RANGING", "TRENDING"):
            b = [r for r in buck[best] if r["adx"] == lab]
            w = [r for r in buck[worst] if r["adx"] == lab]
            strat[lab] = (st.mean(x["R"] for x in b) - st.mean(x["R"] for x in w)
                          if len(b) >= 20 and len(w) >= 20 else None)

        # T3 — overlap of the good state with ADX RANGING
        ov = (100 * sum(1 for r in buck[best] if r["adx"] == "RANGING")
              / len(buck[best]))
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        f1 = f"{strat['RANGING']:+.4f}" if strat["RANGING"] is not None else "  n不足"
        f2 = f"{strat['TRENDING']:+.4f}" if strat["TRENDING"] is not None else "  n不足"
        print(f"{gname:<18} {gap:+9.4f} {cis:>22} {f1:>11} {f2:>11} "
              f"{ov:12.0f}%")
        res[gname] = {"best": best, "worst": worst, "gap": round(gap, 4),
                      "gap_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                      "within_ranging": (None if strat["RANGING"] is None
                                         else round(strat["RANGING"], 4)),
                      "within_trending": (None if strat["TRENDING"] is None
                                          else round(strat["TRENDING"], 4)),
                      "overlap_ranging_pct": round(ov, 1)}

    print("\n── 判讀（判準跑數前凍結）──")
    survivors = []
    for gname, r in sorted(res.items(), key=lambda kv: -kv[1]["gap"]):
        ci = r["gap_ci"]
        if ci is None or ci[0] <= 0:
            v = "差的 CI 含零 —— 這個儀表從來沒有差距"
        else:
            wr_, wt_ = r["within_ranging"], r["within_trending"]
            got = [x for x in (wr_, wt_) if x is not None]
            if not got:
                v = "分層後樣本不足，無法判斷邊際價值"
            elif all(x > 0 for x in got) and len(got) == 2:
                v = "**通過**：差距在 ADX 兩層內都存活且同號 —— 獨立儀表"
                survivors.append(gname)
            elif max(got) < r["gap"] * 0.4:
                v = f"ADX 代理：分層後差距塌到 {max(got):+.4f}（原 {r['gap']:+.4f}）"
            else:
                v = "部分存活，單層 —— 列觀察不列證據"
        print(f"  {gname:<18} {v}")

    print(f"\n  存活：{survivors or '無'}")
    print("  提醒：5 儀表 × 2 策略 = 10 次比較。若存活者只有一個且是邊際的，"
          "它更可能是那 0.5 個僥倖。")
    res["survivors"] = survivors
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
