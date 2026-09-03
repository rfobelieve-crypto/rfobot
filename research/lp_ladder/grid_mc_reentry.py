"""TODO 0.93 十 — the two items 八 left open, on the same paired-MC rig.

1. Re-entry delay after a hard stop. Seven/eight re-anchored the instant a
   stop fired ("停損完立刻站回去"). Here the ladder stays flat for
   {0, 24, 72, 168, 336} h after a stop, then re-anchors at the price then.
2. Range width x policy. Eight fixed the range at -25%; here -25% and -35%
   (the 六 sweet spot's two ends) run against every policy.

Same discipline as grid_mc_policy.py: block bootstrap over whole hourly bars
(wicks preserved -- the grid is fed by wicks), every configuration in a
scenario sees the SAME synthetic paths (paired comparison), nothing is
fitted. Verdict metrics are the fat-left-tail ones (loss probability, p5,
MDD p95, beat-hold), not a mean CI -- see 八's note on 判準.

Run: python research/lp_ladder/grid_mc_reentry.py --paths 100
Out: research/results/lp_grid_mc_reentry.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grid_exec import simulate  # noqa: E402
from grid_mc_policy import load, bar_stats, synth  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "lp_grid_mc_reentry.json"

DELAYS_H = [0, 24, 72, 168, 336]
DROPS = [0.25, 0.35]
SCEN = [("去漂移", dict(demean=True)),
        ("原樣（BTC 歷史漂移）", dict()),
        ("空頭 −30%/年", dict(drift_ann=-0.30)),
        ("空頭 −60%/年", dict(drift_ann=-0.60))]


def configs():
    for drop in DROPS:
        yield f"上緣重錨·無停損·−{drop:.0%}", dict(drop=drop, reanchor="above", stop=None)
        yield f"每90天重錨·無停損·−{drop:.0%}", dict(drop=drop, reanchor="time", stop=None)
        for d in DELAYS_H:
            yield (f"每90天重錨＋停損·延遲{d:>3}h·−{drop:.0%}",
                   dict(drop=drop, reanchor="time", stop="hard", stop_delay_h=d))
        for d in DELAYS_H:
            yield (f"上緣重錨＋停損·延遲{d:>3}h·−{drop:.0%}",
                   dict(drop=drop, reanchor="above", stop="hard", stop_delay_h=d))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--bins", type=int, default=30)
    a = ap.parse_args()

    low, high, close = load("BTC")
    r, hi_r, lo_r = bar_stats(low, high, close)
    n = a.days * 24
    rng = np.random.default_rng(20260903)
    res = {"params": vars(a), "delays_h": DELAYS_H, "drops": DROPS}
    t0 = time.time()
    print("=" * 110)
    print(f"  0.93 十 — 再進場延遲 × 區間寬度 × 政策；{a.paths} 條 {a.days} 天路徵/情境，"
          f"區塊 {a.block}h，{a.bins} 格 nested")
    print("=" * 110)
    for sname, kw in SCEN:
        paths = [synth(r, hi_r, lo_r, n, a.block, rng, **kw) for _ in range(a.paths)]
        bh = np.array([p[2][-1] / p[2][0] - 1 for p in paths])
        print(f"\n  [{sname}]  買進持有中位 {np.median(bh):+.1%}")
        print(f"  {'配置':<34}{'年化中位':>9}{'p5':>9}{'虧損':>6}{'MDD中':>8}{'MDDp95':>8}"
              f"{'贏持有':>7}{'停損/年':>8}")
        res[sname] = {}
        for cname, ckw in configs():
            rets, mdds, stp, beat = [], [], [], []
            for k, (pl, ph, pc) in enumerate(paths):
                m, _ = simulate(pl, ph, pc, N=a.bins, profile="nested", **ckw)
                rets.append(m["cagr"]); mdds.append(m["mdd"]); stp.append(m["stops"])
                beat.append((m["final"] - 1) > bh[k])
            rets = np.array(rets); mdds = np.array(mdds)
            v = {"med": float(np.median(rets)), "mean": float(rets.mean()),
                 "p5": float(np.percentile(rets, 5)),
                 "loss": float((rets < 0).mean()),
                 "mdd_med": float(np.median(mdds)),
                 "mdd_p95": float(np.percentile(mdds, 5)),
                 "beat_hold": float(np.mean(beat)),
                 "stops": float(np.mean(stp))}
            res[sname][cname] = v
            print(f"  {cname:<34}{v['med']:>+9.2%}{v['p5']:>+9.2%}{v['loss']:>6.0%}"
                  f"{v['mdd_med']:>+8.1%}{v['mdd_p95']:>+8.1%}{v['beat_hold']:>7.0%}"
                  f"{v['stops']:>8.2f}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
