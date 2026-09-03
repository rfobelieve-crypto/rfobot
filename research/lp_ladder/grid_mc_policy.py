# -*- coding: utf-8 -*-
"""Monte Carlo over POLICIES, not parameters — the fix for "個位數決策點".

TODO 0.93 七 compared re-anchor / stop policies on the one real BTC path and
could not decide between fast and slow re-anchoring: 2.55 years contains only
2-14 re-anchor or stop events per asset, and the ranking flipped when the
asset changed. That is not a close call, it is an absence of evidence.

This file resamples the path instead. Stationary block bootstrap on BTC's
hourly BARS (not just closes): each sampled bar contributes its own
close-to-close return AND its own high/low as multiples of its close, so the
synthetic tape keeps realistic intrabar range — which matters here, because a
grid is filled by wicks, and a close-only path would understate every policy's
fills by the same amount but not by the same shape.

Drift is set explicitly (the历史 path's own drift is one draw, not a law):
as-is / demeaned / +30%/yr / -30%/yr. A grid earns from oscillation and dies
in trend, so the drift sweep IS the stress test.

Benchmark on every path: buy-and-hold the same capital. "Beats holding" is
reported as a fraction of paths, because a mean return that loses to holding
on 70% of paths is not a strategy, it is a lottery ticket.

Run: python research/lp_ladder/grid_mc_policy.py --paths 80
Out: research/results/lp_grid_mc_policy.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_exec import simulate  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "lp_grid_mc_policy.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load(sym="BTC"):
    lo, hi, cl = [], [], []
    with open(CACHE / f"{sym}USDT_1h.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            lo.append(float(row["low"])); hi.append(float(row["high"]))
            cl.append(float(row["close"]))
    return np.array(lo), np.array(hi), np.array(cl)


def bar_stats(low, high, close):
    """Per-bar return + intrabar range as ratios to that bar's close."""
    r = np.diff(np.log(close))
    hi_r = (high[1:] / close[1:])
    lo_r = (low[1:] / close[1:])
    return r, hi_r, lo_r


def synth(r, hi_r, lo_r, n, block, rng, drift_ann=None, demean=False):
    idx = np.empty(n, np.int64)
    i = 0
    while i < n:
        s = rng.integers(0, len(r) - block)
        take = min(block, n - i)
        idx[i:i + take] = np.arange(s, s + take)
        i += take
    rr = r[idx]
    if demean or drift_ann is not None:
        rr = rr - rr.mean()
    if drift_ann is not None:
        rr = rr + math.log(1 + drift_ann) / (365 * 24)
    c = 100_000.0 * np.exp(np.cumsum(rr))
    return c * lo_r[idx], c * hi_r[idx], c


POLICIES = {
    "上緣重錨": dict(reanchor="above", stop=None),
    "上緣重錨＋停損": dict(reanchor="above", stop="hard"),
    "每90天重錨": dict(reanchor="time", stop=None),
    "每90天重錨＋停損": dict(reanchor="time", stop="hard"),
}
SCEN = [("原樣（BTC 歷史漂移）", dict()),
        ("去漂移", dict(demean=True)),
        ("多頭 +30%/年", dict(drift_ann=0.30)),
        ("空頭 −30%/年", dict(drift_ann=-0.30))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=80)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--profile", default="nested")
    a = ap.parse_args()

    low, high, close = load("BTC")
    r, hi_r, lo_r = bar_stats(low, high, close)
    n = a.days * 24
    rng = np.random.default_rng(20260903)
    res = {"params": vars(a)}
    t0 = time.time()
    print("=" * 104)
    print(f"  MC over POLICIES — {a.paths} 條 {a.days} 天合成路徑/情境, "
          f"區塊 {a.block}h, 區間 −{a.drop:.0%}, {a.bins} 格, {a.profile}")
    print("=" * 104)

    for sname, kw in SCEN:
        # same paths for every policy in a scenario: paired comparison, so a
        # ranking cannot come from one policy drawing kinder tapes
        paths = [synth(r, hi_r, lo_r, n, a.block, rng, **kw)
                 for _ in range(a.paths)]
        bh = np.array([p[2][-1] / p[2][0] - 1 for p in paths])
        print(f"\n  [{sname}]  買進持有中位 {np.median(bh):+.1%}")
        print(f"  {'政策':<20}{'年化中位':>10}{'年化平均':>10}{'p5':>9}{'p95':>9}"
              f"{'虧損機率':>9}{'MDD中位':>9}{'MDD p95':>9}{'贏過持有':>9}"
              f"{'重錨/停損':>11}")
        res[sname] = {}
        for pname, pkw in POLICIES.items():
            rets, mdds, anc, stp, beat = [], [], [], [], []
            for k, (pl, ph, pc) in enumerate(paths):
                m, _ = simulate(pl, ph, pc, drop=a.drop, N=a.bins,
                                profile=a.profile, **pkw)
                rets.append(m["cagr"]); mdds.append(m["mdd"])
                anc.append(m["anchors"]); stp.append(m["stops"])
                beat.append((m["final"] - 1) > bh[k])
            rets = np.array(rets); mdds = np.array(mdds)
            v = {"med": float(np.median(rets)), "mean": float(rets.mean()),
                 "p5": float(np.percentile(rets, 5)),
                 "p95": float(np.percentile(rets, 95)),
                 "loss": float((rets < 0).mean()),
                 "mdd_med": float(np.median(mdds)),
                 "mdd_p95": float(np.percentile(mdds, 5)),
                 "beat_hold": float(np.mean(beat)),
                 "anchors": float(np.mean(anc)), "stops": float(np.mean(stp))}
            res[sname][pname] = v
            print(f"  {pname:<20}{v['med']:>+10.2%}{v['mean']:>+10.2%}"
                  f"{v['p5']:>+9.2%}{v['p95']:>+9.2%}{v['loss']:>9.0%}"
                  f"{v['mdd_med']:>+9.1%}{v['mdd_p95']:>+9.1%}"
                  f"{v['beat_hold']:>9.0%}"
                  f"{v['anchors']:>6.0f}/{v['stops']:<5.1f}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
