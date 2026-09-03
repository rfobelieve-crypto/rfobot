# -*- coding: utf-8 -*-
"""Does the survival layer's ADX regime instrument help the grid?

The grid's two structural weaknesses (TODO 0.93 八) are the same thing seen
twice: it loses to holding in an uptrend and it bleeds in a sustained
downtrend. The repo already owns a validated trend instrument -- ADX(14)
25/20 with the §0.54b direction split (§0.49d, frozen 2026-08-17; the same
`adx_state` the shadow recorder freezes into `regime_cell`).

But it was validated on the sweep-failure line and on V7, NOT on a grid.
Transferring it is a NEW claim, so it gets tested the same way the policies
were: paired Monte Carlo, same synthetic paths for every arm.

Gates tested (all of them block only NEW rungs -- sells, stops and
re-anchors always run; a filter that blocked exits would be a different
strategy):
    TRENDING-off     stop buying whenever ADX > 25 (either direction)
    TREND_DOWN-off   stop buying only in a down-trend -- the mechanism says
                     that is the one that kills a ladder
    TRENDING-half    keep buying at half size in a trend (the §0.52 reefing
                     shape: do not predict the storm, just shorten sail)

Causality: the label for bar i is computed from bars up to i-1 and applied
to bar i. A grid fills on the bar's wick, so using bar i's own close to
decide bar i's buys would be look-ahead.

Run: python research/lp_ladder/grid_adx_gate.py --paths 200
Out: research/results/lp_grid_adx_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from grid_exec import simulate  # noqa: E402
from grid_mc_policy import bar_stats, load, synth  # noqa: E402

OUT = ROOT / "research" / "results" / "lp_grid_adx_gate.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def adx_labels(high, low, close, n_=14, lb=24):
    """Per-bar 'TRENDING'/'RANGING'/'NEUTRAL' + up/down split.

    Same recursion as research/crowd_battery2.adx_state (the frozen §0.49d
    instrument), rewritten over arrays; verified against it on real bars in
    __main__ so this cannot quietly become a second implementation.
    Returns (state, is_up) arrays aligned to the input bars.
    """
    n = len(close)
    st = np.full(n, "", dtype=object)
    tr_s = pdm_s = ndm_s = 0.0
    adx = None
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        um, dm = high[i] - high[i - 1], low[i - 1] - low[i]
        pdm = um if (um > dm and um > 0) else 0.0
        ndm = dm if (dm > um and dm > 0) else 0.0
        if i <= n_:
            tr_s += tr; pdm_s += pdm; ndm_s += ndm
            continue
        tr_s = tr_s - tr_s / n_ + tr
        pdm_s = pdm_s - pdm_s / n_ + pdm
        ndm_s = ndm_s - ndm_s / n_ + ndm
        if tr_s <= 0:
            continue
        pdi, ndi = 100 * pdm_s / tr_s, 100 * ndm_s / tr_s
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi > 0 else 0.0
        adx = dx if adx is None else (adx * (n_ - 1) + dx) / n_
        if i > 2 * n_:
            st[i] = ("TRENDING" if adx > 25 else
                     "RANGING" if adx < 20 else "NEUTRAL")
    up = np.zeros(n, bool)
    up[lb:] = close[lb:] > close[:-lb]
    return st, up


def gates(high, low, close):
    st, up = adx_labels(high, low, close)
    trending = (st == "TRENDING")
    # shift by one bar: decide bar i from information available at i-1
    def shift(x):
        y = np.zeros(len(x), bool)
        y[1:] = x[:-1]
        return y
    tr = shift(trending)
    td = shift(trending & ~up)
    return {"TRENDING-off": ~tr, "TREND_DOWN-off": ~td, "TRENDING-half": ~tr,
            "ALWAYS-OFF": np.zeros(len(close), bool)}


BASE = dict(reanchor="time", stop="hard")       # the tail-controlled policy
ARMS = [("無濾網（基準）", None, 0.0),
        ("TRENDING 關閉買進", "TRENDING-off", 0.0),
        ("TREND_DOWN 關閉買進", "TREND_DOWN-off", 0.0),
        ("TRENDING 半倉（縮帆）", "TRENDING-half", 0.5),
        # THE control that decides whether ADX carries information at all.
        # "Half size half the time" averages ~75% deployment, so if a flat
        # 75% does the same thing, the regime label added nothing and the
        # gain was just de-risking wearing an indicator's clothes.
        ("固定 75% 倉位（對照）", "ALWAYS-OFF", 0.75)]
SCEN = [("去漂移", dict(demean=True)),
        ("多頭 +30%/年", dict(drift_ann=0.30)),
        ("空頭 −30%/年", dict(drift_ann=-0.30)),
        ("空頭 −60%/年", dict(drift_ann=-0.60))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--bins", type=int, default=30)
    a = ap.parse_args()

    low, high, close = load("BTC")
    # parity check against the frozen instrument before using it
    sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
    from research.crowd_battery2 import adx_state
    import sweep_core as SC
    bars = SC.load_csv(str(ROOT / "research" / "sweep_failure" / ".cache"
                           / "BTCUSDT_1h.csv"))
    ref = adx_state(bars)
    mine, _ = adx_labels(np.array([b[SC.H] for b in bars]),
                         np.array([b[SC.L] for b in bars]),
                         np.array([b[SC.C] for b in bars]))
    same = sum(1 for i, b in enumerate(bars)
               if ref.get(b[0] // 3600 * 3600) == mine[i] and mine[i])
    tot = sum(1 for i, b in enumerate(bars) if mine[i])
    print(f"  儀器對帳：與 crowd_battery2.adx_state 相同標籤 {same}/{tot} "
          f"({100*same/max(tot,1):.2f}%)")
    if same / max(tot, 1) < 0.999:
        print("  !! 標籤不一致，這是第二份實作——停手，先修對帳再談結果")
        return 1

    r, hi_r, lo_r = bar_stats(low, high, close)
    n = a.days * 24
    rng = np.random.default_rng(20260903)
    res = {"params": vars(a)}
    t0 = time.time()
    print("=" * 100)
    print(f"  ADX 趨勢閘門 × 網格 — {a.paths} 條 {a.days} 天路徑/情境，"
          f"基準政策＝每90天重錨＋硬停損，區間 −{a.drop:.0%}")
    print("=" * 100)

    for sname, kw in SCEN:
        paths = [synth(r, hi_r, lo_r, n, a.block, rng, **kw)
                 for _ in range(a.paths)]
        bh = np.array([p[2][-1] / p[2][0] - 1 for p in paths])
        gts = [gates(p[1], p[0], p[2]) for p in paths]
        print(f"\n  [{sname}]  買進持有中位 {np.median(bh):+.1%}")
        print(f"  {'閘門':<22}{'年化中位':>10}{'年化平均':>10}{'p5':>9}"
              f"{'虧損機率':>9}{'MDD中位':>9}{'MDD p95':>9}{'贏過持有':>9}"
              f"{'擋掉時間':>9}")
        res[sname] = {}
        for aname, gkey, gscale in ARMS:
            rets, mdds, beat, gof = [], [], [], []
            for k, (pl, ph, pc) in enumerate(paths):
                g = None if gkey is None else gts[k][gkey]
                m, _ = simulate(pl, ph, pc, drop=a.drop, N=a.bins,
                                gate=g, gate_scale=gscale, **BASE)
                rets.append(m["cagr"]); mdds.append(m["mdd"])
                beat.append((m["final"] - 1) > bh[k]); gof.append(m["gated_frac"])
            rets = np.array(rets); mdds = np.array(mdds)
            v = {"med": float(np.median(rets)), "mean": float(rets.mean()),
                 "p5": float(np.percentile(rets, 5)),
                 "loss": float((rets < 0).mean()),
                 "mdd_med": float(np.median(mdds)),
                 "mdd_p95": float(np.percentile(mdds, 5)),
                 "beat_hold": float(np.mean(beat)),
                 "gated": float(np.mean(gof))}
            res[sname][aname] = v
            print(f"  {aname:<22}{v['med']:>+10.2%}{v['mean']:>+10.2%}"
                  f"{v['p5']:>+9.2%}{v['loss']:>9.0%}{v['mdd_med']:>+9.1%}"
                  f"{v['mdd_p95']:>+9.1%}{v['beat_hold']:>9.0%}"
                  f"{v['gated']:>9.0%}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
