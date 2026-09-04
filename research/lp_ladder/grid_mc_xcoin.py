# -*- coding: utf-8 -*-
"""0.93 XI -- is the frozen policy fitted to BTC's volatility? (pre-registered)

0.93 eight/ten established the policy (slow re-anchor + hard stop, delay 0h)
on paths resampled from BTC's own hourly bars. Seven ran the SAME policy on
nine coins' HISTORICAL paths and got 7/9 positive -- but with only 2-14
decision points per coin, which is the exact weakness the MC was built to
fix. So the policy conclusions rest on one asset's bar statistics.

WHAT THIS DOES AND DOES NOT ASK
The frozen candidate rule's first line is "標的 BTC". So this is NOT testing
a claim the rule makes. It asks the diagnostic question behind that line:
is "BTC only" a CONSERVATIVE restriction (the policy also works elsewhere)
or a NECESSARY one (it only works where volatility is small enough not to
punch through a 25% range)?

PRE-REGISTERED PREDICTIONS (written before the run)
  X1  the frozen policy's median annual return is positive on >= 5 of 9
      coins in the DE-DRIFTED scenario (the one that isolates "does it
      harvest chop" from "did the asset go up")
  X2  loss probability is monotone in the coin's realised volatility --
      i.e. the failures are the high-vol coins punching through the range,
      not something idiosyncratic
  X3  BTC is NOT the best coin. If BTC ranks first on median return, the
      policy is suspected of being fitted to it and the "BTC only" line
      stays as a hard restriction rather than a conservative default.
READ: X1 tells us whether the restriction is conservative. X3 is the
overfit check on ourselves. Neither authorises trading anything.

Same rig as ten: block bootstrap over whole hourly bars (wicks preserved),
every coin gets the SAME number of paths and the same policy, no fitting.

Run: python research/lp_ladder/grid_mc_xcoin.py --paths 100
Out: research/results/lp_grid_mc_xcoin.json
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
OUT = ROOT / "research" / "results" / "lp_grid_mc_xcoin.json"
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
FROZEN = dict(drop=0.25, N=30, profile="nested",
              reanchor="time", stop="hard", stop_delay_h=0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--block", type=int, default=48)
    a = ap.parse_args()
    print("=" * 104)
    print(f"  §0.93 十一 跨幣政策 MC——凍結規則是不是只對 BTC 有效（預註冊）"
          f"｜{a.paths} 條 {a.days} 天路徑/幣/情境")
    print("=" * 104)
    res, t0 = {"params": vars(a), "policy": FROZEN, "coins": {}}, time.time()
    for scen_name, kw in (("去漂移", dict(demean=True)), ("原樣", dict())):
        print(f"\n  [{scen_name}]  {'幣':<6}{'年化中位':>10}{'p5':>9}{'虧損':>7}"
              f"{'MDD中':>8}{'MDDp95':>9}{'實現年化波動':>12}{'停損/年':>9}")
        res["coins"].setdefault(scen_name, {})
        for sym in SYMS:
            try:
                low, high, close = load(sym)
            except Exception as e:                       # noqa: BLE001
                print(f"  {sym:<6} 無資料 ({e})")
                continue
            r, hi_r, lo_r = bar_stats(low, high, close)
            vol = float(np.std(r) * np.sqrt(365 * 24))
            rng = np.random.default_rng(20260904)
            rets, mdds, stops = [], [], []
            for _ in range(a.paths):
                pl, ph, pc = synth(r, hi_r, lo_r, a.days * 24, a.block, rng, **kw)
                m, _ = simulate(pl, ph, pc, **FROZEN)
                rets.append(m["cagr"]); mdds.append(m["mdd"]); stops.append(m["stops"])
            rets, mdds = np.array(rets), np.array(mdds)
            v = {"med": float(np.median(rets)), "p5": float(np.percentile(rets, 5)),
                 "loss": float((rets < 0).mean()), "mdd_med": float(np.median(mdds)),
                 "mdd_p95": float(np.percentile(mdds, 5)),
                 "vol_ann": round(vol, 3), "stops": float(np.mean(stops))}
            res["coins"][scen_name][sym] = v
            print(f"          {sym:<6}{v['med']:>+10.2%}{v['p5']:>+9.2%}{v['loss']:>7.0%}"
                  f"{v['mdd_med']:>+8.1%}{v['mdd_p95']:>+9.1%}{vol:>12.2f}{v['stops']:>9.2f}")

    d = res["coins"].get("去漂移", {})
    pos = [k for k, v in d.items() if v["med"] > 0]
    by_med = sorted(d, key=lambda k: -d[k]["med"])
    # X2: is loss probability monotone in vol? use rank correlation
    ks = list(d)
    if len(ks) >= 3:
        import statistics as st
        vr = {k: i for i, k in enumerate(sorted(ks, key=lambda k: d[k]["vol_ann"]))}
        lr = {k: i for i, k in enumerate(sorted(ks, key=lambda k: d[k]["loss"]))}
        n = len(ks)
        rho = 1 - 6 * sum((vr[k] - lr[k]) ** 2 for k in ks) / (n * (n * n - 1))
    else:
        rho = float("nan")
    bars = {
        "X1 去漂移下 ≥5/9 幣中位為正": len(pos) >= 5,
        "X2 虧損機率與波動同向（rank rho ≥ 0.5）": rho >= 0.5,
        "X3 BTC 不是最好的那個幣": bool(by_med and by_med[0] != "BTC"),
    }
    print(f"\n  去漂移下中位為正：{len(pos)}/{len(d)} → {pos}")
    print(f"  依中位排名：{by_med}")
    print(f"  虧損機率 vs 波動 rank 相關 ρ = {rho:+.2f}")
    for k, v in bars.items():
        print(f"    {'✅' if v else '❌'} {k}")
    res.update({"bars": bars, "positive_coins": pos, "rank_by_median": by_med,
                "rho_loss_vol": round(rho, 3)})
    res["verdict"] = ("「BTC only」是保守限制——政策在多數幣上也活"
                      if bars["X1 去漂移下 ≥5/9 幣中位為正"]
                      else "「BTC only」是必要限制——政策只在低波動標的成立")
    print(f"  → {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {time.time()-t0:.0f}s  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
