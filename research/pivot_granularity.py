# -*- coding: utf-8 -*-
"""Does a finer PIVOT buy samples without buying noise? — TODO §0.73.

The operator's TV screenshot shows the reference indicator marking roughly
twice as many swing points as this system does. The cause is not a bug:
the indicator uses len_l=4, the frozen rule uses PIVOT=10.

THIS IS NOT A PARAMETER HUNT. The motivation is sample starvation, which
was measured directly: BTC produced 2 variant-B events in a full week
(§0.72). Everything on this line — Gate F, §0.59, the entry model, the
confluence candidate — is rate-limited by event count, and the clocks run
in months because of it.

sweep_core's own header already records that PIVOT in {5, 10} was
robustness-checked and both were positive, so 5 is not a new claim. 4 has
not been tested and sits next to 5.

THE CRITERION IS NOT "WHICH PIVOT SCORES BEST". Picking the top scorer
across a parameter grid is the 2026-06-20 trap. The question is narrower
and pre-committed here:

    does a finer pivot multiply EVENTS while leaving meanR intact?

  * events rise materially AND meanR holds within ~20% AND per-coin
    breadth holds  -> a finer pivot buys clock speed at no cost, and is
    worth pre-registering as a sample-rate change
  * meanR degrades with granularity -> PIVOT=10 is load-bearing, the
    coarseness IS the filter, and the sample starvation is the price of
    the edge
  * meanR IMPROVES at finer pivots -> treat with suspicion, not delight:
    that is the shape of a fitted parameter and needs its own forward test

Every value is reported. Nothing is selected.
"""
from __future__ import annotations

import importlib
import json
import os
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

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "pivot_granularity.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
PIVOTS = [4, 5, 8, 10, 14]
random.seed(101)


def clustered_ci(pairs, n_boot=2000):
    by = defaultdict(list)
    for d, v in pairs:
        by[d].append(v)
    days = list(by)
    if len(days) < 4:
        return None
    m = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        vals = [x for d in pick for x in by[d]]
        if vals:
            m.append(st.mean(vals))
    m.sort()
    return m[int(.025 * len(m))], m[int(.975 * len(m))]


def run_for(pivot: int):
    os.environ["PIVOT"] = str(pivot)
    import sweep_core
    importlib.reload(sweep_core)
    SC = sweep_core
    assert SC.PIVOT == pivot, f"reload failed: {SC.PIVOT} != {pivot}"
    allev, vb = [], []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        for fill_ts, _x, R, _l, _A, _s, pierce, _sd in SC.backtest_symbol(bars):
            rec = {"ts": int(fill_ts), "R": float(R), "sym": sym}
            allev.append(rec)
            if pierce <= 0.25:
                vb.append(rec)
    return allev, vb


def summarise(ev):
    if not ev:
        return None
    m = st.mean(x["R"] for x in ev)
    ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in ev])
    per = defaultdict(list)
    for x in ev:
        if x["sym"] in CORE9:
            per[x["sym"]].append(x["R"])
    br = sum(1 for s in per if st.mean(per[s]) > 0)
    return {"n": len(ev), "meanR": round(m, 4),
            "wr": round(100 * sum(1 for x in ev if x["R"] > 0) / len(ev), 1),
            "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "breadth": f"{br}/{len(per)}"}


def main() -> int:
    print("§0.73 PIVOT 粒度 —— 問的是「同樣的 edge 能不能拿到更多樣本」\n")
    print("  參考指標用 len_l=4，凍結規則用 PIVOT=10。")
    print("  sweep_core 檔頭已記載 PIVOT∈{5,10} 皆為正 —— 5 不是新主張。")
    print("  **判準不是哪個分數最高**，是事件變多而 meanR 不垮。\n")
    res = {}
    print(f"{'PIVOT':>6} {'全部事件':>9} {'變體B':>8} {'B meanR':>9} "
          f"{'B 勝率':>7} {'B 日聚類CI':>20} {'B 廣度':>7}")
    for pv in PIVOTS:
        allev, vb = run_for(pv)
        s_all, s_vb = summarise(allev), summarise(vb)
        res[pv] = {"all": s_all, "variant_b": s_vb}
        mark = "  ← 現行" if pv == 10 else ("  ← TV 指標" if pv == 4 else "")
        print(f"{pv:>6} {s_all['n']:>9} {s_vb['n']:>8} "
              f"{s_vb['meanR']:>+9.4f} {s_vb['wr']:>6.1f}% "
              f"{str(s_vb['ci']):>20} {s_vb['breadth']:>7}{mark}")

    base = res[10]["variant_b"]
    print(f"\n── 對照現行 PIVOT=10（事件 {base['n']}、meanR {base['meanR']:+.4f}）──")
    verdict_rows = []
    for pv in PIVOTS:
        if pv == 10:
            continue
        v = res[pv]["variant_b"]
        n_mult = v["n"] / base["n"]
        r_keep = v["meanR"] / base["meanR"] if base["meanR"] else 0
        br = int(v["breadth"].split("/")[0])
        ok = n_mult > 1.2 and r_keep >= 0.8 and br >= 6
        print(f"   PIVOT={pv:<3} 事件 ×{n_mult:.2f}  meanR 保留 {100*r_keep:.0f}%"
              f"  廣度 {v['breadth']}  {'✓ 買到樣本沒付代價' if ok else ''}")
        verdict_rows.append((pv, n_mult, r_keep, ok))

    finer = [r for r in verdict_rows if r[0] < 10]
    if finer and all(r[2] >= 0.8 for r in finer) and any(r[3] for r in finer):
        v = ("**細 pivot 買到樣本而 meanR 沒垮** —— 值得預註冊為取樣率變更"
             "（不是 edge 變更）。判決仍需前瞻樣本。")
    elif finer and all(r[2] < 0.8 for r in finer):
        v = ("**粗糙本身就是濾網** —— PIVOT=10 是承重的，細化會稀釋 edge。"
             "樣本稀少是這個 edge 的代價，不是缺陷。")
    else:
        v = "混合結果，逐列判讀（見上表），不得只取有利的那一格。"
    print(f"\n判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
