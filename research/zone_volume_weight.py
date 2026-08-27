# -*- coding: utf-8 -*-
"""Does a pool's ACCUMULATED VOLUME matter? — TODO §0.74.

The operator supplied LuxAlgo "Liquidity Swings". Its pivot length (14) is
past this system's measured peak (§0.73: PIVOT=10 scores +0.0754, 14 scores
+0.0658 at 7/9 breadth), so that part carries nothing. The part that does:

    count += low[length] < top and high[length] > btm ? 1 : 0
    vol   += low[length] < top and high[length] > btm ? volume[length] : 0

It gives every pool a WEIGHT — how much has traded inside the zone since
it formed. flow_system's pools are binary: they exist, then they are
swept. Nothing anywhere in this repo measures how much business has been
done at a level.

And its ZONE is mechanically specific rather than arbitrary: for a pivot
high, top = the bar's HIGH and bottom = max(close, open) — i.e. the UPPER
WICK. The wick is exactly where price went, got rejected, and left stops
behind. That is a real definition of "the area", which is what the
operator has been pointing at since yesterday's map.

TWO OPPOSING PRIORS, which is why this is a question and not a hunch:
  HEAVY IS STRONGER   more volume transacted there = more positions built
                      = more stops resting = sweeping it releases more fuel
  HEAVY IS WORN OUT   a level repeatedly traded through has already had its
                      liquidity consumed; the untouched one is the loaded gun

§0.71b leans to the second: levels where several pool families coincided —
the obvious, widely-watched prices — scored WORSE (+0.061 vs +0.111), as
did dense terrain (D5: >=3 pools ahead 54% vs <=1 pool 62%). If volume
weight follows that pattern, three independent measures agree.

CAUSALITY, checked by construction: the accumulation window runs from the
pivot's CONFIRMATION bar to the bar BEFORE the sweep. Nothing from the
sweep bar or later enters it. (§0.65's G4 died for exactly this; the check
is not optional.)

NORMALISATION: raw volume is not comparable across coins or across time,
so both measures are expressed relative to the same coin's trailing
720-bar median bar-volume. No per-coin fitting.

Pre-committed reading, buckets frozen below before any number is seen:
  * heavy zones score materially better, breadth >=6/9, both halves agree
        -> "loaded" reading; a new feature with a mechanism
  * heavy zones score WORSE with the same consistency
        -> "worn out" reading, and it corroborates §0.71b and D5
  * no separation -> volume weight is decoration; drop it
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

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "zone_volume_weight.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
PIERCE_B = 0.25
VOL_WIN = 720
random.seed(103)
# frozen quantile cuts, not tuned: quartiles of the weight distribution
QCUTS = [0.25, 0.50, 0.75]


def clustered_ci(pairs, n_boot=2500):
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


def collect():
    rows = []
    P = SC.PIVOT
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        n = len(bars)
        h = [b[SC.H] for b in bars]
        lo = [b[SC.L] for b in bars]
        o = [b[SC.O] for b in bars]
        c = [b[SC.C] for b in bars]
        vol = [b[SC.V] for b in bars]
        idx = {b[0]: i for i, b in enumerate(bars)}
        sw_by_lvl = defaultdict(list)
        for e in SC.detect_sweeps(bars):
            sw_by_lvl[round(float(e["level"]), 8)].append(e["j"])

        for fill_ts, _x, R, lvl, A, _s, pierce, side in \
                SC.backtest_symbol(bars):
            if pierce > PIERCE_B:
                continue
            fi = idx.get(fill_ts)
            if fi is None:
                continue
            cands = [j for j in sw_by_lvl.get(round(float(lvl), 8), [])
                     if j < fi and fi - j <= SC.W]
            if not cands:
                continue
            j = max(cands)
            ext = j  # placeholder; the pivot's extreme bar is found below
            # locate the pivot bar whose extreme equals the level
            piv = None
            for k in range(max(P, j - 400), j - P):
                if abs(h[k] - lvl) < 1e-9 or abs(lo[k] - lvl) < 1e-9:
                    piv = k
            if piv is None:
                continue
            is_high = abs(h[piv] - lvl) < 1e-9
            # LuxAlgo "Wick Extremity": the rejected wick of the pivot bar
            if is_high:
                z_top, z_btm = h[piv], max(c[piv], o[piv])
            else:
                z_top, z_btm = min(c[piv], o[piv]), lo[piv]
            if z_top <= z_btm:
                continue
            start = piv + P                       # confirmation bar
            if start >= j:
                continue
            # accumulate strictly BEFORE the sweep bar
            touches = 0
            v_sum = 0.0
            for k in range(start, j):
                if lo[k] < z_top and h[k] > z_btm:
                    touches += 1
                    v_sum += vol[k]
            base = [vol[k] for k in range(max(0, j - VOL_WIN), j) if vol[k] > 0]
            if not base:
                continue
            med = sorted(base)[len(base) // 2]
            if med <= 0:
                continue
            rows.append({"ts": int(fill_ts), "R": float(R), "sym": sym,
                         "touches": touches, "vw": v_sum / med,
                         "age": j - start})
    return rows


def report(rows, key, title):
    vals = sorted(r[key] for r in rows)
    cuts = [vals[int(q * len(vals))] for q in QCUTS]
    def b(x):
        return ("Q1 最輕" if x <= cuts[0] else "Q2" if x <= cuts[1]
                else "Q3" if x <= cuts[2] else "Q4 最重")
    buck = defaultdict(list)
    for r in rows:
        buck[b(r[key])].append(r)
    mid = sorted(r["ts"] for r in rows)[len(rows) // 2]
    print(f"── {title}（四分位切點 "
          f"{cuts[0]:.2f} / {cuts[1]:.2f} / {cuts[2]:.2f}）──")
    print(f"   {'桶':<10} {'n':>6} {'meanR':>9} {'勝率':>7} "
          f"{'日聚類CI':>20} {'廣度':>7} {'前半':>9} {'後半':>9}")
    out = {}
    for k in ("Q1 最輕", "Q2", "Q3", "Q4 最重"):
        v = buck.get(k, [])
        if len(v) < 100:
            print(f"   {k:<10} {len(v):6d}   樣本不足")
            continue
        m = st.mean(x["R"] for x in v)
        wr = 100 * sum(1 for x in v if x["R"] > 0) / len(v)
        ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in v])
        per = defaultdict(list)
        for x in v:
            if x["sym"] in CORE9:
                per[x["sym"]].append(x["R"])
        br = sum(1 for s in per if st.mean(per[s]) > 0)
        h1 = [x["R"] for x in v if x["ts"] < mid]
        h2 = [x["R"] for x in v if x["ts"] >= mid]
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"   {k:<10} {len(v):6d} {m:+9.4f} {wr:6.1f}% {cis:>20} "
              f"{br:3d}/{len(per):<3d} "
              f"{(st.mean(h1) if h1 else float('nan')):+9.4f} "
              f"{(st.mean(h2) if h2 else float('nan')):+9.4f}")
        out[k] = {"n": len(v), "meanR": round(m, 4), "wr": round(wr, 1),
                  "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                  "breadth": f"{br}/{len(per)}",
                  "h1": round(st.mean(h1), 4) if h1 else None,
                  "h2": round(st.mean(h2), 4) if h2 else None,
                  "established": bool(len(v) >= 200 and ci and ci[0] > 0
                                      and br >= 6)}
    ok = [k for k in ("Q1 最輕", "Q2", "Q3", "Q4 最重")
          if k in out and out[k]["established"]]
    print(f"   成立的桶：{ok or '無'}")
    if len(ok) >= 2:
        ms = [out[k]["meanR"] for k in ok]
        print(f"   成立桶最大差：{max(ms) - min(ms):+.4f}R"
              f"（最好 = {ok[ms.index(max(ms))]}）")
    print()
    return out, ok


def main() -> int:
    rows = collect()
    print("§0.74 池的成交量權重 —— LuxAlgo 的核心概念，本系統完全沒有\n")
    print(f"  母體：變體 B 成交 n={len(rows)}｜PIVOT={SC.PIVOT}（凍結值）")
    print("  區域 = pivot 棒的影線（LuxAlgo 'Wick Extremity'）")
    print("  累積窗 = 確認棒 → 掃單棒**前一根**（因果，不含掃單棒本身）\n")

    res = {}
    res["volume"], ok_v = report(rows, "vw", "區域累積成交量／該幣中位棒量")
    res["touches"], ok_t = report(rows, "touches", "區域被觸碰的棒數")

    def verdict(out, ok, name):
        if len(ok) < 2:
            return f"{name}：成立的桶不足兩個，判不出來"
        ms = {k: out[k]["meanR"] for k in ok}
        best = max(ms, key=ms.get)
        gap = max(ms.values()) - min(ms.values())
        if gap < 0.03:
            return f"{name}：成立桶只差 {gap:+.4f}R —— **無分離**"
        if "Q1" in best:
            return (f"{name}：**輕的較好**（差 {gap:+.4f}R）—— 「已被消耗」"
                    "的讀法，與 §0.71b（堆疊越多越差）及地形 D5（池越密越差）"
                    "同向，三個獨立量測一致")
        if "Q4" in best:
            return (f"{name}：**重的較好**（差 {gap:+.4f}R）—— 「上膛」的讀法，"
                    "但與 §0.71b／D5 相反，需要機制解釋才可用")
        return f"{name}：最好的是 {best}，非單調，列觀察"

    v1 = verdict(res["volume"], ok_v, "成交量權重")
    v2 = verdict(res["touches"], ok_t, "觸碰次數")
    print(f"判讀：\n  {v1}\n  {v2}")
    res["verdict_volume"], res["verdict_touches"] = v1, v2
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
