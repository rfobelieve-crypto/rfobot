# -*- coding: utf-8 -*-
"""Do tonight's findings hold on the 79% that was never tested? — §0.71c.

A scope limit surfaced when the operator asked "我的是 pivot 嗎". Only ONE
of the four pool families is pivot-based, and every analysis tonight ran on
`SC.backtest_symbol`, which walks `detect_sweeps` — swing only. The live
shadow log trades all four:

    B:swing    239  (21%)   <- everything tonight
    B:session  585  (51%)
    B:pdh_pdl  273  (24%)
    B:pwh_pwl   49  ( 4%)

So the night's work covered a fifth of what actually trades. That is not
wrong, but it is a limit that has to be stated and then removed.

This file rebuilds the population the way the recorder does — via
`level_types.trade_levels` over all four families — and re-tests the one
candidate that survived: §0.71b, confluence (how many OTHER families sit
on the same price).

WHY THIS IS A REAL TEST AND NOT A REPEAT: confluence was measured on swing
sweeps only. The time-defined families are a different animal — a session
high is fixed the moment the session closes, a PDH the moment the day
rolls. If "obvious levels pay less" is a property of liquidity rather than
of pivots, it should appear in all four. If it appears only in swing, it
was a property of the pivot construction and the finding shrinks to 21%
of the book.

Pre-committed reading:
  * the confluence gap holds in the pooled four-family population AND in
    at least 3 of the 4 families individually -> a property of liquidity;
    the candidate covers the whole book
  * it holds only in swing -> a pivot artefact; the candidate is real but
    applies to a fifth of trades and its value drops accordingly
  * it disappears pooled -> swing was the exception; candidate dies

Every family is reported whether or not it helps the story.
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

import heapq                                            # noqa: E402
from bisect import bisect_left, bisect_right           # noqa: E402

import sweep_core as SC                                    # noqa: E402
import level_types as LT                                   # noqa: E402
from research.liquidity_map_check import swing_levels, first_hit  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "confluence_all_families.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
PIERCE_B = 0.25
TOL = 0.10                      # the middle of the three agreeing tolerances
random.seed(107)


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


def first_hits_batch(bars, levels):
    """first_hit for EVERY level in one pass — O(n log m), not O(n x m).

    The per-level scan (liquidity_map_check.first_hit) is fine for a chart
    with a handful of levels; here there are thousands per family per coin
    and two runs were killed inside it. Same definition, different
    algorithm: walk the bars once, and at each bar retire every pending
    level the bar has now traded through. Buy-side levels are retired in
    ascending price (a rising high takes the lowest ones first); sell-side
    in descending.
    """
    n = len(bars)
    out = [None] * len(levels)
    buy, sell = [], []          # heaps of (key, price, idx), pending
    by_est = defaultdict(list)
    for i, (est, price, side) in enumerate(levels):
        # Enter the pending set at est+1, not est. The reference scans
        # range(est+1, n), so the establishment bar itself never counts. The
        # first version pushed at est, and any level the establishment bar
        # already traded through was popped there, failed the `j > est`
        # guard, and vanished from the heap unmarked — silently "never
        # swept" forever. 90 of 600 sampled levels disagreed with the
        # reference; the shape (ref=est+1, fast=None) named the bug.
        if est + 1 < n:
            by_est[est + 1].append(i)
    for j in range(n):
        for i in by_est.get(j, ()):
            est, price, side = levels[i]
            if side == 1:
                heapq.heappush(buy, (price, i))
            else:
                heapq.heappush(sell, (-price, i))
        hi, lo_ = bars[j][SC.H], bars[j][SC.L]
        while buy and buy[0][0] < hi:
            _, i = heapq.heappop(buy)
            if j > levels[i][0]:            # strictly after establishment
                out[i] = j
        while sell and -sell[0][0] > lo_:
            _, i = heapq.heappop(sell)
            if j > levels[i][0]:
                out[i] = j
    return out


def collect(only=None):
    """All four families, matching what shadow_engine actually records.

    The first version scanned every other family's full level list for
    every fill — O(fills x families x levels) — and was killed before it
    finished. Levels are now sorted by price once per family and the
    candidate window is found by binary search, so each fill touches only
    the handful of levels that could possibly be within tolerance.
    """
    rows = []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        if only and sym not in only:
            continue
        bars = SC.load_csv(str(fp))
        a = SC.atr14(bars)
        idx = {b[0]: i for i, b in enumerate(bars)}
        fam = {k: list(v) for k, v in LT.build_levels(bars).items()}
        fam["swing"] = swing_levels(bars)
        # every family's live windows, PRICE-SORTED for binary search
        live, own_hits = {}, {}
        for k, items in fam.items():
            hits = first_hits_batch(bars, items)
            arr = sorted((p, est, s, h)
                         for (est, p, s), h in zip(items, hits))
            live[k] = (arr, [x[0] for x in arr])
            # the level's own sweep bar, needed to count AT the sweep
            byp = defaultdict(list)
            for (est, pr, sd2), hh in zip(items, hits):
                if hh is not None:
                    byp[round(pr, 8)].append(hh)
            own_hits[k] = byp

        for kind, items in fam.items():
            for tr in LT.trade_levels(bars, items):
                fill_ts, exit_ts, R, pierce, lvl, A, stopped, side = tr
                if pierce > PIERCE_B:
                    continue
                fi = idx.get(fill_ts)
                if fi is None or not A or A <= 0:
                    continue
                d = side if isinstance(side, int) else (
                    1 if str(side).upper() == "LONG" else -1)
                want = 1 if d == -1 else -1
                # COUNT AT THE SWEEP BAR, not the fill bar. The sweep itself
                # takes out every co-located level, so at the fill bar their
                # hit is already in the past and a `hit >= fill` test removes
                # exactly the levels confluence is meant to count. That
                # inverted the whole measure: 158 events with 2+ became 5.
                # The sweep bar is the level's own first_hit, inside
                # [fill-W, fill-1].
                cand = [hh for hh in own_hits[kind].get(round(lvl, 8), [])
                        if fi - SC.W <= hh < fi]
                if not cand:
                    continue
                jsw = max(cand)
                tol = TOL * A
                conf = 0
                for k2, (arr, keys) in live.items():
                    if k2 == kind:
                        continue
                    a0 = bisect_left(keys, lvl - tol)
                    a1 = bisect_right(keys, lvl + tol)
                    for p2, est2, s2, h2 in arr[a0:a1]:
                        if (s2 == want and est2 <= jsw
                                and (h2 is None or h2 >= jsw)):
                            conf += 1
                            break
                rows.append({"ts": int(fill_ts), "R": float(R), "sym": sym,
                             "kind": kind, "conf": conf})
    return rows


def arm(v, lab):
    if len(v) < 60:
        return {"label": lab, "n": len(v), "established": False}
    m = st.mean(x["R"] for x in v)
    ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in v])
    per = defaultdict(list)
    for x in v:
        if x["sym"] in CORE9:
            per[x["sym"]].append(x["R"])
    br = sum(1 for s in per if st.mean(per[s]) > 0)
    return {"label": lab, "n": len(v), "meanR": round(m, 4),
            "wr": round(100 * sum(1 for x in v if x["R"] > 0) / len(v), 1),
            "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "breadth": f"{br}/{len(per)}",
            "established": bool(len(v) >= 200 and ci and ci[0] > 0 and br >= 6)}


def show(a, ind="   "):
    if not a["n"]:
        print(f"{ind}{a['label']:<22} 無樣本")
        return
    ci = f"[{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}]" if a.get("ci") else "—"
    print(f"{ind}{a['label']:<22} n={a['n']:<6} meanR {a.get('meanR',0):+.4f}  "
          f"WR {a.get('wr',0):5.1f}%  CI {ci:<20} 廣度 {a.get('breadth','—'):<6}"
          f"{'  ✓成立' if a['established'] else '  ·未成立'}")


def main() -> int:
    # core9 first: it is the evidence base for §0.58/§0.59, and the full
    # 29-coin run can follow once the shape is known.
    only = None if "--all" in sys.argv else CORE9
    rows = collect(only)
    print("§0.71c 堆疊在**四種池全部**上還成立嗎"
          " —— 今晚之前只測過 swing（21%）\n")
    tot = len(rows)
    print(f"  母體 n={tot}（shadow log 的真正口徑）")
    for k in ("swing", "session", "pdh_pdl", "pwh_pwl"):
        v = [x for x in rows if x["kind"] == k]
        print(f"    {k:<10} {len(v):6d}  ({100*len(v)/tot:.0f}%)")
    print()

    print("── 合併四種池 ──")
    res = {"pooled": {}}
    lo = arm([x for x in rows if x["conf"] <= 1], "堆疊 ≤1 種")
    hi = arm([x for x in rows if x["conf"] >= 2], "堆疊 ≥2 種")
    allr = arm(rows, "全部")
    for a in (allr, lo, hi):
        show(a)
        res["pooled"][a["label"]] = a
    if lo["established"] and hi["established"]:
        gap = lo["meanR"] - hi["meanR"]
        print(f"   → ≤1 − ≥2 = **{gap:+.4f}R**")
        res["pooled_gap"] = round(gap, 4)
    else:
        gap = None
        print("   → 有一臂未成立，合併層不下結論")

    print("\n── 逐家族（全部報告，不論是否有利）──")
    fam_ok = 0
    res["by_family"] = {}
    for k in ("swing", "session", "pdh_pdl", "pwh_pwl"):
        sub = [x for x in rows if x["kind"] == k]
        l2 = arm([x for x in sub if x["conf"] <= 1], f"{k} · 堆疊≤1")
        h2 = arm([x for x in sub if x["conf"] >= 2], f"{k} · 堆疊≥2")
        show(l2)
        show(h2)
        if l2["established"] and h2["established"]:
            g = l2["meanR"] - h2["meanR"]
            ok = g > 0.03
            fam_ok += ok
            print(f"      → 差 {g:+.4f}R {'✓' if ok else ''}")
            res["by_family"][k] = {"gap": round(g, 4), "ok": ok}
        else:
            print("      → 有一臂未成立，不參與計數")
            res["by_family"][k] = {"gap": None, "ok": False}
        print()

    if gap is not None and gap > 0.03 and fam_ok >= 3:
        v = ("**流動性的性質，不是 pivot 的產物** —— 合併成立且 "
             f"{fam_ok}/4 個家族各自成立。候選涵蓋整本帳。")
    elif gap is not None and gap > 0.03:
        v = (f"合併成立但只有 {fam_ok}/4 個家族各自成立 —— "
             "候選為真但強度依家族而異，預註冊要逐家族報告")
    elif gap is not None:
        v = (f"合併只差 {gap:+.4f}R —— **swing 是例外**，"
             "§0.71b 的發現不能外推到另外 79%")
    else:
        v = "合併層有臂未成立，判不出來"
    print(f"判讀：{v}")
    res["verdict"] = v
    res["families_passing"] = fam_ok
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
