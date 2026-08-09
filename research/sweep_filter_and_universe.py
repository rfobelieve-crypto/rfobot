"""Two questions the 2026-08-09 registry surfaced, answered on forward data.

Q1 IS THE PIERCE FILTER EARNING ITS KEEP?
    Gate F's official track is variant A (no filter): clustered CI-low
    +0.063, 7/9 symbols positive. Variant B (A + pierce <= 0.25 ATR) sits at
    CI-low -0.016. So the filter may be costing, not adding. Tested PAIRED
    the same way the exit study is: B is a strict subset of A, so compare
    B's trades against the A-trades it *rejected*, plus the pooled effect of
    applying the filter at all. A filter that adds nothing should be removed —
    fewer parameters is fewer overfitting surfaces, not a cosmetic win.

Q2 WHICH SYMBOLS SHOULD SURVIVE?
    Operator constraint (2026-08-09): 29 symbols firing ~38 trades/day
    against a concurrency cap of 5-10 means capital cannot cover the
    simultaneous signals — the executor would drop trades arbitrarily,
    which is a silent, unmeasured selection.  Better to choose the basket
    ON PURPOSE, from criteria fixed before looking:

      C1 capacity   median risk-$ per trade at 1% participation >= $50
                    (from sweep_capacity_estimate: ATR% x hourly volume).
                    Below this the coin cannot absorb meaningful size, so
                    its signals are decoration.
      C2 sample     >= 8 genuinely-prospective closed trades (else unrankable)
      C3 NOT performance-ranked. Picking the best-performing coins on the
         same data that measures them is the selection trap this repo has
         been burned by (08-07: V∧LIQ 100% -> 0%). Performance is REPORTED
         for the chosen basket, never used to choose it.

    The output is a candidate basket + the honest counterfactual: what the
    forward record would have looked like on that basket.

Usage: python research/sweep_filter_and_universe.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
VOLS = ROOT / "research" / "results" / "sweep_capacity_volumes.json"
MIN_CAPACITY_USD = 50.0      # C1, at 1% participation
MIN_TRADES = 8               # C2
RNG = np.random.default_rng(20260809)


def load():
    rows = []
    for r in csv.DictReader(LOG.open(encoding="utf-8")):
        if r.get("status") != "CLOSED" or r.get("net_r") in (None, ""):
            continue
        if not (r.get("first_seen_utc") and r.get("exit_utc")
                and r["first_seen_utc"] < r["exit_utc"]):
            continue                      # genuinely prospective only
        try:
            rows.append({"sym": r["symbol"], "day": r["fill_utc"][:10],
                         "r": float(r["net_r"]),
                         "b": r.get("variant_b") == "1",
                         "atr_pct": float(r["atr"]) / float(r["entry_px"])})
        except (ValueError, ZeroDivisionError, KeyError):
            continue
    return rows


def cluster_ci(rows, key=lambda x: x["r"], iters=20000):
    """Day-clustered bootstrap — same arithmetic the gate uses."""
    by = defaultdict(list)
    for x in rows:
        by[x["day"]].append(key(x))
    days = list(by)
    if len(days) < 3:
        return None, None
    means = []
    for _ in range(iters):
        pick = RNG.integers(0, len(days), len(days))
        vals = [v for i in pick for v in by[days[i]]]
        means.append(float(np.mean(vals)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def line(tag, rows):
    if not rows:
        print(f"  {tag:<26} (無樣本)")
        return
    r = np.array([x["r"] for x in rows])
    lo, hi = cluster_ci(rows)
    syms = {s for s in (x["sym"] for x in rows)}
    pos = sum(1 for s in syms
              if sum(x["r"] for x in rows if x["sym"] == s) > 0)
    ci = f"[{lo:+.4f},{hi:+.4f}]" if lo is not None else "—"
    print(f"  {tag:<26} n={len(r):>4}  meanR={r.mean():+.4f}  WR={100*(r>0).mean():>3.0f}%"
          f"  CI{ci}  幣正 {pos}/{len(syms)}")


def main() -> int:
    rows = load()
    print(f"真前瞻已平倉 {len(rows)} 筆 · {len({x['sym'] for x in rows})} 幣\n")

    print("═" * 74)
    print("  Q1 淺穿越濾網有沒有用（B = A ∧ pierce≤0.25ATR）")
    print("═" * 74)
    line("A 全部（無濾網）", rows)
    line("B 濾網留下的", [x for x in rows if x["b"]])
    line("濾網丟掉的", [x for x in rows if not x["b"]])
    kept = np.array([x["r"] for x in rows if x["b"]])
    drop = np.array([x["r"] for x in rows if not x["b"]])
    if len(kept) and len(drop):
        print(f"\n  留下 − 丟掉 = {kept.mean() - drop.mean():+.4f}R")
        print("  濾網要有用，被丟掉的那組必須明顯較差；若丟掉的反而較好，"
              "濾網就是在扔錢。")

    print("\n" + "═" * 74)
    print("  Q2 幣種精簡（並發塞不下 → 主動選，不要讓 executor 亂丟）")
    print("═" * 74)
    vols = json.loads(VOLS.read_text()) if VOLS.exists() else {}
    per = defaultdict(list)
    for x in rows:
        per[x["sym"]].append(x)
    table = []
    for s, xs in per.items():
        v = vols.get(s)
        atr = float(np.median([x["atr_pct"] for x in xs]))
        cap = v["median"] * 0.01 * atr if v else None
        rr = np.array([x["r"] for x in xs])
        table.append({"sym": s, "n": len(xs), "cap": cap, "atr": atr,
                      "mean": float(rr.mean()), "sum": float(rr.sum())})
    table.sort(key=lambda t: -(t["cap"] or 0))
    print(f"  {'幣':<7}{'筆數':>5}{'容量$/筆':>11}{'ATR%':>7}"
          f"{'meanR':>9}{'ΣR':>8}  判定")
    keep = []
    for t in table:
        ok_cap = t["cap"] is not None and t["cap"] >= MIN_CAPACITY_USD
        ok_n = t["n"] >= MIN_TRADES
        verdict = "保留" if (ok_cap and ok_n) else (
            "剔除·容量" if not ok_cap else "剔除·樣本")
        if ok_cap and ok_n:
            keep.append(t["sym"])
        capstr = f"${t['cap']:,.0f}" if t["cap"] is not None else "—"
        print(f"  {t['sym']:<7}{t['n']:>5}{capstr:>11}{t['atr']*100:>6.2f}%"
              f"{t['mean']:>+9.3f}{t['sum']:>+8.2f}  {verdict}")

    print(f"\n  → 建議籃子 {len(keep)} 幣：{' '.join(sorted(keep))}")
    sub = [x for x in rows if x["sym"] in keep]
    print("\n  精簡後的前瞻紀錄（純為誠實揭露，未用績效挑幣）：")
    line("  精簡籃 · 無濾網(A)", sub)
    line("  精簡籃 · 濾網後(B)", [x for x in sub if x["b"]])
    print("\n  ⚠ C3：籃子只用『容量 + 樣本量』選，**不曾用績效排序**。"
          "上面的績效是選完之後才報的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
