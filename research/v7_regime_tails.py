# -*- coding: utf-8 -*-
"""Right-tail test for §0.60 Q2 — does filtering TREND_UP x UP cut winners?

Q2 proposed: in TREND_UP, accept only DOWN signals. Full history backs it
(that cell is the only one negative on BOTH win rate and bps: 48.1%,
mean -8.0, median -15.1). But the first new sample after the hypothesis was
formed contradicted it — 11 blocked signals returned 55% WR / +23.5 bps,
including +153.8 / +107.0 / +67.9 bps from the 08-19~21 run.

Three candidate explanations were pre-listed; this file settles #2:

  #2 THE CELL HAS A FAT RIGHT TAIL. Its mean (-8.0) sits ABOVE its median
     (-15.1), which means a few large winners are holding it up. If so, the
     win-rate advantage of filtering is illusory: V7's trade layer earns
     through trailing stops letting winners run (§0.51b showed the exit
     side's positive contribution comes from exactly that), so cutting the
     right tail would hurt the thing that pays.

Tests, all pre-stated:
  T1 tail share      what fraction of each cell's TOTAL positive bps comes
                     from its top 10% of signals, vs the other cells
  T2 filter impact   total bps with and without the cell, and how much of
                     the loss is tail vs body
  T3 trimmed compare the cell's mean after removing the top 10% both sides
                     — if it stays clearly worst, the tail is not what
                     makes it look bad, and Q2 survives this test
  T4 skew           mean-minus-median per cell, ranked

Reading committed before running:
  - if the cell's tail share is NOT unusually high and T3 keeps it worst,
    explanation #2 is dead and Q2 survives to face #1 and #3
  - if the tail share IS high and T3 pulls it toward the pack, Q2 dies here
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.v7_regime_axis import load                   # noqa: E402

OUT = ROOT / "research" / "results" / "v7_regime_tails.json"
CELLS = ("RANGING", "TREND_UP", "TREND_DOWN", "NEUTRAL")
TARGET = ("TREND_UP", "UP")


def cell_rows(rows, cell, d):
    return [r for r in rows if r["cell"] == cell and r["dir"] == d
            and r["bps"] is not None]


def main() -> int:
    rows = load()
    groups = {}
    for cell in CELLS:
        for d in ("UP", "DOWN"):
            v = cell_rows(rows, cell, d)
            if len(v) >= 30:
                groups[f"{cell} × {d}"] = [r["bps"] for r in v]
    key = f"{TARGET[0]} × {TARGET[1]}"

    print("§0.60 Q2 右尾檢定 —— 濾掉 TREND_UP×UP 會不會砍到贏家\n")

    # T1 tail share
    print("── T1  前 10% 訊號貢獻了該格多少正報酬 ──")
    print(f"{'格':<22} {'n':>4} {'平均':>8} {'中位':>8} "
          f"{'前10%佔正報酬':>13} {'正報酬筆佔比':>12}")
    t1 = {}
    for g, v in sorted(groups.items()):
        vs = sorted(v, reverse=True)
        k = max(1, len(vs) // 10)
        pos_total = sum(x for x in vs if x > 0)
        top_total = sum(x for x in vs[:k] if x > 0)
        share = 100 * top_total / pos_total if pos_total else 0
        t1[g] = {"n": len(v), "mean": st.mean(v), "median": st.median(v),
                 "tail_share": share,
                 "pos_frac": 100 * sum(1 for x in v if x > 0) / len(v)}
        mark = "  ←" if g == key else ""
        print(f"{g:<22} {len(v):4d} {st.mean(v):+8.1f} {st.median(v):+8.1f} "
              f"{share:12.1f}% {t1[g]['pos_frac']:11.1f}%{mark}")
    others = [t1[g]["tail_share"] for g in t1 if g != key]
    print(f"\n  該格尾部佔比 {t1[key]['tail_share']:.1f}%  vs 其他格中位 "
          f"{st.median(others):.1f}%  → "
          f"{'異常高' if t1[key]['tail_share'] > st.median(others) + 10 else '不異常'}")

    # T2 filter impact on totals
    print("\n── T2  濾掉該格對總報酬的影響（全歷史，訊號層 bps 合計）──")
    allv = [r["bps"] for r in rows if r["bps"] is not None]
    tgt = groups[key]
    kept = [x for x in allv if True]
    total_all = sum(allv)
    total_wo = total_all - sum(tgt)
    tail_k = max(1, len(sorted(tgt, reverse=True)) // 10)
    tail_sum = sum(sorted(tgt, reverse=True)[:tail_k])
    body_sum = sum(tgt) - tail_sum
    print(f"  全部訊號        n={len(allv):<5} 合計 {total_all:+9.0f} bps  "
          f"平均 {total_all/len(allv):+6.1f}")
    print(f"  濾掉該格後      n={len(allv)-len(tgt):<5} 合計 {total_wo:+9.0f} bps  "
          f"平均 {total_wo/(len(allv)-len(tgt)):+6.1f}")
    print(f"  被濾掉的        n={len(tgt):<5} 合計 {sum(tgt):+9.0f} bps  "
          f"= 尾部 {tail_sum:+.0f} + 本體 {body_sum:+.0f}")
    print(f"  → 平均改善 {total_wo/(len(allv)-len(tgt)) - total_all/len(allv):+.1f} bps/訊號")

    # T3 trimmed means
    print("\n── T3  去掉兩端各 10% 後的均值（尾部拿掉還是不是最差）──")
    print(f"{'格':<22} {'原均值':>9} {'截尾均值':>10} {'排名變化':>10}")
    raw_rank = sorted(groups, key=lambda g: st.mean(groups[g]))
    trim = {}
    for g, v in groups.items():
        vs = sorted(v)
        k = max(1, len(vs) // 10)
        trim[g] = st.mean(vs[k:-k]) if len(vs) > 2 * k else st.mean(vs)
    trim_rank = sorted(groups, key=lambda g: trim[g])
    t3 = {}
    for g in sorted(groups, key=lambda x: st.mean(groups[x])):
        r1, r2 = raw_rank.index(g) + 1, trim_rank.index(g) + 1
        t3[g] = {"raw": st.mean(groups[g]), "trimmed": trim[g],
                 "rank_raw": r1, "rank_trim": r2}
        mark = "  ←" if g == key else ""
        print(f"{g:<22} {st.mean(groups[g]):+9.1f} {trim[g]:+10.1f} "
              f"{r1:>4} → {r2:<4}{mark}")

    # T4 skew
    print("\n── T4  偏度（平均 − 中位，正值＝右尾撐著）──")
    sk = sorted(((t1[g]["mean"] - t1[g]["median"], g) for g in t1), reverse=True)
    for v, g in sk:
        mark = "  ←" if g == key else ""
        print(f"  {g:<22} {v:+8.1f}{mark}")

    # verdict per the pre-committed reading
    tail_unusual = t1[key]["tail_share"] > st.median(others) + 10
    still_worst = trim_rank.index(key) == 0
    print()
    if not tail_unusual and still_worst:
        v = ("右尾解釋 #2 死亡 —— 該格尾部不異常，且去尾後仍是最差。"
             "Q2 存活，進入 #1（趨勢強度分層）與 #3（運氣）的檢驗")
    elif tail_unusual and not still_worst:
        v = ("右尾解釋 #2 成立 —— 該格靠少數大贏撐著，去尾後不再最差。"
             "Q2 死於此：濾掉它等於砍右尾，而 V7 交易層靠讓 winner 跑")
    else:
        v = (f"混合：尾部{'異常' if tail_unusual else '不異常'}、"
             f"去尾後{'仍最差' if still_worst else '不再最差'}，"
             "需人工判讀（見上表）")
    print(f"判讀：{v}")
    OUT.write_text(json.dumps(
        {"t1": t1, "t3": t3, "verdict": v}, indent=1), encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
