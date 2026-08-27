# -*- coding: utf-8 -*-
"""Is liquidity a LINE or a ZONE? — TODO §0.71.

The operator, after looking at yesterday's map: "這是個事件 / 我覺得是流動性
獵取的區域位置沒有處理好 / 昨天看圖表就覺得怪怪的".

This is the difference I dismissed. The reference indicator they supplied
draws a BAND (LSH .. LSH*thresh); the frozen rule draws a LINE and calls a
sweep on `h[j] > h[i]` — a single tick through the extreme counts. If
resting stops actually sit stacked ABOVE the extreme rather than exactly
on it, then a one-tick penetration has taken nothing, and the event that
the whole strategy is built on has been mis-defined at its root.

Note what variant B already implies: `pierce_atr <= 0.25` is a zone
constraint that only has a CEILING. Nothing stops an event that barely
grazed the level from entering the population.

THE QUESTION IS SHAPE, NOT THRESHOLD. Reporting the best bucket and
adopting its edge would be the 2026-06-20 sweep trap. What is being asked
is which of two shapes the data has:

  MONOTONE   shallower is always better -> a line is the right model, the
             existing ceiling-only filter is correct, nothing to change
  HUMPED     an interior band is best, and very shallow pierces are WORSE
             -> liquidity is a zone, the event needs a FLOOR as well, and
             the operator's read is right

Buckets are fixed here before running, spaced to give every one a usable
count rather than chosen to make a shape appear. All are reported.

Secondary check, same idea from the other side: does a very shallow pierce
also fail to REACH the stops? If shallow events show a lower stop-out rate
AND a lower win rate, they are simply weaker events rather than safer ones.
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
OUT = ROOT / "research" / "results" / "pierce_zone_shape.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
random.seed(89)

# frozen before the run — spacing chosen for sample balance, not for shape
EDGES = [0.0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.40, 0.70, 1.50, 99.0]


def label(i):
    a, b = EDGES[i], EDGES[i + 1]
    return f"{a:.2f}–{b:.2f}" if b < 90 else f"{a:.2f}+"


def clustered_ci(pairs, n_boot=3000):
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


def main() -> int:
    rows = []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        for fill_ts, _x, R, _l, _A, stopped, pierce, _s in \
                SC.backtest_symbol(bars):
            rows.append({"ts": int(fill_ts), "R": float(R),
                         "p": float(pierce), "sym": sym,
                         "stopped": bool(stopped)})
    print("§0.71 流動性是線還是區間 —— 問的是形狀不是門檻\n")
    print(f"  母體：所有掃單失敗成交 n={len(rows)}"
          f"（**不套變體 B 的 0.25 上限**，否則看不到上半段）\n")

    print(f"{'穿越深度(ATR)':<14} {'n':>6} {'meanR':>9} {'勝率':>7} "
          f"{'停損率':>7} {'日聚類CI':>20} {'廣度':>7}")
    res = {}
    for i in range(len(EDGES) - 1):
        v = [r for r in rows if EDGES[i] <= r["p"] < EDGES[i + 1]]
        if len(v) < 40:
            print(f"{label(i):<14} {len(v):6d}   （樣本不足，仍列出）")
            continue
        m = st.mean(r["R"] for r in v)
        wr = 100 * sum(1 for r in v if r["R"] > 0) / len(v)
        so = 100 * sum(1 for r in v if r["stopped"]) / len(v)
        ci = clustered_ci([(r["ts"] // 86400, r["R"]) for r in v])
        per = defaultdict(list)
        for r in v:
            if r["sym"] in CORE9:
                per[r["sym"]].append(r["R"])
        br = sum(1 for s in per if st.mean(per[s]) > 0)
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"{label(i):<14} {len(v):6d} {m:+9.4f} {wr:6.1f}% {so:6.1f}% "
              f"{cis:>20} {br:3d}/{len(per):<3d}")
        res[label(i)] = {"n": len(v), "meanR": round(m, 4),
                         "wr": round(wr, 1), "stop_rate": round(so, 1),
                         "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                         "breadth": f"{br}/{len(per)}",
                         "established": bool(len(v) >= 200 and ci
                                             and ci[0] > 0 and br >= 6)}

    ok = [k for k, r in res.items() if r["established"]]
    print(f"\n  成立的桶（n≥200 ∧ CI 離零 ∧ 廣度≥6）：{ok or '無'}")
    if len(ok) >= 3:
        ms = [res[k]["meanR"] for k in ok]
        first_best = ms.index(max(ms)) == 0
        mono = all(ms[i] >= ms[i + 1] for i in range(len(ms) - 1))
        if mono and first_best:
            v = ("**單調**：越淺越好，一路遞減 —— **線**是對的模型，"
                 "現行「只有上限」的濾網正確，區間下限不需要。")
        elif not first_best:
            top = ok[ms.index(max(ms))]
            v = (f"**駝峰**：最好的是中段的「{top}」而不是最淺的那桶 —— "
                 "流動性是**區間**，事件需要一個**下限**。"
                 "使用者對圖表的直覺成立。")
        else:
            v = "**非單調但最淺仍最好** —— 形狀不乾淨，列觀察不列結論。"
    else:
        v = f"成立的桶只有 {len(ok)} 個，形狀判不出來"
    print(f"\n判讀：{v}")

    # secondary: are the shallowest events simply weaker events?
    sh = [r for r in rows if r["p"] < 0.03]
    md = [r for r in rows if 0.07 <= r["p"] < 0.25]
    if len(sh) >= 100 and len(md) >= 100:
        print(f"\n  次要檢查 —— 最淺的是不是「根本沒碰到停損堆」？")
        print(f"    穿越<0.03  n={len(sh):5d}  停損率 "
              f"{100*sum(1 for r in sh if r['stopped'])/len(sh):.1f}%"
              f"  勝率 {100*sum(1 for r in sh if r['R']>0)/len(sh):.1f}%"
              f"  meanR {st.mean(r['R'] for r in sh):+.4f}")
        print(f"    0.07–0.25  n={len(md):5d}  停損率 "
              f"{100*sum(1 for r in md if r['stopped'])/len(md):.1f}%"
              f"  勝率 {100*sum(1 for r in md if r['R']>0)/len(md):.1f}%"
              f"  meanR {st.mean(r['R'] for r in md):+.4f}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
