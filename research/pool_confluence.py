# -*- coding: utf-8 -*-
"""Does stacked liquidity behave differently? — TODO §0.71b.

The operator: "這是個事件 / 我覺得是流動性獵取的區域位置沒有處理好 /
昨天看圖表就覺得怪怪的".

First reading was pierce DEPTH (§0.71) — tested, monotone, no floor
needed, a line is fine for that axis. This is the second reading, and it
is the one yesterday's map actually surfaced:

    76,670.01   PDH/PDL
    76,670.01   session     <- one price, three kinds of pool
    76,670.01   swing

The frozen backtest population is SWING-ONLY (`backtest_symbol` walks
`detect_sweeps`). It cannot see that a swing high it is about to trade is
ALSO yesterday's low and ALSO a session extreme. If liquidity is a zone
rather than a line, a price where several pool families coincide is a
THICKER zone — more resting orders, a different event.

That is what "區域位置" means here, and nothing in the strategy currently
represents it.

METHOD: for every swing sweep in the frozen population, count how many
OTHER pool families (session / PDH-PDL / PWH-PWL) had a live, unswept
level within a tolerance of the swept price at that moment. Bucket by
that count and report every bucket.

TOLERANCE is the one free number, so it is handled by reporting THREE
fixed values (0.05 / 0.10 / 0.20 ATR) side by side rather than choosing
one. A real effect should not appear at exactly one tolerance and vanish
at the neighbours; that pattern is the tell for a fitted number.

Pre-committed reading:
  * confluent sweeps separate consistently across all three tolerances,
    with breadth and CI -> "zone thickness" is real and the event
    definition is genuinely incomplete; the operator's read is right
  * no separation, or separation at only one tolerance -> stacking is
    bookkeeping, not mechanism, and the single-price model is adequate
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
import level_types as LT                                   # noqa: E402
from research.liquidity_map_check import first_hit          # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "pool_confluence.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
TOLS = [0.05, 0.10, 0.20]
random.seed(97)


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


def main() -> int:
    rows = []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        bars = SC.load_csv(str(fp))
        a = SC.atr14(bars)
        idx = {b[0]: i for i, b in enumerate(bars)}
        # the OTHER families, with their live windows
        others = []
        for kind, items in LT.build_levels(bars).items():
            for est, price, side in items:
                others.append((est, price, side,
                               first_hit(bars, est, price, side), kind))
        sw_by_lvl = defaultdict(list)
        for e in SC.detect_sweeps(bars):
            sw_by_lvl[round(float(e["level"]), 8)].append(e["j"])

        for fill_ts, _x, R, lvl, A, _st, pierce, side in \
                SC.backtest_symbol(bars):
            if pierce > 0.25:
                continue
            fi = idx.get(fill_ts)
            if fi is None or not A or A <= 0:
                continue
            cands = [j for j in sw_by_lvl.get(round(float(lvl), 8), [])
                     if j < fi and fi - j <= SC.W]
            if not cands:
                continue
            j = max(cands)
            d = 1 if str(side).upper() == "LONG" else -1
            # a swing high swept upward is buy-side (side=+1 in build_levels)
            want = 1 if d == -1 else -1
            counts = {}
            for tol in TOLS:
                kinds = {k for est, p, s2, hit, k in others
                         if s2 == want and est <= j
                         and (hit is None or hit >= j)
                         and abs(p - lvl) <= tol * A}
                counts[tol] = len(kinds)
            rows.append({"ts": int(fill_ts), "R": float(R), "sym": sym,
                         "c": counts})

    print("§0.71b 同一個價位堆疊了幾種流動性 —— 「區域」的第二種讀法\n")
    print(f"  母體：變體 B 成交 n={len(rows)}")
    print("  計數 = 開掃當下，**其他**池種（session/PDH/PWH）有幾種"
          "落在該價位的容差內\n")

    res = {}
    for tol in TOLS:
        buck = defaultdict(list)
        for r in rows:
            c = r["c"][tol]
            buck["0 無堆疊" if c == 0 else "1 種" if c == 1
                 else "2+ 種"].append(r)
        print(f"── 容差 {tol:.2f} ATR ──")
        print(f"   {'堆疊':<10} {'n':>6} {'meanR':>9} {'勝率':>7} "
              f"{'日聚類CI':>20} {'廣度':>7}")
        tr = {}
        for k in ("0 無堆疊", "1 種", "2+ 種"):
            v = buck.get(k, [])
            if len(v) < 40:
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
            cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
            print(f"   {k:<10} {len(v):6d} {m:+9.4f} {wr:6.1f}% {cis:>20} "
                  f"{br:3d}/{len(per):<3d}")
            tr[k] = {"n": len(v), "meanR": round(m, 4), "wr": round(wr, 1),
                     "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                     "breadth": f"{br}/{len(per)}",
                     "established": bool(len(v) >= 200 and ci and ci[0] > 0
                                         and br >= 6)}
        res[str(tol)] = tr
        if "0 無堆疊" in tr and "2+ 種" in tr:
            print(f"   → 2+ 種 − 無堆疊 = "
                  f"{tr['2+ 種']['meanR'] - tr['0 無堆疊']['meanR']:+.4f}R")
        print()

    gaps = []
    for tol in TOLS:
        t = res[str(tol)]
        if "0 無堆疊" in t and "2+ 種" in t:
            if t["0 無堆疊"]["established"] and t["2+ 種"]["established"]:
                gaps.append(t["2+ 種"]["meanR"] - t["0 無堆疊"]["meanR"])
    print(f"  三個容差下的差（僅計兩臂皆成立者）："
          f"{[round(g, 4) for g in gaps] or '不足'}")
    if len(gaps) == 3 and all(g > 0.03 for g in gaps):
        v = ("**堆疊有效且三個容差一致** —— 流動性厚度是真的，"
             "事件定義確實不完整。使用者的直覺成立。")
    elif len(gaps) == 3 and all(g < -0.03 for g in gaps):
        v = ("**方向相反且一致** —— 堆疊處反而較差。同樣是實質發現，"
             "但與「厚度=更多停損可吃」的先驗相反，需要機制解釋才可用。")
    elif len(gaps) < 3:
        v = "成立的臂不足三個容差，形狀判不出來（不得只取有結果的那一個）"
    else:
        v = ("**無一致分離** —— 堆疊在這個母體上是記帳現象不是機制，"
             "單一價位的模型足夠。")
    print(f"\n判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
