# -*- coding: utf-8 -*-
"""Does the position-count cap cost anything? — TODO §0.62.

Product side reports that signals blocked by the slot cap averaged 4.3x
better than the ones taken, and asks (a) whether Gate F is contaminated,
(b) for an allocation rule ranked by R instead of first-come-first-served.

THE CLAIM IS TESTED, NOT ASSUMED. Two mechanisms produce exactly that
shape with ZERO cap effect, and both are live in this system:

  A. SCORING ASYMMETRY. Blocked signals are scored from the shadow log
     (level fill, no execution cost); taken ones are scored from real
     fills (slippage measured at 0.19R, TODO §0.57 put the structural
     entry gap at 0.1328R). Against a base of ~0.07R that gap alone can
     manufacture a multiple of this size. Apples to oranges.

  B. BUSY-PERIOD CONFOUND. A signal can only be blocked when concurrency
     is already at the cap, so "blocked" is CONDITIONED ON high
     concurrency. If mean R varies with how busy the tape is — for any
     reason unrelated to the cap — the comparison attributes that
     variation to the cap.

This file removes A by construction (both arms scored from the SAME
shadow log, which has no cap and no execution cost) and measures B
directly, so whatever is left is attributable to the cap mechanism
itself: first-come-first-served ordering inside a cluster.

Pre-committed reading, written before the run:
  * |Δ| small and the day-clustered CI of the difference spans zero
        -> the cap is not what produced the product-side 4.3x; the
           finding is A and/or B, and the allocation rule question is
           premature
  * blocked materially better AND mean R is flat across arrival
        concurrency -> real cap cost, ordering inside clusters matters,
        allocation becomes a live research question
  * blocked better BUT mean R rises with arrival concurrency
        -> confounded; report both and do not attribute to the cap

Read-only. Scores no gate. K is reported across a range because the
question is "what does capping cost", not "which K is best" — picking a
K off this table would be the threshold-sweep trap (mistake 2026-06-20).
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
OUT = ROOT / "research" / "results" / "raid_slot_bias.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
KS = (3, 5, 8, 12)          # reported as sensitivity, NOT to choose from
random.seed(23)


def clustered_ci(pairs, n_boot=4000):
    """Day-clustered bootstrap CI of a mean. pairs = [(day, value)]."""
    if not pairs:
        return None
    by_day = defaultdict(list)
    for d, v in pairs:
        by_day[d].append(v)
    days = list(by_day)
    if len(days) < 3:
        return None
    means = []
    for _ in range(n_boot):
        pick = [random.choice(days) for _ in days]
        vals = [x for d in pick for x in by_day[d]]
        if vals:
            means.append(st.mean(vals))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def load():
    rows = []
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if (r.get("variant_b") != "1" or r.get("status") != "CLOSED"
                    or r.get("universe") != "core9"
                    or r.get("symbol") not in CORE9):
                continue
            try:
                f, x, R = (int(float(r["fill_ts"])), int(float(r["exit_ts"])),
                           float(r["net_r"]))
            except (ValueError, TypeError, KeyError):
                continue
            if x <= f:
                continue
            rows.append({"f": f, "x": x, "R": R, "sym": r["symbol"],
                         "kind": r.get("level_kind", ""),
                         "cell": r.get("regime_cell", "")})
    rows.sort(key=lambda z: z["f"])
    return rows


def simulate(rows, K):
    """First-come-first-served slot allocation, exactly as the bot does."""
    open_until: list[int] = []
    for r in rows:
        open_until = [t for t in open_until if t > r["f"]]
        r["conc"] = len(open_until)          # concurrency AT ARRIVAL
        if len(open_until) < K:
            r["taken"] = True
            open_until.append(r["x"])
        else:
            r["taken"] = False
    return rows


def main() -> int:
    rows = load()
    print("§0.62 倉數上限的代價 —— 同一批訊號、同一套計分，只差那道閘")
    print(f"  母體：變體 B · core9 · 已結算 n={len(rows)}")
    print("  兩臂都從 shadow log 計分（level 價、零執行成本），"
          "所以「計分不對稱」這個機制在此被建構性排除\n")

    res = {"n": len(rows), "by_k": {}}
    print(f"{'K':>3} {'進場 n':>7} {'被擋 n':>7} {'被擋佔比':>9} "
          f"{'進場 meanR':>11} {'被擋 meanR':>11} {'差':>9} {'差的日聚類 CI':>20}")
    for K in KS:
        for r in rows:
            r.pop("taken", None)
        simulate(rows, K)
        tk = [r for r in rows if r["taken"]]
        bk = [r for r in rows if not r["taken"]]
        if not bk:
            print(f"{K:3d}   （此 K 之下無訊號被擋）")
            continue
        mt, mb = st.mean([r["R"] for r in tk]), st.mean([r["R"] for r in bk])
        # CI of the DIFFERENCE, clustered by day, resampling days once so
        # both arms move together — a paired-by-day comparison.
        by_day = defaultdict(lambda: ([], []))
        for r in rows:
            by_day[r["f"] // 86400][0 if r["taken"] else 1].append(r["R"])
        days = list(by_day)
        diffs = []
        for _ in range(4000):
            pick = [random.choice(days) for _ in days]
            a = [v for d in pick for v in by_day[d][0]]
            b = [v for d in pick for v in by_day[d][1]]
            if a and b:
                diffs.append(st.mean(b) - st.mean(a))
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        ratio = (mb / mt) if mt not in (0,) else float("nan")
        print(f"{K:3d} {len(tk):7d} {len(bk):7d} {100*len(bk)/len(rows):8.1f}% "
              f"{mt:+11.4f} {mb:+11.4f} {mb-mt:+9.4f} "
              f"  [{lo:+.4f},{hi:+.4f}]")
        res["by_k"][K] = {"taken_n": len(tk), "blocked_n": len(bk),
                          "taken_R": round(mt, 4), "blocked_R": round(mb, 4),
                          "delta": round(mb - mt, 4),
                          "ratio": round(ratio, 2) if ratio == ratio else None,
                          "ci": [round(lo, 4), round(hi, 4)],
                          "ci_spans_zero": lo <= 0 <= hi}

    # ── mechanism B: is mean R a function of how busy the tape is? ───────
    print("\n── 混淆檢查：meanR 隨「到達時的並發數」變化嗎 ──")
    print("   （被擋 ⇔ 並發已滿，所以「被擋」天生條件在高並發上）")
    simulate(rows, 10**9)                    # no cap: pure concurrency
    buckets = defaultdict(list)
    for r in rows:
        c = r["conc"]
        b = "0" if c == 0 else "1-2" if c <= 2 else "3-5" if c <= 5 else \
            "6-11" if c <= 11 else "12+"
        buckets[b].append(r["R"])
    order = ["0", "1-2", "3-5", "6-11", "12+"]
    conc = {}
    for b in order:
        v = buckets.get(b, [])
        if not v:
            continue
        conc[b] = {"n": len(v), "meanR": round(st.mean(v), 4)}
        print(f"   並發 {b:>5}  n={len(v):<5} meanR {st.mean(v):+.4f}")
    res["by_concurrency"] = conc

    vals = [conc[b]["meanR"] for b in order if b in conc]
    spread = (max(vals) - min(vals)) if vals else 0
    print(f"\n   跨桶最大差 {spread:+.4f}R")

    # ── verdict per the pre-committed reading ───────────────────────────
    k5 = res["by_k"].get(5)
    print()
    if k5 is None:
        v = "K=5 之下無樣本被擋，無法判讀"
    elif k5["ci_spans_zero"]:
        v = (f"倉數上限**不是** 4.3 倍的來源。同計分之下 K=5 的差是 "
             f"{k5['delta']:+.4f}R，日聚類 CI {k5['ci']} 含零。"
             f"產品端看到的倍數應歸因於計分不對稱（§0.57 的 0.1328R 執行落差）"
             f"與高並發條件化，不是抽籤偏誤。")
    elif spread > abs(k5["delta"]) * 0.5:
        v = (f"被擋確實較好（{k5['delta']:+.4f}R）**但混淆嚴重**："
             f"meanR 跨並發桶自己就差 {spread:+.4f}R。不得歸因於上限。")
    else:
        v = (f"倉數上限有真實代價：{k5['delta']:+.4f}R，CI {k5['ci']} 離零，"
             f"且並發桶間差異僅 {spread:+.4f}R 不足以解釋。"
             f"叢集內的先後順序真的有資訊 —— 分配規則成為活的研究問題。")
    print(f"判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
