# -*- coding: utf-8 -*-
"""Does a level's ROUNDNESS matter? — TODO §0.76.

Origin: the 7-day verification map showed 6 traded events, two of which sat
on exact round numbers (76,500.00 and 80,000.00). Stops cluster at round
prices — every retail platform nudges users toward them — so "how round is
this level" is a candidate the system has never measured.

PRIOR, from this project's own data and stated before running: §0.71b/D5
both found that OBVIOUS locations pay less (stacked pools +0.061 vs single
+0.111; dense terrain 54% vs sparse 62%). A round number is the most
obvious location there is. So the in-house prior says ROUNDER = WORSE —
the opposite of the folk intuition that round numbers are "strong levels".

DEFINITION, frozen and scale-free (DOGE at $0.21 and BTC at $78,000 must
be comparable): take the level's first five significant digits and count
TRAILING ZEROS. 76,500 -> 76500 -> 2. 80,000 -> 80000 -> 3. 74,514.1 ->
74514 -> 0. DOGE 0.21000 -> 21000 -> 3. Buckets: 0 / 1 / 2+, fixed here.

Population: the AUTHORITATIVE one (shadow_review.rederive — four families,
scenario-A costs), per the §0.70b lesson. Variant-B, settled only.

Pre-committed reading:
  * rounder is worse, established arms, families agree -> third independent
    confirmation of "obvious pays less", candidate for the entry model
  * rounder is better -> contradicts the in-house prior; report as such,
    needs a mechanism before use
  * no separation -> roundness is decoration; drop it
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

from shadow_review import rederive                          # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "level_roundness.json"
CORE9 = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"}
random.seed(223)


def trailing_zeros(price: float) -> int:
    """Trailing zeros of the first five significant digits."""
    if price <= 0:
        return 0
    s = f"{price:.10e}"                    # d.dddddddddde±xx
    digits = (s.split("e")[0].replace(".", "") + "0" * 5)[:5]
    tz = 0
    for ch in reversed(digits):
        if ch == "0":
            tz += 1
        else:
            break
    return tz


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
    # sanity of the instrument first, on known answers
    # my own first expectation for 80000 was wrong (3): its five
    # significant digits are 80000 — four trailing zeros. The check caught
    # the CHECKER, which is fine; that is what known answers are for.
    known = [(76500.0, 2), (80000.0, 4), (74514.1, 0), (0.21, 3),
             (0.215, 2), (123.45, 0), (70000.0, 4)]
    bad = [(p, want, trailing_zeros(p)) for p, want in known
           if trailing_zeros(p) != want]
    if bad:
        raise SystemExit(f"instrument failed known answers: {bad}")
    print("§0.76 池價的「整數程度」—— 儀器先過已知答案 ✓\n")

    rows = []
    for fp in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = fp.name.replace("USDT_1h.csv", "")
        try:
            _b, trades, _p = rederive(sym)
        except Exception:
            continue
        for t in trades:
            if t["net"] is None or not t["b"]:
                continue
            rows.append({"ts": int(t["fill_ts"]), "R": float(t["net"]),
                         "sym": sym, "kind": t["kind"],
                         "tz": min(trailing_zeros(t["lvl"]), 2)})

    print(f"  母體：權威口徑（rederive）變體 B 已結算 n={len(rows)}\n")
    lab = {0: "0 不整", 1: "1 個零", 2: "2+ 個零（整數關卡）"}
    print(f"  {'桶':<20} {'n':>6} {'佔比':>6} {'meanR':>9} {'勝率':>7} "
          f"{'日聚類CI':>20} {'廣度':>7}")
    res = {}
    for tz in (0, 1, 2):
        v = [x for x in rows if x["tz"] == tz]
        if len(v) < 60:
            print(f"  {lab[tz]:<20} {len(v):6d}   樣本不足")
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
        est = bool(len(v) >= 200 and ci and ci[0] > 0 and br >= 6)
        print(f"  {lab[tz]:<20} {len(v):6d} {100*len(v)/len(rows):5.1f}% "
              f"{m:+9.4f} {wr:6.1f}% {cis:>20} {br:3d}/{len(per):<3d}"
              f"{'  ✓成立' if est else '  ·未成立'}")
        res[str(tz)] = {"n": len(v), "meanR": round(m, 4), "wr": round(wr, 1),
                        "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                        "breadth": f"{br}/{len(per)}", "established": est}

    ok = [k for k, r in res.items() if r["established"]]
    if len(ok) >= 2:
        ms = {k: res[k]["meanR"] for k in ok}
        lo_k, hi_k = min(ok), max(ok)
        gap = ms[hi_k] - ms[lo_k]          # rounder minus less-round
        print(f"\n  成立桶：{ok}｜最整 − 最不整 = {gap:+.4f}R")
        if gap < -0.03:
            v = ("**越整越差，第三個獨立確認「明顯的位置付得少」**"
                 "（§0.71b 堆疊、D5 密度之後）—— 進場模型候選特徵")
        elif gap > 0.03:
            v = ("**越整越好** —— 與自家先驗相反（民間直覺的方向），"
                 "需要機制解釋才可用，列觀察")
        else:
            v = f"**無分離**（{gap:+.4f}R）—— 整數程度是裝飾，不列入"
    else:
        v = f"成立桶不足（{ok}），判不出來"
    print(f"\n判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
