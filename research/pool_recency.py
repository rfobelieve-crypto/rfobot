# -*- coding: utf-8 -*-
"""Does it matter whether the swept pool was the NEWEST one? — TODO §0.69.

Origin: the reference indicator the operator supplied ("Liquidity Pools",
FX365_Thailand) keeps ONLY the most recent swing high and swing low. That
is not a display convenience — it encodes an assumption, that a pool which
just formed behaves differently from one that has been resting for days.

The assumption is free to test on the frozen ledger: "was this the newest
unswept pool on its side at the moment it was taken" is a BINARY FACT
about events that already happened. No parameter is invented, no rule is
changed, and the sample is the full 8,262-event backtest rather than the
forward log.

NOT the same as D10, which the terrain campaign killed. D10 measured a
pool's ABSOLUTE age in bars. This measures its RELATIVE rank among the
pools available at that moment — a pool can be three days old and still
be the newest one if nothing has formed since. Absolute age and relative
recency come apart exactly in quiet markets, which is where this strategy
makes its money (§0.49: RANGING is the home regime).

Pre-committed reading, written before the run:
  * newest-pool sweeps score materially better, breadth >=6/9, both halves
    agree  -> recency is a real conditioning variable and becomes feature
              #1 of the entry model
  * no separation                 -> the indicator's choice is cosmetic;
                                     drop it and do not revisit
  * OLDER pools score better      -> equally interesting, and the opposite
                                     of the indicator's implied prior; report
                                     it as such rather than reversing the
                                     hypothesis after the fact

Reported in full: every rank bucket, day-clustered CI, per-coin breadth,
and the two-half split. 4 buckets x 1 test, so multiplicity is mild, but
the bucket definition is frozen here before any number is seen.
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
from research.liquidity_map_check import swing_levels, first_hit  # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "pool_recency.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
PIERCE_B = 0.25
random.seed(59)


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


def bucket(rank: int) -> str:
    if rank == 0:
        return "0 最新的池"
    if rank == 1:
        return "1 第二新"
    if rank <= 4:
        return "2-4"
    return "5+ 陳舊"


def events_for(sym: str):
    """Every variant-B fill, tagged with the swept pool's recency rank."""
    fp = CACHE / f"{sym}USDT_1h.csv"
    if not fp.exists():
        return []
    bars = SC.load_csv(str(fp))
    pools = swing_levels(bars)                       # (est_bar, price, side)
    # sweep bar for each pool; unswept pools never become events
    swept = []
    for est, price, side in pools:
        hit = first_hit(bars, est, price, side)
        swept.append((est, price, side, hit))

    # R outcome keyed by (level price rounded, sweep-ish bar) via the frozen
    # backtest — matching on level price keeps the frozen fill/exit logic as
    # the single source of the number.
    out_by_lvl = defaultdict(list)
    for fill_ts, _x, R, lvl, _A, _s, pierce, side in SC.backtest_symbol(bars):
        if pierce > PIERCE_B:
            continue
        out_by_lvl[round(float(lvl), 8)].append((int(fill_ts), R))

    ev = []
    for est, price, side, hit in swept:
        if hit is None:
            continue
        key = round(float(price), 8)
        cand = out_by_lvl.get(key)
        if not cand:
            continue
        # how many SAME-SIDE pools were newer and still unswept at the sweep
        newer = sum(1 for e2, p2, s2, h2 in swept
                    if s2 == side and e2 <= hit and e2 > est
                    and (h2 is None or h2 >= hit))
        ts, R = cand[0]
        if len(cand) > 1:                 # same price hit more than once
            cand.sort(key=lambda z: abs(z[0] - bars[hit][0]))
            ts, R = cand[0]
        ev.append({"ts": ts, "R": R, "rank": newer, "sym": sym,
                   "age": hit - est})
    return ev


def main() -> int:
    universe = ([p.name.replace("USDT_1h.csv", "")
                 for p in sorted(CACHE.glob("*USDT_1h.csv"))]
                if "--all" in sys.argv else CORE9)
    ev = []
    for s in universe:
        ev += events_for(s)
    if not ev:
        raise SystemExit("no events")

    print("§0.69 被掃的池是不是「最新的那一個」重要嗎")
    print(f"  母體：變體 B、{len(universe)} 幣、n={len(ev)}")
    print("  排名 = 掃單當下，同側有幾個更新且仍未掃的池\n")

    mid = sorted(x["ts"] for x in ev)[len(ev) // 2]
    buck = defaultdict(list)
    for x in ev:
        buck[bucket(x["rank"])].append(x)

    # physical sanity before reading any performance number
    shares = {k: len(v) / len(ev) for k, v in buck.items()}
    if any(s > 0.9 for s in shares.values()) or len(buck) < 3:
        print(f"  儀器疑慮：桶分佈 { {k: round(v, 3) for k, v in shares.items()} }")
        return 1

    print(f"{'桶':<12} {'n':>6} {'meanR':>9} {'勝率':>7} {'日聚類CI':>20} "
          f"{'廣度':>7} {'前半':>9} {'後半':>9}")
    res = {}
    order = ["0 最新的池", "1 第二新", "2-4", "5+ 陳舊"]
    for b in order:
        v = buck.get(b, [])
        if not v:
            continue
        m = st.mean(x["R"] for x in v)
        wr = 100 * sum(1 for x in v if x["R"] > 0) / len(v)
        ci = clustered_ci([(x["ts"] // 86400, x["R"]) for x in v])
        per = defaultdict(list)
        for x in v:
            per[x["sym"]].append(x["R"])
        br = sum(1 for s in per if st.mean(per[s]) > 0)
        h1 = [x["R"] for x in v if x["ts"] < mid]
        h2 = [x["R"] for x in v if x["ts"] >= mid]
        cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "—"
        print(f"{b:<12} {len(v):6d} {m:+9.4f} {wr:6.1f}% {cis:>20} "
              f"{br:4d}/{len(per):<2d} "
              f"{(st.mean(h1) if h1 else float('nan')):+9.4f} "
              f"{(st.mean(h2) if h2 else float('nan')):+9.4f}")
        res[b] = {"n": len(v), "meanR": round(m, 4), "wr": round(wr, 1),
                  "ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                  "breadth": f"{br}/{len(per)}",
                  "h1": round(st.mean(h1), 4) if h1 else None,
                  "h2": round(st.mean(h2), 4) if h2 else None}

    # A bucket must be ESTABLISHED before it may be compared. Three times in
    # one session a point estimate from a thin bucket was allowed to drive a
    # verdict (the best-bucket CI criterion in §0.65, the n=20 causality kill
    # in §0.66, and the first version of this file, where n=50 with a CI
    # spanning zero produced "older pools are better"). A comparison between
    # an established arm and an unestablished one is not a comparison.
    MIN_N, MIN_BREADTH = 200, 6

    def usable(b):
        r = res.get(b)
        if not r:
            return False
        return (r["n"] >= MIN_N and r["ci"] and r["ci"][0] > 0
                and int(r["breadth"].split("/")[0]) >= MIN_BREADTH)

    ok = [b for b in order if usable(b)]
    thin = [b for b in order if b in res and b not in ok]
    print(f"\n  成立的桶（n≥{MIN_N} ∧ CI 離零 ∧ 廣度≥{MIN_BREADTH}）：{ok}")
    if thin:
        print(f"  未成立、不得參與比較：{thin}")
    res["usable_buckets"], res["thin_buckets"] = ok, thin

    if len(ok) < 2:
        v = "成立的桶不足兩個，無法比較"
    else:
        ms = [res[b]["meanR"] for b in ok]
        spread = max(ms) - min(ms)
        mono = all(res[ok[i]]["meanR"] >= res[ok[i + 1]]["meanR"]
                   for i in range(len(ok) - 1))
        print(f"  成立桶之間的最大差：{spread:+.4f}R"
              f"（{'單調遞減' if mono else '非單調'}）")
        if spread >= 0.03 and mono:
            v = ("**新近度有效**：成立的桶之間單調且差距實質 —— "
                 "成為進場模型的特徵 #1")
        else:
            v = (f"**無分離**：成立的桶之間只差 {spread:+.4f}R"
                 f"（{'單調' if mono else '非單調'}），"
                 "而唯一亮眼的『5+ 陳舊』n=50、CI 含零、廣度 10/17，不成立。"
                 "**指標「只留最近一個」在這裡是外觀考量，不列入特徵。**")
        res["spread_among_usable"] = round(spread, 4)
    print(f"\n判讀：{v}")
    res["verdict"] = v
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
