# -*- coding: utf-8 -*-
"""Does a raid trade need ROOM to run? (pre-registered, exploration on backtest)

Why this question and why now
-----------------------------
§0.57 settled the arithmetic: the mechanical rule earns +0.084 R/trade at the
level price and loses 0.133 R/trade to the publishing gate. So the rule alone
is not a strategy; it needs conditioning that lifts per-trade edge several
fold. The two conditionings that already have evidence are regime (§0.59b,
+0.106 forward) and the survivor stack (§0.472) -- but the stack is BTC-only
by construction: three of its seven features are Coinglass/V7 (BTC only) and
four need 1-minute bars, of which this repo holds 110 days for BTC/ETH only.

This asks something that costs no new data at all and is computable on every
coin over the whole history: WHERE IS THE TRADE GOING, and is anything in
the way?

The mechanism is the strategy's own: a failed sweep fades back INTO the
range. If another untouched pool sits right where the fade wants to go, the
move has a wall in front of it. If a pool sits just beyond the break, the
break has a magnet pulling it away from us. The V7 line proved exactly these
two shapes on a different strategy (§0.480/§0.481/§0.482: 前方跑道乾淨 65%
vs 近牆 57%, permutation p=0.028; 背靠支撐 +12.1pp, p=0.0005) -- but that was
V7 signals, so transferring it here is a NEW claim and gets tested like one,
the same way the ADX->grid transfer was (and failed, §0.93 九).

PRE-REGISTERED PREDICTIONS (written before the first run; do not edit after)
---------------------------------------------------------------------------
  P1 ROOM     netR RISES with the distance from the swept level to the
              nearest still-untouched pool IN THE FADE DIRECTION.
              Named worst cell: bottom tercile (a wall close in front).
  P2 MAGNET   netR FALLS with closeness of the nearest still-untouched pool
              BEYOND the level in the break direction.
              Named worst cell: bottom tercile (a magnet just past the break).

SURVIVAL BAR (all four; a dimension that misses any one is dead, not "nearly")
  1 direction as named (top tercile minus bottom tercile carries the
    predicted sign)
  2 both halves of the sample carry that sign
  3 per-coin breadth >= 6/9 on the sign of the gap
  4 day-clustered bootstrap CI95 of the gap excludes zero
Every cell is reported, including the middle one and the "no pool at all"
count -- no cell is dropped for being inconvenient. A bucket that is empty
or holds >90% of the sample means the instrument is broken, not that the
market is strange (mistake.md 2026-08-02).

Scope discipline
  * Universe core9, FULL history: this is EXPLORATION on backtest data.
    Nothing here may be read on the forward rows -- variant A's forward
    sample is reserved for confirmation, and re-cutting it after seeing
    this would be the C/D trap (§0.92 rule 3).
  * Entry/exit rules frozen: every trade is exactly the trade the Gate F
    scorer already books (SLIP=0 gross engine + scenario-A bps, identical
    to sweep_forward.rescore). Only the LABEL on each trade is new.
  * If a dimension survives, it does NOT go live and does NOT get a clock
    here: it must first be written into §0.43 as a named variant and then
    accumulate from zero.

Causality
  A pool counts at the fill bar only if it was established at or before that
  bar AND had not been traded through by that bar. Both facts are known at
  the fill bar; nothing downstream of the fill is consulted.

Run: python research/sweep_failure/room_ahead.py
Out: research/results/sweep_room_ahead.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"          # gross engine; bps costs applied below
import sweep_core as SC            # noqa: E402
import level_types as LT           # noqa: E402
from sweep_forward import SCEN, boot_ci_clustered  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/sweep_room_ahead.json"
CACHE = HERE / ".cache"
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
NOPOOL = 99.0                     # sentinel distance when nothing is in the way


def all_pools(bars):
    """[(est_bar, price, death_bar)] for every causal pool the raid line uses.

    Sources: swing pivots (the frozen engine's own level definition) plus the
    three time-defined families level_types already validated (session,
    PDH/PDL, PWH/PWL). death_bar = first bar strictly after establishment
    whose range trades through the price; the pool is alive on [est, death).
    """
    n = len(bars)
    h = [b[SC.H] for b in bars]
    l = [b[SC.L] for b in bars]
    raw = []                        # (est, price, side)

    for i in range(SC.PIVOT, n - SC.PIVOT):
        seg = range(i - SC.PIVOT, i + SC.PIVOT + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            raw.append((i + SC.PIVOT, h[i], 1))
        if all(l[i] <= l[k] for k in seg) and any(l[i] < l[k] for k in seg if k != i):
            raw.append((i + SC.PIVOT, l[i], -1))
    for kind, items in LT.build_levels(bars).items():
        raw.extend(items)

    # death by one sweep over the bars: a resistance pool dies at the first
    # later bar whose high exceeds it, so among the alive resistances the
    # LOWEST always dies first -> a min-heap answers every bar in log time.
    # (The naive per-pool rescan is O(pools x bars) = ~130M per coin.)
    import heapq
    born = defaultdict(list)
    for est, price, side in raw:
        born[est].append((price, side))
    up, dn = [], []                 # min-heap of resistances, max-heap of supports
    death = {}
    for j in range(n):
        while up and up[0][0] < h[j]:
            price, key = heapq.heappop(up)
            death[key] = j
        while dn and -dn[0][0] > l[j]:
            negp, key = heapq.heappop(dn)
            death[key] = j
        for price, side in born.get(j, ()):
            key = (j, price, side)
            (heapq.heappush(up, (price, key)) if side == 1
             else heapq.heappush(dn, (-price, key)))
    out = []
    for est, price, side in raw:
        out.append((est, price, death.get((est, price, side), n)))
    return out


def distances(pools, f, lvl, d, atr):
    """(room, magnet) in ATR units at fill bar f, for fade direction d."""
    ahead = behind = None
    for est, price, death in pools:
        if est > f or death <= f:
            continue
        gap = (price - lvl) * d
        if gap > 0:
            ahead = gap if ahead is None else min(ahead, gap)
        elif gap < 0:
            behind = -gap if behind is None else min(behind, -gap)
    room = NOPOOL if ahead is None else ahead / atr
    magnet = NOPOOL if behind is None else behind / atr
    return room, magnet


def cell_stats(rows):
    """rows = [(ts, r, sym)] -> n, mean, wr."""
    if not rows:
        return dict(n=0, mean=float("nan"), wr=float("nan"))
    rs = [r for _, r, _ in rows]
    return dict(n=len(rs), mean=sum(rs) / len(rs),
                wr=100.0 * sum(1 for x in rs if x > 0) / len(rs))


def report(name, pred, labelled, lo_q, hi_q):
    """Score one dimension against the four pre-registered bars."""
    cells = {"低（近）": [], "中": [], "高（遠）": []}
    for ts, r, sym, v in labelled:
        k = "低（近）" if v <= lo_q else ("高（遠）" if v > hi_q else "中")
        cells[k].append((ts, r, sym))

    print(f"\n  【{name}】預測：{pred}")
    print(f"     三等分切點 {lo_q:.2f} / {hi_q:.2f} ATR"
          f"（無池記為 {NOPOOL:.0f}）")
    print(f"     {'格':<10}{'n':>7}{'meanR':>10}{'勝率':>8}")
    st = {}
    for k in ("低（近）", "中", "高（遠）"):
        s = cell_stats(cells[k])
        st[k] = s
        print(f"     {k:<10}{s['n']:>7}{s['mean']:>+10.4f}{s['wr']:>7.1f}%")

    hi, lo = cells["高（遠）"], cells["低（近）"]
    gap = st["高（遠）"]["mean"] - st["低（近）"]["mean"]
    # day-clustered CI of the gap: resample days, recompute both cell means
    pairs = [(ts, r, 1) for ts, r, _ in hi] + [(ts, r, 0) for ts, r, _ in lo]
    ci = gap_ci(pairs)
    half = len(labelled) // 2
    g1 = half_gap(labelled[:half], lo_q, hi_q)
    g2 = half_gap(labelled[half:], lo_q, hi_q)
    per = per_coin_gap(labelled, lo_q, hi_q)
    pos = sum(1 for v in per.values() if v is not None and v > 0)
    tot = sum(1 for v in per.values() if v is not None)

    bars_ = {
        "1 方向如宣稱": gap > 0,
        "2 兩半同號": (g1 is not None and g2 is not None
                       and g1 > 0 and g2 > 0),
        f"3 廣度 {pos}/{tot} ≥6": pos >= 6,
        "4 日聚類 CI 離零": ci[0] > 0,
    }
    print(f"     高−低 = {gap:+.4f} R，日聚類 CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"     兩半 {fmt(g1)} / {fmt(g2)}；逐幣 "
          + " ".join(f"{k}{fmt(v)}" for k, v in per.items()))
    for b, ok in bars_.items():
        print(f"       {'✅' if ok else '❌'} {b}")
    verdict = "存活" if all(bars_.values()) else "陣亡"
    print(f"     → {name}：{verdict}")
    return dict(cells={k: st[k] for k in st}, gap=gap, ci=list(ci),
                half1=g1, half2=g2, per_coin=per, bars=bars_,
                verdict=verdict, cuts=[lo_q, hi_q])


def fmt(x):
    return "  n/a" if x is None else f"{x:+.3f}"


def half_gap(rows, lo_q, hi_q):
    hi = [(t, r, s) for t, r, s, v in rows if v > hi_q]
    lo = [(t, r, s) for t, r, s, v in rows if v <= lo_q]
    if not hi or not lo:
        return None
    return cell_stats(hi)["mean"] - cell_stats(lo)["mean"]


def per_coin_gap(rows, lo_q, hi_q):
    out = {}
    for sym in SYMS:
        sub = [r for r in rows if r[2] == sym]
        out[sym] = half_gap(sub, lo_q, hi_q)
    return out


def gap_ci(triples, nb=3000, seed=7):
    """Day-clustered CI of (mean of group 1) - (mean of group 0)."""
    import random
    byd = defaultdict(list)
    for ts, r, g in triples:
        from datetime import datetime, timezone
        byd[datetime.fromtimestamp(ts, tz=timezone.utc).date()].append((r, g))
    days = list(byd.values())
    rng = random.Random(seed)
    out = []
    for _ in range(nb):
        s1 = c1 = s0 = c0 = 0.0
        for _ in range(len(days)):
            for r, g in days[rng.randrange(len(days))]:
                if g:
                    s1 += r; c1 += 1
                else:
                    s0 += r; c0 += 1
        if c1 and c0:
            out.append(s1 / c1 - s0 / c0)
    out.sort()
    if not out:
        return (float("nan"), float("nan"))
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main() -> int:
    print("=" * 84)
    print("  獵取的「空間」——前方跑道與後方磁鐵（預註冊，回測資料，core9 全歷史）")
    print("=" * 84)
    rows_room, rows_mag = [], []
    nopool_room = nopool_mag = 0
    for sym in SYMS:
        p = CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            print(f"  !! {sym} 無資料，跳過")
            continue
        bars = SC.load_csv(str(p))
        idx = {b[0]: i for i, b in enumerate(bars)}
        pools = all_pools(bars)
        trades = SC.backtest_symbol(bars)
        s = SCEN["A"]
        for fill_ts, _ex, r, lvl, atr, stopped, _pierce, side in trades:
            f = idx.get(fill_ts)
            if f is None:
                continue
            d = 1 if side == "LONG" else -1
            room, magnet = distances(pools, f, lvl, d, atr)
            cost = (s["entry"] + (s["sexit"] if stopped else s["texit"])) \
                / 1e4 * lvl / (SC.DIS * atr)
            net = r - cost
            rows_room.append((fill_ts, net, sym, room))
            rows_mag.append((fill_ts, net, sym, magnet))
            nopool_room += room == NOPOOL
            nopool_mag += magnet == NOPOOL
        print(f"  {sym:<5} 交易 {len(trades):>5}  池 {len(pools):>6}")

    rows_room.sort(key=lambda x: x[0])
    rows_mag.sort(key=lambda x: x[0])
    base = cell_stats([(t, r, s) for t, r, s, _ in rows_room])
    print(f"\n  全體基準 n={base['n']}  meanR {base['mean']:+.4f}  "
          f"勝率 {base['wr']:.1f}%")
    print(f"  無池（前方 {nopool_room} / 後方 {nopool_mag}）——"
          f"佔比 {100*nopool_room/max(base['n'],1):.1f}% / "
          f"{100*nopool_mag/max(base['n'],1):.1f}%")

    def terciles(rows):
        v = sorted(x[3] for x in rows)
        return v[len(v) // 3], v[2 * len(v) // 3]

    res = {"base": base, "nopool_room": nopool_room, "nopool_magnet": nopool_mag}
    lo, hi = terciles(rows_room)
    res["room"] = report("前方跑道（到最近未掃池的距離）",
                         "距離越遠 netR 越高；最差＝近牆", rows_room, lo, hi)
    lo, hi = terciles(rows_mag)
    res["magnet"] = report("後方磁鐵（突破方向最近未掃池的距離）",
                           "距離越遠 netR 越高（近磁鐵最差）", rows_mag, lo, hi)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
