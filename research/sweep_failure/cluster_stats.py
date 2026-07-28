# -*- coding: utf-8 -*-
"""Strategy #3 robustness — the repo's own discipline applied to the pooled t.

Three measurements of the FROZEN rules under the scenario-A cost model
(nothing is tuned; Gate F unchanged):

1. Monthly consistency — pooled t=+3.35 is an aggregate; mistake.md
   2026-06-02 says aggregates get vetted by per-period consistency
   (share of positive months, t over monthly means).
2. Day-clustered bootstrap — README admits the 9-symbol pooled t is
   inflated by cross-symbol correlation (nine coins take the same market
   shock). Resampling calendar DAYS (all trades of a day move together)
   gives the honest CI and a variance-inflation factor vs the iid
   bootstrap; effective t = iid t / sqrt(VIF).
3. Concurrency & portfolio path — README warns "extreme sessions open 9
   correlated positions at once". Measure the simultaneous-position
   distribution and the equity path / MDD at 0.5% risk per trade
   (PnL booked at exit; intrabar MTM would cut slightly deeper).

Run: python research/sweep_failure/cluster_stats.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import sweep_forward as SF

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def month(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def main() -> int:
    trades = []      # (fill_ts, exit_ts, netR)
    for s in SF.SYMS:
        raw = SF.SC.backtest_symbol(SF.SC.load_csv(str(SF.CACHE / f"{s}USDT_1h.csv")))
        net = SF.rescore(raw, "A")
        for (fill_ts, exit_ts, *_), (_, r) in zip(raw, net):
            trades.append((fill_ts, exit_ts, r))
    trades.sort()
    rs = [t[2] for t in trades]
    n = len(rs)
    mu = sum(rs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in rs) / (n - 1))
    t_iid = mu / (sd / math.sqrt(n))
    print(f"pool n={n}  meanR={mu:+.4f}  sd={sd:.3f}  iid t={t_iid:+.2f}")

    # ── 1. monthly consistency ────────────────────────────────────────
    bym: dict[str, list[float]] = defaultdict(list)
    for ts, _, r in trades:
        bym[month(ts)].append(r)
    months = sorted(bym)
    mmeans = [sum(bym[m]) / len(bym[m]) for m in months]
    pos = sum(1 for x in mmeans if x > 0)
    mm = sum(mmeans) / len(mmeans)
    msd = math.sqrt(sum((x - mm) ** 2 for x in mmeans) / (len(mmeans) - 1))
    t_month = mm / (msd / math.sqrt(len(mmeans)))
    print(f"\n[1] monthly consistency: {len(months)} months, positive "
          f"{pos}/{len(months)} ({100*pos/len(months):.0f}%),  mean-of-monthly-means "
          f"{mm:+.4f},  t={t_month:+.2f}")
    ranked = sorted(zip(mmeans, months))
    print("    worst:", ", ".join(f"{m} {x:+.3f}" for x, m in ranked[:3]))
    print("    best :", ", ".join(f"{m} {x:+.3f}" for x, m in ranked[-3:]))
    yearly: dict[str, list[float]] = defaultdict(list)
    for ts, _, r in trades:
        yearly[month(ts)[:4]].append(r)
    for y in sorted(yearly):
        v = yearly[y]
        print(f"    {y}: n={len(v):>5}  meanR={sum(v)/len(v):+.4f}")

    # ── 2. day-clustered bootstrap ────────────────────────────────────
    byd: dict[str, list[float]] = defaultdict(list)
    for ts, _, r in trades:
        byd[day(ts)].append(r)
    days = list(byd.values())
    rng = random.Random(11)
    NB = 4000
    cl_means = []
    for _ in range(NB):
        acc, cnt = 0.0, 0
        for _ in range(len(days)):
            grp = days[rng.randrange(len(days))]
            acc += sum(grp)
            cnt += len(grp)
        cl_means.append(acc / cnt)
    iid_means = []
    for _ in range(NB):
        acc = 0.0
        for _ in range(n):
            acc += rs[rng.randrange(n)]
        iid_means.append(acc / n)
    cl_means.sort()
    iid_means.sort()
    var_cl = sum((x - mu) ** 2 for x in cl_means) / NB
    var_iid = sum((x - mu) ** 2 for x in iid_means) / NB
    vif = var_cl / var_iid if var_iid > 0 else float("nan")
    t_eff = t_iid / math.sqrt(vif) if vif > 0 else float("nan")
    lo, hi = cl_means[int(0.025 * NB)], cl_means[int(0.975 * NB)]
    print(f"\n[2] day-clustered bootstrap ({len(days)} days): "
          f"CI95[{lo:+.4f},{hi:+.4f}]  VIF={vif:.2f}  effective t={t_eff:+.2f}"
          f"  (iid CI was [{iid_means[int(0.025*NB)]:+.4f},{iid_means[int(0.975*NB)]:+.4f}])")

    # ── 3. concurrency & portfolio path ───────────────────────────────
    events = []
    for f, e, _ in trades:
        events.append((f, +1))
        events.append((e, -1))
    events.sort()
    cur = mx = 0
    span: dict[int, int] = defaultdict(int)
    prev = events[0][0]
    for ts, d in events:
        span[cur] += ts - prev
        prev = ts
        cur += d
        mx = max(mx, cur)
    tot = sum(span.values())
    heavy = sum(v for k, v in span.items() if k >= 5) / tot
    print(f"\n[3] concurrency: max simultaneous={mx},  time at >=5 open "
          f"{100*heavy:.1f}%,  at 0 open {100*span[0]/tot:.1f}%")
    RISK = 0.005
    eq = peak = 1.0
    mdd = 0.0
    for _, _, r in sorted(trades, key=lambda x: x[1]):
        eq *= 1 + RISK * r
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    days_span = (trades[-1][1] - trades[0][0]) / 86400
    ann = (eq ** (365.0 / days_span) - 1) * 100
    print(f"    portfolio @0.5%/trade: total {100*(eq-1):+.1f}% over "
          f"{days_span/30.4:.0f} months  (~{ann:+.1f}%/yr),  MDD {100*mdd:.1f}% "
          f"(exit-booked; intrabar MTM deeper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
