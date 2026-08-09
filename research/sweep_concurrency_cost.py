"""What does the concurrency cap actually cost us?

Found 2026-08-09 while testing capacity weighting: of 1068 genuinely
prospective signals, **815 (76%) never get taken** — not because capital runs
out (peak usage was 16% of a $2000 account) but because the frozen rule caps
simultaneous positions at 5-10 and the executor takes them first-come.

That is a large, silent, unmeasured selection. Three questions, in order:

  Q1 ARE THE REJECTED TRADES WORSE THAN THE TAKEN ONES?
     If first-come selection happens to skim the good ones, the cap is cheap.
     If rejected trades are just as good, the cap is throwing away 3/4 of the
     edge. Compared on netR (unit-risk) so sizing does not confound it.
     Naive comparison is not enough: rejections CLUSTER in busy hours, and
     busy hours may themselves be better or worse. So also compare WITHIN
     the same hour-block — taken vs rejected side by side.

  Q2 HOW MUCH IS RECOVERED BY RAISING THE CAP?
     Walk the timeline at caps 5..20 and measure $ P&L, peak capital, and
     worst single hour. Capacity-weighted sizing throughout (option C), since
     that is the only executable scheme.

  Q3 WHAT DOES IT COST IN TAIL RISK?
     More concurrency = more correlated exposure. The binding number is the
     worst single hour-block, which in this log is -5.57R with 15 firing at
     once. Reported per cap so the trade-off is explicit, not assumed.

Nothing here changes a frozen rule. The concurrency cap lives in the
portfolio risk framework (TODO 0.4), which is exactly where a change would
have to be argued.

Usage: python research/sweep_concurrency_cost.py
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
ACCOUNT, RISK_PCT, PARTICIPATION = 2000.0, 0.02, 0.01
RNG = np.random.default_rng(20260809)


def load():
    out = []
    for r in csv.DictReader(LOG.open(encoding="utf-8")):
        if r.get("status") != "CLOSED" or r.get("net_r") in (None, ""):
            continue
        if not (r.get("first_seen_utc") and r.get("exit_utc")
                and r["first_seen_utc"] < r["exit_utc"]):
            continue
        try:
            out.append({"sym": r["symbol"], "day": r["fill_utc"][:10],
                        "hour": r["fill_utc"][:13],
                        "t0": int(r["fill_ts"]), "t1": int(r["exit_ts"]),
                        "r": float(r["net_r"]),
                        "atr_pct": float(r["atr"]) / float(r["entry_px"])})
        except (ValueError, ZeroDivisionError, KeyError):
            continue
    return sorted(out, key=lambda x: x["t0"])


def capacities(rows):
    vols = json.loads(VOLS.read_text()) if VOLS.exists() else {}
    per = defaultdict(list)
    for x in rows:
        per[x["sym"]].append(x["atr_pct"])
    return {s: (vols[s]["median"] * PARTICIPATION * float(np.median(a))
                if vols.get(s) else 0.0) for s, a in per.items()}


def walk(rows, cap_conc, size_fn):
    slots, taken, rejected = [], [], []
    peak_cap = 0.0
    worst_hour = defaultdict(float)
    for x in rows:
        slots = [o for o in slots if o[0] > x["t0"]]
        risk = size_fn(x["sym"])
        if len(slots) >= cap_conc or risk <= 0:
            rejected.append(x)
            continue
        slots.append((x["t1"], risk))
        peak_cap = max(peak_cap, sum(o[1] for o in slots))
        pnl = x["r"] * risk
        taken.append({**x, "risk_usd": risk, "pnl": pnl})
        worst_hour[x["hour"]] += pnl
    return taken, rejected, peak_cap, (min(worst_hour.values()) if worst_hour else 0.0)


def ci_day(vals_by_day, iters=20000):
    days = list(vals_by_day)
    if len(days) < 3:
        return None, None
    tot = []
    for _ in range(iters):
        pick = RNG.integers(0, len(days), len(days))
        tot.append(sum(v for i in pick for v in vals_by_day[days[i]]))
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def main() -> int:
    rows = load()
    cap = capacities(rows)
    slot = ACCOUNT * RISK_PCT
    size = lambda s: min(slot, cap.get(s, 0.0))  # noqa: E731

    print(f"真前瞻 {len(rows)} 筆 · {len({x['sym'] for x in rows})} 幣 · "
          f"{len({x['day'] for x in rows})} 天")
    print(f"容量加權（方案 C）· 本金 ${ACCOUNT:,.0f} · 目標每筆 ${slot:,.0f}\n")

    # ── Q1 被擋掉的比較差嗎 ────────────────────────────────────────────
    taken, rejected, _, _ = walk(rows, 8, size)
    tr = np.array([x["r"] for x in taken])
    rr = np.array([x["r"] for x in rejected])
    print("═" * 76)
    print("  Q1 被並發擋掉的訊號，品質比較差嗎？（用 netR 比，排除 sizing 干擾）")
    print("═" * 76)
    print(f"  成交的   n={len(tr):>4}  meanR={tr.mean():+.4f}  WR={100*(tr>0).mean():>3.0f}%")
    print(f"  被擋的   n={len(rr):>4}  meanR={rr.mean():+.4f}  WR={100*(rr>0).mean():>3.0f}%")
    print(f"  差異     {tr.mean()-rr.mean():+.4f}R")

    # within-hour paired view — rejections cluster in busy hours
    pair_t, pair_r = [], []
    by_hour_t = defaultdict(list)
    by_hour_r = defaultdict(list)
    for x in taken:
        by_hour_t[x["hour"]].append(x["r"])
    for x in rejected:
        by_hour_r[x["hour"]].append(x["r"])
    for h in set(by_hour_t) & set(by_hour_r):
        pair_t.append(np.mean(by_hour_t[h]))
        pair_r.append(np.mean(by_hour_r[h]))
    if pair_t:
        d = np.array(pair_t) - np.array(pair_r)
        boot = [np.mean(RNG.choice(d, len(d), replace=True)) for _ in range(20000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n  同一小時內配對（n={len(d)} 個小時，排除『忙碌時段本身好壞』的干擾）")
        print(f"  成交 − 被擋 = {d.mean():+.4f}R   bootstrap CI[{lo:+.4f},{hi:+.4f}]")
        print("  → " + ("先到先得**確實**挑到較好的（CI 不含零）"
                        if lo * hi > 0 else
                        "先到先得與被擋的**沒有品質差異**（CI 含零）"
                        " —— 擋掉的是等值的 edge"))

    # ── Q2/Q3 放寬並發的收益與代價 ────────────────────────────────────
    print("\n" + "═" * 76)
    print("  Q2/Q3 放寬並發上限：拿回多少、賠上多少尾部")
    print("═" * 76)
    print(f"  {'上限':>4}{'成交':>6}{'被擋':>6}{'總損益':>10}{'日聚類CI':>22}"
          f"{'峰值佔用':>10}{'最差單小時':>11}")
    for c in (5, 8, 10, 12, 15, 20, 30):
        tk, rj, pk, worst = walk(rows, c, size)
        by_day = defaultdict(list)
        for x in tk:
            by_day[x["day"]].append(x["pnl"])
        lo, hi = ci_day(by_day)
        ci = f"[{lo:+,.0f},{hi:+,.0f}]" if lo is not None else "—"
        pnl = sum(x["pnl"] for x in tk)
        print(f"  {c:>4}{len(tk):>6}{len(rj):>6}${pnl:>+9,.0f}{ci:>22}"
              f"${pk:>9,.0f}${worst:>10,.0f}")
    print("\n  峰值佔用 = 同時在場的風險金額總和（不是保證金，是最大可能虧損）")
    print("  最差單小時 = 同一小時所有部位加總的最壞損益（相關性風險的實測值）")
    print("\n  ⚠ 並發上限屬於統一風控框架（TODO §0.4）的決定，不是策略參數。")
    print("     本檔只量成本，不改任何凍結規則。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
