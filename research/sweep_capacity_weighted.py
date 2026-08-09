"""Option C: capacity-weighted sizing across all 29 symbols.

WHY (2026-08-09): the 29-symbol basket has a structural conflict — the coins
with the most edge are not the coins that can absorb size. Trimming to the 18
that can (capacity >= $50/trade at 1% participation) throws away real signal:
HBAR +12.65R, SAND +8.03R, VET +7.30R were all cut, and the trimmed basket's
clustered CI-low fell from +0.024 to -0.017.

Option C keeps every symbol but sizes each by what it can absorb:

    risk_usd(sym) = min(account x risk_pct, capacity_usd(sym))

Big coins get the full slot; small coins get whatever they can take. Nothing
is discarded, nothing is oversized.

WHAT CHANGES IN THE MEASUREMENT
  R is risk-normalised, so it silently assumes every trade risks the same
  amount. Once sizing differs per symbol that assumption breaks, and the
  honest unit becomes DOLLARS. This script therefore reports $ P&L and $
  return on the capital actually required, not mean R.

CONCURRENCY IS SIMULATED, NOT ASSUMED
  Signals are walked in time order. A trade is taken only if a slot is free
  (frozen cap 5-10; default 8). Rejected signals are COUNTED — under
  equal-weight sizing the executor drops them arbitrarily, which is a silent
  selection; here it is measured.

CAPITAL REQUIREMENT
  Tracked as the peak sum of concurrent risk-dollars. This is the number that
  answers "does $2000 actually cover this basket".

Capacity source: research/results/sweep_capacity_volumes.json (median hourly
quote volume x participation x that symbol's median ATR%). It is an UPPER
bound — see sweep_capacity_estimate.py for what it ignores.

Usage:
    python research/sweep_capacity_weighted.py
    python research/sweep_capacity_weighted.py --account 2000 --risk 0.02
"""
from __future__ import annotations

import argparse
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
PARTICIPATION = 0.01
TRIM_FLOOR = 50.0            # the 18-coin basket's capacity cut
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
    cap = {}
    for s, a in per.items():
        v = vols.get(s)
        cap[s] = (v["median"] * PARTICIPATION * float(np.median(a))) if v else 0.0
    return cap


def walk(rows, size_fn, concurrency, allowed=None):
    """Time-ordered fill simulation with a hard concurrency cap."""
    open_slots = []          # list of (exit_ts, risk_usd)
    taken, rejected_cap, rejected_uni = [], 0, 0
    peak_capital = 0.0
    for x in rows:
        open_slots = [o for o in open_slots if o[0] > x["t0"]]
        if allowed is not None and x["sym"] not in allowed:
            rejected_uni += 1
            continue
        if len(open_slots) >= concurrency:
            rejected_cap += 1
            continue
        risk = size_fn(x["sym"])
        if risk <= 0:
            rejected_uni += 1
            continue
        open_slots.append((x["t1"], risk))
        peak_capital = max(peak_capital, sum(o[1] for o in open_slots))
        taken.append({**x, "risk_usd": risk, "pnl": x["r"] * risk})
    return taken, rejected_cap, rejected_uni, peak_capital


def day_ci(taken, iters=20000):
    by = defaultdict(list)
    for x in taken:
        by[x["day"]].append(x["pnl"])
    days = list(by)
    if len(days) < 3:
        return None, None
    tot = []
    for _ in range(iters):
        pick = RNG.integers(0, len(days), len(days))
        tot.append(sum(v for i in pick for v in by[days[i]]))
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def report(tag, taken, rej_cap, rej_uni, peak, account):
    if not taken:
        print(f"  {tag:<22} (無成交)")
        return
    pnl = np.array([x["pnl"] for x in taken])
    lo, hi = day_ci(taken)
    ci = f"[{lo:+,.0f},{hi:+,.0f}]" if lo is not None else "—"
    syms = {x["sym"] for x in taken}
    pos = sum(1 for s in syms if sum(x["pnl"] for x in taken if x["sym"] == s) > 0)
    print(f"  {tag:<22} 成交 {len(taken):>4}  總損益 ${pnl.sum():>+9,.0f}  "
          f"日聚類CI {ci:>20}  幣正 {pos}/{len(syms)}")
    print(f"  {'':<22} 峰值佔用 ${peak:>7,.0f}（本金 ${account:,.0f} 的 "
          f"{peak/account*100:>3.0f}%）  被並發擋 {rej_cap}  被籃子擋 {rej_uni}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=2000.0)
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    rows = load()
    cap = capacities(rows)
    slot = a.account * a.risk
    trimmed = {s for s, c in cap.items() if c >= TRIM_FLOOR}
    days = len({x["day"] for x in rows})

    print(f"真前瞻 {len(rows)} 筆 · {len(cap)} 幣 · {days} 天 · "
          f"並發上限 {a.concurrency}")
    print(f"本金 ${a.account:,.0f} · 目標每筆風險 {a.risk*100:.1f}% = ${slot:,.0f}\n")
    print(f"{'方案':<22} 說明")
    print(f"  {'A 齊頭·29幣':<20} 每筆都 ${slot:,.0f}（小幣其實吃不下）")
    print(f"  {'B 精簡·18幣':<20} 只做容量≥${TRIM_FLOOR:.0f} 的幣，每筆 ${slot:,.0f}")
    print(f"  {'C 容量加權·29幣':<20} 每筆 min(${slot:,.0f}, 該幣容量)\n")
    print("═" * 78)

    schemes = [
        ("A 齊頭·29幣", lambda s: slot, None),
        ("B 精簡·18幣", lambda s: slot, trimmed),
        ("C 容量加權·29幣", lambda s: min(slot, cap.get(s, 0.0)), None),
    ]
    for tag, fn, allowed in schemes:
        taken, rc, ru, peak = walk(rows, fn, a.concurrency, allowed)
        report(tag, taken, rc, ru, peak, a.account)
        print()

    print("讀法：A 的總損益最高但**不可執行** —— 小幣吃不下 ${:.0f}，實際下單會被".format(slot))
    print("      交易所最小量放大或直接被吃掉滑價，那個數字是紙上的。")
    print("      B 可執行但丟掉真有 edge 的小幣訊號。C 兩者兼顧，代價是小幣貢獻縮小。")
    print("      峰值佔用是『這個籃子到底需要多少錢』的答案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
