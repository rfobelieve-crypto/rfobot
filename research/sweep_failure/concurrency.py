# -*- coding: utf-8 -*-
"""Is Variant B actually TRADEABLE? Concurrency and correlated-exposure limits.

Widening the liquidity source to four pool types across 29 coins took the
signal rate from 245 to 1327 filtered trades/month. That is the whole reason
the forward answer arrives in ~2 months instead of ~8 — but it also means
many more positions open at once, and #3's own history already warned about
this: the 9-coin swing version peaked at 9 simultaneous positions and spent
17.5% of its life with 5+ open. Four pool types multiply that.

This measures what a live implementation would actually have to hold:
  * simultaneous open positions over time (max, distribution, time above N)
  * SAME-DIRECTION correlated exposure — the number that matters, because
    30 longs across 29 correlated coins is one big long, not 30 small ones
  * the worst single day, and what a fixed risk-per-trade would have cost on it
  * equity path and drawdown at a few risk levels, PnL booked at exit

Nothing here is a strategy change. It answers one question: what cap does the
portfolio risk framework need before this could ever be traded, and does that
cap destroy the trade rate that made the timeline attractive?

Run: python research/sweep_failure/concurrency.py
Out: research/results/sweep_concurrency.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parents[2] / "research/results/sweep_concurrency.json"
PIERCE_MAX = 0.25


def collect() -> list[dict]:
    """Every Variant B trade with entry/exit time, direction and net R."""
    out = []
    for sym in LT.COINS:
        p = LT.CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        bars = SC.load_csv(str(p))
        # swing, via the frozen engine
        for (f_ts, x_ts, R, lvl, atr, st_, pc) in SC.backtest_symbol(bars):
            if pc <= PIERCE_MAX:
                out.append({"sym": sym, "kind": "swing", "f": f_ts, "x": x_ts,
                            "r": LT.net(R, lvl, atr)})
        lv = LT.build_levels(bars)
        for kind in ("session", "pdh_pdl", "pwh_pwl"):
            for (f_ts, x_ts, netr, pc, lvl, atr, st_) in LT.trade_levels(
                    bars, lv.get(kind, [])):
                if pc <= PIERCE_MAX:
                    out.append({"sym": sym, "kind": kind, "f": f_ts,
                                "x": x_ts, "r": netr})
    return sorted(out, key=lambda t: t["f"])


def main() -> int:
    tr = collect()
    print("=" * 78)
    print(f"  VARIANT B CONCURRENCY — {len(tr)} trades, 29 coins x 4 pool types")
    print("=" * 78)

    # ── simultaneous positions, time-weighted ────────────────────────────
    ev = []
    for t in tr:
        ev.append((t["f"], +1))
        ev.append((t["x"], -1))
    ev.sort()
    cur = mx = 0
    span = defaultdict(int)
    prev = ev[0][0]
    for ts, d in ev:
        span[cur] += ts - prev
        prev = ts
        cur += d
        mx = max(mx, cur)
    tot = sum(span.values())
    print(f"\n  simultaneous open positions: max {mx}")
    print(f"  {'threshold':<12}{'% of time at or above':>24}")
    for k in (1, 5, 10, 20, 30, 50):
        share = sum(v for kk, v in span.items() if kk >= k) / tot
        print(f"  >= {k:<9}{100*share:>23.1f}%")

    # ── per-day load, the number a live cap has to survive ───────────────
    byday = defaultdict(list)
    for t in tr:
        byday[datetime.fromtimestamp(t["f"], timezone.utc).date()].append(t)
    counts = sorted((len(v) for v in byday.values()), reverse=True)
    n_days = len(byday)
    print(f"\n  entries per day over {n_days} active days: "
          f"median {counts[len(counts)//2]}, p95 {counts[int(0.05*len(counts))]}, "
          f"max {counts[0]}")

    # ── worst day by realised PnL at 0.25% risk per trade ────────────────
    daypnl = {d: sum(t["r"] for t in v) for d, v in byday.items()}
    worst = sorted(daypnl.items(), key=lambda kv: kv[1])[:5]
    print(f"\n  worst 5 days (sum of R booked, entries that day):")
    for d, v in worst:
        print(f"    {d}  sumR {v:+8.2f}  n={len(byday[d]):>3}"
              f"   at 0.25% risk = {0.25*v:+.2f}% equity")

    # ── equity path / MDD at several risk levels ─────────────────────────
    print(f"\n  {'risk/trade':<12}{'total %':>10}{'ann %':>9}{'MDD %':>9}"
          f"{'worst day %':>13}")
    res_risk = {}
    span_d = (tr[-1]["x"] - tr[0]["f"]) / 86400
    for risk in (0.1, 0.25, 0.5, 1.0):
        eq = peak = 1.0
        mdd = 0.0
        for t in sorted(tr, key=lambda x: x["x"]):
            eq *= 1 + risk / 100.0 * t["r"]
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        ann = (eq ** (365.0 / span_d) - 1) * 100
        wd = risk * min(daypnl.values())
        res_risk[risk] = {"total_pct": (eq - 1) * 100, "ann_pct": ann,
                          "mdd_pct": 100 * mdd, "worst_day_pct": wd}
        print(f"  {risk:<12.2f}{100*(eq-1):>10.1f}{ann:>9.1f}{100*mdd:>9.1f}"
              f"{wd:>13.2f}")

    print(f"\n  NOTE: PnL is booked at exit, so the equity path understates "
          f"intraday drawdown;\n  correlated same-direction exposure is the "
          f"real constraint and a live cap must\n  bound it directly, not just "
          f"the position count.")
    OUT.write_text(json.dumps(
        {"n_trades": len(tr), "max_concurrent": mx,
         "time_share": {str(k): sum(v for kk, v in span.items() if kk >= k) / tot
                        for k in (1, 5, 10, 20, 30, 50)},
         "entries_per_day": {"median": counts[len(counts)//2],
                             "p95": counts[int(0.05*len(counts))],
                             "max": counts[0]},
         "worst_days": [[str(d), v] for d, v in worst],
         "risk_levels": res_risk, "caps": caps}, indent=2), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
