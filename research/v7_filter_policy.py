# -*- coding: utf-8 -*-
"""Filter policy tiers — the frequency/quality menu for adopting the three
verified structural conditions (2026-08-02, operator: 往這塊前進, 勢必會
減少訊號).

Tiers (cumulative, pre-stated):
  T0 現狀        every Strong signal
  T1 +追突破veto  drop signals that chase a <=4h raid break
  T2 +前方牆veto  additionally drop signals with a wall < ~1.4 ATR ahead
                  (the in-sample near-tercile boundary, rounded)
  T3 +背靠支撐    additionally REQUIRE support within ~1.8 ATR behind
                  (near-tercile boundary, rounded)

Each tier: kept share, WR, directional 4h return, signals/month — the
menu the operator picks a live tier from AFTER the forward trigger fires.
This file changes nothing in production.

Run: python research/v7_filter_policy.py
Out: research/results/v7_filter_policy.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from v7_price_location_verify import build_rows  # noqa: E402
from shared.db import get_db_conn  # noqa: E402
from datetime import timezone  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_filter_policy.json"
WALL_ATR = 1.4      # rounded near-tercile boundary (in-sample 1.39)
SUPPORT_ATR = 1.8   # rounded near-tercile boundary (in-sample 1.79)


def enrich_ret(rows):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, actual_return_4h "
                "FROM tracked_signals WHERE strength='Strong' "
                "AND correct IS NOT NULL")
            m = {}
            for r in cur.fetchall():
                ts = int(r["signal_time"].replace(
                    tzinfo=timezone.utc).timestamp())
                sgn = 1 if r["direction"] == "UP" else -1
                m[ts] = (float(r["actual_return_4h"]) * sgn
                         if r["actual_return_4h"] is not None else None)
    finally:
        conn.close()
    for r in rows:
        r["ret"] = m.get(r["ts"])
    return rows


def main() -> int:
    print("=" * 78)
    print("  FILTER POLICY TIERS — 頻率/品質菜單（production 不動, 供扳機後選檔）")
    print("=" * 78)
    rows = enrich_ret(build_rows())
    n = len(rows)
    span_mo = (rows[-1]["ts"] - rows[0]["ts"]) / 86400 / 30.4

    tiers = {
        "T0 現狀": lambda r: True,
        "T1 +追突破veto": lambda r: r["ctx"] != "follow",
        "T2 +前方牆veto": lambda r: r["ctx"] != "follow"
        and not (r["ahead"] is not None and r["ahead"] <= WALL_ATR),
        "T3 +需背靠支撐": lambda r: r["ctx"] != "follow"
        and not (r["ahead"] is not None and r["ahead"] <= WALL_ATR)
        and (r["behind"] is not None and r["behind"] <= SUPPORT_ATR),
    }
    res = {}
    print(f"  樣本 {n} 筆 Strong / {span_mo:.1f} 個月\n")
    print(f"  {'檔位':<14}{'保留':>7}{'佔比':>7}{'筆/月':>7}{'勝率':>7}{'4h方向報酬':>10}")
    for name, pred in tiers.items():
        g = [r for r in rows if pred(r)]
        wr = 100 * sum(r["c"] for r in g) / len(g)
        xs = [r["ret"] for r in g if r["ret"] is not None]
        rt = 100 * sum(xs) / len(xs)
        pm = len(g) / span_mo
        print(f"  {name:<14}{len(g):>7}{100*len(g)/n:>6.0f}%{pm:>7.1f}"
              f"{wr:>6.0f}%{rt:>+9.2f}%")
        res[name] = {"n": len(g), "share_pct": round(100 * len(g) / n, 1),
                     "per_month": round(pm, 1), "wr": round(wr, 1),
                     "ret_pct": round(rt, 2)}
    # halves stability of the tier WRs (the menu must not be an H2 artifact)
    half = n // 2
    print("\n  [halves] 各檔位勝率 前半/後半")
    for name, pred in tiers.items():
        parts = []
        for seg in (rows[:half], rows[half:]):
            g = [r for r in seg if pred(r)]
            parts.append(f"{100*sum(r['c'] for r in g)/len(g):.0f}%(n={len(g)})")
        print(f"  {name:<14}" + " / ".join(parts))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
