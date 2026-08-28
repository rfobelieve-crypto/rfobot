# -*- coding: utf-8 -*-
"""§0.75 arbitrage clock — the verdict scorer, frozen 2026-08-28.

This file IS the registered criteria in executable form, and its main job
before day 7 is to REFUSE to render a verdict (the v7_regime_q2_clock
pattern: the failure mode to guard against is peeking until a number looks
right, and an EV-first line with zero fees is exactly where a good-looking
half-day of data would tempt an early call).

FROZEN GATE (written before more than two minutes of data existed):
  after >= 7 FULL days of recording, the line proceeds to the engineering
  gate iff there EXISTS a threshold band with
    * fee-net round-trip >= 1.0 bps        (fees are 0+0 on this pair, so
                                            net == raw executable room)
    * fired on average >= 10 times/day
    * BOTH halves of the recording satisfy the above independently
  else the line closes. The band is NOT swept for the best number: the
  candidate band is analyze.py's suggestion methodology (p90 of executable
  room), applied identically to both halves.

Progress source: ../entropy-arb/logs/minutes.csv (outside this repo — the
third-party clone carries the recorder; THIS repo carries the judgment).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CSV = ROOT.parent / "entropy-arb" / "logs" / "minutes.csv"
OUT = ROOT / "research" / "results" / "arb_premium_verdict.json"
START = datetime(2026, 8, 28, 10, 28, tzinfo=timezone.utc)   # first minute
GATE_DAYS = 7
NET_BPS_MIN = 1.0
FIRES_PER_DAY_MIN = 10.0


def load():
    rows = []
    with open(CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "ts": int(r["minute_ts"]),
                    "sell_max": float(r["sell_edge_max_bps"]),
                    "buy_max": float(r["buy_edge_max_bps"]),
                    "n": int(r["samples"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def side_stats(rows, key):
    """p90 candidate band per analyze.py's methodology, then fire counts."""
    vals = sorted(x[key] for x in rows)
    if not vals:
        return None
    p90 = vals[int(0.9 * len(vals))]
    band = max(p90, NET_BPS_MIN)
    days = max((rows[-1]["ts"] - rows[0]["ts"]) / 86400, 1e-9)
    fires = sum(1 for x in rows if x[key] >= band)
    return {"p90_bps": round(p90, 3), "band_bps": round(band, 3),
            "fires": fires, "fires_per_day": round(fires / days, 1)}


def main() -> int:
    if not CSV.exists():
        print("§0.75 時鐘：錄製檔不存在 —— freshness board 應該已經在響")
        return 1
    rows = load()
    now = datetime.now(timezone.utc)
    days = (now - START).total_seconds() / 86400
    print("§0.75 兩場館套利時鐘（判準 2026-08-28 凍結）")
    print(f"  配對 SNDK · Entropy vs lighter-rh（taker 0+0 bps）")
    print(f"  已錄 {len(rows)} 分鐘｜經過 {days:.1f}／{GATE_DAYS} 天")

    res = {"minutes": len(rows), "days": round(days, 2),
           "gate_days": GATE_DAYS, "gate_met": days >= GATE_DAYS}
    if days < GATE_DAYS:
        # interim observation only, and only the distribution — no verdict
        if len(rows) >= 60:
            s = side_stats(rows, "sell_max")
            b = side_stats(rows, "buy_max")
            print(f"\n  期中觀察（**不是判決**）：")
            print(f"    sell 側 p90 可成交空間 {s['p90_bps']:+.2f} bps")
            print(f"    buy  側 p90 可成交空間 {b['p90_bps']:+.2f} bps")
            res["interim"] = {"sell": s, "buy": b}
        print(f"\n  → 閘門未達，**不出判決**。零費率＋看起來很肥的半天資料"
              f"正是最誘人提早開獎的組合——判準凍結的意義就在此刻。")
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        return 0

    # ── verdict path ────────────────────────────────────────────────────
    mid = rows[len(rows) // 2]["ts"]
    halves = ([r for r in rows if r["ts"] < mid],
              [r for r in rows if r["ts"] >= mid])
    verdict_sides = {}
    ok_any = False
    for key, lab in (("sell_max", "sell"), ("buy_max", "buy")):
        full = side_stats(rows, key)
        h = [side_stats(hh, key) for hh in halves]
        passed = (full["band_bps"] >= NET_BPS_MIN
                  and full["fires_per_day"] >= FIRES_PER_DAY_MIN
                  and all(x["fires_per_day"] >= FIRES_PER_DAY_MIN
                          and x["band_bps"] >= NET_BPS_MIN for x in h))
        verdict_sides[lab] = {"full": full, "halves": h, "passed": passed}
        ok_any = ok_any or passed
        print(f"  {lab}: 帶 {full['band_bps']:.2f} bps、"
              f"{full['fires_per_day']:.1f} 次/天、"
              f"兩半 {h[0]['fires_per_day']:.1f}/{h[1]['fires_per_day']:.1f}"
              f" → {'✓' if passed else '✗'}")
    v = ("**過閘** —— 進工程閘門討論（審計下單路徑、統一風控、資金拆分）。"
         "注意這只證明溢價存在，不證明抓得到它。"
         if ok_any else
         "**關線** —— 扣費後的可成交空間撐不起門檻；一週結案，成本≈零。")
    print(f"\n判決：{v}")
    res.update({"sides": verdict_sides, "verdict": v})
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
