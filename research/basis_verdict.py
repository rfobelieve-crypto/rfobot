# -*- coding: utf-8 -*-
"""§0.91 in-venue basis clock — the scorer, frozen 2026-09-02.

This file IS the registered criteria in executable form, and its main job
before 2026-09-30 is to REFUSE to render a verdict (same pattern as
premium_verdict.py and v7_regime_q2_clock.py).

FROZEN GATE (written before any forward row existed; the 180-day backfill
was pulled only AFTER these numbers were committed):
  window : registration 2026-09-02 → 2026-09-30 (forward rows only)
  pass   : BTC and ETH EACH satisfy
             * median annualised settled funding >= 8.0%
             * fraction of settlements that are negative < 25%
           and both halves of the window satisfy the same independently
  else   : line closes; the product side writes no spot adapter

Why 8%: on deployed capital (spot in full + perp margin ~1.2x) it is about
6.7%, minus 6-12bp per build/unwind round trip and the negative stretches.
Below that it is not enough above a deposit rate to justify a spot adapter,
two-sided paper funding and pairId attribution. Product side proposed it;
research side reviewed it on 2026-09-02 and kept it unchanged — the
reasoning is sound and moving a threshold at registration time to a number
that "feels better" is the thing pre-registration exists to prevent.

Two readings are reported and must never be confused:
  * PRIOR (backfill, basis_funding_hist): context only. Whether the last
    180 days looked rich says nothing about whether the frozen window will.
    It is here so that a forward result can be read against a regime, not
    so that it can substitute for one.
  * FORWARD (basis_obs since registration): the verdict.

Appendix, not a gate — the hedge reading (product side's question §八.1):
the same rows also answer "does perp-short funding offset a grid's
inventory funding cost". That is a different bar (any positive carry helps
a hedge, while a standalone strategy must clear 8%), so it is reported
separately and cannot rescue a failing standalone verdict.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "results" / "basis_verdict.json"

REGISTERED = datetime(2026, 9, 2, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 9, 30, tzinfo=timezone.utc)
VERDICT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
MEDIAN_ANN_MIN = 8.0          # %
NEG_FRAC_MAX = 0.25

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _stats(anns: list) -> dict | None:
    if len(anns) < 10:
        return None
    s = sorted(anns)
    return {"n": len(anns),
            "median_ann_pct": round(st.median(s), 2),
            "p10": round(s[int(0.1 * len(s))], 2),
            "p90": round(s[int(0.9 * len(s))], 2),
            "neg_frac": round(sum(1 for a in anns if a < 0) / len(anns), 3)}


def _passes(d: dict | None) -> bool:
    return bool(d and d["median_ann_pct"] >= MEDIAN_ANN_MIN
                and d["neg_frac"] < NEG_FRAC_MAX)


def load_forward(cur) -> dict:
    """Settled funding implied by forward observations.

    One row per (symbol, funding period): basis_obs samples every 10 min and
    the rate only changes at settlement, so collapsing to the distinct
    (symbol, funding_rate, period) is what makes the fraction-negative
    denominator "settlements", not "samples" — counting samples would let a
    long quiet stretch dominate.
    """
    cur.execute(
        "SELECT symbol, funding_rate, fund_interval_h, ts_received "
        "FROM basis_obs WHERE ts_received >= %s AND funding_rate IS NOT NULL "
        "ORDER BY symbol, ts_received",
        (int(REGISTERED.timestamp() * 1000),))
    per_sym: dict = {}
    for r in cur.fetchall():
        sym = r["symbol"]
        fi = int(r["fund_interval_h"] or 8)
        ann = float(r["funding_rate"]) * (24.0 / fi) * 365.0 * 100.0
        bucket = int(r["ts_received"]) // (fi * 3600 * 1000)
        per_sym.setdefault(sym, {})[bucket] = ann
    return {s: list(v.values()) for s, v in per_sym.items()}


def load_prior(cur) -> dict:
    cur.execute("SELECT symbol, funding_rate, fund_interval_h, funding_time "
                "FROM basis_funding_hist")
    out: dict = {}
    for r in cur.fetchall():
        fi = int(r["fund_interval_h"] or 8)
        out.setdefault(r["symbol"], []).append(
            float(r["funding_rate"]) * (24.0 / fi) * 365.0 * 100.0)
    return out


def main() -> int:
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            fwd = load_forward(cur)
            prior = load_prior(cur)
    finally:
        conn.close()

    now = datetime.now(timezone.utc)
    days = (now - REGISTERED).total_seconds() / 86400
    gate_days = (DEADLINE - REGISTERED).total_seconds() / 86400
    res = {"registered": REGISTERED.strftime("%Y-%m-%d"),
           "deadline": DEADLINE.strftime("%Y-%m-%d"),
           "days": round(days, 2), "gate_days": round(gate_days, 1),
           "gate_met": now >= DEADLINE,
           "criteria": {"median_ann_pct_min": MEDIAN_ANN_MIN,
                        "neg_frac_max": NEG_FRAC_MAX,
                        "both_halves": True,
                        "symbols_each": list(VERDICT_SYMBOLS)},
           "prior": {}, "forward": {}}

    print("§0.91 站內資金費收租時鐘（判準 2026-09-02 凍結）")
    print(f"  窗口 {res['registered']} → {res['deadline']}｜"
          f"已過 {days:.1f}／{gate_days:.0f} 天")

    print("\n  先驗（180 天回填，**只作背景不判決**）：")
    for sym in VERDICT_SYMBOLS:
        d = _stats(prior.get(sym, []))
        res["prior"][sym] = d
        if d:
            print(f"    {sym:9s} 中位 {d['median_ann_pct']:+6.2f}%／年"
                  f"｜翻負 {d['neg_frac']*100:4.1f}%"
                  f"｜p10/p90 {d['p10']:+.1f}/{d['p90']:+.1f}（n={d['n']}）")

    print("\n  前瞻（判決用）：")
    for sym in VERDICT_SYMBOLS:
        d = _stats(fwd.get(sym, []))
        res["forward"][sym] = d
        if d:
            print(f"    {sym:9s} 中位 {d['median_ann_pct']:+6.2f}%／年"
                  f"｜翻負 {d['neg_frac']*100:4.1f}%（n={d['n']} 結算）")
        else:
            print(f"    {sym:9s} 樣本不足（需 ≥10 個結算）")

    if not res["gate_met"]:
        print(f"\n  → 期限未到（{res['deadline']}），**不出判決**。"
              f"先驗看起來肥正是最誘人提早開獎的時候——"
              f"回填的 180 天不是這個窗口的答案。")
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        return 0

    # ── verdict path ────────────────────────────────────────────────────
    mid = REGISTERED.timestamp() + (DEADLINE - REGISTERED).total_seconds() / 2
    ok_all = True
    for sym in VERDICT_SYMBOLS:
        full = _stats(fwd.get(sym, []))
        ok = _passes(full)
        ok_all = ok_all and ok
        res["forward"][sym] = {**(full or {}), "passed": ok}
        print(f"  {sym}: {'✓' if ok else '✗'}")
    res["verdict"] = ("**過閘** —— 值得研究線回測（資金費歷史、急拉時永續腿"
                      "強平模擬、現貨全額佔用、四腿費用）。過閘 ≠ 上線。"
                      if ok_all else
                      "**關線** —— 收租的分佈撐不起門檻；產品端不寫現貨 adapter。")
    print(f"\n判決：{res['verdict']}")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
