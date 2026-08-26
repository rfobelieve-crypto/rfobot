# -*- coding: utf-8 -*-
"""Is the LONG's opp_signal exit actually starved? — TODO §0.63b.

§0.63 measured STRONG DOWN firing and found 1.54%/bar in TREND_UP against
13-18% elsewhere (post-DECODE_EPOCH), which would mean a LONG held through
an uptrend loses its reverse-signal exit.

BUT THE LIVE BOT DOES NOT REQUIRE STRONG. jarvis `src/v7bot.js:108`:

    const opp = !!sig && sig.direction !== 'NEUTRAL' && ...

Any reverse reading fires it — Moderate counts. So §0.63 measured a
NARROWER event than the one that actually exits the position, and its
number cannot be read as the exit's coverage. Measuring the trigger the
code actually uses is the whole point of this file.

Second thing this checks: the OKX-era third exit, `conviction_decay`
(OKX_CONVICTION_DECAY_BARS=2, built in 2026-07-25 precisely because
opp_signal starves at low signal frequency), did NOT migrate to the
product side. So today a LONG in an uptrend has exactly two exits: the
3xATR trail and whatever coverage opp gives. That makes opp's real
coverage an operational question, not a research curiosity — V7 trades
real money on that book.

Reported per frozen regime cell, post-DECODE_EPOCH only (CLAUDE.md core
principle 7 — the pre-fix era is a different machine and pooling the two
is what made §0.63's first run give the wrong answer):

  * STRONG DOWN per bar        — what §0.63 measured
  * ANY DOWN per bar           — what the bot actually reacts to
  * expected bars to first opp — 1/rate, i.e. how long a LONG would
    typically wait for this exit to become available

Pre-committed reading:
  * ANY-DOWN rate in TREND_UP comparable to other cells -> the exit is
    NOT starved; §0.63's alarm was an artefact of measuring Strong only
  * ANY-DOWN rate still an order of magnitude lower -> the concern
    survives the correction and the missing conviction_decay becomes a
    real gap to close
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sweep_core as SC                                    # noqa: E402
from research.crowd_battery2 import adx_state              # noqa: E402
from shared.db import get_db_conn                          # noqa: E402

CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "v7_opp_coverage.json"
EPOCH = int(datetime(2026, 8, 12, 16, tzinfo=timezone.utc).timestamp())
CELLS = ("RANGING", "TREND_UP", "TREND_DOWN", "NEUTRAL")
LB = 24


def main() -> int:
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    c = [b[SC.C] for b in bars]
    adx = adx_state(bars)
    cell = {}
    for i in range(LB, len(bars)):
        ts = bars[i][0]
        lab = adx.get(ts // 3600 * 3600)
        if lab is None:
            continue
        cell[ts] = ("RANGING" if lab == "RANGING" else
                    "NEUTRAL" if lab != "TRENDING" else
                    ("TREND_UP" if c[i] / c[i - LB] - 1 > 0 else "TREND_DOWN"))

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, strength FROM tracked_signals "
                "WHERE direction IN ('UP','DOWN') ORDER BY signal_time")
            rows = cur.fetchall()
    finally:
        conn.close()

    hit = defaultdict(lambda: {"strong_dn": 0, "any_dn": 0,
                               "strong_up": 0, "any_up": 0})
    hrs = []
    for r in rows:
        ts = int(r["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        if ts < EPOCH:
            continue
        h = ts // 3600 * 3600
        cl = cell.get(h)
        if cl is None:
            continue
        hrs.append(h)
        d = "dn" if r["direction"] == "DOWN" else "up"
        hit[cl][f"any_{d}"] += 1
        if r["strength"] == "Strong":
            hit[cl][f"strong_{d}"] += 1

    if not hrs:
        print("解碼修法後尚無樣本")
        return 0
    lo, hi = min(hrs), max(hrs)
    nb = defaultdict(int)
    for h, cl in cell.items():
        if lo <= h <= hi:
            nb[cl] += 1

    span_d = (hi - lo) / 86400
    print("§0.63b 多單的反向訊號出場真的餓死了嗎")
    print(f"  只取解碼修法之後（2026-08-12 16:00 起，{span_d:.1f} 天）")
    print("  live bot 觸發條件是 direction != NEUTRAL —— **不要求 Strong**\n")
    print(f"{'cell':12} {'bar':>5} {'Strong↓':>8} {'任何↓':>7} "
          f"{'Strong↓/bar':>12} {'任何↓/bar':>11} {'平均等待':>10}")
    res = {"span_days": round(span_d, 1), "cells": {}}
    for cl in CELLS:
        b = nb.get(cl, 0)
        if not b:
            continue
        s, a = hit[cl]["strong_dn"], hit[cl]["any_dn"]
        sr, ar = 100 * s / b, 100 * a / b
        wait = (b / a) if a else float("inf")
        mark = "  ←" if cl == "TREND_UP" else ""
        print(f"{cl:12} {b:5d} {s:8d} {a:7d} {sr:11.2f}% {ar:10.2f}% "
              f"{('%.0f 小時' % wait) if a else '   從未':>10}{mark}")
        res["cells"][cl] = {"bars": b, "strong_down": s, "any_down": a,
                            "strong_rate": round(sr, 2),
                            "any_rate": round(ar, 2),
                            "mean_wait_bars": None if not a else round(wait, 1)}

    tu = res["cells"].get("TREND_UP")
    others = [v["any_rate"] for k, v in res["cells"].items() if k != "TREND_UP"]
    if tu and others:
        med = sorted(others)[len(others) // 2]
        rel = tu["any_rate"] / med if med else 0
        print(f"\n  TREND_UP 的『任何↓』開火率 {tu['any_rate']:.2f}%/bar "
              f"vs 其他格中位 {med:.2f}%/bar → {rel:.2f}×")
        print(f"  對照 Strong 口徑：{tu['strong_rate']:.2f}% "
              f"（§0.63 用的就是這個，被擔心的也是這個數字）")
        starved = rel < 0.5
        v = ("**擔心成立**：即使放寬到任何反向讀數，上升段仍明顯偏低——"
             "多單在上升趨勢裡實質只剩 3×ATR 移動停損一道出場，"
             "而 conviction_decay 沒有隨遷移帶過來。"
             if starved else
             "**擔心不成立（在正確口徑下）**：live bot 不要求 Strong，"
             "放寬到任何反向讀數之後上升段的開火率並不偏低。"
             "§0.63 的警報來自量了一個比實際觸發條件更窄的事件。")
        print(f"\n判讀：{v}")
        res["verdict"] = v
        res["trend_up_vs_median"] = round(rel, 3)

    res["live_exits"] = ["trail(3xATR)", "opp(any reverse)", "time"]
    res["conviction_decay_migrated"] = False
    print("\n  live 出場盤點（jarvis src/v7bot.js）："
          "trail(3×ATR) / opp(任何反向) / time")
    print("  ⚠ conviction_decay **未隨遷移帶過來**"
          "（OKX 時代的第三道，正是為 opp 餓死而設計的）")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
