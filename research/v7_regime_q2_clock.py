# -*- coding: utf-8 -*-
"""§0.60 Q2 pre-registration clock — "in TREND_UP, take DOWN signals only".

This file IS the pre-registration. The rule, the sample floor, the gate and
the criteria are frozen in the constants below; the script's only job is to
report progress and to REFUSE to render a verdict before the gate is met.
That refusal is the point: the hypothesis already survived one adverse
sample, so the failure mode to guard against is peeking until the number
looks right.

HOW THIS HYPOTHESIS GOT HERE, in order, because the order is what makes the
bar legitimate:

  1. Q2 found `TREND_UP x UP` at 48.1% (n=77) against 59.3% overall — the
     only cell negative on BOTH win rate and bps. It was, however, the most
     extreme of EIGHT cells, uncorrected.
  2. The first new sample after the hypothesis was formed CONTRADICTED it:
     11 blocked signals returned 55% WR / +23.5 bps, three of them
     +153.8 / +107.0 / +67.9 bps from the 08-19~21 run.
  3. Three explanations were pre-listed and two were killed:
       #2 fat right tail  — dead. Trimming 10% each side moved the cell
          -8.0 -> -9.9 bps and it stayed rank 1 of 8. Its total is tail
          +932 / body -1550: the body loses, the tail only partly repays.
       #1 trend strength unstratified — dead. Splitting the cell by the
          concurrent 24h gain gave 44.0 / 48.0 / 51.9% WR and -5.2 / -9.7 /
          -9.1 bps: no monotonicity, all three negative. The 08-19~21 run
          sits in the most violent third, which still averages -9.1.
       #3 luck — the only survivor, and consistent with Wilson [37.3, 59.0].

So the rule is re-proposed, but the bar is raised and the counter-example
is carried forward rather than forgotten. See KNOWN_COUNTEREXAMPLE.

WHAT PRE-REGISTRATION ACTUALLY BUYS HERE: the original finding picked the
most extreme of 8 cells. Naming that single cell in advance and testing it
on data that does not yet exist RESOLVES the multiplicity — it does not
merely disclose it. That is the whole function of this file, and it is void
the moment the cell, the gate or the criteria are edited to fit a result.

CRITERION CORRECTION (2026-08-26): the TODO first wrote the first-tier bar
as "the cell's CI LOW stays below overall". That is not a bar at all — any
wide interval satisfies it. The bar that made the original finding "barely
stand" was the CI HIGH: 59.0% < 59.3% overall. Corrected here and in TODO.

Scope: SIGNAL layer (tracked_signals.correct, 4h TWAP direction). Not trade
P&L — different question, never conflate. V7's real execution now runs on
the product side (Bitget), so a PASS is a proposal to hand over, not an
executor change on this side.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from research.v7_regime_axis import load, wilson                # noqa: E402

OUT = ROOT / "research" / "results" / "v7_regime_q2_clock.json"

# ── FROZEN 2026-08-26. Editing anything below voids the pre-registration ──
RULE = "在 TREND_UP 格只接受 DOWN 訊號（擋掉 UP 訊號）"
CELL, SIDE = "TREND_UP", "UP"          # the single pre-named cell
SAMPLE_FLOOR = int(datetime(2026, 8, 26, tzinfo=timezone.utc).timestamp())
GATE_N = 60                            # fresh signals IN THE NAMED CELL
GATE_DAYS = 30                         # and at least this much wall time
KNOWN_COUNTEREXAMPLE = {
    "window": "2026-08-17 ~ 2026-08-26", "n": 11, "wr": 55.0, "bps": 23.5,
    "note": "已用掉的樣本，不得再計入判決；列此以免事後遺忘",
}
LIMITS = [
    "原始發現是 8 格中最極端的一格，多重比較未校正——預註冊單格即為其解方",
    "假設在一批反向新樣本之後仍提出，該批 11 筆列為已知反例",
    "訊號層結論不等於交易層獲利；出場端另有貢獻（§0.51b）",
]
# ── end frozen block ──────────────────────────────────────────────────────


def _tier1(cell_k, cell_n, overall_wr):
    """一級：該格 Wilson CI 上緣 < 全體點估計。

    上緣（不是下緣）才是門檻——下緣低於全體對任何寬區間幾乎必然成立。
    這條正是原始發現「勉強站住」的那條（59.0% < 59.3%）。
    """
    ci = wilson(cell_k, cell_n)
    return (ci is not None and 100 * ci[1] < overall_wr), ci


def main() -> int:
    rows = [r for r in load() if r["ts"] >= SAMPLE_FLOOR]
    cell = [r for r in rows if r["cell"] == CELL and r["dir"] == SIDE]
    n, k = len(cell), sum(r["ok"] for r in cell)
    now = int(datetime.now(timezone.utc).timestamp())
    days = (now - SAMPLE_FLOOR) / 86400

    print("§0.60 Q2 預註冊時鐘（2026-08-26 凍結）")
    print(f"  規則：{RULE}")
    print(f"  樣本 floor：2026-08-26（此前全部已用掉，含 "
          f"{KNOWN_COUNTEREXAMPLE['n']} 筆已知反例）")
    print(f"  閘門：該格新樣本 n≥{GATE_N} 且經過 ≥{GATE_DAYS} 天\n")

    print(f"  已累積：全體 Strong n={len(rows)}｜"
          f"{CELL}×{SIDE} n={n}／{GATE_N}｜經過 {days:.1f}／{GATE_DAYS} 天")

    # honest clock projection from the observed rate, not from a stale one
    if days >= 1 and n > 0:
        eta = (GATE_N - n) / (n / days)
        print(f"  依實測速率推估：還需約 {eta:.0f} 天"
              f"（該格 {n/days:.2f} 筆/天）")
    elif days >= 3:
        print("  該格尚無樣本——速率無法推估。開火率本身會隨行情變動，"
              "不要用歷史佔比硬推時程")

    gate_met = n >= GATE_N and days >= GATE_DAYS
    res = {"rule": RULE, "cell": f"{CELL}x{SIDE}", "n": n, "gate_n": GATE_N,
           "days": round(days, 1), "gate_days": GATE_DAYS,
           "gate_met": gate_met, "limits": LIMITS,
           "known_counterexample": KNOWN_COUNTEREXAMPLE}

    if not gate_met:
        # interim OBSERVATION only, and only once there is something to see.
        if n >= 30:
            ovr = 100 * sum(r["ok"] for r in rows) / len(rows) if rows else 0
            print(f"\n  ── 期中觀察（n≥30，**不是判決**）──")
            print(f"     該格 {100*k/n:.1f}%（{k}/{n}）vs 全體 {ovr:.1f}%")
            print("     方向若與假設相反且持續，判決日照樣照規矩跑，"
                  "不提前收也不提前棄")
        print(f"\n  → 閘門未達，**不出判決**。這個拒絕是刻意的："
              f"假設已經歷一次反向樣本，最該防的就是看到好看的數字就收。")
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nwritten {OUT.name}")
        return 0

    # ── verdict path (only reachable once the gate is met) ────────────────
    ovr_k = sum(r["ok"] for r in rows)
    ovr = 100 * ovr_k / len(rows)
    t1, ci = _tier1(k, n, ovr)
    cb = [r["bps"] for r in cell if r["bps"] is not None]
    kept = [r["bps"] for r in rows
            if r["bps"] is not None and not (r["cell"] == CELL
                                             and r["dir"] == SIDE)]
    allb = [r["bps"] for r in rows if r["bps"] is not None]
    t2 = bool(cb) and st.mean(cb) < 0 and st.mean(kept) > st.mean(allb)
    mid = SAMPLE_FLOOR + (now - SAMPLE_FLOOR) // 2
    h = [[r for r in cell if r["ts"] < mid], [r for r in cell if r["ts"] >= mid]]
    hw = [100 * sum(x["ok"] for x in g) / len(g) if g else None for g in h]
    halves_ok = all(w is not None and w < ovr for w in hw)

    print(f"\n  ── 判決 ──")
    print(f"  該格 {100*k/n:.1f}%（{k}/{n}）CI [{100*ci[0]:.1f},{100*ci[1]:.1f}]"
          f"  全體 {ovr:.1f}%")
    print(f"  一級 CI 上緣 {100*ci[1]:.1f} < 全體 {ovr:.1f}：{'✓' if t1 else '✗'}")
    print(f"  二級 該格 bps<0 且濾掉後全體改善："
          f"{'✓' if t2 else '✗'}"
          + (f"（{st.mean(cb):+.1f} bps；{st.mean(allb):+.1f}→{st.mean(kept):+.1f}）"
             if cb else ""))
    print(f"  兩半皆低於全體：{'✓' if halves_ok else '✗'}"
          + (f"（{hw[0]:.1f}% / {hw[1]:.1f}%）"
             if all(w is not None for w in hw) else ""))
    verdict = ("PASS —— 交付產品端作為 V7 條件式濾網的提案"
               if (t1 and t2 and halves_ok) else
               "FAIL —— Q2 到此為止，不得再以更寬的判準重提")
    print(f"\n  判決：{verdict}")
    print("  限制（判決時一併考慮）：")
    for x in LIMITS:
        print(f"    · {x}")
    res.update({"cell_wr": round(100 * k / n, 2), "overall_wr": round(ovr, 2),
                "ci": [round(100 * ci[0], 2), round(100 * ci[1], 2)],
                "tier1": t1, "tier2": t2, "halves_ok": halves_ok,
                "verdict": verdict})
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nwritten {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
