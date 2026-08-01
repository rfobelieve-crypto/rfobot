# -*- coding: utf-8 -*-
"""V7 signals conditioned on raid context — the mirror of pred_align.

User idea (2026-08-02): V7's judgment uses no price structure / location.
History: the FEATURE direction was tested 2026-06-01 (liquidity-distance
features into the ensemble: +0.0006 AUC, redundancy — mistake.md) and is
dead. This tests the EVENT direction instead: does a live V7 Strong
signal's win rate depend on what the liquidity map was doing when it
fired?

Buckets per signal (raid = any BTC pool sweep within the prior 4 hours,
matching the V7 horizon):
  無獵取   no raid in the prior 4h
  fade順   signal fades the raid (DOWN after a high sweep / UP after low)
           — same side as the sweep-failure strategy
  follow逆 signal FOLLOWS the raid's break direction — fighting the
           raid-reversal dynamics this whole research line documents

Sample: tracked_signals Strong with realized outcomes — genuinely live
forward signals; only the raid overlay is retro-computed (causal, bar
labels on both sides share the open-time convention).

Run: python research/v7_raid_context.py
Out: research/results/v7_raid_context.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from shared.db import get_db_conn  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_raid_context.json"
LOOKBACK_H = 4


def main() -> int:
    print("=" * 78)
    print("  V7 x RAID CONTEXT — Strong 訊號勝率按獵取情境分桶（pred_align 的鏡像）")
    print("=" * 78)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, correct FROM tracked_signals "
                "WHERE strength='Strong' AND correct IS NOT NULL "
                "ORDER BY signal_time")
            sigs = cur.fetchall()
    finally:
        conn.close()
    by_hh: dict[int, list[int]] = {}
    for r in raids_with_fill("BTC"):
        by_hh.setdefault(r["ts"] // 3600, []).append(r["side"])

    def ctx(ts_h: int, direction: str) -> str:
        for k in range(0, LOOKBACK_H + 1):
            sides = by_hh.get(ts_h - k)
            if sides:
                s = sides[0]
                fade = ((s == 1 and direction == "DOWN")
                        or (s == -1 and direction == "UP"))
                return "fade順" if fade else "follow逆"
        return "無獵取"

    rows = []
    for s in sigs:
        ts_h = int(s["signal_time"].replace(
            tzinfo=timezone.utc).timestamp()) // 3600
        rows.append((ctx(ts_h, s["direction"]), int(s["correct"])))
    n = len(rows)
    res = {"n": n, "base_wr": round(100 * sum(c for _, c in rows) / n, 1)}
    print(f"  Strong 訊號 n={n} · 整體勝率 {res['base_wr']}%\n")
    half = n // 2
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        parts, rec = [], {}
        for b in ("無獵取", "fade順", "follow逆"):
            g = [c for x, c in seg if x == b]
            if len(g) >= 15:
                wr = 100 * sum(g) / len(g)
                rec[b] = {"n": len(g), "wr": round(wr, 1)}
                parts.append(f"{b} {wr:.0f}% (n={len(g)})")
            else:
                parts.append(f"{b} thin(n={len(g)})")
        res[tag] = rec
        print(f"  [{tag}] " + " | ".join(parts))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  讀法: follow逆（追著獵取的突破方向開火）在兩半皆為最差桶 →")
    print("  「獵取逆向 veto」進十月清單（V7 側濾網候選）；不動 production。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
