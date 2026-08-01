# -*- coding: utf-8 -*-
"""Faster evidence for the V7 terrain filters — WITHOUT lowering the bar.

The frozen trigger needs +60 new Strong signals; at the current rate
(12-28/month) that is ~3 months. These two tests buy evidence today from
populations the four dims were NOT derived on. Neither replaces the
forward trigger; both make the eventual decision better informed.

  TEST 1 — Moderate tier. The dims were found on Strong only. Moderate
  fires 3-5x more often (67-77/month vs 12-28). If terrain is a property
  of PRICE LOCATION it must show up on Moderate too; if it only exists
  on Strong, that is a red flag about the finding, not about Moderate.
  Out-of-tier generality, available immediately.

  TEST 2 — clean walk-forward OOS decode. gate_a_revalidate_clean_oos
  holds fold-model predictions with a strong/none label produced without
  look-ahead. Its signals span a different (longer, multi-fold) slice
  than the live tracked_signals cohort, so the dims meet bars the live
  book never contained.

Pre-registered before running: SAME thresholds as the confirmed dims
(follow-context, wall <=1.4 ATR, support <=1.8 ATR, density >=3 vs <=1),
no re-fitting, no threshold search, all four dims reported whatever they
do. A dim that flips sign in either population is a problem for that
dim, and gets recorded as one.

Run: python research/terrain_fast_evidence.py
Out: research/results/terrain_fast_evidence.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import pandas as pd  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shared.db import get_db_conn  # noqa: E402
from v7_price_location import pool_lifecycle  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_fast_evidence.json"
WALL, SUP, RANGE_ATR = 1.4, 1.8, 3.0
OOS = ROOT / "research/results/gate_a_revalidate_clean_oos.parquet"


def terrain_ctx(bars):
    """Shared terrain machinery: returns (ts2i, atr, close, pools, raids)."""
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = pool_lifecycle(bars)
    by_hh = defaultdict(list)
    for r in raids_with_fill("BTC"):
        by_hh[r["ts"] // 3600].append(r["side"])
    return ts2i, atr, cl, pools, by_hh


def annotate(ts, direction, correct, ctxs):
    ts2i, atr, cl, pools, by_hh = ctxs
    j = ts2i.get(ts)
    if j is None or atr[j] in (None, 0):
        return None
    up = direction == "UP"
    c = cl[j]
    above = [p[2] for p in pools if p[0] <= j
             and (p[1] is None or p[1] > j) and p[2] > c]
    below = [p[2] for p in pools if p[0] <= j
             and (p[1] is None or p[1] > j) and p[2] < c]
    ahead = ((min(above) - c) / atr[j] if up and above else
             (c - max(below)) / atr[j] if (not up) and below else None)
    behind = ((c - max(below)) / atr[j] if up and below else
              (min(above) - c) / atr[j] if (not up) and above else None)
    na = 0
    for p in pools:
        if p[0] <= j and (p[1] is None or p[1] > j):
            d_ = (p[2] - c) / atr[j]
            if 0 < (d_ if up else -d_) <= RANGE_ATR:
                na += 1
    ctx = "none"
    for k in range(0, 5):
        sides = by_hh.get(ts // 3600 - k)
        if sides:
            ctx = "fade" if ((sides[0] == 1 and not up)
                             or (sides[0] == -1 and up)) else "follow"
            break
    return {"ts": ts, "dir": direction, "c": int(correct), "ahead": ahead,
            "behind": behind, "na": na, "ctx": ctx}


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def cell(g):
    return f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"


DIMS = (
    ("D1 追突破 veto", lambda r: r["ctx"] == "follow",
     lambda r: r["ctx"] != "follow", "追突破", "其他", -1),
    ("D2 前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL,
     lambda r: r["ahead"] is not None and r["ahead"] > WALL, "有牆", "跑道淨", -1),
    ("D3 背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP,
     lambda r: r["behind"] is None or r["behind"] > SUP, "有墊背", "背後空", +1),
    ("D5 池子密度", lambda r: r["na"] <= 1, lambda r: r["na"] >= 3, "疏", "密", +1),
)


def report(title, rows, res):
    rows = sorted(rows, key=lambda r: r["ts"])
    n = len(rows)
    half = n // 2
    print(f"\n  ── {title}  n={n} · 整體 {wr(rows):.0f}%")
    res[title] = {"n": n, "base_wr": wr(rows)}
    for name, pa, pb, la, lb, want in DIMS:
        a_ = [r for r in rows if pa(r)]
        b_ = [r for r in rows if pb(r)]
        gap = wr(a_) - wr(b_) if len(a_) >= 15 and len(b_) >= 15 else None
        h1 = h2 = None
        for tag, seg in (("H1", rows[:half]), ("H2", rows[half:])):
            aa = [r for r in seg if pa(r)]
            bb = [r for r in seg if pb(r)]
            v = wr(aa) - wr(bb) if len(aa) >= 12 and len(bb) >= 12 else None
            if tag == "H1":
                h1 = v
            else:
                h2 = v
        ok = gap is not None and ((gap > 0) == (want > 0))
        halves = (f"H1 {h1:+.0f}/H2 {h2:+.0f}"
                  if h1 is not None and h2 is not None else "halves thin")
        print(f"    {name:<14} {la} {cell(a_):<12} {lb} {cell(b_):<12}"
              + (f" gap {gap:+.0f}pp  {halves}"
                 f"  {'方向一致 ✓' if ok else '⚠ 翻轉'}"
                 if gap is not None else "  樣本不足"))
        res[title][name] = {"gap": gap, "h1": h1, "h2": h2,
                            "na": len(a_), "nb": len(b_), "sign_ok": ok}


def main() -> int:
    print("=" * 78)
    print("  快速取證：地形四維在「非推導母體」上還成不成立")
    print("  ※ 不取代 forward 扳機；這是把決策做得更有依據，不是把門檻放低")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ctxs = terrain_ctx(bars)
    res = {}

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT signal_time, direction, strength, correct "
                        "FROM tracked_signals WHERE correct IS NOT NULL "
                        "AND strength IN ('Strong','Moderate') "
                        "ORDER BY signal_time")
            sigs = cur.fetchall()
    finally:
        conn.close()
    for tier in ("Strong", "Moderate"):
        rows = []
        for s in sigs:
            if s["strength"] != tier:
                continue
            ts = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
            r = annotate(ts, s["direction"], s["correct"], ctxs)
            if r:
                rows.append(r)
        report(f"live {tier} tier", rows, res)

    if OOS.exists():
        df = pd.read_parquet(OOS)
        df = df[df["strong"] != "none"]
        rows = []
        for dt, r in df.iterrows():
            ts = int(dt.timestamp())
            up = r["pred_ret"] > 0
            correct = int((r["y"] > 0) == up)
            a = annotate(ts, "UP" if up else "DOWN", correct, ctxs)
            if a:
                rows.append(a)
        report("clean WF OOS 訊號", rows, res)
    else:
        print("\n  （找不到 clean WF OOS parquet，跳過測試 2）")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
