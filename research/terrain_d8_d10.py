# -*- coding: utf-8 -*-
"""D8 + D10 — the catalog's last two dims, one run each.

D8 獵取風暴: raids in the 24h before the signal. Two frozen predictions
CLASH (storm=chaos hurts vs storm=cleared-liquidity helps), no lean
recorded — data decides. Buckets 0 / 1-2 / 3+; headline 0 vs 3+.

D10 牆齡: age (bars since confirmation) of the nearest un-swept pool
ahead, for signals with a wall (<=1.4 ATR). Recorded lean: NO effect
(pool age already died once in the raid line); run for completeness.
Median split young vs old.

Both use the same G1 bar (|gap|>=4pp + halves same sign). Survivors
would proceed to G2/G3 in a dedicated script; a FAIL ends here.

Run: python research/terrain_d8_d10.py
Out: research/results/terrain_d8_d10.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402
from v7_price_location import pool_lifecycle  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_d8_d10.json"
WALL = 1.4


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def sh(g):
    return f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"


def g1(rows, a_pred, b_pred, la, lb):
    half = len(rows) // 2
    out = []
    for seg in (rows, rows[:half], rows[half:]):
        a_ = [r for r in seg if a_pred(r)]
        b_ = [r for r in seg if b_pred(r)]
        out.append(wr(a_) - wr(b_)
                   if len(a_) >= 15 and len(b_) >= 15 else None)
    ok = (None not in out and out[1] * out[2] > 0 and abs(out[0]) >= 4)
    show = " · ".join("thin" if d is None else f"{d:+.0f}" for d in out)
    print(f"  {la}−{lb} gap: {show} → G1 {'PASS' if ok else 'FAIL'}")
    return out, ok


def main() -> int:
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    res = {}

    print("=" * 78)
    print("  D8 獵取風暴 — 訊號前 24h 掃池數（兩派預測打架, 無傾向）")
    print("=" * 78)
    raid_hh = {}
    for r in raids_with_fill("BTC"):
        raid_hh[r["ts"] // 3600] = raid_hh.get(r["ts"] // 3600, 0) + 1
    rows = []
    for r in build_rows():
        hh = r["ts"] // 3600
        storm = sum(raid_hh.get(hh - k, 0) for k in range(0, 24))
        r2 = dict(r)
        r2["storm"] = storm
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    print(f"\n  n={len(rows)} · 整體 {wr(rows):.0f}%")
    for lab, pred in (("0 場", lambda r: r["storm"] == 0),
                      ("1-2 場", lambda r: 1 <= r["storm"] <= 2),
                      ("3+ 場", lambda r: r["storm"] >= 3)):
        print(f"    {lab:<6} {sh([r for r in rows if pred(r)])}")
    ds, ok = g1(rows, lambda r: r["storm"] == 0, lambda r: r["storm"] >= 3,
                "0場", "3+場")
    res["d8"] = {"deltas": ds, "g1_pass": ok}

    print()
    print("=" * 78)
    print("  D10 牆齡 — 前方牆(≤1.4 ATR)的池齡（傾向: 無效, 補完整性）")
    print("=" * 78)
    pools = pool_lifecycle(bars)
    wrows = []
    for r in build_rows():
        if r["ahead"] is None or r["ahead"] > WALL:
            continue
        j = ts2i[r["ts"]]
        c = cl[j]
        up = r["dir"] == "UP"
        best = None
        for p in pools:
            if p[0] <= j and (p[1] is None or p[1] > j):
                d_ = ((p[2] - c) if up else (c - p[2])) / atr[j]
                if d_ > 0 and (best is None or d_ < best[0]):
                    best = (d_, j - p[0])
        if best is None:
            continue
        r2 = dict(r)
        r2["age"] = best[1]
        wrows.append(r2)
    wrows.sort(key=lambda r: r["ts"])
    ages = sorted(r["age"] for r in wrows)
    med = ages[len(ages) // 2]
    print(f"\n  有牆訊號 n={len(wrows)} · 牆齡中位 {med} bars")
    for lab, pred in (("年輕牆(<中位)", lambda r: r["age"] < med),
                      ("老牆(≥中位)", lambda r: r["age"] >= med)):
        print(f"    {lab:<10} {sh([r for r in wrows if pred(r)])}")
    ds2, ok2 = g1(wrows, lambda r: r["age"] < med, lambda r: r["age"] >= med,
                  "年輕", "老")
    res["d10"] = {"median_bars": med, "deltas": ds2, "g1_pass": ok2}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
