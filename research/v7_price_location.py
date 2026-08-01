# -*- coding: utf-8 -*-
"""V7 signals x TRUE price location — distance to resting liquidity pools
at fire time. The raid-context veto measured WHEN (a sweep just happened);
this measures WHERE (how far the nearest un-swept pool sits ahead of /
behind the signal), which is what 價格位置 actually means.

Causal reconstruction: the four pool types with their lifecycle (established
-> swept). At a signal's bar T, resting pools = established <= T and not yet
swept. Distances in ATR(T):
  dist_ahead  nearest resting pool IN the signal direction (UP -> above)
  dist_behind nearest resting pool on the opposite side

Two textbook stories CONFLICT and are both stated up front:
  magnet     price seeks liquidity -> near pool ahead helps the 4h move
  resistance near pool ahead = a wall of orders -> hurts
Report terciles x (WR / directional 4h return) + halves; the data picks.
~4 looks. BTC, live Strong signals with outcomes.

Run: python research/v7_price_location.py
Out: research/results/v7_price_location.json
"""
from __future__ import annotations

import json
import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shared.db import get_db_conn  # noqa: E402
from shadow_review import build_pools_with_origin  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_price_location.json"


def pool_lifecycle(bars):
    """[(est_i, swept_i_or_None, lvl, side)] under the frozen sweep rule."""
    H, L = SC.H, SC.L
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    pools = []
    for kind, plist in build_pools_with_origin(bars).items():
        for p in plist:
            pools.append([p["est"], None, p["lvl"], p["side"]])
    pools.sort(key=lambda x: x[0])
    live = []
    idx = 0
    for j in range(len(bars)):
        while idx < len(pools) and pools[idx][0] <= j:
            live.append(pools[idx])
            idx += 1
        for p in list(live):
            lvl, s = p[2], p[3]
            if (h[j] > lvl if s == 1 else lo[j] < lvl):
                p[1] = j
                live.remove(p)
    return pools


def main() -> int:
    print("=" * 78)
    print("  V7 x PRICE LOCATION — 開火時距未掃池的距離（磁鐵 vs 阻力, 資料裁決）")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = pool_lifecycle(bars)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, correct, actual_return_4h "
                "FROM tracked_signals WHERE strength='Strong' "
                "AND correct IS NOT NULL ORDER BY signal_time")
            sigs = cur.fetchall()
    finally:
        conn.close()

    rows = []
    for s in sigs:
        ts = int(s["signal_time"].replace(tzinfo=timezone.utc).timestamp())
        j = ts2i.get(ts)
        if j is None or atr[j] in (None, 0):
            continue
        up = s["direction"] == "UP"
        c = cl[j]
        above = [p[2] for p in pools if p[0] <= j and (p[1] is None or p[1] > j)
                 and p[2] > c]
        below = [p[2] for p in pools if p[0] <= j and (p[1] is None or p[1] > j)
                 and p[2] < c]
        ahead = (min(above) - c) / atr[j] if up and above else \
                (c - max(below)) / atr[j] if (not up) and below else None
        behind = (c - max(below)) / atr[j] if up and below else \
                 (min(above) - c) / atr[j] if (not up) and above else None
        sgn = 1 if up else -1
        rows.append({"ts": ts, "c": int(s["correct"]),
                     "ret": (float(s["actual_return_4h"]) * sgn
                             if s["actual_return_4h"] is not None else None),
                     "ahead": ahead, "behind": behind})
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    print(f"  matched Strong signals: {n}")
    res = {}

    def wr(g):
        return 100 * sum(r["c"] for r in g) / len(g) if g else None

    def rt(g):
        xs = [r["ret"] for r in g if r["ret"] is not None]
        return 100 * sum(xs) / len(xs) if xs else None

    def terc(rows_, key, label):
        vals = sorted(r[key] for r in rows_ if r[key] is not None)
        if len(vals) < 90:
            print(f"  {label} thin"); return None
        lo_c, hi_c = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        out = {}
        parts = []
        for nm, pr in (("近", lambda v: v <= lo_c),
                       ("中", lambda v: lo_c < v < hi_c),
                       ("遠", lambda v: v >= hi_c)):
            g = [r for r in rows_ if r[key] is not None and pr(r[key])]
            out[nm] = {"n": len(g), "wr": wr(g), "ret": rt(g)}
            parts.append(f"{nm} {wr(g):.0f}%/{rt(g):+.2f}% (n={len(g)})")
        print(f"  {label:<16}" + " | ".join(parts)
              + f"   [切點 {lo_c:.2f}/{hi_c:.2f} ATR]")
        return out

    print("\n  [前方距離]（訊號方向上最近的未掃池）")
    res["ahead"] = terc(rows, "ahead", "dist_ahead")
    print("  [後方距離]")
    res["behind"] = terc(rows, "behind", "dist_behind")

    half = n // 2
    print("\n  [halves]")
    for tag, seg in (("H1", rows[:half]), ("H2", rows[half:])):
        terc(seg, "ahead", f"{tag} ahead")
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  兩個對立預測都已事前聲明：磁鐵(近=好) vs 阻力(近=壞)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
