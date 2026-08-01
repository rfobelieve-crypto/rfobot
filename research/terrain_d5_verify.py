# -*- coding: utf-8 -*-
"""D5 adversarial battery — the first terrain survivor gets shot at
before its seat is real (operator standing order: 數據太好看記得反復驗證).

The sharpest attack, declared first: HIGH ATR widens the 3-ATR window in
dollars, mechanically catching more pools — so 密 could be a volatility
regime proxy, and vol regime moves V7 WR on its own. The claim must
survive INSIDE ATR terciles.

Battery:
  1 thirds stability          — halves can hide a middle inversion
  2 direction split           — all-DOWN artifact check
  3 regime split              — regime proxy check
  4 ATR-tercile control       — THE mechanical confound (above)
  5 count monotonicity        — na=0,1,2,3,4+ should trend down, not
                                be one lucky bucket boundary
  6 boundary sensitivity      — gap must survive moving the 密 cut to
                                >=4 and the 疏 cut to ==0
Verdict: reversal in any well-populated slice = kill; shrink = note.

Run: python research/terrain_d5_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from v7_price_location import pool_lifecycle  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RANGE_ATR = 3.0


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def sh(g):
    return f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"


def main() -> int:
    print("=" * 78)
    print("  D5 對抗性彈藥庫 — 密度效果是不是 ATR regime 換皮")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = pool_lifecycle(bars)
    rows = []
    for r in build_rows():
        j = ts2i[r["ts"]]
        c = cl[j]
        up = r["dir"] == "UP"
        na = 0
        for p in pools:
            if p[0] <= j and (p[1] is None or p[1] > j):
                d_ = (p[2] - c) / atr[j]
                ad = d_ if up else -d_
                if 0 < ad <= RANGE_ATR:
                    na += 1
        r2 = dict(r)
        r2["na"] = na
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)

    def gap(seg, lo_max=1, hi_min=3):
        a_ = [r for r in seg if r["na"] <= lo_max]
        b_ = [r for r in seg if r["na"] >= hi_min]
        if len(a_) < 15 or len(b_) < 15:
            return None, len(a_), len(b_)
        return wr(a_) - wr(b_), len(a_), len(b_)

    print(f"\n  [1] 三分穩定（全期 gap +8.2）")
    third = n // 3
    for i, tag in enumerate(("T1", "T2", "T3")):
        seg = rows[i * third:(i + 1) * third if i < 2 else n]
        d_, na_, nb_ = gap(seg)
        print(f"    {tag}: " + ("thin" if d_ is None
                                else f"{d_:+.0f}pp (n={na_}/{nb_})"))

    print(f"\n  [2] 方向拆分")
    for d0 in ("UP", "DOWN"):
        seg = [r for r in rows if r["dir"] == d0]
        d_, na_, nb_ = gap(seg)
        print(f"    {d0:<5}: " + ("thin" if d_ is None
                                  else f"{d_:+.0f}pp (n={na_}/{nb_})"))

    print(f"\n  [3] regime 拆分")
    for rg in ("CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"):
        seg = [r for r in rows if r["regime"] == rg]
        d_, na_, nb_ = gap(seg)
        print(f"    {rg:<14}: " + ("thin" if d_ is None
                                   else f"{d_:+.0f}pp (n={na_}/{nb_})"))

    print(f"\n  [4] ATR 三分位控制（關鍵攻擊）")
    vs = sorted(r["volp"] for r in rows)
    v1, v2 = vs[n // 3], vs[2 * n // 3]
    for tag, pred in (("低vol", lambda r: r["volp"] <= v1),
                      ("中vol", lambda r: v1 < r["volp"] <= v2),
                      ("高vol", lambda r: r["volp"] > v2)):
        seg = [r for r in rows if pred(r)]
        d_, na_, nb_ = gap(seg)
        dens = 100 * sum(1 for r in seg if r["na"] >= 3) / len(seg)
        print(f"    {tag}: " + ("thin" if d_ is None
                                else f"{d_:+.0f}pp (n={na_}/{nb_})")
              + f" · 密佔比 {dens:.0f}%")

    print(f"\n  [5] 逐 count 單調性")
    for k in (0, 1, 2, 3, 4):
        g = [r for r in rows if (r["na"] == k if k < 4 else r["na"] >= 4)]
        print(f"    na={'4+' if k == 4 else k}: {sh(g)}")

    print(f"\n  [6] 桶界敏感度")
    for lab, lo_max, hi_min in (("≤1 vs ≥3(原)", 1, 3), ("≤1 vs ≥4", 1, 4),
                                ("==0 vs ≥3", 0, 3), ("≤2 vs ≥3", 2, 3)):
        d_, na_, nb_ = gap(rows, lo_max, hi_min)
        print(f"    {lab:<14}: " + ("thin" if d_ is None
                                    else f"{d_:+.0f}pp (n={na_}/{nb_})"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
