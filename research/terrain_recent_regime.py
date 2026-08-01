# -*- coding: utf-8 -*-
"""Do the four confirmed terrain dims survive the CURRENT decode regime?

Why this exists: tracked_signals shows a structural break in Strong
volume — 114-175/month through 2026-03, then 12-28/month from April on.
The timing matches the 2026-04-19 warmup-buffer fix (before it, the
rolling-percentile decode ran on fallback thresholds and over-issued
signals). So the 766-signal terrain sample is dominated by a decode
regime that no longer exists, and only ~79 signals come from the
current one.

Declared BEFORE running, because n is small: this is a SIGN-REVERSAL
check, not a re-verification. At ~20-40 per cell nothing here can pass
the three gates, and a shrunken-but-same-sign gap is the expected
outcome of small samples, not evidence of decay. The only result that
would matter is a dimension flipping sign in the current regime.

Run: python research/terrain_recent_regime.py
Out: research/results/terrain_recent_regime.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from v7_price_location import pool_lifecycle  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_recent_regime.json"
WALL, SUP, RANGE_ATR = 1.4, 1.8, 3.0
CUTS = ("2026-04-20", "2026-05-01")


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def cell(g):
    return f"{wr(g):.0f}%({len(g)})" if len(g) >= 10 else f"thin({len(g)})"


def main() -> int:
    print("=" * 78)
    print("  地形四維 × 現行解碼 regime（2026-04 buffer 修復後）")
    print("  ※ 這是「方向有沒有翻」的檢查，不是重新驗證——n 不夠跑三關")
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
                if 0 < (d_ if up else -d_) <= RANGE_ATR:
                    na += 1
        r2 = dict(r)
        r2["na"] = na
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])

    dims = (
        ("D1 追突破 veto", lambda r: r["ctx"] == "follow",
         lambda r: r["ctx"] != "follow", "追突破", "其他", -1),
        ("D2 前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL,
         lambda r: r["ahead"] is not None and r["ahead"] > WALL,
         "有牆", "跑道淨", -1),
        ("D3 背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP,
         lambda r: r["behind"] is None or r["behind"] > SUP,
         "有墊背", "背後空", +1),
        ("D5 池子密度", lambda r: r["na"] <= 1, lambda r: r["na"] >= 3,
         "疏", "密", +1),
    )
    res = {}
    for cut in ("全樣本",) + CUTS:
        if cut == "全樣本":
            seg = rows
        else:
            t0 = datetime.strptime(cut, "%Y-%m-%d").replace(
                tzinfo=timezone.utc).timestamp()
            seg = [r for r in rows if r["ts"] >= t0]
        print(f"\n  ── {cut}  n={len(seg)} · 整體 {wr(seg):.0f}%")
        res[cut] = {"n": len(seg), "base_wr": wr(seg)}
        for name, pa, pb, la, lb, want in dims:
            a_ = [r for r in seg if pa(r)]
            b_ = [r for r in seg if pb(r)]
            gap = (wr(a_) - wr(b_)
                   if len(a_) >= 10 and len(b_) >= 10 else None)
            mark = ""
            if gap is not None:
                same = (gap > 0) == (want > 0)
                mark = "  方向一致 ✓" if same else "  ⚠ 方向翻轉"
            print(f"    {name:<14} {la} {cell(a_):<12} {lb} {cell(b_):<12}"
                  + (f" gap {gap:+.0f}pp{mark}" if gap is not None
                     else " 樣本不足"))
            res[cut][name] = {"gap": gap, "na": len(a_), "nb": len(b_)}

    # The gaps came back BIGGER in the small sample, which is the shape a
    # standing order says to distrust. Cheapest decisive check: is any of
    # it carried by one clustered day? Leave-one-day-out on the recent cut
    # — if dropping a single day flips a sign, that dim is one cluster,
    # not a regime-stable effect.
    t0 = datetime.strptime(CUTS[0], "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp()
    seg = [r for r in rows if r["ts"] >= t0]
    days = sorted({datetime.fromtimestamp(r["ts"], timezone.utc).date()
                   for r in seg})
    print(f"\n  ── 集中度：{CUTS[0]} 起 {len(seg)} 筆分佈在 {len(days)} 天，"
          f"逐日剔除後的 gap 範圍")
    res["leave_one_day_out"] = {}
    for name, pa, pb, _la, _lb, want in dims:
        gaps = []
        for d in days:
            sub = [r for r in seg
                   if datetime.fromtimestamp(r["ts"], timezone.utc).date() != d]
            a_ = [r for r in sub if pa(r)]
            b_ = [r for r in sub if pb(r)]
            if len(a_) >= 10 and len(b_) >= 10:
                gaps.append(wr(a_) - wr(b_))
        if not gaps:
            continue
        flip = any((g > 0) != (want > 0) for g in gaps)
        print(f"    {name:<14} [{min(gaps):+.0f}, {max(gaps):+.0f}]pp"
              + ("  ⚠ 有翻轉" if flip else "  無翻轉 ✓"))
        res["leave_one_day_out"][name] = {
            "min": round(min(gaps), 1), "max": round(max(gaps), 1),
            "sign_flip": flip}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
