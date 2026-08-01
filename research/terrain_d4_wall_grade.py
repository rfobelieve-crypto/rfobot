# -*- coding: utf-8 -*-
"""D4 — grade of the wall ahead (liquidity layer, first dim after the
structure layer was closed with zero seats).

Setup: signals whose nearest UN-SWEPT pool ahead is within the confirmed
D2 wall margin (<=1.4 ATR). The wall's KIND comes from the same pool
builder the whole system shares (swing / session / pdh_pdl / pwh_pwl).

Frozen prediction (TODO 0.484): hardness 上週 > 昨日 > 波段 > 時段.
Gate contrast (declared BEFORE running, to keep cells honest): binary
coarsening of that order — 日週級牆 (pwh_pwl+pdh_pdl) vs 盤中級牆
(swing+session). Harder wall = worse for the signal pushing into it, so
predicted 盤中級 WR > 日週級 WR. The 4-way table is printed for the
record but never argued from.

Gates: G1 halves on the binary contrast; G2 residual inside D1 ctx and
D3 support buckets; G3 permutation + bootstrap + quarters.

Run: python research/terrain_d4_wall_grade.py
Out: research/results/terrain_d4_wall_grade.json
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

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shadow_review import build_pools_with_origin  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_d4_wall_grade.json"
WALL, SUP = 1.4, 1.8
ZH = {"swing": "波段", "session": "時段", "pdh_pdl": "昨日", "pwh_pwl": "上週"}


def kinded_lifecycle(bars):
    """pool_lifecycle, but each pool keeps its kind:
    [est, swept_or_None, lvl, side, kind]."""
    H, L = SC.H, SC.L
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    pools = []
    for kind, plist in build_pools_with_origin(bars).items():
        for p in plist:
            pools.append([p["est"], None, p["lvl"], p["side"], kind])
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


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  D4 牆的等級 — 前方 ≤1.4 ATR 牆的池種（日週級 vs 盤中級）")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = kinded_lifecycle(bars)
    rows = []
    for r in build_rows():
        if r["ahead"] is None or r["ahead"] > WALL:
            continue
        j = ts2i[r["ts"]]
        c = cl[j]
        up = r["dir"] == "UP"
        # a physical level often carries several pool identities (a swing
        # high is usually also a session high, sometimes PDH) — the wall's
        # grade is the HIGHEST label stacked within eps of the nearest
        # distance, not whichever copy happened to confirm first
        cands = []
        for p in pools:
            if p[0] <= j and (p[1] is None or p[1] > j):
                if up and p[2] > c:
                    d_ = (p[2] - c) / atr[j]
                elif (not up) and p[2] < c:
                    d_ = (c - p[2]) / atr[j]
                else:
                    continue
                cands.append((d_, p[4]))
        if not cands:
            continue
        d0 = min(d_ for d_, _k in cands)
        stack = {k for d_, k in cands if d_ <= d0 + 0.05}
        rank = ["pwh_pwl", "pdh_pdl", "swing", "session"]
        kind = next(k for k in rank if k in stack)
        r2 = dict(r)
        r2["kind"] = kind
        r2["cls"] = "日週級" if kind in ("pdh_pdl", "pwh_pwl") else "盤中級"
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    res = {}

    print(f"\n  [記錄用] 四池種各自（不作論證）")
    for k in ("pwh_pwl", "pdh_pdl", "swing", "session"):
        g = [r for r in rows if r["kind"] == k]
        w = f"{wr(g):.0f}%" if len(g) >= 15 else "thin"
        print(f"    {ZH[k]:<4} n={len(g):>4}  WR {w}")
        res[f"kind_{k}"] = [wr(g), len(g)]

    half = n // 2
    print(f"\n  [G1] 二元對比（n={n}, 有牆訊號整體 {wr(rows):.0f}%）")
    segs = {}
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        soft = [r for r in seg if r["cls"] == "盤中級"]
        hard = [r for r in seg if r["cls"] == "日週級"]
        d_ = (wr(soft) - wr(hard)
              if len(soft) >= 15 and len(hard) >= 15 else None)
        segs[tag] = d_
        print(f"  {tag:<4} 盤中級 {wr(soft):.0f}%({len(soft)}) | "
              f"日週級 {wr(hard):.0f}%({len(hard)})"
              + (f" | gap {d_:+.0f}pp" if d_ is not None else " | thin"))
    ds = segs["全期"], segs["H1"], segs["H2"]
    g1_pass = (None not in ds and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
    show = " · ".join("thin" if d_ is None else f"{d_:+.0f}" for d_ in ds)
    print(f"  盤中−日週 gap: {show} → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1"] = {"deltas": ds, "pass": g1_pass}
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D4 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if ds[0] > 0 else -1
    good, bad = ("盤中級", "日週級") if sign == 1 else ("日週級", "盤中級")
    print(f"\n  [G2] 殘餘檢定（{good}−{bad}）")
    ok = tot = 0
    for name, pred in (
            ("ctx=none", lambda r: r["ctx"] == "none"),
            ("ctx=fade", lambda r: r["ctx"] == "fade"),
            ("ctx=follow", lambda r: r["ctx"] == "follow"),
            ("背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP),
            ("背後空", lambda r: r["behind"] is not None and r["behind"] > SUP)):
        seg = [r for r in rows if pred(r)]
        gg = [r for r in seg if r["cls"] == good]
        bb = [r for r in seg if r["cls"] == bad]
        if len(gg) >= 20 and len(bb) >= 20:
            d_ = wr(gg) - wr(bb)
            tot += 1
            ok += d_ > 0
            print(f"    {name:<10} {d_:+.0f}pp (n={len(gg)}/{len(bb)})")
    g2_pass = tot >= 3 and ok / tot >= 0.67
    print(f"  桶內同向 {ok}/{tot} → G2 {'PASS' if g2_pass else 'FAIL'}")
    res["g2"] = {"ok": ok, "tot": tot, "pass": g2_pass}
    if not g2_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D4 止步於 G2 — 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關（{good} vs {bad}）")
    ga = np.array([r["c"] for r in rows if r["cls"] == good])
    ba = np.array([r["c"] for r in rows if r["cls"] == bad])
    obs = 100 * (ga.mean() - ba.mean())
    rgen = np.random.default_rng(7)
    pool_ = np.concatenate([ga, ba])
    null = []
    for _ in range(2000):
        p_ = rgen.permutation(pool_)
        null.append(100 * (p_[:len(ga)].mean() - p_[len(ga):].mean()))
    pval = float((np.array(null) >= obs).mean())
    boots = []
    for _ in range(2000):
        boots.append(100 * (rgen.choice(ga, len(ga), True).mean()
                            - rgen.choice(ba, len(ba), True).mean()))
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    byq = {}
    for r in rows:
        dt = datetime.fromtimestamp(r["ts"], timezone.utc)
        byq.setdefault(f"{dt.year}-Q{(dt.month-1)//3+1}", []).append(r)
    qsigns = []
    for q in sorted(byq):
        gg = [r["c"] for r in byq[q] if r["cls"] == good]
        bb = [r["c"] for r in byq[q] if r["cls"] == bad]
        if len(gg) >= 10 and len(bb) >= 10:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ D4 取得席位' if g3_pass else 'FAIL — 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass, "good_side": good}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
