# -*- coding: utf-8 -*-
"""D6 — flipped levels (swept 4-24h ago) acting as support behind.

Frozen prediction (TODO 0.484): a just-swept level behind the signal
behaves ≈ like the un-swept-pool support (D3: behind<=1.8 ATR -> 68%).

Cleanest faithful test (declared before running): among signals WITHOUT
un-swept support behind (behind>1.8 ATR or none — D3's 背後空), does a
flipped level within 1.8 ATR behind substitute for the missing support?
  有翻轉墊背  vs  真空墊背   (predicted: flip -> toward 68%, vacuum ~57%)
This framing controls D3 by construction (everyone in the sample lacks
real support). Within-支撐 additivity is printed for the record only.

Flip candidates: any pool swept 4-24 bars before the signal whose level
sits on the support side (below close for UP, above for DOWN) within
1.8 ATR. Textbook-flip subtype (break direction matches) recorded, not
gated. Gates G1/G2/G3 as fixed by the protocol.

Run: python research/terrain_d6_flip.py
Out: research/results/terrain_d6_flip.json
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
from v7_price_location import pool_lifecycle  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_d6_flip.json"
WALL, SUP = 1.4, 1.8
AGE_MIN, AGE_MAX = 4, 24


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  D6 翻轉位 — 剛掃過(4-24h)的價位能否頂替墊背支撐")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = pool_lifecycle(bars)     # [est, swept_or_None, lvl, side]
    rows = []
    for r in build_rows():
        j = ts2i[r["ts"]]
        c = cl[j]
        up = r["dir"] == "UP"
        flip = False
        flip_tb = False
        for p in pools:
            if p[1] is None or not (j - AGE_MAX <= p[1] <= j - AGE_MIN):
                continue
            lvl, side = p[2], p[3]
            d_ = (c - lvl) / atr[j] if up else (lvl - c) / atr[j]
            if 0 < d_ <= SUP:
                flip = True
                # textbook flip: broken resistance under an UP signal
                # (swept high, price above) / broken support over a DOWN
                if (up and side == 1) or ((not up) and side == -1):
                    flip_tb = True
        r2 = dict(r)
        r2["flip"] = flip
        r2["flip_tb"] = flip_tb
        r2["has_sup"] = r["behind"] is not None and r["behind"] <= SUP
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    res = {}

    vac = [r for r in rows if not r["has_sup"]]      # D3 背後空 population
    n = len(vac)
    print(f"\n  [記錄用] 全樣本 {len(rows)} · 背後空(無未掃池支撐) {n}")
    sup_g = [r for r in rows if r["has_sup"]]
    print(f"    參考: 背後支撐 {wr(sup_g):.0f}%({len(sup_g)}) | "
          f"背後空 {wr(vac):.0f}%({n})")
    for tag, g in (("支撐內·另有翻轉", [r for r in sup_g if r["flip"]]),
                   ("支撐內·無翻轉", [r for r in sup_g if not r["flip"]])):
        s = f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"
        print(f"    {tag}: {s}")

    half = n // 2
    print(f"\n  [G1] 背後空樣本內：有翻轉墊背 vs 真空墊背")
    segs = {}
    for tag, seg in (("全期", vac), ("H1", vac[:half]), ("H2", vac[half:])):
        f_ = [r for r in seg if r["flip"]]
        v_ = [r for r in seg if not r["flip"]]
        d_ = wr(f_) - wr(v_) if len(f_) >= 15 and len(v_) >= 15 else None
        segs[tag] = d_
        sf = f"{wr(f_):.0f}%({len(f_)})" if f_ else "—"
        sv = f"{wr(v_):.0f}%({len(v_)})" if v_ else "—"
        print(f"  {tag:<4} 翻轉 {sf} | 真空 {sv}"
              + (f" | gap {d_:+.0f}pp" if d_ is not None else " | thin"))
    tb = [r for r in vac if r["flip_tb"]]
    print(f"    （教科書翻轉子集: {wr(tb):.0f}%({len(tb)})" if len(tb) >= 15
          else f"    （教科書翻轉子集 thin({len(tb)})", "— 記錄不論證）")
    ds = segs["全期"], segs["H1"], segs["H2"]
    g1_pass = (None not in ds and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
    show = " · ".join("thin" if d_ is None else f"{d_:+.0f}" for d_ in ds)
    print(f"  翻轉−真空 gap: {show} → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1"] = {"deltas": ds, "pass": g1_pass}
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D6 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if ds[0] > 0 else -1
    lab = ("翻轉", "真空") if sign == 1 else ("真空", "翻轉")
    print(f"\n  [G2] 殘餘檢定（{lab[0]}−{lab[1]}，背後空樣本內）")
    ok = tot = 0
    for name, pred in (
            ("ctx=none", lambda r: r["ctx"] == "none"),
            ("ctx=fade", lambda r: r["ctx"] == "fade"),
            ("ctx=follow", lambda r: r["ctx"] == "follow"),
            ("前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL),
            ("前方淨", lambda r: r["ahead"] is not None and r["ahead"] > WALL)):
        seg = [r for r in vac if pred(r)]
        a_ = [r for r in seg if r["flip"] == (sign == 1)]
        b_ = [r for r in seg if r["flip"] != (sign == 1)]
        if len(a_) >= 20 and len(b_) >= 20:
            d_ = wr(a_) - wr(b_)
            tot += 1
            ok += d_ > 0
            print(f"    {name:<10} {d_:+.0f}pp (n={len(a_)}/{len(b_)})")
    g2_pass = tot >= 3 and ok / tot >= 0.67
    print(f"  桶內同向 {ok}/{tot} → G2 {'PASS' if g2_pass else 'FAIL'}")
    res["g2"] = {"ok": ok, "tot": tot, "pass": g2_pass}
    if not g2_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D6 止步於 G2 — 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關")
    ga = np.array([r["c"] for r in vac if r["flip"] == (sign == 1)])
    ba = np.array([r["c"] for r in vac if r["flip"] != (sign == 1)])
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
    for r in vac:
        dt = datetime.fromtimestamp(r["ts"], timezone.utc)
        byq.setdefault(f"{dt.year}-Q{(dt.month-1)//3+1}", []).append(r)
    qsigns = []
    for q in sorted(byq):
        gg = [r["c"] for r in byq[q] if r["flip"] == (sign == 1)]
        bb = [r["c"] for r in byq[q] if r["flip"] != (sign == 1)]
        if len(gg) >= 10 and len(bb) >= 10:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ D6 取得席位' if g3_pass else 'FAIL — 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
