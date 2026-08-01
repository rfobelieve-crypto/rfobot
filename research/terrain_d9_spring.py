# -*- coding: utf-8 -*-
"""D9 — the spring: deep raid + reclaim, then the aligned (fade) signal.

Frozen prediction (TODO 0.484): within fade-context signals, the subset
whose triggering raid was DEEP and RECLAIMED (Wyckoff spring shape) is
the fattest fade bucket.

Causal parts, zero tuned numbers:
  deep     pierce >= expanding causal median of all raid pierces so far
  reclaim  the raid bar's own close is back inside the swept level
           (known at signal time — V7 fires on bar close)
Population: ctx=fade signals; the joined raid is the same one the veto
line uses (first raid hour within 0-4 bars back, first raid that hour).
Contrast: 彈簧(深∧收回) vs 其他fade. Gates G1/G2/G3 per protocol.

Run: python research/terrain_d9_spring.py
Out: research/results/terrain_d9_spring.json
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
from sweep_raid_postflow import raids_with_fill  # noqa: E402
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_d9_spring.json"
WALL, SUP = 1.4, 1.8


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  D9 彈簧後開槍 — 深獵取∧收回 的 fade 是否最肥")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    cl = [b[SC.C] for b in bars]
    raids = sorted(raids_with_fill("BTC"), key=lambda r: r["ts"])
    # expanding causal median of pierce + per-hour first raid with flags
    by_hh = {}
    seen = []
    for r in raids:
        med = float(np.median(seen)) if len(seen) >= 30 else None
        seen.append(r["pierce"])
        hh = r["ts"] // 3600
        if hh in by_hh:
            continue
        j = ts2i.get(r["ts"])
        if j is None:
            continue
        reclaimed = (cl[j] <= r["lvl"]) if r["side"] == 1 else (cl[j] >= r["lvl"])
        deep = med is not None and r["pierce"] >= med
        by_hh[hh] = {"side": r["side"], "deep": deep, "reclaimed": reclaimed}
    fades = []
    for r in build_rows():
        if r["ctx"] != "fade":
            continue
        rd = None
        for k in range(0, 5):
            rd = by_hh.get(r["ts"] // 3600 - k)
            if rd is not None:
                break
        if rd is None:
            continue
        r2 = dict(r)
        r2["spring"] = rd["deep"] and rd["reclaimed"]
        r2["deep"] = rd["deep"]
        r2["reclaimed"] = rd["reclaimed"]
        fades.append(r2)
    fades.sort(key=lambda r: r["ts"])
    n = len(fades)
    res = {}

    print(f"\n  [記錄用] fade 訊號 n={n} · 整體 {wr(fades):.0f}%")
    for tag, pred in (("深∧收回(彈簧)", lambda r: r["spring"]),
                      ("深∧未收回", lambda r: r["deep"] and not r["reclaimed"]),
                      ("淺∧收回", lambda r: (not r["deep"]) and r["reclaimed"]),
                      ("淺∧未收回", lambda r: (not r["deep"]) and not r["reclaimed"])):
        g = [r for r in fades if pred(r)]
        s = f"{wr(g):.0f}%({len(g)})" if len(g) >= 15 else f"thin({len(g)})"
        print(f"    {tag:<12} {s}")

    half = n // 2
    print(f"\n  [G1] 彈簧 vs 其他 fade")
    segs = {}
    for tag, seg in (("全期", fades), ("H1", fades[:half]), ("H2", fades[half:])):
        a_ = [r for r in seg if r["spring"]]
        b_ = [r for r in seg if not r["spring"]]
        d_ = wr(a_) - wr(b_) if len(a_) >= 15 and len(b_) >= 15 else None
        segs[tag] = d_
        sa = f"{wr(a_):.0f}%({len(a_)})" if a_ else "—"
        sb = f"{wr(b_):.0f}%({len(b_)})" if b_ else "—"
        print(f"  {tag:<4} 彈簧 {sa} | 其他 {sb}"
              + (f" | gap {d_:+.0f}pp" if d_ is not None else " | thin"))
    ds = segs["全期"], segs["H1"], segs["H2"]
    g1_pass = (None not in ds and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
    show = " · ".join("thin" if d_ is None else f"{d_:+.0f}" for d_ in ds)
    print(f"  彈簧−其他 gap: {show} → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1"] = {"deltas": ds, "pass": g1_pass}
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D9 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if ds[0] > 0 else -1
    print(f"\n  [G2] 殘餘檢定（fade 樣本內, {'彈簧優' if sign==1 else '彈簧劣'}）")
    ok = tot = 0
    for name, pred in (
            ("前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL),
            ("前方淨", lambda r: r["ahead"] is not None and r["ahead"] > WALL),
            ("背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP),
            ("背後空", lambda r: r["behind"] is not None and r["behind"] > SUP)):
        seg = [r for r in fades if pred(r)]
        a_ = [r for r in seg if r["spring"] == (sign == 1)]
        b_ = [r for r in seg if r["spring"] != (sign == 1)]
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
        print("\n  D9 止步於 G2 — 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關")
    ga = np.array([r["c"] for r in fades if r["spring"] == (sign == 1)])
    ba = np.array([r["c"] for r in fades if r["spring"] != (sign == 1)])
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
    for r in fades:
        dt = datetime.fromtimestamp(r["ts"], timezone.utc)
        byq.setdefault(f"{dt.year}-Q{(dt.month-1)//3+1}", []).append(r)
    qsigns = []
    for q in sorted(byq):
        gg = [r["c"] for r in byq[q] if r["spring"] == (sign == 1)]
        bb = [r["c"] for r in byq[q] if r["spring"] != (sign == 1)]
        if len(gg) >= 10 and len(bb) >= 10:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ D9 取得席位' if g3_pass else 'FAIL — 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
