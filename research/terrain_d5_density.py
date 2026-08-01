# -*- coding: utf-8 -*-
"""D5 — pool density within 3 ATR ahead of the signal.

Frozen prediction (TODO 0.484): denser cluster ahead = increasing
penalty (a stack of magnets/resistance in the path). Behind-version is
printed for the record only, per the catalog note.

Buckets declared before running: 疏 (<=1 pool) / 中 (2) / 密 (>=3);
headline contrast 疏 vs 密, predicted 疏 wins. The anti-repackaging
check that matters: D2 already knows the NEAREST pool ahead — density
must still separate WITHIN 前方牆 and 前方淨 buckets or it is D2 in a
new hat. Gates G1/G2/G3 per protocol.

Run: python research/terrain_d5_density.py
Out: research/results/terrain_d5_density.json
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

OUT = ROOT / "research/results/terrain_d5_density.json"
WALL, SUP = 1.4, 1.8
RANGE_ATR = 3.0


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  D5 池子密度 — 前方 3 ATR 內未掃池數（疏/中/密）")
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
        na = nb = 0
        for p in pools:
            if p[0] <= j and (p[1] is None or p[1] > j):
                d_ = (p[2] - c) / atr[j]
                if up:
                    ahead_d, behind_d = d_, -d_
                else:
                    ahead_d, behind_d = -d_, d_
                if 0 < ahead_d <= RANGE_ATR:
                    na += 1
                if 0 < behind_d <= RANGE_ATR:
                    nb += 1
        r2 = dict(r)
        r2["na"] = na
        r2["nb"] = nb
        r2["bk"] = "疏" if na <= 1 else ("中" if na == 2 else "密")
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    res = {}
    half = n // 2

    print(f"\n  [G1] 前方密度分桶（n={n}, 整體 {wr(rows):.0f}%）")
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        parts = []
        for b in ("疏", "中", "密"):
            g = [r for r in seg if r["bk"] == b]
            parts.append(f"{b} {wr(g):.0f}%({len(g)})" if len(g) >= 15
                         else f"{b} thin({len(g)})")
        print(f"  {tag:<4}" + " | ".join(parts))
        res[f"g1_{tag}"] = {b: [wr([r for r in seg if r["bk"] == b]),
                                len([r for r in seg if r["bk"] == b])]
                            for b in ("疏", "中", "密")}
    print("  [記錄用] 背後密度（不論證）: "
          + " | ".join(
              f"{lab} {wr(g):.0f}%({len(g)})" if len(
                  g := [r for r in rows
                        if (r['nb'] <= 1) == (lab == '疏背')]) >= 15
              else f"{lab} thin"
              for lab in ("疏背", "密背")))

    def gap(seg):
        a_ = [r for r in seg if r["bk"] == "疏"]
        b_ = [r for r in seg if r["bk"] == "密"]
        return (wr(a_) - wr(b_)
                if len(a_) >= 15 and len(b_) >= 15 else None)

    ds = gap(rows), gap(rows[:half]), gap(rows[half:])
    g1_pass = (None not in ds and ds[1] * ds[2] > 0 and abs(ds[0]) >= 4)
    show = " · ".join("thin" if d_ is None else f"{d_:+.0f}" for d_ in ds)
    print(f"  疏−密 gap: {show} → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1"] = {"deltas": ds, "pass": g1_pass}
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D5 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if ds[0] > 0 else -1
    good, bad = ("疏", "密") if sign == 1 else ("密", "疏")
    print(f"\n  [G2] 殘餘檢定（{good}−{bad}；重點=D2 桶內是否仍分離）")
    ok = tot = 0
    key_cells = {}
    for name, pred in (
            ("前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL),
            ("前方淨", lambda r: r["ahead"] is not None and r["ahead"] > WALL),
            ("ctx=none", lambda r: r["ctx"] == "none"),
            ("背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP),
            ("背後空", lambda r: r["behind"] is not None and r["behind"] > SUP)):
        seg = [r for r in rows if pred(r)]
        gg = [r for r in seg if r["bk"] == good]
        bb = [r for r in seg if r["bk"] == bad]
        if len(gg) >= 20 and len(bb) >= 20:
            d_ = wr(gg) - wr(bb)
            tot += 1
            ok += d_ > 0
            key_cells[name] = round(d_, 1)
            print(f"    {name:<10} {d_:+.0f}pp (n={len(gg)}/{len(bb)})")
    both_d2 = all(key_cells.get(k, -1) > 0 for k in ("前方牆", "前方淨")
                  if k in key_cells)
    g2_pass = tot >= 3 and ok / tot >= 0.67 and both_d2
    print(f"  桶內同向 {ok}/{tot}（D2 兩桶內須同向: {'✓' if both_d2 else '✗'}）"
          f" → G2 {'PASS' if g2_pass else 'FAIL'}")
    res["g2"] = {"ok": ok, "tot": tot, "cells": key_cells, "pass": g2_pass}
    if not g2_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  D5 止步於 G2（疑似 D2 換皮）— 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關（{good} vs {bad}）")
    ga = np.array([r["c"] for r in rows if r["bk"] == good])
    ba = np.array([r["c"] for r in rows if r["bk"] == bad])
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
        gg = [r["c"] for r in byq[q] if r["bk"] == good]
        bb = [r["c"] for r in byq[q] if r["bk"] == bad]
        if len(gg) >= 10 and len(bb) >= 10:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ D5 取得席位' if g3_pass else 'FAIL — 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass, "good_side": good}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
