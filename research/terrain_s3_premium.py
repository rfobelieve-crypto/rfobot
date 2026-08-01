# -*- coding: utf-8 -*-
"""S3 — premium/discount within the dealing range (catalog order #2).

Definition (causal): dealing range at a signal bar = most recent CONFIRMED
swing high price <-> most recent confirmed swing low price (PIVOT=10,
confirmation at est=i+P+1). Position pos = (close - low) / (high - low),
degenerate ranges (high<=low) skipped. Direction-aligned position
dpos = pos for UP signals, 1-pos for DOWN — low dpos means the signal
fires from the CHEAP side of its own trade (discount for longs, premium
for shorts).

Frozen prediction (TODO 0.484): dpos low (折價側) wins — buy cheap, sell
dear. Buckets 折價 <=0.33 / 中間 / 溢價 >=0.67; headline = 折價 vs 溢價.
Same three gates as S1; a pretty number triggers extra scrutiny.

Run: python research/terrain_s3_premium.py
Out: research/results/terrain_s3_premium.json
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
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/terrain_s3_premium.json"
WALL, SUP = 1.4, 1.8


def range_series(bars):
    """bar -> (last confirmed swing high, last confirmed swing low)."""
    H, L = SC.H, SC.L
    n = len(bars)
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    P = SC.PIVOT
    piv = []
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            piv.append((i + P + 1, i, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            piv.append((i + P + 1, i, lo[i], -1))
    piv.sort()
    out = [None] * n
    hi_last = lo_last = None
    hi_i = lo_i = -1
    pi = 0
    for j in range(n):
        while pi < len(piv) and piv[pi][0] <= j:
            _e, i_, price, side = piv[pi]
            if side == 1 and i_ > hi_i:
                hi_last, hi_i = price, i_
            if side == -1 and i_ > lo_i:
                lo_last, lo_i = price, i_
            pi += 1
        if hi_last is not None and lo_last is not None and hi_last > lo_last:
            out[j] = (hi_last, lo_last)
    return out


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  S3 折價/溢價 — 三關協議")
    print("=" * 78)
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    cl = [b[SC.C] for b in bars]
    rng_s = range_series(bars)
    rows = []
    for r in build_rows():
        j = ts2i.get(r["ts"])
        if j is None or rng_s[j] is None:
            continue
        hi, lo_ = rng_s[j]
        pos = (cl[j] - lo_) / (hi - lo_)
        pos = min(max(pos, -0.5), 1.5)
        dpos = pos if r["dir"] == "UP" else 1 - pos
        r2 = dict(r)
        r2["dpos"] = dpos
        r2["bk"] = ("折價" if dpos <= 0.33 else
                    "溢價" if dpos >= 0.67 else "中間")
        rows.append(r2)
    rows.sort(key=lambda r: r["ts"])
    n = len(rows)
    res = {}

    print(f"\n  [G1] 分桶（n={n}, 整體 {wr(rows):.0f}%）")
    half = n // 2
    for tag, seg in (("全期", rows), ("H1", rows[:half]), ("H2", rows[half:])):
        parts = []
        for b in ("折價", "中間", "溢價"):
            g = [r for r in seg if r["bk"] == b]
            parts.append(f"{b} {wr(g):.0f}% (n={len(g)})" if len(g) >= 15
                         else f"{b} thin({len(g)})")
        print(f"  {tag:<4}" + " | ".join(parts))
        res[f"g1_{tag}"] = {b: wr([r for r in seg if r["bk"] == b])
                            for b in ("折價", "中間", "溢價")}
    d_full = (res["g1_全期"]["折價"] or 0) - (res["g1_全期"]["溢價"] or 0)
    d1 = (res["g1_H1"]["折價"] or 0) - (res["g1_H1"]["溢價"] or 0)
    d2 = (res["g1_H2"]["折價"] or 0) - (res["g1_H2"]["溢價"] or 0)
    g1_pass = d1 * d2 > 0 and abs(d_full) >= 4
    print(f"  折價−溢價 gap: 全期 {d_full:+.0f}pp · H1 {d1:+.0f} · H2 {d2:+.0f}"
          f" → G1 {'PASS' if g1_pass else 'FAIL'}")
    res["g1_pass"] = g1_pass
    if not g1_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  S3 止步於 G1 — 記錄後收檔")
        return 0

    sign = 1 if d_full > 0 else -1
    good, bad = ("折價", "溢價") if sign == 1 else ("溢價", "折價")
    print(f"\n  [G2] 殘餘檢定（{good}−{bad}）")
    ok = tot = 0
    for name, pred in (
            ("ctx=none", lambda r: r["ctx"] == "none"),
            ("ctx=fade", lambda r: r["ctx"] == "fade"),
            ("ctx=follow", lambda r: r["ctx"] == "follow"),
            ("前方牆", lambda r: r["ahead"] is not None and r["ahead"] <= WALL),
            ("前方淨", lambda r: r["ahead"] is not None and r["ahead"] > WALL),
            ("背後支撐", lambda r: r["behind"] is not None and r["behind"] <= SUP),
            ("背後空", lambda r: r["behind"] is not None and r["behind"] > SUP)):
        seg = [r for r in rows if pred(r)]
        gg = [r for r in seg if r["bk"] == good]
        bb = [r for r in seg if r["bk"] == bad]
        if len(gg) >= 20 and len(bb) >= 20:
            d_ = wr(gg) - wr(bb)
            tot += 1
            ok += d_ > 0
            print(f"    {name:<10} {d_:+.0f}pp (n={len(gg)}/{len(bb)})")
    g2_pass = tot >= 4 and ok / tot >= 0.7
    print(f"  桶內同向 {ok}/{tot} → G2 {'PASS' if g2_pass else 'FAIL'}")
    res["g2"] = {"ok": ok, "tot": tot, "pass": g2_pass}
    if not g2_pass:
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print("\n  S3 止步於 G2 — 記錄後收檔")
        return 0

    print(f"\n  [G3] 統計關（{good} vs {bad}）")
    ga = np.array([r["c"] for r in rows if r["bk"] == good])
    ba = np.array([r["c"] for r in rows if r["bk"] == bad])
    obs = 100 * (ga.mean() - ba.mean())
    rgen = np.random.default_rng(7)
    pool = np.concatenate([ga, ba])
    null = []
    for _ in range(2000):
        p_ = rgen.permutation(pool)
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
        if len(gg) >= 12 and len(bb) >= 12:
            qsigns.append((q, round(100 * (np.mean(gg) - np.mean(bb)), 1)))
    adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
    g3_pass = pval < 0.05 and lo_ci > 0 and adverse <= 1
    print(f"    gap {obs:+.1f}pp · 置換 p={pval:.4f} · CI [{lo_ci:+.1f},{hi_ci:+.1f}]")
    print(f"    逐季 " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
          + f" → 逆風 {adverse}/{len(qsigns)}")
    print(f"    G3 {'PASS ✅ S3 取得席位' if g3_pass else 'FAIL — 記錄收檔'}")
    res["g3"] = {"gap": round(float(obs), 1), "p": pval,
                 "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                 "quarters": qsigns, "pass": g3_pass, "good_side": good}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
