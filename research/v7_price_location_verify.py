# -*- coding: utf-8 -*-
"""Adversarial verification of the price-location findings (2026-08-02).

The claims under attack:
  A1 ahead-far (clean runway) > ahead-near      (65% vs 57%)
  A2 behind-near (support at the back) > rest   (+12-14pp both halves)
  A3 composite (behind-near ∧ ahead-far) 66% vs 60% base

Battery (each can kill or downgrade a claim):
  1 thirds stability      — the veto's T1 inversion showed halves can hide
  2 direction split       — is it all DOWN (the known-strong class)?
  3 regime split          — structure or regime proxy?
  4 volatility control    — ATR-normalized distance could proxy vol;
                            effect must survive within ATR-regime terciles
  5 veto overlap          — composite must add WITHIN raid-context buckets,
                            else it is the same effect wearing a new hat
  6 permutation placebo   — shuffle outcomes 2000x: the observed composite
                            gap must be far outside the null (mechanics)

Verdict standard: a claim keeps its seat only if no check reverses its
sign in a well-populated slice; shrinkage without reversal = noted.

Run: python research/v7_price_location_verify.py
Out: research/results/v7_price_location_verify.json
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

import numpy as np  # noqa: E402
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shared.db import get_db_conn  # noqa: E402
from v7_price_location import pool_lifecycle  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_price_location_verify.json"


def build_rows():
    bars = SC.load_csv(str(LT.CACHE / "BTCUSDT_1h.csv"))
    ts2i = {b[0]: i for i, b in enumerate(bars)}
    atr = SC.atr14(bars)
    cl = [b[SC.C] for b in bars]
    pools = pool_lifecycle(bars)
    by_hh = defaultdict(list)
    for r in raids_with_fill("BTC"):
        by_hh[r["ts"] // 3600].append(r["side"])
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, regime, correct "
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
        above = [p[2] for p in pools if p[0] <= j
                 and (p[1] is None or p[1] > j) and p[2] > c]
        below = [p[2] for p in pools if p[0] <= j
                 and (p[1] is None or p[1] > j) and p[2] < c]
        ahead = ((min(above) - c) / atr[j] if up and above else
                 (c - max(below)) / atr[j] if (not up) and below else None)
        behind = ((c - max(below)) / atr[j] if up and below else
                  (min(above) - c) / atr[j] if (not up) and above else None)
        # raid-context bucket (the veto's definition, 4h)
        ctx = "none"
        for k in range(0, 5):
            sides = by_hh.get(ts // 3600 - k)
            if sides:
                sd = sides[0]
                fade = ((sd == 1 and not up) or (sd == -1 and up))
                ctx = "fade" if fade else "follow"
                break
        rows.append({"ts": ts, "dir": s["direction"], "regime": s["regime"] or "?",
                     "c": int(s["correct"]), "ahead": ahead, "behind": behind,
                     "ctx": ctx, "volp": atr[j] / c})
    rows.sort(key=lambda r: r["ts"])
    return rows


def wr(g):
    return 100 * sum(r["c"] for r in g) / len(g) if g else None


def main() -> int:
    print("=" * 78)
    print("  LOCATION VERIFY — 專攻三個宣稱（sign 反轉即擊殺, 縮水記錄在案）")
    print("=" * 78)
    rows = build_rows()
    n = len(rows)
    av = sorted(r["ahead"] for r in rows if r["ahead"] is not None)
    bv = sorted(r["behind"] for r in rows if r["behind"] is not None)
    a_lo, a_hi = av[len(av) // 3], av[2 * len(av) // 3]
    b_lo = bv[len(bv) // 3]

    def a_far(r):
        return r["ahead"] is not None and r["ahead"] >= a_hi

    def a_near(r):
        return r["ahead"] is not None and r["ahead"] <= a_lo

    def b_near(r):
        return r["behind"] is not None and r["behind"] <= b_lo

    def comp(r):
        return b_near(r) and a_far(r)

    res = {}
    print(f"  n={n} · 整體 {wr(rows):.0f}% · 複合格 n={sum(1 for r in rows if comp(r))}\n")

    def gap_line(seg, tag):
        af, an = [r for r in seg if a_far(r)], [r for r in seg if a_near(r)]
        bn, br_ = [r for r in seg if b_near(r)], [r for r in seg if not b_near(r) and r["behind"] is not None]
        cp, ncp = [r for r in seg if comp(r)], [r for r in seg if not comp(r)]
        def d(x, y):
            return (wr(x) - wr(y)) if (x and y and len(x) >= 12 and len(y) >= 12) else None
        parts = []
        for nm, v in (("A1 ahead遠-近", d(af, an)), ("A2 behind近-餘", d(bn, br_)),
                      ("A3 複合-其餘", d(cp, ncp))):
            parts.append(f"{nm} {v:+.0f}pp" if v is not None else f"{nm} thin")
        print(f"  {tag:<14}" + " | ".join(parts)
              + f"  (n={len(seg)}, comp n={len(cp)})")
        return {"A1": d(af, an), "A2": d(bn, br_), "A3": d(cp, ncp)}

    print("  [1] 三等分")
    third = n // 3
    for i, tag in ((0, "T1"), (1, "T2"), (2, "T3")):
        seg = rows[i * third:(i + 1) * third if i < 2 else n]
        res[f"third_{tag}"] = gap_line(seg, tag)

    print("\n  [2] 方向")
    for d_ in ("UP", "DOWN"):
        res[f"dir_{d_}"] = gap_line([r for r in rows if r["dir"] == d_], d_)

    print("\n  [3] regime（n≥100 者）")
    for g in sorted({r["regime"] for r in rows}):
        seg = [r for r in rows if r["regime"] == g]
        if len(seg) >= 100:
            res[f"reg_{g}"] = gap_line(seg, g[:12])

    print("\n  [4] 波動度控制（ATR/價 三分位內）")
    vv = sorted(r["volp"] for r in rows)
    v1, v2 = vv[n // 3], vv[2 * n // 3]
    for nm, pr in (("低波動", lambda r: r["volp"] <= v1),
                   ("中波動", lambda r: v1 < r["volp"] < v2),
                   ("高波動", lambda r: r["volp"] >= v2)):
        res[f"vol_{nm}"] = gap_line([r for r in rows if pr(r)], nm)

    print("\n  [5] veto 情境內（複合格要在桶內仍加分才不是換皮）")
    for b in ("none", "fade", "follow"):
        res[f"ctx_{b}"] = gap_line([r for r in rows if r["ctx"] == b], b)

    print("\n  [6] 置換檢定（洗牌結果 2000 次, 複合格 gap 的虛無分佈）")
    rng = np.random.default_rng(7)
    cs = np.array([r["c"] for r in rows])
    mask = np.array([comp(r) for r in rows])
    obs = 100 * (cs[mask].mean() - cs[~mask].mean())
    null = []
    for _ in range(2000):
        p_ = rng.permutation(cs)
        null.append(100 * (p_[mask].mean() - p_[~mask].mean()))
    null = np.array(null)
    pval = float((null >= obs).mean())
    print(f"    觀察 gap {obs:+.1f}pp · 虛無 95 分位 {np.percentile(null, 95):+.1f}pp"
          f" · p={pval:.3f}")
    res["permutation"] = {"obs_pp": round(float(obs), 1), "p": pval}

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
