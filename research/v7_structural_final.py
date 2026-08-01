# -*- coding: utf-8 -*-
"""FINAL verification — the three surviving V7 structural claims, all
through the same hardest statistical battery (2026-08-02, operator: 在做
最後一次驗證). The composite already died under permutation; the
survivors get the identical treatment plus bootstrap CIs and a quarter-
by-quarter sign table.

  V  raid-chase veto     follow-the-break vs everything else
  A1 wall ahead          ahead-near tercile vs ahead-far tercile
  A2 support behind      behind-near tercile vs the rest

For each: observed WR gap, permutation p (2000 shuffles of outcomes),
bootstrap 95% CI of the gap (2000 resamples), and per-quarter gap signs.
Keep-the-seat bar: permutation p < 0.05 AND bootstrap CI excluding zero
AND no more than one adverse quarter among well-populated quarters.

Run: python research/v7_structural_final.py
Out: research/results/v7_structural_final.json
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
from v7_price_location_verify import build_rows  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/v7_structural_final.json"


def main() -> int:
    print("=" * 78)
    print("  FINAL — 三個存活宣稱的最終統計關（置換 p + bootstrap CI + 逐季）")
    print("=" * 78)
    rows = build_rows()
    n = len(rows)
    av = sorted(r["ahead"] for r in rows if r["ahead"] is not None)
    bv = sorted(r["behind"] for r in rows if r["behind"] is not None)
    a_lo, a_hi = av[len(av) // 3], av[2 * len(av) // 3]
    b_lo = bv[len(bv) // 3]

    claims = {
        "V 追突破veto": (lambda r: r["ctx"] == "follow",
                        lambda r: r["ctx"] != "follow"),
        "A1 前方有牆": (lambda r: r["ahead"] is not None and r["ahead"] <= a_lo,
                       lambda r: r["ahead"] is not None and r["ahead"] >= a_hi),
        # slot order is (bad_pred, good_pred): behind-NEAR is the GOOD side
        "A2 背靠支撐": (lambda r: r["behind"] is not None and r["behind"] > b_lo,
                       lambda r: r["behind"] is not None and r["behind"] <= b_lo),
    }
    # sign convention: gap = WR(good side) - WR(bad side), positive = claim
    good_bad = {"V 追突破veto": ("rest", "follow"),
                "A1 前方有牆": ("far", "near"),
                "A2 背靠支撐": ("near", "rest")}

    rng = np.random.default_rng(7)
    res = {}
    print(f"  n={n} · 整體 WR {100*np.mean([r['c'] for r in rows]):.0f}%\n")
    for name, (bad_pred, good_pred) in claims.items():
        bad = np.array([r["c"] for r in rows if bad_pred(r)])
        good = np.array([r["c"] for r in rows if good_pred(r)])
        obs = 100 * (good.mean() - bad.mean())
        pool = np.concatenate([good, bad])
        ng = len(good)
        null = []
        for _ in range(2000):
            p_ = rng.permutation(pool)
            null.append(100 * (p_[:ng].mean() - p_[ng:].mean()))
        pval = float((np.array(null) >= obs).mean())
        boots = []
        for _ in range(2000):
            g_ = rng.choice(good, len(good), replace=True)
            b_ = rng.choice(bad, len(bad), replace=True)
            boots.append(100 * (g_.mean() - b_.mean()))
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        # quarters
        qsigns = []
        byq = {}
        for r in rows:
            q = f"{datetime.fromtimestamp(r['ts'], timezone.utc):%Y-Q}" + str(
                (datetime.fromtimestamp(r['ts'], timezone.utc).month - 1) // 3 + 1)
            byq.setdefault(q, []).append(r)
        for q in sorted(byq):
            g_ = [r["c"] for r in byq[q] if good_pred(r)]
            b_ = [r["c"] for r in byq[q] if bad_pred(r)]
            if len(g_) >= 12 and len(b_) >= 12:
                d_ = 100 * (np.mean(g_) - np.mean(b_))
                qsigns.append((q, round(float(d_), 1)))
        adverse = sum(1 for _q, d_ in qsigns if d_ < 0)
        keep = pval < 0.05 and lo_ci > 0 and adverse <= 1
        gtag, btag = good_bad[name]
        print(f"  {name:<12} gap({gtag}−{btag}) {obs:+.1f}pp · 置換 p={pval:.4f}"
              f" · bootstrap CI [{lo_ci:+.1f}, {hi_ci:+.1f}]")
        print(f"    逐季: " + "  ".join(f"{q} {d_:+.0f}" for q, d_ in qsigns)
              + f"  → 逆風季 {adverse}/{len(qsigns)}")
        print(f"    verdict: {'✅ 保留席位' if keep else '⚠️ 未達最終關'}")
        res[name] = {"gap_pp": round(float(obs), 1), "p": pval,
                     "ci": [round(float(lo_ci), 1), round(float(hi_ci), 1)],
                     "quarters": qsigns, "adverse": adverse,
                     "verdict": "KEEP" if keep else "WEAK"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
