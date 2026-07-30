# -*- coding: utf-8 -*-
"""The textbook OI x CVD quadrants at raid moments — the user's combination.

User pushback (2026-07-30): single-variable terciles are not how order-flow
people actually read the tape — they read COMBINATIONS, canonically OI
change x aggression direction. That is a fair methodological objection:
round 2 tested oi_chg_raid and fut_taker_signed separately and both looked
dead alone, but the classic claim is about their JOINT state.

ONE named, pre-stated hypothesis (not a combo search) — the standard matrix,
read in the break direction s:
  OI UP   + taker WITH the break   = new money driving  -> 真突破 (highest
                                     breakout rate)
  OI DOWN + taker WITH the break   = covering/stop-driven push (軋空/止損
                                     驅動) -> exhausts, best reversal netR
  OI UP   + taker AGAINST          = absorption/fading into the break
  OI DOWN + taker AGAINST          = positions leaving both ways, apathy

Cuts: OI up/down = sign of oi_chg_raid; taker with/against = sign of
fut_taker_signed. Sign cuts, no tuned thresholds. Also reported with the
magnitude version (|oi_chg| top half) and inside the shallow-pierce subset,
plus first/second-half consistency. BTC only (Coinglass OI), 2,741 raids.

Run: python research/sweep_raid_quadrants.py
Out: research/results/sweep_raid_quadrants.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_quadrants.json"

Q_ZH = {
    ("up", "with"): "OI↑ 追價順破 (新錢進場→真突破形)",
    ("dn", "with"): "OI↓ 追價順破 (止損驅動→反轉形)",
    ("up", "anti"): "OI↑ 追價逆破 (吸收/對做)",
    ("dn", "anti"): "OI↓ 追價逆破 (雙向撤退)",
}


def stat(rs):
    n = len(rs)
    if n < 60:
        return None
    br = 100 * sum(1 for r in rs if r["cls"] == "BREAKOUT") / n
    rv = 100 * sum(1 for r in rs if r["cls"] == "REVERSAL") / n
    nets = [r["netR"] for r in rs if r["netR"] is not None]
    m = sum(nets) / len(nets) if nets else float("nan")
    return {"n": n, "breakout_pct": br, "reversal_pct": rv, "netR": m}


def show(rows, label):
    print(f"\n  {label}")
    rec = {}
    for key, zh in Q_ZH.items():
        g = [r for r in rows if r.get("quad") == key]
        s = stat(g)
        rec["/".join(key)] = s
        if s:
            print(f"    {zh:<28} n={s['n']:>4}  突破{s['breakout_pct']:>4.0f}%  "
                  f"反轉{s['reversal_pct']:>3.0f}%  netR|回踩 {s['netR']:+.3f}")
        else:
            print(f"    {zh:<28} n={len(g)} thin")
    return rec


def main() -> int:
    print("=" * 78)
    print("  OI x CVD QUADRANTS AT RAIDS — 教科書組合, 單一具名假設")
    print("=" * 78)
    S = D.load_state()
    rows = D.attach(A.raids("BTC"), S)
    rows = [r for r in rows
            if r.get("oi_chg_raid") is not None
            and r.get("fut_taker_signed") is not None]
    for r in rows:
        r["quad"] = ("up" if r["oi_chg_raid"] > 0 else "dn",
                     "with" if r["fut_taker_signed"] > 0 else "anti")
    print(f"  BTC raids with OI+CVD coverage: {len(rows)}")
    res = {"all": show(rows, "[全部]")}

    big = [r for r in rows if abs(r["oi_chg_raid"]) >= 0.3]
    res["material_oi"] = show(big, "[|OI 變化| >= 0.3% — 排除噪音級變動]")

    sh = [r for r in rows if r["pierce"] <= 0.25]
    res["shallow"] = show(sh, "[淺穿越子集內 — 濾網之上還加不加值]")

    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    res["H1"] = show(rows_sorted[:half], "[前半]")
    res["H2"] = show(rows_sorted[half:], "[後半]")

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  預先聲明的教科書預測: OI↑順破=突破率最高; OI↓順破=反轉 netR 最好。"
          "\n  判準: 兩個預測方向正確 + 前後半一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
