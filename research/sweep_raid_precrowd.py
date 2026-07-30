# -*- coding: utf-8 -*-
"""Pre-raid crowding — the OI x CVD joint state BEFORE the raid (user asked
"有試過 OI 加 CVD 嗎", 2026-07-30).

What was already covered: the raid-HOUR joint state (quadrants: Q = OI down +
taker with break = stop-driven, confirmed) and Q inside combos (QV richest).
What was NOT covered: the BUILDUP. oi_chg_4h and raid-hour CVD were tested
individually (dead alone); their PRE-RAID joint state was never read.

One named mechanism (squeeze fuel), stated before looking:
  raid of a high hurts shorts. If in the 4h BEFORE the raid OI grew while
  CVD pushed AGAINST the eventual break direction, the crowd was building
  wrong-way positions into the level -> the raid is the squeeze that clears
  them -> expect the best reversal netR there.
  Conversely OI up + pre-CVD WITH the break = trend pressing into the level
  -> expect the highest breakout rate.

Cuts are signs only (no tuned thresholds). BTC only (Coinglass OI).
Verdict bar: both named predictions right + halves same direction.

Run: python research/sweep_raid_precrowd.py
Out: research/results/sweep_raid_precrowd.json
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

OUT = ROOT / "research/results/sweep_raid_precrowd.json"

Q_ZH = {
    ("up", "with"): "OI↑ 前流順破向 (趨勢壓進水位→突破形)",
    ("up", "anti"): "OI↑ 前流逆破向 (反向擁擠=擠壓燃料→反轉形)",
    ("dn", "with"): "OI↓ 前流順破向",
    ("dn", "anti"): "OI↓ 前流逆破向",
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
        g = [r for r in rows if r.get("pquad") == key]
        s = stat(g)
        rec["/".join(key)] = s
        if s:
            print(f"    {zh:<30} n={s['n']:>4}  突破{s['breakout_pct']:>4.0f}%  "
                  f"反轉{s['reversal_pct']:>3.0f}%  netR|回踩 {s['netR']:+.3f}")
        else:
            print(f"    {zh:<30} n={len(g)} thin")
    return rec


def main() -> int:
    print("=" * 78)
    print("  PRE-RAID CROWDING — 獵取前 4h 的 OI x CVD 佈局（單一具名假設）")
    print("=" * 78)
    S = D.load_state()
    rows = []
    for r in D.attach(A.raids("BTC"), S):
        if r.get("oi_chg_4h") is None:
            continue
        hh = r["ts"] // 3600
        s = r["side"]
        num = vol = 0.0
        ok = True
        for k in range(1, 5):
            fb, fs = S["fut_b"].get(hh - k), S["fut_s"].get(hh - k)
            if fb is None or fs is None:
                ok = False
                break
            num += fb - fs
            vol += fb + fs
        if not ok or vol <= 0:
            continue
        r["pre4_cvd_signed"] = s * num / vol
        r["pquad"] = ("up" if r["oi_chg_4h"] > 0 else "dn",
                      "with" if r["pre4_cvd_signed"] > 0 else "anti")
        rows.append(r)
    print(f"  BTC raids with 4h buildup coverage: {len(rows)}")

    res = {"all": show(rows, "[全部]")}
    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    res["H1"] = show(rows_sorted[:half], "[前半]")
    res["H2"] = show(rows_sorted[half:], "[後半]")

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("\n  預先聲明: OI↑+前流逆破=反轉 netR 最好; OI↑+前流順破=突破率最高。"
          "\n  判準: 兩個預測都對 + 前後半同向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
