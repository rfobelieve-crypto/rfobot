# -*- coding: utf-8 -*-
"""Raid combination search — which JOINT states best separate reversal from
continuation. The user's explicit request (2026-07-30), run with guardrails.

Building blocks are ONLY the features that already survived earlier named
tests (no fresh dredging):
  P  pierce <= 0.25 ATR          (registered filter; resolution king)
  A  attack <= 5 minutes         (round 4; both symbols monotone)
  R  raid bar closed back inside (round 4; SFP close-back, 0% vs 22%)
  V  attack-minute volume, top tercile per symbol (round 4; netR +0.15 both)
  Q  OI down + taker with break  (quadrants; stop-driven push, halves-stable)
  L  liq_burst top tercile       (derivs; violence, halves-stable)  [BTC]

Tier 1 combos over {P,A,R,V} on BTC+ETH; tier 2 adds {Q,L} on BTC only
(Coinglass). 25 declared combos x 2 targets — expect a handful of chance
standouts; the bar for naming anything: n >= 80, material separation, and
halves-consistent. Exploratory ranking to DESIGN a future pre-registration
(the October minute-cancel checkpoint) — nothing here changes any current
registration.

Run: python research/sweep_raid_combos.py
Out: research/results/sweep_raid_combos.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_attack as K  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_combos.json"

FLAGS_ZH = {"P": "淺穿越", "A": "攻擊≤5分", "R": "收回內側",
            "V": "攻擊量能高", "Q": "OI↓順破(止損驅動)", "L": "清算爆量"}


def build():
    rows = []
    for sym, canon in A.SYMS.items():
        m1 = K.fetch_1m(sym)
        flow, _i, _c = A.load_flow(canon)
        rr = K.attach_attack(K.raids_with_level(sym), flow, m1)
        vs = sorted(r["att_vshock"] for r in rr if r.get("att_vshock") is not None)
        vcut = vs[2 * len(vs) // 3]
        for r in rr:
            r["P"] = r["pierce"] <= 0.25
            r["A"] = r["att_min"] <= 5
            r["R"] = r["reject_in_hour"] == 1
            r["V"] = (r.get("att_vshock") or 0) >= vcut
        rows += rr
    # BTC-only derivatives flags, joined by raid timestamp
    S = D.load_state()
    dmap = {}
    for r in D.attach(A.raids("BTC"), S):
        dmap[r["ts"]] = r
    lb = sorted(v["liq_burst"] for v in dmap.values()
                if v.get("liq_burst") is not None)
    lcut = lb[2 * len(lb) // 3] if lb else None
    for r in rows:
        d = dmap.get(r["ts"]) if r["sym"] == "BTC" else None
        if d and d.get("oi_chg_raid") is not None and d.get("fut_taker_signed") is not None:
            r["Q"] = d["oi_chg_raid"] < 0 and d["fut_taker_signed"] > 0
        else:
            r["Q"] = None
        if d and d.get("liq_burst") is not None and lcut:
            r["L"] = d["liq_burst"] >= lcut
        else:
            r["L"] = None
    return rows


def stat(rs):
    n = len(rs)
    if n < 80:
        return None
    br = 100 * sum(1 for r in rs if r["cls"] == "BREAKOUT") / n
    rv = 100 * sum(1 for r in rs if r["cls"] == "REVERSAL") / n
    nets = [r["netR"] for r in rs if r["netR"] is not None]
    if len(nets) < 40:
        return None
    m = sum(nets) / len(nets)
    sd = math.sqrt(sum((x - m) ** 2 for x in nets) / (len(nets) - 1))
    return {"n": n, "breakout_pct": br, "reversal_pct": rv, "netR": m,
            "t": m / (sd / math.sqrt(len(nets)))}


COMBOS = (["P", "A", "R", "V", "PA", "PR", "PV", "AR", "AV", "RV",
           "PAV", "PRV", "ARV", "PAR", "PARV"]
          + ["Q", "L", "QP", "QA", "QV", "QL", "LP", "LV", "QPV", "LPV"])


def sel(rows, combo):
    need_btc = any(c in combo for c in "QL")
    out = []
    for r in rows:
        if need_btc and r["sym"] != "BTC":
            continue
        ok = True
        for c in combo:
            v = r.get(c)
            if v is None or not v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def main() -> int:
    rows = build()
    base = stat(rows)
    print("=" * 78)
    print("  RAID COMBO SEARCH — 25 declared combos; bar = n>=80 + separation "
          "+ halves")
    print("=" * 78)
    print(f"  universe: {len(rows)} raids (BTC+ETH); base 突破 "
          f"{base['breakout_pct']:.0f}%  反轉 {base['reversal_pct']:.0f}%  "
          f"netR {base['netR']:+.3f}\n")

    res = {}
    table = []
    for cb in COMBOS:
        s = stat(sel(rows, cb))
        res[cb] = s
        if s:
            table.append((cb, s))
    # rank by netR t (quality) — print all valid, no cherry-pick
    print("  依 netR t 排序（品質軸）")
    print(f"  {'combo':<7}{'條件':<34}{'n':>5}{'突破%':>6}{'反轉%':>6}"
          f"{'netR':>8}{'t':>7}{'佔全體%':>8}")
    for cb, s in sorted(table, key=lambda x: -x[1]["t"]):
        zh = "+".join(FLAGS_ZH[c] for c in cb)
        print(f"  {cb:<7}{zh:<34}{s['n']:>5}{s['breakout_pct']:>6.0f}"
              f"{s['reversal_pct']:>6.0f}{s['netR']:>+8.3f}{s['t']:>+7.2f}"
              f"{100*s['n']/len(rows):>7.0f}%")

    # halves consistency for the top 3 by t
    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    top3 = [cb for cb, _ in sorted(table, key=lambda x: -x[1]["t"])[:3]]
    print(f"\n  [halves] 前後半 — top3: {', '.join(top3)}")
    for cb in top3:
        for tag, seg in (("H1", rows_sorted[:half]), ("H2", rows_sorted[half:])):
            s = stat(sel(seg, cb))
            if s:
                print(f"  {tag} {cb:<7} n={s['n']:>4}  突破{s['breakout_pct']:.0f}%  "
                      f"反轉{s['reversal_pct']:.0f}%  netR {s['netR']:+.3f} "
                      f"(t{s['t']:+.1f})")
            else:
                print(f"  {tag} {cb:<7} thin")

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
