# -*- coding: utf-8 -*-
"""Second combination menu (user paste, 2026-07-30) — the three combos that
are genuinely NEW and testable with retained data. Everything else on the
A-L menu is already covered, data-impossible, or post-event (mapping in the
conversation).

  MENU-A  attack intensity x absorption efficiency
          declared: reversal quality best where ABSORPTION is high;
          breakout rate highest where absorption LOW and attack volume
          extreme (low-resistance push). BTC+ETH.
  MENU-I  absorption x OI exhaustion (stop-run deleveraging)
          declared: ABS_hi AND OI-down(raid hour) = best netR of the four
          cells. BTC only (Coinglass OI).
  MENU-L  leverage environment filter — OI level percentile vs trailing 30d
          declared: high-OI environment -> higher reversal rate / netR
          (more fuel to burn). BTC only.

Cuts: terciles / signs only, per symbol, no tuned thresholds. 3 tests x 2
targets = 6 looks — chance alone can hand one of them a pattern; the bar
stays named-prediction-correct + halves same direction + material size.

Skipped from the menu ON PURPOSE (reasons in chat): K/H post-raid OI path
(entry-time causality muddy: conditioning on slow retests = selected
subpopulation; adjacent buildup family just FAILed), C/D/J (order-book
queue / trade-size data not retained — October depth_deltas checkpoint),
F (breakout-side confirmation, post-event), G (ML vector — premature until
the rule set survives a forward pre-registration).

Run: python research/sweep_raid_menu2.py
Out: research/results/sweep_raid_menu2.json
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
import sweep_raid_attack as K  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_menu2.json"


def stat(rs):
    n = len(rs)
    if n < 60:
        return None
    br = 100 * sum(1 for r in rs if r["cls"] == "BREAKOUT") / n
    rv = 100 * sum(1 for r in rs if r["cls"] == "REVERSAL") / n
    nets = [r["netR"] for r in rs if r["netR"] is not None]
    m = sum(nets) / len(nets) if nets else float("nan")
    return {"n": n, "breakout_pct": br, "reversal_pct": rv, "netR": m}


def cell(rows, pred, label):
    s = stat([r for r in rows if pred(r)])
    if s:
        print(f"    {label:<34} n={s['n']:>4}  突破{s['breakout_pct']:>4.0f}%  "
              f"反轉{s['reversal_pct']:>3.0f}%  netR|回踩 {s['netR']:+.3f}")
    else:
        print(f"    {label:<34} thin")
    return s


def build():
    rows = []
    for sym, canon in A.SYMS.items():
        m1 = K.fetch_1m(sym)
        flow, _i, _c = A.load_flow(canon)
        rr = K.attach_attack(K.raids_with_level(sym), flow, m1)
        # absorption recomputed here (round-3 definition): signed taker share
        # into the break over the raid HOUR / penetration depth
        keep = []
        for r in rr:
            m0 = r["ts"] // 60
            win = [flow[m] for m in range(m0, m0 + 60) if m in flow]
            vol = sum(w[0] for w in win)
            if len(win) < 40 or vol <= 0:
                continue
            into = r["side"] * sum(w[1] for w in win) / vol
            r["absorption"] = into / max(r["pierce"], 0.10)
            keep.append(r)
        ab = sorted(x["absorption"] for x in keep)
        vs = sorted(x["att_vshock"] for x in keep
                    if x.get("att_vshock") is not None)
        a_hi, a_lo = ab[2 * len(ab) // 3], ab[len(ab) // 3]
        v_hi = vs[2 * len(vs) // 3]
        for r in keep:
            r["ABS"] = ("hi" if r["absorption"] >= a_hi
                        else "lo" if r["absorption"] <= a_lo else "mid")
            r["Vhi"] = (r.get("att_vshock") or 0) >= v_hi
        rows += keep
    # BTC Coinglass: raid-hour OI change + OI level percentile (30d)
    S = D.load_state()
    oi = S["oi"]
    dmap = {r["ts"]: r for r in D.attach(A.raids("BTC"), S)}
    for r in rows:
        r["oi_dn"] = None
        r["oi_pct"] = None
        if r["sym"] != "BTC":
            continue
        d = dmap.get(r["ts"])
        if d and d.get("oi_chg_raid") is not None:
            r["oi_dn"] = d["oi_chg_raid"] < 0
        hh = r["ts"] // 3600
        v = oi.get(hh)
        win = [oi[hh - k] for k in range(1, 721) if hh - k in oi]
        if v is not None and len(win) >= 300:
            r["oi_pct"] = 100 * sum(1 for x in win if x < v) / len(win)
    return rows


def halves(rows, pred, label):
    rs = sorted(rows, key=lambda r: r["ts"])
    half = len(rs) // 2
    for tag, seg in (("H1", rs[:half]), ("H2", rs[half:])):
        s = stat([r for r in seg if pred(r)])
        if s:
            print(f"    [{tag}] {label:<28} n={s['n']:>4}  "
                  f"突破{s['breakout_pct']:>4.0f}%  netR {s['netR']:+.3f}")


def main() -> int:
    rows = build()
    res = {}
    base = stat(rows)
    print("=" * 78)
    print("  MENU-2 COMBOS — A(吸收x強度) / I(吸收xOI衰竭) / L(OI環境)")
    print("  6 looks; 判準 = 具名預測正確 + 前後半同向 + 幅度")
    print("=" * 78)
    print(f"  universe: {len(rows)} raids (BTC+ETH); base 突破 "
          f"{base['breakout_pct']:.0f}% 反轉 {base['reversal_pct']:.0f}% "
          f"netR {base['netR']:+.3f}")

    print("\n  [MENU-A] 吸收 x 攻擊量能（2x2, BTC+ETH）")
    res["A"] = {
        "abs_hi": cell(rows, lambda r: r["ABS"] == "hi", "吸收高（全部）"),
        "abs_hi_v": cell(rows, lambda r: r["ABS"] == "hi" and r["Vhi"],
                         "吸收高 ∧ 量能極端"),
        "abs_hi_nv": cell(rows, lambda r: r["ABS"] == "hi" and not r["Vhi"],
                          "吸收高 ∧ 量能普通"),
        "abs_lo_v": cell(rows, lambda r: r["ABS"] == "lo" and r["Vhi"],
                         "吸收低 ∧ 量能極端 →預測突破最高"),
        "abs_lo_nv": cell(rows, lambda r: r["ABS"] == "lo" and not r["Vhi"],
                          "吸收低 ∧ 量能普通"),
    }
    halves(rows, lambda r: r["ABS"] == "hi" and r["Vhi"], "吸收高∧量能極端")

    print("\n  [MENU-I] 吸收 x raid 小時 OI（BTC）")
    btc = [r for r in rows if r["oi_dn"] is not None]
    res["I"] = {
        "hi_dn": cell(btc, lambda r: r["ABS"] == "hi" and r["oi_dn"],
                      "吸收高 ∧ OI↓ →預測 netR 最佳"),
        "hi_up": cell(btc, lambda r: r["ABS"] == "hi" and not r["oi_dn"],
                      "吸收高 ∧ OI↑"),
        "lo_dn": cell(btc, lambda r: r["ABS"] == "lo" and r["oi_dn"],
                      "吸收低 ∧ OI↓"),
        "lo_up": cell(btc, lambda r: r["ABS"] == "lo" and not r["oi_dn"],
                      "吸收低 ∧ OI↑"),
    }
    halves(btc, lambda r: r["ABS"] == "hi" and r["oi_dn"], "吸收高∧OI↓")

    print("\n  [MENU-L] OI 環境分位（30d, BTC terciles）")
    lp = [r for r in rows if r["oi_pct"] is not None]
    res["L"] = {
        "low": cell(lp, lambda r: r["oi_pct"] < 33.3, "OI 低分位 (<33)"),
        "mid": cell(lp, lambda r: 33.3 <= r["oi_pct"] < 66.7, "OI 中分位"),
        "high": cell(lp, lambda r: r["oi_pct"] >= 66.7,
                     "OI 高分位 (>=67) →預測反轉較高"),
    }
    halves(lp, lambda r: r["oi_pct"] >= 66.7, "OI 高分位")

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
