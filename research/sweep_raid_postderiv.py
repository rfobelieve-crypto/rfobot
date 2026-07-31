# -*- coding: utf-8 -*-
"""Post-raid DERIVATIVES dynamics — OI unwind, second liquidation wave,
funding drift, futures CVD, and their JOINT OI x CVD state, in the window
between the raid hour and the retest.

TODO 0.469 item 1 + user's standing methodological point (2026-08-02):
joint states must be tested, not just marginals — the during-raid OI x CVD
quadrant survived (Q), the pre-raid one failed; the POST-raid quadrant is
the missing third window. Coinglass is hourly, so this is BTC-only and the
window is whole hours: fills at bar+1 have no complete gap hour and are
excluded (same selection caveat as the flow-side post study — reported).

  F1 (entry quality, fills with >=1 complete gap hour): features over
     [raid_hour+1, fill_hour), target netR
       pd_oi_chg    OI % change (prediction: OI keeps FALLING = the
                    deleveraging continues -> fatter fade)
       pd_cvd      s x taker share over gap hours (prediction: flow
                    flipped against the break -> fatter fade; the
                    futures-CVD twin of the flow-side chase veto)
       pd_liq2     gap liq/hour vs 24h baseline (second wave -> ?)
       pd_fund     funding drift x s (further into break = crowding)
     JOINT: OI(down/up) x CVD(with/against break) — 4 cells, all reported.
     Named: OI down AND CVD against = unwind+flip -> best netR;
            OI up AND CVD with = fresh positioning chase -> worst.
  F2 (resolution among survivors not retested by sweep+2h): the same
     features measured on hour [raid+1h, raid+2h) -> eventual BREAKOUT vs
     retest, plus the quadrant.

~10 looks; bar = monotone/halves for continuous, named-cell + halves for
the quadrant. Descriptive research — no registration touched.

Run: python research/sweep_raid_postderiv.py
Out: research/results/sweep_raid_postderiv.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_derivs as D  # noqa: E402
from sweep_raid_postflow import raids_with_fill  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_postderiv.json"

Q_ZH = {
    ("dn", "anti"): "OI↓∧CVD逆破 (去槓桿+流翻轉→具名最佳)",
    ("dn", "with"): "OI↓∧CVD順破",
    ("up", "anti"): "OI↑∧CVD逆破",
    ("up", "with"): "OI↑∧CVD順破 (新倉追價→具名最差)",
}


def feats_over(S, s, h0, h1):
    """Derivative features over whole hours [h0, h1)."""
    if h1 <= h0:
        return None
    oi0, oi1 = S["oi"].get(h0 - 1), S["oi"].get(h1 - 1)
    fb = [S["fut_b"].get(h) for h in range(h0, h1)]
    fs = [S["fut_s"].get(h) for h in range(h0, h1)]
    if oi0 is None or oi1 is None or any(v is None for v in fb + fs):
        return None
    tot = sum(fb) + sum(fs)
    liq = [(S["liq_l"].get(h, 0) + S["liq_s"].get(h, 0)) for h in range(h0, h1)]
    base = [S["liq_l"].get(h0 - 1 - k, 0) + S["liq_s"].get(h0 - 1 - k, 0)
            for k in range(24)]
    base = [b for b in base if b > 0]
    f0, f1v = S["funding"].get(h0 - 1), S["funding"].get(h1 - 1)
    out = {
        "pd_oi_chg": (oi1 / oi0 - 1) * 100,
        "pd_cvd": s * (sum(fb) - sum(fs)) / tot if tot > 0 else None,
        "pd_liq2": ((sum(liq) / len(liq)) / (sum(base) / len(base))
                    if base else None),
        "pd_fund": (s * (f1v - f0)
                    if f0 is not None and f1v is not None else None),
    }
    out["quad"] = ("dn" if out["pd_oi_chg"] < 0 else "up",
                   "with" if (out["pd_cvd"] or 0) > 0 else "anti")
    return out


def terc(rows, key, target):
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    if len(vals) < 90:
        return None, f"  {key:<10} thin (n={len(vals)})"
    lo_c, hi_c = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    if lo_c == hi_c:
        return None, f"  {key:<10} degenerate — skipped"
    parts, rec = [], {}
    for nm, pr in (("low", lambda v: v <= lo_c), ("mid", lambda v: lo_c < v < hi_c),
                   ("high", lambda v: v >= hi_c)):
        g = [r for r in rows if r.get(key) is not None and pr(r[key])]
        if not g:
            continue
        if target == "netR":
            xs = [r["netR"] for r in g if r["netR"] is not None]
            m = sum(xs) / len(xs)
            wr = 100 * sum(1 for x in xs if x > 0) / len(xs)
            rec[nm] = {"n": len(g), "netR": m, "wr": wr}
            parts.append(f"{nm} {m:+.3f}/{wr:.0f}% (n={len(g)})")
        else:
            br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
            rec[nm] = {"n": len(g), "breakout_pct": br}
            parts.append(f"{nm} 突破{br:.0f}% (n={len(g)})")
    return rec, f"  {key:<10}" + " | ".join(parts)


def quad_table(rows, target, label):
    print(f"  [{label}] OI×CVD 四象限")
    rec = {}
    for key, zh in Q_ZH.items():
        g = [r for r in rows if r.get("quad") == key]
        if len(g) < 30:
            print(f"    {zh:<34} n={len(g)} thin")
            rec["/".join(key)] = None
            continue
        if target == "netR":
            xs = [r["netR"] for r in g if r["netR"] is not None]
            m = sum(xs) / len(xs)
            wr = 100 * sum(1 for x in xs if x > 0) / len(xs)
            rec["/".join(key)] = {"n": len(g), "netR": m, "wr": wr}
            print(f"    {zh:<34} n={len(g):>4}  netR {m:+.3f} / WR {wr:.0f}%")
        else:
            br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
            rec["/".join(key)] = {"n": len(g), "breakout_pct": br}
            print(f"    {zh:<34} n={len(g):>4}  突破 {br:.0f}%")
    return rec


def main() -> int:
    print("=" * 78)
    print("  POST-RAID DERIVATIVES — 獵取後的 OI/CVD/清算/資金費率 + 聯合象限 (BTC)")
    print("=" * 78)
    S = D.load_state()
    rr = raids_with_fill("BTC")
    res = {}

    f1 = []
    for r in rr:
        if r["fill_ts"] is None:
            continue
        h0, h1 = r["ts"] // 3600 + 1, r["fill_ts"] // 3600
        f = feats_over(S, r["side"], h0, h1)
        if f:
            r2 = dict(r)
            r2.update(f)
            f1.append(r2)
    f1.sort(key=lambda r: r["ts"])
    print(f"\n  [F1] 進場品質（gap ≥1 完整小時, n={len(f1)}）")
    for k in ("pd_oi_chg", "pd_cvd", "pd_liq2", "pd_fund"):
        rec, line = terc(f1, k, "netR")
        res[f"F1_{k}"] = rec
        print(line)
    res["F1_quad"] = quad_table(f1, "netR", "F1")
    half = len(f1) // 2
    print("  [F1 halves] 具名象限")
    for tag, seg in (("H1", f1[:half]), ("H2", f1[half:])):
        for key in (("dn", "anti"), ("up", "with")):
            g = [r for r in seg if r.get("quad") == key]
            xs = [r["netR"] for r in g if r["netR"] is not None]
            if len(xs) >= 20:
                print(f"    {tag} {'∧'.join(key):<10} netR {sum(xs)/len(xs):+.3f} (n={len(xs)})")

    f2 = []
    for r in rr:
        if r["fill_ts"] is not None and r["fill_ts"] <= r["ts"] + 7200:
            continue
        f = feats_over(S, r["side"], r["ts"] // 3600 + 1, r["ts"] // 3600 + 2)
        if f:
            r2 = dict(r)
            r2.update(f)
            f2.append(r2)
    f2.sort(key=lambda r: r["ts"])
    base_br = 100 * sum(1 for r in f2 if r["cls"] == "BREAKOUT") / max(len(f2), 1)
    print(f"\n  [F2] 決議預測（+2h 倖存者, n={len(f2)}, 基準突破 {base_br:.0f}%）")
    for k in ("pd_oi_chg", "pd_cvd", "pd_liq2", "pd_fund"):
        rec, line = terc(f2, k, "cls")
        res[f"F2_{k}"] = rec
        print(line)
    res["F2_quad"] = quad_table(f2, "cls", "F2")

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
