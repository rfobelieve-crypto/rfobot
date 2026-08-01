# -*- coding: utf-8 -*-
"""The operator's SEQUENCE hypothesis, tested as one joint condition.

User (2026-08-02): 反轉 = 獵取時 OI 先下降(止損)且清算放大 → 獵取後 CVD
快速反轉且 OI 增加(反向新倉) → 往反向走；延續 = 相反。

Every stage was validated separately (Q quadrant, liq burst, chase veto,
OI-bleed veto); this asks whether the STAGES CHAIN — the named composite:

  raid state  R+ = OI down AND taker with break during the raid hour (Q)
              [liq-burst variant reported as a sub-line: R+ ∧ liq >= median]
  post state  P+ = CVD flipped against the break (pd_cvd < 0)
                   AND OI rising (pd_oi_chg > 0) over the gap hours

  Named predictions (stated before running):
    H_rev:  R+ ∧ P+  -> best netR (the user's full reversal sequence)
    H_cont: R- ∧ chase (pd_cvd > 0)  -> worst netR / most breakouts

Grids: F1 = fills with >=1 complete gap hour (netR); F2 = survivors not
retested by +2h, post state measured on [raid+1h, +2h) (resolution).
2x2 all-cells reported; strict cells small-n flagged; halves on the named
cells. BTC only (Coinglass). ~8 looks.

Run: python research/sweep_raid_sequence.py
Out: research/results/sweep_raid_sequence.json
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
from sweep_raid_postderiv import feats_over  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_sequence.json"


def cell(rows, label, target):
    n = len(rows)
    if n < 25:
        print(f"    {label:<26} n={n} thin")
        return None
    if target == "netR":
        xs = [r["netR"] for r in rows if r["netR"] is not None]
        m = sum(xs) / len(xs)
        wr = 100 * sum(1 for x in xs if x > 0) / len(xs)
        print(f"    {label:<26} n={n:>4}  netR {m:+.3f} / WR {wr:.0f}%")
        return {"n": n, "netR": round(m, 3), "wr": round(wr, 1)}
    br = 100 * sum(1 for r in rows if r["cls"] == "BREAKOUT") / n
    print(f"    {label:<26} n={n:>4}  突破 {br:.0f}%")
    return {"n": n, "breakout_pct": round(br, 1)}


def main() -> int:
    print("=" * 78)
    print("  SEQUENCE — 使用者的三段式假設：獵取(OI↓+清算)→獵取後(CVD翻+OI回升)")
    print("=" * 78)
    S = D.load_state()
    from sweep_raid_anatomy import raids as _raids
    dmap = {r["ts"]: r for r in D.attach(_raids("BTC"), S)}
    lbs = sorted(v["liq_burst"] for v in dmap.values()
                 if v.get("liq_burst") is not None)
    liq_med = lbs[len(lbs) // 2]
    rr = raids_with_fill("BTC")
    res = {}

    def attach_states(r, h0, h1):
        d = dmap.get(r["ts"])
        if not d or d.get("oi_chg_raid") is None \
                or d.get("fut_taker_signed") is None:
            return None
        f = feats_over(S, r["side"], h0, h1)
        if not f or f.get("pd_cvd") is None or f.get("pd_oi_chg") is None:
            return None
        x = dict(r)
        x["Rplus"] = d["oi_chg_raid"] < 0 and d["fut_taker_signed"] > 0
        x["liq_hi"] = (d.get("liq_burst") or 0) >= liq_med
        x["Pplus"] = f["pd_cvd"] < 0 and f["pd_oi_chg"] > 0
        x["chase"] = f["pd_cvd"] > 0
        return x

    # ── F1: entry quality ────────────────────────────────────────────────
    f1 = []
    for r in rr:
        if r["fill_ts"] is None:
            continue
        h0, h1 = r["ts"] // 3600 + 1, r["fill_ts"] // 3600
        if h1 <= h0:
            continue
        x = attach_states(r, h0, h1)
        if x:
            f1.append(x)
    f1.sort(key=lambda r: r["ts"])
    print(f"\n  [F1] 進場品質（n={len(f1)}, 全體均 netR "
          f"{sum(r['netR'] for r in f1)/len(f1):+.3f}）")
    grid = {}
    for rname, rpred in (("R+", lambda r: r["Rplus"]),
                         ("R-", lambda r: not r["Rplus"])):
        for pname, ppred in (("P+", lambda r: r["Pplus"]),
                             ("P-", lambda r: not r["Pplus"])):
            g = [r for r in f1 if rpred(r) and ppred(r)]
            grid[f"{rname}{pname}"] = cell(g, f"{rname} ∧ {pname}", "netR")
    res["F1_grid"] = grid
    strict = [r for r in f1 if r["Rplus"] and r["liq_hi"] and r["Pplus"]]
    res["F1_strict"] = cell(strict, "嚴格序列 R+∧清算高∧P+", "netR")
    cont = [r for r in f1 if not r["Rplus"] and r["chase"]]
    res["F1_cont"] = cell(cont, "延續假設 R-∧追殺", "netR")
    half = len(f1) // 2
    print("  [halves] 具名格")
    for tag, seg in (("H1", f1[:half]), ("H2", f1[half:])):
        cell([r for r in seg if r["Rplus"] and r["Pplus"]],
             f"{tag} R+∧P+", "netR")
        cell([r for r in seg if not r["Rplus"] and r["chase"]],
             f"{tag} R-∧追殺", "netR")

    # ── F2: resolution among +2h survivors ──────────────────────────────
    f2 = []
    for r in rr:
        if r["fill_ts"] is not None and r["fill_ts"] <= r["ts"] + 7200:
            continue
        x = attach_states(r, r["ts"] // 3600 + 1, r["ts"] // 3600 + 2)
        if x:
            f2.append(x)
    base = 100 * sum(1 for r in f2 if r["cls"] == "BREAKOUT") / max(len(f2), 1)
    print(f"\n  [F2] 決議（+2h 倖存者 n={len(f2)}, 基準突破 {base:.0f}%）")
    grid2 = {}
    for rname, rpred in (("R+", lambda r: r["Rplus"]),
                         ("R-", lambda r: not r["Rplus"])):
        for pname, ppred in (("P+", lambda r: r["Pplus"]),
                             ("P-", lambda r: not r["Pplus"])):
            g = [r for r in f2 if rpred(r) and ppred(r)]
            grid2[f"{rname}{pname}"] = cell(g, f"{rname} ∧ {pname}", "cls")
    res["F2_grid"] = grid2
    res["F2_strict"] = cell([r for r in f2 if r["Rplus"] and r["liq_hi"]
                             and r["Pplus"]], "嚴格序列", "cls")
    res["F2_cont"] = cell([r for r in f2 if not r["Rplus"] and r["chase"]],
                          "延續假設 R-∧追殺", "cls")

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  具名預測: R+∧P+ 的 netR 全場最佳；R-∧追殺 最差且突破率最高。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
