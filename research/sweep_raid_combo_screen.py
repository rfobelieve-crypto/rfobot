# -*- coding: utf-8 -*-
"""SYSTEMATIC combo screen -> frozen watchlist -> shadow scoreboard.

User plan (2026-08-02): screen MANY liquidity-raid x order-flow combos,
freeze the survivors, track them in the shadow, and let forward data
decide what goes live. This is stage 1 (the screen) + the nomination.

Design that keeps the multiple-comparison problem honest:
  - the screen enumerates the FULL grid (~288 cells, all reported, no
    cherry-picking): depth {shallow<=0.25, mid, deep>0.6, any} x
    close-back {R, notR, any} x volume {V, notV, any} x attack-speed
    {fast<=5m, any} x BTC-flag {none, Q, LIQ, PA+}
  - flags are only features with a survival record (or their negations);
    thresholds are the registered/causal ones, nothing tuned here
  - NOMINATION RULE (pre-stated): eligible = n>=80 AND halves same sign
    AND (symbol-agnostic combos: BTC & ETH same sign) AND |t|>=2.5;
    rank by |t|, greedy de-dup on trade-set overlap (Jaccard>0.7), cap 8
  - the screen NOMINATES ONLY. Nothing goes live from here: nominees are
    frozen into combo_watchlist.py, the shadow scores them on FORWARD
    rows with the same clustered-CI arithmetic as Gate F, and promotion
    needs that forward evidence (the in-sample screen is just the entry
    ticket — the same two-stage shape as variant B itself).

Run: python research/sweep_raid_combo_screen.py
Out: research/results/sweep_raid_combo_screen.json (full grid + nominees)
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

import pandas as pd  # noqa: E402
import sweep_raid_menu2 as M  # noqa: E402
import sweep_raid_anatomy as A  # noqa: E402
import sweep_raid_derivs as D  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_combo_screen.json"
OOS_PQ = ROOT / "research/results/dual_model/direction_reg_oos_mse.parquet"

DEPTH = {"淺": lambda r: r["pierce"] <= 0.25,
         "中": lambda r: 0.25 < r["pierce"] <= 0.60,
         "深": lambda r: r["pierce"] > 0.60,
         "*": lambda r: True}
RFLAG = {"R": lambda r: r["reject_in_hour"] == 1,
         "¬R": lambda r: r["reject_in_hour"] == 0,
         "*": lambda r: True}
VFLAG = {"V": lambda r: r["Vhi"],
         "¬V": lambda r: not r["Vhi"],
         "*": lambda r: True}
AFLAG = {"快": lambda r: r["att_min"] <= 5,
         "*": lambda r: True}
BFLAG = {"": None,
         "Q": lambda r: r.get("q_flag") == 1,
         "LIQ": lambda r: r.get("liq_hi") is True,
         "PA": lambda r: (r.get("pred_align") or 0) > 0}


def stats(g):
    xs = [r["netR"] for r in g if r["netR"] is not None]
    if len(xs) < 40:
        return None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    t = m / (sd / math.sqrt(len(xs))) if sd > 0 else 0.0
    br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
    wr = 100 * sum(1 for x in xs if x > 0) / len(xs)
    return {"n": len(g), "n_fill": len(xs), "netR": round(m, 4),
            "t": round(t, 2), "wr": round(wr, 1), "breakout_pct": round(br, 1)}


def main() -> int:
    print("=" * 78)
    print("  COMBO SCREEN — 全網格篩選 + 事前提名規則（提名≠上線, forward 才算數）")
    print("=" * 78)
    rows = M.build()
    # BTC overlay flags
    S = D.load_state()
    dmap = {r["ts"]: r for r in D.attach(A.raids("BTC"), S)}
    lbs = sorted(v["liq_burst"] for v in dmap.values()
                 if v.get("liq_burst") is not None)
    liq_med = lbs[len(lbs) // 2]
    P = pd.read_parquet(OOS_PQ)
    for r in rows:
        if r["sym"] != "BTC":
            continue
        d = dmap.get(r["ts"])
        if d and d.get("oi_chg_raid") is not None \
                and d.get("fut_taker_signed") is not None:
            r["q_flag"] = int(d["oi_chg_raid"] < 0
                              and d["fut_taker_signed"] > 0)
        if d and d.get("liq_burst") is not None:
            r["liq_hi"] = d["liq_burst"] >= liq_med
        dt = pd.Timestamp(r["ts"], unit="s", tz="UTC")
        if dt in P.index:
            r["pred_align"] = -r["side"] * float(P.loc[dt, "pred_ret"])
    rows.sort(key=lambda r: r["ts"])
    half_ts = rows[len(rows) // 2]["ts"]

    grid = {}
    cells = []
    for dn, dp in DEPTH.items():
        for rn, rp in RFLAG.items():
            for vn, vp in VFLAG.items():
                for an, ap in AFLAG.items():
                    for bn, bp in BFLAG.items():
                        name = "∧".join(x for x in (dn if dn != "*" else "",
                                                    rn if rn != "*" else "",
                                                    vn if vn != "*" else "",
                                                    an if an != "*" else "",
                                                    bn) if x)
                        if not name:
                            continue
                        base = [r for r in rows
                                if dp(r) and rp(r) and vp(r) and ap(r)]
                        if bp is not None:
                            g = [r for r in base if r["sym"] == "BTC"
                                 and bp(r)]
                        else:
                            g = base
                        st = stats(g)
                        grid[name] = st
                        if st:
                            cells.append((name, st, g, bp is not None))
    print(f"  評估 {len(grid)} 格, 有效(n_fill>=40) {len(cells)} 格")

    # nomination
    def halves_ok(g):
        a = [r["netR"] for r in g if r["netR"] is not None
             and r["ts"] < half_ts]
        b = [r["netR"] for r in g if r["netR"] is not None
             and r["ts"] >= half_ts]
        if len(a) < 20 or len(b) < 20:
            return False
        return (sum(a) / len(a)) * (sum(b) / len(b)) > 0

    def syms_ok(g, btc_only):
        if btc_only:
            return True
        a = [r["netR"] for r in g if r["netR"] is not None
             and r["sym"] == "BTC"]
        b = [r["netR"] for r in g if r["netR"] is not None
             and r["sym"] == "ETH"]
        if len(a) < 20 or len(b) < 20:
            return False
        return (sum(a) / len(a)) * (sum(b) / len(b)) > 0

    elig = [(nm, st, g) for nm, st, g, btc in cells
            if st["n"] >= 80 and abs(st["t"]) >= 2.5
            and halves_ok(g) and syms_ok(g, btc)]
    elig.sort(key=lambda x: -abs(x[1]["t"]))
    nominees, used = [], []
    for nm, st, g in elig:
        ids = {(r["sym"], r["ts"]) for r in g}
        if any(len(ids & u) / len(ids | u) > 0.7 for u in used):
            continue
        nominees.append((nm, st))
        used.append(ids)
        if len(nominees) >= 8:
            break

    print(f"\n  合格 {len(elig)} 格 → 去重後提名 {len(nominees)} 個：")
    for nm, st in nominees:
        print(f"    {nm:<22} n={st['n']:>4} netR {st['netR']:+.3f} "
              f"t{st['t']:+.1f} WR {st['wr']:.0f}% 突破{st['breakout_pct']:.0f}%")
    OUT.write_text(json.dumps(
        {"grid": grid, "nominees": [{"name": nm, **st} for nm, st in nominees],
         "eligible": len(elig), "liq_med_insample": liq_med},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  下一步: 提名凍結進 combo_watchlist.py → shadow --combos 前瞻記帳")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
