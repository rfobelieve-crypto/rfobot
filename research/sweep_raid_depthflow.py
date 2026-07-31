# -*- coding: utf-8 -*-
"""Depth x flow script table — what does the order flow of a SHALLOW vs
DEEP raid look like, and which combination identifies the coming script.

User direction (2026-08-01): variant D (close-back AND volume) is where
order flow earns its seat — sharpen it by conditioning on 獵殺深淺 (price
behaviour) so a raid can be read as a SCRIPT at decision time.

Depth buckets (pre-stated, no tuning): 淺 <=0.25 ATR (the registered B
cut), 中 0.25-0.60, 深 >0.60 (round number near the historical deep-
tercile boundary ~0.53-0.55; not load-bearing). Flow state within each
depth: R = closed back inside the raid hour, V = attack-minute volume in
the symbol's top tercile (research framing; the causal per-symbol-median
version is what variant D runs live).

Layers:
  L1 descriptive — 獵殺深淺的訂單流長怎樣: median attack minutes / vshock
     / taker share / absorption per depth bucket.
  L2 script table — 12 cells (3 depth x R/V combinations), ALL reported:
     n, breakout%, reversal%, netR|retested. Named hypotheses stated up
     front; verdicts need both-symbol + halves agreement on headline cells.
       H1 淺∧R∧V   -> best reversal quality (the D recipe's home turf)
       H2 深∧¬R∧V  -> continuation (real breakout being driven)
       H3 深∧R     -> trapped-breakout spring: deep pierce that still
                      closed back = the failed deep raid — does flow
                      rescue deep pierces? (genuinely new question)
Banner: 12 cells x 2 targets ~= 24 looks — several chance patterns
expected; only the three NAMED hypotheses can produce verdicts, the rest
is a reported map. BTC+ETH, ~100d.

Run: python research/sweep_raid_depthflow.py
Out: research/results/sweep_raid_depthflow.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_raid_menu2 as M  # noqa: E402  (build(): flags + absorption)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_depthflow.json"


def depth_of(p: float) -> str:
    return "淺" if p <= 0.25 else ("中" if p <= 0.60 else "深")


def stat(rs):
    n = len(rs)
    if n < 40:
        return None
    br = 100 * sum(1 for r in rs if r["cls"] == "BREAKOUT") / n
    rv = 100 * sum(1 for r in rs if r["cls"] == "REVERSAL") / n
    nets = [r["netR"] for r in rs if r["netR"] is not None]
    m = sum(nets) / len(nets) if nets else float("nan")
    return {"n": n, "breakout_pct": br, "reversal_pct": rv, "netR": m}


def med(rs, key):
    vs = sorted(r[key] for r in rs if r.get(key) is not None)
    return vs[len(vs) // 2] if vs else None


def cell_line(rs, label):
    s = stat(rs)
    if not s:
        return None, f"    {label:<16} n={len(rs)} thin"
    return s, (f"    {label:<16} n={s['n']:>4}  突破{s['breakout_pct']:>4.0f}%  "
               f"反轉{s['reversal_pct']:>3.0f}%  netR {s['netR']:+.3f}")


def main() -> int:
    print("=" * 78)
    print("  DEPTH x FLOW — 獵殺深淺 × 訂單流 → 劇本識別（24 looks, 只有 3 個具名假設可下判決）")
    print("=" * 78)
    rows = M.build()
    for r in rows:
        r["depth"] = depth_of(r["pierce"])
        r["R"] = r["reject_in_hour"] == 1
    res = {}

    print("\n  [L1] 各深度的訂單流長相（中位數）")
    print(f"  {'深度':<4}{'n':>6}{'攻擊分鐘':>9}{'量能倍數':>9}{'追價佔比':>9}{'吸收':>8}{'突破%':>7}")
    for d in ("淺", "中", "深"):
        g = [r for r in rows if r["depth"] == d]
        br = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
        print(f"  {d:<4}{len(g):>6}{med(g,'att_min'):>9.0f}"
              f"{med(g,'att_vshock'):>9.2f}{med(g,'att_taker'):>9.3f}"
              f"{med(g,'absorption'):>8.2f}{br:>6.0f}%")
        res[f"L1_{d}"] = {"n": len(g), "att_min": med(g, "att_min"),
                          "vshock": med(g, "att_vshock"),
                          "taker": med(g, "att_taker"),
                          "absorption": med(g, "absorption"),
                          "breakout_pct": br}

    print("\n  [L2] 劇本表（12 格全報告；R=收回內側, V=量能高·幣內三分位）")
    for d in ("淺", "中", "深"):
        print(f"  ── {d}穿越 ──")
        for rname, rpred in (("R", lambda r: r["R"]), ("¬R", lambda r: not r["R"])):
            for vname, vpred in (("V", lambda r: r["Vhi"]),
                                 ("¬V", lambda r: not r["Vhi"])):
                g = [r for r in rows if r["depth"] == d
                     and rpred(r) and vpred(r)]
                s, line = cell_line(g, f"{d}∧{rname}∧{vname}")
                res[f"L2_{d}_{rname}_{vname}"] = s
                print(line)

    # named-hypothesis verification: both symbols + halves on headline cells
    print("\n  [具名假設驗證] 雙幣 + 前後半")
    named = {
        "H1 淺∧R∧V": lambda r: r["depth"] == "淺" and r["R"] and r["Vhi"],
        "H2 深∧¬R∧V": lambda r: r["depth"] == "深" and not r["R"] and r["Vhi"],
        "H3 深∧R": lambda r: r["depth"] == "深" and r["R"],
    }
    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    half = len(rows_sorted) // 2
    for name, pred in named.items():
        for tag, seg in (("BTC", [r for r in rows if r["sym"] == "BTC"]),
                         ("ETH", [r for r in rows if r["sym"] == "ETH"]),
                         ("H1半", rows_sorted[:half]), ("H2半", rows_sorted[half:])):
            _, line = cell_line([r for r in seg if pred(r)], f"{name}[{tag}]")
            print(line)

    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
