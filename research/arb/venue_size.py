# -*- coding: utf-8 -*-
"""1.03 -- is a BIG venue vs a SMALL venue really an order of magnitude wider
than SMALL vs SMALL? (pre-registered, exploration on the scan window)

Why
---
CoinGecko's 2026 tokenized-stock report says Binance's Samsung perp traded
0.93% above Hyperliquid's on average (SK Hynix 1.03%, peaks 2.3%), and
explains it as "late-entrant venues hold higher prices because less arbitrage
capital flows in". Our own family measures 2-15 bps -- but BOTH our legs are
late entrants (Entropy io, Lighter-RH). If the report's mechanism is right,
the pair we chose is the reason our bands are thin, not the method.

Binance's USDT-M futures does not actually list equity perps (checked
2026-09-04: SPXUSDT is the memecoin, MUSDT is a memecoin), so the report's
"Binance" leg is a different product. We do not need it: OKX and Bitget list
the same Korean and US equity perps that xyz/Lighter do, and the scanner has
been recording all of them for 4 days. So the comparison is available now.

PRE-REGISTERED (written before looking at any band from this cut)
  Venue size classes, FROZEN by what the venue is, not by its numbers:
    BIG    okx, bitget                 -- established CEXes, deep books
    SMALL  every HL builder dex (io, xyz, para, mkts, ...), lighter,
           lighter-rh, and HL core     -- late entrants
  P1  For EQUITY/COMMODITY assets, median band of BIG-SMALL pairs is
      >= 3x the median band of SMALL-SMALL pairs.
      (The report implies ~10x+; 3x is the bar that would still change our
      pair choice. Choosing the bar before the run is the point.)
  P2  BIG-BIG (okx<->bitget) is the NARROWEST of the three -- the control.
      If it is not, the classification is measuring something else and P1
      means nothing.
  P3  The same ordering holds for CRYPTO assets, i.e. it is a venue-size
      effect and not an equity-only story. (Directional, reported either
      way; it does not gate P1.)

EXCLUSIONS, frozen here
  * Leveraged wrappers (CSOP...2L, ...3L, ...2S) -- a 2x product is a
    different underlying, not a price gap.
  * Pairs already excluded by the scanner's own scale guard.
  * Any pair with < 200 quote rows in the window.
  * The band is the SAME metric the promotion ranking uses (p90 of positive
    executable room per side), so this cut cannot invent a new number.

This decides ONE thing: whether to re-point the recording family at
big-vs-small pairs. It is not a verdict and opens no clock.

Run: python research/arb/venue_size.py
Out: research/results/arb_venue_size.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_rank as SR            # noqa: E402  frozen loader + band metric

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = HERE.parents[1] / "research" / "results" / "arb_venue_size.json"
BIG = {"okx", "bitget"}
LEVERAGED = re.compile(r"(CSOP|[0-9]+[LS]$|2L|3L|2S|3S)", re.I)
MIN_ROWS = 200
BAR_P1 = 3.0

EQUITY = {
    # US equities / indices / ETFs
    "NVDA", "TSLA", "AAPL", "MSTR", "COIN", "SPY", "QQQ", "SPX", "NDX100",
    "META", "GOOGL", "AMD", "INTC", "MSFT", "PLTR", "MU", "SNDK", "NBIS",
    "SPCX", "VVV", "OAI", "ANTH", "IONQ", "EWY",
    # Korean
    "SAMSUNG", "SKHYNIX", "SKHY", "SKHX", "KR200",
    # commodities / metals
    "GOLD", "XAU", "XAUT", "PAXG", "SILVER", "XAG", "COPPER", "PLATINUM",
    "PALLADIUM", "NATGAS", "BRENTOIL", "WTI", "CL", "URANIUM", "LIT",
}


def klass(a: str, b: str) -> str:
    ab = (a in BIG, b in BIG)
    return "BIG-BIG" if all(ab) else ("SMALL-SMALL" if not any(ab) else "BIG-SMALL")


def main() -> int:
    print("=" * 96)
    print("  §1.03 大所 vs 小所——價差是不是真的寬一個數量級（預註冊，掃描窗，探索）")
    print("=" * 96)
    df = SR.load()
    if df.empty:
        print("  無掃描資料")
        return 1
    df = df[~df.pair.str.upper().str.contains(LEVERAGED, regex=True, na=False)]
    rows = []
    for pair, g in df.groupby("pair"):
        if len(g) < MIN_ROWS:
            continue
        la, lb = g.leg_a.iloc[0], g.leg_b.iloc[0]
        asset = pair.split("@")[0].upper()
        best = None
        for col, dep in (("sell_edge_bps", ("a_bid_usd", "b_ask_usd")),
                         ("buy_edge_bps", ("b_bid_usd", "a_ask_usd"))):
            if col not in g.columns:
                continue
            m = SR.side_metric(g, col, dep)
            if m["band_bps"] is not None and (best is None
                                              or m["band_bps"] > best["band_bps"]):
                best = m
        if not best:
            continue
        rows.append({"pair": pair, "asset": asset, "legs": f"{la}-{lb}",
                     "klass": klass(la, lb),
                     "kind": "equity" if asset in EQUITY else "crypto",
                     "band_bps": best["band_bps"], "depth_usd": best["depth_usd"],
                     "n": len(g)})
    r = pd.DataFrame(rows)
    if r.empty:
        print("  無合格配對")
        return 1
    span = (df.ts.max() - df.ts.min()) / 86400
    print(f"  {len(r)} 個配對（≥{MIN_ROWS} 筆報價）｜跨度 {span:.2f} 天｜"
          f"帶＝凍結的排名指標（正向可成交空間 p90，取較寬側）\n")

    res = {"span_days": round(span, 2), "pairs": len(r), "cells": {}}
    for kind in ("equity", "crypto"):
        sub = r[r.kind == kind]
        if sub.empty:
            continue
        print(f"  [{kind}]  {'類別':<12}{'n配對':>6}{'帶中位':>9}{'帶p75':>9}"
              f"{'深度中位$':>12}")
        cell = {}
        for k in ("BIG-BIG", "BIG-SMALL", "SMALL-SMALL"):
            s = sub[sub.klass == k]
            if s.empty:
                continue
            cell[k] = {"pairs": len(s), "band_med": round(float(s.band_bps.median()), 2),
                       "band_p75": round(float(s.band_bps.quantile(0.75)), 2),
                       "depth_med": round(float(s.depth_usd.median()), 0)}
            print(f"           {k:<12}{len(s):>6}{cell[k]['band_med']:>9.2f}"
                  f"{cell[k]['band_p75']:>9.2f}{cell[k]['depth_med']:>12,.0f}")
        res["cells"][kind] = cell
        if kind == "equity":
            print("\n  逐配對（股票/商品，帶寬前 20）：")
            for _, x in sub.sort_values("band_bps", ascending=False).head(20).iterrows():
                print(f"    {x.pair:<30}{x.klass:<12}{x.band_bps:>8.2f} bps"
                      f"  深度 ${x.depth_usd:>10,.0f}  n={x.n}")
        print()

    eq = res["cells"].get("equity", {})
    cr = res["cells"].get("crypto", {})

    def ratio(c):
        bs, ss = c.get("BIG-SMALL"), c.get("SMALL-SMALL")
        if not bs or not ss or ss["band_med"] <= 0:
            return None
        return round(bs["band_med"] / ss["band_med"], 2)

    r_eq, r_cr = ratio(eq), ratio(cr)
    bars = {
        f"P1 股票/商品 BIG-SMALL 帶中位 ≥ {BAR_P1}× SMALL-SMALL":
            r_eq is not None and r_eq >= BAR_P1,
        "P2 對照組 BIG-BIG 最窄":
            bool(eq.get("BIG-BIG") and eq.get("BIG-SMALL") and eq.get("SMALL-SMALL")
                 and eq["BIG-BIG"]["band_med"] < min(eq["BIG-SMALL"]["band_med"],
                                                     eq["SMALL-SMALL"]["band_med"])),
    }
    print(f"  股票/商品 BIG-SMALL ÷ SMALL-SMALL = {r_eq}×"
          f"｜加密同一比值 = {r_cr}×（P3，方向性報告）")
    for k, v in bars.items():
        print(f"    {'✅' if v else '❌'} {k}")
    res.update({"ratio_equity": r_eq, "ratio_crypto": r_cr, "bars": bars})
    res["verdict"] = ("大所 vs 小所確實更寬——錄製家族該重新指向這一格"
                      if all(bars.values()) else
                      "沒有支持「寬一個數量級」——不因為這份外部報告改變配對選擇")
    print(f"  → {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
