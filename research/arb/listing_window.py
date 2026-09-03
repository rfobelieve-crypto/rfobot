# -*- coding: utf-8 -*-
"""1.05 -- the cross-listing window: is the band widest right after a SECOND
venue lists an asset the first one already had? (pre-registered)

Why this and why it is the right question
-----------------------------------------
1.03 killed "pick bigger venues" (BIG-SMALL was 1.07x SMALL-SMALL, not the
order of magnitude the external report implied) and found the real divide is
whether a SMALL venue is involved at all -- two mature CEXes sit 3.2 bps
apart. 1.01 killed "trade the US open". Both were attempts to find a
STANDING edge in a steady-state market, and both said the same thing: in
steady state the gap is arbitraged down to roughly the cost of trading it.

That is exactly what the two brothers' account says their edge never was.
Theirs was: be there in the first days of a venue/market, before the capital
arrives. CoinGecko's report says the same mechanism from the other side --
"late-entrant venues hold higher prices because less arbitrage capital flows
in ... these gaps recur in the early phase as more exchanges launch."

So the hypothesis is not about WHERE but about WHEN: the band on a pair is a
decaying function of the time since the pair came into existence, i.e. since
the second venue listed the asset.

HONEST PROVENANCE (this decides how the result may be used)
The idea came from reading the listings file, where PONS -- listed hl_core
08-31, lighter-rh 09-01, lighter 09-02 -- turned out to be the widest
zero-fee pair in the whole scan (45.95 bps). That is a hypothesis found in
the data. So the cut below is EXPLORATION and can only decide whether to
open a forward clock; confirmation has to come from listings that happen
after registration. Same rule as variant M (TODO 0.94).

PRE-REGISTERED PREDICTIONS (written before any band was computed by age)
  P1 DECAY    median band of pairs aged < 24h is >= 2x the median band of
              the same pairs aged > 72h (paired: the SAME pair, so cross-
              sectional differences between assets cannot produce it)
  P2 CONTROL  pairs with no listing event in the window (present since the
              scan began) show no such decay: their first-24h-equivalent
              slice vs later slice is within 1.3x. Without this, P1 is just
              "the market was wilder on the day most listings happened".
  P3 BREADTH  the decay holds for >= 60% of the individual listing events,
              not one dominant event.
SURVIVAL: all three. Passing opens a forward clock and a listing watcher;
it does NOT authorise a trade.

INSTRUMENT NOTE, fixed before this ran (2026-09-04)
2,006 of the 2,023 recorded "listed" events were a venue's whole market list
reappearing after a failed fetch. The detector now labels those
mass_reappear and this script reads only `listed`. Without that fix every
number below would be noise.

Run: python research/arb/listing_window.py
Out: research/results/arb_listing_window.json
"""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_rank as SR            # noqa: E402  frozen loader + band metric

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "arb_listing_window.json"
LISTINGS = ROOT.parent / "entropy-arb" / "logs" / "scan" / "listings.csv"
YOUNG_H, OLD_H = 24.0, 72.0
BAR_DECAY, BAR_CONTROL, BAR_BREADTH = 2.0, 1.3, 0.60
MIN_ROWS_CELL = 30


def band_of(g: pd.DataFrame) -> float | None:
    """The frozen ranking band: p90 of positive executable room, wider side."""
    best = None
    for col, dep in (("sell_edge_bps", ("a_bid_usd", "b_ask_usd")),
                     ("buy_edge_bps", ("b_bid_usd", "a_ask_usd"))):
        if col not in g.columns:
            continue
        m = SR.side_metric(g, col, dep)
        if m["band_bps"] is not None and (best is None or m["band_bps"] > best):
            best = m["band_bps"]
    return best


def main() -> int:
    print("=" * 96)
    print("  §1.05 交叉上市窗口——第二個場館掛上去之後，帶是不是隨時間衰減（預註冊，探索）")
    print("=" * 96)
    if not LISTINGS.exists():
        print("  無 listings.csv")
        return 1
    ev = [r for r in csv.DictReader(io.open(LISTINGS, encoding="utf-8", errors="replace"))
          if r["event"] == "listed"]
    # the historical file predates the mass_* labels, so re-apply the same
    # rule here: >=5 in one (ts, venue) is a fetch artifact, not a listing.
    cnt = defaultdict(int)
    for r in ev:
        cnt[(r["ts"], r["venue"])] += 1
    ev = [r for r in ev if cnt[(r["ts"], r["venue"])] < 5]
    print(f"  真上市事件 {len(ev)} 筆（已剔除抓取產物）")
    for r in ev:
        print(f"    {datetime.fromtimestamp(int(r['ts']), timezone.utc):%m-%d %H:%M}"
              f"  {r['venue']:<12}{r['symbol']}")

    df = SR.load()
    if df.empty:
        print("  無掃描資料")
        return 1
    # a listing makes a PAIR only if some other venue already had the asset;
    # the pair name is <ASSET>@<legA>-<legB>, so match on the asset and on
    # the listing venue appearing as one of the legs.
    born = {}
    for r in ev:
        sym, ven, ts = r["symbol"].upper(), r["venue"], int(r["ts"])
        for pair in df.pair.unique():
            asset, _, legs = pair.partition("@")
            if asset.upper() != sym:
                continue
            v = ven.replace("hl_", "").replace("core", "HL")
            if v.lower() in legs.lower() or (v == "HL" and legs.startswith("HL")):
                born[pair] = min(ts, born.get(pair, 1 << 62))
    print(f"\n  對應到掃描中的配對 {len(born)} 個")

    rows, per_event = [], []
    for pair, t0 in sorted(born.items()):
        g = df[df.pair == pair]
        young = g[(g.ts - t0) / 3600 <= YOUNG_H]
        old = g[(g.ts - t0) / 3600 > OLD_H]
        if len(young) < MIN_ROWS_CELL or len(old) < MIN_ROWS_CELL:
            per_event.append({"pair": pair, "young_n": len(young), "old_n": len(old),
                              "skip": "樣本不足"})
            continue
        by, bo = band_of(young), band_of(old)
        if by is None or bo is None or bo <= 0:
            per_event.append({"pair": pair, "skip": "帶算不出來"})
            continue
        rows.append({"pair": pair, "young": by, "old": bo, "ratio": by / bo,
                     "young_n": len(young), "old_n": len(old)})
    res = {"events": len(ev), "pairs_matched": len(born), "measured": len(rows)}
    if rows:
        r = pd.DataFrame(rows).sort_values("ratio", ascending=False)
        print(f"\n  {'配對':<30}{'<24h 帶':>9}{'>72h 帶':>9}{'比值':>7}{'n young/old':>14}")
        for _, x in r.iterrows():
            print(f"  {x.pair:<30}{x.young:>9.2f}{x.old:>9.2f}{x.ratio:>7.2f}"
                  f"{x.young_n:>7.0f}/{x.old_n:<6.0f}")
        med = float(r.ratio.median())
        breadth = float((r.ratio > 1.0).mean())
        res.update({"median_ratio": round(med, 2), "breadth": round(breadth, 2),
                    "rows": rows})
    else:
        med = breadth = None
        print("\n  沒有任何配對同時有 <24h 與 >72h 的足夠樣本——"
              "掃描窗只有 4 天，而多數上市發生在最後一天。")

    # control: pairs present for the whole window, cut at the same clock
    ctrl = []
    t_ref = int(df.ts.min())
    for pair, g in df.groupby("pair"):
        if pair in born or len(g) < MIN_ROWS_CELL * 2:
            continue
        young = g[(g.ts - t_ref) / 3600 <= YOUNG_H]
        old = g[(g.ts - t_ref) / 3600 > OLD_H]
        if len(young) < MIN_ROWS_CELL or len(old) < MIN_ROWS_CELL:
            continue
        by, bo = band_of(young), band_of(old)
        if by is None or bo is None or bo <= 0:
            continue
        ctrl.append(by / bo)
    c_med = float(pd.Series(ctrl).median()) if ctrl else None
    res["control_median_ratio"] = round(c_med, 2) if c_med else None
    res["control_n"] = len(ctrl)
    print(f"\n  對照組（整段都在的配對，同一把尺）：n={len(ctrl)}  中位比值 {c_med}")

    bars = {
        f"P1 新配對 <24h 帶 ≥ {BAR_DECAY}× >72h": med is not None and med >= BAR_DECAY,
        f"P2 對照組無同樣衰減（< {BAR_CONTROL}×）":
            c_med is not None and c_med < BAR_CONTROL,
        f"P3 ≥{BAR_BREADTH:.0%} 的事件同向": breadth is not None and breadth >= BAR_BREADTH,
    }
    for k, v in bars.items():
        print(f"    {'✅' if v else '❌'} {k}")
    res["bars"] = bars
    res["verdict"] = ("存活：開前瞻時鐘＋上市監看" if all(bars.values())
                      else "未通過——不開時鐘（多半是樣本不足，見上）")
    print(f"  → {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
