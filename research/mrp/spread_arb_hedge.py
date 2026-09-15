# -*- coding: utf-8 -*-
"""價差套利：換對沖腿 ＋ 審 para 那個唯一的正值（2026-09-13）

===========================================================================
第一版的表裡最大的數字是我們自己的費用
===========================================================================
`venue_spread_arb.py` 用 binance 當對沖腿，結果 10 個場館裡 9 個毛邊際為負。
而綁束不是價差不夠寬，是 **binance 的 taker 5.00 bps 比幾乎每個場館的
半價差都大**。

所以這支掃「對沖腿換成誰」。其中有一個選項是**散戶專屬**的：
**Lighter Standard 的 taker 是 0.00 bps**，而它的代價（300ms 延遲、
60 req/min）**只擋做市，不擋小時級的對沖**（CLAUDE.md §HFT 已記，
而我們現在用 Premium 跑慢策略＝白付 2.80）。機構進不了那一級
——官方把 Standard 的適用對象寫成零售／中低頻／方向性／Funding Arb。

配置（照 small-trader-alpha-1 的「the favorite」）：
    不流動場館**掛單**（收半價差、付 maker 費）
    流動場館**吃單**對沖（付半價差、付 taker 費）
    毛邊際 = illiq_半價差 − maker(illiq) − liq_半價差 − taker(liq)

===========================================================================
第二件事：審 para，因為它是唯一的正值
===========================================================================
自己的規矩（mistake.md 2026-08-02）：**跟先驗矛盾的漂亮結果要查儀器，
符合先驗的漂亮結果更要查。** para 的半價差中位 17.968 bps 是 binance 的
9 倍，而它的**凍結比例 34%** —— 那正好是 backtest-audit #13 說的形狀
（「API 壞掉的標的逆選擇最低」）。所以這支逐 ticker 攤開它：
凍結比例、半價差、深度，以及**寬價差是不是集中在幾個死掉的 ticker 上**。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 2026-09-15: arb location comes from research/arb_home.py (ARB_HOME overrides), not a hardcoded path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arb_home  # noqa: E402
arb_home.add_to_path()

from venue_spread_arb import LIQUID, load_quotes  # noqa: E402

# 候選對沖腿。`lighter-std` 是同一個場館的 Standard 費率級別
# （arblib/fees.py 已有這一列，taker 0.0 / maker 0.0）。
HEDGES = ["binance", "okx", "bitget", "lighter", "lighter-std", "HL"]


def main():
    from arblib import fees
    p = load_quotes()

    # 逐場館的半價差中位與凍結比例（與第一版同算法）
    from venue_spread_arb import MIN_ROWS, frozen_frac
    st = {}
    for v, g in p.groupby("venue"):
        if len(g) < MIN_ROWS:
            continue
        st[v] = dict(hs=float(g.hs_bps.median()), frozen=frozen_frac(g),
                     n=len(g), syms=g.sym.nunique())

    print("對沖腿掃描：每單位名目的毛邊際（掛單在左欄、吃單對沖在上欄）")
    print("  毛邊際 = 掛單場館半價差 − 它的 maker − 對沖場館半價差 − 它的 taker\n")
    hdr = [h for h in HEDGES if h in st or h.endswith("-std")]
    print("  %-12s %9s %7s  %s" % ("掛單場館", "半價差", "凍結",
                                   "".join("%11s" % h for h in hdr)))
    rows = []
    for v in sorted(st, key=lambda k: -st[k]["hs"]):
        cells, best, bestv = [], -1e9, None
        for h in hdr:
            # lighter-std 的簿口就是 lighter 的簿口，只有費率級別不同
            hv = "lighter" if h == "lighter-std" else h
            if hv not in st or hv == v:
                cells.append("%11s" % "—")
                continue
            g = (st[v]["hs"] - fees.fee_bps(v, True)
                 - st[hv]["hs"] - fees.fee_bps(h, False))
            cells.append("%11.2f" % g)
            if g > best:
                best, bestv = g, h
        rows.append(dict(venue=v, hs=st[v]["hs"], frozen=st[v]["frozen"],
                         best=best, best_hedge=bestv))
        print("  %-12s %9.3f %6.0f%%  %s"
              % (v, st[v]["hs"], 100 * st[v]["frozen"], "".join(cells)))

    r = pd.DataFrame(rows)
    print("\n最好的對沖腿 ＋ 毛邊際為正的掛單場館")
    pos = r[(r.best > 0)].sort_values("best", ascending=False)
    for _, x in pos.iterrows():
        tag = "  **凍結 %.0f%%，先查儀器**" % (100 * x.frozen) \
            if x.frozen > 0.25 else ""
        print("  %-12s 毛邊際 **%+.2f bps**（對沖在 %s）%s"
              % (x.venue, x.best, x.best_hedge, tag))
    if pos.empty:
        print("  **沒有** —— 任何對沖腿組合都為負")

    # ── 審 para：逐 ticker ────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("審 para（唯一的正值，凍結 34%）—— 逐 ticker")
    print("=" * 92)
    g = p[p.venue == "para"]
    out = []
    for sym, s in g.groupby("sym"):
        if len(s) < 200:
            continue
        s = s.sort_values("t")
        same = ((s.bid.diff() == 0) & (s.ask.diff() == 0)).iloc[1:]
        out.append(dict(sym=sym, n=len(s), hs=float(s.hs_bps.median()),
                        frozen=float(same.mean()),
                        # 最長連續不動（取樣數 x 180 秒）
                        run=int(max((len(list(v)) for k, v in
                                     __import__("itertools").groupby(same)
                                     if k), default=0))))
    d = pd.DataFrame(out).sort_values("hs", ascending=False)
    print("  %-14s %7s %9s %8s %14s" % ("ticker", "n", "半價差", "凍結",
                                        "最長不動"))
    for _, x in d.iterrows():
        print("  %-14s %7d %9.3f %7.0f%% %11.1f 小時"
              % (x.sym, x.n, x.hs, 100 * x.frozen, x.run * 180 / 3600.0))
    live = d[d.frozen <= 0.25]
    print("\n  **凍結 <= 25%% 的 ticker：%d / %d**" % (len(live), len(d)))
    if len(live):
        print("  它們的半價差中位 **%.3f bps**（全體 %.3f）"
              % (live.hs.median(), d.hs.median()))
        print("  -> 寬價差%s集中在凍結的那些上"
              % ("不" if live.hs.median() > d.hs.median() * 0.7 else "**確實**"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
