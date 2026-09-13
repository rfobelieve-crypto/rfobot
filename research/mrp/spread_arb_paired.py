# -*- coding: utf-8 -*-
"""價差套利的 Gate 0：**同一個 ticker 兩條腿**，帶深度與凍結（2026-09-13）

===========================================================================
更正 `spread_arb_hedge.py` 的一個嚴重缺陷
===========================================================================
那一支把對沖腿的半價差用了**場館全體中位**（binance 1.99 / lighter 3.97），
然後跟掛單場館的全體中位相減。**跨場館對沖需要的是同一個 ticker 在對沖
場館的價差，而且那個 ticker 必須真的掛在那裡。**

para 上面那 19 個是**美股永續**（SOFI / SMCI / MELI / AVGO / LRCX …）。
如果 Lighter 沒有掛 SOFI，那條「對沖在 lighter-std」的腿**根本不存在**，
而 +12.50 bps 是拿兩個不同標的的價差相減得到的數字。

這是 mistake.md 2026-09-07「配對設計對差值有效、對水準無效」的同一族：
**一個在 A 構造下算的量拿去配 B。** 這支逐 ticker 重算。

===========================================================================
這支算什麼
===========================================================================
對每一個 (ticker, 掛單場館, 對沖場館) 且**兩邊都真的有報價**：

    毛邊際 = 掛單腿半價差 − maker(掛單腿) − 對沖腿半價差 − taker(對沖腿)

並同時報三件會殺死它的東西：
  * **兩條腿的凍結比例**（對沖腿凍結也是謊）
  * **深度**（factor-research #10：容量是獨立的一關，不是成本的附註）
  * **同時有報價的時間佔比**（兩邊都活著才交易得了）

固定成本（提款／gas／資金費）仍然不在這裡 —— 這是上界。
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "C:/Users/rfo/Desktop/flowbot/arb")

BAR_SEC = 180
MIN_BOTH = 1000        # 兩邊同時有報價的最少格數
FROZEN_CAP = 0.25      # 任一腿凍結超過這個就不解讀


def load_full():
    """逐 (場館, ticker, 時間) 的頂檔 ＋ **深度**。"""
    from arblib import scan_rank
    d = scan_rank.load()
    parts = []
    for pre in ("a", "b"):
        c = ["ts", "pair", "leg_%s" % pre, "%s_bid" % pre, "%s_ask" % pre,
             "%s_bid_usd" % pre, "%s_ask_usd" % pre]
        x = d[c].copy()
        x.columns = ["ts", "pair", "venue", "bid", "ask", "bid_usd", "ask_usd"]
        parts.append(x)
    p = pd.concat(parts, ignore_index=True)
    del parts, d
    p["sym"] = p["pair"].str.split("@").str[0]
    p = p.drop(columns=["pair"])
    p = p[(p.bid > 0) & (p.ask > 0) & (p.ask >= p.bid)]
    p["t"] = (p.ts // BAR_SEC) * BAR_SEC
    p = p.groupby(["venue", "sym", "t"], as_index=False).last()
    p["mid"] = (p.bid + p.ask) / 2.0
    p["hs"] = (p.ask - p.bid) / 2.0 / p.mid * 1e4
    p["dep"] = p[["bid_usd", "ask_usd"]].min(axis=1)
    return p


def frozen(s):
    if len(s) < 50:
        return np.nan
    s = s.sort_values("t")
    return float((((s.bid.diff() == 0) & (s.ask.diff() == 0)).iloc[1:]).mean())


def main():
    from arblib import fees
    p = load_full()
    print("面板：%s 列｜%d 場館｜%d ticker"
          % (format(len(p), ","), p.venue.nunique(), p.sym.nunique()))

    # 逐 (venue, sym) 先算統計，再配對
    st = {}
    for (v, s), g in p.groupby(["venue", "sym"]):
        if len(g) < 300:
            continue
        st[(v, s)] = dict(n=len(g), hs=float(g.hs.median()),
                          dep=float(g.dep.median()), fz=frozen(g),
                          idx=set(g.t.values))

    rows = []
    syms = sorted({s for _, s in st})
    for s in syms:
        vs = [v for (v, ss) in st if ss == s]
        for a, b in itertools.permutations(vs, 2):       # a 掛單、b 對沖
            A, B = st[(a, s)], st[(b, s)]
            both = len(A["idx"] & B["idx"])
            if both < MIN_BOTH:
                continue
            # Lighter 的簿口同一本，只有費率級別不同 -> 對沖腿另算 std 版
            for lab, hb in ((b, b), ("lighter-std", b)) if b == "lighter" \
                    else ((b, b),):
                g = (A["hs"] - fees.fee_bps(a, True)
                     - B["hs"] - fees.fee_bps(lab, False))
                rows.append(dict(
                    sym=s, post=a, hedge=lab, both=both,
                    hs_post=A["hs"], hs_hedge=B["hs"],
                    fz_post=A["fz"], fz_hedge=B["fz"],
                    dep=min(A["dep"], B["dep"]), gross=g))
    r = pd.DataFrame(rows)
    print("配對：%s 個 (ticker, 掛單場館, 對沖場館) 且兩邊同時有報價 >= %d 格"
          % (format(len(r), ","), MIN_BOTH))

    print("\n自曝檢查")
    print("  S1 任一腿凍結 > %.0f%% 的配對：%d / %d（**不解讀**）"
          % (100 * FROZEN_CAP,
             int(((r.fz_post > FROZEN_CAP) | (r.fz_hedge > FROZEN_CAP)).sum()),
             len(r)))
    live = r[(r.fz_post <= FROZEN_CAP) & (r.fz_hedge <= FROZEN_CAP)]
    print("  S2 **para 的美股永續有幾個在別的場館也有**："
          "%d 個 ticker" % live[live.post == "para"].sym.nunique())

    pos = live[live.gross > 0].sort_values("gross", ascending=False)
    print("\n" + "=" * 104)
    print("毛邊際為正（同一 ticker 兩條腿、兩腿都不凍結、未扣固定成本）")
    print("=" * 104)
    if pos.empty:
        print("  **一個都沒有。**")
    else:
        print("  %-10s %-11s %-12s %7s %9s %9s %9s %10s"
              % ("ticker", "掛單", "對沖", "共同格", "掛單價差", "對沖價差",
                 "深度$", "毛邊際"))
        for _, x in pos.head(25).iterrows():
            print("  %-10s %-11s %-12s %7d %9.2f %9.2f %9.0f %10.2f"
                  % (x.sym, x.post, x.hedge, x.both, x.hs_post, x.hs_hedge,
                     x.dep, x.gross))
        print("\n  共 **%d** 個配對為正，涵蓋 **%d** 個 ticker、"
              "掛單場館：%s"
              % (len(pos), pos.sym.nunique(), ", ".join(sorted(set(pos.post)))))
        print("  深度中位 **$%.0f**（這是單邊頂檔，容量的上界）"
              % pos.dep.median())

    # 寫出給 G1（流量）那一關接 —— 下游不得自己重算毛邊際（第二份實作）
    out = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       "research", "results", "spread_arb_pairs.json")
    live.to_json(out, orient="records", force_ascii=False)
    print("\n寫出 %s（%d 個不凍結的配對，含負的）" % (out, len(live)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
