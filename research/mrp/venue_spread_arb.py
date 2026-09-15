# -*- coding: utf-8 -*-
"""逐場館的價差寬度 —— 「價差套利」（不是中價套利）的 Gate 0（2026-09-13）

===========================================================================
為什麼量這個
===========================================================================
`small-trader-alpha-1`（2023-07-03）把機會分成兩類，而他推薦的是第二類：

  1. **中價套利**（兩腿都吃單）：兩所中價嚴重錯位。他的評語是「太容易套，
     所以在轉帳到達之前就回去了」——死於收斂。
  2. **價差套利**（不流動腿**掛單** ／ 流動腿吃單）：**兩所中價可以完全
     一樣**，賺的是不流動那所的價差寬。他叫讀者把注意力放這裡。

第二類**不需要錯價存在**。所以它的 Gate 0 不是「帶有多寬」（那是 §0.75 /
§1.39 量的，已經 NO-GO），而是：

    在不流動場館掛單收到的半價差
      − 那裡的 maker 費
      − 流動場館的半價差（對沖腿吃單要付）
      − 那裡的 taker 費
      = 每單位名目的毛邊際      <- 這支算這個

固定成本（提款費／gas）不在這裡：它按筆攤提，要等有了可成交量才算得出來。
**所以這支算的是上界**，而上界為負就不必往下做了。

===========================================================================
兩個性質讓這個量測現在就能做
===========================================================================
1. **對今天那個腿差 53 秒的 bug 免疫。** 價差是同一個報價裡買賣兩邊的差，
   不跨腿、不跨抓取段（TODO §1.39）。
2. **不需要帳戶、不需要金鑰、不需要動錢。**

===========================================================================
自曝檢查（凍結報價會同時偽造寬價差與寬帶）
===========================================================================
S1  **凍結比例**：連續兩筆取樣的 bid 與 ask 完全相同的比例。小場館的
    「寬價差」最常見的假來源就是報價不動（backtest-audit #13「API 壞掉的
    標的逆選擇最低」、#20 洗量／假簿口）。**凍結比例高的場館不准解讀。**
S2  **binance 當雜訊底**：最深的簿口、最緊的價差。任何場館的半價差要是
    沒有明顯大於 binance，就沒有「把流動性搬過去」的空間。
S3  全格報告：每個場館都印，不挑。
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
ARB = str(arb_home.HOME)
sys.path.insert(0, ARB)

BAR_SEC = 180
LIQUID = "binance"          # 對沖腿（他的「流動場館」）
MIN_ROWS = 2000             # 每個場館至少這麼多筆才列


def load_quotes():
    """逐 (場館, ticker, 時間) 的頂檔 —— 保留 bid/ask，不只 mid。"""
    from arblib import scan_rank
    d = scan_rank.load()
    parts = []
    for pre in ("a", "b"):
        x = d[["ts", "pair", "leg_%s" % pre,
               "%s_bid" % pre, "%s_ask" % pre]].copy()
        x.columns = ["ts", "pair", "venue", "bid", "ask"]
        parts.append(x)
    p = pd.concat(parts, ignore_index=True)
    del parts, d
    p["sym"] = p["pair"].str.split("@").str[0]
    p = p.drop(columns=["pair"])
    p = p[(p.bid > 0) & (p.ask > 0) & (p.ask >= p.bid)]
    p["t"] = (p.ts // BAR_SEC) * BAR_SEC
    p = p.groupby(["venue", "sym", "t"], as_index=False).last()
    p["mid"] = (p.bid + p.ask) / 2.0
    p["hs_bps"] = (p.ask - p.bid) / 2.0 / p.mid * 1e4      # 半價差
    return p


def frozen_frac(g):
    """S1：連續取樣的 bid 與 ask 都沒動的比例（逐 ticker 算再合池）。"""
    num = den = 0
    for _, s in g.groupby("sym"):
        if len(s) < 50:
            continue
        s = s.sort_values("t")
        same = (s.bid.diff() == 0) & (s.ask.diff() == 0)
        num += int(same.iloc[1:].sum())
        den += len(s) - 1
    return (num / den) if den else np.nan


def main():
    from arblib import fees
    p = load_quotes()
    print("報價面板：%s 列｜%d 個場館｜%d 個 ticker｜%.1f 天"
          % (format(len(p), ","), p.venue.nunique(), p.sym.nunique(),
             (p.t.max() - p.t.min()) / 86400.0))

    rows = []
    for v, g in p.groupby("venue"):
        if len(g) < MIN_ROWS:
            continue
        rows.append(dict(
            venue=v, n=len(g), syms=g.sym.nunique(),
            hs_p25=float(g.hs_bps.quantile(.25)),
            hs_med=float(g.hs_bps.median()),
            hs_p75=float(g.hs_bps.quantile(.75)),
            frozen=frozen_frac(g),
            maker=fees.fee_bps(v, maker=True, rebate=True),
            taker=fees.fee_bps(v, maker=False, rebate=True)))
    r = pd.DataFrame(rows).sort_values("hs_med", ascending=False)

    liq = r[r.venue == LIQUID]
    if liq.empty:
        print("**%s 不在表上 —— 沒有流動腿可當雜訊底，停**" % LIQUID)
        return 1
    L = liq.iloc[0]
    # 他的配置：不流動腿掛單、流動腿吃單
    r["gross"] = (r.hs_med - r.maker) - (L.hs_med + L.taker)

    print("\nS2 流動腿（%s）：半價差中位 **%.3f bps**、taker %.2f、maker %.2f"
          % (LIQUID, L.hs_med, L.taker, L.maker))
    print("S1 凍結比例 = 連續取樣 bid 與 ask 都沒動的比例（高的不准解讀）")

    print("\n" + "=" * 104)
    print("逐場館：在這裡掛單、在 %s 對沖吃單，每單位名目的毛邊際（未扣提款費）"
          % LIQUID)
    print("=" * 104)
    print("  %-12s %8s %6s %9s %9s %9s %8s %7s %7s %10s"
          % ("場館", "n", "ticker", "半價差p25", "中位", "p75",
             "凍結", "maker", "taker", "毛邊際"))
    for _, x in r.iterrows():
        flag = "  **凍結高，不解讀**" if (x.frozen or 0) > 0.5 else ""
        print("  %-12s %8s %6d %9.3f %9.3f %9.3f %7.0f%% %7.2f %7.2f %10.3f%s"
              % (x.venue, format(int(x.n), ","), x.syms, x.hs_p25, x.hs_med,
                 x.hs_p75, 100 * (x.frozen or 0), x.maker, x.taker,
                 x.gross, flag))

    ok = r[(r.gross > 0) & (r.frozen <= 0.5) & (r.venue != LIQUID)]
    print("\n判讀")
    print("  **毛邊際 > 0 且凍結比例 <= 50%% 的場館：%d 個**%s"
          % (len(ok), ("：" + ", ".join(ok.venue)) if len(ok) else ""))
    print("  毛邊際是**上界**：沒扣提款費／gas（按筆攤提，要有可成交量才算得出來）、")
    print("  沒扣逆選擇（掛單被打到的那一側是資訊流）、沒扣未成交的放棄邊際。")
    print("  凍結比例高 = 那個「寬價差」可能是報價不動不是市場"
          "（backtest-audit #13 / #20）。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
