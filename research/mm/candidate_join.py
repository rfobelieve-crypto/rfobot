# -*- coding: utf-8 -*-
"""把 Stage 0 的四個量測 join 起來 —— 交集才是候選（2026-09-13，TODO §1.40）

===========================================================================
為什麼要 join：兩個量測指向相反的方向
===========================================================================
`quote_life.py`      慢而寬的市場 = 股票/商品永續（MSFT 1061ms/1.21bps、
                     ANSEM 1028/52.1、DELL 668/39.5）-> **50ms 很舒服**
`inventory_bound.py` 而那一批的**流量是方向性的**（換邊率 QQQ 21%、MU 15%、
                     SPY 18%、TSLA 15%、US100 6%）-> **會輾過做市方**

原因一致：股票永續的流量由標的自己的走勢驅動，所以大家同一邊。
**慢，是因為它跟著別的東西走；而跟著別的東西走，就意味著單向。**

所以候選不是任一個量測的 top-N，是**四個條件的交集**：

    A  報價存活 >= 50ms 的比例夠高      （我們來得及撤）
    B  半價差夠寬                        （有東西可賺）
    C  換邊率在 25%~75%                  （非洗量、非方向性）
    D  單邊性低                          （庫存平得掉）
    ＋ markout > maker 費                （真的有淨值）

全格報告：四個條件逐一列出誰過誰不過，**不挑**。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "research", "results")
MAKER_FEE = 0.40            # Lighter Premium 零質押


def main():
    ql = pd.read_json(os.path.join(RES, "quote_life_by_coin.json"))
    ib = pd.read_json(os.path.join(RES, "inventory_bound.json"))
    mo = pd.read_json(os.path.join(RES, "sweep_markout.json"))
    j = (ql[["coin", "usd", "exp", "mo"]]
         .rename(columns={"usd": "usd_ql", "mo": "mo_ql"})
         .merge(ib[["coin", "clipr", "alt", "ratio_p50_300", "net_p95_300",
                    "usd"]], on="coin", how="inner")
         .merge(mo[["coin", "hs", "mo_1.0"]], on="coin", how="left"))
    print("三份量測都有的幣：**%d 個**" % len(j))

    j["A"] = j.exp <= 15.0                 # 曝露 <=15% = 多數移動我們追得上
    j["B"] = j.hs >= 2.0                   # 半價差 >= 2 bps
    j["C"] = j.alt.between(0.25, 0.75)     # 非洗量、非方向性
    j["D"] = j.ratio_p50_300 <= 0.40       # 單邊性低（庫存平得掉）
    j["E"] = j["mo_1.0"] > MAKER_FEE       # markout 蓋得過 maker 費
    for c in "ABCDE":
        j[c] = j[c].fillna(False)
    j["pass_n"] = j[list("ABCDE")].sum(axis=1)

    print("\n逐條件通過數（%d 個幣）" % len(j))
    names = dict(A="A 來得及撤（曝露<=15%%）", B="B 半價差>=2bps",
                 C="C 換邊率 25-75%%", D="D 單邊性<=0.40",
                 E="E markout>maker 費")
    for c in "ABCDE":
        print("  %-26s **%2d / %d**" % (names[c] % () if "%%" not in names[c]
                                        else names[c].replace("%%", "%"),
                                        int(j[c].sum()), len(j)))

    print("\n全格（按過關數排序）")
    print("  %-10s %6s %7s %6s %7s %8s  %s"
          % ("coin", "曝露%", "半價差", "換邊%", "單邊性", "mo@1s", "ABCDE"))
    for _, x in j.sort_values(["pass_n", "usd"], ascending=False).iterrows():
        flags = "".join(c if x[c] else "·" for c in "ABCDE")
        print("  %-10s %5.1f %7s %5.0f %7s %8s  %s"
              % (x.coin, x.exp,
                 "—" if pd.isna(x.hs) else "%.2f" % x.hs,
                 100 * x.alt,
                 "—" if pd.isna(x.ratio_p50_300) else "%.3f" % x.ratio_p50_300,
                 "—" if pd.isna(x["mo_1.0"]) else "%+.3f" % x["mo_1.0"],
                 flags))

    full = j[j.pass_n == 5]
    print("\n判讀")
    print("  **五個條件全過：%d 個**%s"
          % (len(full), ("：" + ", ".join(full.coin)) if len(full) else ""))
    four = j[j.pass_n == 4]
    print("  過 4 個的：%d 個%s"
          % (len(four), ("：" + ", ".join("%s(缺%s)" % (
              x.coin, "".join(c for c in "ABCDE" if not x[c]))
              for _, x in four.iterrows())) if len(four) else ""))
    print("\n  **這 %d 個幣全部是 tob 錄的前 80 名（流動端）。**" % len(j))
    print("  而 §1.40 的候選（半價差 20-60 bps）住在長尾，"
          "而長尾今天才開始錄 —— 所以這張表回答的是")
    print("  「流動端有沒有」而不是「這條線有沒有」。")
    j.to_json(os.path.join(RES, "mm_candidates.json"), orient="records",
              force_ascii=False)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
