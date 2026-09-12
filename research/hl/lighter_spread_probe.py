# -*- coding: utf-8 -*-
"""量 Lighter 的頂檔半價差 —— §1.27 那個死結的分母。

為什麼這是決定性的數字：§1.27/1.28 量到吃單淨 −2.42 bps（毛利 +0.58、
**費 3.00**），而那個 3.00 是 **Binance** 的。Lighter 在 arblib/fees.py 裡是
`taker_bps 0.0 / maker_bps 0.0`（標 verified）。費用那一項歸零之後，
**剩下的唯一障礙就是價差與逆選擇**，而價差現在就量得出來。

對照基準（都來自我們自己的量測，不是論證）：
    Binance 11 個厚標的   半價差中位 **0.50 bps**   <- 毛利 +0.58 是在這上面量的
    HL 尾 50 名          頂檔價差中位 38.5 bps（半價差 ~19）
    §1.29 倉位型做市門檻  ≈0.5 bps

每個配對的 hedge 腿都是 Lighter（arblib/premium_verdict.py 的 PAIRS）；
GOLD_LL / NVDA_LL 兩腿都是 Lighter。
"""
import csv
import glob
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/rfo/Desktop/flowbot/arb/engine/logs"

BOTH_LIGHTER = {"GOLD_LL", "NVDA_LL"}


def half_spread(rows, bid_col, ask_col):
    vals = []
    for r in rows:
        try:
            b, a = float(r[bid_col]), float(r[ask_col])
        except (TypeError, ValueError):
            continue
        if b <= 0 or a <= 0 or a <= b:
            continue
        mid = (a + b) / 2.0
        vals.append((a - b) / 2.0 / mid * 1e4)
    return vals


print("%-9s %5s │ %-26s │ %-26s" % ("配對", "列數", "Lighter 腿（hedge）半價差 bps",
                                    "另一腿（entropy）半價差 bps"))
print("%-9s %5s │ %-26s │ %-26s" % ("", "", "中位 / p25 / p75", "中位 / p25 / p75"))
print("-" * 74)

lighter_all = []
for d in sorted(os.listdir(ROOT)):
    f = os.path.join(ROOT, d, "minutes.csv")
    if not os.path.exists(f):
        continue
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    h = half_spread(rows, "hedge_bid", "hedge_ask")
    e = half_spread(rows, "entropy_bid", "entropy_ask")

    def fmt(v):
        if not v:
            return "%-26s" % "（無）"
        v = sorted(v)
        return "%-26s" % ("%7.2f / %6.2f / %6.2f"
                          % (st.median(v), v[len(v) // 4], v[3 * len(v) // 4]))

    tag = d + (" *" if d in BOTH_LIGHTER else "")
    print("%-9s %5d │ %s │ %s" % (tag, len(rows), fmt(h), fmt(e)))
    lighter_all += h
    if d in BOTH_LIGHTER:
        lighter_all += e

print("-" * 74)
if lighter_all:
    v = sorted(lighter_all)
    print("Lighter 合池 n=%d：中位 **%.2f bps**、p25 %.2f、p75 %.2f、p90 %.2f"
          % (len(v), st.median(v), v[len(v) // 4], v[3 * len(v) // 4],
             v[int(len(v) * 0.9)]))
    print()
    print("對照：Binance 厚標的 0.50 bps（毛利 +0.58 就是在這上面量的）")
    print("      HL 尾 50 名 ~19 bps")
    print("      §1.29 倉位型做市門檻 ≈0.5 bps")
print()
print("* = 兩腿都是 Lighter。**列數少是因為 55 欄錄製器今天才重啟**，")
print("  半價差是穩定量所以有指示性，但這不是判決 —— 判決要夠長的樣本。")
