# -*- coding: utf-8 -*-
"""把份額假設換成量到的競爭者數 ＋ 用場館自報成交額看那 57 個看不見的標的

===========================================================================
更正：12.69 美金/天 用的假設跟同一份輸出裡的量測自相矛盾
===========================================================================
`lighter_flow_gate0.py` 同時印了兩件事：

    份額假設 **50%**（`small-trader-alpha-1` 的起始值）
    做市帳戶數中位 **23 個**（我們自己從成交帶量到的）

**50% 是他給「沒人競爭的不流動場館」的起始值**，而我們量到的是一個
有 23 個人在掛的簿口。兩個數字不能並存。這支把份額換成量測：

    我方份額 = 1 / (現有做市帳戶數 + 1)        <- 等分
    另外報    最大帳戶佔比的倒推（它拿 31%，剩下的人分 69%）

===========================================================================
第二件事：那 57 個看不見的標的，今天就有流量的代理值
===========================================================================
掃描器的 HEADER 有 **`b_vol24_usd`** —— 場館自己報的 24 小時成交額，
只存 b 腿。所以凡是 lighter 當過 b 腿的 ticker，**不用等成交帶，
今天就有一個流量數字**。

**它是代理不是量測**，兩個已知的落差要寫下來：
  1. 交易所自報的成交額**包含洗量**（backtest-audit 第 20 項），
     而我們沒有辦法從這個欄位分辨。
  2. 它是**全日全簿口**的量，不是「打到頂檔的量」。成交帶量到的
     前 80 名裡，這兩者的比值可以算出來 —— 所以這支**用那個比值去校準**，
     而不是直接把 vol24 當可賺的流量。

校準是在**有兩種資料的那 9 個標的上**算的，然後外推到 57 個。
外推就是外推，要標明。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
# 2026-09-15: arb location comes from research/arb_home.py (ARB_HOME overrides), not a hardcoded path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arb_home  # noqa: E402
arb_home.add_to_path()

PAIRS = os.path.join(ROOT, "research", "results", "spread_arb_pairs.json")
FLOW = os.path.join(ROOT, "research", "results", "lighter_flow_gate0.json")
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"


def vol24_by_sym():
    """場館自報的 24h 成交額，逐 (ticker) 取 lighter 當 b 腿那些列的中位。"""
    from arblib import scan_rank
    d = scan_rank.load()
    x = d[d.leg_b == "lighter"][["pair", "b_vol24_usd"]].copy()
    x["sym"] = x["pair"].str.split("@").str[0]
    v = x.groupby("sym")["b_vol24_usd"].median()
    return v[v > 0]


def main():
    fl = pd.read_json(FLOW)
    pr = pd.read_json(PAIRS)
    sys.stdout.reconfigure(encoding="utf-8")

    # ── 1. 份額：用量到的競爭者數取代 50% 的假設 ──────────────────
    ok = fl[fl.fills_day >= 10].copy()
    ok["share_eq"] = 1.0 / (ok.makers + 1.0)
    ok["pnl_eq"] = ok.usd_day * ok.share_eq * ok.gross / 1e4
    ok["pnl_50"] = ok.usd_day * 0.50 * ok.gross / 1e4
    print("一、份額假設換成量測（看得見的 %d 個標的）" % len(ok))
    print("  %-10s %8s %9s %8s %10s %10s"
          % ("ticker", "毛邊際", "流量$/天", "做市帳戶", "50%假設", "等分實測"))
    for _, x in ok.sort_values("pnl_eq", ascending=False).iterrows():
        print("  %-10s %8.2f %9.0f %8.0f %10.3f %10.3f"
              % (x.sym, x.gross, x.usd_day, x.makers, x.pnl_50, x.pnl_eq))
    print("  ---")
    print("  **合計：50%% 假設 $%.2f/天  ->  按現有做市者等分 $%.2f/天**"
          % (ok.pnl_50.sum(), ok.pnl_eq.sum()))
    print("  （等分也是假設，只是比 50%% 誠實 —— 實際份額取決於排隊位置，"
          "而我們沒有簿口佇列資料）")

    # ── 2. 校準：自報 vol24 vs 成交帶量到的真實流量 ─────────────────
    v = vol24_by_sym()
    cal = ok.merge(v.rename("vol24"), left_on="sym", right_index=True)
    print("\n二、用那 9 個有兩種資料的標的，校準「自報成交額 -> 真實流量」")
    print("  %-10s %12s %12s %8s" % ("ticker", "自報 vol24", "帶子流量/天", "比值"))
    cal["ratio"] = cal.usd_day / cal.vol24
    for _, x in cal.sort_values("ratio").iterrows():
        print("  %-10s %12.0f %12.0f %8.3f" % (x.sym, x.vol24, x.usd_day,
                                               x.ratio))
    k = float(cal.ratio.median())
    print("  **校準係數中位 %.3f**（自報量的 %.0f%% 才是我們量到的成交流量）"
          % (k, 100 * k))
    print("  p25~p75 %.3f~%.3f —— **離散很大，所以下面是量級不是估計**"
          % (cal.ratio.quantile(.25), cal.ratio.quantile(.75)))

    # ── 3. 外推到看不見的那些 ────────────────────────────────────
    miss = pr[(pr.post == "lighter") & (pr.gross > 0)
              & (~pr.sym.isin(fl.sym))].copy()
    miss = (miss.sort_values("gross", ascending=False)
                .groupby("sym", as_index=False).first())
    miss = miss.merge(v.rename("vol24"), left_on="sym", right_index=True,
                      how="left")
    have = miss[miss.vol24.notna()].copy()
    print("\n三、外推到看不見的 %d 個（其中 %d 個有自報成交額）"
          % (len(miss), len(have)))
    have["flow_est"] = have.vol24 * k
    # 競爭者數未知 -> 兩個情境都報
    have["pnl_solo"] = have.flow_est * 0.50 * have.gross / 1e4
    have["pnl_23"] = have.flow_est * (1.0 / 24) * have.gross / 1e4
    have = have.sort_values("pnl_solo", ascending=False)
    print("  %-10s %-11s %8s %12s %12s %10s %10s"
          % ("ticker", "對沖", "毛邊際", "自報 vol24", "推估流量/天",
             "若沒人搶", "若 23 人搶"))
    for _, x in have.head(20).iterrows():
        print("  %-10s %-11s %8.2f %12.0f %12.0f %10.3f %10.3f"
              % (x.sym, x.hedge, x.gross, x.vol24, x.flow_est,
                 x.pnl_solo, x.pnl_23))
    print("  ---")
    print("  **合計：若沒人搶 $%.2f/天  ／  若同樣 23 人搶 $%.2f/天**"
          % (have.pnl_solo.sum(), have.pnl_23.sum()))
    print("  %d 個沒有自報成交額 -> 連代理值都沒有，仍然是未知"
          % int(miss.vol24.isna().sum()))

    print("\n四、寬價差的兩種成因，而它們的結論相反")
    print("  一個 20-60 bps 的半價差可能是：")
    print("    (a) **沒人在搶** -> 我們去掛就是第一個，那是機會")
    print("    (b) **沒人在交易** -> 掛了也沒人打，那是死市場")
    print("  **分辨它們只需要一個數字：流量。** 而那正是成交帶沒錄的。")
    print("  上面第三節的「推估流量」是自報量校準來的，"
          "而自報量含洗量（backtest-audit #20）—— 所以它分辨不了 (a) 和 (b)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
