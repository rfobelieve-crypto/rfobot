# -*- coding: utf-8 -*-
"""G1 重算：分母改成**實錄時間**，並量「放寬宇宙買到多少」（2026-09-13）

===========================================================================
更正一個讓前面每個 $/天 都錯 15 倍的錯
===========================================================================
`lighter_flow_gate0.py` 的分母寫的是

    span = (d.ts.max() - d.ts.min()) / 86400000      # 日曆跨度

而那是**日曆跨度不是實錄時間**。實際狀況：

| 日期 | 一般成交 | 清算 |
|---|---|---|
| 08-20 ~ 09-10 | **0** | 1–529/天 |
| 09-11 | 116 | 568 |
| 09-12 | 390,461 | 165 |
| 09-13（8 小時）| 340,959 | 60 |

**一般成交是 09-11 才開始的**（CLAUDE.md §HFT 2026-09-12 就記了「成交帶今天
才上線」），09-10 之前的檔案裡只有清算 —— 那是 `liq_recorder` 在同一個目錄。

所以 `ts.max()-ts.min()` = 23.6 天量到的是**清算的跨度**，而一般成交只有
**1.5 天**。分母差 15 倍，每個「筆/天」「$/天」都被低估 15 倍。

**正確的分母 = 有一般成交的那些 UTC 小時數 / 24。** 不是日曆跨度、
也不是檔案數（舊檔只有清算也會產生檔案）。

這是這個專案反覆的同一個形狀：**一個看起來合理的分母，量的是別的東西。**
同族：mistake.md 2026-09-03「把分鐘當交易」（144 次/天那個恆等式）。

===========================================================================
順便回答「放寬 TOP_N 買到什麼」
===========================================================================
`lighter_tape.py` 的凍結設計記著一個量測（不是拍的）：
    227 個 active 市場裡，前 40 名 = 98.0% 成交額 / 94.0% 筆數，
    **前 80 名 = 99.6% / 98.7%**，前 150 名之後幾乎全是 0 筆。

所以 rank 81+ 合計 = 0.4% 成交額 / 1.3% 筆數。這支把它攤到市場數上，
跟 G1 的門檻（>= 10 筆/天）比 —— **那個比較決定放寬值不值得**。
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
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
MIN_FILLS_DAY = 10        # TODO §1.40 凍結
N_ACTIVE = 227            # lighter_tape 檔頭記的 active 市場數
SH_FILLS, SH_USD = 0.987, 0.996   # 前 80 名的覆蓋率，同上


def main():
    fs = sorted(glob.glob(TAPE))
    d = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "usd", "is_liq", "is_maker_ask", "maker_fee",
        "ask_acct", "bid_acct"]) for f in fs], ignore_index=True)
    reg = d[~d.is_liq].copy()
    # **分母 = 有一般成交的 UTC 小時數**
    hrs = (reg.ts // 3600000).nunique()
    span = hrs / 24.0
    print("一般成交 %s 筆｜%d 個幣｜**實錄 %d 小時 = %.2f 天**"
          % (format(len(reg), ","), reg.coin.nunique(), hrs, span))
    print("  （日曆跨度 %.1f 天 —— 那是清算的跨度，**不可當分母**）"
          % ((d.ts.max() - d.ts.min()) / 86400000))

    # ── G1 重算 ────────────────────────────────────────────────
    pr = pd.read_json(PAIRS)
    lt = (pr[(pr.post == "lighter") & (pr.gross > 0)]
          .sort_values("gross", ascending=False)
          .groupby("sym", as_index=False).first())
    rec = {}
    for c, x in reg.groupby("coin"):
        mk = np.where(x.is_maker_ask, x.ask_acct, x.bid_acct)
        s = pd.Series(x.usd.values).groupby(mk).sum()
        rec[c] = dict(fills_day=len(x) / span, usd_day=float(x.usd.sum()) / span,
                      makers=int(s.size),
                      mk_fee=float(np.average(x.maker_fee.fillna(0),
                                              weights=x.usd)) / 100.0)
    fl = pd.DataFrame(rec).T.reset_index().rename(columns={"index": "coin"})
    m = lt.merge(fl, left_on="sym", right_on="coin")
    m["share_eq"] = 1.0 / (m.makers + 1.0)
    m["pnl_eq"] = m.usd_day * m.share_eq * m.gross / 1e4
    m["pnl_50"] = m.usd_day * 0.50 * m.gross / 1e4
    m = m.sort_values("pnl_eq", ascending=False)

    print("\n看得見的 %d 個標的（分母已修正）" % len(m))
    print("  %-10s %8s %10s %12s %8s %10s %10s"
          % ("ticker", "毛邊際", "筆/天", "流量$/天", "做市帳戶",
             "等分$/天", "50%$/天"))
    for _, x in m.iterrows():
        print("  %-10s %8.2f %10.0f %12s %8.0f %10.2f %10.2f"
              % (x.sym, x.gross, x.fills_day, format(int(x.usd_day), ","),
                 x.makers, x.pnl_eq, x.pnl_50))
    ok = m[m.fills_day >= MIN_FILLS_DAY]
    print("  ---")
    print("  G1 過門檻 **%d / %d**｜**等分合計 $%.2f/天**（50%% 假設下 $%.2f）"
          % (len(ok), len(m), ok.pnl_eq.sum(), ok.pnl_50.sum()))
    s = ok.pnl_eq.sort_values(ascending=False).values
    print("  G3 集中度：最大 1 個 %.0f%%｜前 3 個 %.0f%%"
          % (100 * s[0] / s.sum(), 100 * s[:3].sum() / s.sum()))

    # ── 放寬買到什麼 ───────────────────────────────────────────
    f80 = len(reg) / span
    u80 = reg.usd.sum() / span
    n80 = reg.coin.nunique()
    tail_f = f80 / SH_FILLS - f80
    tail_u = u80 / SH_USD - u80
    n_tail = N_ACTIVE - n80
    print("\n放寬 TOP_N 買到什麼（用 lighter_tape 檔頭凍結的覆蓋率）")
    print("  前 %d 名實測：%s 筆/天、$%s/天"
          % (n80, format(int(f80), ","), format(int(u80), ",")))
    print("  rank 81+ 合計（0.4%% 成交額 / 1.3%% 筆數）：**%s 筆/天、$%s/天**"
          % (format(int(tail_f), ","), format(int(tail_u), ",")))
    print("  攤到 %d 個市場：**每個 %.0f 筆/天、$%.0f/天**"
          % (n_tail, tail_f / n_tail, tail_u / n_tail))
    print("  G1 門檻 10 筆/天 -> **%s**"
          % ("過（平均 %.0f 倍餘裕）" % (tail_f / n_tail / 10)
             if tail_f / n_tail >= 10 else "不過"))

    miss = pr[(pr.post == "lighter") & (pr.gross > 0) & (~pr.sym.isin(fl.coin))]
    miss = (miss.sort_values("gross", ascending=False)
                .groupby("sym", as_index=False).first())
    gm = float(miss.gross.median())
    print("\n  §1.40 看不見的 %d 個：毛邊際中位 **%.2f bps**"
          "（看得見的是 %.2f）" % (len(miss), gm, float(m.gross.median())))
    per = tail_u / n_tail
    for lab, sh in (("若也是 23 人搶（1/24）", 1 / 24.0),
                    ("若只有 3 人搶（1/4）", 0.25),
                    ("若沒人搶（50%）", 0.50)):
        print("    %-22s 每個 $%.3f/天 -> **%d 個合計 $%.2f/天**"
              % (lab, per * sh * gm / 1e4, len(miss),
                 len(miss) * per * sh * gm / 1e4))
    print("\n  **放寬的好處就是上面那三行選哪一行** —— 而選哪一行只取決於")
    print("  長尾的做市帳戶數，那個數字跟流量在同一份資料裡（ask_acct/bid_acct）。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
