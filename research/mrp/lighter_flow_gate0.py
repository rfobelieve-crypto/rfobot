# -*- coding: utf-8 -*-
"""G1：Lighter 長尾的吃單流量 —— §1.40 的判決數字（2026-09-13）

===========================================================================
為什麼這是唯一的判決數字
===========================================================================
§1.40 量到 290 個配對毛邊際為正、深度中位 **$49**。
$49 x 20 bps = **一趟約 $0.01**。所以這門生意完全取決於**次數**，
而次數就是「每天有多少錢打到那個簿口上」。

判準在跑之前凍結（TODO §1.40，commit 2b93640）：
  **G1  逐 ticker 每天 >= 10 筆可能打到我們的成交。**
  低於 10 筆就不是生意（backtest-audit 第 2 項：「3 isn't many」），
  而且在任何資金規模下都不是。

===========================================================================
順手把一個「未量測」變成量測
===========================================================================
§1.40 寫下的第一個未量測是：**我們掛進去會改變簿口，而哪一種會發生取決於
有沒有人跟我們競價**。成交帶有 `ask_acct` / `bid_acct`，所以**做市方的
帳戶數與集中度量得出來** —— 那就是競價強度的直接讀數：

  * 做市方帳戶數少 -> 我們加入一個薄隊伍
  * 一個帳戶吃掉大部分做市量 -> 那是一個專職做市者，我們在跟它搶

而 `maker_fee` 欄位直接給「現在的做市方實際付多少費」——
對照我們自己的 0.40 bps，就知道我們在不在平價上（TODO §1.31/§1.33）。

===========================================================================
口徑（寫死，不要事後放寬）
===========================================================================
* **吃單流量 = 全部成交**（每筆成交都有一個做市方，我們本來可以是它），
  但**排除清算**（`is_liq`）—— 清算不是可競爭的流，它是強制平倉。
* **我們的份額假設 50%**，照 `small-trader-alpha-1` 的原話
  「先假設你能拿到買/賣任一側流量的一半，再從那裡修」。
  這是**假設不是量測**，所以單獨印出來、不藏在淨值裡。
* 毛邊際**不在這裡重算**，直接讀 `spread_arb_pairs.json`
  （§1.40 的計分器寫的）—— 第二份實作會安靜地不同意。
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAIRS = os.path.join(ROOT, "research", "results", "spread_arb_pairs.json")
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
OUT = os.path.join(ROOT, "research", "results", "lighter_flow_gate0.json")

MIN_FILLS_DAY = 10        # G1 門檻，TODO §1.40 凍結
SHARE = 0.50              # 他的起始假設，不是量測


def load_tape():
    fs = sorted(glob.glob(TAPE))
    cols = ["ts", "coin", "usd", "is_liq", "is_maker_ask",
            "maker_fee", "taker_fee", "ask_acct", "bid_acct"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs],
                  ignore_index=True)
    d["day"] = (d.ts // 86400000).astype("int64")
    return d


def main():
    if not os.path.exists(PAIRS):
        print("找不到 %s —— 先跑 spread_arb_paired.py" % PAIRS)
        return 2
    pr = pd.read_json(PAIRS)
    # 每個 ticker 取它最好的那個配對（掛單場館 + 對沖場館）
    best = (pr.sort_values("gross", ascending=False)
              .groupby(["sym", "post"], as_index=False).first())
    lt = best[(best.post == "lighter") & (best.gross > 0)]
    print("§1.40 的配對：%d 個不凍結｜其中掛單在 lighter 且毛邊際 > 0 的 "
          "**%d 個 ticker**" % (len(pr), len(lt)))

    d = load_tape()
    days = d.day.nunique()
    span = (d.ts.max() - d.ts.min()) / 86400000.0
    print("成交帶：%s 筆｜%d 個幣｜%d 個日檔｜%.1f 天｜清算佔 %.1f%%"
          % (format(len(d), ","), d.coin.nunique(), days, span,
             100 * d.is_liq.mean()))

    live = d[~d.is_liq]
    rec = dict()
    for c, x in live.groupby("coin"):
        mk = np.where(x.is_maker_ask, x.ask_acct, x.bid_acct)
        s = pd.Series(x.usd.values).groupby(mk).sum()
        rec[c] = dict(
            fills=len(x), usd=float(x.usd.sum()),
            med_sz=float(x.usd.median()),
            makers=int(s.size),
            top_mk=float(s.max() / s.sum()) if s.size else np.nan,
            # **`maker_fee` 的單位是 1e-6，raw/100 = bps。**
            # 逐值對上 CLAUDE.md 的費率表才確定的，不是猜的：
            # 28->0.28（Premium 質押）、40->0.40（Premium 零質押）、
            # 50->0.50（Plus）、196->1.96、280->2.80（吃單側）。
            # 第一版寫 /10 跑出「做市方付 2.80」—— 那是吃單費率，不可能，
            # 而「數字不合理先查儀器」就是這樣抓到的。
            # 注意 **空白 = Standard = 0 費率**（CLAUDE.md 2026-09-12：
            # 零費率是用「不寫那個欄位」表示的），所以 fillna(0) 是對的。
            mk_fee=float(np.average(x.maker_fee.fillna(0),
                                    weights=x.usd)) / 100.0,
            # 吃單側零費率（Standard）佔多少量 —— 那是散戶流的比例
            tk_free=float(x.usd[x.taker_fee.isna()].sum() / x.usd.sum()))
    flow = pd.DataFrame(rec).T.reset_index().rename(columns={"index": "coin"})
    flow["fills_day"] = flow.fills / span
    flow["usd_day"] = flow.usd / span

    m = lt.merge(flow, left_on="sym", right_on="coin", how="left")
    # **「沒錄到」不是「沒有流量」。** lighter_tape 的宇宙是日成交額前 80 名
    # （`TOP_N_DEFAULT = 80`），而這門生意要的標的正是長尾 —— 所以不在帶子裡
    # 的 ticker 必須標成**未知**，不可以 fillna(0) 當成零流量。
    # 那是這個專案反覆踩的同一個形狀：把「我不知道」記成一個看起來合理的值。
    m["in_tape"] = m.coin.notna()
    miss = m[~m.in_tape]
    print("\n**儀器範圍**：%d 個目標 ticker 裡，**%d 個在成交帶宇宙內、"
          "%d 個不在（= 未知，不是 0）**" % (len(m), int(m.in_tape.sum()), len(miss)))
    if len(miss):
        print("  不在帶子裡的：%s%s"
              % (", ".join(miss.sym.head(12)),
                 " …" if len(miss) > 12 else ""))
        print("  原因：`lighter_tape.py` 錄日成交額**前 80 名**，"
              "而寬價差住在長尾。**這是 WS 串流，不可回填。**")
    m = m[m.in_tape].copy()
    # 我們的份額 x 流量 x 毛邊際
    m["mine_usd_day"] = m.usd_day * SHARE
    m["pnl_day"] = m.mine_usd_day * m.gross / 1e4
    m = m.sort_values("pnl_day", ascending=False)

    print("\n" + "=" * 112)
    print("G1：掛單在 lighter 的 %d 個 ticker —— 每天的吃單流量與它換成的錢"
          % len(m))
    print("  門檻：**每天 >= %d 筆**（TODO §1.40 凍結）｜份額假設 %.0f%%（**假設**）"
          % (MIN_FILLS_DAY, 100 * SHARE))
    print("=" * 112)
    print("  %-10s %-11s %8s %9s %10s %8s %7s %7s %9s"
          % ("ticker", "對沖", "毛邊際", "筆/天", "流量$/天",
             "做市帳戶", "最大佔", "maker費", "我方$/天"))
    for _, x in m.head(25).iterrows():
        print("  %-10s %-11s %8.2f %9.1f %10.0f %8s %6.0f%% %7s %9.3f"
              % (x.sym, x.hedge, x.gross, x.fills_day, x.usd_day,
                 ("%.0f" % x.makers) if np.isfinite(x.makers) else "—",
                 100 * (x.top_mk if np.isfinite(x.top_mk) else 0),
                 ("%.2f" % x.mk_fee) if np.isfinite(x.mk_fee) else "—",
                 x.pnl_day))

    ok = m[m.fills_day >= MIN_FILLS_DAY]
    print("\n判決（G1）")
    print("  **過門檻（>= %d 筆/天）：%d / %d 個 ticker**"
          % (MIN_FILLS_DAY, len(ok), len(m)))
    print("  它們合計 **$%.2f / 天**（份額 %.0f%% 的假設下，未扣逆選擇、"
          "未成交、固定成本、資金費）" % (ok.pnl_day.sum(), 100 * SHARE))
    nf = m[m.fills_day < MIN_FILLS_DAY]
    print("  沒過的 %d 個裡，**完全沒有成交的 %d 個**"
          % (len(nf), int((nf.fills_day == 0).sum())))

    # G3 集中度（backtest-audit 第 2 項，我們從來沒做過這一項）
    if ok.pnl_day.sum() > 0:
        s = ok.pnl_day.sort_values(ascending=False).values
        tot = s.sum()
        print("\nG3 集中度：最大 1 個 ticker 佔 **%.0f%%**｜前 5 個 **%.0f%%**"
              "｜前 10 個 **%.0f%%**"
              % (100 * s[0] / tot, 100 * s[:5].sum() / tot,
                 100 * s[:10].sum() / tot))
        print("  （集中在 1-2 個就不是 %d 個機會，是 1-2 個）" % len(ok))

    print("\n做市競價強度（把 §1.40 的「未量測 1」變成量測）")
    if len(ok):
        print("  過關 ticker 的做市帳戶數中位 **%.0f**｜最大帳戶佔做市量中位 "
              "**%.0f%%**" % (ok.makers.median(), 100 * ok.top_mk.median()))
        mf = float(ok.mk_fee.median())
        print("  現在的做市方實際付的 maker 費（量加權中位）**%.3f bps**，"
              "我們是 **0.40** -> **我們貴 %.3f bps**" % (mf, 0.40 - mf))
        print("  吃單側零費率（Standard＝散戶）佔量中位 **%.0f%%**"
              % (100 * ok.tk_free.median()))
        bad = ok[ok.gross <= (0.40 - mf)]
        print("  **毛邊際小於那個費率劣勢的 ticker：%d / %d**%s"
              % (len(bad), len(ok),
                 ("：" + ", ".join(bad.sym)) if len(bad) else ""))

    m.to_json(OUT, orient="records", force_ascii=False)
    print("\n寫出 %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
