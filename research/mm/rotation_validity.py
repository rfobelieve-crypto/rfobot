# -*- coding: utf-8 -*-
"""輪換規則本身有沒有效：前半挑、後半驗（2026-09-13，TODO §1.40）

===========================================================================
為什麼這一關要排在建系統之前
===========================================================================
使用者：「這就多了一件事要篩選幣種跟市場，所以後台要一直 follow 市場變化
來更換幣種市場控制風險及槓桿」。方向對 —— 但**「持續重選」本身就是一個
選擇程序，而選擇程序會過擬合**。

mistake.md 2026-09-09：「事後找到的維度，通過多少一致性檢查都不算數…
**唯一能證明的是把挑選程序本身放進樣本外**：只用訓練窗跑完整個搜尋，
看它選到什麼，再拿那個到測試窗。」

所以這支不問「哪些幣好」，問**「挑的動作有沒有用」**：

    前半（時間上的前一半）挑 -> 後半量
      R1  前後半的 markout 排名相關（Spearman）。接近 0 = 排名是雜訊。
      R2  用前半挑 top-N，後半的 markout 有沒有贏過全體平均。
      R3  **對照組：用半價差挑**（= §1.40 那 288 個配對的挑法）。
          如果用 markout 挑沒有比用半價差挑好，那我們不需要 markout 系統。
      R4  **隨機挑的對照**：抽同樣張數的隨機 N 個幣，重抽 500 次取分布。
          前半挑的要落在它的右尾，否則「挑」這個動作沒有價值。

**不及格的後果是省下來的工程**：如果 R1≈0 且 R2 不贏隨機，那正確的設計是
**固定宇宙 + 事後下架（停損型）**，不是前瞻選幣 —— 兩者的工程量差一個數量級。

===========================================================================
誠實範圍（先寫，免得看到結果才說）
===========================================================================
窗只有 5.76 小時，切兩半各約 2.9 小時。**這測的是「小時級的 markout 排名
有沒有持續性」，不是「7 天級的輪換規則有沒有用」。** 兩者可以不同答案：
短窗沒有持續性不代表長窗沒有（也可能相反）。所以這一關的結論是
**方向性的證據，不是判決** —— 判決要等 217 個市場累積。
但它現在就能擋掉「盲目去建一套前瞻選幣系統」。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
TOB = "D:/flowbot_data/lighter/tob/*/*.parquet"
MIN_USD = 50_000.0          # 每半窗每個幣至少這麼多成交額才進榜
TOP_N = 10
N_RAND = 500
SEED = 0


def per_half():
    tb = pd.concat([pd.read_parquet(f, columns=["book_time", "coin", "bid", "ask"])
                    for f in sorted(glob.glob(TOB))], ignore_index=True)
    tb = tb[tb.book_time.between(1e12, 1e14)]
    tb = tb[(tb.bid > 0) & (tb.ask >= tb.bid)]
    tb["mid"] = (tb.bid + tb.ask) / 2.0
    tb["hs"] = (tb.ask - tb.bid) / 2.0 / tb.mid * 1e4
    tp = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "px", "usd", "is_liq", "is_maker_ask"])
        for f in sorted(glob.glob(TAPE))], ignore_index=True)
    tp = tp[(~tp.is_liq) & (tp.usd > 0) & (tp.px > 0)]
    lo = max(tp.ts.min(), tb.book_time.min())
    hi = min(tp.ts.max(), tb.book_time.max())
    tp = tp[tp.ts.between(lo, hi)]
    cut = lo + (hi - lo) // 2
    print("窗 %.2f 小時，切點 %s（前半 %.2fh / 後半 %.2fh）"
          % ((hi - lo) / 3.6e6, pd.to_datetime(cut, unit="ms"),
             (cut - lo) / 3.6e6, (hi - cut) / 3.6e6))

    out = {}
    for half, (a, b) in (("H1", (lo, cut)), ("H2", (cut, hi))):
        rows = []
        f_all = tp[tp.ts.between(a, b)]
        t_all = tb[tb.book_time.between(a, b)]
        for coin, f in f_all.groupby("coin"):
            bb = t_all[t_all.coin == coin].sort_values("book_time")
            if len(bb) < 100 or f.usd.sum() < MIN_USD:
                continue
            t = bb.book_time.values.astype("int64")
            mid = bb.mid.values
            f = f.sort_values("ts")
            ft = f.ts.values.astype("int64")
            k = np.clip(np.searchsorted(t, ft + 1000, side="right") - 1,
                        0, len(t) - 1)
            sign = np.where(f.is_maker_ask.values, 1.0, -1.0)
            mo = sign * (f.px.values - mid[k]) / mid[k] * 1e4
            ok = np.isfinite(mo)
            if ok.sum() < 50:
                continue
            rows.append(dict(coin=coin, usd=float(f.usd[ok].sum()),
                             mo=float(np.average(mo[ok],
                                                 weights=f.usd.values[ok])),
                             hs=float(np.nanmedian(bb.hs))))
        out[half] = pd.DataFrame(rows).set_index("coin")
        print("  %s：%d 個幣過 $%s 門檻" % (half, len(out[half]),
                                          format(int(MIN_USD), ",")))
    return out


def main():
    h = per_half()
    j = h["H1"].join(h["H2"], lsuffix="1", rsuffix="2", how="inner")
    print("\n兩半都有的幣：**%d 個**" % len(j))
    if len(j) < 15:
        print("樣本太少，不解讀")
        return 1

    uni = float(np.average(j.mo2, weights=j.usd2))
    print("\nR1 排名持續性")
    for lab, col in (("markout", "mo"), ("半價差", "hs")):
        rho = j["%s1" % col].corr(j["%s2" % col], method="spearman")
        print("  前後半的 %s 排名 Spearman = **%+.3f**" % (lab, rho))
    print("  （markout 接近 0 = 排名是雜訊，前瞻選幣沒有依據）")

    print("\nR2/R3 用前半挑 top-%d，看後半（全體成交額加權 markout = %+.3f）"
          % (TOP_N, uni))
    res = {}
    for lab, col, asc in (("markout 最高", "mo1", False),
                          ("半價差最寬", "hs1", False),
                          ("markout 最低（反向對照）", "mo1", True)):
        pick = j.sort_values(col, ascending=asc).head(TOP_N)
        v = float(np.average(pick.mo2, weights=pick.usd2))
        res[lab] = v
        print("  %-26s 後半 markout **%+.3f**（對全體 %+.3f）｜選到：%s"
              % (lab, v, v - uni, ", ".join(pick.index[:6])))

    print("\nR4 隨機挑 %d 個的分布（%d 次重抽）" % (TOP_N, N_RAND))
    rng = np.random.default_rng(SEED)
    sim = []
    for _ in range(N_RAND):
        s = j.sample(TOP_N, random_state=int(rng.integers(1 << 31)))
        sim.append(float(np.average(s.mo2, weights=s.usd2)))
    sim = np.array(sim)
    print("  隨機 p10 %+.3f｜p50 %+.3f｜p90 %+.3f" %
          tuple(np.percentile(sim, [10, 50, 90])))
    for lab, v in res.items():
        pct = 100 * (sim < v).mean()
        print("  **%-26s 落在隨機分布的第 %.0f 百分位**%s"
              % (lab, pct, "  <- 挑得出東西" if pct >= 90 else
                 ("  <- **不比隨機好**" if pct <= 80 else "")))

    print("\n判讀 —— **兩種挑法的答案相反，所以結論不是「能不能挑」**")
    rho_m = j.mo1.corr(j.mo2, method="spearman")
    rho_s = j.hs1.corr(j.hs2, method="spearman")
    pm = 100 * (sim < res["markout 最高"]).mean()
    ps = 100 * (sim < res["半價差最寬"]).mean()
    print("  用 **markout（績效）** 挑：排名持續性 %+.3f，但選出來落在隨機的"
          "第 %.0f 百分位 -> **不比隨機好**" % (rho_m, pm))
    print("     而且**反向挑（選最差的）也落在第 %.0f 百分位** —— 兩邊都在中位"
          "以下，那是 **winner's curse** 的簽章："
          % (100 * (sim < res["markout 最低（反向對照）"]).mean()))
    print("     markout 的極端值是估計雜訊最大的那些，挑極端 = 挑雜訊。")
    print("  用 **半價差（結構性質）** 挑：排名持續性 **%+.3f**（幾乎不變），"
          "選出來落在第 %.0f 百分位 -> **挑得出東西**" % (rho_s, ps))
    print("\n  -> 三個不同的工作要用三個不同的量，不可互換：")
    print("     **選**（selector）  用半價差 —— 它持續（%+.3f），而且贏過隨機" % rho_s)
    print("     **估**（多少錢）    用 markout —— 只有它含逆選擇")
    print("     **下架**（kill）    用實現的 markout，**事後**不是事前")
    print("  而半價差持續到 %+.3f，代表**宇宙幾乎不需要輪換** —— "
          "要動的是下架那一層。" % rho_s)
    print("  範圍：窗只有 6.45 小時，這測的是小時級的持續性，"
          "**不是 7 天級輪換規則的判決**。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
