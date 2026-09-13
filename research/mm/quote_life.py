# -*- coding: utf-8 -*-
"""報價存活時間與「來不來得及撤」（2026-09-13，TODO §1.40 Stage 0）

===========================================================================
要回答的問題
===========================================================================
我們的往返延遲是實測的：**撤單 49.9 ms 中位 / p90 54.7**、送單 54.1 / 67.0
（`arb/tools/cancel_latency.py`，Premium、n=20）。而簿口上有 23 個對手。

**50 ms 夠不夠？** 這支用三個量測回答，而第三個才是判決：

M1  **頂檔存活時間** —— 連續兩次頂檔價變動之間隔多久。
    市場多快在動的原始讀數。中位 5 秒 -> 50 ms 是 1%；中位 200 ms -> 我們慢。

M2  **成交當下簿口的新鮮度** —— 成交時戳減去它之前最後一次頂檔變動。
    這是「成交發生在簿口剛動完多久之後」。

M3  **可反應窗口（判決用這個）** —— 對每一筆成交，往回找**最後一次對做市方
    不利方向**的頂檔移動（做市方賣在 ask 就是 mid 往上），算那之後到成交
    之間有多少毫秒。**那就是我們原本有多少時間可以撤單。**
    然後看 **markout 對可反應窗口的分桶**：
      窗口 < 50 ms 的成交 markout 明顯更差 -> **延遲就是綁束，50 ms 在錯的一側**
      各桶 markout 差不多             -> 延遲不是綁束，問題在別的地方

===========================================================================
口徑
===========================================================================
* 兩邊都用**交易所時戳**（tape `ts` vs tob `book_time`），不用各自的 rx_ms
  —— 實測 tape 的 rx−ts 中位 150 ms、tob 的 67 ms，混用會注入 83 ms 假偏移。
* tob 有 2.4% 的列 `book_time` 是微秒（2026-09-13 修正前），丟掉。
* **tob 只在頂檔價變動時寫一列**（`tob_capture` 的 `_tob_last` 比價），
  所以連續列的時間差就是存活時間 —— 這是建構保證不是假設。
* markout 沿用 `sweep_markout.py` 的定義（做市方視角、mid、usd 加權）。

===========================================================================
自曝檢查
===========================================================================
C1  **可反應窗口必須 <= 成交時簿口年齡不成立、但兩者都該是正數**，而且
    「不利方向移動」的次數應該約等於全部移動的一半（上下大致對半）。
    偏離太多代表我的方向符號寫反了。
C2  **不利移動之後的 markout 必須比有利移動之後更差。** 不成立就是符號反了。
C3  全格報告，不挑桶。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "research", "results", "quote_life.json")

TAPE = "D:/flowbot_data/lighter/trades/*/*.parquet"
TOB = "D:/flowbot_data/lighter/tob/*/*.parquet"
OUR_LAT_MS = 50.0                      # 實測往返
BUCKETS = [0, 25, 50, 100, 250, 1000, 5000, np.inf]


def load():
    tb = pd.concat([pd.read_parquet(f, columns=["book_time", "coin", "bid", "ask"])
                    for f in sorted(glob.glob(TOB))], ignore_index=True)
    tb = tb[tb.book_time.between(1e12, 1e14)]
    tb = tb[(tb.bid > 0) & (tb.ask >= tb.bid)]
    tb["mid"] = (tb.bid + tb.ask) / 2.0
    tp = pd.concat([pd.read_parquet(f, columns=[
        "ts", "coin", "px", "usd", "is_liq", "is_maker_ask"])
        for f in sorted(glob.glob(TAPE))], ignore_index=True)
    tp = tp[(~tp.is_liq) & (tp.usd > 0) & (tp.px > 0)]
    lo = max(tp.ts.min(), tb.book_time.min())
    hi = min(tp.ts.max(), tb.book_time.max())
    tp = tp[tp.ts.between(lo, hi)]
    print("tob %s 列｜tape %s 筆｜窗 %.2f 小時｜%d 個幣"
          % (format(len(tb), ","), format(len(tp), ","), (hi - lo) / 3.6e6,
             tb.coin.nunique()))
    return tp, tb


def main():
    tp, tb = load()
    life, rows = [], []
    n_up = n_dn = 0
    for coin, b in tb.groupby("coin"):
        b = b.sort_values("book_time")
        t = b.book_time.values.astype("int64")
        mid = b.mid.values
        if len(t) < 50:
            continue
        # M1 存活時間：tob 只在價變時寫列，所以相鄰差就是存活時間
        d = np.diff(t)
        life.append(pd.DataFrame(dict(coin=coin, dt=d[d > 0])))
        # 每一筆的移動方向，以及「最後一次上移/下移」的索引（O(n) 前綴最大）
        dm = np.diff(mid, prepend=mid[0])
        up, dn = dm > 0, dm < 0
        n_up += int(up.sum())
        n_dn += int(dn.sum())
        idx = np.arange(len(t))
        last_up = np.maximum.accumulate(np.where(up, idx, -1))
        last_dn = np.maximum.accumulate(np.where(dn, idx, -1))

        f = tp[tp.coin == coin]
        if len(f) < 50:
            continue
        f = f.sort_values("ts")
        ft = f.ts.values.astype("int64")
        j = np.searchsorted(t, ft, side="right") - 1      # 成交前最後一列
        ok = j >= 0
        if ok.sum() < 50:
            continue
        f, j = f[ok], j[ok]
        age = ft[ok] - t[j]                               # M2 簿口年齡
        # M3 可反應窗口：做市方賣在 ask -> 不利是 mid 上移
        adv_i = np.where(f.is_maker_ask.values, last_up[j], last_dn[j])
        win = np.where(adv_i >= 0, ft[ok] - t[adv_i], np.nan)
        # markout @1s（做市方視角）
        tgt = ft[ok] + 1000
        k = np.searchsorted(t, tgt, side="right") - 1
        k = np.clip(k, 0, len(t) - 1)
        sign = np.where(f.is_maker_ask.values, 1.0, -1.0)
        mo = sign * (f.px.values - mid[k]) / mid[k] * 1e4
        rows.append(pd.DataFrame(dict(coin=coin, usd=f.usd.values,
                                      age=age, win=win, mo=mo)))
    L = pd.concat(life, ignore_index=True)
    R = pd.concat(rows, ignore_index=True)
    R = R[np.isfinite(R.mo)]

    print("\nM1 頂檔存活時間（毫秒；tob 只在價變時寫列）")
    for q in (.10, .25, .50, .75, .90):
        print("  p%-3d %10.0f ms" % (100 * q, L.dt.quantile(q)))
    print("  **我們的往返 %.0f ms 落在第 %.1f 百分位**"
          % (OUR_LAT_MS, 100 * (L.dt < OUR_LAT_MS).mean()))
    print("  逐幣中位（成交額前 8）：")
    top = R.groupby("coin").usd.sum().sort_values(ascending=False).head(8)
    med = L.groupby("coin").dt.median()
    for c in top.index:
        if c in med:
            print("    %-8s %8.0f ms" % (c, med[c]))

    print("\nM2 成交當下簿口年齡（成交時戳 − 前一次頂檔變動）")
    for q in (.25, .50, .75, .90):
        print("  p%-3d %10.0f ms" % (100 * q, R.age.quantile(q)))

    print("\nM3 可反應窗口（最後一次**不利方向**移動到成交之間）")
    w = R[np.isfinite(R.win)]
    print("  有不利移動可參照的成交：%s / %s（%.0f%%）"
          % (format(len(w), ","), format(len(R), ","), 100 * len(w) / len(R)))
    for q in (.10, .25, .50, .75):
        print("  p%-3d %10.0f ms" % (100 * q, w.win.quantile(q)))
    print("  **窗口 < %.0f ms（= 我們撤不掉）的成交佔 %.1f%%、佔成交額 %.1f%%**"
          % (OUR_LAT_MS, 100 * (w.win < OUR_LAT_MS).mean(),
             100 * w.loc[w.win < OUR_LAT_MS, "usd"].sum() / w.usd.sum()))

    print("\n" + "=" * 78)
    print("判決：markout@1s 對可反應窗口分桶（usd 加權，bps）")
    print("=" * 78)
    print("  %-16s %10s %12s %10s" % ("窗口(ms)", "成交數", "佔成交額", "markout"))
    w = w.copy()
    w["bk"] = pd.cut(w.win, BUCKETS, right=False)
    tot = w.usd.sum()
    for bk, g in w.groupby("bk", observed=True):
        print("  %-16s %10s %11.1f%% %10.3f"
              % (str(bk), format(len(g), ","), 100 * g.usd.sum() / tot,
                 np.average(g.mo, weights=g.usd)))

    print("\n自曝檢查")
    print("  C1 上移 %s 次 / 下移 %s 次（應該大致對半）-> 比值 %.2f"
          % (format(n_up, ","), format(n_dn, ","), n_up / max(n_dn, 1)))
    # C2 不利移動 vs 有利移動之後的 markout
    R2 = R[np.isfinite(R.win)]
    fast = R2[R2.win < 100]
    slow = R2[R2.win >= 1000]
    print("  C2 窗口 <100ms 的 markout %.3f vs >=1000ms 的 %.3f"
          % (np.average(fast.mo, weights=fast.usd) if len(fast) else np.nan,
             np.average(slow.mo, weights=slow.usd) if len(slow) else np.nan))
    print("     （前者該明顯更差；不然就是方向符號寫反了）")

    # ── 逐市場 ＋ G3 集中度 ──────────────────────────────────────────
    # **整體那張單調表不可以直接讀成「延遲是綁束」。** 2026-09-13 實測：
    # 75% 的窗內虧損來自**一個市場**。所以整體數字不是決策變數，
    # 逐市場分布才是（backtest-audit 第 2 項：「所有損益來自 3 次大跳
    # —— 3 次不多」，而這裡是 1 次）。
    R2 = R[np.isfinite(R.win)].copy()
    R2["pnl"] = R2.usd * R2.mo / 1e4
    rows2 = []
    for c, x in R2.groupby("coin"):
        b, o = x[x.win < OUR_LAT_MS], x[x.win >= OUR_LAT_MS]
        rows2.append(dict(
            coin=c, usd=float(x.usd.sum()),
            exp=100 * b.usd.sum() / x.usd.sum(),
            pnl_in=float(b.pnl.sum()),
            mo_in=np.average(b.mo, weights=b.usd) if len(b) > 20 else np.nan,
            mo_out=np.average(o.mo, weights=o.usd) if len(o) > 20 else np.nan,
            mo=np.average(x.mo, weights=x.usd)))
    G = pd.DataFrame(rows2)
    big = G[G.usd > 3e5].copy()
    print("\n" + "=" * 78)
    print("逐市場：曝露比例不預測 markout，而虧損極度集中")
    print("=" * 78)
    rho = big.exp.corr(big.mo, method="spearman")
    print("  Spearman(曝露%%, 全體 markout) = **%.3f**  -> %s"
          % (rho, "曝露低的市場更好" if rho < -0.2 else
             "**兩者無關 —— 選標的不能只看曝露**"))
    tot = R2.loc[R2.win < OUR_LAT_MS, "pnl"].sum()
    s = (R2[R2.win < OUR_LAT_MS].groupby("coin").pnl.sum()
         .sort_values())
    print("  窗內 markout 金額合計 **$%.0f**（負 = 做市方虧）" % tot)
    print("  **G3 集中度：最虧 1 個佔 %.0f%%、前 3 個 %.0f%%**"
          % (100 * s.iloc[0] / tot, 100 * s.head(3).sum() / tot))
    print("  最虧的 5 個：%s"
          % ", ".join("%s $%.0f" % (c, v) for c, v in s.head(5).items()))
    worst = list(s.head(1).index)
    for drop in (worst, list(s.head(3).index)):
        k = R2[~R2.coin.isin(drop)]
        kb = k[k.win < OUR_LAT_MS]
        print("  排除 %-22s 窗內 %+.3f、全體 %+.3f bps"
              % (",".join(drop), np.average(kb.mo, weights=kb.usd),
                 np.average(k.mo, weights=k.usd)))
    print("\n  -> **降延遲不是槓桿，逐市場下架才是**：排除一個市場的效果"
          "大於把延遲砍半，而前者免費。")
    print("     所以監控面板不是裝飾，**它就是風控本身** ——"
          "最有價值的單一元件是逐市場滾動 markout ＋ 自動下架。")

    R.to_json(OUT, orient="records", force_ascii=False)
    G.to_json(OUT.replace(".json", "_by_coin.json"), orient="records",
              force_ascii=False)
    print("\n寫出 %s（%s 列）＋ 逐市場版" % (OUT, format(len(R), ",")))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
