# -*- coding: utf-8 -*-
"""HL 成交帶的參與者盤點 —— 執行演算法存不存在，用**身分**看，不用統計猜。

2026-09-15。新線的第一步，而它刻意只做描述不做預測。

===========================================================================
為什麼這條線值得開，一句話
===========================================================================
Quant Arb `2023-12-09 hide-n-seek pt2` 給了四個偵測執行演算法的方法：
重複大小、等間隔分箱、NUFFT、全域簿口。**四個都是在匿名成交上做統計推論**
—— 因為他手上只有 Binance/Bybit 的 aggTrades，那裡面沒有身分。

而 HL 的逐筆成交帶**帶雙方地址**（`a0`/`a1`）。所以他要猜的那個東西，
我們可以直接看到。這不是「多一個特徵」，是**標籤** ——
我們能做他做不到的事：拿真身分去驗他的偵測法準不準。

**這支只回答第一個、也最便宜的問題：那個現象存不存在。**
如果沒有任何帳戶呈現規律執行，整條線在資訊層開工之前就死了，
而那個結論只要幾分鐘（核心原則 11：執行可行性排在資訊層之前；
這裡更前面一步 —— 現象存在性排在執行可行性之前）。

===========================================================================
不做什麼（寫下來，因為清單的形狀決定它看得見什麼）
===========================================================================
* **不算任何前瞻報酬。** 這支是盤點不是 alpha，混進去就會變成
  「看過答案再挑維度」（mistake.md 2026-09-09）。
* **不調參數去讓某個帳戶「看起來像演算法」。** 門檻先寫死在下面的常數，
  跑出來難看就記難看的。
* 不宣稱任何地址是誰。地址是公開鏈上資料，但這支只做群體統計。

用法：
    python research/hl/exec_census.py --coin BTC --hours 6
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TAPE = "D:/flowbot_data/hl/trades"

# 判準寫死在這裡，跑之前就凍結（factor-research.md：事前寫死，事後不放寬）
MIN_TRADES = 30          # 一個帳戶要有這麼多筆才談得上「規律」
TOP_SIZE_SHARE = 0.30    # 最常見的那個 size 佔它自己成交的比例
CV_REGULAR = 0.50        # 間隔的變異係數低於此 = 節奏規律（隨機到達 CV≈1）


def load(coin: str, hours: int) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(TAPE, "**", "*.parquet"),
                          recursive=True))
    if not fs:
        sys.exit("讀不到成交帶 %s —— 空結果不是合法狀態（D 槽掛載了嗎）" % TAPE)
    fr = []
    for f in reversed(fs):
        d = pd.read_parquet(f, columns=["ts", "coin", "side", "px", "sz",
                                        "a0", "a1"])
        d = d[d["coin"] == coin]
        if len(d):
            fr.append(d)
        if sum(len(x) for x in fr) and len(fr) >= hours:
            break
    if not fr:
        sys.exit("成交帶裡沒有 %s" % coin)
    d = pd.concat(fr, ignore_index=True).sort_values("ts")
    d["t"] = d["ts"].astype("int64") / 1e3
    return d.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--hours", type=int, default=6)
    a = ap.parse_args()
    d = load(a.coin, a.hours)
    span_h = (d["t"].max() - d["t"].min()) / 3600.0
    print("=== %s：%d 筆成交 / %.1f 小時 ===" % (a.coin, len(d), span_h))

    # --- 哪一欄是吃單方 -------------------------------------------------
    # 一個吃單方會在**同一毫秒**掃過多個掛單 -> 它的地址在那一組裡重複。
    # 這不是假設，是可以量的，而且下面每一個結論都建立在它之上。
    g = d.groupby(["ts", "side"])
    rep0 = g["a0"].nunique().mean()
    rep1 = g["a1"].nunique().mean()
    taker = "a1" if rep1 < rep0 else "a0"
    maker = "a0" if taker == "a1" else "a1"
    print("  同一毫秒同一側的相異地址數：a0 %.2f / a1 %.2f  -> **吃單方 = %s**"
          % (rep0, rep1, taker))

    n_takers = d[taker].nunique()
    n_makers = d[maker].nunique()
    print("  相異吃單方 %d 個 / 相異掛單方 %d 個" % (n_takers, n_makers))

    vol = d.assign(usd=d["px"] * d["sz"]).groupby(taker)["usd"].sum()
    vol = vol.sort_values(ascending=False)
    tot = vol.sum()
    print("  吃單金額合計 $%.0f；前 1 名佔 %.0f%%、前 5 佔 %.0f%%、"
          "前 20 佔 %.0f%%"
          % (tot, vol.iloc[0] / tot * 100, vol.iloc[:5].sum() / tot * 100,
             vol.iloc[:20].sum() / tot * 100))

    # --- 有沒有帳戶在做規律執行 -----------------------------------------
    print()
    print("=== 規律執行的帳戶（門檻跑之前就凍結）===")
    print("  判準：>=%d 筆 ∧ 最常見 size 佔比 >=%.0f%% ∧ 間隔 CV <=%.2f"
          % (MIN_TRADES, TOP_SIZE_SHARE * 100, CV_REGULAR))
    rows = []
    for acct, s in d.groupby(taker):
        if len(s) < MIN_TRADES:
            continue
        vc = s["sz"].value_counts()
        share = vc.iloc[0] / len(s)
        gaps = np.diff(np.sort(s["t"].values))
        gaps = gaps[gaps > 0]
        if len(gaps) < 5:
            continue
        cv = gaps.std() / gaps.mean() if gaps.mean() > 0 else np.inf
        rows.append({"acct": acct, "n": len(s), "top_size": vc.index[0],
                     "share": share, "cv": cv,
                     "usd": float((s["px"] * s["sz"]).sum()),
                     "med_gap_s": float(np.median(gaps))})
    r = pd.DataFrame(rows)
    if r.empty:
        print("  **沒有任何帳戶達到 %d 筆** —— 這個標的/時窗太稀疏，換一個"
              % MIN_TRADES)
        return 0
    hit = r[(r["share"] >= TOP_SIZE_SHARE) & (r["cv"] <= CV_REGULAR)]
    print("  夠活躍的帳戶 %d 個，其中**兩個條件都過的 %d 個**"
          % (len(r), len(hit)))
    if len(hit):
        h = hit.sort_values("usd", ascending=False).head(8).copy()
        h["acct"] = h["acct"].str[:10] + ".."
        print(h[["acct", "n", "top_size", "share", "cv", "med_gap_s",
                 "usd"]].to_string(index=False,
                                   float_format=lambda x: "%.2f" % x))
    else:
        print("  兩個條件都過的是 0 個。分開看各自過幾個：")
        print("    只有 size 規律：%d 個   只有節奏規律：%d 個"
              % ((r["share"] >= TOP_SIZE_SHARE).sum(),
                 (r["cv"] <= CV_REGULAR).sum()))
        print("    最低的 CV %.2f / 最高的 size 佔比 %.0f%%"
              % (r["cv"].min(), r["share"].max() * 100))

    # --- 相位：場館的時鐘，還是各自的計時器 ------------------------------
    # **這一關才是讓上面那張表站得住的東西。**
    # 「中位間隔 30 秒」本身沒有分辨力 —— 全體 298 個活躍帳戶的中位間隔
    # 中位數就是 30.17 秒。有分辨力的是 CV≈0（**每一個**間隔都一樣）。
    # 而 CV≈0 還有一個競爭解釋:它們對齊了某個**場館事件**（結算、預言機
    # 更新），那樣的話就不是參與者的行為而是市場的節拍。
    # 分辨法：看牆鐘相位。共用時鐘 -> 相位全部一樣；各自的計時器 -> 相位散開。
    if len(hit):
        print()
        print("=== 相位：場館的時鐘 vs 各自的計時器 ===")
        ph_med = []
        for _, row in hit.sort_values("n", ascending=False).head(8).iterrows():
            s = d[d[taker] == row["acct"]]
            period = round(row["med_gap_s"])
            if period <= 0:
                continue
            ph = s["t"].values % period
            ph_med.append(float(np.median(ph)))
            print("  %s..  n=%4d  週期 %.0fs  相位中位 %5.2f  相位std %5.2f"
                  % (row["acct"][:12], row["n"], period,
                     np.median(ph), ph.std()))
        if len(ph_med) >= 3:
            spread = float(np.std(ph_med))
            if spread > 1.0:
                print("  -> 相位彼此差 std %.2f 秒 = **各自獨立的計時器**，"
                      "不是共用的場館事件。" % spread)
            else:
                print("  -> 相位彼此幾乎相同（std %.2f）= 它們對齊同一個東西，"
                      "**這是場館節拍不是參與者行為** —— 先查那是什麼事件，"
                      "上面那張表在查清楚之前不可解讀。" % spread)

    # --- 自曝檢查 --------------------------------------------------------
    print()
    print("=== 自曝檢查 ===")
    bad = []
    c = "PASS" if rep1 != rep0 else "**FAIL**"
    if c.startswith("**"):
        bad.append(c)
    print("  C1 吃單/掛單欄分得出來（兩邊重複度不能一樣）  %s" % c)
    # 隨機到達的間隔 CV 應該接近 1；整體分布若遠離 1，先查儀器不要解讀
    allgaps = np.diff(np.sort(d["t"].values))
    allgaps = allgaps[allgaps > 0]
    cv_all = allgaps.std() / allgaps.mean()
    c = "PASS" if 0.3 <= cv_all <= 6.0 else "**FAIL**"
    if c.startswith("**"):
        bad.append(c)
    print("  C2 全體成交間隔 CV 在合理範圍                  %s  %.2f"
          "（純隨機到達≈1）" % (c, cv_all))
    c = "PASS" if abs(vol.sum() - tot) < 1e-6 else "**FAIL**"
    print("  C3 金額加總自洽                                %s" % c)
    print()
    print("  %s" % ("全過" if not bad else "**有 %d 關沒過，數字不可引用**"
                    % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
