# -*- coding: utf-8 -*-
"""HFT Alphas Pt.1 的特徵，跑在**我們自己的** Lighter 5 秒中價上（2026-09-13）

===========================================================================
這是什麼，以及**不是**什麼
===========================================================================
使用者 2026-09-13：「開始探索 quant arb 的 alpha 策略，搞不好會有跟我系統
相呼應的，沒有也沒關係我們建設出一個更強或衍生的」。

**這不是複現。** 差異逐項寫出來，因為不寫出來就會被當成複現：

| 項目 | 他（HFT Alphas Pt.1） | 我們 |
|---|---|---|
| 場館 | Binance USD-M 期貨 | **Lighter mainnet 永續** |
| 宇宙 | BTC / ETH / SOL | **前 80 名**（lighter_mid 的宇宙）|
| 期間 | 2025 全年，逐 tick | **約 10 小時**，5 秒取樣 |
| bar | 由 quote 中價組 5 秒 OHLC | **5 秒的點取樣**（沒有 O/H/L）|
| 目標 | 5 秒與**不重疊**的 15 秒（t+5 -> t+15）| 同（不重疊，照抄）|

所以**只測得動不需要 OHLC 的那兩個**，而那剛好是他說最強的兩個：

    obi_1bp             = (bid_d1 − ask_d1) / (bid_d1 + ask_d1)
    best_size_imbalance = (bid_sz  − ask_sz) / (bid_sz  + ask_sz)

原文的結論（**這就是我們的已知答案**）：
  「Orderbook imbalance dominates performance wise」
  「the reversal effect was rather weak despite its high strength on the
    1min-1h timeframes (known prior)」
  兩個簿口特徵在 5s 與 15s **都**很強，而 obi_1bp 與 best_size_imbalance
  的曲線「look very similar」（他打算在第 3 篇做殘差化）。

===========================================================================
自曝檢查（跑之前寫死）
===========================================================================
K1  **已知答案對照**：兩個簿口特徵的 |IC| 在 5 秒上都要 > 0.02，而且
    5 秒 > 15 秒（他量到 0.126–0.139 -> 0.107）。差一個數量級就先查儀器。
K2  **相關性**：corr(obi_1bp, best_size_imbalance) 應該很高（他說曲線幾乎
    一樣）。若接近 0，代表我們算的不是同一對東西。
K3  **不重疊**：15 秒的目標是 t+5 -> t+15，**不是** t -> t+15。
    累積區間會讓一個只有 5 秒的效應在 15 秒也顯示為有效
    （mistake.md 2026-09-11）。
K4  **逐幣全格報告**，不只池化。池化的 IC 會被最活躍的幾個幣主導。
K5  **用中價不用成交價**（lighter_mid 本來就是中價）—— 成交價在薄簿口上
    自帶負自相關，會偽裝成反轉（mistake.md 2026-09-11）。

**這一支不判過不過。** 樣本只有 10 小時，它回答的是「這個效應在我們的
場館上存不存在、量級對不對」，不是「可不可以交易」。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MID = "D:/flowbot_data/lighter/mid"
OUT = os.path.join(ROOT, "research", "results", "hft_alphas_lighter.json")

BAR_SEC = 5.0                 # lighter_mid 的取樣週期
MIN_BARS = 300                # 每幣至少要這麼多根才報


def load(days=None):
    fs = sorted(glob.glob(MID + "/*/*.parquet"))
    if days:
        fs = [f for f in fs if os.path.basename(os.path.dirname(f)) in days]
    cols = ["bucket_ts", "coin", "mid", "bid_sz", "ask_sz",
            "bid_d1", "ask_d1", "bid_d5", "ask_d5", "bid_d10", "ask_d10",
            "spread_bps", "stale_ms"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs],
                  ignore_index=True)
    return d.sort_values(["coin", "bucket_ts"]).reset_index(drop=True)


def imb(b, a):
    """他給的 safe_imbalance：兩邊都 0 時回 0，不是 nan。"""
    b = np.asarray(b, dtype="float64")
    a = np.asarray(a, dtype="float64")
    t = b + a
    out = np.zeros_like(b)
    m = t > 1e-10
    out[m] = (b[m] - a[m]) / t[m]
    return out


def build(d):
    g = d.groupby("coin", sort=False)
    d = d.copy()
    d["obi_1bp"] = imb(d.bid_d1, d.ask_d1)
    d["obi_5bp"] = imb(d.bid_d5, d.ask_d5)
    d["obi_10bp"] = imb(d.bid_d10, d.ask_d10)
    d["best_size_imbalance"] = imb(d.bid_sz, d.ask_sz)

    # **不重疊的區間**（K3）。lighter_mid 是 5 秒一格，所以
    #   r5  = mid(t+1格) / mid(t) − 1
    #   r15 = mid(t+3格) / mid(t+1格) − 1     <- 從 t+5s 到 t+15s，不含 t..t+5
    lm = np.log(d["mid"].astype("float64"))
    d["_lm"] = lm
    d["r5"] = g["mid"].shift(-1) / d["mid"] - 1.0
    d["r15"] = g["mid"].shift(-3) / g["mid"].shift(-1) - 1.0
    # 跨幣邊界：shift 會把下一個幣的值帶進來，用 bucket 連續性擋掉
    dt = g["bucket_ts"].shift(-1) - d["bucket_ts"]
    d.loc[dt.isna() | (dt > BAR_SEC * 1000 * 1.5), "r5"] = np.nan
    dt3 = g["bucket_ts"].shift(-3) - d["bucket_ts"]
    d.loc[dt3.isna() | (dt3 > BAR_SEC * 1000 * 3.5), "r15"] = np.nan
    return d


FEATS = ["obi_1bp", "obi_5bp", "obi_10bp", "best_size_imbalance"]


def ic(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    xs, ys = pd.Series(x[m]), pd.Series(y[m])
    if xs.std() == 0 or ys.std() == 0:
        return np.nan, int(m.sum())
    return float(xs.corr(ys, method="spearman")), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=None)
    a = ap.parse_args()
    d = load(a.days)
    print("lighter_mid：%s 列｜%d 個標的｜%.1f 小時"
          % (format(len(d), ","), d.coin.nunique(),
             (d.bucket_ts.max() - d.bucket_ts.min()) / 3.6e6))
    d = build(d)

    print("\n" + "=" * 78)
    print("池化 IC（Spearman）｜r5 = t->t+5s｜r15 = **t+5s->t+15s（不重疊）**")
    print("=" * 78)
    print("%-22s %10s %10s %12s %12s" % ("特徵", "IC(5s)", "IC(15s)", "n(5s)", "5s>15s?"))
    print("-" * 70)
    res = {}
    for f in FEATS:
        i5, n5 = ic(d[f].values, d["r5"].values)
        i15, _ = ic(d[f].values, d["r15"].values)
        ok = "YES" if (np.isfinite(i5) and np.isfinite(i15)
                       and abs(i5) > abs(i15)) else "no"
        print("%-22s %10.4f %10.4f %12s %12s"
              % (f, i5, i15, format(n5, ","), ok))
        res[f] = {"ic5": i5, "ic15": i15, "n": n5}

    print("\nK1 已知答案對照（他：5s 0.126–0.139、15s 0.107，簿口最強）")
    best = max(FEATS, key=lambda f: abs(res[f]["ic5"]) if np.isfinite(res[f]["ic5"]) else -1)
    print("   我們最強的是 **%s**（|IC5| %.4f）" % (best, abs(res[best]["ic5"])))
    k1 = all(abs(res[f]["ic5"]) > 0.02 for f in ("obi_1bp", "best_size_imbalance")
             if np.isfinite(res[f]["ic5"]))
    print("   兩個簿口特徵 |IC5| > 0.02：%s" % ("PASS" if k1 else "**FAIL -> 先查儀器**"))

    c = d[["obi_1bp", "best_size_imbalance"]].corr().iloc[0, 1]
    print("\nK2 corr(obi_1bp, best_size_imbalance) = %.3f"
          "（他說兩條曲線幾乎一樣 -> 應該高）" % c)

    print("\nK4 逐幣（|IC5| 前 10，n>=%d）" % MIN_BARS)
    rows = []
    for coin, g in d.groupby("coin"):
        if len(g) < MIN_BARS:
            continue
        i5, n5 = ic(g["obi_1bp"].values, g["r5"].values)
        j5, _ = ic(g["best_size_imbalance"].values, g["r5"].values)
        rows.append((coin, n5, i5, j5, float(g.spread_bps.median())))
    r = pd.DataFrame(rows, columns=["coin", "n", "obi_1bp", "best_size_imb",
                                    "spread_bps"])
    r = r.reindex(r.obi_1bp.abs().sort_values(ascending=False).index)
    print(r.head(10).to_string(index=False, float_format=lambda v: "%9.4f" % v))
    pos = int((r.obi_1bp > 0).sum())
    print("\n   逐幣符號：obi_1bp 正 %d / %d（%.0f%%）｜中位 IC %.4f"
          % (pos, len(r), 100 * pos / max(len(r), 1), r.obi_1bp.median()))

    import json
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"pooled": res, "corr_obi_best": float(c),
                   "per_coin": r.to_dict("records"),
                   "hours": float((d.bucket_ts.max() - d.bucket_ts.min()) / 3.6e6),
                   "coins": int(d.coin.nunique())}, fh, ensure_ascii=False, indent=1)
    print("\n寫出 %s" % OUT)
    print("**這一支不判過不過** —— 10 小時只能回答「存不存在、量級對不對」。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
