# -*- coding: utf-8 -*-
"""MRP 第一測：portmanteau GEP 解出**權重**（2026-09-13）

===========================================================================
MRP = Mean-Reverting Portfolio（均值回歸組合）
===========================================================================
傳統配對交易：挑兩個綁在一起的標的，賭價差收斂。
MRP：**不預設是哪幾個、也不預設權重**，解一個最佳化，找出走勢最會回到
自己均值的那一籃子。

作者的界定（`pairs-trading-framework-and-process`，2023-03-23）：
  「Modern pairs trading can involve a variety of assets, with a mix of long
   and short exposures. **They don't have to be cointegrated or stationary
   either.**」
三個要一起最佳化的目標：可預測性／**成本相對利潤**／統計性質的穩定性，
而他明說第三項「常常值得放棄去換 1 和 2 的更好條件」。

===========================================================================
為什麼這一族值得做：**我們從來沒有解過權重**
===========================================================================
| 線 | 籃子 | 權重 | 進出時機 |
|---|---|---|---|
| V7 | 寫死（BTC 一個） | 寫死 1.0 | 學出來（XGBoost）|
| §0.75 套利 | 寫死（同標的兩場館） | **寫死 (+1,−1)** | 寫死（p90 帶）|
| §4.65 分鐘級 | 寫死 | 寫死 | 學出來 |
| **MRP** | **解出來** | **解出來** | z-score／布林 |

而 V7 是均值回歸這件事是我們自己記過的（`project_onchain_etf_overlay_nogo`：
「模型是 4h 均值回歸專家……**因為 4h TWAP target 本身就是均值回歸的**」）。
所以這不是換一條新線，是**把同一族裡唯一沒被解過的自由度解開**。

===========================================================================
方法（逐行照抄 `non-sparse-synthetic-portfolios`，2023-03-08）
===========================================================================
把「最小化 portmanteau 統計量」寫成廣義特徵值問題（GEP）——他自己說
「if this sounds too complicated you'll be happy to hear **we are basically
using PCA**」。**最小特徵值**對應的特徵向量 -> 最會均值回歸的權重；
**最大**的 -> 動量組合（這給了我們一道免費的對照關，見 K2）。

===========================================================================
我們與他不同的四處，全部寫出來
===========================================================================
| 項目 | 他 | 我們 | 後果 |
|---|---|---|---|
| 宇宙 | S&P500 的電力公用事業（同業、同量級） | **29 個加密貨幣**（core 宇宙）| 不同業 -> 均值回歸的機制先驗弱得多 |
| 價格 | 日線收盤（原始價） | **對數價** | 加密幣價差 5 個數量級（BTC 77000 vs DOGE 0.2），不取對數等於只解到 BTC |
| 頻率 | 日 | **小時** | 買賣價跳動的污染更重，見 K4 |
| 資料 | 未明說 | 成交價 bar（**不是中價**）| 他自己警告過：S&P500 選得好是因為「fake mean-reversion from the bid/ask bounce is less pronounced」。我們沒有這個保護 |

===========================================================================
自曝檢查（跑之前寫死）
===========================================================================
K1  **樣本內必然贏**（建構保證），所以樣本內數字**不可引用**。
    只報**樣本外**：每個訓練窗解出的權重，套到下一段沒看過的資料上。
K2  **免費對照關（他的對稱性）**：最大特徵值那個組合應該是**動量**。
    若 MR 組合與動量組合的樣本外 portmanteau **沒有分開**，代表這個解
    在我們的資料上沒有抓到任何東西 —— 這一關不判過不過，它問
    「它在做我以為的事嗎」（mistake.md 2026-09-09）。
K3  **基準**：等權多空、單一標的、以及**隨機權重**（同樣做 1000 次）。
    贏不過隨機權重就不是發現。
K4  **買賣價跳動關（我們加的，他沒有）**：跳動是**每根 bar 一次**的噪音，
    所以它造成的假均值回歸**隨頻率降低而縮小**。把同一組權重在 1h／4h／1d
    上各算一次：真的均值回歸應該撐得住，跳動造的會塌掉。
    （mistake.md 2026-09-11、backtest-audit 第 17/18 項）

**這一支不判過不過。** 它回答「解出來的權重在樣本外是不是真的比寫死的好」。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(ROOT, "research", "sweep_failure", ".cache")
OUT = os.path.join(ROOT, "research", "results", "mrp_portmanteau.json")

# ── 宇宙分組（2026-09-13 加）────────────────────────────────────────
# 他的處方，原話兩次：
#   「To form our portfolios we should **select the optimal assets ahead of
#     time**. Otherwise, we will not be likely to generate a robust portfolio」
#   「**Fundamentally connected assets are the best**」
# 他用的是**同一個產業**（S&P500 的電力公用事業）。我們第一測把 29 個不相干
# 的幣丟進去 -> K2 只有 12/28（擲硬幣），而 lags 掃 10/48/120 一動也不動。
#
# 分組是**事前手寫的、不看資料**（不是分群）—— 用相關性分群會看到全樣本，
# 那是 factor-research 第 2 條擋的事。分完就凍結，不因為結果難看而調整。
GROUPS = {
    "L1_major":  ["BTC", "ETH", "SOL", "BNB"],
    "L1_alt":    ["ADA", "AVAX", "DOT", "ATOM", "NEAR", "ALGO", "HBAR",
                  "APT", "SUI", "ICP"],
    "L1_old":    ["LTC", "ETC", "XRP", "TRX", "VET"],
    "L2":        ["ARB", "OP"],
    "defi":      ["LINK", "UNI", "AAVE", "INJ"],
    "gamefi":    ["SAND", "AXS"],
    "storage":   ["FIL", "ICP"],
    "all":       None,
}

LAGS = 10            # 他的值。「generally recommended to go lower for
                     # mean-reversion, and higher for momentum」
TRAIN = 24 * 90      # 訓練窗：90 天的小時 bar
TEST = 24 * 30       # 測試窗：30 天
STEP = 24 * 30


# ── 他的兩支函式，逐行照抄（只加型別與註解）────────────────────────
def autocov_matrix_calc(arr: np.ndarray, p: int) -> np.ndarray:
    m = arr.shape[0]
    dm = arr - np.nanmean(arr, axis=0)
    return 1 / (m - p - 1) * dm[p:].T @ dm[:m - p]


def portmanteau_gep(df: pd.DataFrame, lags: int):
    """回傳 (特徵值, 權重矩陣)，**由小到大**排序。

    第 0 行（最小特徵值）= 最會均值回歸；最後一行 = 動量。
    """
    dfn = df - df.mean(0)
    rho = dfn.cov().values
    rho_inv_sqrt = np.linalg.inv(sqrtm(rho))
    pmt = 0.0
    for i in range(1, lags):
        ac = autocov_matrix_calc(df.values, i)
        pmt = pmt + np.square(rho_inv_sqrt @ ac @ rho_inv_sqrt)
    pmt /= lags
    ev, vec = np.linalg.eig(pmt)
    asc = np.argsort(ev)
    ev, vec = ev[asc], vec[:, asc]
    return np.real(ev), np.real(rho_inv_sqrt @ vec)


# ── 評分：組合序列有多會均值回歸 ─────────────────────────────────
def portmanteau_stat(x: np.ndarray, lags: int = LAGS) -> float:
    """單一序列的 portmanteau（Ljung-Box 的核心）。**越小越均值回歸。**"""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags * 5:
        return np.nan
    x = x - x.mean()
    d = float(x @ x)
    if d <= 0:
        return np.nan
    s = 0.0
    for k in range(1, lags):
        s += (float(x[k:] @ x[:-k]) / d) ** 2
    return s / lags


def halflife(x: np.ndarray) -> float:
    """OU 半衰期（小時）。dx_t = a + b*x_{t-1} -> hl = −ln2/ln(1+b)。"""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.nan
    y, lag = np.diff(x), x[:-1]
    A = np.vstack([np.ones_like(lag), lag]).T
    try:
        b = np.linalg.lstsq(A, y, rcond=None)[0][1]
    except Exception:                                   # noqa: BLE001
        return np.nan
    if b >= 0 or 1 + b <= 0:
        return np.inf                                   # 不回歸
    return float(-np.log(2) / np.log(1 + b))


BARS_PER = {"1h": 1, "4h": 4, "1d": 24}


def load(freq: str = "1h"):
    """小時快取 -> 指定頻率的收盤價。

    **降頻取「每個區塊的最後一根收盤」**，不是平均 —— 平均會人為引入
    負自相關（平滑），那正好會偽裝成均值回歸，是這條線最該避免的事。
    """
    fs = sorted(glob.glob(os.path.join(CACHE, "*_1h.csv")))
    ser = {}
    for f in fs:
        sym = os.path.basename(f).split("_")[0]
        d = pd.read_csv(f, usecols=["time", "close"])
        ser[sym] = pd.Series(d["close"].values, index=d["time"].values)
    px = pd.DataFrame(ser).sort_index().dropna()
    n = BARS_PER[freq]
    if n > 1:
        # 對齊到區塊邊界再取最後一根：用時戳而不是位置，否則資料頭部
        # 一移動（.cache 是滾動 930 天窗）分界就跟著漂。
        sec = px.index.values.astype("int64")
        sec = sec // (1000 if sec.max() > 1e11 else 1)
        blk = sec // (3600 * n)
        px = px.groupby(blk).last()
        px.index = (pd.Series(sec).groupby(blk).last().values)
    return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, default=LAGS)
    ap.add_argument("--group", default="all", choices=sorted(GROUPS))
    ap.add_argument("--freq", default="1h", choices=["1h", "4h", "1d"])
    ap.add_argument("--train", type=int, default=TRAIN)
    ap.add_argument("--test", type=int, default=TEST)
    a = ap.parse_args()

    px = load(a.freq)
    g = GROUPS[a.group]
    if g:
        keep = [c for c in px.columns if c.replace("USDT", "") in g]
        if len(keep) < 3:
            print("group %s 只湊到 %d 個標的，至少要 3 個" % (a.group, len(keep)))
            return
        px = px[keep]
    # **對數價**（與他不同的一處，理由見檔頭）
    lp = np.log(px)
    print("分組 = %s" % a.group)
    print("頻率 %s" % a.freq)
    print("宇宙 %d 個標的｜%s 根 bar｜完整重疊"
          % (lp.shape[1], format(len(lp), ",")))
    print("標的：%s" % ", ".join(lp.columns))

    rng = np.random.default_rng(0)
    rows = []
    i = a.train
    while i + a.test <= len(lp):
        tr = lp.iloc[i - a.train:i]
        te = lp.iloc[i:i + a.test]
        try:
            ev, W = portmanteau_gep(tr, a.lags)
        except Exception as e:                          # noqa: BLE001
            print("  窗 %d 解不出來：%s" % (i, str(e)[:50])); i += STEP; continue
        w_mr, w_mo = W[:, 0], W[:, -1]
        w_mr = w_mr / np.abs(w_mr).sum()
        w_mo = w_mo / np.abs(w_mo).sum()
        # 樣本外：把權重套到沒看過的那一段
        s_mr = te.values @ w_mr
        s_mo = te.values @ w_mo
        s_eq = te.values @ (np.ones(lp.shape[1]) / lp.shape[1])
        # K3：隨機權重（同樣是多空、同樣 L1 正規化）
        rnd = []
        for _ in range(200):
            w = rng.normal(size=lp.shape[1])
            w /= np.abs(w).sum()
            rnd.append(portmanteau_stat(te.values @ w, a.lags))
        # **樣本內的 K2 純診斷用**（樣本內 MR<動量 是建構保證，不是績效）。
        # 要看的是：樣本內對、樣本外**穩定反過來** = 過擬合的一種具體形狀，
        # 而那與「樣本外只是變雜訊」是不同的病，處置也不同。
        pm_mr_is = portmanteau_stat(tr.values @ w_mr, a.lags)
        pm_mo_is = portmanteau_stat(tr.values @ w_mo, a.lags)
        rows.append(dict(
            i=int(i), t0=int(te.index[0]), t1=int(te.index[-1]),
            pm_mr_is=pm_mr_is, pm_mo_is=pm_mo_is,
            pm_mr=portmanteau_stat(s_mr, a.lags),
            pm_mo=portmanteau_stat(s_mo, a.lags),
            pm_eq=portmanteau_stat(s_eq, a.lags),
            pm_rnd_med=float(np.nanmedian(rnd)),
            pm_rnd_p05=float(np.nanpercentile(rnd, 5)),
            hl_mr=halflife(s_mr), hl_mo=halflife(s_mo),
            ev_min=float(ev[0]), ev_max=float(ev[-1]),
            nz=int((np.abs(w_mr) > 0.02).sum()),
            top=", ".join("%s%+.2f" % (lp.columns[k], w_mr[k])
                          for k in np.argsort(-np.abs(w_mr))[:4])))
        i += a.test       # 步長 = 測試窗（原本寫死 24*30，在日線上那是 720 天 -> 只有 1 個窗）

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 96)
    print("樣本外 portmanteau（**越小越均值回歸**）｜每窗訓練 %d 根、測試 %d 根"
          % (a.train, a.test))
    print("=" * 96)
    print(r[["t0", "pm_mr", "pm_mo", "pm_eq", "pm_rnd_med", "pm_rnd_p05",
             "hl_mr", "nz"]].to_string(index=False,
                                       float_format=lambda v: "%9.4f" % v))

    n = len(r)
    print("\nK1 樣本外窗數 %d（樣本內必然贏，不報）" % n)
    k2 = int((r.pm_mr < r.pm_mo).sum())
    k2is = int((r.pm_mr_is < r.pm_mo_is).sum())
    print("K2 他的對稱性：MR 比動量更均值回歸的窗 **%d / %d**"
          "（沒分開 = 這個解沒抓到東西）" % (k2, n))
    print("   樣本內同一關 %d / %d（**建構保證，純診斷**）-> %s"
          % (k2is, n,
             "樣本內對、樣本外**穩定反過來** = 過擬合的反轉"
             if k2is >= n * 0.9 and k2 <= n * 0.2 else
             "樣本外只是退化成雜訊" if k2is >= n * 0.9 else
             "**樣本內都不對 -> 先查程式**"))
    k3a = int((r.pm_mr < r.pm_rnd_med).sum())
    k3b = int((r.pm_mr < r.pm_rnd_p05).sum())
    print("K3 對隨機權重：贏中位 %d/%d｜贏 p05 %d/%d"
          "（**贏不過隨機就不是發現**）" % (k3a, n, k3b, n))
    print("   MR 中位 %.4f｜動量中位 %.4f｜等權中位 %.4f｜隨機中位 %.4f"
          % (r.pm_mr.median(), r.pm_mo.median(), r.pm_eq.median(),
             r.pm_rnd_med.median()))
    fin = r.hl_mr[np.isfinite(r.hl_mr)]
    print("   MR 半衰期中位 %.1f 根 bar（%d/%d 個窗有限）"
          % (fin.median() if len(fin) else float("nan"), len(fin), n))
    print("\n權重最大的四個（每窗）：")
    for _, x in r.iterrows():
        print("   %s  nz=%2d  %s" % (str(int(x.t0))[:10], int(x.nz), x.top))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "group": a.group, "freq": a.freq,
                   "universe": list(lp.columns),
                   "lags": a.lags, "train": a.train, "test": a.test,
                   "k2": k2, "k3_med": k3a, "k3_p05": k3b, "n": n},
                  fh, ensure_ascii=False, indent=1)
    print("\n寫出 %s" % OUT)
    print("**這一支不判過不過** —— 它回答「解出來的權重在樣本外是不是真的"
          "比寫死的好」。K4（1h/4h/1d 的跳動關）另一支。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
