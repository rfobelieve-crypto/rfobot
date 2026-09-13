# -*- coding: utf-8 -*-
"""稀疏 MRP：截斷法 vs 貪婪法 × 三個目標函數（2026-09-13）

接 §1.38。非稀疏解每個標的都有權重 -> 不可交易，而且作者自己說
「there is a much larger probability of overfitting」——我們也看到了那個簽名
（每個窗的主權重標的都不一樣）。這支做稀疏化。

===========================================================================
三篇原文，逐項照抄
===========================================================================
`truncation-method-for-smrps`（2023-03-16）
    非稀疏解 -> 按**對名目的貢獻**排序 -> 取前 k 檔。
    兩個變體：截斷後**重訓** vs **不重訓**。
    他的已知答案：「the portfolio that **wasn't retrained** is stronger OOS.
    **Less is more**... reducing the amount of fitting we perform is a great
    way to ensure robust portfolios.」-> 這是 **K5**。

`greedy-method-for-smrps`（2023-03-20，付費）
    cardinality k = **6**（「可交易，且在捕捉均值回歸與過擬合之間取得平衡」）。
    **前兩檔用暴力搜尋**，之後每次貪婪加一檔。
    他說貪婪**遠優於**截斷 -> 這是 **K6**。

    **為什麼不從「單一最均值回歸的資產」起步**（本系列最有價值的一段）：
      「那檔通常也是**最不流動**的一檔。看到的均值回歸很可能是**買賣價跳動**，
       因為資料是**成交價不是中價**。」
    —— 也就是說**跳動防禦被直接做進演算法**。我們原本打算另外加的 K4
    在這裡變成了方法的一部分，而我們的資料正是成交價 bar，所以更需要它。

`greedy-method-for-smrps-part-2`（2023-03-29，付費）
    另外兩個目標函數：
      **BTCD**（Box-Tiao）：最小化 VAR(1) 的可預測性
        = 最小化「VAR(1) 預測值的變異數 ÷ 實現值的變異數」
        直覺：最小化漂移項，剩下的就只有雜訊。
      **Crossing**：價格穿越均值的次數，鬆弛成 **lag=1 的 portmanteau**。

===========================================================================
我們與他不同的地方（除了 §1.38 已列的四處）
===========================================================================
* **對名目的貢獻在對數價空間就是 |w| 本身。** 他用原始價，所以要用價格調整；
  我們用對數價，而 d(ln p) = 報酬，所以 w 直接就是資金權重。少一步轉換。
* cardinality 掃 3/4/6/8，不是只用他的 6 —— 但**全格報告**，不挑最好的那格。

===========================================================================
自曝檢查
===========================================================================
K5  截斷**不重訓** 應該勝過 **重訓**（他的已知答案）
K6  **貪婪** 應該勝過 **截斷**（他的已知答案）
K7  三個目標函數都要贏過**隨機挑同樣檔數 + 隨機權重**
    （贏不過隨機就不是發現；§1.38 的 K3 同一條）
K8  **跳動關**：把選出來的稀疏組合在 1h / 4h / 1d 各算一次 portmanteau。
    跳動是每根 bar 一次的噪音 -> 假均值回歸**隨頻率降低而縮小**。
    真的均值回歸應該撐得住。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "research", "results", "mrp_sparse.json")

from mrp_portmanteau import (GROUPS, autocov_matrix_calc, halflife,  # noqa: E402
                             load, portmanteau_gep, portmanteau_stat)

LAGS = 10


# ── 三個目標函數：回傳 (值, 權重) ──────────────────────────────────
def obj_portmanteau(df, lags=LAGS):
    ev, W = portmanteau_gep(df, lags)
    return float(ev[0]), W[:, 0]


def obj_crossing(df, lags=None):
    """穿越統計量，鬆弛成 lag=1 的 portmanteau（他 part 2 的說法）。"""
    ev, W = portmanteau_gep(df, 2)
    return float(ev[0]), W[:, 0]


def obj_btcd(df, lags=None):
    """Box-Tiao：最小化 VAR(1) 的可預測性。

    VAR(1) 用最小平方直接解（不引 statsmodels）：X_t = A X_{t-1} + e。
    predictability 矩陣 = C^{-1/2} A C A' C^{-1/2}，取最小特徵值。
    """
    X = (df - df.mean(0)).values
    Y, L = X[1:], X[:-1]
    A = np.linalg.lstsq(L, Y, rcond=None)[0].T      # (n,n)
    C = np.cov(X, rowvar=False)
    Ci = np.linalg.inv(sqrtm(C))
    M = Ci @ A @ C @ A.T @ Ci
    ev, vec = np.linalg.eig(M)
    asc = np.argsort(np.real(ev))
    return float(np.real(ev[asc][0])), np.real(Ci @ vec[:, asc])[:, 0]


OBJ = {"portmanteau": obj_portmanteau, "crossing": obj_crossing,
       "btcd": obj_btcd}


# ── 兩種稀疏化 ────────────────────────────────────────────────────
def truncate(tr, k, kind, retrain):
    """全解 -> 取 |w| 前 k 檔（對數價空間裡 |w| 就是名目貢獻）。"""
    _, w = OBJ[kind](tr)
    idx = np.argsort(-np.abs(w))[:k]
    cols = [tr.columns[i] for i in idx]
    if retrain:
        _, w2 = OBJ[kind](tr[cols])
        return cols, w2
    return cols, w[idx]


def greedy(tr, k, kind):
    """**前兩檔暴搜**，之後每次加一檔。

    不從「單一最均值回歸的資產」長出來 —— 那檔通常最不流動，看到的
    均值回歸很可能是買賣價跳動（他的原話，見檔頭）。
    """
    cols = list(tr.columns)
    best, pair = np.inf, None
    for a, b in itertools.combinations(cols, 2):
        try:
            v, _ = OBJ[kind](tr[[a, b]])
        except Exception:                               # noqa: BLE001
            continue
        if v < best:
            best, pair = v, [a, b]
    sel = list(pair)
    while len(sel) < k:
        bv, bc = np.inf, None
        for c in cols:
            if c in sel:
                continue
            try:
                v, _ = OBJ[kind](tr[sel + [c]])
            except Exception:                           # noqa: BLE001
                continue
            if v < bv:
                bv, bc = v, c
        if bc is None:
            break
        sel.append(bc)
    _, w = OBJ[kind](tr[sel])
    return sel, w


def score(te, cols, w, lags=LAGS):
    w = np.asarray(w, dtype="float64")
    w = w / (np.abs(w).sum() + 1e-15)
    s = te[cols].values @ w
    return portmanteau_stat(s, lags), halflife(s), s


def resample_log(lp, hours):
    """對數價降頻：每 `hours` 根取最後一根（跳動關 K8 用）。"""
    return lp.iloc[::hours]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=sorted(GROUPS))
    ap.add_argument("--k", type=int, nargs="*", default=[3, 4, 6, 8])
    ap.add_argument("--train", type=int, default=24 * 90)
    ap.add_argument("--test", type=int, default=24 * 30)
    ap.add_argument("--objs", nargs="*", default=["portmanteau", "crossing", "btcd"])
    a = ap.parse_args()

    px = load()
    g = GROUPS[a.group]
    if g:
        px = px[[c for c in px.columns if c.replace("USDT", "") in g]]
    lp = np.log(px)
    print("分組 %s｜%d 標的｜%s 根" % (a.group, lp.shape[1], format(len(lp), ",")))
    if lp.shape[1] < max(a.k) + 1:
        print("標的數 %d 不夠做 k=%s" % (lp.shape[1], a.k))

    rng = np.random.default_rng(0)
    rows = []
    step = a.test
    i = a.train
    while i + a.test <= len(lp):
        tr, te = lp.iloc[i - a.train:i], lp.iloc[i:i + a.test]
        for kind in a.objs:
            for k in a.k:
                if k >= lp.shape[1]:
                    continue
                try:
                    c1, w1 = truncate(tr, k, kind, retrain=False)
                    c2, w2 = truncate(tr, k, kind, retrain=True)
                    c3, w3 = greedy(tr, k, kind)
                except Exception as e:                  # noqa: BLE001
                    print("  窗 %d %s k=%d 失敗：%s" % (i, kind, k, str(e)[:40]))
                    continue
                p1, h1, _ = score(te, c1, w1)
                p2, h2, _ = score(te, c2, w2)
                p3, h3, s3 = score(te, c3, w3)
                # K7 隨機：隨機挑 k 檔 + 隨機權重
                rnd = []
                for _ in range(200):
                    cs = list(rng.choice(lp.columns, size=k, replace=False))
                    ww = rng.normal(size=k)
                    rnd.append(score(te, cs, ww)[0])
                # K8 跳動關：同一組權重在 4h / 1d 上重算
                p3_4h = portmanteau_stat(
                    resample_log(te[c3], 4).values @ (w3 / np.abs(w3).sum()), LAGS)
                # 1d 不做：測試窗 30 天降到日線只有 30 點，而 portmanteau
                # 需要 lags*5 = 50 點 -> 必然 nan。用 8h（90 點）代替，
                # 跳動效應一樣隨 horizon 遞減，而且量得到。
                p3_8h = portmanteau_stat(
                    resample_log(te[c3], 8).values @ (w3 / np.abs(w3).sum()), LAGS)
                rows.append(dict(i=int(i), obj=kind, k=k,
                                 trunc=p1, trunc_retrain=p2, greedy=p3,
                                 rnd_med=float(np.nanmedian(rnd)),
                                 rnd_p05=float(np.nanpercentile(rnd, 5)),
                                 hl=h3, greedy_4h=p3_4h, greedy_8h=p3_8h,
                                 cols=",".join(c3)))
        i += step

    r = pd.DataFrame(rows)
    if r.empty:
        print("沒有結果"); return
    pd.set_option("display.width", 220)
    print("\n" + "=" * 104)
    print("樣本外 portmanteau（越小越均值回歸）｜每格是 %d 個窗的中位"
          % r.i.nunique())
    print("=" * 104)
    h = ("%-12s %3s %9s %9s %9s %9s %9s   %6s %6s %6s"
         % ("目標", "k", "截斷", "截斷重訓", "貪婪", "隨機中位", "隨機p05",
            "K5", "K6", "K7"))
    print(h); print("-" * len(h))
    summ = []
    for (kind, k), g2 in r.groupby(["obj", "k"]):
        n = len(g2)
        k5 = int((g2.trunc < g2.trunc_retrain).sum())
        k6 = int((g2.greedy < g2.trunc).sum())
        k7 = int((g2.greedy < g2.rnd_p05).sum())
        print("%-12s %3d %9.4f %9.4f %9.4f %9.4f %9.4f   %s %s %s"
              % (kind, k, g2.trunc.median(), g2.trunc_retrain.median(),
                 g2.greedy.median(), g2.rnd_med.median(), g2.rnd_p05.median(),
                 "%2d/%2d" % (k5, n), "%2d/%2d" % (k6, n), "%2d/%2d" % (k7, n)))
        summ.append(dict(obj=kind, k=k, n=n, k5=k5, k6=k6, k7=k7,
                         trunc=float(g2.trunc.median()),
                         greedy=float(g2.greedy.median()),
                         rnd_med=float(g2.rnd_med.median())))
    print("\nK5 截斷不重訓 < 截斷重訓（他：less is more）")
    print("K6 貪婪 < 截斷（他：greedy 遠優於 truncation）")
    print("K7 貪婪 < 隨機 p05（**贏不過隨機就不是發現**）")

    print("\nK8 跳動關（同一組貪婪權重，換頻率重算；假均值回歸會隨頻率降低而縮小）")
    print("%-12s %3s %9s %9s %9s" % ("目標", "k", "1h", "4h", "8h"))
    for (kind, k), g2 in r.groupby(["obj", "k"]):
        print("%-12s %3d %9.4f %9.4f %9.4f"
              % (kind, k, g2.greedy.median(), g2.greedy_4h.median(),
                 g2.greedy_8h.median()))
    print("  （1h 明顯低而 4h/1d 塌回去 = 跳動造的；三個都低 = 真的）")

    fin = r.hl[np.isfinite(r.hl)]
    print("\n貪婪組合的半衰期中位 %.1f 小時（%d/%d 有限）"
          % (fin.median() if len(fin) else float("nan"), len(fin), len(r)))
    top = r[r.obj == "portmanteau"].cols.value_counts().head(5)
    print("\n最常被貪婪選中的組合（portmanteau）：")
    for c, n in top.items():
        print("   %2d 次  %s" % (n, c))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"group": a.group, "rows": rows, "summary": summ},
                  fh, ensure_ascii=False, indent=1, default=str)
    print("\n寫出 %s" % OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
