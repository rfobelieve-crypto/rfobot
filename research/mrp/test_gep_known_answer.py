# -*- coding: utf-8 -*-
"""GEP 的已知答案對照 —— **在用它推翻任何事之前先跑這支**。

mistake.md 2026-07-29：「新寫的診斷在用來推翻任何結論之前，先在一個答案
已知的資料上跑一次」「**自己剛寫的工具比別人的舊工具更危險**」。

造一份藏了**已知答案**的資料：
    5 條獨立隨機漫步 x1..x5（沒有均值回歸）
    再讓 x5 = 0.6*x1 + 0.4*x2 + OU(半衰期 h)
    -> 於是 w = (0.6, 0.4, 0, 0, −1) 這個組合**依建構是 OU**，
       而任何其他方向都是隨機漫步。

如果 `portmanteau_gep` 是對的，它的**最小特徵值**那個向量應該與
(0.6, 0.4, 0, 0, −1) 幾乎共線（|cos| 接近 1）。

反向對照同樣重要：**最大特徵值**那個向量不應該共線 —— 否則代表這支在
「最小」與「最大」之間分不出來，而那正是我們在真實資料上看到的症狀
（K2 只有 12/28）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrp_portmanteau import halflife, portmanteau_gep, portmanteau_stat


def synth(n=4000, hl=24.0, seed=0, noise=0.15):
    """5 條序列，答案是 w=(0.6, 0.4, 0, 0, −1)。"""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1.0, size=(n, 4)), axis=0)
    phi = 2.0 ** (-1.0 / hl)                 # OU 的 AR(1) 係數
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + rng.normal(0, noise)
    x5 = 0.6 * x[:, 0] + 0.4 * x[:, 1] - ou  # 使 0.6x1+0.4x2−x5 = ou
    d = pd.DataFrame(np.column_stack([x, x5]),
                     columns=["x1", "x2", "x3", "x4", "x5"])
    truth = np.array([0.6, 0.4, 0.0, 0.0, -1.0])
    return d, truth / np.abs(truth).sum()


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-15)
    b = b / (np.linalg.norm(b) + 1e-15)
    return abs(float(a @ b))


def main():
    print("=" * 76)
    print("GEP 已知答案對照：藏一個 OU 組合 w=(0.6,0.4,0,0,−1)，看它找不找得到")
    print("=" * 76)
    ok_all = True
    print("%-8s %-7s %10s %10s %12s %12s" %
          ("半衰期", "lags", "|cos|最小", "|cos|最大", "pm(找到的)", "pm(真答案)"))
    print("-" * 68)
    for hl in (6.0, 24.0, 96.0):
        for lags in (5, 10, 20):
            d, truth = synth(hl=hl)
            ev, W = portmanteau_gep(d, lags)
            w_mr = W[:, 0] / np.abs(W[:, 0]).sum()
            w_mo = W[:, -1] / np.abs(W[:, -1]).sum()
            c_mr, c_mo = cosine(w_mr, truth), cosine(w_mo, truth)
            pm_found = portmanteau_stat(d.values @ w_mr, lags)
            pm_true = portmanteau_stat(d.values @ truth, lags)
            # **「找不到」要分成兩種**：實作壞了 vs 解析度不夠。
            # lags 個 lag 的自相關窗，對半衰期遠大於它的過程本來就分不出
            # 隨機漫步 —— 那不是 bug，是這個統計量的量測範圍。
            good = c_mr > 0.9 and c_mr > c_mo
            inrange = hl <= 2.5 * lags
            if inrange:
                ok_all &= good
            print("%-8.0f %-7d %10.3f %10.3f %12.4f %12.4f  %s"
                  % (hl, lags, c_mr, c_mo, pm_found, pm_true,
                     ("OK" if good else "**不合格**") if inrange
                     else ("(超出解析度，預期找不到)" if not good
                           else "(超出解析度但仍找到)")))
    d, truth = synth(hl=24.0)
    print("\n真答案那條序列的半衰期：%.1f 小時（造的時候設 24）"
          % halflife(d.values @ truth))
    print("\n結論：%s" % ("**GEP 實作是對的** —— 它在答案已知的資料上找得到，"
                          "所以真實資料上的負面結果是資料的事不是程式的事。"
                          if ok_all else
                          "**實作有問題** —— 在答案已知的資料上都找不到，"
                          "真實資料的結果一律作廢。"))
    print("\n**而真正要帶走的是解析度那一條**：半衰期超過約 2.5 x lags 就量不到。")
    print("   他用**日線** lags=10 -> 自相關窗 = 10 **天**。")
    print("   我們用**小時線**照抄 lags=10 -> 窗只有 10 **小時**，")
    print("   而真實資料量到的半衰期是 **88 小時**，正好落在失效區。")
    print("   **抄參數要連它的單位一起抄。**")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
