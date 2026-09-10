# -*- coding: utf-8 -*-
"""那一格是不是運氣：置換檢定（2026-09-10）

`sdv_divergence.py` 測了 10 格（5 因子 × 2 母體），過了 1 格：
所有掃單、分歧變化 5 分、樣本外 +0.0772、SE 0.0399、CI 下緣 +0.0011、逐幣 8/9。

**10 格過 1 格，本來就是純運氣也會發生的事。** 使用者不想再等前瞻，
而回測上還能做的最有價值的一件事就是這個：直接量「隨機情況下，最好的
那一格能有多好」。

做法
    每一輪：把因子值**在幣內隨機重排**（保留每個幣的分布與時間結構，
            只打斷「這個因子值對應到這一筆交易」的關係），然後跑完
            **一模一樣**的流程 —— 前半學方向、後半算報酬 —— 並取
            10 格裡最好的那一格。
    重複 N 輪，看真實的 +0.0772 排在這 N 個「隨機最佳」的哪個位置。

**這自動控制了多重比較**：因為每一輪隨機也享有同樣的 10 次挑選機會。
p 值 = 隨機最佳 >= 真實最佳的比例。

這不是新的探索，是對既有結果的質問，所以不另立判準；照慣例 p < 0.05
才算它撐得住。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNAP = HERE / "data" / "sweep_snapshot.parquet"
N_PERM = 200
SEED = 20260910
WINS = (5, 10, 15)


def prep():
    d = pd.read_parquet(SNAP)
    d["day"] = d.ts // 86_400_000
    d = d.reset_index(drop=True)
    d["div_pos"] = d.pre_ls_top_pos - d.pre_ls_retail
    d["div_acct"] = d.pre_ls_top_acct - d.pre_ls_retail
    for k in WINS:
        d[f"divchg{k}"] = d[f"post{k}_ls_top_pos"] - d[f"post{k}_ls_retail"]
    return d


def one_cell(sub, fac, ycol, mid):
    """一格：前半學方向 -> 後半的每筆淨值。回傳 (m, npos, nsym)。"""
    m_first = sub.day < mid
    f = sub[fac].to_numpy(float)
    ok = np.isfinite(f)
    if ok.sum() < 200:
        return None
    med = np.nanmedian(f[m_first & ok])
    yw = sub[ycol].to_numpy(float)
    ya = sub[ycol.replace("with", "against")].to_numpy(float)
    hi = f > med
    a = m_first & ok
    sign = 1 if np.nanmean(yw[a & hi]) >= np.nanmean(yw[a & ~hi]) else -1
    b = (~m_first) & ok
    dirs = np.where(hi[b], sign, -sign)
    v = np.where(dirs > 0, yw[b], ya[b])
    syms = sub["sym"].to_numpy()[b]
    per = pd.DataFrame({"s": syms, "v": v}).groupby("s").v.mean()
    return float(np.mean(v)), int((per > 0).sum()), int(len(per))


CELLS = [("div_pos", "y_with_d0"), ("div_acct", "y_with_d0")] + \
        [(f"divchg{k}", f"y_with_d{k}") for k in WINS]


def best_over_cells(d, shuffled=None):
    """跑全部 10 格（2 母體 × 5 因子），回傳最好那格的每筆淨值。"""
    best, where = -9e9, None
    for mname, sub in (("所有掃單", d), ("SDV", d[d.is_sdv])):
        sub = sub.reset_index(drop=True)
        if shuffled is not None:
            sub = sub.copy()
            for fac, _ in CELLS:
                # 幣內重排：保留每個幣的分布，只打斷與報酬的對應
                v = sub[fac].to_numpy(float).copy()
                for s, idx in sub.groupby("sym").indices.items():
                    v[idx] = shuffled.permutation(v[idx])
                sub[fac] = v
        mid = float(sub.day.median())
        for fac, ycol in CELLS:
            r = one_cell(sub, fac, ycol, mid)
            if r and r[0] > best:
                best, where = r[0], f"{mname}·{fac}"
    return best, where


def main():
    d = prep()
    real, where = best_over_cells(d)
    print(f"真實資料：10 格裡最好的是 {where}，每筆 {real:+.4f}")
    print(f"跑 {N_PERM} 輪置換（幣內重排因子，流程完全相同）…")

    rng = np.random.default_rng(SEED)
    hits, vals = 0, []
    for i in range(N_PERM):
        b, _ = best_over_cells(d, shuffled=rng)
        vals.append(b)
        hits += (b >= real)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_PERM}  隨機最佳的中位 {np.median(vals):+.4f}"
                  f"  p 目前 {(hits + 1) / (i + 2):.3f}")
    vals = np.array(vals)
    p = (hits + 1) / (N_PERM + 1)
    print()
    print("=" * 66)
    print(f"隨機最佳：中位 {np.median(vals):+.4f}   p95 {np.percentile(vals, 95):+.4f}"
          f"   最大 {vals.max():+.4f}")
    print(f"真實最佳：{real:+.4f}")
    print(f"**p = {p:.4f}**（隨機最佳 >= 真實的比例，已含多重比較）")
    print()
    if p < 0.05:
        print("  -> 撐得住：隨機給同樣 10 次挑選機會，也很少做到這麼好")
    else:
        print("  -> **撐不住：純靠 10 次挑選的運氣就能達到這個水準**")


if __name__ == "__main__":
    main()
