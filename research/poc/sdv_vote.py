# -*- coding: utf-8 -*-
"""掃單之後：三個因子各自的方向由**前半資料**決定，再投票（2026-09-10 預註冊）

接續 `sdv_after_sweep.py`。那一輪照使用者的假設把方向寫成
「OI 降 -> 順勢、OI 升 -> 逆勢」，三個 K 全部 FAIL，而分桶顯示**方向是反的**
（OI 升最多那一桶順勢 +0.197，最低那桶 +0.016）。

**所以這一輪絕對不能由我指定方向** —— 我已經看過分桶了，手動把它翻過來
就是 mistake.md 2026-09-09 判掉的那件事（事後找到的維度，通過再多一致性
檢查都不算數；唯一算數的是把挑選程序本身放進樣本外）。

===========================================================================
乾淨的流程：方向是「學」來的，不是「挑」來的
===========================================================================
    前半資料   對每個因子 f：把樣本用前半中位數切兩半，看哪一邊順勢比較賺
               -> 得到 sign_f ∈ {+1, −1}（+1 = 高值時順勢好）
               中位數與 sign 全部只用前半算，後半一個數字都不看
    後半資料   套用學到的 sign_f 決定方向；後半是**真樣本外**

這樣「方向反了」這件事如果是真的，前半自己就會學到；如果只是我事後看到的
雜訊，前半會學到別的方向，後半就會打臉 —— 兩種結局都誠實。

===========================================================================
臂（全格報告，不挑）
===========================================================================
    S1/S2/S3   單因子：各自用學到的方向
    VOTE       三個因子各投一票（多數決；平手時順勢）
    B          對照：全部順勢（不用任何因子）
    C          對照：全部逆勢

判準（跑之前凍結）
    P1  VOTE 的樣本外每筆淨值 > max(B, C, S1, S2, S3)
        —— 多因子要贏過所有對照**和**每一個單因子，否則它只是把最好的
           那個因子稀釋掉
    P2  VOTE 的日聚類 bootstrap CI 下緣 > 0
    P3  VOTE 逐幣 ≥ 6/9
    P1∧P2∧P3 -> 採用候選；否則不過

**使用者已明說「單因子 fail 正常」**，所以單因子不設過閘門檻，只報數字；
判準只加在 VOTE 上。但單因子的方向若在前半／後半之間翻號，要如實標出來
——那代表那個因子不穩定，投票裡帶著它是負擔不是資產。

自曝檢查
    A1  前半學到的 sign 與後半「事後最佳」的 sign 是否一致 —— 不一致的
        因子要標明，那正是「我事後看到的方向」與「真的能事先學到的方向」
        的差別。
    A2  B 臂（全部順勢）必須重現 sdv_after_sweep 的數字（同母體同進出場）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import conj_backtest as cb  # noqa: E402
import sdv_after_sweep as sas  # noqa: E402

OUT = HERE / "data" / "results"
KS = (5, 10, 15)
FACTORS = (("f1", "OI 變化"), ("f2", "散戶多空比變化"), ("f3", "平均單筆大小"))
SEED = 20260910


def collect(k):
    """所有掃單的逐筆記錄 + 每筆在順勢／逆勢下的報酬（方向先不決定）。"""
    recs = []
    for s in cb.CORE9:
        rows, op, hi, lo, cl, n = sas.build(s, k)
        for r in rows:
            d_s = r["d_sweep"]
            r["R_with"] = sas.run_trade(op, hi, lo, cl, r["j0"], d_s, r["A"])
            r["R_against"] = sas.run_trade(op, hi, lo, cl, r["j0"], -d_s, r["A"])
            recs.append(r)
    return pd.DataFrame(recs)


def learn(first, col):
    """只用前半：中位數切兩半，哪一邊順勢比較賺 -> sign。"""
    med = float(first[col].median())
    hi = first[first[col] > med]["R_with"].mean()
    lo = first[first[col] <= med]["R_with"].mean()
    return med, (1 if hi >= lo else -1), float(hi), float(lo)


def apply_sign(d, col, med, sign):
    """+1 表示「高於中位數就順勢」。回傳每筆的方向係數 ±1。"""
    high = (d[col] > med).to_numpy()
    return np.where(high, sign, -sign)


def pick(d, dirs):
    """dirs = ±1 陣列；回傳每筆實際拿到的報酬。"""
    return np.where(dirs > 0, d["R_with"].to_numpy(), d["R_against"].to_numpy())


def ci_lo(days, vals):
    rng = np.random.default_rng(SEED)
    by = {}
    for dd, v in zip(days, vals):
        by.setdefault(int(dd), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return np.nan
    arr = [np.array(by[x]) for x in ks]
    idx = rng.integers(0, len(ks), size=(2000, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(2000)])
    return float(np.percentile(o, 2.5))


def main():
    allout = {}
    for k in KS:
        d = collect(k)
        mid = float(d.day.median())
        first, second = d[d.day < mid], d[d.day >= mid].copy()

        print("=" * 76)
        print(f"K = {k} 分鐘   全部 {len(d):,} 筆（前半 {len(first):,} / 後半 {len(second):,}）")
        print()
        print("只用前半學方向：")
        learned = {}
        for col, nm in FACTORS:
            med, sign, hi, lo = learn(first, col)
            # A1：後半事後最佳的 sign（只印出來對照，不參與任何決策）
            h2 = second[second[col] > med]["R_with"].mean()
            l2 = second[second[col] <= med]["R_with"].mean()
            post = 1 if h2 >= l2 else -1
            flag = "" if post == sign else "   ← 後半翻號（此因子不穩定）"
            learned[col] = (med, sign)
            arrow = "高值→順勢" if sign > 0 else "高值→逆勢"
            print(f"  {nm:14} 前半 高{hi:+.3f} / 低{lo:+.3f} -> {arrow}"
                  f"     後半 高{h2:+.3f} / 低{l2:+.3f}{flag}")

        # ---- 各臂在後半的表現 ----
        arms = {}
        for col, nm in FACTORS:
            med, sign = learned[col]
            arms[nm] = pick(second, apply_sign(second, col, med, sign))
        votes = np.zeros(len(second))
        for col, _ in FACTORS:
            med, sign = learned[col]
            votes += apply_sign(second, col, med, sign)
        arms["VOTE 三因子多數決"] = pick(second, np.where(votes >= 0, 1, -1))
        arms["B 對照：全部順勢"] = second["R_with"].to_numpy()
        arms["C 對照：全部逆勢"] = second["R_against"].to_numpy()

        print()
        print(f"後半（真樣本外） n = {len(second):,}")
        print(f"{'臂':22} {'淨/筆':>9} {'CI下緣':>9} {'幣+':>6}")
        res = {}
        for nm, v in arms.items():
            per = pd.DataFrame({"sym": second.sym.values, "v": v}).groupby("sym").v.mean()
            lo = ci_lo(second.day.values, v)
            res[nm] = dict(m=float(np.mean(v)), lo=lo,
                           npos=int((per > 0).sum()), nsym=int(len(per)))
            print(f"{nm:22} {np.mean(v):+9.4f} {lo:+9.4f} {int((per > 0).sum()):3d}/{len(per)}")

        vt = res["VOTE 三因子多數決"]
        others = [res[n]["m"] for n in res if n != "VOTE 三因子多數決"]
        p1, p2, p3 = vt["m"] > max(others), vt["lo"] > 0, vt["npos"] >= 6
        print()
        print(f"  P1 VOTE 贏過所有對照與單因子 {'PASS' if p1 else 'FAIL'} / "
              f"P2 CI下緣>0 {'PASS' if p2 else 'FAIL'}（{vt['lo']:+.4f}）/ "
              f"P3 逐幣≥6/9 {'PASS' if p3 else 'FAIL'}（{vt['npos']}/{vt['nsym']}）"
              f"  -> {'採用候選' if (p1 and p2 and p3) else '不過'}")
        print()
        allout[str(k)] = res

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_vote.json"
    p.write_text(json.dumps(allout, indent=2, default=float), encoding="utf-8")
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
