# -*- coding: utf-8 -*-
"""
========================================================================
2026-09-07 **D2：未成交不是判決的原因，判決不變。**
========================================================================
未成交 868 筆（12.32%）的最近距離分布：

    = 0 tick（碰到了）        0    0.00%   <- D1 自曝檢查 PASS
    (0, 1]                   7    0.81%
    (1, 2]                  10    1.15%
    (2, 5]                  32    3.69%
    (5, 10]                 57    6.57%
    **> 10 tick（真的沒回來） 762   87.79%**
    中位 **108 tick**、p75 527、p90 4,216

**87.8% 的未成交離掛單價超過 10 tick。不是「差一點」，是價格根本沒回來。**

放寬敏感度（k = 最近距離 ≤ k tick 即成交）：

    k= 0  成交 87.68%  E[R|成交] −0.0341  每事件 **−0.0299**（現行）
    k= 1  成交 87.78%            −0.0337          −0.0295
    k= 2  成交 87.92%            −0.0328          **−0.0289**
    k= 5  成交 88.37%            −0.0310          −0.0274
    k=10  成交 89.18%            −0.0258          −0.0231

從 0 放寬到 10 tick，成交率只動 1.5 個百分點。**離 +0.01 還很遠。**

使用者提出的「隊伍論不適用小資金」是對的，我先前那段機制敘述確實是推的
不是量的，已收回。但它不影響判決，而且現在有三層獨立理由：
  (1) 成交規則本來就是「碰到即成交」，沒有隊伍假設（D1 證明）
  (2) 未成交中位距離 108 tick —— 隊伍前面有多少量都不改變「價格沒回來」
  (3) 決定判決的 `honest_fill` 乾淨臂是**市價成交、成交率 100%**，不用成交率

**要標記的不一致**：本檔的 E[R|成交] −0.0341 與 `exec_ladder` 的 −0.0504
不同，因為**成交時點不同**——本檔填在「最接近的那一分鐘」（argmin），
`exec_ladder` 填在「第一次碰到」。兩個是不同的估計量。兩者都為負且都離
+0.01 很遠，結論不受影響，但這個差異不得消失在「反正都是負的」裡面。
沒成交的那些，價格離掛單價到底多遠？——57.9% 是假設造成的還是市場造成的

使用者 2026-09-07：
    「如果成交規則是『價格穿越 k tick 才算成交』，那 57.9% 就不是隊伍造成的，
     是你自己設的保守假設造成的……先跑未成交事件的距離分布。那個數字決定
     57.9% 是真的還是假設造成的。」

===========================================================================
先更正三個前提（不更正的話這支會回答錯的問題）
===========================================================================
1. **57.9% 不是成交率。** 它是「掃單 bar 收盤時市場已在價位另一側」的比例
   （`honest_fill.py` 的 `inside`），一個**狀態分類**。
   真正的掛單成交率是 **87.68%**（δ=0，`resting_limit.py`）。

2. **成交規則本來就是「碰到就算成交」，沒有要求穿越任何 tick。**
   `exec_ladder.py`：`hit = np.flatnonzero(mhi[i0:i1] >= lvl)`。
   使用者建議的主假設就是已經在用的那條。
   （穿越 k tick 的版本是 `resting_fill.py`，**已整組作廢**——作廢理由是
   分層變數 Spearman +0.6884 與結果機械相關，不是保守假設。）

3. **決定判決的那條路徑沒有未成交。** `honest_fill.py` 的乾淨臂是
   「B 情境用真實可成交價**市價**成交」，成交率 100%，−0.0483 完全不依賴
   任何成交率假設；它依賴的是**進場價**（市場中位在價位之外 42.6 bps）。

   使用者關於「隊伍論不適用於小資金」的更正是對的——那段機制敘述是推的
   不是量的，已收回。但它不影響判決，因為判決那條路徑沒用到隊伍假設。

===========================================================================
本檔要回答的（使用者提的檢查，仍然該做）
===========================================================================
掛單路徑裡**沒成交**的那 12.32%，價格最近曾經離掛單價多遠（以 tick 計）：

    = 0 tick   碰到了 -> 依現行規則本來就算成交，這一格應為 **0**
                （若不為 0，代表我的實作有 bug，本檔當場自曝）
    1–2 tick   差一點 -> 以小資金的量，很可能實際會成交 -> 敏感度要測
    ≥ 3 tick   真的沒回來 -> 市場造成，不可修

並直接做敏感度：把成交條件放寬成「最近距離 ≤ k tick 即成交」，
k ∈ {0（現行）, 1, 2, 5, 10}，各自重算**每事件值**。

判準（跑之前寫死）
    D1 自曝檢查：k=0 的分布裡「= 0 tick」必須是 0 筆。不是 0 -> 實作有 bug，
       以下不解讀。
    D2 若放寬到 k=2 之後，每事件值仍 < +0.01 R
       -> **57.9%／未成交不是判決的原因**，判決不變。
    D3 若 k=2 之後每事件值 ≥ +0.01 R
       -> **判決下得太早**，整條線要用新的成交規則重算。
    D4 全格報告 k 的每一階，不挑對自己有利的那一階。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402

BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
HOUR_MS = 3_600_000
KS = [0, 1, 2, 5, 10]
RNG = np.random.default_rng(20260907)


def exit_from(h, lo, cl, n, f_bar, d, entry, risk, A):
    stop = entry - d * risk
    for q in range(f_bar + 1, min(f_bar + sc.HOLD + 1, n)):
        if (d == 1 and lo[q] <= stop) or (d == -1 and h[q] >= stop):
            return -1.0 - sc.SLIP / sc.DIS
    exb = min(f_bar + sc.HOLD, n - 1)
    return d * (cl[exb] - d * sc.SLIP * A - entry) / risk


def main():
    rows = []
    for sym in CORE9:
        b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        h = [x[sc.H] for x in b1]
        lo = [x[sc.L] for x in b1]
        cl = [x[sc.C] for x in b1]
        n = len(b1)
        hts = np.array([int(x[0]) for x in b1], np.int64) * 1000

        m = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "high", "low", "tick_size"])
        mts = m["ts"].to_numpy(np.int64)
        mhi = np.nan_to_num(m["high"].to_numpy(float), nan=-np.inf)
        mlo = np.nan_to_num(m["low"].to_numpy(float), nan=np.inf)
        tick = m["tick_size"].to_numpy(float)
        nm = len(mts)

        for e in sc.backtest_symbol(b1, detail=True):
            j, lvl, A, d, risk = (e["j"], e["level"], e["atr"], e["d"], e["risk"])
            t0 = int(b1[j][0]) * 1000 + HOUR_MS
            t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
            i0 = int(np.searchsorted(mts, t0, side="left"))
            i1 = int(np.searchsorted(mts, t1, side="left"))
            if i0 >= nm or i1 <= i0:
                continue
            tk = float(np.nanmedian(tick[i0:i1]))
            if not np.isfinite(tk) or tk <= 0:
                continue

            # 現行規則：碰到即成交（不要求穿越）
            if d == -1:
                reach = mhi[i0:i1]                 # 做空掛 lvl，價格要漲上來
                gapv = lvl - reach                 # >0 表示還沒到
            else:
                reach = mlo[i0:i1]                 # 做多掛 lvl，價格要跌下來
                gapv = reach - lvl
            best = float(np.nanmin(gapv))          # 最近曾經差多少（價格單位）
            best_tick = best / tk                  # 換成 tick

            row = dict(sym=sym, best_tick=float(best_tick), tick=tk,
                       R_frozen=float(e["R"]),
                       day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                        tz="UTC").strftime("%Y-%m-%d"))
            # 各 k 之下的成交與 R
            for k in KS:
                ok = best_tick <= k
                if not ok:
                    row[f"f{k}"] = 0
                    row[f"r{k}"] = np.nan
                    continue
                idx = int(np.nanargmin(gapv))
                kk = i0 + idx
                fb = int(np.searchsorted(hts, int(mts[kk]), side="right")) - 1
                if fb < 0 or fb + 1 >= n:
                    row[f"f{k}"] = 0
                    row[f"r{k}"] = np.nan
                    continue
                row[f"f{k}"] = 1
                row[f"r{k}"] = exit_from(h, lo, cl, n, fb, d,
                                         lvl + d * sc.SLIP * A, risk, A)
            rows.append(row)

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "nofill_distance.parquet", index=False)

    def day_ci(x, days, b=2000):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        x, days = x[ok], np.asarray(days)[ok]
        if len(x) < 30:
            return (float("nan"),) * 3
        uq, inv = np.unique(days, return_inverse=True)
        ix = [np.where(inv == kk)[0] for kk in range(len(uq))]
        reps = np.empty(b)
        for i in range(b):
            p = RNG.integers(0, len(uq), len(uq))
            reps[i] = x[np.concatenate([ix[kk] for kk in p])].mean()
        return (float(x.mean()), float(np.percentile(reps, 2.5)),
                float(np.percentile(reps, 97.5)))

    days = d.day.to_numpy()
    nf = d[d.f0 == 0]
    print("=== 沒成交的那些，價格最近曾經離掛單價多遠 ===")
    print(f"全部 {len(d):,} 筆   現行規則（碰到即成交）成交 "
          f"{int(d.f0.sum()):,} = {d.f0.mean()*100:.2f}%   "
          f"未成交 {len(nf):,} = {(1-d.f0.mean())*100:.2f}%")
    print()
    bt = nf.best_tick.to_numpy(float)
    bt = bt[np.isfinite(bt)]
    buckets = [("= 0 tick（碰到了）", (bt <= 0).sum()),
               ("(0, 1] tick", ((bt > 0) & (bt <= 1)).sum()),
               ("(1, 2] tick", ((bt > 1) & (bt <= 2)).sum()),
               ("(2, 5] tick", ((bt > 2) & (bt <= 5)).sum()),
               ("(5, 10] tick", ((bt > 5) & (bt <= 10)).sum()),
               ("> 10 tick（真的沒回來）", (bt > 10).sum())]
    print(f"{'最近距離':24s} {'n':>7s} {'佔未成交':>9s}")
    dist = {}
    for lab, c in buckets:
        dist[lab] = int(c)
        print(f"{lab:24s} {int(c):7,d} {c/max(len(bt),1)*100:8.2f}%")
    print()
    print(f"  中位 {np.median(bt):,.1f} tick   p25 {np.percentile(bt,25):,.1f}   "
          f"p75 {np.percentile(bt,75):,.1f}   p90 {np.percentile(bt,90):,.1f}")

    d1 = int((bt <= 0).sum()) == 0
    print()
    print(f"D1 自曝檢查：未成交裡「= 0 tick」應為 0 筆，實測 {int((bt<=0).sum())}"
          f"  -> {'PASS' if d1 else '**FAIL — 實作有 bug，以下不解讀**'}")

    print()
    print("=== D4 放寬成交條件的敏感度（全格，不挑）===")
    print()
    print(f"{'k (tick)':10s} {'成交率':>8s} {'E[R|成交]':>11s} "
          f"{'日聚類 CI95':>24s} {'每事件':>9s}")
    res = {"n": int(len(d)), "fill_rate_k0": float(d.f0.mean()),
           "dist": dist, "median_tick": float(np.median(bt)), "D1": bool(d1),
           "k": {}}
    for k in KS:
        fr = float(d[f"f{k}"].mean())
        mm, ll, hh = day_ci(d[f"r{k}"].to_numpy(), days)
        pe = mm * fr
        res["k"][str(k)] = dict(fill=fr, e_r=mm, ci=[ll, hh], per_event=pe)
        print(f"{k:10d} {fr*100:7.2f}% {mm:+11.4f}  [{ll:+.4f},{hh:+.4f}] "
              f"{pe:+9.4f}")

    pe2 = res["k"]["2"]["per_event"]
    v = ("**判決下得太早 —— 整條線要用新的成交規則重算**" if pe2 >= 0.01
         else "**未成交不是判決的原因，判決不變**")
    print()
    print(f"D2/D3 放寬到 k=2 之後每事件 {pe2:+.4f} R"
          f"（門檻 +0.01）-> {v}")
    res["verdict"] = v

    (OUT / "nofill_distance.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "nofill_distance.json")
    print()
    print("提醒：決定判決的那條路徑（honest_fill 乾淨臂）**成交率 100%**，")
    print("      本檔測的是掛單路徑。兩者都為負才是判決的依據。")


if __name__ == "__main__":
    main()
