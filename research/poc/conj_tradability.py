# -*- coding: utf-8 -*-
"""交會事件的可交易性換算 —— 0.35 ATR 有多少進得了口袋。

為什麼要單獨做這件事
    事件研究量的是**收盤到收盤、零成本、無並發限制**的價格移動。
    `TRIAGE.md` 的 +0.3485 ATR 是那種數字。把它當成「每筆賺 0.35 ATR」
    是 mistake.md 2026-09-03 那個 $30/天 的同一種錯：
    **頻率 x 幅度 x 規模的估計，先問每一項數的是什麼單位。**

    三關，全部用這條線既有的凍結成本模型，不自己另編：

    1. 成本    `sweep_forward.py` 的分類成本（2026-07-28 凍結，未調參）
                 情境 A 目標執行：進場 stop-market taker 5 + slip 2 = 7 bps
                                  時間出場 worked limit maker 2 + miss 1 = 3 bps
                 情境 B 全 taker ：進場 7 / 時間出場 6
                 換算：cost_ATR = sum(leg_bps)/1e4 * price / ATR
                 **逐幣真實 bps，不用統一 ATR 單位**——統一單位會奉承低波動
                 幣，把 BTC/BNB 的相對成本低估一半以上。
    2. 容量    交會事件約 3.3 筆/天（九幣合計）。`max_position_count=1` 下
                 60 分鐘持有期會互相擠掉多少？**用真實毫秒時戳做跨幣單槽
                 貪婪模擬**，不是同日總量近似。

                 第一版就是用同日總量算的，報出「100% 都開得到」——那個
                 檢查對真正的問題（同一個 60 分鐘窗內的重疊）**免疫**，
                 因為級聯會同時打九個幣而日總量看不到這件事。
                 這是「守衛存在但沒有測量能力」，同族見 mistake.md
                 2026-08-11 的 G4 離散度比值。已修，此註解留著。
    3. 淨值    效應 - 成本，逐幣報告，並問「幾個幣扣完成本後還是正的」。

**這是刻畫不是判決。** 輸入的 +0.3485 本身是 in-sample，前瞻時鐘
（`conj_clock.py`，凍結日 2026-09-07）還沒有樣本。本檔回答的是
「**如果**前瞻確認了，它值多少」——不是「它已經值多少」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
HOLD_MIN = 60
SCEN = {"A 目標執行": 7 + 3, "B 全 taker": 7 + 6}


def main():
    src = OUT / "conj_causal.parquet"
    if not src.exists():
        sys.exit("先跑 conj_causal.py（因果門檻版的配對輸出）")
    d = pd.read_parquet(src)
    if "ts_ms" not in d.columns:
        sys.exit("配對輸出沒有 ts_ms —— 重跑 conj_causal.py")
    conj = d[d.lane == "掃單+強制流"].copy()
    conj["x"] = conj["e60"] - conj["c60"]
    conj = conj[np.isfinite(conj.x)]

    # 逐幣 ATR%（中位）—— 成本換算的分母
    atrpct = {}
    for sym in ec.CORE9:
        b = pd.read_parquet(BARS / f"{sym}.parquet", columns=["close", "atr_h14"])
        a = b["atr_h14"].to_numpy(float)
        c = b["close"].to_numpy(float)
        g = np.isfinite(a) & np.isfinite(c) & (c > 0) & (a > 0)
        atrpct[sym] = float(np.median(a[g] / c[g]))

    print("=== 1. 成本（逐幣真實 bps，凍結成本模型）===")
    print()
    print(f"{'幣':6s} {'ATR%':>8s} {'效應(ATR)':>10s} {'效應(bps)':>10s} "
          + "".join(f"{'淨 ' + k:>16s}" for k in SCEN))
    rows = {}
    for sym in ec.CORE9:
        g = conj[conj.sym == sym]
        if len(g) < 30:
            continue
        eff = float(g.x.mean())
        ap = atrpct[sym]
        cell = {k: eff - bps / 1e4 / ap for k, bps in SCEN.items()}
        rows[sym] = dict(n=int(len(g)), atr_pct=ap, eff_atr=eff,
                         eff_bps=eff * ap * 1e4, **cell)
        print(f"{sym:6s} {ap*100:7.3f}% {eff:+10.4f} {eff*ap*1e4:+10.1f} "
              + "".join(f"{cell[k]:+16.4f}" for k in SCEN))

    pooled = float(conj.x.mean())
    w = conj.groupby("sym").size()
    ap_w = float(sum(w[s] * atrpct[s] for s in w.index) / w.sum())
    print(f"\n{'池化':6s} {ap_w*100:7.3f}% {pooled:+10.4f} "
          f"{pooled * ap_w * 1e4:+10.1f} "
          + "".join(f"{pooled - SCEN[k] / 1e4 / ap_w:+16.4f}" for k in SCEN))
    for k, bps in SCEN.items():
        pos = sum(1 for s in rows if rows[s][k] > 0)
        print(f"  情境 {k}：成本 {bps} bps，扣完後逐幣為正 {pos}/{len(rows)}")

    print()
    print("=== 2. 容量（max_position_count=1 的真實擠出）===")
    print()
    print("  跨幣單槽貪婪模擬：按毫秒時戳排序，槽空著才開，開了就鎖 60 分鐘。")
    print("  這是真的重疊計算，不是同日總量近似（後者對級聯同時打九幣免疫）。")
    print()
    g = conj.sort_values("ts_ms").reset_index(drop=True)
    hold_ms = HOLD_MIN * 60_000
    free_at = -1
    take = np.zeros(len(g), dtype=bool)
    for i, tms in enumerate(g.ts_ms.to_numpy(np.int64)):
        if tms >= free_at:
            take[i] = True
            free_at = tms + hold_ms
    n_total, n_take = int(len(g)), int(take.sum())
    n_days = int(g.day.nunique())
    eff_t = float(g.x[take].mean())
    eff_d = float(g.x[~take].mean()) if (~take).sum() else float("nan")
    burst = int(g.groupby(g.ts_ms // hold_ms).size().max())
    print(f"  事件 {n_total:,} 筆 / {n_days:,} 個 UTC 日 = "
          f"{n_total / n_days:.2f} 筆/天（九幣合計）")
    print(f"  單槽實際開得到 {n_take:,} / {n_total:,} = "
          f"{n_take / n_total * 100:.1f}%   被擠掉 {n_total - n_take:,} 筆")
    print(f"  被取的效應 {eff_t:+.4f}   被擠掉的效應 {eff_d:+.4f}   "
          f"差 {eff_t - eff_d:+.4f}")
    print(f"  同一個 60 分鐘桶內最多 {burst} 筆同時發生")
    print()
    print("  逐槽位數（放寬併發上限會拿回多少）：")
    for slots in (1, 2, 3, 5, 9):
        free = [-1] * slots
        cnt = 0
        for tms in g.ts_ms.to_numpy(np.int64):
            k = int(np.argmin(free))
            if tms >= free[k]:
                free[k] = tms + hold_ms
                cnt += 1
        print(f"    {slots} 槽 -> {cnt:,} / {n_total:,} = "
              f"{cnt / n_total * 100:.1f}%")

    res = dict(pooled_eff_atr=pooled, atr_pct_weighted=ap_w,
               pooled_eff_bps=pooled * ap_w * 1e4,
               scenarios={k: dict(bps=v,
                                  net_atr=pooled - v / 1e4 / ap_w,
                                  coins_pos=sum(1 for s in rows if rows[s][k] > 0))
                          for k, v in SCEN.items()},
               per_coin=rows, n=n_total, days=n_days,
               per_day=n_total / n_days, n_taken=n_take,
               slot_take_rate=n_take / n_total,
               eff_taken=eff_t, eff_dropped=eff_d, max_burst=burst)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_tradability.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_tradability.json")
    print()
    print("**這是刻畫不是判決**：輸入的效應是 in-sample，前瞻時鐘尚無樣本。")


if __name__ == "__main__":
    main()
