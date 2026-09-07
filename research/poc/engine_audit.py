# -*- coding: utf-8 -*-
"""凍結引擎自己有沒有問題 —— 使用者 2026-09-07：「怎麼都沒思考過會不會是你的引擎有問題」

這個質問是對的。我一路把 `sweep_core` 當地基，每次量出 null 就去找機制解釋
（回踩與延續互斥、功效不足），**沒有回頭審引擎**。而且使用者自己的事件總表
E2 早就列了一條待查，我沒跑：

    「層級成立時離當時價格多遠（ATR 計）？若集中在 0.5 ATR 內，層級是
     『剛剛的價格附近』而非『遠處的流動性聚集』，意義要重寫。」

本檔量六件事，全部只用引擎自己的定義，不引入新規則：

    A1 層級確認時，它離當時價格多遠（ATR）
       樞紐 i 要到 i+PIVOT 才確認。用 close[i+PIVOT] 當「當時價格」。
       若中位 < 0.5 ATR -> 層級是「剛剛的價格附近」，語意要重寫。
    A2 掃單發生時，價格離層級多遠（掃單 bar 開盤 vs 層級，ATR）
       這決定「掃單」是不是「價格本來就貼著它」。
    A3 穿透深度（pierce）的分布
       引擎只要求 h[j] > h[i] **一個 tick**。若穿透中位極小，
       「掃流動性」實際上是「創了一個微小的新極值」。
    A4 層級重複度：多少層級彼此在 0.25 ATR 之內（近似同一個價位）
       樞紐很密，相鄰極值常常幾乎同價，會讓事件數灌水。
    A5 樞紐確認到被掃的間隔（bar 數）
       若中位很短，層級根本沒「放置」多久，談不上流動性堆積。
    A6 每個層級被掃幾次
       同一個價位反覆產生事件 = 事件數灌水。

**本檔不下判決，只出分布。** 但 A1 的中位若 < 0.5 ATR，依使用者事前寫下的
話，這條線的語意就要重寫——那個門檻是他先寫的，不是我事後訂的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402

CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
A4_WIN_H = 24       # A4 的時間窗（小時）
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def q(x, name, unit="ATR"):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return None
    return dict(n=int(len(x)), p10=float(np.percentile(x, 10)),
                p25=float(np.percentile(x, 25)), med=float(np.median(x)),
                p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)),
                mean=float(x.mean()))


def show(name, s, fmt="{:.3f}"):
    if s is None:
        print(f"{name:34s} (無資料)")
        return
    print(f"{name:34s} n={s['n']:7,d}  "
          f"p10 {fmt.format(s['p10'])}  p25 {fmt.format(s['p25'])}  "
          f"**中位 {fmt.format(s['med'])}**  p75 {fmt.format(s['p75'])}  "
          f"p90 {fmt.format(s['p90'])}")


def main():
    A1, A2, A3, A5, A6 = [], [], [], [], []
    dup_near, dup_tot = 0, 0
    lvl_hits = []
    n_ev = 0
    for sym in CORE9:
        bars = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        atr = sc.atr14(bars)
        c = [x[sc.C] for x in bars]
        o = [x[sc.O] for x in bars]
        n = len(bars)
        ev = sc.detect_sweeps(bars)
        n_ev += len(ev)

        # A6：同一個 origin 樞紐被掃幾次（引擎每個樞紐只發一次，所以這裡
        # 量的是**不同樞紐落在同一價位**的重複度）
        by_origin = {}
        levels = []
        for e in ev:
            i = e["origin"]
            j = e["j"]
            A = atr[j]
            if A is None or A == 0:
                continue
            conf = min(i + sc.PIVOT, n - 1)          # 樞紐確認的那根
            A_conf = atr[conf]
            if A_conf:
                A1.append(abs(e["level"] - c[conf]) / A_conf)
            A2.append(abs(e["level"] - o[j]) / A)
            kd = 1 if e["kind"] == "buy" else -1
            hj = bars[j][sc.H] if kd == 1 else bars[j][sc.L]
            A3.append((hj - e["level"]) / A if kd == 1
                      else (e["level"] - hj) / A)
            A5.append(j - conf)
            by_origin[i] = by_origin.get(i, 0) + 1
            levels.append((e["level"], A, bars[j][0]))
        A6.extend(by_origin.values())

        # A4：層級彼此有多近。**必須同時看價格與時間** —— 第一版只比價格，
        # 把相差半年、價位剛好接近的兩個層級也算成「重複」，那不是重複。
        # 這裡改成：同一個事件的 ±WINDOW 小時之內，有沒有另一個事件的層級
        # 落在 0.25 ATR 之內。
        if len(levels) > 1:
            lv = np.array([x[0] for x in levels])
            aa = np.array([x[1] for x in levels])
            tt = np.array([x[2] for x in levels], dtype=np.int64)
            srt = np.argsort(tt)
            lv, aa, tt = lv[srt], aa[srt], tt[srt]
            for k in range(len(lv)):
                lo_k = np.searchsorted(tt, tt[k] - A4_WIN_H * 3600, "left")
                hi_k = np.searchsorted(tt, tt[k] + A4_WIN_H * 3600, "right")
                if hi_k - lo_k <= 1:
                    dup_tot += 1
                    continue
                near = np.abs(lv[lo_k:hi_k] - lv[k]) / max(aa[k], 1e-12) <= 0.25
                dup_near += int(near.sum() > 1)      # 扣掉自己
                dup_tot += 1

    print("=== 凍結引擎自審（只用引擎自己的定義，不引入新規則）===")
    print(f"掃單事件總數 {n_ev:,}（九幣、全歷史）")
    print()
    show("A1 層級確認時離當時價格", q(A1, "A1"))
    show("A2 掃單當下離層級（bar 開盤）", q(A2, "A2"))
    show("A3 穿透深度 pierce", q(A3, "A3"))
    show("A5 確認到被掃的 bar 數", q(A5, "A5"), fmt="{:.0f}")
    show("A6 同一樞紐產生的事件數", q(A6, "A6"), fmt="{:.1f}")
    print()
    print(f"A4 同一事件 ±{A4_WIN_H} 小時內、有另一層級在 0.25 ATR 之內的比例："
          f"{dup_near}/{dup_tot} = {dup_near/max(dup_tot,1)*100:.1f}%")
    print()

    s1 = q(A1, "A1")
    if s1:
        verdict = ("**層級是「剛剛的價格附近」——語意要重寫**"
                   if s1["med"] < 0.5 else "層級確實離價格有距離")
        print(f"使用者事前寫下的門檻：A1 中位 < 0.5 ATR -> 語意要重寫")
        print(f"實測中位 {s1['med']:.3f} ATR  ->  {verdict}")

    res = dict(n_events=n_ev, A1=q(A1, "A1"), A2=q(A2, "A2"), A3=q(A3, "A3"),
               A5=q(A5, "A5"), A6=q(A6, "A6"),
               A4_near_share=dup_near / max(dup_tot, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine_audit.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "engine_audit.json")


if __name__ == "__main__":
    main()
