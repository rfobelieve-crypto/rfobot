# -*- coding: utf-8 -*-
"""凍結參數是不是配出來的 —— 它在整個網格裡排第幾？

使用者 2026-09-07：「基礎沒建好做什麼都是錯的」。

基礎最大的破口是：`sweep_core` 的 PIVOT=10 / W=8 / HOLD=8 / DIS=3.5 是在
**看得到全歷史**的情況下定的，而且這條規則的出身自帶 snooping
（`sweep_core` 檔頭：它是在觀察另一個測試的**反面**時得到的）。同家族的
變體 B 已經前瞻 FAIL，所以「這一族會過擬合」不是理論擔憂。

回溯式的 hold-out 救不了已經被看過的參數。但有一個**便宜且決定性**的問題
可以現在回答：

    **凍結的那一組，在網格裡排第幾？**

    · 若它是 81 組裡最好的 -> 它是被挑出來的，效應大半來自選擇
    · 若它落在中段、而且**大多數組合都是正的** -> 效應不是參數造出來的，
      凍結值只是這個高原上的一點

判準（跑之前寫死）
    P1 選擇溢價
        凍結組的 meanR 相對網格**中位**的溢價。
        溢價 / 凍結值 > 50% -> 效應有一半以上來自參數選擇，紅旗
    P2 高原寬度
        網格中 meanR > 0 的組合佔比。
        < 60% -> 效應依賴特定參數，不是穩健現象
    P3 排名
        凍結組的百分位。若 > 90th -> 它坐在峰頂，紅旗
    P4 兩半一致
        把歷史切兩半，各自算網格。**在前半最好的那一組**，在後半排第幾？
        若前半最佳在後半掉到中位以下 -> 這個網格上的「最佳」不可轉移，
        任何靠選參數得到的優勢都是幻覺（**這一項才是真正的 hold-out**）

**本檔不改任何參數。** 它只回答「凍結值有多特別」。
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SF = HERE.parents[0] / "sweep_failure"
sys.path.insert(0, str(SF))
import sweep_core as sc  # noqa: E402

CACHE = SF / ".cache"
OUT = HERE / "data" / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]

GRID = dict(PIVOT=[8, 10, 12], W=[6, 8, 12], HOLD=[6, 8, 12], DIS=[3.0, 3.5, 4.0])
FROZEN = dict(PIVOT=10, W=8, HOLD=8, DIS=3.5)

_BARS: dict = {}


def bars(sym):
    if sym not in _BARS:
        _BARS[sym] = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    return _BARS[sym]


def run(combo, half=None):
    """跑一組參數，回傳 (meanR, n, 逐幣為正數)。half: None/'a'/'b'。"""
    for k, v in combo.items():
        os.environ[k] = str(v)
    importlib.reload(sc)                 # 模組層常數在 import 時才讀 env
    rs, per = [], []
    for sym in CORE9:
        b = bars(sym)
        if half is not None:
            mid = len(b) // 2
            b = b[:mid] if half == "a" else b[mid:]
        r = [t[2] for t in sc.backtest_symbol(b)]
        rs.extend(r)
        per.append(np.mean(r) if r else np.nan)
    if not rs:
        return np.nan, 0, 0
    return float(np.mean(rs)), len(rs), int(np.nansum(np.array(per) > 0))


def grid_runs(half=None):
    out = []
    for p in GRID["PIVOT"]:
        for w in GRID["W"]:
            for h in GRID["HOLD"]:
                for d in GRID["DIS"]:
                    c = dict(PIVOT=p, W=w, HOLD=h, DIS=d)
                    m, n, pos = run(c, half=half)
                    out.append(dict(**c, mean=m, n=n, coins_pos=pos))
    return out


def key(c):
    return (c["PIVOT"], c["W"], c["HOLD"], c["DIS"])


def main():
    fk = (FROZEN["PIVOT"], FROZEN["W"], FROZEN["HOLD"], FROZEN["DIS"])

    print("=== 全歷史網格（81 組）===")
    g = grid_runs()
    means = np.array([x["mean"] for x in g], float)
    fz = next(x for x in g if key(x) == fk)
    rank = int((means < fz["mean"]).sum())
    pct = rank / len(means) * 100
    med = float(np.median(means))
    pos_share = float((means > 0).mean())
    print(f"  凍結組 PIVOT10/W8/HOLD8/DIS3.5  meanR {fz['mean']:+.4f}  "
          f"n={fz['n']:,}  逐幣為正 {fz['coins_pos']}/9")
    print(f"  網格中位 {med:+.4f}   最佳 {means.max():+.4f}   "
          f"最差 {means.min():+.4f}")
    print(f"  凍結組排名 {rank + 1}/{len(means)}（第 {pct:.0f} 百分位）")
    print(f"  meanR > 0 的組合佔比 {pos_share * 100:.1f}%")
    print()

    prem = fz["mean"] - med
    ratio = prem / fz["mean"] if fz["mean"] else np.nan
    v1 = "PASS" if ratio <= 0.50 else "**紅旗 — 一半以上來自參數選擇**"
    print(f"P1 選擇溢價 {prem:+.4f} = 凍結值的 {ratio * 100:.1f}%  -> {v1}")
    v2 = "PASS" if pos_share >= 0.60 else "**紅旗 — 效應依賴特定參數**"
    print(f"P2 高原寬度 {pos_share * 100:.1f}% 為正  -> {v2}")
    v3 = "PASS" if pct <= 90 else "**紅旗 — 坐在峰頂**"
    print(f"P3 排名第 {pct:.0f} 百分位  -> {v3}")

    print()
    print("=== P4 真正的 hold-out：前半選最佳，後半看它排第幾 ===")
    ga = grid_runs(half="a")
    gb = grid_runs(half="b")
    ba = max(ga, key=lambda x: (x["mean"] if np.isfinite(x["mean"]) else -9))
    mb = np.array([x["mean"] for x in gb], float)
    hit = next(x for x in gb if key(x) == key(ba))
    r_b = int((mb < hit["mean"]).sum()) / len(mb) * 100
    fz_b = next(x for x in gb if key(x) == fk)
    r_fz = int((mb < fz_b["mean"]).sum()) / len(mb) * 100
    print(f"  前半最佳 {key(ba)}  前半 meanR {ba['mean']:+.4f}")
    print(f"    -> 它在**後半**的 meanR {hit['mean']:+.4f}，"
          f"排第 {r_b:.0f} 百分位")
    print(f"  凍結組在後半 meanR {fz_b['mean']:+.4f}，排第 {r_fz:.0f} 百分位")
    v4 = ("PASS（選參數的優勢不可轉移，凍結值不比它差）"
          if r_b <= 50 or fz_b["mean"] >= hit["mean"]
          else "**選出來的參數在後半仍然領先 —— 選擇是有效的，凍結值可能落後**")
    print(f"  -> {v4}")

    res = dict(frozen=fz, median=med, best=float(means.max()),
               worst=float(means.min()), rank_pct=pct,
               pos_share=pos_share, premium=prem, premium_ratio=ratio,
               P1=v1, P2=v2, P3=v3, P4=v4,
               half_a_best=ba, half_b_of_a_best=hit, half_b_frozen=fz_b,
               grid=g)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "param_robustness.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "param_robustness.json")


if __name__ == "__main__":
    main()
