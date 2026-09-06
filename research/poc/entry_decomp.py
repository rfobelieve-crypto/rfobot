# -*- coding: utf-8 -*-
"""執行落差的拆解 — §0.57 當時做不到的那一步。

背景
    §0.57（2026-08-24）量到批次發布架構下每筆損失 **0.1328 R**：
        凍結（level 價）  meanR +0.0839   CI [+0.052,+0.117]  9/9 幣
        可實現（fill bar 收盤）    −0.0489   CI [−0.076,−0.023]  3/9
    但「fill bar 收盤價」是被**只有 1 小時 bar**逼出來的最壞情況代理。
    產品端每 60 秒輪詢，實際拿得到的價格在「level 價」與「小時收盤」之間，
    而那個中間值從來沒有被量過。

    現在有九幣 935 天的完整 1 分鐘 bar，可以把落差拆成兩段：

        A  lvl                      掛限價在價位上（凍結假設 / 理論上限）
        B  觸價那一分鐘的收盤        60 秒輪詢 + 市價單（真正的可實現值）
        C  小時 fill bar 的收盤      批次發布（§0.57 的悲觀代理）

        A - B  = 盤中無法避免的滑價（掛單架構也拿不回來的部分）
        B - C  = **批次發布的懲罰** = `raid_pending_levels` 要拿回的那一段

    三臂**共用同一個出場**，所以差值裡沒有市場路徑，只有進場價。

凍結規則一個字不動：PIVOT/W/HOLD/DIS/SLIP 全部沿用 sweep_core。
變體 A（無濾網、core9）——變體 B 已於 §0.92 判 FAIL，不拿它的數字當目標。

**這是刻畫，不是判決。** 它回答「落差由什麼組成、哪一段拿得回來」。
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
OUT = HERE / "data" / "results"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
MIN_MS = 60_000
HOUR_MS = 3_600_000
RNG = np.random.default_rng(20260906)


def build(sym):
    b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    atr = sc.atr14(b1)
    h = [x[sc.H] for x in b1]
    lo = [x[sc.L] for x in b1]
    cl = [x[sc.C] for x in b1]
    n = len(b1)

    m = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low", "close"])
    mts = m["ts"].to_numpy(np.int64)
    mhi = np.nan_to_num(m["high"].to_numpy(float), nan=-np.inf)
    mlo = np.nan_to_num(m["low"].to_numpy(float), nan=np.inf)
    mcl = m["close"].to_numpy(float)

    rows = []
    last_exit = -1
    for e in sc.detect_sweeps(b1):
        j, lvl = e["j"], e["level"]
        A = atr[j]
        if A is None or A == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd                                   # 反轉方向
        fill = None
        for f in range(j + 1, min(j + 1 + sc.W, n)):
            if kd == 1 and lo[f] <= lvl:
                fill = f
                break
            if kd == -1 and h[f] >= lvl:
                fill = f
                break
        if fill is None or fill <= last_exit or fill + 1 >= n:
            continue

        # --- 三個進場價 -------------------------------------------------
        e_A = lvl + d * sc.SLIP * A               # 掛限價在價位（凍結）
        e_C = cl[fill] - d * sc.SLIP * A          # 小時 fill bar 收盤（批次）
        # 60 秒輪詢：fill 那個小時之內，第一根觸價的分鐘，成交在它的收盤
        # 單位:1h 快取的 time 是**秒**、1 分鐘 parquet 的 ts 是**毫秒**
        # (mistake.md 2026-04-12:同一供應商不同端點不同單位,轉換要顯式)
        hour_ms = int(b1[fill][0]) * 1000
        i0 = int(np.searchsorted(mts, hour_ms, side="left"))
        e_B = np.nan
        touch_min = -1
        for k in range(i0, min(i0 + 60, len(mts))):
            if mts[k] >= hour_ms + HOUR_MS:
                break
            if (kd == 1 and mlo[k] <= lvl) or (kd == -1 and mhi[k] >= lvl):
                e_B = mcl[k] - d * sc.SLIP * A
                touch_min = int((mts[k] - hour_ms) // MIN_MS)
                break
        if not np.isfinite(e_B):
            continue

        # --- 共用出場：凍結的時間出場（三臂同一根 bar，路徑被消掉）------
        # 配對設計刻意**不含災難停損**:停損是進場相對的,三臂會有三個停損價、
        # 三個出場,配對性就沒了。所以本表的「差值」乾淨,「水準」則不是完整
        # 凍結規則的 meanR —— 後者另外算(R_A_full),兩者不可混用。
        exitbar = min(fill + sc.HOLD, n - 1)
        ex = cl[exitbar] - d * sc.SLIP * A
        risk = sc.DIS * A

        # --- 完整凍結規則(含災難停損)下的 A 臂:水準用這個 ----------------
        e_full = lvl + d * sc.SLIP * A
        stop = e_full - d * risk
        R_full = None
        for k in range(fill + 1, min(fill + sc.HOLD + 1, n)):
            if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                R_full = -1.0 - sc.SLIP / sc.DIS
                break
        if R_full is None:
            R_full = d * (cl[min(fill + sc.HOLD, n - 1)] - d * sc.SLIP * A
                          - e_full) / risk
        rows.append(dict(
            sym=sym, fill_ts=hour_ms, hour_ts=hour_ms,
            day=pd.Timestamp(hour_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
            side="LONG" if d == 1 else "SHORT", lvl=lvl, atr=A,
            touch_minute=touch_min,
            R_A=d * (ex - e_A) / risk,
            R_A_full=R_full,
            R_B=d * (ex - e_B) / risk,
            R_C=d * (ex - e_C) / risk,
            slip_AB=d * (e_B - e_A) / risk,       # A->B 讓掉多少 R
            slip_BC=d * (e_C - e_B) / risk,       # B->C 再讓掉多少 R
        ))
        last_exit = exitbar
    return pd.DataFrame(rows)


def day_cluster_ci(x, days, b=2000):
    uq, inv = np.unique(days, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([idx[k] for k in p])].mean()
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), \
        float(np.std(reps, ddof=1))


def main():
    d = pd.concat([build(s) for s in CORE9], ignore_index=True)
    if d.empty or "day" not in d.columns:
        sys.exit("no rows produced -- check the seconds/milliseconds conversion")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "entry_decomp.parquet", index=False)
    days = d["day"].to_numpy()
    print(f"變體 A（無濾網、core9）成交 n={len(d):,}  UTC 日={d.day.nunique():,}  "
          f"幣={d.sym.nunique()}\n")

    print("=== 三種進場，共用出場（時間出場，同一根 bar）===\n")
    print(f"{'臂':34s} {'meanR':>9s} {'日聚類 CI95':>22s} {'逐幣正':>7s}")
    res = {}
    x = d["R_A_full"].to_numpy(float)
    lo_, hi_, se_ = day_cluster_ci(x, days)
    pos = int((d.groupby("sym")["R_A_full"].mean() > 0).sum())
    res["R_A_full"] = dict(mean=float(x.mean()), ci=[lo_, hi_], se=se_, coins_pos=pos)
    print(f"{'[水準] A + 完整凍結規則(含停損)':34s} {x.mean():+9.4f}  "
          f"[{lo_:+.4f}, {hi_:+.4f}]   {pos}/9   t={x.mean()/se_:+.2f}")
    print()
    for col, name in (("R_A", "A  掛限價在 level（無停損,配對用）"),
                      ("R_B", "B  觸價分鐘收盤（60 秒輪詢）"),
                      ("R_C", "C  小時 bar 收盤（批次發布）")):
        x = d[col].to_numpy(float)
        lo, hi, se = day_cluster_ci(x, days)
        pos = int((d.groupby("sym")[col].mean() > 0).sum())
        res[col] = dict(mean=float(x.mean()), ci=[lo, hi], se=se, coins_pos=pos)
        print(f"{name:34s} {x.mean():+9.4f}  [{lo:+.4f}, {hi:+.4f}]   {pos}/9")

    print("\n=== 落差拆解 ===\n")
    for col, name in (("slip_AB", "A->B  盤中滑價（掛單也拿不回）"),
                      ("slip_BC", "B->C  **批次發布的懲罰**（可拿回）")):
        x = d[col].to_numpy(float)
        lo, hi, se = day_cluster_ci(x, days)
        res[col] = dict(mean=float(x.mean()), ci=[lo, hi], se=se)
        print(f"{name:34s} {x.mean():+9.4f}  [{lo:+.4f}, {hi:+.4f}]")
    tot = d["slip_AB"].mean() + d["slip_BC"].mean()
    print(f"{'合計 A->C':34s} {tot:+9.4f}   "
          f"（§0.57 在變體 B 上量到 0.1328）")
    res["total_gap"] = float(tot)

    print("\n=== 觸價發生在小時的第幾分鐘 ===\n")
    tm = d["touch_minute"]
    print(f"  中位 {tm.median():.0f} 分  q25 {tm.quantile(.25):.0f}  "
          f"q75 {tm.quantile(.75):.0f}  在前 5 分鐘內 {(tm<5).mean()*100:.1f}%")
    res["touch_minute_median"] = float(tm.median())

    print("\n=== 逐幣（meanR）===\n")
    g = d.groupby("sym")[["R_A", "R_B", "R_C", "slip_AB", "slip_BC"]].mean()
    print(g.to_string(float_format=lambda x: f"{x:+.4f}"))
    res["per_coin"] = g.to_dict("index")

    (OUT / "entry_decomp.json").write_text(json.dumps(res, indent=2, default=float),
                                           encoding="utf-8")
    print("\nwritten ->", OUT / "entry_decomp.json")


if __name__ == "__main__":
    main()
