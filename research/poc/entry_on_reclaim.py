# -*- coding: utf-8 -*-
"""進場改成「收回確認後的可成交價」—— 使用者 2026-09-07 提案

    「收回確認後的可成交價進場，如果進場改成這樣呢」

這與 `exec_ladder.py` 的「市價立刻成交」**不是同一條規則**，必須分開量：

    exec_ladder 的市價臂   掃單 bar 一收盤就進場
                          -> 對「已越過」那 57.9% 來說收回確實發生了，相同
                          -> 但對另外 42.1%（收盤還在穿越那一側），**收回
                             還沒發生**，那一臂等於在未確認時搶進。不是本提案。

    本檔（使用者提案）      **等收回被確認，才用當下拿得到的價進場**
                          沒收回就**不進場**（沒有部位，不是虧損）

===========================================================================
規則（跑之前寫死）
===========================================================================
    價位 lvl、掃單 bar j、方向 d 全部沿用凍結引擎，一個字不動。

    收回確認：掃單 bar j 收盤之後，**第一根收盤落在價位內側的 1 分鐘 K**。
        買側被掃（做空 d=−1）-> 內側 = close < lvl
        賣側被掃（做多 d=+1）-> 內側 = close > lvl
        若掃單 bar j 自己的收盤已在內側，則確認時刻就是 j 收盤，
        進場用其後的第一根分鐘 K。

    等待窗：與凍結規則相同的回踩窗（到第 j+W 根小時收盤為止）。
            窗內沒有任何一分鐘收在內側 -> **不成交**。

    進場價（兩種都報，不挑對自己有利的）
        E1 確認那根分鐘 K 的**收盤**（本專案既有慣例：§0.98 的 I1 修正版
           就是「收盤收回正確一側就在那根收盤進場」）
        E2 **下一根**分鐘 K 的收盤（更保守，涵蓋「你不可能在收盤那一刻
           成交」的現實）
        兩者都再付凍結的 SLIP=0.05 ATR 逆向滑價。

    出場：完整凍結規則（3.5 ATR 災難停損 + HOLD=8），從進場所屬的小時 bar
          起算，停損從下一根查起。SLIP 不動。

===========================================================================
判準（寫在 CI 上，不寫在點估計上）
===========================================================================
    N1 零成本每事件值（＝ 成交率 × E[R|成交]，未成交貢獻 0）
       日聚類 CI。E1 / E2 都報。
    N2 扣成本後（逐幣真實 bps，cost_R = bps/1e4 / (DIS × ATR%)）
       **若 E1 與 E2 扣完都 < +0.01 R -> 這個進場改法救不回來**
       （使用者事前寫下的門檻，與 `exec_ladder` 的 L3 同一條）
    N3 成交率必須報，且 <50% 要明講「這是換策略不是改進場」
    N4 已知答案的對照（**必須**過，否則 N1/N2 不解讀）
       「掃單 bar 收盤已在內側」那一格（gap ≤ 0，57.9%），本規則的確認
       時刻就是 j 收盤，所以它的 E1 應該與 `exec_ladder` 市價臂在同一格的
       值接近（容差 0.01）。差太多代表兩份實作不同意。
    N5 全格報告，不挑格。任一格若要單獨採用需要它自己的前瞻註冊。

**這是新的進場規則 = 新變體，全部 in-sample。** 就算為正也不得直接採用，
必須另開前瞻時鐘（§0.92 變體 B 的教訓：事後挑出來的變體會前瞻 FAIL）。
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
EDGES = [-np.inf, 0.0, 0.10, 0.25, 0.50, 1.00, np.inf]
LABELS = ["≤0 已越過", "0–0.10", "0.10–0.25", "0.25–0.50",
          "0.50–1.00", ">1.00 最遠"]
SCEN = {"A 目標執行": 10, "B 全 taker": 13}
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
    atrpct = {}
    for sym in CORE9:
        b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        h = [x[sc.H] for x in b1]
        lo = [x[sc.L] for x in b1]
        cl = [x[sc.C] for x in b1]
        n = len(b1)
        hts = np.array([int(x[0]) for x in b1], np.int64) * 1000

        m = pd.read_parquet(BARS / f"{sym}.parquet", columns=["ts", "close"])
        mts = m["ts"].to_numpy(np.int64)
        mcl = m["close"].to_numpy(float)
        nm = len(mts)
        atrpct[sym] = float(np.nanmedian(
            pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["atr_h14"])["atr_h14"].to_numpy(float)
            / np.where(mcl > 0, mcl, np.nan)))

        for e in sc.backtest_symbol(b1, detail=True):
            j, lvl, A, d, risk = (e["j"], e["level"], e["atr"], e["d"], e["risk"])
            t0 = int(b1[j][0]) * 1000 + HOUR_MS
            t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
            i0 = int(np.searchsorted(mts, t0, side="left"))
            i1 = int(np.searchsorted(mts, t1, side="left"))
            if i0 >= nm or i1 <= i0:
                continue
            gap = d * (lvl - cl[j]) / A

            # 收回確認：掃單 bar 收盤已在內側 -> 確認時刻就是 j 收盤，
            # 進場用其後第一根分鐘 K；否則等窗內第一根收在內側的分鐘。
            inside_at_j = (cl[j] < lvl) if d == -1 else (cl[j] > lvl)
            if inside_at_j:
                k = i0
            else:
                seg = mcl[i0:i1]
                hit = np.flatnonzero(seg < lvl) if d == -1 else \
                    np.flatnonzero(seg > lvl)
                k = i0 + int(hit[0]) if len(hit) else -1

            if k < 0 or k + 1 >= nm:
                rows.append(dict(sym=sym, gap=float(gap), filled=0,
                                 R_e1=np.nan, R_e2=np.nan,
                                 R_frozen=float(e["R"]),
                                 day=pd.Timestamp(int(b1[j][0]) * 1000,
                                                  unit="ms", tz="UTC"
                                                  ).strftime("%Y-%m-%d")))
                continue

            def r_at(idx):
                px = float(mcl[idx])
                fb = int(np.searchsorted(hts, int(mts[idx]), side="right")) - 1
                if fb < 0 or fb + 1 >= n:
                    return np.nan
                return exit_from(h, lo, cl, n, fb, d,
                                 px + d * sc.SLIP * A, risk, A)

            rows.append(dict(
                sym=sym, gap=float(gap), filled=1,
                R_e1=r_at(k), R_e2=r_at(k + 1),
                wait_min=int((mts[k] - mts[i0]) // 60_000),
                R_frozen=float(e["R"]),
                day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d")))

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "entry_on_reclaim.parquet", index=False)
    d["bin"] = pd.cut(d.gap, EDGES, labels=LABELS, right=False)

    def day_ci(x, days, b=2000):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        x, days = x[ok], np.asarray(days)[ok]
        if len(x) < 30:
            return (float("nan"),) * 3
        uq, inv = np.unique(days, return_inverse=True)
        ix = [np.where(inv == k)[0] for k in range(len(uq))]
        reps = np.empty(b)
        for i in range(b):
            p = RNG.integers(0, len(uq), len(uq))
            reps[i] = x[np.concatenate([ix[k] for k in p])].mean()
        return (float(x.mean()), float(np.percentile(reps, 2.5)),
                float(np.percentile(reps, 97.5)))

    fr = float(d.filled.mean())
    print("=== 進場改成「收回確認後的可成交價」（使用者提案）===")
    print(f"掃單事件 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日")
    print(f"N3 成交率 {fr*100:.2f}%"
          + ("   （<50%，這是換策略不是改進場）" if fr < 0.50 else ""))
    if "wait_min" in d.columns:
        w = d.loc[d.filled == 1, "wait_min"]
        print(f"   收回確認等待時間：中位 {w.median():.0f} 分、"
              f"p90 {w.quantile(.9):.0f} 分")
    print()
    print("**全格報告，不挑格。**")
    print()
    print(f"{'gap 分層':12s} {'n':>6s} {'佔比':>7s} {'成交率':>7s} "
          f"{'E1 E[R|成交]':>12s} {'E2':>9s} {'E1 貢獻':>9s} {'凍結對照':>9s}")
    tbl, c1, c2 = {}, 0.0, 0.0
    for lab in LABELS:
        g = d[d.bin == lab]
        if not len(g):
            continue
        sh = len(g) / len(d)
        f_ = float(g.filled.mean())
        e1 = float(g.R_e1.mean()) if g.filled.sum() >= 30 else np.nan
        e2 = float(g.R_e2.mean()) if g.filled.sum() >= 30 else np.nan
        con1 = sh * f_ * (e1 if np.isfinite(e1) else 0.0)
        con2 = sh * f_ * (e2 if np.isfinite(e2) else 0.0)
        c1 += con1
        c2 += con2
        tbl[lab] = dict(n=int(len(g)), share=sh, fill=f_, e1=e1, e2=e2,
                        contrib_e1=con1, frozen=float(g.R_frozen.mean()))
        print(f"{lab:12s} {len(g):6,d} {sh*100:6.2f}% {f_*100:6.2f}% "
              f"{e1:+12.4f} {e2:+9.4f} {con1:+9.4f} "
              f"{g.R_frozen.mean():+9.4f}")

    print()
    print("=== N1 每事件值（零成本）===")
    print()
    m1, l1, h1 = day_ci(d.R_e1.fillna(0).to_numpy(), d.day.to_numpy())
    m2, l2, h2 = day_ci(d.R_e2.fillna(0).to_numpy(), d.day.to_numpy())
    print(f"  E1 確認那根收盤   每事件 {m1:+.4f}  CI [{l1:+.4f}, {h1:+.4f}]"
          f"   （積分 {c1:+.4f}）")
    print(f"  E2 下一根收盤     每事件 {m2:+.4f}  CI [{l2:+.4f}, {h2:+.4f}]"
          f"   （積分 {c2:+.4f}）")

    w = d.groupby("sym").size()
    ap = float(sum(w[s] * atrpct[s] for s in w.index) / w.sum())
    print()
    print(f"=== N2 扣成本後（加權 ATR% = {ap*100:.3f}%）===")
    print()
    best = -9.0
    nets = {}
    for nm_, pe in (("E1", m1), ("E2", m2)):
        for k_, bps in SCEN.items():
            net = pe - bps / 1e4 / (sc.DIS * ap)
            nets[f"{nm_}/{k_}"] = net
            best = max(best, net)
            print(f"  {nm_} × {k_:10s} 成本 {bps} bps = "
                  f"{bps/1e4/(sc.DIS*ap):.4f} R -> 每事件 {net:+.4f} R")
    v2 = ("**這個進場改法救不回來**（都 < +0.01 R）" if best < 0.01
          else f"最佳 {best:+.4f} ≥ +0.01，有可談的空間")
    print()
    print(f"N2 使用者門檻（< +0.01 R 即救不回來）：最佳 {best:+.4f}  -> {v2}")

    # N4 已知答案對照
    ref = None
    p = OUT / "exec_ladder.parquet"
    if p.exists():
        el = pd.read_parquet(p)
        ref = float(el.loc[el.gap <= 0, "R_mkt"].mean())
    g0 = d[d.gap <= 0]
    own = float(g0.R_e1.mean())
    ok4 = ref is None or abs(own - ref) < 0.01
    print()
    print(f"N4 已知答案對照：gap≤0 格 本檔 E1 {own:+.4f} vs "
          f"exec_ladder 市價臂 {ref if ref is None else f'{ref:+.4f}'}"
          f"  -> {'PASS' if ok4 else '**FAIL — 兩份實作不同意，上面不解讀**'}")

    res = dict(n=int(len(d)), fill_rate=fr, e1=m1, e1_ci=[l1, h1],
               e2=m2, e2_ci=[l2, h2], table=tbl, net=nets,
               best=best, verdict=v2, N4_ref=ref, N4_own=own, N4=bool(ok4),
               atr_pct=ap)
    (OUT / "entry_on_reclaim.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "entry_on_reclaim.json")
    print()
    print("**這是新變體，全部 in-sample。就算為正也不得直接採用，必須另開前瞻時鐘。**")


if __name__ == "__main__":
    main()
