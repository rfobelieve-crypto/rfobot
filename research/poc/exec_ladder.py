# -*- coding: utf-8 -*-
"""
========================================================================
2026-09-07 **L3 觸發：換進場價救不回來。真實 edge ≈ 0，扣成本後為負。**
========================================================================
    路徑                每事件 R      日聚類 CI95
    凍結（已推翻）        +0.0366    [+0.0128, +0.0615]
    掛限價在價位          −0.0504    （積分值）
    **市價立刻成交**      **−0.0008**  **[−0.0251, +0.0238]**  <- 零，不是負

    扣成本（逐幣真實 bps，cost_R = bps/1e4 / (DIS × ATR%)）
      掛限價 × A(10bps) −0.0789   × B(13bps) −0.0874
      市價   × A(10bps) **−0.0293**  × B(13bps) −0.0378

    L1 已知答案對照 **PASS**（積分 −0.0504 vs honest_fill B2 −0.0463，
       差 0.0041 < 0.005 —— 兩份獨立實作同意）
    L3 使用者事前門檻（< +0.01 R 即救不回來）：最佳 −0.0293 -> **觸發**

六格的 E[R|成交] **全部為負**，沒有一格是活的。凍結的 +0.0366 完全來自
「≤0 已越過」那一格（0.579 × +0.0946 = +0.0548，其餘五格合計 −0.0182），
而那一格誠實執行是 −0.0613 —— 一來一回擺盪 **0.156 R**。

markout 欄把逆選擇講得很清楚：離價位越遠（gap 越大）成交後越糟
（>1.00 那格 5 分鐘 −0.366）——在市場遠離價位時「成交在價位」帳面好看，
但那個位置本來就在對你不利的一側。

**跑完之後修掉的兩個自己的錯（留檔）**
  1. 分層標籤方向反了（≤0 才是「已越過」）。計算一直是對的——≤0 格
     n=4,081、凍結 +0.0946 與 `honest_fill` 的 B 情境逐項吻合——錯的只有
     名字，但這種錯會嚴重誤導讀表的人。
  2. 成本換算漏掉 DIS=3.5，成本被高估 3.5 倍（−0.0995 R vs 正確 −0.0284）。
     檔頭公式一直是對的，程式碼漏了。注意 `conj_tradability.py` **沒有**
     這個問題：那裡的效應是 ATR 單位，不需要再除 DIS。**單位不同公式就不同。**
  兩個錯都不改變 L3 的判決（修正前後最佳值都遠低於 +0.01）。
執行階梯 —— 真實 edge 還剩多少（TODO §1.02 的重算）

使用者 2026-09-07：
    「真實 edge 是多少……需要用分層表的成交率和 markout 重算一次歷史……
     如果連 0.01R 都不到，換進場價也救不回來，那是另一種結論。」

===========================================================================
為什麼前兩張分層表作廢，這張不會
===========================================================================
`resting_fill.py`  分層變數 Spearman(深度, R) = **+0.6884**  -> 整組作廢
`resting_limit.py` 分層變數 Spearman           = **−0.7127**  -> Q2 不可解讀

兩個都是「價格位移」的變形，與結果機械相關。差別不在符號，在**用途**：

    前兩次問「**哪一格**有 edge」   -> 挑格，套套邏輯致命
    本檔問「**全部加起來**還剩多少」 -> 積分，不挑格

    可達 edge = Σ_格 佔比 × 成交率 × E[R | 成交]
                未成交貢獻 **0**（沒有部位，不是反事實利潤）

在「積分」這個用途下，分層變數與結果相關**不是缺陷而是必要**——我們就是要
知道每一種市場狀態下實際拿得到什麼。**挑格才是陷阱，本檔不挑格。**

分層變數（決策時點已知，這是唯一會致命的地方）
    gap = d × (lvl − close(掃單 bar j)) / ATR

    掃單 bar j 收盤之後才下單，所以 close(j) 在決策當下已知。
    gap ≤ 0   **市場已越過價位** -> 凍結記的「成交在 lvl」拿不到
    gap > 0   市場還在穿越那一側、離價位 gap 個 ATR
    （第一版檔頭把方向寫反了，程式碼一直是對的；已在 LABELS 旁留註。）

    **不得**用成交 bar 或之後的任何資訊（mistake.md 2026-09-03 的 +0.52R）。

每一格報告（全格，不挑）
    n / 佔比
    成交率      在 lvl 掛限價，回踩窗 W=8 根內價格是否回到 lvl
    E[R|成交]   完整凍結出場（3.5 ATR 災難停損 + HOLD=8）
    markout     成交後 5 / 15 / 60 分鐘，方向對齊 / risk（逆選擇的直接度量）
    每事件貢獻  佔比 × 成交率 × E[R|成交]
    凍結對照    同一格凍結記的 R（看它多給了多少）

    並列第二條路徑「市價立刻成交」：成交率 100%，但進場價是真實可成交價。

成本：逐幣真實 bps（凍結成本模型，2026-07-28 定，未調參）
    情境 A 目標執行 進場 7 + 時間出場 3 = 10 bps
    情境 B 全 taker 進場 7 + 時間出場 6 = 13 bps
    cost_R = bps/1e4 × price / (DIS × ATR)   **逐幣**，不用統一 ATR 單位

===========================================================================
判準（跑之前寫死）
===========================================================================
    L1 已知答案的對照（**必須**過，否則以下不解讀）
       積分出來的「掛單路徑」每事件值，必須重現 `honest_fill.py` 的
       B2 每事件 **−0.0463**（容差 0.005）。重現不出來代表我又寫了第二份
       實作而且它們不同意（mistake.md 2026-08-26）。
    L2 真實 edge 還剩多少（主結論，零成本）
       兩條路徑的每事件值，日聚類 CI。
    L3 扣成本後
       兩條路徑各扣情境 A / B 成本後的每事件值。
       **若兩條路徑扣完都 < +0.01 R -> 換進場價救不回來**（使用者事前寫下
       的門檻），這條線的結論是「訊號有效但不可執行」。
    L4 不挑格的紀律
       全格報告。任何單一格都不構成「只做這一格」的依據——那需要它自己的
       前瞻註冊。本檔輸出**不得**被用來事後選格。
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
MARKOUTS = (5, 15, 60)
# 分層邊界（ATR 單位）。事前寫死，不因結果調整。
EDGES = [-np.inf, 0.0, 0.10, 0.25, 0.50, 1.00, np.inf]
# 標籤方向:gap = d×(lvl − close(j))/ATR。做空 d=−1 時,市場已越過價位
# (收在 lvl 下方)算出來是**負的**。所以 ≤0 才是「已越過」= 凍結價拿不到的
# 那一側。第一版把 ≤0 標成「A情境」是反的——計算沒錯,只有名字錯,但這種錯
# 會嚴重誤導讀表的人,所以在原地留註。
LABELS = ["≤0 已越過(凍結價拿不到)", "0–0.10", "0.10–0.25",
          "0.25–0.50", "0.50–1.00", ">1.00 未越過(離價位最遠)"]
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

        m = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "high", "low", "close"])
        mts = m["ts"].to_numpy(np.int64)
        mhi = np.nan_to_num(m["high"].to_numpy(float), nan=-np.inf)
        mlo = np.nan_to_num(m["low"].to_numpy(float), nan=np.inf)
        mcl = m["close"].to_numpy(float)
        nm = len(mts)
        pxs = m["close"].to_numpy(float)
        atrpct[sym] = float(np.nanmedian(
            pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["atr_h14"])["atr_h14"].to_numpy(float)
            / np.where(pxs > 0, pxs, np.nan)))

        for e in sc.backtest_symbol(b1, detail=True):
            j, lvl, A, d, risk = (e["j"], e["level"], e["atr"], e["d"], e["risk"])
            t0 = int(b1[j][0]) * 1000 + HOUR_MS
            t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
            i0 = int(np.searchsorted(mts, t0, side="left"))
            i1 = int(np.searchsorted(mts, t1, side="left"))
            if i0 >= nm or i1 <= i0:
                continue

            # 分層變數：決策時點（掃單 bar 收盤）市場已越過價位多遠
            gap = d * (lvl - cl[j]) / A

            # 路徑 1 掛限價在 lvl：價格回到 lvl 才成交
            if d == -1:
                hit = np.flatnonzero(mhi[i0:i1] >= lvl)
            else:
                hit = np.flatnonzero(mlo[i0:i1] <= lvl)
            if len(hit):
                k = i0 + int(hit[0])
                fb = int(np.searchsorted(hts, int(mts[k]), side="right")) - 1
                if 0 <= fb and fb + 1 < n:
                    r_rest = exit_from(h, lo, cl, n, fb, d,
                                       lvl + d * sc.SLIP * A, risk, A)
                    mo = {t: (float(d * (mcl[k + t] - lvl) / risk)
                              if k + t < nm else np.nan) for t in MARKOUTS}
                    filled = 1
                else:
                    r_rest, mo, filled = np.nan, {t: np.nan for t in MARKOUTS}, 0
            else:
                r_rest, mo, filled = np.nan, {t: np.nan for t in MARKOUTS}, 0

            # 路徑 2 市價立刻成交：掃單 bar 收盤後第一分鐘的收盤價
            px = float(mcl[i0])
            fb2 = min(j + 1, n - 1)
            r_mkt = exit_from(h, lo, cl, n, fb2, d, px + d * sc.SLIP * A, risk, A)

            rows.append(dict(
                sym=sym, gap=float(gap), lvl=float(lvl),
                R_frozen=float(e["R"]), R_rest=float(r_rest) if filled else np.nan,
                filled=int(filled), R_mkt=float(r_mkt),
                **{f"mo{t}": mo[t] for t in MARKOUTS},
                day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d")))

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "exec_ladder.parquet", index=False)
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

    print("=== 執行階梯：決策時點市場已越過價位多遠（ATR）===")
    print(f"凍結交易 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日")
    print("**全格報告，不挑格。** 任何單一格都不構成「只做這一格」的依據。")
    print()
    hdr = (f"{'gap 分層':13s} {'n':>6s} {'佔比':>7s} {'成交率':>7s} "
           f"{'E[R|成交]':>10s} {'貢獻':>9s} {'凍結對照':>9s}   "
           + "".join(f"{'mo' + str(t) + 'm':>9s}" for t in MARKOUTS))
    print(hdr)
    tbl = {}
    contrib_rest = 0.0
    for lab in LABELS:
        g = d[d.bin == lab]
        if not len(g):
            continue
        share = len(g) / len(d)
        fr = float(g.filled.mean())
        er = float(g.R_rest.mean()) if g.filled.sum() >= 30 else np.nan
        con = share * fr * (er if np.isfinite(er) else 0.0)
        contrib_rest += con
        fz = float(g.R_frozen.mean())
        mos = [float(g[f"mo{t}"].mean()) for t in MARKOUTS]
        tbl[lab] = dict(n=int(len(g)), share=share, fill_rate=fr,
                        e_r=er, contrib=con, frozen=fz,
                        markout={str(t): mo for t, mo in zip(MARKOUTS, mos)})
        print(f"{lab:13s} {len(g):6,d} {share*100:6.2f}% {fr*100:6.2f}% "
              f"{er:+10.4f} {con:+9.4f} {fz:+9.4f}   "
              + "".join(f"{mo:+9.4f}" for mo in mos))

    print()
    print("=== 兩條路徑的每事件值（積分，不挑格）===")
    print()
    rest_pe = contrib_rest
    mkt_pe, mlo_, mhi_ = day_ci(d.R_mkt.to_numpy(), d.day.to_numpy())
    fz_pe, flo, fhi = day_ci(d.R_frozen.to_numpy(), d.day.to_numpy())
    print(f"{'路徑':22s} {'每事件 R':>10s} {'日聚類 CI95':>24s}")
    print(f"{'凍結（已推翻）':22s} {fz_pe:+10.4f}  [{flo:+.4f}, {fhi:+.4f}]")
    print(f"{'掛限價在價位':22s} {rest_pe:+10.4f}  （積分值，見上表）")
    print(f"{'市價立刻成交':22s} {mkt_pe:+10.4f}  [{mlo_:+.4f}, {mhi_:+.4f}]")

    # L1 已知答案對照
    ref = -0.0463
    ok1 = abs(rest_pe - ref) < 0.005
    print()
    print(f"L1 已知答案對照：掛單路徑每事件 {rest_pe:+.4f} vs "
          f"honest_fill B2 {ref:+.4f}  差 {abs(rest_pe-ref):.4f}（需 <0.005）"
          f"  -> {'PASS' if ok1 else '**FAIL — 兩份實作不同意，以下不解讀**'}")

    print()
    print("=== L3 扣成本後（逐幣真實 bps）===")
    print()
    w = d.groupby("sym").size()
    ap_w = float(sum(w[s] * atrpct[s] for s in w.index) / w.sum())
    print(f"  加權 ATR% = {ap_w*100:.3f}%")
    res = {"n": int(len(d)), "table": tbl, "rest_per_event": rest_pe,
           "mkt_per_event": mkt_pe, "frozen_per_event": fz_pe,
           "atr_pct_weighted": ap_w, "L1": bool(ok1)}
    verdicts = {}
    for name, pe in (("掛限價", rest_pe), ("市價", mkt_pe)):
        for k, bps in SCEN.items():
            # cost_R = bps/1e4 × price / (DIS × ATR) = (bps/1e4)/(DIS × ATR%)
            # 第一版漏了 DIS=3.5,成本被高估 3.5 倍(-0.0995 R vs 正確 -0.0284)。
            # 注意 conj_tradability 沒有這個問題——那裡的效應是 ATR 單位,
            # 不需要再除 DIS。單位不同,公式就不同。
            net = pe - bps / 1e4 / (sc.DIS * ap_w)
            verdicts[f"{name}/{k}"] = net
            print(f"  {name:6s} × {k:10s} 成本 {bps:2d} bps "
                  f"= {bps/1e4/(sc.DIS*ap_w):.4f} R "
                  f"-> 每事件 {net:+.4f} R")
    res["net"] = verdicts

    best = max(verdicts.values())
    v3 = ("**換進場價救不回來** — 兩條路徑扣成本後都 < +0.01 R"
          if best < 0.01 else
          f"最好的一格 {best:+.4f} R ≥ +0.01，還有可談的空間")
    print()
    print(f"L3 使用者事前門檻（< +0.01 R 即救不回來）：最佳 {best:+.4f}  -> {v3}")
    res["L3"] = dict(best=best, verdict=v3)

    (OUT / "exec_ladder.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "exec_ladder.json")
    print()
    print("**L4：全格報告，不得事後選格。** 任一格若要單獨採用，需要它自己的前瞻註冊。")


if __name__ == "__main__":
    main()
