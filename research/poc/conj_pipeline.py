# -*- coding: utf-8 -*-
"""現行管線接得住交會事件嗎 —— 每小時 vs 三分鐘

`conj_entry.py`（2026-09-07）量到：交會事件的可執行窗口約**三分鐘**
（P1 +0.293、P2 +0.206、P3 +0.155、P5 +0.103、P10 +0.056 CI 含零）。

而現行的訊號管線是**每小時**：
    SweepShadow 排程 PT1H -> shadow_engine.bat -> raid_signals_publish.py
    -> `/public/signal-feed` -> 產品端輪詢

**20 倍的落差。** 這件事決定這條線能不能用現有基礎設施跑，而且它應該在
**開時鐘之前**就問——掃單線的教訓正是「可執行性要早驗」。時鐘要跑四個月，
如果管線根本接不住，那四個月是白等。

===========================================================================
本檔量兩件事
===========================================================================
1. **固定延遲的完整衰減曲線**，延伸到 60 分鐘（`conj_entry` 只到 10 分）。
2. **每小時批次實際交付什麼**：訊號在小時內的哪一分鐘觸發是均勻的，
   所以批次管線給的延遲是 **Uniform(0, 60] 分鐘**，不是固定 60 分。
   直接用每個事件在小時內的真實位置算，不用模擬。

    d_batch(t) = 60 − (t 在該小時的第幾分鐘)
    也就是「等到下一個整點批次」還要多久。

出場：與 `conj_entry` 一致，固定 close(t+60)。**延遲越久，價格越差且持有
越短**——兩個效應都算進去。

成本：凍結模型（A 10 bps / B 13 bps），cost_ATR = bps/1e4 / ATR%
      （效應是 ATR 單位，不再除 DIS —— `exec_ladder` 檔頭記載的那個錯）

===========================================================================
判準（跑之前寫死）
===========================================================================
    Y1 已知答案的對照（**必須**過，否則以下不解讀）
       延遲 0 與 1 分鐘必須重現 `conj_entry` 的 +0.2927 / +0.2930
       （容差 0.02）。
    Y2 現行每小時管線能不能用
       批次延遲下、扣成本後的每事件值，日聚類 CI **下緣 > 0** -> 能用
       CI 含零或為負 -> **現行管線接不住這條線**，時鐘要不要繼續跑必須
       重新決定（不是自動繼續）
    Y3 需要多快
       找出「扣成本後 CI 下緣仍 > 0」的最大延遲。那個數字就是管線的規格。
    Y4 全格報告，不挑延遲。
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
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_clock as ck  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
DELAYS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
HOLD_MIN = 60
W = 5
SCEN = {"A 目標執行": 10, "B 全 taker": 13}
RNG = np.random.default_rng(20260907)


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    atrpct = {}
    for sym in ec.CORE9:
        cand, ts, cl, at, day = ck.frozen_cand(sym, liq)
        b = pd.read_parquet(BARS / f"{sym}.parquet", columns=["open", "close"])
        op = b["open"].to_numpy(float)
        n = len(ts)
        atrpct[sym] = float(np.nanmedian(at / np.where(cl > 0, cl, np.nan)))

        pairs = []
        for nm in et.NAMES:
            v = cand.get(nm)
            if v is None or len(v) == 0:
                continue
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))
        FLOW = {"delta_ext", "vol_burst", "oi_crash"}
        for a, sig in et.cluster(pairs):
            if not ("sweep" in sig and (sig & FLOW)):
                continue
            if a < W or a + HOLD_MIN + max(DELAYS) >= n:
                continue
            A = float(at[a])
            if not np.isfinite(A) or A <= 0:
                continue
            imp = np.sign(cl[a] - cl[a - W]) or 1.0
            exit_px = float(cl[a + HOLD_MIN])
            # 每小時批次：等到下一個整點還要幾分鐘
            minute_in_hour = int((int(ts[a]) % 3_600_000) // 60_000)
            d_batch = 60 - minute_in_hour
            row = dict(sym=sym, d_batch=int(d_batch),
                       day=pd.Timestamp(int(ts[a]), unit="ms",
                                        tz="UTC").strftime("%Y-%m-%d"))
            for dl in DELAYS:
                ent = float(cl[a]) if dl == 0 else float(op[a + dl])
                row[f"r{dl}"] = float(imp * (exit_px - ent) / A)
            q = a + d_batch
            row["r_batch"] = (float(imp * (exit_px - op[q]) / A)
                              if q < n else np.nan)
            rows.append(row)

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "conj_pipeline.parquet", index=False)

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

    days = d.day.to_numpy()
    w = d.groupby("sym").size()
    ap = float(sum(w[s] * atrpct[s] for s in w.index) / w.sum())
    cost = {k: bps / 1e4 / ap for k, bps in SCEN.items()}
    print("=== 交會事件：延遲多久還剩多少（單位 ATR，出場固定 close(t+60)）===")
    print(f"交會事件 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日   "
          f"加權 ATR% {ap*100:.3f}%   "
          f"成本 A {cost['A 目標執行']:.4f} / B {cost['B 全 taker']:.4f} ATR")
    print("**全格報告，不挑延遲。**")
    print()
    print(f"{'延遲':>6s} {'每事件':>9s} {'日聚類 CI95':>22s} {'相對 0':>7s} "
          f"{'淨 A':>9s} {'淨 A CI 下緣':>12s}")
    res = {"n": int(len(d)), "atr_pct": ap, "cost": cost, "delays": {}}
    base = None
    max_ok = None
    for dl in DELAYS:
        m, lo, hi = day_ci(d[f"r{dl}"].to_numpy(), days)
        if base is None:
            base = m
        netA = m - cost["A 目標執行"]
        netA_lo = lo - cost["A 目標執行"]
        if netA_lo > 0:
            max_ok = dl
        res["delays"][str(dl)] = dict(mean=m, ci=[lo, hi], rel=m / base,
                                      netA=netA, netA_lo=netA_lo)
        print(f"{dl:5d}m {m:+9.4f}  [{lo:+.4f},{hi:+.4f}] "
              f"{m/base*100:6.1f}% {netA:+9.4f} {netA_lo:+12.4f}")

    print()
    print("=== Y2 現行每小時批次管線實際交付什麼 ===")
    print()
    print(f"  批次延遲分布：中位 {d.d_batch.median():.0f} 分、"
          f"平均 {d.d_batch.mean():.1f} 分、p90 {d.d_batch.quantile(.9):.0f} 分")
    mb, lb, hb = day_ci(d.r_batch.to_numpy(), days)
    netb = mb - cost["A 目標執行"]
    netb_lo = lb - cost["A 目標執行"]
    print(f"  每事件 {mb:+.4f}  CI [{lb:+.4f}, {hb:+.4f}]   "
          f"相對零延遲 {mb/base*100:.1f}%")
    print(f"  扣成本 A：{netb:+.4f}   CI 下緣 {netb_lo:+.4f}")
    v2 = ("能用" if netb_lo > 0 else
          "**現行管線接不住這條線**")
    print(f"  -> {v2}")
    res["batch"] = dict(mean=mb, ci=[lb, hb], netA=netb, netA_lo=netb_lo,
                        median_delay=float(d.d_batch.median()), verdict=v2)

    print()
    print(f"Y3 管線規格：扣成本後 CI 下緣仍 > 0 的**最大延遲** = "
          f"{max_ok if max_ok is not None else '（沒有任何一格）'} 分鐘")
    res["Y3_max_delay_min"] = max_ok

    ref0, ref1 = 0.2927, 0.2930
    ok1 = (abs(res["delays"]["0"]["mean"] - ref0) < 0.02
           and abs(res["delays"]["1"]["mean"] - ref1) < 0.02)
    print(f"Y1 已知答案對照：0 分 {res['delays']['0']['mean']:+.4f}（{ref0:+.4f}）、"
          f"1 分 {res['delays']['1']['mean']:+.4f}（{ref1:+.4f}）"
          f"  -> {'PASS' if ok1 else '**FAIL — 兩份實作不同意，以上不解讀**'}")
    res["Y1"] = bool(ok1)

    (OUT / "conj_pipeline.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_pipeline.json")


if __name__ == "__main__":
    main()
