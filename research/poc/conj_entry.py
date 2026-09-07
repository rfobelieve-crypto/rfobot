# -*- coding: utf-8 -*-
"""交會事件的進場設計 + **同時**做執行檢查（不等訊號驗完）

使用者 2026-09-07：「那先做 2」。

為什麼執行檢查要跟進場設計同時做，而不是等時鐘跑完
    掃單失敗那條線的教訓（TODO §1.02）：訊號驗了兩年半、前瞻時鐘跑了一年，
    最後發現**進場價市場不給**，+0.0366 R 全部是記帳。那個缺陷在第一天就
    量得出來，只是沒有人去量。**這次先量。**

===========================================================================
候選進場規則（沿用 `conj_clock.py` 的凍結事件定義，一個字不動）
===========================================================================
    事件：交會時刻 t（簽名同時含 sweep 與至少一個 {delta_ext, vol_burst,
          oi_crash}，因果門檻、併窗 5 分、冷卻 60 分、錨點取群內最早那分鐘）
    方向：impulse = sign(close(t) − close(t−5m))   **只用 t 之前**
    這是**延續**交易（順著 impulse），與掃單失敗的反轉方向相反。

    訊號在 close(t) 才成立（所有事件定義都用 ≤ t 的資料），所以**最早的
    可成交價是 open(t+1)**。「在 close(t) 成交」是不可能的——那正是掃單線
    死掉的那種假設，本檔不用它當結果，只留作參考基準。

進場路徑（全部報告，不挑）
    P0  close(t)         **參考基準，不可實現**（事件研究用的就是它）
    P1  open(t+1)        訊號成立後第一個可成交價
    P2  open(t+2)
    P3  open(t+3)
    P5  open(t+5)
    P10 open(t+10)       模擬「偵測→決策→送單」較慢的管線
    L   限價掛在 close(t)，窗內未成交就不進場（成交率另計）

出場：固定在 close(t+60)。**進場越晚，價格越差而且持有越短**——兩個效應
      都算進去，這才是真實的。另外並報「固定持有 60 分鐘」的版本。

成本：凍結成本模型（2026-07-28 定，未調參）
      情境 A 目標執行 10 bps ／ 情境 B 全 taker 13 bps
      cost_ATR = bps/1e4 / ATR%（本檔的效應是 **ATR 單位**，不再除 DIS
      —— 單位不同公式就不同，見 `exec_ladder` 檔頭記載的那個錯）

===========================================================================
判準（跑之前寫死）
===========================================================================
    X1 全格報告，每條路徑的每事件值與日聚類 CI。**不挑延遲。**
    X2 扣成本後，哪些路徑的 CI 下緣仍 > 0
    X3 已知答案的對照（**必須**過，否則以下不解讀）
       P0 必須重現 `TRIAGE.md` 因果門檻版的 **+0.3485**（容差 0.02）。
       重現不出來代表我又寫了第二份實作而且它們不同意。
    X4 衰減速度（本檔真正要回答的）
       P1 相對 P0 掉多少？若一分鐘的延遲就吃掉一半以上，這條線與
       §0.98「行情在確認出現之前就走完了」是同一個形狀，**要當場講明**。
    X5 限價路徑的成交率與逆選擇一併報，不挑對自己有利的。

**全部 in-sample。** 就算全部為正也不得採用——`conj_clock.py` 的前瞻時鐘
（0/300，約四個月）才是判準。本檔只回答「**可執行性的上限在哪**」。
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
DELAYS = [0, 1, 2, 3, 5, 10]
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
                continue                      # 只取交會
            if a < W or a + HOLD_MIN + max(DELAYS) >= n:
                continue
            A = float(at[a])
            if not np.isfinite(A) or A <= 0:
                continue
            imp = np.sign(cl[a] - cl[a - W]) or 1.0
            exit_px = float(cl[a + HOLD_MIN])
            row = dict(sym=sym, atr=A,
                       day=pd.Timestamp(int(ts[a]), unit="ms",
                                        tz="UTC").strftime("%Y-%m-%d"))
            for dl in DELAYS:
                ent = float(cl[a]) if dl == 0 else float(op[a + dl])
                # 固定出場時刻 close(t+60)：進場越晚，價格越差且持有越短
                row[f"r{dl}"] = float(imp * (exit_px - ent) / A)
                # 固定持有 60 分鐘（並報）
                q = a + dl + HOLD_MIN
                row[f"h{dl}"] = (float(imp * (cl[q] - ent) / A)
                                 if q < n else np.nan)
            # 限價路徑：掛在 close(t)，60 分鐘窗內回到該價才成交
            lp = float(cl[a])
            seg_lo = np.min(np.minimum(op[a + 1:a + 1 + HOLD_MIN],
                                       cl[a + 1:a + 1 + HOLD_MIN]))
            seg_hi = np.max(np.maximum(op[a + 1:a + 1 + HOLD_MIN],
                                       cl[a + 1:a + 1 + HOLD_MIN]))
            # 順勢延續：impulse>0 -> 做多 -> 掛買單在 lp，價格要跌回來
            filled = (seg_lo <= lp) if imp > 0 else (seg_hi >= lp)
            row["l_filled"] = int(bool(filled))
            row["r_limit"] = float(imp * (exit_px - lp) / A) if filled else np.nan
            rows.append(row)

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "conj_entry.parquet", index=False)

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
    print("=== 交會事件的進場設計 + 執行檢查（單位 ATR）===")
    print(f"交會事件 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日、"
          f"{d.sym.nunique()} 幣   加權 ATR% {ap*100:.3f}%")
    print("**全格報告，不挑延遲。** 出場固定 close(t+60)。")
    print()
    print(f"{'進場':22s} {'每事件':>9s} {'日聚類 CI95':>22s} "
          f"{'逐幣正':>7s} {'相對 P0':>8s}   "
          + "".join(f"{'淨 ' + k:>13s}" for k in SCEN))
    res = {"n": int(len(d)), "atr_pct": ap, "paths": {}}
    base = None
    for dl in DELAYS:
        col = f"r{dl}"
        m, lo, hi = day_ci(d[col].to_numpy(), days)
        if base is None:
            base = m
        pc = d.groupby("sym")[col].mean()
        nm = ("P0 close(t) **不可實現**" if dl == 0
              else f"P{dl} open(t+{dl})")
        nets = {k: m - bps / 1e4 / ap for k, bps in SCEN.items()}
        res["paths"][nm] = dict(mean=m, ci=[lo, hi], coins_pos=int((pc > 0).sum()),
                                rel=m / base if base else np.nan, net=nets)
        print(f"{nm:22s} {m:+9.4f}  [{lo:+.4f},{hi:+.4f}] {int((pc>0).sum()):>6d}/9 "
              f"{m/base*100 if base else float('nan'):7.1f}%   "
              + "".join(f"{nets[k]:+13.4f}" for k in SCEN))

    print()
    print("並報：固定持有 60 分鐘（出場 = 進場後 60 分）")
    for dl in DELAYS:
        m, lo, hi = day_ci(d[f"h{dl}"].to_numpy(), days)
        print(f"  {'P' + str(dl):5s} {m:+9.4f}  [{lo:+.4f},{hi:+.4f}]")

    fr = float(d.l_filled.mean())
    ml, ll, hl = day_ci(d.r_limit.to_numpy(), days)
    print()
    print(f"X5 限價路徑（掛在 close(t)，60 分鐘窗內回到該價才成交）")
    print(f"  成交率 {fr*100:.2f}%   E[R|成交] {ml:+.4f} [{ll:+.4f},{hl:+.4f}]"
          f"   每事件 {ml*fr:+.4f}")
    res["limit"] = dict(fill=fr, e_r=ml, ci=[ll, hl], per_event=ml * fr)

    print()
    print("=== 預註冊判準 ===")
    print()
    ref = 0.3485
    p0 = res["paths"]["P0 close(t) **不可實現**"]["mean"]
    ok3 = abs(p0 - ref) < 0.02
    print(f"X3 已知答案對照：P0 {p0:+.4f} vs TRIAGE 因果門檻版 {ref:+.4f}"
          f"  差 {abs(p0-ref):.4f}（需 <0.02）-> "
          f"{'PASS' if ok3 else '**FAIL — 兩份實作不同意，以下不解讀**'}")
    p1 = res["paths"]["P1 open(t+1)"]["mean"]
    decay = 1 - p1 / p0 if p0 else np.nan
    v4 = ("**一分鐘就吃掉一半以上 —— 與 §0.98「行情在確認出現之前就走完了」"
          "同形狀**" if decay > 0.5 else "一分鐘的延遲沒有吃掉一半")
    print(f"X4 衰減：P1 相對 P0 掉 {decay*100:.1f}%  -> {v4}")
    res["X3"] = bool(ok3)
    res["X4"] = dict(decay=decay, verdict=v4)

    alive = [k for k, v in res["paths"].items()
             if not k.startswith("P0") and
             min(v["net"][s] for s in SCEN) > 0]
    print()
    print(f"X2 扣成本後兩種情境都為正的可實現路徑：{alive if alive else '（無）'}")
    res["X2_alive"] = alive

    (OUT / "conj_entry.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_entry.json")
    print()
    print("**全部 in-sample。前瞻時鐘 conj_clock.py（0/300）才是判準。**")
    print("**本檔只回答可執行性的上限在哪，不得據此採用任何路徑。**")


if __name__ == "__main__":
    main()
