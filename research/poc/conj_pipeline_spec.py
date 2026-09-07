# -*- coding: utf-8 -*-
"""2 分鐘管線的規格 —— 哪些資料來得及，拿掉來不及的還剩多少

`conj_pipeline.py` 判出：可執行窗口 2 分鐘，現行每小時批次接不住。
這支回答「要做到 2 分鐘，需要改什麼」，而第一件事是**資料本身來不來得及**。

===========================================================================
逐一資料來源的延遲下界（不是估的，是資料本身的粒度決定的）
===========================================================================
    sweep（掃單）
        價位來自已確認的樞紐（前後各 10 根小時 K）——**可以事先算好**。
        偵測「價格穿過價位」只需要當下的分鐘價。
        -> 延遲下界 ≈ **1 根分鐘 K**（收盤即知）

    delta_ext（主動量極端）
        |delta| 的 5 分鐘後向和。delta 來自 1 分鐘 kline 的 field[9]
        （taker buy base，交易所原生標記）。滾動 30 日門檻可**每日預算**。
        -> 延遲下界 ≈ **1 根分鐘 K**

    vol_burst（量能爆發）
        5 分鐘量 / 前 30 日同時段均值。同樣來自 1 分鐘 kline，基準可預算。
        -> 延遲下界 ≈ **1 根分鐘 K**

    oi_crash（OI 崩落）
        來自 Binance `metrics`，**粒度就是 5 分鐘**；而且 `event_census`
        檔頭記載的前視修正要求只用 `create_time <= t − 5min` 的列
        （那一列不是 create_time 當下的快照，帶著整個區間的資訊）。
        -> 延遲下界 ≈ **5–10 分鐘**，**結構上不可能壓進 2 分鐘**

所以問題變成一句話：**把 oi_crash 拿掉，還剩多少交會、效應剩多少？**

本檔全格報告三種簽名母體：
    ALL      現行定義：sweep ∧ (delta_ext ∨ vol_burst ∨ oi_crash)
    NO-OI    只用分鐘級資料：sweep ∧ (delta_ext ∨ vol_burst)
    OI-ONLY  只有 OI 那一支：sweep ∧ oi_crash ∧ ¬(delta_ext ∨ vol_burst)

並在每個母體上重跑延遲曲線（0/1/2/3/5 分），因為母體換了效應也會換。

===========================================================================
判準（跑之前寫死）
===========================================================================
    S1 已知答案的對照（**必須**過，否則以下不解讀）
       ALL 母體在延遲 0 與 2 分鐘要重現 `conj_pipeline` 的
       +0.2927 / +0.2064（容差 0.02）。
    S2 NO-OI 母體在 **2 分鐘延遲、扣成本後**的日聚類 CI 下緣 > 0
       -> **2 分鐘管線做得到，而且不需要 OI**，規格成立
       CI 含零或為負 -> 拿掉 OI 就不成立，2 分鐘管線救不了這條線
    S3 覆蓋率要報：NO-OI 佔 ALL 的多少事件。若 < 50%，
       要明講「這是換了母體不是改管線」。
    S4 全格報告三個母體 × 五個延遲，不挑。
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
DELAYS = [0, 1, 2, 3, 5]
HOLD_MIN = 60
W = 5
SCEN = {"A 目標執行": 10, "B 全 taker": 13}
RNG = np.random.default_rng(20260907)
MINUTE = {"delta_ext", "vol_burst"}     # 只需要 1 分鐘 kline
SLOW = {"oi_crash"}                     # 5 分鐘粒度，結構上壓不進 2 分鐘


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
        FLOW = MINUTE | SLOW
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
            row = dict(sym=sym,
                       has_minute=bool(sig & MINUTE),
                       has_oi=bool(sig & SLOW),
                       day=pd.Timestamp(int(ts[a]), unit="ms",
                                        tz="UTC").strftime("%Y-%m-%d"))
            for dl in DELAYS:
                ent = float(cl[a]) if dl == 0 else float(op[a + dl])
                row[f"r{dl}"] = float(imp * (exit_px - ent) / A)
            rows.append(row)

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "conj_pipeline_spec.parquet", index=False)

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

    w = d.groupby("sym").size()
    ap = float(sum(w[s] * atrpct[s] for s in w.index) / w.sum())
    costA = SCEN["A 目標執行"] / 1e4 / ap
    costB = SCEN["B 全 taker"] / 1e4 / ap

    pops = {
        "ALL 現行定義": np.ones(len(d), bool),
        "NO-OI 只用分鐘級": d.has_minute.to_numpy(bool),
        "OI-ONLY 只有 OI": (d.has_oi & ~d.has_minute).to_numpy(bool),
    }
    print("=== 2 分鐘管線規格：拿掉來不及的資料還剩多少 ===")
    print(f"交會事件 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日   "
          f"ATR% {ap*100:.3f}%   成本 A {costA:.4f} / B {costB:.4f} ATR")
    print()
    print(f"{'母體':20s} {'n':>6s} {'佔 ALL':>7s}   "
          + "".join(f"{str(x) + 'm':>10s}" for x in DELAYS))
    res = {"n": int(len(d)), "atr_pct": ap, "costA": costA, "pops": {}}
    for nm, sel in pops.items():
        g = d[sel]
        if len(g) < 30:
            print(f"{nm:20s} {len(g):6,d}  (n<30)")
            continue
        cells, cell_res = [], {}
        for dl in DELAYS:
            m, lo, hi = day_ci(g[f"r{dl}"].to_numpy(), g.day.to_numpy())
            cell_res[str(dl)] = dict(mean=m, ci=[lo, hi],
                                     netA=m - costA, netA_lo=lo - costA)
            cells.append(f"{m:+.4f}")
        res["pops"][nm] = dict(n=int(len(g)), share=len(g) / len(d),
                               delays=cell_res)
        print(f"{nm:20s} {len(g):6,d} {len(g)/len(d)*100:6.2f}%   "
              + "".join(f"{c:>10s}" for c in cells))

    print()
    print("=== 各母體在 2 分鐘延遲、扣成本 A 之後 ===")
    print()
    print(f"{'母體':20s} {'每事件':>9s} {'CI95':>22s} {'淨 A':>9s} {'淨 A CI 下緣':>12s}")
    for nm in pops:
        if nm not in res["pops"]:
            continue
        c = res["pops"][nm]["delays"]["2"]
        print(f"{nm:20s} {c['mean']:+9.4f}  "
              f"[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"{c['netA']:+9.4f} {c['netA_lo']:+12.4f}")

    print()
    print("=== 預註冊判準 ===")
    print()
    a0 = res["pops"]["ALL 現行定義"]["delays"]["0"]["mean"]
    a2 = res["pops"]["ALL 現行定義"]["delays"]["2"]["mean"]
    ok1 = abs(a0 - 0.2927) < 0.02 and abs(a2 - 0.2064) < 0.02
    print(f"S1 已知答案對照：ALL 0 分 {a0:+.4f}（+0.2927）、"
          f"2 分 {a2:+.4f}（+0.2064）-> "
          f"{'PASS' if ok1 else '**FAIL — 兩份實作不同意，以上不解讀**'}")

    nk = "NO-OI 只用分鐘級"
    if nk in res["pops"]:
        c = res["pops"][nk]["delays"]["2"]
        v2 = ("**成立 —— 2 分鐘管線做得到，而且不需要 OI**"
              if c["netA_lo"] > 0 else
              "**不成立 —— 拿掉 OI 就撐不住，2 分鐘管線救不了這條線**")
        print(f"S2 NO-OI 在 2 分鐘、扣成本 A：{c['netA']:+.4f}  "
              f"CI 下緣 {c['netA_lo']:+.4f}  -> {v2}")
        sh = res["pops"][nk]["share"]
        v3 = ("PASS" if sh >= 0.50
              else "**覆蓋率 <50%，這是換了母體不是改管線**")
        print(f"S3 NO-OI 覆蓋率 {sh*100:.1f}%  -> {v3}")
        res["S2"] = v2
        res["S3"] = dict(share=sh, verdict=v3)
    res["S1"] = bool(ok1)

    (OUT / "conj_pipeline_spec.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_pipeline_spec.json")


if __name__ == "__main__":
    main()
