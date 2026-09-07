# -*- coding: utf-8 -*-
"""掃單失敗：在**穿越的那一分鐘**進場——四條路徑沒問過的那一格

===========================================================================
為什麼是這一格
===========================================================================
§1.02 判掉掃單線時比較了四種進場，全部為負：

    凍結（幻影價）      +0.0366   不可得（下一根小時 K 記「成交在價位」，
                                  但 44.72% 的成交那根 K 從沒碰過價位）
    掃單 bar 收盤市價   -0.0008 / 扣成本 -0.0293
    收回確認後進        -0.0495
    掛限價在價位等      -0.0504

**四條的共同點：全部發生在掃單那根小時 K 收盤「之後」，或等價格回到價位。**
而判決自己寫下的機制是「回踩確實發生了，只是發生在那一根之內」。

沒有一條問：**穿越發生的那一分鐘**。那一分鐘價格在價位**外側**（穿越的
定義），而這是 fade 單——buyside 掃單（向上刺穿前高）做空，進場價越高越好。
它真實存在、當下可偵測（`conj_watch.py` 每分鐘做的正是「穿越了哪個活價位」）、
不需要等任何確認（§0.98 已四度證明「等確認」必敗）。

===========================================================================
設計：只換進場價，其餘一個字不動
===========================================================================
**第一版的對照設計是錯的，記在這裡**（與 `conj_watch_parity` 第一版同病）：
我拿「全部 12,503 個事件、從穿越分鐘持有 480 分」去對「凍結的 7,083 筆、
從回踩那根小時 K 持有 8 小時」——兩個不同母體、不同持有窗。那個 M1
不管實作對錯都不可能過（實測幻影臂 -0.0599 vs 凍結 +0.0366），**它沒有
分辨力**，所以它底下的每一格都不能解讀。

現在的做法：直接吃 `sweep_core.backtest_symbol(detail=True)` 吐出來的
**凍結交易本身**，逐筆只把進場價換掉：

    母體      凍結交易（含它的非重疊、回踩濾網、W/HOLD，全部照舊）
    出場時刻  凍結的 exit_ts，**不動**
    風險單位  凍結的 risk = DIS x ATR，**不動**
    停損      entry - d x risk（entry 換了，停損跟著平移，這是規則本身）
    只換的    entry 價格：幻影 lvl -> 穿越那一分鐘的收盤

    穿越分鐘  掃單那根小時 K 之內，第一根 high > lvl（buy）/ low < lvl（sell）
              的分鐘。**穿越測試不是水準測試**——這正是 §1.02 判掉主線的
              那個錯，也是我 2026-09-07 在 conj_watch 又寫了一次的那個錯。

停損以**分鐘**高低價判定（進場變早了，中間那段必須用分鐘資料看），
出場價一律用凍結 exit_ts 那一分鐘的收盤。

成本   凍結模型 A 10 bps / B 13 bps，逐幣真實 bps。
       cost_R = bps/1e4 / (DIS x ATR%)  <- 除以 DIS，單位是 R
       （`exec_ladder` 檔頭記載過漏掉 DIS 會把成本高估 3.5 倍）

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
M1  **已知答案對照，必須先過，否則以下一律不解讀。**
    用凍結的 entry 價跑同一台重算機器，meanR 必須落在凍結 meanR 的
    ±0.005 之內。對不上代表重算機器寫錯了。
M2  穿越分鐘進場、**扣成本 A** 之後，日聚類 bootstrap CI 下緣 > 0
    -> 這條線在分鐘解析度上可執行，§1.02 的結案要加註
M3  逐幣 >= 6/9 為正（與 Gate F 同門檻）
M4  全格報告延遲 0/1/2/5/10 分 x 兩個成本情境，不挑格
M5  報「穿越分鐘 vs 幻影價」的價差分佈——若進場價其實比幻影價**差**，
    本檔的整個前提就錯了，要當場說出來
M6  **無濾網臂（決定性的那一關）**。上面的母體是**凍結交易**，而凍結交易
    的入選條件是「8 根小時 K 內發生過回踩」—— 那是掃單當下**還不知道**的事。
    在穿越分鐘進場的人不可能先知道未來會不會回踩，所以那個母體帶前視。
    無濾網臂：`detect_sweeps` 的**每一個**掃單都在穿越分鐘進場，
    出場 = 停損 或 進場後 480 分鐘，兩者先到。
    這一臂才是可執行的那一個；它的扣成本 CI 下緣 > 0 才算 M2 真的過。
    自帶對照：把無濾網臂限制在凍結交易那個子集上，必須重現上面的數字。
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
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402

BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
DELAYS = [0, 1, 2, 5, 10]
SCEN = {"A 目標執行": 10, "B 全 taker": 13}
RNG = np.random.default_rng(20260907)
TOL = 0.005
HOUR_MS = 3_600_000


def collect(sym):
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None, None
    bars = sc.load_csv(str(p))
    det = sc.backtest_symbol(bars, detail=True)
    if not det:
        return None, None

    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close",
                                 "atr_h14"])
    mts = b["ts"].to_numpy(np.int64)
    mop = b["open"].to_numpy(float)
    mhi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    mlo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    mcl = b["close"].to_numpy(float)
    n = len(mts)
    atrpct = float(np.nanmedian(
        b["atr_h14"].to_numpy(float) / np.where(mcl > 0, mcl, np.nan)))

    # 小時 K 快取的時間戳是**秒**，分鐘 parquet 是**毫秒**
    # （2026-09-07 levels_asof 因此靜默失效過，見 mistake.md）
    def to_ms(t):
        t = int(t)
        return t * 1000 if t < 1_000_000_000_000 else t

    rows = []
    for t in det:
        sw = to_ms(t["sweep_ts"])
        ex = to_ms(t["exit_ts"])
        lvl, A, d = float(t["level"]), float(t["atr"]), float(t["d"])
        risk = sc.DIS * A
        if not np.isfinite(A) or A <= 0:
            continue
        s0 = int(np.searchsorted(mts, sw))
        s1 = int(np.searchsorted(mts, sw + HOUR_MS))
        # **小時 K 的時間戳是開盤標籤，不是時點**：exit_ts 那根的收盤在 +1h。
        # 第一版用 exit_ts 那一分鐘的收盤出場 = 提早一小時平倉，重算 +0.0997
        # vs 凍結 +0.0366（三倍，正是「數字大到不合理先查儀器」擋下來的）。
        # 這是 mistake.md 2026-09-03 那條在本 session 的第三個實例。
        xi = int(np.searchsorted(mts, ex + HOUR_MS)) - 1
        if s0 >= s1 or xi >= n or s1 > xi:
            continue
        # 穿越測試：掃單那根小時 K 之內，第一根真的越過價位的分鐘
        seg = slice(s0, s1)
        if d == -1.0:                       # buy 掃單（向上刺穿）-> SHORT
            hit = np.flatnonzero(mhi[seg] > lvl)
        else:                               # sell 掃單 -> LONG
            hit = np.flatnonzero(mlo[seg] < lvl)
        if not len(hit):
            continue                        # 分鐘資料沒覆蓋到這根小時 K
        pm = s0 + int(hit[0])               # 穿越的那一分鐘

        row = {"sym": sym, "day": pd.Timestamp(int(mts[pm]), unit="ms",
                                               tz="UTC").strftime("%Y-%m-%d"),
               "R_frozen": float(t["R"]) if "R" in t else np.nan}
        # M1 對照臂：用凍結的 entry 價、凍結的出場時刻，走同一台重算機器
        for tag, ent, j0 in (("m1", float(t["entry"]),
                              int(np.searchsorted(mts, to_ms(t["fill_ts"])))),
                             *[(f"r{dl}", None, pm + dl) for dl in DELAYS]):
            if j0 >= xi or j0 >= n:
                row[tag] = np.nan
                continue
            if ent is not None:
                entry = ent               # 凍結的 entry 已含進場滑價
            else:
                px = float(mcl[pm]) if j0 == pm else float(mop[j0])
                entry = px + d * sc.SLIP * A      # 市價單的不利滑價，同凍結形式
            stop = entry - d * risk
            sl = slice(j0 + 1, xi + 1)
            hitk = (np.flatnonzero(mlo[sl] <= stop) if d == 1
                    else np.flatnonzero(mhi[sl] >= stop))
            # 出場也付不利滑價 —— 凍結是 `ex = c[exitbar] - d*SLIP*A`。
            # 漏掉它會讓重算比凍結好 SLIP/DIS = 0.0143 R，正是 M1 第二版
            # 那個 +0.0096 的缺口。**任何宣稱含成本的重算，兩條腿都要在。**
            exq = float(mcl[xi]) - d * sc.SLIP * A
            row[tag] = (-1.0 - sc.SLIP / sc.DIS if len(hitk)
                        else float(d * (exq - entry) / risk))
        row["edge_vs_phantom"] = float(d * (float(mcl[pm]) - lvl) / risk)
        rows.append(row)

    # ---- M6 無濾網臂：每一個掃單都進場，不問未來會不會回踩 ----
    at1 = sc.atr14(bars)
    h1 = [x[sc.H] for x in bars]
    l1 = [x[sc.L] for x in bars]
    ts1 = np.array([to_ms(x[0]) for x in bars], np.int64)
    nf = []
    for e in sc.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        A = at1[j]
        if A is None or A == 0:
            continue
        A = float(A)
        d = -1.0 if e["kind"] == "buy" else 1.0
        risk = sc.DIS * A
        s0 = int(np.searchsorted(mts, int(ts1[j])))
        s1 = int(np.searchsorted(mts, int(ts1[j]) + HOUR_MS))
        if s0 >= s1:
            continue
        seg = slice(s0, s1)
        hit = (np.flatnonzero(mhi[seg] > lvl) if d == -1.0
               else np.flatnonzero(mlo[seg] < lvl))
        if not len(hit):
            continue
        pm = s0 + int(hit[0])
        row = {"sym": sym, "day": pd.Timestamp(int(mts[pm]), unit="ms",
                                               tz="UTC").strftime("%Y-%m-%d")}
        for dl in DELAYS:
            j0 = pm + dl
            end = j0 + 8 * 60                # 與凍結 HOLD=8 根小時 K 同牆鐘長
            if end >= n:
                row[f"r{dl}"] = np.nan
                continue
            px = float(mcl[pm]) if dl == 0 else float(mop[j0])
            entry = px + d * sc.SLIP * A
            stop = entry - d * risk
            sl = slice(j0 + 1, end + 1)
            hitk = (np.flatnonzero(mlo[sl] <= stop) if d == 1
                    else np.flatnonzero(mhi[sl] >= stop))
            exq = float(mcl[end]) - d * sc.SLIP * A
            row[f"r{dl}"] = (-1.0 - sc.SLIP / sc.DIS if len(hitk)
                             else float(d * (exq - entry) / risk))
        nf.append(row)
    return pd.DataFrame(rows), atrpct, pd.DataFrame(nf)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return (float("nan"),) * 3
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == m)[0] for m in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([ix[m] for m in p])].mean()
    return (float(x.mean()), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)))


def main():
    frames, ap, froz, nfs = [], {}, [], []
    for sym in ec.CORE9:
        d, a, nf = collect(sym)
        if d is None or not len(d):
            continue
        frames.append(d)
        nfs.append(nf)
        ap[sym] = a
        bars = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        froz += [t[2] for t in sc.backtest_symbol(bars)]
    d = pd.concat(frames, ignore_index=True)
    nf = pd.concat(nfs, ignore_index=True)
    frozen_mean = float(np.mean(froz))

    w = d.groupby("sym").size()
    apw = float(sum(w[s] * ap[s] for s in w.index) / w.sum())
    cost = {k: bps / 1e4 / (sc.DIS * apw) for k, bps in SCEN.items()}

    print("=== 掃單失敗：在穿越的那一分鐘進場（只換進場價）===")
    print(f"凍結交易 {len(froz):,} 筆 -> 分鐘資料覆蓋得到的 {len(d):,} 筆、"
          f"{d.day.nunique():,} 個 UTC 日")
    print(f"加權 ATR% {apw*100:.3f}%   成本（R）A {cost['A 目標執行']:.4f} / "
          f"B {cost['B 全 taker']:.4f}")
    print()

    m1, m1lo, m1hi = day_ci(d["m1"].to_numpy(), d.day.to_numpy())
    ok1 = abs(m1 - frozen_mean) < TOL
    print("=== M1 已知答案對照（用凍結的 entry 價跑同一台重算機器）===")
    print(f"  重算 {m1:+.4f}   凍結全樣本 {frozen_mean:+.4f}   容差 {TOL}")
    print(f"  -> {'PASS' if ok1 else '**FAIL —— 重算機器寫錯了，以下不解讀**'}")
    print()

    e = d["edge_vs_phantom"]
    print("=== M5 穿越分鐘進場價 vs 價位（R，正 = 對 fade 更有利）===")
    print(f"  中位 {e.median():+.4f}   平均 {e.mean():+.4f}   "
          f"為正 {(e > 0).mean()*100:.1f}%")
    print("  -> " + ("前提成立：穿越分鐘給出比價位更有利的進場"
                     if e.median() > 0 else
                     "**前提錯誤**：穿越分鐘並沒有比價位有利"))
    print()

    print("=== M4 全格報告（不挑格）===")
    print(f"{'延遲':>5s} {'零成本':>9s} {'日聚類 CI95':>22s} "
          f"{'淨 A':>9s} {'淨 A 下緣':>10s} {'淨 B':>9s} {'幣 +':>5s}")
    res = {"n": int(len(d)), "n_frozen": len(froz), "atr_pct": apw,
           "cost": cost, "frozen_mean": frozen_mean,
           "M1": {"recomputed": m1, "ok": bool(ok1)},
           "M5_edge_vs_phantom_median": float(e.median()), "delays": {}}
    for dl in DELAYS:
        col = d[f"r{dl}"].to_numpy()
        m, lo, hi = day_ci(col, d.day.to_numpy())
        per = d.groupby("sym")[f"r{dl}"].mean() - cost["A 目標執行"]
        npos = int((per > 0).sum())
        res["delays"][str(dl)] = dict(
            mean=m, ci=[lo, hi], netA=m - cost["A 目標執行"],
            netA_lo=lo - cost["A 目標執行"],
            netB=m - cost["B 全 taker"], coins_pos=npos)
        print(f"{dl:4d}m {m:+9.4f}  [{lo:+.4f},{hi:+.4f}] "
              f"{m-cost['A 目標執行']:+9.4f} "
              f"{lo-cost['A 目標執行']:+10.4f} "
              f"{m-cost['B 全 taker']:+9.4f} {npos:4d}/9")

    print()
    print("=== 判準 ===")
    c0 = res["delays"]["0"]
    ok2 = ok1 and c0["netA_lo"] > 0
    ok3 = ok1 and c0["coins_pos"] >= 6
    print(f"M2 CI 下緣 {c0['netA_lo']:+.4f} > 0 ? -> "
          f"{'**PASS —— 分鐘解析度上可執行**' if ok2 else 'FAIL'}")
    print(f"M3 逐幣 {c0['coins_pos']}/9 >= 6 ? -> {'PASS' if ok3 else 'FAIL'}")
    print()
    if not ok1:
        print("M1 沒過 -> 以上不構成任何判決。")
    elif ok2 and ok3:
        print("**§1.02 的結案要加註：它測的是四種小時級進場；"
              "分鐘級的穿越進場沒被測過，而且是正的。**")
    else:
        print("這一格也沒救 —— §1.02「訊號有效但交易設計不可執行」的結案確定，"
              "而且現在連分鐘解析度的穿越進場都試過了。")

    # ---- M6 無濾網臂 ----
    print()
    print("=== M6 無濾網臂：每一個掃單都在穿越分鐘進場（沒有回踩濾網）===")
    print(f"  掃單事件 {len(nf):,} 筆（凍結交易 {len(d):,} 筆是它的子集，"
          f"入選條件用到未來資訊）")
    print(f"{'延遲':>5s} {'零成本':>9s} {'日聚類 CI95':>22s} "
          f"{'淨 A':>9s} {'淨 A 下緣':>10s} {'幣 +':>5s}")
    res["nofilter"] = {"n": int(len(nf)), "delays": {}}
    for dl in DELAYS:
        m, lo, hi = day_ci(nf[f"r{dl}"].to_numpy(), nf.day.to_numpy())
        per = nf.groupby("sym")[f"r{dl}"].mean() - cost["A 目標執行"]
        npos = int((per > 0).sum())
        res["nofilter"]["delays"][str(dl)] = dict(
            mean=m, ci=[lo, hi], netA=m - cost["A 目標執行"],
            netA_lo=lo - cost["A 目標執行"], coins_pos=npos)
        print(f"{dl:4d}m {m:+9.4f}  [{lo:+.4f},{hi:+.4f}] "
              f"{m-cost['A 目標執行']:+9.4f} "
              f"{lo-cost['A 目標執行']:+10.4f} {npos:4d}/9")
    c6 = res["nofilter"]["delays"]["0"]
    ok6 = ok1 and c6["netA_lo"] > 0 and c6["coins_pos"] >= 6
    print()
    print(f"M6 無濾網、扣成本 A 的 CI 下緣 {c6['netA_lo']:+.4f} > 0 "
          f"且逐幣 {c6['coins_pos']}/9 >= 6 ? -> "
          + ("**PASS —— 這條線在分鐘解析度上真的可執行**" if ok6 else
             "FAIL —— 上面那個正數是回踩濾網的存活偏誤"))
    res["M6"] = bool(ok6)

    res["M2"], res["M3"] = bool(ok2), bool(ok3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pierce_entry.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    d.to_parquet(OUT / "pierce_entry.parquet", index=False)
    print()
    print("written ->", OUT / "pierce_entry.json")


if __name__ == "__main__":
    main()
