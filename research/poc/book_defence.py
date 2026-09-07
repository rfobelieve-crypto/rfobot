# -*- coding: utf-8 -*-
"""掃單 A/B 的判別器候選：**該吸收它的那一側在補單還是在撤單**

===========================================================================
為什麼是這個量（以及我上一版把機制講錯了什麼）
===========================================================================
六條進場路徑 + 48 格出場結構全部測完之後，掃單線只剩一個開著的問題：

    進場價     分鐘解析度已解決（-0.085 R -> -0.0067 R）
    出場結構   48 格全負，零成本下最好的一格 -0.0119 ATR
    **A/B 選擇  唯一還開著的**

而 A/B 的賠率是 B +0.1001 R / A -0.2181 R，扣成本要 **B% >= 77.4%**；
現況基準 49.6%，最好的既有判別器（強制流旗標）只給 6.7 pp。

**更正一個我自己說錯的機制**：我先前說「掛單簿理論上最對——A/B 字面上就是
價位外側有沒有掛單」。`depth_deltas_1m` **量不到那個**——它只有每分鐘的
`bid/ask_add_qty`、`bid/ask_cancel_qty`，是**掛撤流量**，沒有價格分層、
沒有靜態簿深。庫裡沒有「價位外側還剩多少掛單」這個量。
（同一個毛病 mistake.md 2026-09-07 已經記過一次：把儀器量不到的機制
講成它在量的東西。）

它**能**量的是另一個機制，同樣說得通：
**掃單當下，該吸收它的那一側是在補單還是在撤單。**

    buyside 掃單（價格向上刺穿前高）-> 吸收方 = ASK（賣方）
        ask 在補單 -> 有人守 -> 反轉（B）
        ask 在撤單 -> 沒人守 -> 續走（A）
    sellside 掃單 -> 吸收方 = BID，對稱

**先驗是不利的**：這個 repo 的撤單流研究（策略 #3）方向性判決 FAIL——
`cancel_lead_ic` 四個 horizon 全滅，唯一活著的是「撤單強度 -> 波動」。
所以這裡不是「照著一個成功的線往下做」，是「用一個已知沒有 4h 方向性的
資料，去問一個它可能更適合的短程、事件條件化的問題」。

===========================================================================
資料與功效（凍結判準前先算，mistake.md 2026-09-04）
===========================================================================
覆蓋 7/9 幣（缺 SOL、AVAX）、46-60 天 -> 窗內掃單 **n = 592**（全樣本 4.6%）
中位切兩半，每組 ~296 -> 兩比例差 SE ≈ 4.1 pp，日聚類放大後 MDE ≈ 10.5 pp
需要偵測的量級 ~20 pp -> **這個設計有測量能力**（SE < 門檻，不是地形扳機那種病）

===========================================================================
規則（跑之前寫死）
===========================================================================
特徵    defence = (add - cancel) / (add + cancel)，只取**吸收方**那一側，
        窗口 [穿越分鐘 - 4, 穿越分鐘]（5 分鐘，**全部嚴格 <= 決策時刻**，
        不需要等待）。跨交易所加總。
分組    **中位切兩半**。不掃門檻——「看過哪一半贏再去找切點」正是
        mistake.md 2026-06-20 擋的事。
對照    同一條算式套在**另一側**（不該有效應的那一側）當安慰劑，
        比照 §0.99 用 F0 當對照的做法。安慰劑若也顯著，這個效應是
        某個共同因子（波動／活躍度）的翻版，不是機制。

判準
    D1  已知答案對照：窗內子樣本的 B 率必須落在全樣本 49.6% 的 ±5 pp 內。
        偏掉代表這 592 筆不代表母體，以下不解讀。
    D2  高 defence 組的 B% - 低 defence 組的 B% > 0（方向事先寫死：
        補單 -> 反轉），且日聚類 bootstrap CI 不含零。
    D3  逐幣 >= 5/7 同號。
    D4  安慰劑（另一側）的差必須**不顯著**。顯著就是共同因子。
    D5  同時報經濟量：兩組在穿越分鐘進場的 meanR。B% 不是賠率，
        判決要看得到錢。全格報告，不挑格。
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402

BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
HOUR_MS = 3_600_000
WIN = 5                      # 特徵窗（分鐘），全部嚴格 <= 穿越分鐘
HOLD_MIN = 8 * 60
BASE_B = 0.496               # 全樣本 B 率
RNG = np.random.default_rng(20260907)


def to_ms(t):
    t = int(t)
    return t * 1000 if t < 1_000_000_000_000 else t


def main():
    from shared.db import get_db_conn
    conn = get_db_conn()
    dd = pd.read_sql(
        "SELECT canonical_symbol s, minute_start_ms m, "
        "SUM(bid_add_qty) ba, SUM(bid_cancel_qty) bc, "
        "SUM(ask_add_qty) aa, SUM(ask_cancel_qty) ac "
        "FROM depth_deltas_1m GROUP BY canonical_symbol, minute_start_ms",
        conn)
    conn.close()
    dd["sym"] = dd["s"].str.replace("-USD", "", regex=False)

    rows = []
    for sym, g in dd.groupby("sym"):
        if sym not in ec.CORE9:
            continue
        p = CACHE / f"{sym}USDT_1h.csv"
        if not p.exists():
            continue
        g = g.sort_values("m")
        gm = g["m"].to_numpy(np.int64)
        col = {k: g[k].to_numpy(float) for k in ("ba", "bc", "aa", "ac")}

        bars = sc.load_csv(str(p))
        at1 = sc.atr14(bars)
        ts1 = np.array([to_ms(b[0]) for b in bars], np.int64)
        c1 = np.array([b[sc.C] for b in bars], float)

        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "high", "low", "close"])
        mts = b["ts"].to_numpy(np.int64)
        mhi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        mlo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        mcl = b["close"].to_numpy(float)
        n = len(mts)

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
            if s0 >= s1 or s0 >= n:
                continue
            seg = slice(s0, s1)
            hit = (np.flatnonzero(mhi[seg] > lvl) if d == -1.0
                   else np.flatnonzero(mlo[seg] < lvl))
            if not len(hit):
                continue
            pm = s0 + int(hit[0])
            t_pm = int(mts[pm])
            # 特徵窗：[pm-4, pm]，全部 <= 決策時刻
            i1 = int(np.searchsorted(gm, t_pm, side="right"))
            i0 = int(np.searchsorted(gm, t_pm - (WIN - 1) * 60_000))
            if i1 - i0 < WIN:
                continue                    # 窗內掛撤資料不完整就跳過
            add_a, can_a = col["aa"][i0:i1].sum(), col["ac"][i0:i1].sum()
            add_b, can_b = col["ba"][i0:i1].sum(), col["bc"][i0:i1].sum()
            # 吸收方：buyside 掃單(d=-1) -> ASK；sellside -> BID
            if d == -1.0:
                ab_a, ab_c, pl_a, pl_c = add_a, can_a, add_b, can_b
            else:
                ab_a, ab_c, pl_a, pl_c = add_b, can_b, add_a, can_a
            if ab_a + ab_c <= 0 or pl_a + pl_c <= 0:
                continue

            if pm + HOLD_MIN >= n:
                continue
            entry = float(mcl[pm]) + d * sc.SLIP * A
            stop = entry - d * risk
            sl = slice(pm + 1, pm + HOLD_MIN + 1)
            hk = (np.flatnonzero(mlo[sl] <= stop) if d == 1
                  else np.flatnonzero(mhi[sl] >= stop))
            R = (-1.0 - sc.SLIP / sc.DIS if len(hk) else
                 float(d * (float(mcl[pm + HOLD_MIN]) - d * sc.SLIP * A
                            - entry) / risk))
            rows.append(dict(
                sym=sym,
                day=pd.Timestamp(t_pm, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                defence=(ab_a - ab_c) / (ab_a + ab_c),
                placebo=(pl_a - pl_c) / (pl_a + pl_c),
                isB=bool((c1[j] < lvl) if d == -1.0 else (c1[j] > lvl)),
                R=R))
    d = pd.DataFrame(rows)

    def day_ci_diff(x, grp, days, b=2000):
        """高組 - 低組的差，日聚類 bootstrap。"""
        days = np.asarray(days)
        uq, inv = np.unique(days, return_inverse=True)
        ix = [np.where(inv == m)[0] for m in range(len(uq))]
        reps = np.empty(b)
        for i in range(b):
            p = RNG.integers(0, len(uq), len(uq))
            k = np.concatenate([ix[m] for m in p])
            xs, gs = x[k], grp[k]
            reps[i] = (xs[gs].mean() - xs[~gs].mean()
                       if gs.any() and (~gs).any() else np.nan)
        reps = reps[np.isfinite(reps)]
        return (float(x[grp].mean() - x[~grp].mean()),
                float(np.percentile(reps, 2.5)),
                float(np.percentile(reps, 97.5)))

    print("=== 掃單 A/B 判別器：吸收方的補單 vs 撤單 ===")
    print(f"n = {len(d):,}   {d.day.nunique()} 個 UTC 日   "
          f"{d.sym.nunique()} 幣（缺 SOL/AVAX，掛撤資料沒覆蓋）")
    print()

    bsub = d.isB.mean()
    ok1 = abs(bsub - BASE_B) <= 0.05
    print("=== D1 已知答案對照 ===")
    print(f"  窗內子樣本 B 率 {bsub*100:.1f}%   全樣本 {BASE_B*100:.1f}%   "
          f"容差 ±5 pp -> {'PASS' if ok1 else '**FAIL —— 子樣本不代表母體，以下不解讀**'}")
    print()

    res = {"n": int(len(d)), "B_rate": float(bsub), "D1": bool(ok1)}
    days = d.day.to_numpy()
    for name, fcol in (("defence 吸收方", "defence"), ("placebo 另一側", "placebo")):
        hi = (d[fcol] > d[fcol].median()).to_numpy()
        db, lo, up = day_ci_diff(d.isB.to_numpy(float), hi, days)
        dr, rlo, rup = day_ci_diff(d.R.to_numpy(float), hi, days)
        print(f"=== {name} （中位切兩半，不掃門檻）===")
        print(f"  高組 B% {d.isB[hi].mean()*100:.1f}%   "
              f"低組 B% {d.isB[~hi].mean()*100:.1f}%   "
              f"差 {db*100:+.1f} pp   CI [{lo*100:+.1f},{up*100:+.1f}]")
        print(f"  高組 meanR {d.R[hi].mean():+.4f}   "
              f"低組 meanR {d.R[~hi].mean():+.4f}   "
              f"差 {dr:+.4f}   CI [{rlo:+.4f},{rup:+.4f}]")
        per = d.groupby("sym").apply(
            lambda x: (x.isB[x[fcol] > x[fcol].median()].mean()
                       - x.isB[x[fcol] <= x[fcol].median()].mean()))
        same = int((np.sign(per.dropna()) == np.sign(db)).sum())
        print(f"  逐幣同號 {same}/{len(per.dropna())}   "
              + "  ".join(f"{k}:{v*100:+.0f}" for k, v in per.items()))
        print()
        res[fcol] = dict(dB=db, ci=[lo, up], dR=dr, ciR=[rlo, rup],
                         coins_same=same)

    print("=== 判準 ===")
    dfc = res["defence"]
    plc = res["placebo"]
    ok2 = ok1 and dfc["dB"] > 0 and dfc["ci"][0] > 0
    ok3 = ok1 and dfc["coins_same"] >= 5
    ok4 = plc["ci"][0] <= 0 <= plc["ci"][1]
    print(f"D2 補單 -> 反轉，差 {dfc['dB']*100:+.1f} pp、CI 下緣 "
          f"{dfc['ci'][0]*100:+.1f} pp > 0 ? -> {'**PASS**' if ok2 else 'FAIL'}")
    print(f"D3 逐幣 {dfc['coins_same']}/7 >= 5 ? -> {'PASS' if ok3 else 'FAIL'}")
    print(f"D4 安慰劑不顯著（CI [{plc['ci'][0]*100:+.1f},{plc['ci'][1]*100:+.1f}] "
          f"含零）? -> {'PASS' if ok4 else '**FAIL —— 是共同因子不是機制**'}")
    print()
    if not ok1:
        print("D1 沒過 -> 以上不構成判決。")
    elif ok2 and ok3 and ok4:
        print("**掛單簿帶 A/B 資訊。** 下一步：算它把 B% 抬到多少、"
              "離 77.4% 還差多遠，並另開預註冊 + OOS。")
    else:
        print("**掛單簿的掛撤流不帶 A/B 資訊**（在這個樣本、這個 MDE 之下）。"
              "\n這是掃單線最後一個有機制支撐的候選 —— 結案。")

    res.update(D2=bool(ok2), D3=bool(ok3), D4=bool(ok4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "book_defence.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "book_defence.json")


if __name__ == "__main__":
    main()
