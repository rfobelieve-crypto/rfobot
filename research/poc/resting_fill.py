# -*- coding: utf-8 -*-
"""掛單成交率與逆選擇，依穿透深度分層 —— 那 0.0568 R 拿不拿得到。

背景（`entry_decomp.py`，2026-09-07）
    落差 0.0568 R 全部發生在「價格碰到價位那一分鐘之內」，而拿回它的方法
    是**事先把限價單掛在 lvl 等**，不是更快通知。但那個結論建立在一個
    **未經檢驗的假設**上：

        A 臂假設「價格觸到 lvl 就成交在 lvl」。

    這是撮合上的樂觀假設。真實的限價單有佇列：價格只是**碰到** lvl 而
    不穿過時，先到的單先成交，你可能整個沒被吃到。而價格**穿得越深**，
    你越可能成交 —— 也越可能是在「行情繼續往你不利的方向衝」的時候成交。
    這就是逆選擇：**你只在最糟的情況下拿得到部位。**

    所以 0.0568 R 是一個**條件於全部成交**的數字。這支把那個條件拆開。

撮合模型（事前寫死，全格報告，不挑格）
    掛單掛在 lvl，從 sweep bar 之後一直掛到回踩窗 W=8 根小時結束。
    在門檻 δ（ATR 單位）之下：

        成交 = 存在某一分鐘，價格從 lvl 起往穿越方向**穿透 ≥ δ**
        成交時刻 = **第一根**滿足該條件的分鐘；成交價 = lvl
        整個 W 窗內從未穿透 ≥ δ  ->  **未成交**（沒有部位，不是虧損）

    δ = 0.00 精確重現現行 A 臂（觸價即成交）——**這是已知答案的對照組**，
    成交率必須 ≈ 100%、meanR 必須 ≈ entry_decomp 的 R_A_full。對不上就是
    儀器壞了，不是發現（mistake.md 2026-07-29）。

    δ = 0.01 / 0.02 / 0.05 / 0.10 ATR 是逐步嚴格的佇列假設。

**分層變數不是濾網。** depth 是成交之後才知道的，掛單的人不能選它——
它決定「有沒有成交」，不決定「要不要下單」。所以用它分組是在模擬撮合，
不是事後選樣本。任何把它讀成進場條件的用法都是前視（§0.98 的坑）。

出場：**完整凍結規則**（3.5 ATR 災難停損 + HOLD=8 小時），從成交所屬的
小時 bar 起算，停損從**下一根**小時 bar 開始檢查（沿用 sweep_core 的時點
規矩，嚴格晚於成交）。SLIP 也一個字不動——限價進場其實不該付進場滑價，
放寬它會讓數字變好，而**往有利方向放寬判準是不允許的**；另外報一個
零進場滑價的敏感度供參，不作為判準。

判準（跑之前寫死，全部寫在 CI 上不寫在點估計上）
    P1 逆選擇存在嗎
        對每個 δ>0：成交組 meanR − 同 δ 未成交組（用 δ=0 規則算出的
        反事實 R）的日聚類 CI **上緣 < 0**  ->  逆選擇 CONFIRMED
        CI 含零 -> INCONCLUSIVE（不得讀成「沒有逆選擇」）
    P2 掛單架構還活著嗎
        最嚴格的 δ=0.10 下，成交組 meanR 的日聚類 CI **下緣 > 0**
        ->  掛單在最保守的佇列假設下仍是正的 = 0.0568 R 拿得到
        CI 含零 -> INCONCLUSIVE；上緣 < 0 -> REJECT
    P3 儀器
        δ=0 的成交率 ≥ 99% 且 meanR 與 entry_decomp 的 R_A_full 差
        < 0.005 R。不過 -> 停手，先修儀器，不解讀任何一格。

markout（逆選擇的直接度量，不參與判準）
    成交後 5 / 15 / 60 分鐘，方向對齊的價格移動 / risk。負值 = 成交之後
    立刻往不利方向走 = 被逆選擇。
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
DELTAS = [0.00, 0.01, 0.02, 0.05, 0.10]
MARKOUTS = [5, 15, 60]
RNG = np.random.default_rng(20260907)


def build(sym):
    """每個掃單事件一列，含各 δ 下的成交與否、成交時刻、R 與 markout。"""
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
    nm = len(mts)
    hts = np.array([int(x[0]) for x in b1], dtype=np.int64) * 1000   # 秒 -> 毫秒

    rows = []
    for e in sc.detect_sweeps(b1):
        j, lvl = e["j"], e["level"]
        A = atr[j]
        if A is None or A == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd                                    # 反轉方向：買側掃單 -> 做多
        risk = sc.DIS * A

        # 掛單存活的分鐘範圍：sweep bar 收盤之後 -> 回踩窗 W 根小時結束
        t0 = int(b1[j][0]) * 1000 + HOUR_MS        # 1h 快取的 time 是**秒**
        t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
        i0 = int(np.searchsorted(mts, t0, side="left"))
        i1 = int(np.searchsorted(mts, t1, side="left"))
        if i1 <= i0 or i1 > nm:
            continue

        # 每一分鐘從 lvl 起算、往穿越方向的穿透深度（ATR 單位；未觸到為負）
        if kd == 1:                                # 買側流動性被掃 -> 價格往下穿
            pierce = (lvl - mlo[i0:i1]) / A
        else:
            pierce = (mhi[i0:i1] - lvl) / A

        row = dict(sym=sym, j=j, side="LONG" if d == 1 else "SHORT",
                   lvl=lvl, atr=A,
                   day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                    tz="UTC").strftime("%Y-%m-%d"),
                   max_pierce=float(np.nanmax(pierce)) if len(pierce) else np.nan)

        for dl in DELTAS:
            tag = f"{dl:.2f}"
            hit = np.flatnonzero(pierce >= dl)
            if len(hit) == 0:
                row[f"fill_{tag}"] = 0
                row[f"R_{tag}"] = np.nan
                for mo in MARKOUTS:
                    row[f"mo{mo}_{tag}"] = np.nan
                continue
            k = i0 + int(hit[0])                   # 第一根穿透 ≥ δ 的分鐘
            f = int(np.searchsorted([int(x[0]) * 1000 for x in b1],
                                    int(mts[k]), side="right")) - 1
            if f < 0 or f + 1 >= n:
                row[f"fill_{tag}"] = 0
                row[f"R_{tag}"] = np.nan
                for mo in MARKOUTS:
                    row[f"mo{mo}_{tag}"] = np.nan
                continue

            entry = lvl + d * sc.SLIP * A          # 凍結的 SLIP 不動
            stop = entry - d * risk
            R = None
            for q in range(f + 1, min(f + sc.HOLD + 1, n)):
                if (d == 1 and lo[q] <= stop) or (d == -1 and h[q] >= stop):
                    R = -1.0 - sc.SLIP / sc.DIS
                    break
            if R is None:
                exb = min(f + sc.HOLD, n - 1)
                R = d * (cl[exb] - d * sc.SLIP * A - entry) / risk

            row[f"fill_{tag}"] = 1
            row[f"R_{tag}"] = float(R)
            row[f"R0slip_{tag}"] = float(d * (cl[min(f + sc.HOLD, n - 1)]
                                              - d * sc.SLIP * A - lvl) / risk)
            row[f"fillmin_{tag}"] = int((mts[k] - t0) // MIN_MS)
            for mo in MARKOUTS:
                q = k + mo
                row[f"mo{mo}_{tag}"] = (float(d * (mcl[q] - lvl) / risk)
                                        if q < nm else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return float("nan"), float("nan"), float("nan"), float("nan")
    uq, inv = np.unique(days, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([idx[k] for k in p])].mean()
    return (float(x.mean()), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)), float(np.std(reps, ddof=1)))


def main():
    d = pd.concat([build(s) for s in CORE9], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "resting_fill.parquet", index=False)
    days = d["day"].to_numpy()
    print(f"掃單事件 n={len(d):,}   UTC 日={d.day.nunique():,}   幣={d.sym.nunique()}")
    print("（全格報告，五個門檻都印。任何一格都不是單獨的結論。）\n")

    res = {"n": int(len(d)), "days": int(d.day.nunique())}

    # ---- P3 儀器關：δ=0 必須重現 entry_decomp 的 A 臂 --------------------
    ref = None
    p = OUT / "entry_decomp.json"
    if p.exists():
        ref = json.loads(p.read_text(encoding="utf-8")).get("R_A_full", {}).get("mean")
    fr0 = float(d["fill_0.00"].mean())
    m0, lo0, hi0, _ = day_ci(d["R_0.00"], days)
    gap = abs(m0 - ref) if ref is not None else float("nan")
    ok3 = (fr0 >= 0.99) and (not np.isfinite(gap) or gap < 0.005)
    print("=== P3 儀器關（已知答案的對照組）===")
    print(f"  δ=0 成交率 {fr0*100:.2f}%（需 ≥99%）   meanR {m0:+.4f}"
          + (f"   entry_decomp R_A_full {ref:+.4f}   差 {gap:.4f}（需 <0.005）"
             if ref is not None else "   （entry_decomp.json 不在，無法對照）"))
    print(f"  -> {'PASS' if ok3 else '**FAIL — 停手，先修儀器**'}\n")
    res["P3"] = dict(fill_rate=fr0, mean=m0, ref=ref, gap=gap, passed=bool(ok3))
    if not ok3:
        print("儀器沒過，以下各格不解讀。")

    # ---- 主表：五個門檻全格 ---------------------------------------------
    print("=== 成交率、成交組報酬、與未成交組的反事實 ===\n")
    print(f"{'δ(ATR)':>7s} {'成交率':>8s} {'n 成交':>8s} "
          f"{'成交組 meanR':>13s} {'日聚類 CI95':>22s} "
          f"{'未成交反事實':>13s} {'差(成交−未成交)':>24s}")
    per = {}
    for dl in DELTAS:
        tag = f"{dl:.2f}"
        f = d[f"fill_{tag}"].to_numpy(int) == 1
        fr = float(f.mean())
        mF, loF, hiF, seF = day_ci(d.loc[f, f"R_{tag}"], days[f])
        # 未成交組的反事實：用 δ=0 規則（觸價即成交）算出的 R
        nf = ~f
        mN, loN, hiN, _ = day_ci(d.loc[nf, "R_0.00"], days[nf])
        if nf.sum() >= 30:
            # 差值的 CI：兩組獨立，按日聚類各自 bootstrap 後取差
            uqF = d.loc[f, "day"].to_numpy()
            uqN = d.loc[nf, "day"].to_numpy()
            xF = d.loc[f, f"R_{tag}"].to_numpy(float)
            xN = d.loc[nf, "R_0.00"].to_numpy(float)

            def boot(x, dy, b=2000):
                ok = np.isfinite(x)
                x, dy = x[ok], dy[ok]
                uq, inv = np.unique(dy, return_inverse=True)
                idx = [np.where(inv == k)[0] for k in range(len(uq))]
                out = np.empty(b)
                for i in range(b):
                    pp = RNG.integers(0, len(uq), len(uq))
                    out[i] = x[np.concatenate([idx[k] for k in pp])].mean()
                return out

            diff = boot(xF, uqF) - boot(xN, uqN)
            dmean = float(np.nanmean(xF) - np.nanmean(xN))
            dlo, dhi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            ds = f"{dmean:+.4f} [{dlo:+.4f},{dhi:+.4f}]"
        else:
            dmean = dlo = dhi = float("nan")
            ds = "(未成交組 n<30)"
        print(f"{dl:7.2f} {fr*100:7.2f}% {int(f.sum()):8,d} "
              f"{mF:+13.4f}  [{loF:+.4f},{hiF:+.4f}] "
              f"{mN:+13.4f} {ds:>24s}")
        per[tag] = dict(delta=dl, fill_rate=fr, n_fill=int(f.sum()),
                        mean_fill=mF, ci_fill=[loF, hiF], se_fill=seF,
                        mean_nofill_cf=mN,
                        diff=dmean, diff_ci=[dlo, dhi])
    res["by_delta"] = per

    # ---- markout：成交之後立刻發生什麼 ----------------------------------
    print("\n=== 成交後 markout（方向對齊 / risk；負 = 成交後立刻往不利方向）===\n")
    print(f"{'δ(ATR)':>7s}" + "".join(f"{str(x)+'m':>26s}" for x in MARKOUTS))
    mo_res = {}
    for dl in DELTAS:
        tag = f"{dl:.2f}"
        f = d[f"fill_{tag}"].to_numpy(int) == 1
        cells, cell_res = [], {}
        for mo in MARKOUTS:
            mm, ll, hh, _ = day_ci(d.loc[f, f"mo{mo}_{tag}"], days[f])
            cells.append(f"{mm:+.4f} [{ll:+.4f},{hh:+.4f}]")
            cell_res[str(mo)] = dict(mean=mm, ci=[ll, hh])
        mo_res[tag] = cell_res
        print(f"{dl:7.2f}" + "".join(f"{c:>26s}" for c in cells))
    res["markout"] = mo_res

    # ---- 判準 -----------------------------------------------------------
    print("\n=== 預註冊判準 ===\n")
    print("P1 逆選擇（成交組 − 未成交組反事實，CI 上緣 < 0 才算 CONFIRMED）")
    p1 = {}
    for dl in DELTAS[1:]:
        tag = f"{dl:.2f}"
        hi = per[tag]["diff_ci"][1]
        v = ("CONFIRMED" if np.isfinite(hi) and hi < 0 else
             "INCONCLUSIVE" if np.isfinite(hi) else "N/A")
        p1[tag] = v
        print(f"    δ={dl:.2f}   差 {per[tag]['diff']:+.4f}  "
              f"CI 上緣 {hi:+.4f}   -> {v}")
    res["P1"] = p1

    tagm = f"{DELTAS[-1]:.2f}"
    lo_ = per[tagm]["ci_fill"][0]
    hi_ = per[tagm]["ci_fill"][1]
    v2 = ("PASS" if np.isfinite(lo_) and lo_ > 0 else
          "REJECT" if np.isfinite(hi_) and hi_ < 0 else "INCONCLUSIVE")
    print(f"\nP2 最嚴格門檻 δ={DELTAS[-1]:.2f} 下掛單是否仍為正 "
          f"（CI 下緣 > 0 才 PASS）")
    print(f"    meanR {per[tagm]['mean_fill']:+.4f}  "
          f"CI [{lo_:+.4f},{hi_:+.4f}]  成交率 {per[tagm]['fill_rate']*100:.1f}%"
          f"   -> {v2}")
    res["P2"] = dict(verdict=v2, delta=DELTAS[-1], **per[tagm])

    print("\n=== 敏感度（不參與判準）：限價進場不付滑價 ===")
    for dl in DELTAS:
        tag = f"{dl:.2f}"
        c = f"R0slip_{tag}"
        if c not in d.columns:
            continue
        f = d[f"fill_{tag}"].to_numpy(int) == 1
        mm, ll, hh, _ = day_ci(d.loc[f, c], days[f])
        print(f"    δ={dl:.2f}   meanR {mm:+.4f}  [{ll:+.4f},{hh:+.4f}]")

    (OUT / "resting_fill.json").write_text(json.dumps(res, indent=2, default=float),
                                           encoding="utf-8")
    print("\nwritten ->", OUT / "resting_fill.json")


if __name__ == "__main__":
    main()
