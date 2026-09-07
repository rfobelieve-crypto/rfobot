# -*- coding: utf-8 -*-
"""掃單失敗：**等它收回價位內側再進場**——最後那一格

===========================================================================
為什麼是這一格（前面五條路徑把問題縮到只剩它）
===========================================================================
`pierce_entry.py`（2026-09-07）把失敗精確定位成兩件事，第一件已解決：

    進場價與價位的落差   小時級 -0.085 R  ->  分鐘級 -0.0067 R   已解決
    交易選擇（哪些會回踩）                     值 -0.06 R        還開著

而選擇這件事，兩端都測過、中間整段是空的：

    穿越分鐘進場   選擇資訊 = 無      價格最好      -0.0602
    ???            ???                ???           **沒測過**
    小時 K 收盤進  選擇資訊 = B 已知  價格跑掉      -0.0008

診斷（同日）給了中間格的形狀：

    A 情境（收盤仍在價位外）50.4%  meanR -0.2181  停損率 23.7%
    B 情境（收盤已回內側）  49.6%  meanR +0.1001  停損率 11.3%
    反轉最早發生在穿越後    中位 1 分鐘，80.9% 在該小時內反轉過

**整條線的價值全在「能不能事先知道這是 B」。** 而「收盤回到價位內側」這件事
本身就是策略的論點，它在分鐘尺度上是可觀測的——不需要預測，只要等。

===========================================================================
規則
===========================================================================
母體    `sweep_core.detect_sweeps` 的**每一個**掃單，無回踩濾網
        （pierce_entry 的 M6 證明過：帶濾網的母體對市價進場是前視）
觸發    穿越之後，第一根**收盤回到價位內側**的分鐘。
        buyside（向上刺穿）-> close < lvl；sellside -> close > lvl。
        只在掃單那根小時 K 之內找（超過就是「等確認」，§0.98 已四度判死）。
進場    那一分鐘的收盤 + 不利滑價（市價單）
        另報 WAIT_K = 反轉後再等 k 分鐘（0/1/2/5）當敏感度，不挑格
風險    R = DIS x ATR = 3.5 x ATR（與凍結同一把尺）
停損    entry - d x risk，以分鐘高低價判定
出場    停損 或 進場後 480 分鐘（= 凍結 HOLD 8 根小時 K）的收盤，先到者
成本    A 10 bps / B 13 bps，cost_R = bps/1e4/(DIS x ATR%)

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
V1  **上界對照，必須先過。** 同一個「已反轉」母體，用**最好的價格**
    （穿越分鐘）進場的 meanR 必須 >= 本格的 meanR。
    等待只會讓價格變差，所以本格不可能贏過上界；若贏了就是實作錯了。
V2  扣成本 A 之後，日聚類 bootstrap CI 下緣 > 0  -> 這條線活了
    CI 含零或為負 -> **五條路徑加這一格全負，掃單線結案確定**
V3  逐幣 >= 6/9（與 Gate F 同門檻）
V4  兩半同號（前半 / 後半樣本各自 meanR 同號）——只在某一段成立的效應
    要當 regime artifact（factor-research #7）
V5  全格報告 WAIT_K 的每一格，不挑格。**不做濾網搜尋**——
    「看過哪一半贏再去找濾網」正是 mistake.md 2026-06-20 擋的那件事。
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
WAIT_K = [0, 1, 2, 5]
HOLD_MIN = 8 * 60
SCEN = {"A 目標執行": 10, "B 全 taker": 13}
RNG = np.random.default_rng(20260907)
HOUR_MS = 3_600_000


def to_ms(t):
    t = int(t)
    return t * 1000 if t < 1_000_000_000_000 else t


def collect(sym):
    p = CACHE / f"{sym}USDT_1h.csv"
    if not p.exists():
        return None, None
    bars = sc.load_csv(str(p))
    at1 = sc.atr14(bars)
    ts1 = np.array([to_ms(b[0]) for b in bars], np.int64)

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

    rows = []
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
        pm = s0 + int(hit[0])                 # 穿越的那一分鐘
        # 觸發：穿越之後第一根收盤回到價位內側的分鐘（只在該小時之內找）
        w = mcl[pm:s1]
        rev = (np.flatnonzero(w < lvl) if d == -1.0
               else np.flatnonzero(w > lvl))
        if not len(rev):
            continue                           # 該小時內從未收回 -> 不交易
        rm = pm + int(rev[0])

        def score(j0):
            end = j0 + HOLD_MIN
            if j0 >= n or end >= n:
                return np.nan
            px = float(mcl[j0]) if j0 == rm else float(mop[j0])
            entry = px + d * sc.SLIP * A
            stop = entry - d * risk
            sl = slice(j0 + 1, end + 1)
            hitk = (np.flatnonzero(mlo[sl] <= stop) if d == 1
                    else np.flatnonzero(mhi[sl] >= stop))
            if len(hitk):
                return -1.0 - sc.SLIP / sc.DIS
            exq = float(mcl[end]) - d * sc.SLIP * A
            return float(d * (exq - entry) / risk)

        row = {"sym": sym, "ts": int(mts[rm]),
               "day": pd.Timestamp(int(mts[rm]), unit="ms",
                                   tz="UTC").strftime("%Y-%m-%d"),
               "rev_min": int(rev[0])}
        for k in WAIT_K:
            row[f"w{k}"] = score(rm + k)
        # V1 上界：同一個母體，用最好的價格（穿越分鐘）進場
        end = pm + HOLD_MIN
        if end < n:
            entry = float(mcl[pm]) + d * sc.SLIP * A
            stop = entry - d * risk
            sl = slice(pm + 1, end + 1)
            hitk = (np.flatnonzero(mlo[sl] <= stop) if d == 1
                    else np.flatnonzero(mhi[sl] >= stop))
            row["ub"] = (-1.0 - sc.SLIP / sc.DIS if len(hitk) else
                         float(d * (float(mcl[end]) - d * sc.SLIP * A
                                    - entry) / risk))
        else:
            row["ub"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), atrpct


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
    frames, ap = [], {}
    for sym in ec.CORE9:
        d, a = collect(sym)
        if d is None or not len(d):
            continue
        frames.append(d)
        ap[sym] = a
    d = pd.concat(frames, ignore_index=True).sort_values("ts")
    w = d.groupby("sym").size()
    apw = float(sum(w[s] * ap[s] for s in w.index) / w.sum())
    cost = {k: bps / 1e4 / (sc.DIS * apw) for k, bps in SCEN.items()}

    print("=== 掃單失敗：等它收回價位內側再進場 ===")
    print(f"觸發 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日、九幣"
          f"   收回時間中位 {d.rev_min.median():.0f} 分")
    print(f"加權 ATR% {apw*100:.3f}%   成本（R）A {cost['A 目標執行']:.4f} / "
          f"B {cost['B 全 taker']:.4f}")
    print()

    ub, ublo, ubhi = day_ci(d["ub"].to_numpy(), d.day.to_numpy())
    print("=== V1 上界對照（同一母體、用最好的價格 = 穿越分鐘進場）===")
    print(f"  上界 meanR {ub:+.4f}  [{ublo:+.4f},{ubhi:+.4f}]")
    print()

    print("=== V5 全格報告（不挑格）===")
    print(f"{'等待':>5s} {'零成本':>9s} {'日聚類 CI95':>22s} "
          f"{'淨 A':>9s} {'淨 A 下緣':>10s} {'幣 +':>5s} {'前半':>9s} {'後半':>9s}")
    half = len(d) // 2
    res = {"n": int(len(d)), "atr_pct": apw, "cost": cost,
           "upper_bound": ub, "waits": {}}
    for k in WAIT_K:
        col = d[f"w{k}"].to_numpy()
        m, lo, hi = day_ci(col, d.day.to_numpy())
        per = d.groupby("sym")[f"w{k}"].mean() - cost["A 目標執行"]
        npos = int((per > 0).sum())
        h1 = float(np.nanmean(col[:half]))
        h2 = float(np.nanmean(col[half:]))
        res["waits"][str(k)] = dict(
            mean=m, ci=[lo, hi], netA=m - cost["A 目標執行"],
            netA_lo=lo - cost["A 目標執行"], coins_pos=npos,
            half1=h1, half2=h2)
        print(f"{k:4d}m {m:+9.4f}  [{lo:+.4f},{hi:+.4f}] "
              f"{m-cost['A 目標執行']:+9.4f} "
              f"{lo-cost['A 目標執行']:+10.4f} {npos:4d}/9 "
              f"{h1:+9.4f} {h2:+9.4f}")

    c0 = res["waits"]["0"]
    ok1 = ub >= c0["mean"] - 1e-9
    ok2 = c0["netA_lo"] > 0
    ok3 = c0["coins_pos"] >= 6
    ok4 = np.sign(c0["half1"]) == np.sign(c0["half2"])
    print()
    print("=== 判準 ===")
    print(f"V1 上界 {ub:+.4f} >= 本格 {c0['mean']:+.4f} ? -> "
          + ("PASS" if ok1 else "**FAIL —— 等待不可能拿到更好的價格，實作錯了**"))
    print(f"V2 扣成本 A 的 CI 下緣 {c0['netA_lo']:+.4f} > 0 ? -> "
          + ("**PASS**" if ok2 else "FAIL"))
    print(f"V3 逐幣 {c0['coins_pos']}/9 >= 6 ? -> {'PASS' if ok3 else 'FAIL'}")
    print(f"V4 兩半同號（{c0['half1']:+.4f} / {c0['half2']:+.4f}）? -> "
          + ("PASS" if ok4 else "FAIL"))
    print()
    if not ok1:
        print("V1 沒過 -> 以上不解讀。")
    elif ok2 and ok3 and ok4:
        print("**掃單線活過來了** —— 等收回內側再進場是可執行且為正的。")
    else:
        print("**六條路徑全負 —— 掃單線結案確定。**")
        print("進場價那一半分鐘資料修好了；選擇那一半，等待也換不到。")

    res.update(V1=bool(ok1), V2=bool(ok2), V3=bool(ok3), V4=bool(ok4))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "revert_entry.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "revert_entry.json")


if __name__ == "__main__":
    main()
