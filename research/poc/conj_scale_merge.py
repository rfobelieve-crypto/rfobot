# -*- coding: utf-8 -*-
"""SDV 的頻率從哪裡來：樞紐尺度 vs 併窗 —— 兩個維度，一套判準（2026-09-10）

起點是使用者的兩句話與一張圖：
    「我們用 5 分鐘級別就用 5 分鐘級別的圖來做 sweep」
    「這邊就是一個流動性獵取然後延續的完美例子為什麼這筆沒抓到」

那張圖（BTC 2026-09-03，78,000 -> 81,348）查下去是：

    sweep      14:23    <- 1h 樞紐**本來就偵測到了**，沒有漏
    delta_ext  14:48
    vol_burst  14:49    <- 流量比掃單晚 25 分鐘

**所以那一筆漏掉的原因不是樞紐尺度，是併窗（MERGE_GAP=5 分）。**
兩個維度必須分開測，否則會拿一個維度的失敗去否定另一個。

===========================================================================
判準（不自己發明，沿用 §0.73 `research/pivot_granularity.py` 已凍結的那套）
===========================================================================

    事件變多 ∧ meanR 保持在 20% 內 ∧ 逐幣廣度保持
        -> 買到頻率，沒有代價
    meanR 退化超過 20%
        -> 粗糙度／窄窗本身就是濾網，維持現行
    meanR 反而改善
        -> 要懷疑不要高興：那是擬合參數的形狀，需要自己的前瞻測試

報告口徑照 CLAUDE.md 核心原則 9：**樣本外放主句，樣本內只在括號裡**。
以事件日中位數切前後半，判準只看後半。**全格報告，不挑格。**

===========================================================================
維度 A — 樞紐尺度（1h / 5m）：**已判 NO-GO，2026-09-10**
===========================================================================

    尺度   期間        n       毛        淨       逐幣
    1h    樣本外     811   +0.2082  +0.1749    7/9
    5m    樣本外   1,474   +0.1047  +0.0761    5/9
    （全期 1h n=1,587 淨 +0.3295 9/9；5m n=2,975 淨 +0.1219 8/9）

meanR 掉 **56%**（門檻 20%）、廣度 7/9 -> 5/9。**兩關都不過。**
而且不是取捨是兩邊都輸：樣本外總報酬 811x0.1749 = 141.8 vs
1,474x0.0761 = 112.2 —— 筆數多 82%，總報酬反而**少 21%**。

與 §0.73 在舊線上的結論同向（細化 PIVOT 到 4：樣本 1.76x、meanR −25%）。
本檔另外排除掉一個競爭解釋：SDV 有流量條件當判別器（§1.03f，兩旗標
都開 +0.331 vs 都沒開 +0.056），原本可能猜它會把更細樞紐的雜訊濾掉 ——
**沒有**。`levels_5m` / `events_5m` 的建置程式碼保留（`--scale`），
資料保留，供日後別的問題使用；SDV 不採用。

===========================================================================
維度 B — 併窗（MERGE_GAP）：本檔要測的
===========================================================================

現行 5 分鐘。九幣 9,267 個掃單裡只有 16.7% 湊得成 SDV；放寬到 30 分
是 31.1%（1.86x）、60 分 37.8%（2.26x）。

**為什麼這個維度值得測，而且 A 的失敗不預測它**：A 失敗的機制是
「更細的樞紐＝更弱的結構」——價位本身變差了。併窗放寬**不會讓價位變差**，
它只是允許流量晚一點到。兩者機制不同，不可互相推論。

**反過來的風險也要寫在前面**：窗放得越寬，「掃單」與「流量」的關聯就越
可能只是巧合（一天有 2.41 個 delta_ext，窗開到 120 分鐘等於隨便一個掃單
都能配到一個流量）。所以本檔一併報**隨機對照**：把流量時點在同一天內
隨機打散，重跑同一套併窗網格 —— 真效應應該隨窗放寬而衰減得比對照慢。
沒有這一關，「窗放寬 -> 事件變多 -> 總報酬變多」會是恆真句。

用法
    python research/poc/conj_scale_merge.py            # 維度 B 全格
    python research/poc/conj_scale_merge.py --scale-ab # 重跑維度 A 的表
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import conj_backtest as cb  # noqa: E402
import conj_clock as ck  # noqa: E402
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402

OUT = HERE / "data" / "results"
FLOW = cb.FLOW
W, DELAY, STOP, HOLD = cb.W, cb.DELAY, cb.STOP, cb.HOLD
GAPS = (5, 10, 15, 20, 30, 45, 60, 90, 120)      # 全格報告，不挑格
SEED = 20260910


def cluster_ci(df, col="R", n=2000, seed=7):
    """日聚類 bootstrap（重抽「日」不是重抽「筆」）。"""
    rng = np.random.default_rng(seed)
    by = {k: v.to_numpy(float) for k, v in df.groupby("day")[col]}
    days = list(by)
    if len(days) < 5:
        return (np.nan, np.nan)
    arrs = [by[d] for d in days]
    idx = rng.integers(0, len(days), size=(n, len(days)))
    out = np.empty(n)
    for i in range(n):
        out[i] = np.concatenate([arrs[j] for j in idx[i]]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# 組裝**只有一份**，在 conj_redef。本檔第一版自己抄了一份迴圈，當場被 S1
# 抓到不一致（1,616 vs 1,587，差在群之間的冷卻）—— 已刪除，改成把 merge_gap
# 傳給正版（mistake.md 2026-08-26：第二份實作會安靜地不同意）。
import conj_redef as cr  # noqa: E402


def ledger_gap(sym, gap_min, shuffle_flow=False, rng=None, scale="1h"):
    """同一套規則，只改併窗。shuffle_flow=True 時把流量時點在同日內打散。"""
    ev_dir = None if scale == "1h" else HERE / "data" / f"events_{scale}"
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq(), ev_dir)
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    n = len(ts)
    day_of = (ts // 86_400_000).astype(np.int64)

    pairs = [(int(m), "sweep") for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in FLOW:
        v = cand.get(nm)
        if v is None or not len(v):
            continue
        ms = [int(x) for x in ec.cooldown_filter(np.sort(v))]
        if shuffle_flow:
            # 同一天之內隨機重放：保留「每天幾個流量事件」的分布，
            # 只破壞它與掃單的時間對應。這才是要打掉的那個結構。
            byday = {}
            for m in ms:
                byday.setdefault(int(day_of[m]), []).append(m)
            ms = []
            for d, lst in byday.items():
                i0 = int(np.searchsorted(day_of, d))
                i1 = int(np.searchsorted(day_of, d + 1))
                if i1 - i0 <= 1:
                    ms += lst
                    continue
                ms += [int(x) for x in rng.integers(i0, i1, size=len(lst))]
            ms.sort()
        pairs += [(m, nm) for m in ms]

    rows = []
    for _a, mem in cr.groups_with_members(pairs, merge_gap=gap_min):
        sig = {t for _, t in mem}
        if not ({"delta_ext", "vol_burst"} <= sig) or "sweep" not in sig:
            continue                                   # 只看 SDV（三者齊發）
        m_sw = min(m for m, t in mem if t == "sweep")
        m_fl = min(m for m, t in mem if t in FLOW)
        ready = max(m_sw, m_fl)                        # 誠實錨點（§1.03b）
        if ready < W or ready + DELAY + HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)
        j0 = ready + DELAY
        ent = float(op[j0])
        end = j0 + HOLD
        adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
               else (hi[j0 + 1:end + 1] - ent)) / A
        hit = np.flatnonzero(adv >= STOP)
        if len(hit):
            R, stopped = -STOP, True
        else:
            R, stopped = float(d * (cl[end] - ent) / A), False
        leg = cb.COST_ENTRY + (cb.COST_STOP if stopped else cb.COST_TIME)
        rows.append(dict(sym=sym, day=int(ts[ready] // 86_400_000),
                         R=R, R_net=R - leg / 1e4 * ent / A,
                         lag=int(abs(m_fl - m_sw))))
    return rows


def table(rows, mid):
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    out = {}
    for lab, sub in (("全期", d), ("前半", d[d.day < mid]), ("樣本外", d[d.day >= mid])):
        if len(sub) < 20:
            out[lab] = None
            continue
        per = sub.groupby("sym").R_net.mean()
        ci = cluster_ci(sub, "R_net")
        out[lab] = dict(n=int(len(sub)), R=float(sub.R.mean()),
                        Rn=float(sub.R_net.mean()), ci_lo=ci[0], ci_hi=ci[1],
                        npos=int((per > 0).sum()), nsym=int(len(per)),
                        total=float(sub.R_net.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1h")
    ap.add_argument("--syms", default="")
    a = ap.parse_args()
    syms = a.syms.split(",") if a.syms else list(cb.CORE9)
    rng = np.random.default_rng(SEED)

    # ---- S1 已知答案對照：gap=5 必須重現 conj_backtest.ledger 的 SDV ----
    ref = []
    for s in syms:
        ref += [t["R"] for t in cb.ledger(s, scale=a.scale)[0] if t["sigk"] == "and"]
    mine = [r["R"] for s in syms for r in ledger_gap(s, et.MERGE_GAP, scale=a.scale)]
    print("=" * 78)
    print(f"S1 已知答案對照（gap={et.MERGE_GAP} 必須等於現行 ledger）")
    print(f"   ledger n={len(ref):5}  毛 {np.mean(ref):+.4f}")
    print(f"   本檔   n={len(mine):5}  毛 {np.mean(mine):+.4f}")
    ok1 = len(ref) == len(mine) and abs(np.mean(ref) - np.mean(mine)) < 1e-9
    print(f"   {'PASS' if ok1 else '**FAIL —— 儀器對不上，下面的數字全部不可用**'}")
    if not ok1:
        sys.exit(1)

    allrows = {}
    for g in GAPS:
        allrows[g] = [r for s in syms for r in ledger_gap(s, g, scale=a.scale)]
    mid = float(np.median([r["day"] for r in allrows[5]]))

    print("\n" + "=" * 78)
    print("維度 B — 併窗網格（全格報告，不挑格）。判準只看樣本外那一列。")
    print(f"{'併窗':>5} {'n':>7} {'淨/筆':>9} {'CI下緣':>9} {'幣+':>6} "
          f"{'總淨值':>9} {'相對5分':>8}")
    res = {}
    base = None
    for g in GAPS:
        t = table(allrows[g], mid)
        res[g] = t
        o = t["樣本外"] if t else None
        if o is None:
            continue
        if base is None:
            base = o
        print(f"{g:5d} {o['n']:7,} {o['Rn']:+9.4f} {o['ci_lo']:+9.4f} "
              f"{o['npos']:3d}/{o['nsym']} {o['total']:9.1f} "
              f"{o['Rn']/base['Rn']:8.2f}x")

    # ---- 隨機對照：流量時點同日打散 ----
    print("\n隨機對照（流量時點同日打散）—— 真效應應該衰減得比它慢")
    print(f"{'併窗':>5} {'n':>7} {'淨/筆':>9} {'幣+':>6}")
    shuf = {}
    for g in GAPS:
        rows = [r for s in syms for r in ledger_gap(s, g, shuffle_flow=True, rng=rng,
                                                    scale=a.scale)]
        t = table(rows, mid)
        shuf[g] = t
        o = t["樣本外"] if t else None
        if o:
            print(f"{g:5d} {o['n']:7,} {o['Rn']:+9.4f} {o['npos']:3d}/{o['nsym']}")

    # ---- 判準 ----
    print("\n" + "=" * 78)
    b5 = res[5]["樣本外"]
    print(f"基準（現行 5 分）樣本外：n={b5['n']:,} 淨 {b5['Rn']:+.4f} "
          f"逐幣 {b5['npos']}/{b5['nsym']}")
    print(f"{'併窗':>5} {'事件變多':>9} {'meanR 保持20%內':>16} {'廣度保持':>9} {'結論':>8}")
    verdict = {}
    for g in GAPS:
        if g == 5 or not res[g] or not res[g]["樣本外"]:
            continue
        o = res[g]["樣本外"]
        c1 = o["n"] > b5["n"] * 1.15
        c2 = o["Rn"] >= b5["Rn"] * 0.80
        c3 = o["npos"] >= b5["npos"]
        v = "採用候選" if (c1 and c2 and c3) else "不過"
        verdict[g] = dict(more=bool(c1), keeps=bool(c2), breadth=bool(c3), verdict=v)
        print(f"{g:5d} {'是' if c1 else '否':>9} {'是' if c2 else '否':>16} "
              f"{'是' if c3 else '否':>9} {v:>8}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"conj_merge_gap_{a.scale}.json"
    p.write_text(json.dumps(dict(scale=a.scale, gaps=res, shuffled=shuf,
                                 verdict=verdict, seed=SEED), indent=2,
                            default=float), encoding="utf-8")
    print(f"\nwritten -> {p}")


if __name__ == "__main__":
    main()
