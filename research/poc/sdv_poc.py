# -*- coding: utf-8 -*-
"""掃單後價格站在成交量 POC 的哪一邊，能不能判斷延續（2026-09-10 預註冊）

使用者 2026-09-10（原話）：「像我手動交易獵取後抓固定成交量 POC 現在的
位置我就可以判斷出價格可能會延續不會反轉」。

POC = 成交量分布裡成交量最大的那個價位（Point of Control）。機制敘述
（**這是使用者的實戰經驗，不是我量出來的**）：掃單之後價格站在 POC 上方
＝買方控制住價值區＝延續；POC 還在價格上方＝價格還沒站穩＝容易被打回。

**優點：事件當下就知道**。它用的是歷史成交量，不像 OI／多空比要等 5 分鐘
粒度的下一格，所以不必延後進場，可以直接掛在現行規格（成立 +3 分）上。

===========================================================================
POC 怎麼算（1 分鐘 bar 的標準近似）
===========================================================================
沒有逐筆成交資料，所以用標準做法：把每根 1 分鐘 bar 的成交量**均勻分配**
到它自己的 [low, high] 區間，累加成分布，取最大的那個價格 bin。
（TradingView 在沒有逐筆資料時也是這樣近似的。）

bin 用 **log 尺度、每格 5 bps** —— 跨幣可比，不會因為 BTC 價格高就分得細。
窗口全格報告 4 / 8 / 24 小時，**不挑最好的**。

因子 = (掃單當下收盤 − POC) / ATR   > 0 表示價格在 POC 上方

===========================================================================
判準（跑之前凍結；置換檢定**內建**，不是事後補做）
===========================================================================
    Q1  樣本外每筆淨值 > 同母體基準（全部順勢）
    Q2  標準誤（這個平均值本身有多不準）< 基準的
    Q3  樣本外日聚類 CI 下緣 > 0
    Q4  逐幣 >= 6/9
    Q5  **置換檢定 p < 0.05**：把因子在幣內隨機重排、跑完全相同的流程
        200 輪，每輪取 6 格（3 窗口 × 2 母體）裡最好的一格。
        p = 隨機最佳 >= 真實最佳的比例 —— 這自動含多重比較。

今天已經吃過一次虧：另一個因子 10 格過 1 格、四條判準全過、CI 下緣
+0.0011，置換檢定一跑 **p = 0.7761**（隨機最佳的中位數還比真實高）。
所以 Q5 不是可選的收尾，是判準的一部分。

方向一樣**由前半資料學**，不由我指定（今天兩次證明使用者的直覺方向
可能是反的，而且前半自己學得到正確的那個）。
"""
from __future__ import annotations

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
import conj_redef as cr  # noqa: E402
import event_census as ec  # noqa: E402

OUT = HERE / "data" / "results"
WINDOWS_H = (4, 8, 24)
BIN_BPS = 5.0
N_PERM = 200
W, DELAY, STOP, HOLD = cb.W, cb.DELAY, cb.STOP, cb.HOLD
FLOW = cb.FLOW
SEED = 20260910


def poc_at(hi, lo, vol, i0, i1, ref, step):
    """[i0, i1) 這段的 POC 價格。把每根 bar 的量均勻攤在它的 [low, high]。"""
    h, l, v = hi[i0:i1], lo[i0:i1], vol[i0:i1]
    ok = np.isfinite(h) & np.isfinite(l) & (v > 0) & (h >= l)
    if not ok.any():
        return np.nan
    h, l, v = h[ok], l[ok], v[ok]
    bh = np.floor(np.log(h / ref) / step).astype(np.int64)
    bl = np.floor(np.log(np.maximum(l, 1e-12) / ref) / step).astype(np.int64)
    lo_b, hi_b = int(bl.min()), int(bh.max())
    nb = hi_b - lo_b + 1
    if nb <= 0 or nb > 40000:
        return np.nan
    prof = np.zeros(nb)
    span = bh - bl + 1
    per = v / span
    # 逐 bar 攤平（span 通常只有幾個 bin，直接迴圈比向量化好讀且夠快）
    for a, b, p in zip(bl - lo_b, bh - lo_b, per):
        prof[a:b + 1] += p
    return float(ref * np.exp((int(np.argmax(prof)) + lo_b + 0.5) * step))


def build(sym):
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close", "volume"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=np.nan)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.nan)
    vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    clv = b["close"].to_numpy(float)
    n = len(ts)
    ref = float(np.nanmedian(clv))
    step = np.log1p(BIN_BPS / 1e4)

    sw = set(int(m) for m in ec.cooldown_filter(np.sort(cand["sweep"])))
    pairs = [(m, "sweep") for m in sorted(sw)]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            pairs += [(int(m), nm) for m in ec.cooldown_filter(np.sort(v))]

    rows = []
    for _a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig:
            continue
        ready = max(min(m for m, t in mem if t == "sweep"),
                    min((m for m, t in mem if t in FLOW), default=10 ** 9))
        if ready >= 10 ** 9:
            ready = min(m for m, t in mem if t == "sweep")
        if ready < 24 * 60 + W or ready + DELAY + HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(clv[ready] - clv[ready - W]) or 1.0)
        j0 = ready + DELAY
        ent = float(op[j0])
        end = j0 + HOLD
        row = dict(sym=sym, day=int(ts[ready]) // 86_400_000,
                   is_sdv=bool({"delta_ext", "vol_burst"} <= sig))
        for wh in WINDOWS_H:
            p = poc_at(hi, lo, vol, ready - wh * 60, ready + 1, ref, step)
            row[f"poc{wh}"] = ((clv[ready] - p) / A) if np.isfinite(p) else np.nan
        for sgn, nm2 in ((d, "with"), (-d, "against")):
            adv = ((ent - lo[j0 + 1:end + 1]) if sgn > 0
                   else (hi[j0 + 1:end + 1] - ent)) / A
            adv = np.nan_to_num(adv, nan=-np.inf)
            if len(np.flatnonzero(adv >= STOP)):
                R, st = -STOP, True
            else:
                R, st = float(sgn * (clv[end] - ent) / A), False
            leg = cb.COST_ENTRY + (cb.COST_STOP if st else cb.COST_TIME)
            row[f"y_{nm2}"] = R - leg / 1e4 * ent / A
        rows.append(row)
    return rows


def boot(days, vals, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    by = {}
    for d, v in zip(days, vals):
        by.setdefault(int(d), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return float(np.mean(vals)), np.nan, np.nan
    arr = [np.array(by[x]) for x in ks]
    idx = rng.integers(0, len(ks), size=(n, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(n)])
    return float(np.mean(vals)), float(o.std(ddof=1)), float(np.percentile(o, 2.5))


def cell(sub, fac, mid):
    f = sub[fac].to_numpy(float)
    ok = np.isfinite(f)
    if ok.sum() < 200:
        return None
    isf = (sub.day < mid).to_numpy()
    med = np.nanmedian(f[isf & ok])
    yw, ya = sub.y_with.to_numpy(), sub.y_against.to_numpy()
    high = f > med
    a = isf & ok
    sign = 1 if np.nanmean(yw[a & high]) >= np.nanmean(yw[a & ~high]) else -1
    b = (~isf) & ok
    v = np.where(np.where(high[b], sign, -sign) > 0, yw[b], ya[b])
    return v, sub.day.to_numpy()[b], sub.sym.to_numpy()[b], sign


def best_cell(d, rng=None):
    best, where = -9e9, None
    for mname, sub in (("所有掃單", d), ("SDV", d[d.is_sdv])):
        sub = sub.reset_index(drop=True)
        if rng is not None:
            sub = sub.copy()
            for wh in WINDOWS_H:
                c = f"poc{wh}"
                v = sub[c].to_numpy(float).copy()
                for s, idx in sub.groupby("sym").indices.items():
                    v[idx] = rng.permutation(v[idx])
                sub[c] = v
        mid = float(sub.day.median())
        for wh in WINDOWS_H:
            r = cell(sub, f"poc{wh}", mid)
            if r and float(np.mean(r[0])) > best:
                best, where = float(np.mean(r[0])), f"{mname}·{wh}h"
    return best, where


def main():
    rows = []
    for s in cb.CORE9:
        r = build(s)
        rows += r
        print(f"  {s:5} {len(r):6,} 筆")
    d = pd.DataFrame(rows).reset_index(drop=True)
    print(f"\n共 {len(d):,} 筆（SDV {int(d.is_sdv.sum()):,}）\n")

    res = {}
    for mname, sub in (("所有掃單", d), ("SDV", d[d.is_sdv])):
        sub = sub.reset_index(drop=True)
        mid = float(sub.day.median())
        sec = sub[sub.day >= mid]
        bm, bse, blo = boot(sec.day.values, sec.y_with.to_numpy())
        perb = sec.groupby("sym").y_with.mean()
        print("=" * 84)
        print(f"母體 {mname}   樣本外 {len(sec):,}")
        print(f"{'窗口':10} {'n':>6} {'淨/筆':>9} {'SE':>7} {'CI下緣':>9} {'幣+':>6}  方向")
        print(f"{'基準(全順勢)':10} {len(sec):6,} {bm:+9.4f} {bse:7.4f} {blo:+9.4f} "
              f"{int((perb > 0).sum()):3d}/{len(perb)}")
        for wh in WINDOWS_H:
            r = cell(sub, f"poc{wh}", mid)
            if not r:
                continue
            v, dd, sy, sign = r
            m, se, lo = boot(dd, v)
            per = pd.DataFrame({"s": sy, "v": v}).groupby("s").v.mean()
            q = (m > bm, se < bse, lo > 0, int((per > 0).sum()) >= 6)
            res[f"{mname}·{wh}h"] = dict(n=int(len(v)), m=m, se=se, lo=lo,
                                         npos=int((per > 0).sum()), nsym=int(len(per)),
                                         sign=int(sign), q=[bool(x) for x in q])
            print(f"{wh:>3}小時    {len(v):6,} {m:+9.4f} {se:7.4f} {lo:+9.4f} "
                  f"{int((per > 0).sum()):3d}/{len(per)}"
                  f"  {'價格在POC上方→順勢' if sign > 0 else '價格在POC上方→逆勢'}"
                  f"  Q1{'✓' if q[0] else '✗'}Q2{'✓' if q[1] else '✗'}"
                  f"Q3{'✓' if q[2] else '✗'}Q4{'✓' if q[3] else '✗'}")
        print()

    real, where = best_cell(d)
    print("=" * 84)
    print(f"Q5 置換檢定：真實 6 格裡最好的是 {where}，每筆 {real:+.4f}")
    rng = np.random.default_rng(SEED)
    hits, vals = 0, []
    for i in range(N_PERM):
        b, _ = best_cell(d, rng)
        vals.append(b)
        hits += (b >= real)
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{N_PERM}  隨機最佳中位 {np.median(vals):+.4f}"
                  f"  p 目前 {(hits+1)/(i+2):.3f}")
    p = (hits + 1) / (N_PERM + 1)
    print(f"\n   隨機最佳：中位 {np.median(vals):+.4f}  p95 {np.percentile(vals,95):+.4f}")
    print(f"   **p = {p:.4f}**")
    anyq = [k for k, v in res.items() if all(v["q"])]
    print()
    print(f"判決：Q1-Q4 全過的格 = {anyq if anyq else '無'}；Q5 p={p:.4f}"
          f" -> {'**有提升**' if (anyq and p < 0.05) else '無提升'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sdv_poc.json").write_text(
        json.dumps(dict(cells=res, perm_p=p, real=real, where=where),
                   indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
