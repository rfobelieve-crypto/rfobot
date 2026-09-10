# -*- coding: utf-8 -*-
"""事件普查 — 換一種事件定義，那把尺會不會變細？

使用者 2026-09-07：
    「把所有假設分成一個個事件，用這個事件發生後的價格走向來做判斷。」

前面三次（POC / OI / delta）都是**條件化**：拿同一批掃單事件，問「其中 X 高
的是不是比 X 低的好」。那是測**事件內部的差異**，尺是 0.048~0.068 R。
事件研究測的是**事件本身的效果**，尺細 2~3 倍。這支先量「換一種事件，尺會不會
真的變細」，**不判決任何假設**。

**這是篩選不是判決**：5 種事件 × 5 個 horizon = 25 格，全格報告。任何一格都
不構成結論——要下結論必須有它自己的預註冊（否則就是挑格，§0.92 擋的那件事）。

事件定義（全部只用 t 之前**已完整**的資料，向後看 5 分鐘）
    sweep       凍結引擎的掃單事件（基準，已知答案）
    liq_burst   清算名目 5 分鐘和 ≥ 該幣 p99          （BTC/ETH、159 天）
    delta_ext   |delta| 5 分鐘和 ≥ 該幣 p99           （九幣全歷史）
    vol_burst   5 分鐘量 / 前 30 日同時段均值 ≥ 該幣 p99
    oi_crash    5 分鐘 OI 變化 ≤ 該幣 p1
去重：同幣 60 分鐘冷卻，保留最早的一筆。

OI 的前視（2026-09-07 查出並修正）
    Binance metrics 的一列，其 `sum_open_interest` **不是 create_time 當下的
    快照**——它帶著該 5 分鐘區間的資訊。在區間內任何一分鐘用它，等於偷看最多
    4 分鐘後的資料。用時間推移驗證：不推移 t=+11.10、推後一格 t=-0.66，
    **效應整個消失**。所以這裡一律只用 create_time <= t - 5min 的列。

方向與標籤（統一、因果）
    impulse = sign(close(t) - close(t-5m))      只用 t 之前
    r_tau   = impulse x (close(t+tau) - close(t)) / ATR    延續為正
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
BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
MIN_MS = 60_000
W = 5
COOLDOWN = 60
TAUS = [5, 15, 30, 60, 240]
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
RNG = np.random.default_rng(20260907)


def cooldown_filter(idx, minutes=COOLDOWN):
    keep, last = [], -10**9
    for i in idx:
        if i - last >= minutes:
            keep.append(i)
            last = i
    return np.array(keep, dtype=np.int64)


def label(cl, at, k, tau):
    """impulse 方向（只用 t 之前）x 之後 tau 分鐘的報酬 / ATR。"""
    n = len(cl)
    ok = (k >= W) & (k + tau < n)
    k = k[ok]
    imp = np.sign(cl[k] - cl[k - W])
    imp[imp == 0] = 1.0
    a = at[k]
    good = np.isfinite(a) & (a > 0)
    k, imp, a = k[good], imp[good], a[good]
    return k, imp * (cl[k + tau] - cl[k]) / a


def day_stats(days, r):
    uq, inv = np.unique(days, return_inverse=True)
    cnt = np.bincount(inv)
    dev = (r - r.mean()) ** 2
    var_by_day = np.bincount(inv, weights=dev)
    share = np.sort(var_by_day)[::-1] / max(var_by_day.sum(), 1e-12)
    idx = [np.where(inv == j)[0] for j in range(len(uq))]
    reps = np.empty(2000)
    for i in range(2000):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = r[np.concatenate([idx[j] for j in p])].mean()
    se = float(np.std(reps, ddof=1))
    top = np.sort(cnt)[::-1]
    return dict(n=int(len(r)), days=int(len(uq)),
                ev_top1=float(top[0] / len(r)),
                var_top1=float(share[0]), var_top5=float(share[:5].sum()),
                mean=float(r.mean()), se=se, mde=float(1.96 * se))


def detect_all(sym, liq, events_dir=None):
    """五種事件的分鐘索引（未去重）與該幣的分鐘序列。

    `events_dir`（2026-09-10）：掃單事件表的目錄。None = `data/events`
    （1h 樞紐，預設，行為與先前逐位元相同）；傳 `data/events_5m` 就是
    5 分鐘樞紐。**只換來源，偵測算法一個字沒動。**

    2026-09-07 抽出來共用：重疊矩陣（`event_overlap.py`）必須用**同一份**
    偵測，否則兩份實作會安靜地不同意（mistake.md 2026-08-26）。函式內容
    逐行來自原本 main() 的迴圈，行為未變。
    """
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "close", "volume", "delta", "atr_h14"])
    ts = b["ts"].to_numpy(np.int64)
    cl = b["close"].to_numpy(float)
    vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
    at = b["atr_h14"].to_numpy(float)
    n = len(ts)
    day = (ts // 86_400_000)

    def back_sum(x):
        c = np.concatenate([[0.0], np.cumsum(x)])
        i = np.arange(n)
        return c[i + 1] - c[np.clip(i + 1 - W, 0, n)]

    cand = {}
    ad = back_sum(np.abs(dl))
    cand["delta_ext"] = np.flatnonzero(ad >= np.nanpercentile(ad, 99))

    v5 = back_sum(vol)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for kd in range(1, 31):
        sh = kd * 1440
        acc[sh:] += v5[:-sh]
        cnt[sh:] += 1
    base = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    vs = np.where(base > 0, v5 / base, np.nan)
    cand["vol_burst"] = np.flatnonzero(vs >= np.nanpercentile(vs, 99))

    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest"])
    oms = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10**6).to_numpy()
    oiv = o["sum_open_interest"].to_numpy(float)
    cut = ts - 5 * MIN_MS          # <-- OI 前視修正，見檔頭
    j_hi = np.searchsorted(oms, cut, side="right") - 1
    j_lo = np.searchsorted(oms, cut - W * MIN_MS, side="right") - 1
    ok = (j_lo >= 0) & (j_hi > j_lo)
    oc = np.full(n, np.nan)
    oc[ok] = (oiv[j_hi[ok]] - oiv[j_lo[ok]]) / oiv[j_lo[ok]] * 100
    cand["oi_crash"] = np.flatnonzero(oc <= np.nanpercentile(oc, 1))

    g = liq[liq.sym == sym].sort_values("w")
    if len(g):
        lw = g["w"].to_numpy(np.int64)
        lu = g["u"].to_numpy(float)
        lc = np.concatenate([[0.0], np.cumsum(lu)])
        hi = np.searchsorted(lw, ts, side="right")
        lo = np.searchsorted(lw, ts - W * MIN_MS, side="left")
        lsum = lc[hi] - lc[lo]
        inwin = (ts >= lw.min() + 86_400_000) & (ts <= lw.max())
        lsum = np.where(inwin, lsum, np.nan)
        pos = lsum[np.isfinite(lsum) & (lsum > 0)]
        thr = np.percentile(pos, 99) if len(pos) else np.inf
        cand["liq_burst"] = np.flatnonzero(np.nan_to_num(lsum, nan=-1) >= thr)

    ev = pd.read_parquet((EVENTS if events_dir is None else events_dir)
                         / f"{sym}.parquet", columns=["t_sweep"])
    cand["sweep"] = np.searchsorted(ts, ev["t_sweep"].to_numpy(np.int64) - MIN_MS)

    # 2026-09-07 附加：把**原始量值**一併回傳。因果版門檻（滾動 30 日分位）
    # 必須用同一份量值重新切門檻，不能自己再算一遍（mistake.md 2026-08-26）。
    # cand 的算法一個字沒動 —— 25 格逐格驗證過與重構前相同。
    q = {"delta_ext": ad, "vol_burst": vs, "oi_crash": oc}
    if "liq_burst" in cand:
        q["liq_burst"] = np.nan_to_num(lsum, nan=-1.0)
    return cand, ts, cl, at, day, q


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, liq_total_usd u "
                      "FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    names = ("sweep", "liq_burst", "delta_ext", "vol_burst", "oi_crash")
    per_type = {k: {} for k in names}

    for sym in CORE9:
        cand, ts, cl, at, day, _q = detect_all(sym, liq)
        for name, idx in cand.items():
            idx = cooldown_filter(np.sort(idx))
            for tau in TAUS:
                k, r = label(cl, at, idx, tau)
                per_type[name].setdefault(tau, {"r": [], "day": []})
                per_type[name][tau]["r"].append(r)
                per_type[name][tau]["day"].append(day[k])

    res = {}
    print("**這是篩選不是判決**：5 種事件 x 5 個 horizon = 25 格，全格報告。")
    print("任何一格都不構成結論——要下結論必須先有它自己的預註冊。")
    print()
    print(f"{'事件':11s} {'n':>7s} {'日':>5s} {'最大日佔變異':>11s}   "
          + "".join(f"{str(x) + 'm':>17s}" for x in TAUS))
    for name in names:
        dd = per_type[name]
        if not dd or sum(len(x) for x in dd[TAUS[0]]["r"]) < 50:
            print(f"{name:11s} (樣本不足)")
            continue
        cells, base = [], None
        for tau in TAUS:
            r = np.concatenate(dd[tau]["r"])
            dy = np.concatenate(dd[tau]["day"])
            s = day_stats(dy, r)
            res[f"{name}_{tau}"] = s
            if base is None:
                base = s
            cells.append(f"{s['mean']:+.4f}/{s['mde']:.4f}")
        print(f"{name:11s} {base['n']:7,d} {base['days']:5,d} "
              f"{base['var_top1'] * 100:10.2f}%   "
              + "".join(f"{c:>17s}" for c in cells))
    print()
    print("每格是「均值 / MDE」（ATR 單位）。**均值 > MDE 才是看得見的**。")
    print("MDE = 該事件主效應的最小可偵測量（日聚類 bootstrap，1.96 x SE）。")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_census.json").write_text(json.dumps(res, indent=2, default=float),
                                           encoding="utf-8")
    print("written ->", OUT / "event_census.json")


if __name__ == "__main__":
    main()
