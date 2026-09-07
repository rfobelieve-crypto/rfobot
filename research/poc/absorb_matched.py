# -*- coding: utf-8 -*-
"""把 absorption 的套套邏輯成分剝掉——在「價格移動相同」的條件下比。

問題
    `absorption` = Kyle λ = 逐分鐘 Δprice 對 delta 的迴歸斜率。它的**分子含
    價格移動**，而事件的定義就是「價格創了新極值」。所以 BRIDGE.md 量到的
    AUC 0.816（vs 同日對照）有一部分是定義帶來的，不是市場結構。

作法
    對每個事件，在**同一天、非事件、離任何事件 >30 分鐘**的分鐘裡，挑一個
    ±5 分鐘窗的 **|Δprice|/ATR 最接近**的（caliper ±20%）。配對之後兩邊的
    價格移動幅度一樣，剩下的差異就不能再用「事件必然有大移動」解釋。

    配對之後 λ = |Δp| / delta 的分子被對齊，所以這個檢定實際在問：
        **同樣大小的價格移動，掃單那一分鐘是不是用更少的主動成交量做到的？**
    也就是「book 是不是更薄」。這是一個乾淨、可解釋、非套套的問題。

    因此同時直接報 |delta|——它比 λ 更直白，而且完全不含價格。

判準（跑之前寫死）
    配對成功率 < 50% -> 事件的移動幅度在同日找不到可比的分鐘，
                        標 INCONCLUSIVE-MATCH（結論不可下）
    AUC(λ) 配對後仍 > 0.60 -> absorption 的分離**不是**套套邏輯
    AUC(λ) 配對後落到 [0.45, 0.55] -> 原本那 0.816 幾乎全是價格移動帶來的
    中間 -> 部分是、部分不是，照數字報，不四捨五入成結論

**這是刻畫不是特徵**（窗口含 t_sweep 之後的資料）。
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
OUT = HERE / "data" / "results"
MIN_MS = 60_000
WIN = 5
GUARD_MIN = 30
CALIPER = 0.20
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
RNG = np.random.default_rng(20260907)


def lam_and_delta(cl, dl, atr, i_lo, i_hi):
    """Kyle λ（正規化）與 |delta| 總量，在 [i_lo, i_hi] 這段分鐘上。"""
    if i_hi - i_lo < 6 or not np.isfinite(atr) or atr <= 0:
        return np.nan, np.nan
    dp = np.diff(cl[i_lo:i_hi + 1])
    dd = dl[i_lo + 1:i_hi + 1]
    g = np.isfinite(dp) & np.isfinite(dd)
    if g.sum() < 6 or np.std(dd[g]) == 0:
        return np.nan, np.nan
    slope = np.polyfit(dd[g], dp[g], 1)[0]
    return slope * np.std(dd[g]) / atr, float(np.abs(dd[g]).sum())


def main():
    rows = []
    matched = tried = 0
    for sym in CORE9:
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "close", "delta", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        cl = b["close"].to_numpy(float)
        dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
        at = b["atr_h14"].to_numpy(float)
        ts0 = int(ts[0])
        n = len(ts)

        # 每一分鐘的 ±5 分鐘 |Δprice| / ATR（向量化，全序列）
        idx = np.arange(n)
        lo = np.clip(idx - WIN, 0, n - 1)
        hi = np.clip(idx + WIN, 0, n - 1)
        move = np.abs(cl[hi] - cl[lo]) / np.where(at > 0, at, np.nan)

        ev = pd.read_parquet(EVENTS / f"{sym}.parquet", columns=["t_sweep"])
        et = ev["t_sweep"].to_numpy(np.int64)
        e_i = ((et - ts0) // MIN_MS).astype(np.int64)
        e_i = e_i[(e_i >= WIN + 1) & (e_i < n - WIN - 1)]

        by_day = {}
        for k in e_i:
            by_day.setdefault(int((ts[k] // 86_400_000)), []).append(int(k))

        for k in e_i:
            tried += 1
            d0 = int(ts[k] // 86_400_000)
            same = np.array(by_day[d0], dtype=np.int64)
            day_lo = int(np.searchsorted(ts, d0 * 86_400_000))
            day_hi = int(np.searchsorted(ts, (d0 + 1) * 86_400_000)) - 1
            cand = np.arange(max(day_lo, WIN + 1), min(day_hi, n - WIN - 1) + 1)
            if len(cand) == 0:
                continue
            far = np.abs(cand[:, None] - same[None, :]).min(axis=1) > GUARD_MIN
            cand = cand[far]
            if len(cand) == 0:
                continue
            m_e = move[k]
            m_c = move[cand]
            ok = np.isfinite(m_c) & (np.abs(m_c - m_e) <= CALIPER * m_e)
            cand = cand[ok]
            if len(cand) == 0:
                continue
            j = int(cand[np.argmin(np.abs(move[cand] - m_e))])
            le, de = lam_and_delta(cl, dl, at[k], k - WIN, k + WIN)
            lc, dc = lam_and_delta(cl, dl, at[j], j - WIN, j + WIN)
            if not (np.isfinite(le) and np.isfinite(lc)):
                continue
            matched += 1
            rows.append(dict(sym=sym, move_e=m_e, move_c=move[j],
                             lam_e=le, lam_c=lc, absdelta_e=de, absdelta_c=dc))

    d = pd.DataFrame(rows)
    rate = matched / tried if tried else 0.0
    print(f"配對:{matched:,} / {tried:,} = {rate*100:.1f}%   "
          f"（caliper ±{CALIPER*100:.0f}%，同日、離事件 >{GUARD_MIN} 分鐘）")
    if rate < 0.50:
        print("\n配對成功率 < 50% -> INCONCLUSIVE-MATCH，結論不可下")
    print(f"配對品質：|Δp|/ATR 事件中位 {d.move_e.median():.4f}  "
          f"對照中位 {d.move_c.median():.4f}  "
          f"相對差中位 {(abs(d.move_e-d.move_c)/d.move_e).median()*100:.2f}%\n")

    def auc(p, q):
        p, q = p[np.isfinite(p)], q[np.isfinite(q)]
        a = np.concatenate([p, q])
        r = pd.Series(a).rank().to_numpy()
        return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))

    res = dict(matched=matched, tried=tried, rate=rate, caliper=CALIPER,
               move_med_event=float(d.move_e.median()),
               move_med_control=float(d.move_c.median()))
    print(f"{'量':16s} {'事件中位':>12s} {'對照中位':>12s} {'AUC':>8s}")
    for col, name, flip in (("lam", "Kyle λ（吸收弱＝高）", False),
                            ("absdelta", "|delta| 總量", True)):
        pe, pc = d[f"{col}_e"].to_numpy(float), d[f"{col}_c"].to_numpy(float)
        a = auc(-pe, -pc) if flip else auc(pe, pc)
        res[f"auc_{col}_matched"] = a
        res[f"{col}_med_event"] = float(np.nanmedian(pe))
        res[f"{col}_med_control"] = float(np.nanmedian(pc))
        print(f"{name:16s} {np.nanmedian(pe):12.4f} {np.nanmedian(pc):12.4f} {a:8.4f}")
    print("\n（|delta| 的 AUC 已翻號：>0.5 代表事件用**更少**主動量做到同樣的移動）")
    print(f"\n對照 BRIDGE.md 未配對時 AUC(λ vs 同日) = 0.8159")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "absorb_matched.json").write_text(json.dumps(res, indent=2, default=float),
                                             encoding="utf-8")
    d.to_parquet(OUT / "absorb_matched.parquet", index=False)
    print("written ->", OUT / "absorb_matched.json")


if __name__ == "__main__":
    main()
