# -*- coding: utf-8 -*-
"""合取檢定 — 插針 × 清算，哪一個才是單位？

使用者 2026-09-06 的修正（比我原本的框架準確）：
    我量的是「事件有沒有伴隨清算」，AUC 0.95。但對照組是隨機分鐘，而隨機
    分鐘大多安靜——那個 0.95 有一部分只是在說「有事 vs 沒事」。
    **清算本身很常見**，所以「我們的事件裡有清算」不必然有判別力。
    有判別力的是**合取**：價格軌跡的運動事件 × 被迫成交。

    而且合取更好測量：價格軌跡是完整記錄的（1 分鐘 OHLCV，缺 0 根），
    清算是殘缺的（23.6%）。用軌跡把時刻框出來、再問框裡的流，比單獨追一條
    殘缺的事件流穩。

問題（跑之前先寫死，跑完照同一份交代）
    Q1 反向基底率：清算爆發裡，有多少落在掃單事件附近？
       若絕大多數不是 -> 清算單獨沒有判別力，合取才是單位（支持使用者）。
       若絕大多數都是 -> 兩者幾乎同一件事，合取沒有新增資訊。
    Q2 合取有沒有新增資訊：比較三種錨點之後的行為
       A 插針 + 清算   （我們的事件，且該窗錄到清算）
       B 插針 + 幾乎無清算
       C 清算爆發 + 無插針（距任何事件 >30 分鐘）
       量 forward |return| 與方向性 return，以 ATR 正規化。
       若 A 與 C 的分布幾乎相同 -> 插針沒有新增資訊。
       若 A 與 B 幾乎相同 -> 清算沒有新增資訊。
       兩者都不同 -> 合取是單位。

    註:A/B 的方向欄以 side 取符號(延續為正);C 沒有 side,方向欄不可與 A/B 比,
    只有 |r| 可比。

    **這是刻畫，不是可行動的判決。** 任何要拿去改規則的主張都要走它自己的
    預註冊。這裡只回答「我們在測的是什麼」。

資料限制（先寫，免得結果被誤讀）
    清算：BTC/ETH、2026-03-31 起、完整率 **23.6%**，且漏失隨強度惡化。
    所以「我們沒錄到清算」≠「市場沒有清算」。方向上幫得上忙——低強度時
    我們錄得相對準（Q1 覆蓋率 0.321 vs Q5 0.110），所以「幾乎沒錄到」比
    「錄到很多」可信。B 組因此比 A 組脆弱，判讀時要記得。
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
WIN = 5                      # +-5 min window
GUARD_MIN = 30               # C group must be >30 min from any event
SYMS = ["BTC", "ETH"]        # the only coins with liquidation coverage
TAU = 3600_000               # 1h forward horizon


def load_liq():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    c = get_db_conn()
    d = pd.read_sql("SELECT canonical_symbol s, window_start w, liq_total_usd u, "
                    "liq_count k FROM liquidation_1m", c)
    c.close()
    d["sym"] = d["s"].str.replace("-USD", "", regex=False)
    return d


def rolling_window_sum(w, u, anchors, half=WIN):
    """sum of u over [anchor-half, anchor+half] minutes, w sorted."""
    cum = np.concatenate([[0.0], np.cumsum(u)])
    lo = np.searchsorted(w, anchors - half * MIN_MS, side="left")
    hi = np.searchsorted(w, anchors + half * MIN_MS, side="right")
    return cum[hi] - cum[lo]


def main():
    liq = load_liq()
    res = {}
    rows_all = []
    print("=== Q1 反向基底率：清算爆發裡有多少落在掃單事件附近？ ===\n")
    for sym in SYMS:
        g = liq[liq.sym == sym].sort_values("w")
        w = g["w"].to_numpy(np.int64)
        u = g["u"].to_numpy(float)
        lo_ms, hi_ms = int(w.min()), int(w.max())

        ev = pd.read_parquet(EVENTS / f"{sym}.parquet", columns=["t_sweep", "side"])
        et = ev["t_sweep"].to_numpy(np.int64)
        et = et[(et >= lo_ms + 86_400_000) & (et <= hi_ms)]

        # 每一分鐘的 +-5min 清算量（只在有錄到的分鐘上算）
        wsum = rolling_window_sum(w, u, w)
        # 每一分鐘離最近事件多遠
        idx = np.searchsorted(et, w)
        d_prev = np.where(idx > 0, w - et[np.clip(idx - 1, 0, len(et) - 1)], 1 << 60)
        d_next = np.where(idx < len(et), et[np.clip(idx, 0, len(et) - 1)] - w, 1 << 60)
        dist = np.minimum(d_prev, d_next) / MIN_MS

        base = float((dist <= WIN).mean())
        print(f"  {sym}: 錄到清算的分鐘 {len(w):,}，其中距事件 <= {WIN} 分鐘的佔 "
              f"**{base*100:.2f}%**（基底率）")
        qs = np.percentile(wsum[wsum > 0], [50, 75, 90, 95, 99])
        for lab, th in zip(["p50", "p75", "p90", "p95", "p99"], qs):
            m = wsum >= th
            near = float((dist[m] <= WIN).mean())
            print(f"      清算窗量 >= {lab} (${th:,.0f}): {int(m.sum()):6,d} 分鐘，"
                  f"其中 {near*100:5.2f}% 在事件附近  (lift {near/base:.2f}x)")
            res[f"{sym}_near_{lab}"] = near
        res[f"{sym}_base_rate"] = base
        print()

    print("=== Q2 三種錨點之後的行為（ATR 正規化，1 小時）===\n")
    for sym in SYMS:
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "close", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        cl = b["close"].to_numpy(float)
        atr = b["atr_h14"].to_numpy(float)
        pos = {t: i for i, t in enumerate(ts)}

        g = liq[liq.sym == sym].sort_values("w")
        w = g["w"].to_numpy(np.int64)
        u = g["u"].to_numpy(float)
        lo_ms, hi_ms = int(w.min()), int(w.max())
        wsum_at = lambda a: rolling_window_sum(w, u, a)

        ev = pd.read_parquet(EVENTS / f"{sym}.parquet", columns=["t_sweep", "side"])
        ev = ev[(ev.t_sweep >= lo_ms + 86_400_000) & (ev.t_sweep <= hi_ms - TAU)]
        et = ev["t_sweep"].to_numpy(np.int64)
        eliq = wsum_at(et)
        # 門檻必須跟被比較的量同尺度:窗口總和要對窗口總和的分布比,不能對
        # 單一分鐘的分位數比(2026-09-06 第一版就是這樣把 B 組清成 n=1 的)。
        all_w = wsum_at(w)
        thr_lo = np.percentile(all_w, 10)          # 「幾乎無清算」= 窗口和的 p10
        thr_hi = np.percentile(all_w, 50)

        # C 組：清算爆發但遠離任何事件
        allev = pd.read_parquet(EVENTS / f"{sym}.parquet", columns=["t_sweep"])
        aet = allev["t_sweep"].to_numpy(np.int64)
        wsum = wsum_at(w)
        idx = np.searchsorted(aet, w)
        dprev = np.where(idx > 0, w - aet[np.clip(idx - 1, 0, len(aet) - 1)], 1 << 60)
        dnext = np.where(idx < len(aet), aet[np.clip(idx, 0, len(aet) - 1)] - w, 1 << 60)
        far = np.minimum(dprev, dnext) / MIN_MS > GUARD_MIN
        cmask = far & (wsum >= np.median(eliq)) & (w <= hi_ms - TAU)

        sides = ev["side"].to_numpy()
        cont = np.where(sides == "sellside", -1.0, 1.0)

        def fwd(anchors, sign=None):
            out = []
            for k, t in enumerate(anchors):
                i, j = pos.get(int(t) - MIN_MS), pos.get(int(t) + TAU - MIN_MS)
                if i is None or j is None:
                    continue
                a = atr[i]
                if not np.isfinite(a) or a <= 0:
                    continue
                s = 1.0 if sign is None else sign[k]
                out.append(s * (cl[j] - cl[i]) / a)
            return np.array(out)

        mA = eliq >= thr_hi
        mB = eliq <= thr_lo
        print(f"    門檻(窗口和): p10=${thr_lo:,.0f}  p50=${thr_hi:,.0f}   "
              f"事件落在 A={int(mA.sum())}  B={int(mB.sum())}  "
              f"中間={int(len(et)-mA.sum()-mB.sum())}")
        groups = {
            "A 插針+清算": fwd(et[mA], cont[mA]),
            "B 插針+幾乎無清算": fwd(et[mB], cont[mB]),
            "C 清算+無插針": fwd(w[cmask]),
        }
        print(f"  {sym}")
        print(f"    {'組':22s} {'n':>6s} {'|r| 中位':>10s} {'|r| q90':>10s} "
              f"{'延續r中位':>10s} {'r std':>8s}")
        for name, x in groups.items():
            if len(x) < 20:
                print(f"    {name:22s} {len(x):6d}   (樣本不足)")
                continue
            print(f"    {name:22s} {len(x):6,d} {np.median(np.abs(x)):10.4f} "
                  f"{np.percentile(np.abs(x),90):10.4f} {np.median(x):+9.4f} "
                  f"{x.std():8.4f}")
            res[f"{sym}_{name}"] = dict(n=int(len(x)),
                                        abs_med=float(np.median(np.abs(x))),
                                        abs_q90=float(np.percentile(np.abs(x), 90)),
                                        med=float(np.median(x)), std=float(x.std()))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conjunction.json").write_text(json.dumps(res, indent=2, default=float),
                                          encoding="utf-8")
    print("written ->", OUT / "conjunction.json")


if __name__ == "__main__":
    main()
