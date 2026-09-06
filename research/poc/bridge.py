# -*- coding: utf-8 -*-
"""橋樑檢定 — 我們標的「價格幾何」事件，伴不伴隨真實的被迫成交？

使用者 2026-09-06 提出的未確認前提：
    系統標的是**價格軌跡的幾何**（影線刺穿前低、收不收回）。
    機制假設講的是**成交的來源**（停損被觸發、清算引擎被觸發）。
    兩者不是同一件事，中間的橋樑從來沒被驗證過。

    如果一半的事件沒有實質被迫成交，那 POC 距離的解釋力當然被稀釋
    ——不是假設錯，是母體混了。而 OI accel 跑出平坦結果時，
    「機制不成立」與「事件裡真清算佔比太低」在現行設計下分不出來。

三個欄位，掃單當下 ±5 分鐘：
    volume_spike  該窗成交量 / 前 30 日同時段同長度窗的中位數
    oi_drop_pct   該窗未平倉量變化（%），負值 = 部位被銷毀
    liq_usd       該窗清算名目（只有 BTC/ETH、159 天、且完整率 23.6%）

**這是刻畫不是特徵。** 窗口含 t_sweep 之後的資料，所以它回答「這個事件
當時發生了什麼」，不能直接拿來當濾網——真要當濾網必須改成只用事前資訊，
否則是前視。

**對照組是這支的重點。** 「多數事件的三個欄位接近零」單獨看沒有意義：
要跟隨機時刻比才知道分不分得開。每個事件配一個對照——**同一幣、同一個
UTC 小時、不同天**，把日內節律與幣別差異都消掉。分離度用 AUC 表示：
0.5 = 掃單事件跟隨機時刻在被迫流足跡上完全一樣（橋樑不成立）。

資料覆蓋（2026-09-06 量過，寫在這裡免得結果被誤讀）
    oi_drop_pct   九幣、全歷史、5 分鐘粒度            完整
    volume_spike  九幣、全歷史、1 分鐘                完整
    liq_usd       BTC/ETH、2026-03-31 起、**完整率 23.6%**，
                  且漏失隨強度惡化（大時段 2.3%）——只能當交叉檢查，
                  不能當主證據。用它檢驗前兩個是不是被迫流的好代理。
"""
from __future__ import annotations

import argparse
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
WIN_MIN = 5                     # +-5 minutes
BASE_DAYS = 30
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
RNG = np.random.default_rng(20260906)


def window_volume(cum, pos, t, ts0):
    """Sum of 1m volume over [t-5min, t+5min], via a prefix-sum lookup."""
    lo = (t - WIN_MIN * MIN_MS - ts0) // MIN_MS
    hi = (t + WIN_MIN * MIN_MS - ts0) // MIN_MS
    lo = np.clip(lo, 0, len(cum) - 1)
    hi = np.clip(hi, 0, len(cum) - 1)
    return cum[hi] - cum[lo]


def build_for(sym, anchors):
    """anchors: int64 ms array.  Returns DataFrame of the three columns."""
    b = pd.read_parquet(BARS / f"{sym}.parquet", columns=["ts", "volume"])
    ts0 = int(b["ts"].iloc[0])
    v = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    cum = np.concatenate([[0.0], np.cumsum(v)])

    win = window_volume(cum, None, anchors, ts0)
    # baseline: the same 11-minute window on each of the previous 30 days
    base = np.empty((BASE_DAYS, len(anchors)))
    for k in range(1, BASE_DAYS + 1):
        base[k - 1] = window_volume(cum, None, anchors - k * 86_400_000, ts0)
    med = np.median(base, axis=0)
    spike = np.where(med > 0, win / med, np.nan)

    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest"])
    o["ms"] = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10**6)
    ots = o["ms"].to_numpy(np.int64)
    oi = o["sum_open_interest"].to_numpy(float)
    i_lo = np.searchsorted(ots, anchors - WIN_MIN * MIN_MS, side="right") - 1
    i_hi = np.searchsorted(ots, anchors + WIN_MIN * MIN_MS, side="right") - 1
    ok = (i_lo >= 0) & (i_hi >= 0) & (i_hi > i_lo)
    oi_pct = np.full(len(anchors), np.nan)
    oi_pct[ok] = (oi[i_hi[ok]] - oi[i_lo[ok]]) / oi[i_lo[ok]] * 100.0
    return pd.DataFrame(dict(sym=sym, anchor=anchors, win_volume=win,
                             volume_spike=spike, oi_drop_pct=oi_pct))


def liq_for(anchors_by_sym):
    """liq_total_usd in the window.  BTC/ETH only, 2026-03-31 onward."""
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    c = get_db_conn()
    d = pd.read_sql("SELECT canonical_symbol s, window_start w, liq_total_usd u "
                    "FROM liquidation_1m", c)
    c.close()
    d["sym"] = d["s"].str.replace("-USD", "", regex=False)
    out = {}
    for sym, anchors in anchors_by_sym.items():
        g = d[d.sym == sym]
        if g.empty:
            continue
        w = g["w"].to_numpy(np.int64)
        u = g["u"].to_numpy(float)
        order = np.argsort(w)
        w, u = w[order], u[order]
        cum = np.concatenate([[0.0], np.cumsum(u)])
        lo = np.searchsorted(w, anchors - WIN_MIN * MIN_MS, side="left")
        hi = np.searchsorted(w, anchors + WIN_MIN * MIN_MS, side="right")
        covered = (anchors >= w.min() + 86_400_000) & (anchors <= w.max())
        vals = np.where(covered, cum[hi] - cum[lo], np.nan)
        out[sym] = vals
    return out


def auc(pos, neg):
    """P(random event > random control), NaN-safe."""
    p = pos[np.isfinite(pos)]
    n = neg[np.isfinite(neg)]
    if len(p) < 20 or len(n) < 20:
        return np.nan
    allv = np.concatenate([p, n])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    a = ap.parse_args()
    syms = [s for s in (x.strip().upper() for x in a.syms.split(",")) if s
            and (OI / f"{s}.parquet").exists()]
    print("coins with OI on disk:", ", ".join(syms), "\n")

    ev_frames, ct_frames, anchors_ev, anchors_ct = [], [], {}, {}
    for s in syms:
        ev = pd.read_parquet(EVENTS / f"{s}.parquet", columns=["t_sweep", "side"])
        t = ev["t_sweep"].to_numpy(np.int64)
        # control: same coin, same UTC hour-of-day, a different day
        offs = RNG.integers(1, 300, len(t)) * 86_400_000 * RNG.choice([-1, 1], len(t))
        lo = pd.read_parquet(BARS / f"{s}.parquet", columns=["ts"])["ts"]
        c = np.clip(t + offs, int(lo.iloc[0]) + 31 * 86_400_000, int(lo.iloc[-1]))
        anchors_ev[s], anchors_ct[s] = t, c
        e = build_for(s, t)
        e["side"] = ev["side"].to_numpy()
        e["kind"] = "event"
        k = build_for(s, c)
        k["side"] = ev["side"].to_numpy()
        k["kind"] = "control"
        ev_frames.append(e)
        ct_frames.append(k)

    liq_e = liq_for(anchors_ev)
    liq_c = liq_for(anchors_ct)
    for f in ev_frames:
        s = f["sym"].iloc[0]
        f["liq_usd"] = liq_e.get(s, np.full(len(f), np.nan))
    for f in ct_frames:
        s = f["sym"].iloc[0]
        f["liq_usd"] = liq_c.get(s, np.full(len(f), np.nan))
    E = pd.concat(ev_frames, ignore_index=True)
    C = pd.concat(ct_frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([E, C], ignore_index=True).to_parquet(OUT / "bridge.parquet", index=False)

    res = {"n_events": int(len(E)), "n_controls": int(len(C)), "coins": syms}
    print(f"events={len(E):,}  controls={len(C):,}\n")
    print("=== 分布：事件 vs 對照（同幣、同 UTC 小時、不同天）===\n")
    print(f"{'欄位':14s} {'組':8s} {'n':>6s} {'q10':>9s} {'q25':>9s} "
          f"{'中位':>9s} {'q75':>9s} {'q90':>9s}")
    for col in ("volume_spike", "oi_drop_pct", "liq_usd"):
        for name, D in (("event", E), ("control", C)):
            x = D[col].dropna()
            if len(x) < 20:
                print(f"{col:14s} {name:8s} {len(x):6d}   (資料不足)")
                continue
            q = x.quantile([.1, .25, .5, .75, .9])
            print(f"{col:14s} {name:8s} {len(x):6,d} " +
                  " ".join(f"{q.iloc[i]:9.3f}" for i in range(5)))
        print()

    print("=== 分離度 AUC（0.5 = 掃單事件與隨機時刻在該欄位上完全一樣）===\n")
    for col in ("volume_spike", "oi_drop_pct", "liq_usd"):
        # oi_drop: 部位銷毀是負值，取負號讓「更多銷毀」= 更大
        pe = -E[col].to_numpy(float) if col == "oi_drop_pct" else E[col].to_numpy(float)
        pc = -C[col].to_numpy(float) if col == "oi_drop_pct" else C[col].to_numpy(float)
        v = auc(pe, pc)
        res[f"auc_{col}"] = v
        print(f"  {col:14s} AUC = {v:.4f}" if np.isfinite(v) else
              f"  {col:14s} AUC = n/a")
        for side in ("sellside", "buyside"):
            m1, m0 = E.side == side, C.side == side
            vs = auc(pe[m1.to_numpy()], pc[m0.to_numpy()])
            res[f"auc_{col}_{side}"] = vs
            print(f"      {side:9s} {vs:.4f}" if np.isfinite(vs) else
                  f"      {side:9s} n/a")

    print("\n=== 「沒有被迫流足跡」的事件佔多少 ===\n")
    for th in (1.0, 1.25, 1.5, 2.0):
        f = float((E.volume_spike.dropna() < th).mean())
        fc = float((C.volume_spike.dropna() < th).mean())
        print(f"  volume_spike < {th:<5.2f}  事件 {f*100:5.1f}%   對照 {fc*100:5.1f}%")
        res[f"frac_spike_lt_{th}"] = f
    fe = float((E.oi_drop_pct.dropna() >= 0).mean())
    fc = float((C.oi_drop_pct.dropna() >= 0).mean())
    print(f"  oi_drop_pct >= 0（沒有部位銷毀）  事件 {fe*100:5.1f}%   對照 {fc*100:5.1f}%")
    res["frac_no_oi_drop"] = fe

    print("\n=== 代理效度：在看得到清算的地方，前兩欄追不追得上 liq_usd ===\n")
    sub = E.dropna(subset=["liq_usd"])
    if len(sub) > 100:
        from scipy.stats import spearmanr
        for col in ("volume_spike", "oi_drop_pct"):
            m = sub.dropna(subset=[col])
            x = -m[col] if col == "oi_drop_pct" else m[col]
            rho = spearmanr(x, m.liq_usd).correlation
            res[f"spearman_{col}_vs_liq"] = float(rho)
            print(f"  spearman({col}, liq_usd) = {rho:+.4f}   n={len(m):,}")
        print(f"  （覆蓋：{len(sub):,} / {len(E):,} 筆事件 = {len(sub)/len(E)*100:.1f}%）")
    else:
        print("  重疊樣本不足")

    (OUT / "bridge.json").write_text(json.dumps(res, indent=2, default=float),
                                     encoding="utf-8")
    print("\nwritten ->", OUT / "bridge.json")


if __name__ == "__main__":
    main()
