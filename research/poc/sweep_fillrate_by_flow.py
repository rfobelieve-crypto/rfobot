# -*- coding: utf-8 -*-
"""回踩要求本身是不是已經把「延續型」掃單濾掉了？

背景
    分流量到：掃單伴隨清算流 -> 60 分鐘延續 +0.278 ATR；純掃單 -> 反轉。
    而凍結引擎是反轉策略，且 90.1% 的成交發生在穿越後 60 分鐘內——所以它
    確實進在延續窗之內，不是「延續走完才進場」。

    但 `sweep_liq_filter.py` 量到兩堆的 R 沒有顯著差別。可能的機制：
    **引擎的成交條件是「價格回踩到價位」，而延續最強的那些一去不回、
    根本不會回踩** —— 它們從來沒有成交過，所以不會出現在 R 的統計裡。

    如果這個機制成立，成交率應該隨清算流事件數**單調下降**。

    這也直接回答使用者的「你每個 swing 都進場啊」：**沒有**。
    掃單事件裡有相當比例從來沒成交，而被擋掉的正好是延續型的。

無前視：清算流的窗是穿越分鐘 ±5 分，嚴格早於成交（成交在 j+1..j+W）。
本檔只數成交率，不碰報酬，所以沒有標籤循環的問題。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402
import conj_causal as cc  # noqa: E402

BARS = HERE / "data" / "bars"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
OUT = HERE / "data" / "results"
FLOW = ("delta_ext", "vol_burst", "oi_crash")
HOUR_MS = 3_600_000
NEAR_MS = 5 * 60_000


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    for sym in ec.CORE9:
        _c, mts, _cl, _at, day, q = ec.detect_all(sym, liq)
        caus = cc.causal_flags(q, day)
        parts = [caus.get(k, np.array([], np.int64)) for k in FLOW]
        fm = np.sort(np.concatenate(parts)) if any(len(p) for p in parts) \
            else np.array([], np.int64)
        fts = mts[fm] if len(fm) else np.array([], np.int64)

        mb = pd.read_parquet(BARS / f"{sym}.parquet", columns=["ts", "high", "low"])
        bts = mb["ts"].to_numpy(np.int64)
        bhi = np.nan_to_num(mb["high"].to_numpy(float), nan=-np.inf)
        blo = np.nan_to_num(mb["low"].to_numpy(float), nan=np.inf)

        b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        h = [x[sc.H] for x in b1]
        lo = [x[sc.L] for x in b1]
        n = len(b1)
        for e in sc.detect_sweeps(b1):
            j, lvl = e["j"], e["level"]
            kd = 1 if e["kind"] == "buy" else -1
            h0 = int(b1[j][0]) * 1000
            a = int(np.searchsorted(bts, h0, side="left"))
            z = int(np.searchsorted(bts, h0 + HOUR_MS, side="left"))
            if z <= a:
                continue
            seg = (bhi[a:z] > lvl) if kd == 1 else (blo[a:z] < lvl)
            nz = np.flatnonzero(seg)
            if not len(nz):
                continue
            pm = int(bts[a + int(nz[0])])
            i0 = int(np.searchsorted(fts, pm - NEAR_MS, side="left"))
            i1 = int(np.searchsorted(fts, pm + NEAR_MS, side="right"))
            filled = False
            for f in range(j + 1, min(j + 1 + sc.W, n)):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    filled = True
                    break
            rows.append((sym, int(i1 - i0), bool(filled)))

    d = pd.DataFrame(rows, columns=["sym", "n_flow", "filled"])
    print(f"全部掃單事件（可定位穿越分鐘）{len(d):,}")
    print(f"其中回踩成交 {int(d.filled.sum()):,}  "
          f"= {d.filled.mean() * 100:.2f}%  "
          f"（**沒成交的那些從來不會出現在回測圖上**）")
    print()
    print(f"{'流事件數':>8s} {'掃單數':>8s} {'佔比':>7s} {'成交':>8s} {'成交率':>8s}")
    cells = {}
    for lab, sel in (("0", d.n_flow == 0), ("1", d.n_flow == 1),
                     ("2", d.n_flow == 2), ("3+", d.n_flow >= 3)):
        g = d[sel]
        if not len(g):
            continue
        cells[lab] = float(g.filled.mean())
        print(f"{lab:>8s} {len(g):8,d} {len(g)/len(d)*100:6.2f}% "
              f"{int(g.filled.sum()):8,d} {g.filled.mean()*100:7.2f}%")
    if "0" in cells and "3+" in cells:
        print()
        print(f"  0 個 vs 3+ 個的成交率差 "
              f"{(cells['0'] - cells['3+']) * 100:+.2f} pp")
        mono = all(cells[a] >= cells[b] for a, b in
                   zip(["0", "1", "2"], ["1", "2", "3+"]) if a in cells and b in cells)
        print(f"  單調下降（流越多越不成交）-> {'是' if mono else '否'}")
    print()
    print("逐幣成交率（0 個 vs 3+ 個）：")
    t = d.assign(grp=np.where(d.n_flow == 0, "0", np.where(d.n_flow >= 3, "3+", "1-2")))
    print(t[t.grp != "1-2"].groupby(["sym", "grp"]).filled.mean().unstack()
          .to_string(float_format=lambda x: f"{x*100:5.1f}%"))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "sweep_fillrate_by_flow.parquet", index=False)
    print()
    print("written ->", OUT / "sweep_fillrate_by_flow.parquet")


if __name__ == "__main__":
    main()
