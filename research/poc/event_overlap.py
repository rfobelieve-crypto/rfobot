# -*- coding: utf-8 -*-
"""事件重疊矩陣 —— 這四種事件是同一個現象的四個窗口，還是四件事？

使用者 2026-09-07 把分岔的處置**寫在看數字之前**：

    高重疊 -> 框架對，oi_crash 當條件變數，接受 MDE 上升，重跑。
    低重疊 -> 母現象是強制流不是 sweep。框架往上抬一層：研究對象是
              「部位被強制銷毀時的價格反應」，sweep 和 oi_crash 是它的
              兩種觀測窗口。兩條線平行走，各自判定，不互相嵌套。

判準的操作型定義（本檔在跑之前寫死，補上使用者未指定的門檻）
    共現窗 ±5 分鐘（＝事件自己的聚合窗長）。±30 分鐘做敏感度。
    以 P(Y|X) 為主軸，並列獨立假設下的期望值與 lift（只看原始數量會被
    基底發生率騙 —— 事件越多的類型天生越容易「重疊」）。

    HIGH    P(oi_crash | sweep) >= 0.50   -> 走第一條路
    LOW     P(oi_crash | sweep) <  0.20   -> 走第二條路
    MIDDLE  介於兩者                       -> 兩格都有樣本，oi_crash 可當
            條件變數，但兩條線同時也是可分的；**兩件事都報，不強行選邊**

    lift = P(Y|X) / P(Y)。lift ≈ 1 代表共現純粹來自基底發生率，
    即使 P(Y|X) 很高也**不構成**「同一個現象」的證據。

偵測邏輯 import 自 `event_census.detect_all` —— 不另寫一份（同一份資料
兩個偵測遲早會不同意，mistake.md 2026-08-26）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402

NAMES = ["sweep", "oi_crash", "vol_burst", "delta_ext", "liq_burst"]
WINDOWS = [5, 30]
OUT = HERE / "data" / "results"


def main():
    liq = None
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    idx = {n: [] for n in NAMES}     # 每幣一組（去重後的分鐘索引）
    minutes = {}
    for sym in ec.CORE9:
        cand, ts, cl, at, day = ec.detect_all(sym, liq)
        minutes[sym] = len(ts)
        for n in NAMES:
            v = cand.get(n)
            idx[n].append(ec.cooldown_filter(np.sort(v))
                          if v is not None and len(v) else np.array([], np.int64))

    tot_min = sum(minutes.values())
    counts = {n: int(sum(len(a) for a in idx[n])) for n in NAMES}
    print(f"分鐘總數 {tot_min:,}（九幣合計）")
    print("事件數（60 分鐘冷卻去重後）：",
          "  ".join(f"{n}={counts[n]:,}" for n in NAMES), "\n")

    res = {"minutes": tot_min, "counts": counts, "windows": {}}
    for W in WINDOWS:
        print(f"=== 共現窗 ±{W} 分鐘 ===\n")
        inter = {}
        for a in NAMES:
            for b in NAMES:
                if a == b:
                    inter[(a, b)] = counts[a]
                    continue
                c = 0
                for ia, ib in zip(idx[a], idx[b]):
                    if len(ia) == 0 or len(ib) == 0:
                        continue
                    pos = np.searchsorted(ib, ia)
                    lo = np.clip(pos - 1, 0, len(ib) - 1)
                    hi = np.clip(pos, 0, len(ib) - 1)
                    d = np.minimum(np.abs(ib[lo] - ia), np.abs(ib[hi] - ia))
                    c += int((d <= W).sum())
                inter[(a, b)] = c

        hdr = f"{'':11s}" + "".join(f"{n:>12s}" for n in NAMES)
        print(hdr)
        for a in NAMES:
            print(f"{a:11s}" + "".join(f"{inter[(a,b)]:12,d}" for b in NAMES))
        print("\n（第 a 列第 b 欄 = 有多少個 a 事件，在 ±%d 分鐘內伴隨至少一個 b 事件）\n" % W)

        print(f"{'X':11s} {'Y':11s} {'P(Y|X)':>8s} {'獨立期望':>9s} {'lift':>7s}")
        cells = {}
        for a in NAMES:
            for b in NAMES:
                if a == b or counts[a] == 0:
                    continue
                p = inter[(a, b)] / counts[a]
                base = min(1.0, counts[b] * (2 * W + 1) / tot_min)
                lift = p / base if base > 0 else float("nan")
                cells[f"{a}|{b}"] = dict(n_inter=inter[(a, b)], p=p,
                                         expected=base, lift=lift)
                if a in ("sweep", "oi_crash") or b in ("sweep", "oi_crash"):
                    print(f"{a:11s} {b:11s} {p*100:7.2f}% {base*100:8.2f}% "
                          f"{lift:7.2f}x")
        res["windows"][str(W)] = dict(
            matrix={f"{a}|{b}": inter[(a, b)] for a in NAMES for b in NAMES},
            cells=cells)
        print()

    p_os = res["windows"]["5"]["cells"]["sweep|oi_crash"]["p"]
    lift_os = res["windows"]["5"]["cells"]["sweep|oi_crash"]["lift"]
    branch = ("HIGH" if p_os >= 0.50 else "LOW" if p_os < 0.20 else "MIDDLE")
    print("=== 預註冊分岔（判準寫在看數字之前）===")
    print(f"  P(oi_crash | sweep) = {p_os*100:.2f}%   lift {lift_os:.2f}x"
          f"   -> **{branch}**")
    print({"HIGH": "  -> 框架對：oi_crash 當條件變數，接受 MDE 上升，重跑。",
           "LOW": "  -> 母現象是強制流不是 sweep：兩條線平行走，各自判定，不互相嵌套。",
           "MIDDLE": "  -> 兩格都有樣本：oi_crash 可當條件變數，但兩條線同時是可分的。"
                     "兩件事都報，不強行選邊。"}[branch])
    res["branch"] = dict(p_sweep_oi=p_os, lift=lift_os, verdict=branch)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_overlap.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\nwritten ->", OUT / "event_overlap.json")


if __name__ == "__main__":
    main()
