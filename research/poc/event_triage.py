# -*- coding: utf-8 -*-
"""事件分流 —— 把五條事件流併成互斥的「事件時刻」，各自量事後反應。

為什麼是現在做這件事
    `event_overlap.py` 判出 **LOW**（P(oi_crash|sweep) = 12.20%，lift 8.42x）。
    使用者對這個分岔的處置寫在看數字之前：

        低重疊 -> 母現象是強制流不是 sweep。框架往上抬一層：研究對象是
                  「部位被強制銷毀時的價格反應」，sweep 和 oi_crash 是它的
                  兩種觀測窗口。**兩條線平行走，各自判定，不互相嵌套。**

    「不互相嵌套」是本檔的設計約束，不是修辭。所以這裡**不是**把 sweep 依
    oi_crash 分桶（那就是嵌套／條件化，也正是 MDE 會爆掉的那條路）。這裡是
    把五條流**併成一條互斥的時刻流**：每個時刻只屬於一個簽名，三條路
    （只有掃單／掃單＋強制流／只有強制流）因此可以放在同一支尺上比，
    而且沒有任何一格是另一格的子集。

分流規則（跑之前寫死）
    1. 五種事件各自的分鐘索引，沿用 `event_census.detect_all`（同一份偵測，
       不寫第二份；mistake.md 2026-08-26）。
    2. 把所有 (分鐘, 類型) 併起來按分鐘排序，**相鄰事件間隔 <= 5 分鐘就併成
       同一個時刻**（gap-based clustering；5 分鐘＝事件自己的聚合窗長）。
    3. 時刻的錨點 = 該群**最早**的那一分鐘（先發生的那件事）。
    4. 時刻之間再套 60 分鐘冷卻，與各類型自己的冷卻一致。
    5. 簽名 = 該群裡出現過的類型集合。

    步驟 2 的方向性要講清楚：合併會讓「只有 X」這一格變小、「X+Y」變大。
    這對「母現象是強制流」這個假設是**保守**的——它讓純掃單格更難拿到樣本，
    而純掃單正是要被挑戰的那一格。

標籤與對照（與 `event_census` 完全一致，不另立）
    impulse = sign(close(t) - close(t-5m))      只用 t 之前，不循環
    r_tau   = impulse x (close(t+tau) - close(t)) / ATR_h14
    MDE     = 1.96 x 日聚類 bootstrap SE（B=2000）

**這是刻畫不是判決。** 簽名是資料長出來的、不是事前註冊的，格數又多，
任何一格都不構成結論——要下結論必須有它自己的預註冊（§0.92 擋的正是挑格）。
本檔全格報告，並在每格印出「均值 / MDE」與最大單日佔變異。
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
SHORT = {"sweep": "S", "oi_crash": "O", "vol_burst": "V",
         "delta_ext": "D", "liq_burst": "L"}
MERGE_GAP = 5          # 分鐘：相鄰事件併成同一時刻
COOLDOWN = 60          # 分鐘：時刻之間的冷卻
TAUS = [5, 15, 30, 60, 240]
MIN_N = 100            # 低於此樣本數的簽名併進「其他」，不逐格解讀
OUT = HERE / "data" / "results"


def cluster(pairs):
    """[(minute, type)] -> [(anchor_minute, frozenset(types))]，gap-based。"""
    if not pairs:
        return []
    pairs = sorted(pairs)
    out, cur_start, cur_last, cur_set = [], pairs[0][0], pairs[0][0], {pairs[0][1]}
    for m, t in pairs[1:]:
        if m - cur_last <= MERGE_GAP:
            cur_last = m
            cur_set.add(t)
        else:
            out.append((cur_start, frozenset(cur_set)))
            cur_start, cur_last, cur_set = m, m, {t}
    out.append((cur_start, frozenset(cur_set)))
    # 時刻之間的冷卻：保留最早的，丟掉 60 分鐘內的後續時刻
    kept, last = [], -10 ** 9
    for a, s in out:
        if a - last >= COOLDOWN:
            kept.append((a, s))
            last = a
    return kept


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = {}          # sig -> {tau: {"r": [], "day": []}}
    sig_n = {}
    for sym in ec.CORE9:
        cand, ts, cl, at, day, _q = ec.detect_all(sym, liq)
        pairs = []
        for n in NAMES:
            v = cand.get(n)
            if v is None or len(v) == 0:
                continue
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), n))
        moments = cluster(pairs)
        by_sig = {}
        for a, s in moments:
            by_sig.setdefault(s, []).append(a)
        for s, idx in by_sig.items():
            sig_n[s] = sig_n.get(s, 0) + len(idx)
            arr = np.array(sorted(idx), dtype=np.int64)
            for tau in TAUS:
                k, r = ec.label(cl, at, arr, tau)
                d = rows.setdefault(s, {}).setdefault(tau, {"r": [], "day": []})
                d["r"].append(r)
                d["day"].append(day[k])

    total = sum(sig_n.values())
    order = sorted(sig_n, key=lambda s: -sig_n[s])
    big = [s for s in order if sig_n[s] >= MIN_N]
    small = [s for s in order if sig_n[s] < MIN_N]

    def name(s):
        return "+".join(SHORT[n] for n in NAMES if n in s)

    print(f"互斥事件時刻 {total:,} 個（併窗 {MERGE_GAP} 分、冷卻 {COOLDOWN} 分、九幣）")
    print(f"簽名 {len(order)} 種，其中 n >= {MIN_N} 的有 {len(big)} 種；"
          f"其餘 {len(small)} 種共 {sum(sig_n[s] for s in small):,} 個併為「其他」不逐格解讀")
    print("S=掃單 O=OI崩落 V=量能爆發 D=主動量極端 L=清算爆發"
          "（L 只覆蓋 BTC/ETH 77 天，含 L 的簽名天生稀少）\n")
    print("**這是刻畫不是判決**：簽名是資料長出來的、不是事前註冊的，任何一格都不構成結論。\n")

    hdr = (f"{'簽名':10s} {'n':>7s} {'佔比':>7s} {'日':>5s} {'最大日佔變異':>11s}   "
           + "".join(f"{str(x) + 'm':>17s}" for x in TAUS))
    print(hdr)
    res = {}
    for s in big:
        cells, base = [], None
        for tau in TAUS:
            r = np.concatenate(rows[s][tau]["r"])
            dy = np.concatenate(rows[s][tau]["day"])
            st = ec.day_stats(dy, r)
            res[f"{name(s)}_{tau}"] = st
            if base is None:
                base = st
            hit = "*" if abs(st["mean"]) > st["mde"] else " "
            cells.append(f"{st['mean']:+.4f}/{st['mde']:.4f}{hit}")
        print(f"{name(s):10s} {base['n']:7,d} {sig_n[s]/total*100:6.2f}% "
              f"{base['days']:5,d} {base['var_top1']*100:10.2f}%   "
              + "".join(f"{c:>17s}" for c in cells))

    print("\n每格「均值 / MDE」（ATR 單位）。**帶 * 的是均值絕對值超過自己的 MDE 的格**。")
    print("MDE = 該格的最小可偵測量（日聚類 bootstrap B=2000，1.96 x SE）。\n")

    # 三條路的並排（分流的重點）
    print("=== 三條路（互斥，沒有任何一格是另一格的子集）===\n")
    FLOW = {"oi_crash", "vol_burst", "delta_ext", "liq_burst"}
    lanes = {"只有掃單": lambda s: s == frozenset({"sweep"}),
             "掃單 + 強制流": lambda s: "sweep" in s and (s & FLOW),
             "只有強制流": lambda s: "sweep" not in s and (s & FLOW)}
    print(f"{'路':14s} {'n':>8s} {'佔比':>7s} {'日':>5s}   "
          + "".join(f"{str(x) + 'm':>17s}" for x in TAUS))
    lane_res = {}
    for ln, pred in lanes.items():
        sel = [s for s in order if pred(s)]
        if not sel:
            continue
        n_tot = sum(sig_n[s] for s in sel)
        cells, base = [], None
        for tau in TAUS:
            r = np.concatenate([x for s in sel for x in rows[s][tau]["r"]])
            dy = np.concatenate([x for s in sel for x in rows[s][tau]["day"]])
            st = ec.day_stats(dy, r)
            lane_res[f"{ln}_{tau}"] = st
            if base is None:
                base = st
            hit = "*" if abs(st["mean"]) > st["mde"] else " "
            cells.append(f"{st['mean']:+.4f}/{st['mde']:.4f}{hit}")
        print(f"{ln:14s} {base['n']:8,d} {n_tot/total*100:6.2f}% {base['days']:5,d}   "
              + "".join(f"{c:>17s}" for c in cells))
    res["lanes"] = lane_res
    res["signature_counts"] = {name(s): int(sig_n[s]) for s in order}
    res["total_moments"] = int(total)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "event_triage.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\nwritten ->", OUT / "event_triage.json")


if __name__ == "__main__":
    main()
