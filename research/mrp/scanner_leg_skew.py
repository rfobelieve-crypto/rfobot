# -*- coding: utf-8 -*-
"""量 arb 掃描器的腿間偏移（TODO §1.39，2026-09-13）

===========================================================================
它在回答什麼
===========================================================================
`arb/engine/tools/scanner.py` 的 `scan_once` 把 `ts = int(time.time())` 在
**週期開頭取一次**，然後 `quote_all` 分**三段序列**抓，而整個週期的每一列
共用那一個 ts：

    1. kind=="hl"   ThreadPoolExecutor(4)   HL / xyz / para / mkts / io
    2. CEX          ThreadPoolExecutor(4)   okx / bitget / binance
    3. **lighter    普通 for 迴圈，每筆 time.sleep(REQ_SPACING)**

所以一列 CSV 的 `a_bid` 與 `b_bid` 可以是隔幾十秒的兩個報價。
偏移 δ 在格寬 T 之下注入約 **σ√(δ/T)** 的**假**價差，而那個假價差依建構
就是「一格之內完成的均值回歸」—— 也就是套利與 MRP 最想找的東西。

**這支不推論，它量。** 兩個量測，互為對照：

  A. **落後相關**：同一個標的在不同場館的報酬，lag-0 相關應該接近 1。
     同段應該 > 0.95；明顯低的那一群就是另一批，而 (1 − ρ) x 格寬 直接給
     偏移的量級。
  B. **跨儀器對照**：錄製家族走兩條常駐 WebSocket（`entropy_arb/feeds.py`，
     雙邊簿口在記憶體裡、同一瞬間讀兩本）= 同步是建構保證。同一個配對，
     錄製器的價差離散度就是真實的雜訊底，掃描器大出來的那一截就是偏移。

**改了 scanner 的抓取順序、worker 數或 REQ_SPACING 之後要回來重跑這一支**，
並同步更新 `mrp_crossvenue.FETCH_GROUPS` 與 `arblib/scan_rank.BLOCK_SKEW_S`
—— 那兩張表是手寫的，它們跟 `quote_all` 漂開的話，乾淨／污染的分組就失效了。

2026-09-13 的基準：okx/bitget 1 秒、HL 8 秒、**lighter 家族 53 秒**；
BTC 價差 std 錄製器 0.978 vs 掃描器 4.829（**4.9 倍**）。預測與實測逐項對上：
okx 0.48/0.71、bitget 0.54/0.75、HL 1.64/1.92、lighter **4.33/4.64**
（殘差一致的 ~0.25 bps 就是真正的跨場館基差雜訊，依 √(真²+假²) 疊加）。

> **⚠ 一個要當場更正的過度宣稱：「同段 = 同步」只對兩個執行緒池那兩段成立。**
> lighter 那一段是**普通 for 迴圈**，所以**段內**的兩條腿也不同時 ——
> 實測 `lighter−lighter-rh` 的價差 std 是 **2.29 bps 不是 0**，而且
> lighter 的 **lag−1 相關 0.295**（其他場館都 ≈0），代表它的報酬有一部分
> 漏到前一格去，偏移不只是次格的。
> 對 §1.39 的判決沒有影響（乾淨面板取的是 CEX 那組，它是真的執行緒池），
> 但 `FETCH_GROUPS` 把 lighter 當成一個同步段是**近似，不是事實**。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 2026-09-15: arb location comes from research/arb_home.py (ARB_HOME overrides), not a hardcoded path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arb_home  # noqa: E402
ARB = str(arb_home.HOME)
LOGS = os.path.join(ARB, "engine", "logs")

from mrp_crossvenue import BAR_SEC, MIN_OBS, VENUE_GROUP, _load  # noqa: E402

BASE = "binance"        # 偏移的基準腿（CEX 段）
SYM = "BTC"             # 控制配對：兩個深簿、最緊的價差，帶應該最小


def lagged(p, sym=SYM, base=BASE):
    """A. 落後相關 -> 推估偏移。"""
    w = p[p.sym == sym].pivot(index="t", columns="venue", values="mid")
    w = w.dropna(axis=1, thresh=MIN_OBS)
    if base not in w.columns:
        print("  **%s 不在 %s 的面板上 —— 沒有基準腿可比**" % (base, sym))
        return {}
    print("  %s，格寬 %d 秒，基準 = %s 的報酬" % (sym, BAR_SEC, base))
    print("  %-12s %-9s %8s %8s %8s   %s"
          % ("場館", "抓取段", "lag−1", "lag0", "lag+1", "推估偏移"))
    out = {}
    for c in w.columns:
        if c == base:
            continue
        k = w[[base, c]].dropna()
        if len(k) < MIN_OBS:
            continue
        d = np.diff(np.log(k.values), axis=0)
        cc = {}
        for L in (-1, 0, 1):
            a, b = d[:, 0], d[:, 1]
            if L > 0:
                a, b = a[:-L], b[L:]
            elif L < 0:
                a, b = a[-L:], b[:L]
            cc[L] = float(np.corrcoef(a, b)[0, 1])
        dt = max(0.0, 1.0 - cc[0]) * BAR_SEC
        out[c] = (cc[0], dt)
        # lag0 不是最大 -> 偏移超過一格，上面的線性推估就不適用了
        flag = "" if cc[0] >= max(cc[-1], cc[1]) else "  **偏移 > 一格**"
        print("  %-12s %-9s %8.3f %8.3f %8.3f   **%5.0f 秒**%s"
              % (c, VENUE_GROUP.get(c, "?"), cc[-1], cc[0], cc[1], dt, flag))
    return out


def injected(p, sym=SYM, base=BASE):
    """偏移注入多少假價差：預測 σ√(δ/T)，再跟實測對一次。"""
    w = p[p.sym == sym].pivot(index="t", columns="venue", values="mid")
    w = w.dropna(axis=1, thresh=MIN_OBS)
    if base not in w.columns:
        return
    r = np.diff(np.log(w[base].dropna().values))
    sd = float(np.nanstd(r) * 1e4)
    print("\n  %s 每 %d 秒的報酬 σ = **%.1f bps**" % (sym, BAR_SEC, sd))
    print("  %-24s %10s %10s" % ("配對", "預測", "實測"))
    for c in w.columns:
        if c == base:
            continue
        k = w[[base, c]].dropna()
        if len(k) < MIN_OBS:
            continue
        d = np.diff(np.log(k.values), axis=0)
        dt = max(0.0, 1.0 - float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])) * BAR_SEC
        s = (np.log(k[c]) - np.log(k[base])) * 1e4
        print("  %-24s %10.2f %10.2f"
              % ("%s−%s" % (c, base), sd * np.sqrt(dt / BAR_SEC), s.std()))
    # 同段的那一對就是雜訊底
    if {"lighter", "lighter-rh"} <= set(w.columns):
        k = w[["lighter", "lighter-rh"]].dropna()
        s = (np.log(k["lighter"]) - np.log(k["lighter-rh"])) * 1e4
        print("  %-24s %10s %10.2f   <- 同一段抓的，這是雜訊底"
              % ("lighter−lighter-rh", "0（同段）", s.std()))


def cross_instrument(p):
    """B. 錄製器（WS，同步）vs 掃描器（REST，分段）—— 同一個配對。"""
    print("\n  錄製家族（WS 雙邊常駐簿口，同步是建構保證）")
    print("  %-12s %8s %10s %10s" % ("配對", "n", "價差 std", "p90|偏離|"))
    for d in sorted(glob.glob(os.path.join(LOGS, "*"))):
        f = os.path.join(d, "minutes.csv")
        if not os.path.isfile(f):
            continue
        try:
            m = pd.read_csv(f, usecols=["premium_close_bps"])
        except Exception:                                   # noqa: BLE001
            continue
        v = pd.to_numeric(m["premium_close_bps"], errors="coerce").dropna()
        if len(v) < 200:
            continue
        print("  %-12s %8d %10.3f %10.3f"
              % (os.path.basename(d), len(v), v.std(),
                 np.percentile((v - v.mean()).abs(), 90)))

    print("\n  掃描器（REST 分段序列）—— 同一批配對")
    print("  %-24s %8s %10s %9s" % ("配對", "n", "價差 std", "腿差"))
    g = p.pivot_table(index=["sym", "t"], columns="venue", values="mid")
    for sym, a, b in (("BTC", "HL", "lighter-rh"), ("BTC", "HL", "lighter"),
                      ("BTC", "HL", "binance"), ("BTC", "binance", "okx"),
                      ("HYPE", "HL", "lighter"), ("NEAR", "HL", "lighter")):
        try:
            w = g.loc[sym, [a, b]].dropna()
        except Exception:                                   # noqa: BLE001
            continue
        if len(w) < MIN_OBS:
            continue
        s = (np.log(w[a]) - np.log(w[b])) * 1e4
        same = VENUE_GROUP.get(a) == VENUE_GROUP.get(b)
        print("  %-24s %8d %10.3f %9s"
              % ("%s %s/%s" % (sym, a, b), len(s), s.std(),
                 "同段" if same else "跨段"))


def main():
    p = _load()
    print("面板：%s 列｜%d 個 ticker｜%d 個場館｜%.1f 天"
          % (format(len(p), ","), p.sym.nunique(), p.venue.nunique(),
             (p.t.max() - p.t.min()) / 86400.0))
    print("\nA 落後相關 -> 推估偏移")
    lagged(p)
    injected(p)
    print("\nB 跨儀器對照")
    cross_instrument(p)
    print("\n判讀：同段的那一列是真實的跨場館雜訊底；跨段大出來的那一截是"
          "抓取順序注入的假價差。\n"
          "      改過 scanner 的抓取順序之後要回來重跑，並同步"
          "mrp_crossvenue.FETCH_GROUPS 與 scan_rank.BLOCK_SKEW_S。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
