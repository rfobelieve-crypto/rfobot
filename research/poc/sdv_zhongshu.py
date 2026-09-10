# -*- coding: utf-8 -*-
"""纏論「中樞」當第五種流動性池子（2026-09-10 預註冊）

===========================================================================
為什麼只測這一個組件，不測那個系統
===========================================================================
使用者提供《白話纏論》。它有四個組件，而**這個專案已經分開量過三個**，
結果正好落在地形戰役那條分界線的兩側：

    分型 / 筆（純價格結構）    地形 S2（BOS/CHoCH）、D6 翻轉位   **全滅**
    MACD 背馳                B-P4 +1.6pp 幅度不足、回歸家族三連死  **FAIL**
    斐波那契 0.5/0.618 進場    2026-09-05 費波那契當均值回歸的錨    **NO-GO**
    **中樞當流動性位置**       地形 D1/D2/D3/D5                  **全部存活**

所以測的是第四個，而且只測「它當價位表好不好」，不測那套買賣點系統。

**明確不測的，以及理由**（不是漏掉）：

    死扛      原文「買入後沒有出現頂背馳一律不許平倉」= 無上限虧損。
              這個專案裸版網格 12 配置 9 爆倉、MDD 86-100%，不重蹈。
    背馳      三度量測為零資訊。
    斐波那契   已 NO-GO。
    絕對斷言   原文「上沿=賣下沿=買，這是 100% 正確的」「99.99% 會跌回去」
              —— 無樣本數、無成本、零失敗案例；不作為任何判準的依據。

===========================================================================
中樞的定義（凍結；照原文，零門檻參數）
===========================================================================
在 **1 小時 K**（與現行 swing 價位同尺度）上：

    分型    頂分型 = high 是連續三根的最大且嚴格大於左右；底分型鏡像
    筆      連接**交替**的頂底分型；同型相鄰時保留較極端的那個
    中樞    連續三筆（下-上-下 或 上-下-上）的重疊區間
            上沿 = 兩個高點裡**較低**的那個（Lower High）
            下沿 = 兩個低點裡**較高**的那個（Higher Low）
            必須真的重疊（下沿 < 上沿），否則不成立
    價位    上沿 -> buyside、下沿 -> sellside，各一個價位

**可知時刻**：第三筆的終點分型要靠右邊那根才能確認，所以價位在
**確認那根 1h K 的收盤**才成立（`ready_ms = 該根開盤 + 1h`）。嚴格無前視。

**一個必須申報的自由度**：原文刻意不給「筆」的最小長度
（「模糊的正確比精准的錯誤好」），而純三根分型在 1h 上會切出大量雜訊筆。
所以**兩種讀法都跑、都報，不挑**：

    zhongshu     MIN_BARS = 1   最貼近原文
    zhongshu5    MIN_BARS = 5   古典纏論的「筆至少五根」

===========================================================================
其餘一律沿用（不得帶進第二份實作）
===========================================================================
穿越判定（盤中價穿過價位 ±2 ticks）、一次消耗、流量條件 D/V、併窗、
冷卻、進場（ready+3 分市價）、出場（3 ATR / 480 分）、成本
—— 全部呼叫 `sdv_pools.run_pool(..., pool_fn=...)`，只換價位表。

===========================================================================
判準（照 §1.03p 原樣，跑之前凍結）
===========================================================================
    R1  該池子的 SDV 樣本外每筆淨值 > 0，且日聚類 CI 下緣 > 0
    R2  逐幣 >= 6/9
    R3  事件數比現行 swing 多（否則沒有「讓機會變多」的意義）
    R1 ∧ R2 ∧ R3 -> 可納入候選

自曝檢查
    S1  swing 那一格必須重現現行 SDV 的 1,584 筆（±60）
    S2  中樞的上沿必須嚴格大於下沿（構造保證，違反率必須為 0）
    S3  兩種讀法的價位數要有差異（若相同，代表 MIN_BARS 沒有起作用）

**附帶問題**（§1.05 留下的）：止損地圖那個變數跟波動度正交（|r| <= 0.09）
是罕見性質。中樞邊界是另一種「非極值的位置變數」，所以一併報
corr(中樞價位密度, ATR 分位) —— 若也正交，那是「重疊區域」這一類構造的
共同性質，不是巧合。
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
import sdv_pools as sp  # noqa: E402

OUT = HERE / "data" / "results" / "sdv_zhongshu.json"
HOUR_MS = 3_600_000
READINGS = {"zhongshu": 1, "zhongshu5": 5}


def _to_hourly(ts, hi, lo):
    """分鐘 -> 1 小時（UTC 整點）。回傳 (開盤 ms, high, low, 分鐘索引)。"""
    b = (np.asarray(ts, np.int64) // HOUR_MS) * HOUR_MS
    d = pd.DataFrame({"b": b, "h": hi, "l": lo})
    g = d.groupby("b").agg(h=("h", "max"), l=("l", "min"))
    return g.index.to_numpy(np.int64), g.h.to_numpy(float), g.l.to_numpy(float)


def _fractals(h, l):
    """三根分型。回傳 [(索引, 'top'|'bot', 價)]，時間序。"""
    out = []
    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1]:
            out.append((i, "top", float(h[i])))
        if l[i] < l[i - 1] and l[i] < l[i + 1]:
            out.append((i, "bot", float(l[i])))
    return out


def _strokes(fr, min_bars):
    """把分型串成交替的筆端點。同型相鄰保留較極端者；跨度不足 min_bars 跳過。"""
    seq = []
    for i, kind, px in fr:
        if seq and seq[-1][1] == kind:
            # 同型：保留更極端的那個
            if (kind == "top" and px > seq[-1][2]) or \
               (kind == "bot" and px < seq[-1][2]):
                seq[-1] = (i, kind, px)
            continue
        if seq and (i - seq[-1][0]) < min_bars:
            continue
        seq.append((i, kind, px))
    return seq


def zhongshu_pools(min_bars):
    """回傳 pool_fn(ts, hi, lo, tick) -> [(ready_ms, price, side)]。"""
    def fn(ts, hi, lo, tick):
        hts, hh, hl = _to_hourly(ts, hi, lo)
        seq = _strokes(_fractals(hh, hl), min_bars)
        out = []
        for k in range(3, len(seq)):
            a, b, c, d = seq[k - 3], seq[k - 2], seq[k - 1], seq[k]
            # 三筆 = 四個端點 a-b, b-c, c-d。兩個高點與兩個低點：
            hs = [p for _, kd, p in (a, b, c, d) if kd == "top"]
            ls = [p for _, kd, p in (a, b, c, d) if kd == "bot"]
            if len(hs) != 2 or len(ls) != 2:
                continue
            up_edge, dn_edge = min(hs), max(ls)      # Lower High / Higher Low
            if not (dn_edge < up_edge):
                continue                              # 沒有重疊 = 不是中樞
            # 可知時刻：第三筆終點分型要靠右邊那根確認 -> 該根收盤
            i_conf = d[0] + 1
            if i_conf >= len(hts):
                continue
            ready = int(hts[i_conf]) + HOUR_MS
            out.append((ready, float(up_edge), "buyside"))
            out.append((ready, float(dn_edge), "sellside"))
        return sorted(out)
    return fn


def main():
    res, base_n = {}, None
    print(f"{'池子':12} {'掃單':>8} {'SDV':>7} {'期間':8} {'淨/筆':>9} "
          f"{'SE':>7} {'CI下緣':>9} {'幣+':>6}")
    for kind in ("swing", *READINGS):
        fn = zhongshu_pools(READINGS[kind]) if kind in READINGS else None
        rows, nsw = [], 0
        for s in cb.CORE9:
            r, k = sp.run_pool(s, kind, pool_fn=fn)
            rows += r
            nsw += k
        d = pd.DataFrame(rows)
        if d.empty:
            print(f"{kind:12} {nsw:8,} {0:7,}  無事件")
            res[kind] = dict(n_sweeps=nsw, empty=True)
            continue
        mid = float(d.day.median())
        line = {"n_sweeps": nsw}
        for lab, sub in (("全期", d), ("樣本外", d[d.day >= mid])):
            m, se, lo = sp.boot(sub.day.values, sub.Rn.to_numpy())
            per = sub.groupby("sym").Rn.mean()
            line[lab] = dict(n=int(len(sub)), m=m, se=se, lo=lo,
                             npos=int((per > 0).sum()), nsym=int(len(per)))
            print(f"{kind if lab == '全期' else '':12} "
                  f"{nsw if lab == '全期' else '':>8} "
                  f"{len(d) if lab == '全期' else '':>7} {lab:8} "
                  f"{m:+9.4f} {se:7.4f} {lo:+9.4f} "
                  f"{int((per > 0).sum()):3d}/{len(per)}")
        res[kind] = line
        if kind == "swing":
            base_n = len(d)
        print()

    print("=" * 76)
    ok1 = abs(res["swing"]["全期"]["n"] - 1584) <= 60
    print(f"S1 swing 重現 {res['swing']['全期']['n']:,} 筆（現行 1,584）"
          f"  {'PASS' if ok1 else '**FAIL：穿越判定與 events.py 不一致**'}")
    # S2/S3：直接查價位表本身
    fn1, fn5 = zhongshu_pools(1), zhongshu_pools(5)
    bad = n1 = n5 = 0
    for s in cb.CORE9:
        b = pd.read_parquet(cb.BARS / f"{s}.parquet",
                            columns=["ts", "high", "low", "tick_size"])
        ts = b.ts.to_numpy(np.int64)
        hi = np.nan_to_num(b.high.to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b.low.to_numpy(float), nan=np.inf)
        tk = float(b.tick_size.iloc[0])
        p1, p5 = fn1(ts, hi, lo, tk), fn5(ts, hi, lo, tk)
        n1 += len(p1)
        n5 += len(p5)
        by = {}
        for r, px, sd in p1:
            by.setdefault(r, {})[sd] = px
        bad += sum(1 for v in by.values()
                   if "buyside" in v and "sellside" in v
                   and not (v["sellside"] < v["buyside"]))
    print(f"S2 上沿 > 下沿 的違反數 {bad}（構造保證，應為 0）"
          f"  {'PASS' if bad == 0 else '**FAIL**'}")
    print(f"S3 兩種讀法的價位數 {n1:,} vs {n5:,}"
          f"  {'PASS' if n1 != n5 else '**FAIL：MIN_BARS 沒起作用**'}")
    res["selfcheck"] = dict(s1=bool(ok1), s2_bad=int(bad),
                            n_levels_min1=int(n1), n_levels_min5=int(n5))

    print()
    print("判準（R1 樣本外淨>0 且 CI 下緣>0 ∧ R2 逐幣>=6/9 ∧ R3 事件比 swing 多）：")
    for kind in READINGS:
        v = res.get(kind, {})
        if v.get("empty") or "樣本外" not in v:
            print(f"  {kind:12} 無事件 -> 不過")
            continue
        o = v["樣本外"]
        r1 = o["m"] > 0 and o["lo"] > 0
        r2 = o["npos"] >= 6
        r3 = v["全期"]["n"] > base_n
        print(f"  {kind:12} R1{'✓' if r1 else '✗'} R2{'✓' if r2 else '✗'} "
              f"R3{'✓' if r3 else '✗'}（{v['全期']['n']:,} vs swing {base_n:,}）"
              f"  -> {'**可納入候選**' if (r1 and r2 and r3) else '不過'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
