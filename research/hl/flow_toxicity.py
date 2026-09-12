# -*- coding: utf-8 -*-
"""逐標的的**流量毒性**：做市方在每一筆成交之後賺還是賠（markout 篩選）

來源：Advanced Market Making（2025-08-02，付費）「Being selective of what
you quote」一節：

    「在你還沒有自己的 markout 可以依靠之前，你可以用**那個標的上每一筆成交
      的平均 markout（比如 5 秒）**，追蹤它的 EWMA。你會發現能維持很好
      markout 的，是超級無毒的配對。……**動態選擇要報什麼、只在有 edge 時
      報價，是把一個缺 edge 的系統變成獲利或打平的好方法。**」

這支回答一個我們一直說「完全未量」的問題：**Lighter / HL 上哪些標的的流量
不帶毒**。它不判決任何東西，也不宣稱任何損益。

── 為什麼是 1 分鐘不是 5 秒 ────────────────────────────────────────
他的 5 秒是 HFT 報價商的尺度。我們的錄製是**分鐘級**（`hl_mid.py` 牆鐘 60 秒
取樣），而且我們要做的是分鐘頻率的策略 —— markout 的期間要配**真正的持有
期**，不然量到的不是我們會遇到的東西。5 秒的版本需要秒級簿口，我們沒有。

── 構造 ───────────────────────────────────────────────────────────
吃單方向：HL 的 `side` 'B' = 吃單方買、'A' = 吃單方賣（與 feeds.py 一致）。

    吃單方 markout = d x (mid(t+H) - px) / px x 1e4      d = +1(B) / -1(A)
    **做市方 markout = 上式取負**

**做市方 markout 為正 = 流量不帶毒**（吃單的人平均是賠的）。
以**成交金額加權**（一筆 100 萬的單與一筆 10 元的單不該同權）。

價格用**中價不用成交價**（mistake.md 2026-09-11：成交價在薄的標的上自帶負
自相關，會偽裝成均值回歸）。所以只取兩邊資料都有的那幾天。

── 自曝關 ─────────────────────────────────────────────────────────
D1 兩個方向的筆數要大致相當（B 與 A 各佔 30~70%）；某一側壓倒性代表方向
   欄位讀錯了。
D2 同一筆成交在 t 的中價與成交價不得差太多（中位 |px/mid-1| < 50 bps），
   否則是對齊錯了而不是市場。
D3 全格報告、不挑標的。

跑法：
    python research/hl/flow_toxicity.py            # 預設 1 分鐘 markout
    python research/hl/flow_toxicity.py --h 5      # 5 分鐘
    python research/hl/flow_toxicity.py --days 10
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
TAPE = r"D:\flowbot_data\hl\trades"
MID = r"D:\flowbot_data\hl\mid"


def day_of(path):
    return os.path.basename(os.path.dirname(path))


def load_days(root, days):
    fs = sorted(glob.glob(os.path.join(root, "*", "*.parquet")))
    by = {}
    for f in fs:
        by.setdefault(day_of(f), []).append(f)
    return by


def main(a):
    tape_days = load_days(TAPE, a.days)
    mid_days = load_days(MID, a.days)
    both = sorted(set(tape_days) & set(mid_days))[-a.days:]
    assert both, "成交帶與中價沒有重疊的日期"
    print("成交帶 %d 天、中價 %d 天，**重疊 %d 天**：%s ~ %s"
          % (len(tape_days), len(mid_days), len(both), both[0], both[-1]))
    print("markout 期間 = %d 分鐘（配分鐘級持有期，不是他文章的 5 秒）" % a.h)
    print()

    rows = []
    for d in both:
        t = pd.concat([pd.read_parquet(f) for f in tape_days[d]],
                      ignore_index=True)
        m = pd.concat([pd.read_parquet(f) for f in mid_days[d]],
                      ignore_index=True)
        m = m[["ts", "coin", "mid"]].dropna()
        if t.empty or m.empty:
            continue
        t = t[t.coin.isin(set(m.coin))].copy()
        if t.empty:
            continue
        t["ts"] = t.ts.astype("int64")
        m["ts"] = m.ts.astype("int64")
        t = t.sort_values("ts")
        m = m.sort_values("ts")

        # 成交當下的中價（as-of backward）與 H 分鐘後的中價（as-of forward）
        cur = pd.merge_asof(t, m.rename(columns={"mid": "mid0"}),
                            on="ts", by="coin", direction="backward",
                            tolerance=120_000)
        cur["ts_fwd"] = cur.ts + a.h * 60_000
        fwd = pd.merge_asof(
            cur.sort_values("ts_fwd"),
            m.rename(columns={"mid": "mid1", "ts": "ts_fwd"}),
            on="ts_fwd", by="coin", direction="forward",
            tolerance=120_000)
        fwd = fwd.dropna(subset=["mid0", "mid1"])
        if fwd.empty:
            continue
        rows.append(fwd[["coin", "side", "px", "sz", "mid0", "mid1"]])

    df = pd.concat(rows, ignore_index=True)
    df["ntl"] = df.px * df.sz
    d_sign = np.where(df.side.astype(str) == "B", 1.0, -1.0)

    # D1 方向平衡
    frac_b = float((d_sign > 0).mean())
    print("D1 吃單方向：買 %.1f%% / 賣 %.1f%%  -> %s"
          % (100 * frac_b, 100 * (1 - frac_b),
             "PASS" if 0.30 <= frac_b <= 0.70 else "**FAIL：方向欄位可能讀錯**"))
    # D2 對齊
    algn = float(np.nanmedian(np.abs(df.px / df.mid0 - 1.0))) * 1e4
    print("D2 成交價 vs 當下中價 中位偏離 %.2f bps -> %s"
          % (algn, "PASS" if algn < 50 else "**FAIL：對齊錯了**"))
    print()

    taker = d_sign * (df.mid1.values - df.px.values) / df.px.values * 1e4
    df["maker_bps"] = -taker

    g = df.groupby("coin")
    out = pd.DataFrame({
        "n": g.size(),
        "ntl_usd": g.ntl.sum(),
        "maker_bps_w": g.apply(
            lambda x: np.average(x.maker_bps, weights=x.ntl)
            if x.ntl.sum() > 0 else np.nan),
        "maker_bps_med": g.maker_bps.median(),
    })
    out = out[out.n >= a.min_trades].sort_values("maker_bps_w", ascending=False)

    print("做市方 markout（**正 = 流量不帶毒 = 吃單的人平均賠錢**），"
          "成交金額加權，單位 bps")
    print("只列成交筆數 >= %d 的標的，共 %d 個\n" % (a.min_trades, len(out)))
    print("%-12s %8s %14s %12s %12s" %
          ("標的", "筆數", "成交額 $", "加權 bps", "中位 bps"))
    print("-" * 62)
    for c, r in out.head(a.top).iterrows():
        print("%-12s %8d %14.0f %12.3f %12.3f"
              % (c, r.n, r.ntl_usd, r.maker_bps_w, r.maker_bps_med))
    if len(out) > a.top * 2:
        print("   …（中間 %d 個略）…" % (len(out) - 2 * a.top))
    for c, r in out.tail(a.top).iterrows():
        print("%-12s %8d %14.0f %12.3f %12.3f"
              % (c, r.n, r.ntl_usd, r.maker_bps_w, r.maker_bps_med))
    print("-" * 62)
    w = np.average(out.maker_bps_w, weights=out.ntl_usd)
    print("全體成交額加權 %.3f bps；為正的標的 %d / %d"
          % (w, int((out.maker_bps_w > 0).sum()), len(out)))
    print()
    print("**這不是判決**：它量的是「平均一筆成交」，不是「我們會拿到的成交」。")
    print("做市方實際被逆選擇，拿到的是對自己不利的那些 —— 所以這只是**篩選**，")
    print("用來決定先報哪些標的，不是損益預測。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h", type=int, default=1, help="markout 分鐘數（預設 1）")
    p.add_argument("--days", type=int, default=14, help="用最近幾天（預設 14）")
    p.add_argument("--min-trades", type=int, default=200)
    p.add_argument("--top", type=int, default=12)
    main(p.parse_args())
