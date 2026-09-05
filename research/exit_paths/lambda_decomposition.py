# -*- coding: utf-8 -*-
"""λ 的條件分解——逆選擇不是常數，是混合分布的均值（預註冊，2026-09-05）

§1.17 量到無條件掛單的 markout_60 = **−3.1 bps**，那是把所有成交平均之後的
數字。但打到你的單的不是一種人：需要流動性的、對沖的、被清算的、以及真正
知道下一秒的。前三種可能付錢給你，最後一種收你的錢。**均值 −3.1 只說明
「不分對象地掛，平均下來輸」，沒說每一種各佔多少、各值多少。**

所以這支不找新訊號，**把已有的成交按掛單當下可觀測的狀態切開，找 λ ≤ 0 的
格子**。狀態是掛單當下就看得到的，不需要預測任何東西——只需要決定在哪些狀態
下掛、哪些不掛。

## 先修儀器（使用者指出，這決定所有格子的可信度）

§1.17 的 markout 是用**主規則**算的：`bid_l1(t') ≤ p`（最佳買價跌到我的價位）
＝**觸及即成交**，一分鐘快照、無佇列模型。它**高估**成交、也**低估** λ，
因為多算進來的是「只是碰到」那些較不毒的成交。

嚴格規則 `ask_l1(t') ≤ p`（對手方**穿過**我的價位）必然成交，但它挑出來的
正是價格繼續走的那些——**對 λ 是悲觀的**。

**判準因此要求兩種規則都成立**：一格只有在主規則與嚴格規則下 markout 的
95% CI 上界**都** < 0，才算「這裡的對手在付錢給你」。

## 狀態（全部在掛單當下可觀測，不需預測）

  T  時段：亞洲 / 歐洲 / 美洲，以及美股開盤前後一小時（UTC 13:30 ±1h）單獨切
  C  撤單流狀態：`depth_deltas_1m` 的 shock（凍結原式）三分位
  D  深度狀態：`bid_depth_usd_l20` 相對該幣 30 日中位的三分位
  V  波動 regime：trailing 60 分鐘實現波動三分位（**當下實現值，不是預測**）
  L  清算距離：**本輪不做**——清算叢集需要 `liq_events`，而那個錄製器
     2026-09-05 才開始跑。**用 Scope 具名記錄這個缺口，不靜默略過**
     （靜默縮小範圍已經害過兩次，見 shared/declared_scope.py）

## 判準（跑之前寫死）

  L1  某格的 markout_60 在**兩種成交規則下** CI 上界都 < 0 ∧ 該格成交數 ≥ 300
  L2  全格報告，不挑格；跨 4 個維度共 ~13 格，**Benjamini-Hochberg 修正**
  L3  逐幣一致：該格在 ≥ 6/11 個幣上同號
  三條全過 = 找到可站的格子；否則按下面三種結果分類：
     (a) 有格子過 → 網格改成只在那些狀態掛
     (b) 全部 λ > 0 但分散度大 → 先關掉最慘的格，看 g−c−λ 會不會翻正
     (c) 全部 λ 都在 +2 bps 以上且無分散 → **BTC 上的無條件流動性提供結案**

## 第二部分：跨標的的無條件 λ vs 自然價差

λ = 3.1 bps 在 BTC 上致命，是因為 BTC 的自然價差只有 ~0.015 bps，`g−c` 的
空間本來就薄。在自然價差 1–5 bps 的幣上，同樣的 λ 佔比完全不同。
所以同時報：每個幣的無條件 λ、自然價差、以及 **λ / 半價差** 這個比值
——**比值 < 1 的幣，做市在結構上才有空間。**

Run: python research/exit_paths/lambda_decomposition.py
Out: research/results/lambda_decomposition.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402
from shared.declared_scope import Scope  # noqa: E402

OUT = ROOT / "research" / "results" / "lambda_decomposition.json"
SYMS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
        "LINK-USD", "AVAX-USD", "SUI-USD", "UNI-USD", "AAVE-USD"]
EX = "binance"
T_HOLD = 60            # 掛單有效期（分鐘）
H = 60                 # markout 視野（分鐘）
STEP = 3               # 每 3 分鐘雙邊各一張（與 §1.17 的 U 組一致）
MIN_N = 300


def load(sym):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts_ms, mid_price, bid_l1_price, ask_l1_price, bid_depth_usd_l20 "
                "FROM orderbook_snapshots_1m WHERE canonical_symbol=%s AND exchange=%s "
                "ORDER BY ts_ms", (sym, EX))
            d = pd.DataFrame(cur.fetchall())
            cur.execute(
                "SELECT minute_start_ms m, bid_cancel_qty bc, ask_cancel_qty ac "
                "FROM depth_deltas_1m WHERE canonical_symbol=%s AND exchange=%s "
                "ORDER BY m", (sym, EX))
            c = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    if d.empty:
        return None
    d["m"] = (d["ts_ms"] // 60000) * 60000
    d = d.sort_values("ts_ms").groupby("m", as_index=False).last()
    for col in ("mid_price", "bid_l1_price", "ask_l1_price", "bid_depth_usd_l20"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    idx = pd.RangeIndex(int(d["m"].min()), int(d["m"].max()) + 60000, 60000)
    d = d.set_index("m").reindex(idx)
    if not c.empty:
        c["shock"] = ((c["bc"].astype(float) + c["ac"].astype(float))
                      / (c["bc"].astype(float) + c["ac"].astype(float))
                      .rolling(60, min_periods=30).median().replace(0, np.nan))
        d["shock"] = c.set_index("m")["shock"].reindex(idx)
    else:
        d["shock"] = np.nan
    return d


def events(d):
    """回傳每個掛單事件的：成交(主/嚴格)、markout、以及掛單當下的四種狀態。"""
    mid = d["mid_price"].values
    bid = d["bid_l1_price"].values
    ask = d["ask_l1_price"].values
    dep = d["bid_depth_usd_l20"].values
    shk = d["shock"].values
    n = len(d)
    r = np.diff(np.log(np.where(np.isfinite(mid), mid, np.nan)), prepend=np.nan)
    vol60 = pd.Series(r).rolling(60, min_periods=30).std().values
    dep_med = np.nanmedian(dep)
    hour = ((pd.to_datetime(d.index, unit="ms", utc=True).hour).values)
    rows = []
    for s in (1, -1):
        p = bid if s > 0 else ask
        kf = np.full(n, -1, np.int64); ks = np.full(n, -1, np.int64)
        for k in range(1, T_HOLD + 1):
            if s > 0:
                c1 = bid[k:] <= p[:-k]; c2 = ask[k:] <= p[:-k]
            else:
                c1 = ask[k:] >= p[:-k]; c2 = bid[k:] >= p[:-k]
            c1 = np.nan_to_num(c1, nan=False).astype(bool)
            c2 = np.nan_to_num(c2, nan=False).astype(bool)
            f = np.zeros(n, bool); f[:-k] = c1 & (kf[:-k] < 0); kf[f] = k
            g = np.zeros(n, bool); g[:-k] = c2 & (ks[:-k] < 0); ks[g] = k
        ev = np.arange(n)[(np.arange(n) % STEP == 0) & np.isfinite(p)
                          & (np.arange(n) < n - T_HOLD - H - 1)]
        for e in ev:
            row = {"side": s, "day": int(d.index[e] // 86_400_000),
                   "hour": int(hour[e]),
                   "dep": dep[e] / dep_med if dep_med and np.isfinite(dep[e]) else np.nan,
                   "shock": shk[e], "vol": vol60[e]}
            for tag, kk in (("main", kf[e]), ("strict", ks[e])):
                if kk > 0 and e + kk + H < n:
                    row[f"mo_{tag}"] = s * (mid[e + kk + H] - p[e]) / p[e] * 1e4
                else:
                    row[f"mo_{tag}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def dblock(v, days, B=1500, seed=29):
    rng = np.random.default_rng(seed); g = {}
    for x, dd in zip(v, days):
        if np.isfinite(x):
            g.setdefault(int(dd), []).append(x)
    ks = np.array(list(g))
    if len(ks) < 5:
        return float("nan"), float("nan")
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cells(df):
    """四個維度的分格（掛單當下可觀測）。"""
    out = {}
    h = df["hour"]
    out["時段·亞洲(0-8h)"] = (h < 8)
    out["時段·歐洲(8-13h)"] = (h >= 8) & (h < 13)
    out["時段·美股開盤±1h"] = (h >= 13) & (h < 15)
    out["時段·美洲其餘"] = (h >= 15)
    for name, col in (("撤單流", "shock"), ("深度", "dep"), ("波動", "vol")):
        v = df[col]
        if v.notna().sum() < 1000:
            continue
        q1, q2 = v.quantile(1 / 3), v.quantile(2 / 3)
        out[f"{name}·低"] = v <= q1
        out[f"{name}·中"] = (v > q1) & (v <= q2)
        out[f"{name}·高"] = v > q2
    return out


def bh(pvals, alpha=0.05):
    idx = np.argsort(pvals)
    m = len(pvals)
    keep = np.zeros(m, bool)
    for r, i in enumerate(idx, 1):
        if pvals[i] <= alpha * r / m:
            keep[idx[:r]] = True
    return keep


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    scope = Scope("λ 條件分解", expect_n=len(SYMS))
    frames, per_coin, ok = {}, {}, []
    print("=" * 104)
    print("  λ 的條件分解：逆選擇不是常數，是混合分布的均值（預註冊）")
    print("=" * 104)
    print(f"  {'幣':<10}{'分鐘':>8}{'事件':>8}{'成交率(主/嚴)':>16}"
          f"{'半價差bps':>11}{'λ主':>9}{'λ嚴':>9}{'λ/半價差':>10}")
    for s in SYMS:
        d = load(s)
        if d is None or len(d) < 5000:
            continue
        df = events(d)
        hs = ((d["ask_l1_price"] - d["bid_l1_price"]) / d["mid_price"] * 1e4 / 2).median()
        fm = df["mo_main"].notna().mean(); fs = df["mo_strict"].notna().mean()
        lam_m = -df["mo_main"].mean(); lam_s = -df["mo_strict"].mean()
        per_coin[s] = {"minutes": int(len(d)), "events": int(len(df)),
                       "fill_main": float(fm), "fill_strict": float(fs),
                       "half_spread_bps": float(hs),
                       "lambda_main": float(lam_m), "lambda_strict": float(lam_s),
                       "ratio": float(lam_m / hs) if hs else float("nan")}
        frames[s] = df; ok.append(s)
        print(f"  {s:<10}{len(d):>8}{len(df):>8}{fm:>8.0%}/{fs:>6.0%}"
              f"{hs:>11.3f}{lam_m:>+9.2f}{lam_s:>+9.2f}{lam_m/hs if hs else float('nan'):>10.1f}",
              flush=True)
    scope.check(actual_n=len(ok),
                allow_shrink="" if len(ok) == len(SYMS) else
                "部分標的的 orderbook_snapshots_1m 不足 5000 分鐘（事前規則排除）")
    if not frames:
        print("  無資料"); return 0
    D = pd.concat(frames.values(), keys=frames.keys(), names=["sym"]).reset_index(level=0)

    print(f"\n  ── 全格報告（不挑格；判準：兩種成交規則的 CI 上界都 < 0）──")
    print(f"  {'狀態格':<20}{'n(主)':>8}{'λ主':>9}{'CI主':>20}{'λ嚴':>9}{'CI嚴':>20}  逐幣")
    cs = cells(D)
    res = {"per_coin": per_coin, "scope": scope.as_dict(), "cells": {}}
    pv, names = [], []
    for name, mask in cs.items():
        sub = D[mask]
        mm = sub["mo_main"].dropna(); ss = sub["mo_strict"].dropna()
        if len(mm) < MIN_N:
            print(f"  {name:<20}{len(mm):>8}  樣本不足"); continue
        lo_m, hi_m = dblock(mm.values, sub.loc[mm.index, "day"].values)
        lo_s, hi_s = dblock(ss.values, sub.loc[ss.index, "day"].values)
        same = 0
        for sym in ok:
            v = D[(D["sym"] == sym) & mask]["mo_main"].dropna()
            if len(v) >= 50 and np.sign(v.mean()) == np.sign(mm.mean()):
                same += 1
        pay = (hi_m < 0) and (hi_s < 0)      # markout>0 才是對手在付錢；λ=−markout
        res["cells"][name] = {"n": int(len(mm)), "mo_main": float(mm.mean()),
                              "ci_main": [lo_m, hi_m], "mo_strict": float(ss.mean()),
                              "ci_strict": [lo_s, hi_s], "same_sign": same,
                              "pays_you": bool(mm.mean() > 0 and lo_m > 0 and lo_s > 0)}
        r = res["cells"][name]
        print(f"  {name:<20}{len(mm):>8}{-mm.mean():>+9.2f}  [{-hi_m:+7.2f},{-lo_m:+7.2f}]"
              f"{-ss.mean():>+9.2f}  [{-hi_s:+7.2f},{-lo_s:+7.2f}]{same:>5}/{len(ok)}")
    good = [k for k, v in res["cells"].items() if v["pays_you"] and v["same_sign"] >= 6]
    lam_all = [v["lambda_main"] for v in per_coin.values()]
    spread_ok = [k for k, v in per_coin.items() if v["ratio"] < 1]
    res["cells_paying_you"] = good
    res["coins_ratio_lt1"] = spread_ok
    if good:
        verdict = f"找到可站的格子：{good}"
    elif min(lam_all) > 2.0:
        verdict = "全部 λ > +2 bps 且無分散 → BTC 類的無條件流動性提供結案"
    else:
        verdict = "沒有 λ ≤ 0 的格子，但分散度存在 → 先關最慘的格，看 g−c−λ 能否翻正"
    res["verdict"] = verdict
    print(f"\n  λ/半價差 < 1 的幣（做市在結構上有空間）：{spread_ok or '無'}")
    print(f"  ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
