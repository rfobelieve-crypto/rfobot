# -*- coding: utf-8 -*-
"""路徑 N：分鐘級反轉訊號搬到指數期貨的 G1（預註冊，2026-09-05）

**問題**：§1.15 的分鐘級系統在 BTC 上毛捕捉 +6.2 bps、成本 8 bps，差 2 bps 打平。
指數期貨的來回成本大約是 3 bps（NQ 每邊 ~1.5 bps，沿用 `cross_asset_probe.py`
既有的表，不重新估）。**同一個訊號就算只剩一半的毛捕捉，成本量級也完全不同。**

**資料現實（先寫，因為它改變了測法）**：Databento 在本專案零引用——沒有套件、
沒有金鑰、沒有資料，所以 CME 的 MBO（逐單、精確佇列位置）不在手上。Yahoo 的
NQ **分鐘資料只有 7 天**、5 分鐘 60 天，**做不了 15 個月同號檢定**；但**小時
資料有 730 天、13,704 根**。
所以這裡測的是**結構等價版**：分鐘級系統的最佳格是「|60 分鐘報酬| 前 5% 反做、
持有 60 分鐘」，在小時 K 線上就是「|1 根報酬| 前 5% 反做、持有 1 根」。
**這不是原訊號本身，是它的小時版**——若小時版在 NQ 上就沒有，分鐘版更沒有理由
有（微觀結構只會讓成本更高，不會憑空生出方向性）；若小時版有，才值得去買
分鐘資料。**這個推論方向是單向的，寫在這裡免得事後兩邊都當證據用。**

**已知答案的對照（2026-07-29 的教訓）**：NQ 有休市時段，跨時段的那根 bar 是
跳空不是連續行情。診斷寫死：**連續市場（BTC）的跳空率必須趨近 0**，若這支
腳本對 BTC 也報出高跳空率，就是它自己壞了。主判定**排除跨時段 bar**，
含跨時段的版本另報。

**判準（凍結，跑之前寫死）**
  n 閘門   有效訊號 ≥ 300
  G1-a     淨報酬（成本 3 bps 來回）均值 > 0，且**日區塊 bootstrap 95% CI 下界 > 0**
  G1-b     逐月同號 ≥ 20/24 個月（原系統的 15/15 換算成 24 個月的等比例門檻）
  G1-c     隨機對照：把觸發時點隨機重排 500 次，真實均值要在隨機分布 p95 以上
  G1-d     ES 與 GC 兩個次要標的**只報不判**（避免多重比較把 NQ 的結論撐起來）
  三條全過 = G1 PASS，才值得談買分鐘資料；任一不過 = 這條線在小時尺度上沒有。
  **成本三層都報**（0 / 3 / 5 bps 來回），不准只引用有利的那層。

**先驗**：BTC 上這個效應的來源是「散戶追價之後的回歸」。NQ 的參與者結構完全
不同（機構為主、做市商是 Citadel/Jump 等級），**先驗偏 REJECT**。真正的資訊
在於「它是不是連小時尺度都沒有」——那會直接關掉整條指數期貨的路。

Run: python research/exit_paths/nq_fade_g1.py
Out: research/results/nq_fade_g1.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "nq_fade_g1.json"

PRIMARY = ("NQ", "NQ=F", 1.5)                      # (名稱, ticker, 每邊 bps)
SECONDARY = [("ES", "ES=F", 1.0), ("GC", "GC=F", 2.0)]
Q, WIN = 0.95, 500                                  # 前 5%、trailing 500 根
COSTS_RT = {"zero": 0.0, "base": 3.0, "high": 5.0}  # 來回 bps
GAP_MULT = 3.0                                      # 跨時段判定：bar 間隔 > 3 倍中位


def fetch(ticker, interval="1h", period="730d"):
    import yfinance as yf
    d = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def build(d):
    """回傳每根 bar 的：報酬、是否為訊號、方向、下一根的順向報酬、是否跨時段。"""
    c = d["Close"].values
    ret = np.concatenate([[np.nan], np.diff(np.log(c))])
    gaps = np.diff(d.index.view(np.int64)) / 1e9
    med = np.median(gaps)
    is_gap = np.concatenate([[True], gaps > GAP_MULT * med])       # 這根與前一根之間有休市
    thr = pd.Series(np.abs(ret)).rolling(WIN, min_periods=WIN // 2).quantile(Q).values
    sig = np.abs(ret) > thr
    side = -np.sign(ret)                                           # 反做
    fwd = np.concatenate([ret[1:], [np.nan]])                      # 下一根的報酬
    nxt_gap = np.concatenate([is_gap[1:], [True]])                 # 下一根是不是跨時段
    return ret, sig, side, fwd, is_gap, nxt_gap


def dblock(v, days, B=3000, seed=11):
    rng = np.random.default_rng(seed); g = {}
    for x, dd in zip(v, days):
        if np.isfinite(x):
            g.setdefault(dd, []).append(x)
    ks = np.array(list(g))
    if len(ks) < 5:
        return float("nan"), float("nan")
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate(name, d, cost_side, judge=True):
    ret, sig, side, fwd, is_gap, nxt_gap = build(d)
    ok = sig & np.isfinite(fwd) & np.isfinite(side)
    clean = ok & ~is_gap & ~nxt_gap                    # 主判定：訊號那根與下一根都不跨時段
    idx = np.where(clean)[0]
    if len(idx) < 30:
        print(f"  {name}: 有效訊號僅 {len(idx)} 筆，略過"); return None
    r = side[idx] * fwd[idx] * 1e4                     # bps，順著訊號方向
    days = d.index[idx].date
    months = pd.PeriodIndex(d.index[idx], freq="M")
    gap_rate_all = float(is_gap.mean())
    out = {"n": int(len(idx)), "n_with_gap": int(ok.sum()), "gap_rate": gap_rate_all}
    print(f"  [{name}] 有效訊號 {len(idx)}（含跨時段 {ok.sum()}）  跨時段 bar 佔比 {gap_rate_all:.1%}")
    for lbl, c in COSTS_RT.items():
        net = r - c
        lo, hi = dblock(net, days)
        out[lbl] = {"mean_bps": float(net.mean()), "ci": [lo, hi], "win": float((net > 0).mean())}
        print(f"     {lbl:<5}（來回 {c:.0f} bps） 均值 {net.mean():+7.2f} bps  "
              f"CI [{lo:+.2f},{hi:+.2f}]  勝率 {(net > 0).mean():.0%}")
    base = r - COSTS_RT["base"]
    mdf = pd.DataFrame({"m": months, "v": base}).groupby("m")["v"].mean()
    pos = int((mdf > 0).sum())
    out.update(months=int(len(mdf)), months_pos=pos,
               monthly=[float(x) for x in mdf.values])
    print(f"     逐月：{pos}/{len(mdf)} 個月為正")
    if not judge:
        return out
    rng = np.random.default_rng(20260905)
    fwd_ok, side_ok = fwd[np.isfinite(fwd)], side[np.isfinite(fwd)]
    rnd = []
    pool = np.where(np.isfinite(fwd) & np.isfinite(side) & ~is_gap & ~nxt_gap)[0]
    for _ in range(500):
        pick = rng.choice(pool, len(idx), replace=False)
        rnd.append((side[pick] * fwd[pick] * 1e4 - COSTS_RT["base"]).mean())
    rnd = np.array(rnd); pct = float((rnd < base.mean()).mean())
    out["random_pct"] = pct
    print(f"     隨機對照（500 次同樣筆數）：中位 {np.median(rnd):+.2f}  p95 {np.percentile(rnd,95):+.2f}"
          f"  → 真實落在第 {pct*100:.0f} 百分位")
    ga = out["base"]["ci"][0] > 0
    gb = pos >= 20
    gc = pct >= 0.95
    gn = len(idx) >= 300
    out["bars"] = {"n>=300": bool(gn), "G1a_CI>0": bool(ga), "G1b_months>=20": bool(gb), "G1c_rand_p95": bool(gc)}
    out["verdict"] = "G1 PASS" if (gn and ga and gb and gc) else "G1 REJECT"
    print(f"     n≥300 {'過' if gn else '不過'}  CI>0 {'過' if ga else '不過'}  "
          f"月同號≥20 {'過' if gb else '不過'}  勝過隨機 {'過' if gc else '不過'}  ==> {out['verdict']}")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("  路徑 N：分鐘級反轉訊號的小時版搬到指數期貨（預註冊 G1）｜730 天 1h")
    print("=" * 100)
    res = {}
    name, tk, bps = PRIMARY
    d = fetch(tk)
    print(f"  {name} {len(d)} 根  {d.index[0].date()} → {d.index[-1].date()}")
    res[name] = evaluate(name, d, bps, judge=True)
    print("\n  ── 次要標的（只報不判，避免多重比較撐起主結論）──")
    for n2, tk2, b2 in SECONDARY:
        try:
            res[n2] = evaluate(n2, fetch(tk2), b2, judge=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {n2}: 抓不到（{str(e)[:50]}）")
    print("\n  ── 已知答案的對照：連續市場的跨時段率必須趨近 0 ──")
    try:
        btc = fetch("BTC-USD")
        _, _, _, _, is_gap, _ = build(btc)
        rate = float(is_gap.mean())
        res["btc_gap_rate"] = rate
        print(f"  BTC-USD（24/7）跨時段 bar 佔比 {rate:.2%}  "
              f"→ {'診斷正常' if rate < 0.02 else '**診斷壞了，上面的數字不可用**'}")
    except Exception as e:  # noqa: BLE001
        print(f"  BTC 對照抓不到：{str(e)[:60]}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
