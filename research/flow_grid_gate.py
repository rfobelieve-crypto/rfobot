# -*- coding: utf-8 -*-
"""網格 × 流動性波動閘門 —— 動工前的一天生死判定（預註冊 2026-09-05）

**問題**：撤單強度衝擊（`shock`）對「未來 4h 的波動」有沒有 **trailing vol
之外的邊際資訊**？沒有 → 候選 A（網格 × 流波動閘門）在動工前就死；
有 → 那個數字直接餵 `prereg_power.py` 定 n。

**為什麼是這個問題**：訂單流在本專案唯一四個 horizon 全過、兩半一致的發現
是「撤單衝擊 → |forward return|」（TEST B，2026-08-13）。它被否決是因為
V7 沒有旋鈕接波動預測。網格有——網格的損益天生是波動的函數，而它唯一的
死法（趨勢／跌出區間）正是波動擴張。§0.93 用 ADX 當閘門 NO-GO，但 ADX
已被證明無前瞻力（ER/VR 雙證偽）；閘門的概念沒被否定，儀器不行而已。

**最大風險**（也是本檔存在的理由）：trailing realized vol 本身就是強基線
（縮帆線儀表排名 trailing vol +0.46 ≫ ADX +0.08）。撤單衝擊若只是 trailing
vol 的翻版，raw IC 再漂亮都沒用。所以主檢定是 **conditional IC**。

**在跑之前寫死的東西**

  特徵   shock(t) = (bid_cancel + ask_cancel)(t) / trailing-60m median
         —— 逐字沿用 market_data/tasks/cancel_playbook_watcher.compute_features
         的凍結定義，trailing-only。**不另造特徵。**
  基線   trail_vol(t) = 過去 240 分鐘 1m log-return 的 std（trailing-only）
  主標籤 fut_vol(t)   = 未來 240 分鐘 1m log-return 的 std，
         窗口從 t+2 起算（留一整分鐘的空隙，比嚴格早一格更保守——
         mistake.md 2026-09-03：同一根 bar 的不同欄位屬於不同時刻）
  副標籤 exit2x(t)    = 未來 4h 的 mid 區間 > 2 × 過去 4h 的 mid 區間
         （「波動加倍」＝網格死法的形狀）
  條件化 (i) Spearman partial correlation：rank(fut_vol) 對 rank(trail_vol)
             回歸取殘差，再與 rank(shock) 取相關
         (ii) 依 trail_vol 五分位分層，各層內算 IC(shock, fut_vol)，全格報告
  判準   撤單流自己凍結的那條，逐字：
         **日區塊 bootstrap 95% CI 離零 ∧ |IC| ≥ 0.02 ∧ 前後兩半同號**
         套在 **conditional IC** 上。raw IC 只當儀器檢查：它必須重現 TEST B
         的正號與量級（~+0.1），否則是我的管線壞了，不是市場變了。
  樣本   主：BTC-USD binance_perp（網格與 V7 的標的，永續是被迫流所在）
         穩健：全部 10 幣，報同號幣數（不得事後挑幣）
  預測   raw IC 為正、~+0.1（已知答案）。conditional IC：**我不知道**，
         先驗五五開。寫在這裡是為了事後不能改口。

**不做的事**：不調 shock 的窗口、不換標籤定義、不因為某一幣好看就只報它。
差 0.0 就是差 0.0。

Run:  python research/flow_grid_gate.py [--all]
Out:  research/results/flow_grid_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "flow_grid_gate.json"
H = 240            # 4h in minutes
GAP = 2            # future window starts at t+GAP (conservative)
IC_FLOOR = 0.02    # frozen with the cancel playbook (CLAUDE.md)
SEED = 20260905


def load(symbol: str, exchange: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms m, bid_cancel_qty bc, ask_cancel_qty ac "
                "FROM depth_deltas_1m WHERE canonical_symbol=%s AND exchange=%s "
                "ORDER BY minute_start_ms", (symbol, exchange))
            d = pd.DataFrame(cur.fetchall())
            cur.execute(
                "SELECT ts_ms, mid_price FROM orderbook_snapshots_1m "
                # price is always the SPOT mid: orderbook_snapshots_1m only
                # records exchange='binance'. Spot and perp 4h vol are the
                # same quantity to within bps; the label is unaffected.
                "WHERE canonical_symbol=%s AND exchange='binance' ORDER BY ts_ms",
                (symbol,))
            p = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    if d.empty or p.empty:
        return pd.DataFrame()
    d["m"] = (d["m"] // 60000) * 60000
    p["m"] = (p["ts_ms"] // 60000) * 60000
    # last snapshot inside each minute = the price known at minute END
    p = p.sort_values("ts_ms").groupby("m", as_index=False).last()[["m", "mid_price"]]
    x = d.merge(p, on="m", how="inner").sort_values("m").reset_index(drop=True)
    for c in ("bc", "ac", "mid_price"):
        x[c] = x[c].astype(float)
    return x


def features(x: pd.DataFrame) -> pd.DataFrame:
    # --- frozen shock, verbatim from cancel_playbook_watcher ---
    tot = x["bc"] + x["ac"]
    base = tot.rolling(60, min_periods=30).median()
    x["shock"] = tot / base.replace(0, np.nan)
    # --- baseline & labels ---
    r = np.log(x["mid_price"]).diff()
    x["trail_vol"] = r.rolling(H, min_periods=H // 2).std()
    # future: std of returns over t+GAP .. t+GAP+H-1  (strictly after t)
    fut = r.shift(-(GAP)).rolling(H, min_periods=H // 2).std().shift(-(H - 1))
    x["fut_vol"] = fut
    mid = x["mid_price"]
    hi_p = mid.rolling(H).max().shift(-(H - 1) - GAP)
    lo_p = mid.rolling(H).min().shift(-(H - 1) - GAP)
    hi_t = mid.rolling(H).max()
    lo_t = mid.rolling(H).min()
    x["fut_range"] = (hi_p - lo_p) / mid
    x["trail_range"] = (hi_t - lo_t) / mid
    x["exit2x"] = (x["fut_range"] > 2.0 * x["trail_range"]).astype(float)
    x["day"] = (x["m"] // 86_400_000).astype(int)
    return x.dropna(subset=["shock", "trail_vol", "fut_vol"])


def partial_ic(a, b, ctrl) -> float:
    """Spearman partial corr of a with b, controlling for ctrl."""
    ra, rb, rc = (pd.Series(v).rank().values for v in (a, b, ctrl))
    # residualise rb on rc
    A = np.vstack([rc, np.ones_like(rc)]).T
    beta = np.linalg.lstsq(A, rb, rcond=None)[0]
    res = rb - A @ beta
    return float(spearmanr(ra, res).correlation)


def day_block_ci(x: pd.DataFrame, fn, B=200, seed=SEED):
    rng = np.random.default_rng(seed)
    days = x["day"].unique()
    groups = {d: g for d, g in x.groupby("day")}
    out = []
    for _ in range(B):
        pick = rng.choice(days, len(days), replace=True)
        s = pd.concat([groups[d] for d in pick])
        try:
            out.append(fn(s))
        except Exception:  # noqa: BLE001
            continue
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)), float(np.nanstd(out))


def evaluate(x: pd.DataFrame, label: str) -> dict:
    raw = lambda s: float(spearmanr(s["shock"], s[label]).correlation)
    cond = lambda s: partial_ic(s["shock"].values, s[label].values, s["trail_vol"].values)
    n = len(x)
    half = n // 2
    r_raw, r_cond = raw(x), cond(x)
    lo, hi, se = day_block_ci(x, cond)
    rlo, rhi, _ = day_block_ci(x, raw, B=100)
    h1, h2 = cond(x.iloc[:half]), cond(x.iloc[half:])
    # within trailing-vol quintile (no regression at all)
    q = pd.qcut(x["trail_vol"], 5, labels=False, duplicates="drop")
    strat = []
    for k in sorted(q.dropna().unique()):
        s = x[q == k]
        strat.append({"q": int(k), "n": int(len(s)),
                      "ic": float(spearmanr(s["shock"], s[label]).correlation)})
    passed = (lo > 0 or hi < 0) and abs(r_cond) >= IC_FLOOR and np.sign(h1) == np.sign(h2)
    return {"label": label, "n": n, "days": int(x["day"].nunique()),
            "raw_ic": r_raw, "raw_ci": [rlo, rhi],
            "cond_ic": r_cond, "cond_ci": [lo, hi], "cond_se": se,
            "halves": [h1, h2], "strata": strat,
            "verdict": "PASS" if passed else "FAIL",
            "power_note": (f"SE {se:.4f} vs floor {IC_FLOOR}: "
                           f"{'floor is >2.8 SE (decidable)' if IC_FLOOR >= 2.8 * se else 'floor < 2.8 SE — underpowered'}")}


def run_symbol(sym: str, ex: str) -> dict | None:
    x = load(sym, ex)
    if x.empty:
        return None
    x = features(x)
    if len(x) < 5000:
        return None
    return {"symbol": sym, "exchange": ex,
            "fut_vol": evaluate(x, "fut_vol"),
            "exit2x": evaluate(x, "exit2x")}


def fmt(r: dict) -> str:
    e = r["fut_vol"]; g = r["exit2x"]
    s = (f"{r['symbol']:9} {r['exchange']:13} n={e['n']:6d} ({e['days']}d)\n"
         f"   主標籤 fut_vol : raw IC {e['raw_ic']:+.4f} [{e['raw_ci'][0]:+.3f},{e['raw_ci'][1]:+.3f}]"
         f"  | cond IC {e['cond_ic']:+.4f} [{e['cond_ci'][0]:+.4f},{e['cond_ci'][1]:+.4f}]"
         f"  halves {e['halves'][0]:+.3f}/{e['halves'][1]:+.3f}  → {e['verdict']}\n"
         f"                    {e['power_note']}\n"
         f"                    分層 IC(依 trail_vol 五分位): "
         + " ".join(f"Q{s['q']+1}:{s['ic']:+.3f}" for s in e["strata"]) + "\n"
         f"   副標籤 exit2x  : raw IC {g['raw_ic']:+.4f} | cond IC {g['cond_ic']:+.4f}"
         f" [{g['cond_ci'][0]:+.4f},{g['cond_ci'][1]:+.4f}] halves {g['halves'][0]:+.3f}/{g['halves'][1]:+.3f} → {g['verdict']}")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    syms = ["BTC-USD"] if not a.all else ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                                          "DOGE-USD", "LINK-USD", "SUI-USD", "UNI-USD", "AAVE-USD"]
    exs = ["binance_perp"]   # perp only: that is where forced flow lives; spot is redundant
    print("=" * 96)
    print("  網格 × 流動性波動閘門：shock 對未來 4h 波動的 conditional IC（對照 trailing vol）")
    print("  判準（凍結）：日區塊 bootstrap CI 離零 ∧ |cond IC| ≥ 0.02 ∧ 兩半同號")
    print("=" * 96)
    res = []
    for ex in exs:
        for s in syms:
            r = run_symbol(s, ex)
            if r:
                res.append(r)
                print(fmt(r), flush=True)
    if a.all:
        for lab in ("fut_vol", "exit2x"):
            pos = sum(1 for r in res if r[lab]["cond_ic"] > 0)
            pas = sum(1 for r in res if r[lab]["verdict"] == "PASS")
            print(f"\n  {lab}: cond IC 正號 {pos}/{len(res)}，過判準 {pas}/{len(res)}")
    OUT.write_text(json.dumps({"prereg": "see module docstring", "results": res},
                              ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
