# -*- coding: utf-8 -*-
"""流訊號的鏡像消費者：做多波動（突破）—— 預註冊 2026-09-05

**為什麼**：`flow_grid_gate.py` 證明撤單衝擊 `shock` 對未來 4h 波動水平有
trailing vol 之外的資訊（10/10 幣、安慰劑歸零）。`lp_ladder/grid_flow_gate.py`
把它放進網格（**做空波動**）當閘門：贏 trailing-vol 閘門四情境全勝，但震盪時
等於少放 10% 資金 → NO-GO。訊號說的是「波動會比 trailing 高」——那是**做多
波動**的訊號，放在做空波動的策略裡最多只能「少做空一點」。鏡像才是直接變現。

**策略（凍結）**：在小時邊界 i，若訊號在 i−1 亮 → 以 i 開頭的 mid 為中心，
掛雙向停損進場：多在 mid×(1+k·σ)，空在 mid×(1−k·σ)，σ = trailing 4h 的
1m 報酬 std × √240（把 trailing 波動放大到 4h 尺度），**k = 1.0**。
未來 240 分鐘內先碰到哪邊就進哪邊（用 1m mid 判斷，成交價＝觸發價）；
進場後：mid 回到原中心 → 停損出場（突破失敗）；否則 240 分鐘窗口結束平倉。
兩邊都沒碰到 → 沒交易。**沒有任何參數是調出來的**：k=1、窗 4h、80 百分位
都是本線既有的數字。

**成本**：停損進場是 taker，在快速盤：每邊 taker 5 + 滑價 5 = **10 bps**，
來回 20 bps（比 V7 的 9 bps 嚴）。另報零成本。

**四臂（同一套進場出場，只差「哪些小時開火」）**
  FLOW      shock_h(i−1) > 其 trailing 168h 第 80 百分位（同 grid_flow_gate 的構造）
  TRAILVOL  對照一：trailing 4h vol 本身 > 其 trailing 168h 第 80 百分位
            —— 「是流，還是只是波動高」
  RANDOM    對照二：隨機 20% 的小時（固定種子）—— 「是訊號，還是突破本身就賺」
  ALL       每個小時都開火 —— 無條件突破的基準

**判準（跑之前凍結，逐字沿用本線既有的形狀）**：GO 必須同時
  (1) FLOW 臂每筆淨 bps > 0，日區塊 bootstrap 95% CI 不含零（10 幣合併）
  (2) FLOW − TRAILVOL 的差 > 0 且 bootstrap CI 不含零（流要贏過波動本身）
  (3) FLOW − RANDOM 的差 > 0 且 CI 不含零（訊號要贏過隨機開火）
  (4) 10 幣裡 FLOW 淨 bps 為正的 ≥ 7
  (5) 前後兩半同號
  任一不過 = NO-GO。不調 k、不換窗、不換百分位、不挑幣。

**預測（寫死）**：ALL 臂淨值為負（無條件突破在 20 bps 成本下是輸的，這是
突破策略的常識）；FLOW 臂 > TRAILVOL 臂（IC 已證）；FLOW 臂是否 > 0——
**不知道**，這才是問題。先驗：(2)(3) 六成過，(1) 五五開。

Run: python research/flow_breakout_gate.py
Out: research/results/flow_breakout_gate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from flow_grid_gate import load  # noqa: E402  (same data, same frozen shock)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "flow_breakout_gate.json"
K = 1.0
W = 240
COST_SIDE = 10.0     # bps: taker 5 + slippage 5, stop-market in a fast tape
LB = 168
Q = 0.80
COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
         "DOGE-USD", "LINK-USD", "SUI-USD", "UNI-USD", "AAVE-USD"]


def prep(sym: str) -> pd.DataFrame:
    x = load(sym, "binance_perp")
    tot = x["bc"] + x["ac"]
    base = tot.rolling(60, min_periods=30).median()
    x["shock"] = tot / base.replace(0, np.nan)
    r = np.log(x["mid_price"]).diff()
    x["tv"] = r.rolling(W, min_periods=W // 2).std()
    x["hour"] = x["m"] // 3_600_000
    x["day"] = x["m"] // 86_400_000
    return x.reset_index(drop=True)


def hourly_signals(x: pd.DataFrame):
    """Per hour: mean shock and last trailing vol; both known at hour END."""
    h = x.groupby("hour").agg(shock=("shock", "mean"), tv=("tv", "last"),
                              start=("m", "min")).reset_index()
    for col in ("shock", "tv"):
        thr = h[col].rolling(LB, min_periods=LB // 2).quantile(Q)
        h[col + "_hot"] = (h[col] > thr) & thr.notna()
    # decide hour i from hour i-1
    h["fire_FLOW"] = h["shock_hot"].shift(1).fillna(False).astype(bool)
    h["fire_TRAILVOL"] = h["tv_hot"].shift(1).fillna(False).astype(bool)
    rng = np.random.default_rng(20260905)
    h["fire_RANDOM"] = rng.random(len(h)) < 0.20
    h["fire_ALL"] = True
    h["sigma4h"] = h["tv"].shift(1) * np.sqrt(W)          # from hour i-1
    return h


def trade(mid: np.ndarray, start_idx: int, sigma: float, cost_side=COST_SIDE):
    """Breakout over the next W minutes from the mid at start_idx.
    Returns net bps or None (no trigger)."""
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    seg = mid[start_idx + 1: start_idx + 1 + W]
    if len(seg) < W // 2:
        return None
    c = mid[start_idx]
    up, dn = c * (1 + K * sigma), c * (1 - K * sigma)
    hit_up = np.argmax(seg >= up) if (seg >= up).any() else None
    hit_dn = np.argmax(seg <= dn) if (seg <= dn).any() else None
    if hit_up is None and hit_dn is None:
        return None
    if hit_dn is None or (hit_up is not None and hit_up <= hit_dn):
        side, e_i, entry = +1, hit_up, up
    else:
        side, e_i, entry = -1, hit_dn, dn
    after = seg[e_i + 1:]
    # stop: mid returns to the original centre (the breakout failed)
    back = (after <= c) if side > 0 else (after >= c)
    exit_px = c if back.any() else (after[-1] if len(after) else entry)
    gross = side * (exit_px / entry - 1) * 1e4
    return gross - 2 * cost_side, gross


def run_coin(sym: str) -> dict:
    x = prep(sym)
    h = hourly_signals(x)
    mid = x["mid_price"].values
    first_idx = x.groupby("hour").apply(lambda g: g.index[0])
    h = h.merge(first_idx.rename("i0").reset_index(), on="hour", how="left")
    res = {}
    for arm in ("FLOW", "TRAILVOL", "RANDOM", "ALL"):
        rows = []
        for _, r in h[h["fire_" + arm]].iterrows():
            t = trade(mid, int(r["i0"]), float(r["sigma4h"]))
            if t is not None:
                rows.append({"day": int(r["start"] // 86_400_000), "net": t[0], "gross": t[1]})
        res[arm] = pd.DataFrame(rows)
    return res


def dblock(df: pd.DataFrame, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    days = df["day"].unique(); g = {d: v["net"].values for d, v in df.groupby("day")}
    out = np.array([np.concatenate([g[d] for d in rng.choice(days, len(days))]).mean()
                    for _ in range(B)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, B=2000, seed=8):
    rng = np.random.default_rng(seed)
    da = {d: v["net"].values for d, v in a.groupby("day")}
    db = {d: v["net"].values for d, v in b.groupby("day")}
    days = sorted(set(da) | set(db))
    out = []
    for _ in range(B):
        pick = rng.choice(days, len(days))
        ma = np.concatenate([da[d] for d in pick if d in da]).mean()
        mb = np.concatenate([db[d] for d in pick if d in db]).mean()
        out.append(ma - mb)
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    print("=" * 100)
    print(f"  流訊號 → 做多波動（突破）  k={K}σ  窗 {W}m  成本 {COST_SIDE:.0f} bps/邊")
    print("  判準：(1) FLOW 淨>0 CI離零 (2) FLOW>TRAILVOL (3) FLOW>RANDOM (4) ≥7/10 幣正 (5) 兩半同號")
    print("=" * 100)
    pooled = {a: [] for a in ("FLOW", "TRAILVOL", "RANDOM", "ALL")}
    per = {}
    print(f"  {'幣':9}" + "".join(f"{a:>18}" for a in pooled) + "   (n / 淨bps/筆)")
    for sym in COINS:
        r = run_coin(sym)
        per[sym] = {a: {"n": int(len(d)), "net": float(d["net"].mean()) if len(d) else None,
                        "gross": float(d["gross"].mean()) if len(d) else None} for a, d in r.items()}
        for a, d in r.items():
            d = d.copy(); d["sym"] = sym; pooled[a].append(d)
        print(f"  {sym:9}" + "".join(f"{per[sym][a]['n']:>7d}/{(per[sym][a]['net'] or 0):>+9.1f}" for a in pooled))
    P = {a: pd.concat(v, ignore_index=True) for a, v in pooled.items()}
    print()
    summ = {}
    for a, d in P.items():
        lo, hi = dblock(d)
        half = len(d) // 2
        d2 = d.sort_values("day")
        summ[a] = {"n": int(len(d)), "net": float(d["net"].mean()), "gross": float(d["gross"].mean()),
                   "ci": [lo, hi], "wr": float((d["net"] > 0).mean()),
                   "halves": [float(d2["net"].iloc[:half].mean()), float(d2["net"].iloc[half:].mean())],
                   "coins_pos": int(sum(1 for s in COINS if (per[s][a]["net"] or 0) > 0))}
        s = summ[a]
        print(f"  {a:9} n={s['n']:5d}  淨 {s['net']:+7.1f} bps [{lo:+.1f}, {hi:+.1f}]  毛 {s['gross']:+6.1f}"
              f"  WR {s['wr']*100:4.1f}%  兩半 {s['halves'][0]:+.1f}/{s['halves'][1]:+.1f}  幣正 {s['coins_pos']}/10")
    d_tv = diff_ci(P["FLOW"], P["TRAILVOL"]); d_rd = diff_ci(P["FLOW"], P["RANDOM"])
    print(f"\n  FLOW − TRAILVOL: {d_tv[0]:+.1f} bps [{d_tv[1]:+.1f}, {d_tv[2]:+.1f}]")
    print(f"  FLOW − RANDOM  : {d_rd[0]:+.1f} bps [{d_rd[1]:+.1f}, {d_rd[2]:+.1f}]")
    F = summ["FLOW"]
    c1 = F["net"] > 0 and F["ci"][0] > 0
    c2 = d_tv[0] > 0 and d_tv[1] > 0
    c3 = d_rd[0] > 0 and d_rd[1] > 0
    c4 = F["coins_pos"] >= 7
    c5 = np.sign(F["halves"][0]) == np.sign(F["halves"][1])
    verdict = "GO" if all((c1, c2, c3, c4, c5)) else "NO-GO"
    print(f"\n  (1) {'過' if c1 else '不過'}  (2) {'過' if c2 else '不過'}  (3) {'過' if c3 else '不過'}"
          f"  (4) {'過' if c4 else '不過'}  (5) {'過' if c5 else '不過'}   ==> {verdict}")
    print(f"  零成本參考：FLOW 毛 {F['gross']:+.1f} bps/筆（成本 {2*COST_SIDE:.0f} bps 來回）")
    OUT.write_text(json.dumps({"summary": summ, "per_coin": per, "diff_tv": d_tv, "diff_rd": d_rd,
                               "verdict": verdict, "c": [bool(c) for c in (c1, c2, c3, c4, c5)]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
