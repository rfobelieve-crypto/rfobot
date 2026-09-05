# -*- coding: utf-8 -*-
"""費波那契回撤位 × V7 Strong 進場——一小時的預註冊測試（2026-09-05）

使用者：「如果均值回歸加入費波那契／黃金比例呢？」
近親已死六個（地形 D6/S3/S1/S2/D4/D10、整數關卡、量能剖面），機制結論是
「系統吃的是還掛著的單，不是價格記憶」。本檔給它一次自己的機會，一個桶、
一個對照、判準寫死。

**定義（凍結）**
  波段   進場 bar t 之前 72 根 1h bar 的最高 H、最低 L（t−72..t−1，因果）
  水位   L + r·(H−L)，r ∈ {0.382, 0.5, 0.618}（三個價位對上下波段都一樣，無方向歧義）
  距離   |entry_price − 最近水位| / ATR14(t−1)
  桶     NEAR：距離 ≤ 0.5 ATR；FAR：其餘。**一個門檻，不掃。**
  樣本   tracked_signals Strong，2026-04-03 起，有 actual_return_4h 者（n≈127）
  命中   sign(方向) × actual_return_4h > 0
  對照   **隨機水位**：在同一波段 [L, H] 內均勻抽 3 個 r，重複 500 次，每次算同樣的
         NEAR−FAR 命中率差 → 分佈。費波那契的差落在這個分佈的哪個分位，就是它
         「比隨機幾何多出來的內容」。這是本檔的主判準。
**判準**
  (1) NEAR − FAR 命中率差 ≥ +8pp（地形扳機的門檻，沿用）
  (2) 費波那契的差在隨機水位分佈的第 95 百分位以上（有超過隨機幾何的內容）
  (3) 前後兩半 NEAR−FAR 同號
  三條全過 = 進地形候選（門口）；否則 NO-GO。功效：n≈127，NEAR 桶約 30–50 →
  差的 SE ≈ 9–10pp，MDE ≈ 27pp。**只測得出大效應**，先寫明。
**先驗**：差 ≈ 0，落在隨機分佈中段。NO-GO。

Run: python research/fib_proximity_test.py
Out: research/results/fib_proximity_test.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "fib_proximity_test.json"
SWING, NEAR_ATR, FIB = 72, 0.5, (0.382, 0.5, 0.618)


def main():
    rows = list(csv.DictReader(open(ROOT / "research/sweep_failure/.cache/BTCUSDT_1h.csv", newline="")))
    k = pd.DataFrame({c: [float(r[c]) for r in rows] for c in ("high", "low", "close")},
                     index=pd.to_datetime([int(r["time"]) for r in rows], unit="s", utc=True))
    tr = np.maximum(k["high"] - k["low"], np.maximum((k["high"] - k["close"].shift()).abs(),
                                                      (k["low"] - k["close"].shift()).abs()))
    k["atr"] = tr.rolling(14).mean()
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT signal_time, direction, entry_price, actual_return_4h FROM tracked_signals "
                        "WHERE strength='Strong' AND signal_time>='2026-04-03' AND actual_return_4h IS NOT NULL "
                        "ORDER BY signal_time")
            s = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    s["t"] = pd.to_datetime(s["signal_time"], utc=True)
    s["sgn"] = np.where(s["direction"].astype(str).str.upper().str.startswith("UP"), 1, -1)
    s["hit"] = (s["sgn"] * s["actual_return_4h"].astype(float) > 0)
    recs = []
    for _, r in s.iterrows():
        if r["t"] not in k.index:
            continue
        i = k.index.get_loc(r["t"])
        if i < SWING + 15:
            continue
        w = k.iloc[i - SWING:i]
        H, L, atr = w["high"].max(), w["low"].min(), k["atr"].iloc[i - 1]
        ep = float(r["entry_price"])
        recs.append({"hit": bool(r["hit"]), "H": H, "L": L, "atr": atr, "ep": ep})
    d = pd.DataFrame(recs); n = len(d)

    def gap(ratios_per_row):
        lv = np.array([[L + q * (H - L) for q in rs] for (H, L), rs in zip(zip(d.H, d.L), ratios_per_row)])
        dist = np.min(np.abs(lv - d.ep.values[:, None]), axis=1) / d.atr.values
        near = dist <= NEAR_ATR
        if near.sum() < 5 or (~near).sum() < 5:
            return np.nan, near
        return d.hit[near].mean() - d.hit[~near].mean(), near

    g_fib, near = gap([FIB] * n)
    half = n // 2
    h1 = d.hit[:half][near[:half]].mean() - d.hit[:half][~near[:half]].mean()
    h2 = d.hit[half:][near[half:]].mean() - d.hit[half:][~near[half:]].mean()
    rng = np.random.default_rng(618)
    rand = np.array([gap([tuple(rng.uniform(0.05, 0.95, 3)) for _ in range(n)])[0] for _ in range(500)])
    rand = rand[np.isfinite(rand)]
    pct = float((rand < g_fib).mean())
    print("=" * 84)
    print(f"  費波那契回撤位 × Strong 進場  n={n}  NEAR(≤{NEAR_ATR} ATR) {near.sum()} / FAR {(~near).sum()}")
    print("=" * 84)
    print(f"  NEAR 命中 {d.hit[near].mean()*100:.1f}%  FAR 命中 {d.hit[~near].mean()*100:.1f}%  差 {g_fib*100:+.1f}pp"
          f"  兩半 {h1*100:+.1f}/{h2*100:+.1f}pp")
    print(f"  隨機水位對照（500 次）：差的分佈 中位 {np.median(rand)*100:+.1f}pp  p5 {np.percentile(rand,5)*100:+.1f}"
          f"  p95 {np.percentile(rand,95)*100:+.1f}  → 費波那契落在第 {pct*100:.0f} 百分位")
    se = np.sqrt(0.25 / max(near.sum(), 1) + 0.25 / max((~near).sum(), 1))
    print(f"  功效：差的 SE {se*100:.1f}pp，MDE {2.802*se*100:.0f}pp")
    c1 = g_fib >= 0.08; c2 = pct >= 0.95; c3 = np.sign(h1) == np.sign(h2) and h1 > 0
    verdict = "門口候選" if (c1 and c2 and c3) else "NO-GO"
    print(f"\n  (1) 差 ≥ 8pp: {'過' if c1 else '不過'}  (2) 超過隨機幾何 p95: {'過' if c2 else '不過'}  (3) 兩半同號: {'過' if c3 else '不過'}"
          f"   ==> {verdict}")
    OUT.write_text(json.dumps({"n": n, "near": int(near.sum()), "gap_pp": g_fib * 100, "halves_pp": [h1 * 100, h2 * 100],
                               "random_pct": pct, "random_p5_p95": [float(np.percentile(rand, 5)) * 100, float(np.percentile(rand, 95)) * 100],
                               "se_pp": se * 100, "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
