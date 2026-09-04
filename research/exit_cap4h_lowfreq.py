# -*- coding: utf-8 -*-
"""「4 小時到就出場」在低頻 regime 的配對驗證 —— 預註冊 2026-09-05

**使用者**：「你只看了進場的落點，我覺得落點在於他的出場。如果我四小時到就出場。」

**為什麼要重測**：`cap_4h` 今天在 `exit_variants_backtest` 輸 baseline 30.9 bps
（§0.88），但那個回測的 regime 裡 `opp_signal` 佔 60% 出場、每筆 +97——trailing
贏是因為它把倉位留給 opp_signal 收。**現在 opp_signal 只剩 11%**（top-5% 重定義
餓死它），trailing 留下的倉位交給 trail_stop（兩個世界一致 −40 bps）。live 18 筆
反算：4h 定時出場 +21 net vs 實際 −5，但 n=18 CI 含零。所以要在**低頻 regime
的全部 Strong**（2026-04-03 起，n=127）上配對驗。

**設計（跑前寫死）**
  訊號   tracked_signals Strong，signal_time ≥ 2026-04-03（dual v7 上線）
  進場   訊號 bar 是標籤，開火在 label+1h 收盤後，executor 約 2.5 分鐘後成交
         → 用 **label+1h bar 的開盤價**（mistake.md 2026-07-28 的時間對齊）
  臂 A   baseline trailing：3×ATR(14) 移動停損（每根 bar 用 high/low 觸發、
         以停損價成交）＋ 反向 Strong 出現即平（live 的 opp_signal 語意）
         ＋ 72h 時間上限。與 live executor 同形（conviction_decay 不在，它 live
         也只佔 11%）。
  臂 B   cap_4h：進場後第 4 根 bar 收盤平倉，中途**沒有**停損（純粹「模型說
         4h 就 4h」）。
  成本   兩臂同 9 bps 來回（實測）。配對差對成本中性。
  獨立   每筆訊號獨立評估（不套單倉制）——問的是「給定這個進場，哪個出場好」，
         不是槽位效應；槽位效應另報（cap_4h 釋放得快，能多吃幾筆）。
  判準   **配對差（B − A）≥ +22 bps ∧ 日區塊 bootstrap 95% CI 不含零 ∧ 前後兩半
         同號**。功效預檢：n=127、配對 sd 101 → 門檻 22 = 2.45×SE（<2.8，略
         不足；MDE≈25 bps）。判準是 CI 型，構造上不會被雜訊矇過，但可能徒勞
         ——徒勞就明寫。
  預測   B − A 為正（機制：opp_signal 餓死後 trailing 只剩 trail_stop）；
         是否 ≥ 22 且 CI 離零：五五開。

Run: python research/exit_cap4h_lowfreq.py
Out: research/results/exit_cap4h_lowfreq.json
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

CSV = ROOT / "research" / "sweep_failure" / ".cache" / "BTCUSDT_1h.csv"
OUT = ROOT / "research" / "results" / "exit_cap4h_lowfreq.json"
COST = 9.0          # bps round trip, measured on live fills
TRAIL = 3.0
CAP_H = 72
HOLD4 = 4


def bars():
    rows = list(csv.DictReader(open(CSV, newline="")))
    df = pd.DataFrame({k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close")})
    df.index = pd.to_datetime([int(r["time"]) for r in rows], unit="s", utc=True)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    df["atr"] = tr.rolling(14).mean()
    return df


def signals():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT signal_time, direction FROM tracked_signals "
                        "WHERE strength='Strong' AND signal_time>='2026-04-03' ORDER BY signal_time")
            rows = cur.fetchall()
    finally:
        conn.close()
    s = pd.DataFrame(rows)
    s["t"] = pd.to_datetime(s["signal_time"], utc=True)
    s["side"] = np.where(s["direction"].astype(str).str.upper().str.startswith("UP"), 1, -1)
    return s[["t", "side"]]


def sim_trailing(df, i0, side, opp_times):
    """Enter at open of bar i0; trail 3xATR on high/low; opp Strong exits at
    that bar's open; 72h cap at close."""
    entry = df["open"].iloc[i0]
    atr = df["atr"].iloc[i0 - 1]
    stop = entry - side * TRAIL * atr
    extreme = entry
    for k in range(i0, min(i0 + CAP_H, len(df))):
        t = df.index[k]
        if k > i0 and t in opp_times:
            return side * (df["open"].iloc[k] / entry - 1) * 1e4, "opp_signal"
        hi, lo = df["high"].iloc[k], df["low"].iloc[k]
        # stop hit this bar?
        if (side > 0 and lo <= stop) or (side < 0 and hi >= stop):
            return side * (stop / entry - 1) * 1e4, "trail_stop"
        # ratchet
        if side > 0 and hi > extreme:
            extreme = hi; stop = max(stop, extreme - TRAIL * atr)
        elif side < 0 and lo < extreme:
            extreme = lo; stop = min(stop, extreme + TRAIL * atr)
    k = min(i0 + CAP_H, len(df)) - 1
    return side * (df["close"].iloc[k] / entry - 1) * 1e4, "time_cap"


def sim_cap4h(df, i0, side):
    entry = df["open"].iloc[i0]
    k = min(i0 + HOLD4 - 1, len(df) - 1)
    return side * (df["close"].iloc[k] / entry - 1) * 1e4


def dblock(v, days, B=3000, seed=5):
    rng = np.random.default_rng(seed)
    g = {}
    for x, d in zip(v, days):
        g.setdefault(d, []).append(x)
    ks = np.array(list(g))
    out = [np.concatenate([g[d] for d in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = bars(); s = signals()
    up_t = set(s[s.side > 0]["t"]); dn_t = set(s[s.side < 0]["t"])
    rows = []
    for _, r in s.iterrows():
        t_entry = r["t"] + pd.Timedelta(hours=1)     # fires at label+1h close → next bar open
        if t_entry not in df.index:
            continue
        i0 = df.index.get_loc(t_entry)
        if i0 < 20 or i0 + CAP_H >= len(df):
            continue
        opp = dn_t if r["side"] > 0 else up_t
        # opp exits trigger at the bar AFTER the reverse signal's label (its fire time)
        opp_fire = {x + pd.Timedelta(hours=1) for x in opp}
        a, why = sim_trailing(df, i0, r["side"], opp_fire)
        b = sim_cap4h(df, i0, r["side"])
        rows.append({"t": r["t"], "side": r["side"], "trail": a - COST, "cap4h": b - COST,
                     "why": why, "day": int(t_entry.timestamp() // 86400)})
    d = pd.DataFrame(rows); d["diff"] = d["cap4h"] - d["trail"]
    n = len(d); half = n // 2
    lo, hi = dblock(d["diff"].values, d["day"].values)
    h1, h2 = d["diff"].iloc[:half].mean(), d["diff"].iloc[half:].mean()
    print("=" * 92)
    print(f"  低頻 regime（2026-04-03 起）Strong n={n}  同一批訊號、同一套 1h K 線、每筆獨立")
    print("=" * 92)
    for arm in ("trail", "cap4h"):
        v = d[arm]
        print(f"  {arm:6} 淨 {v.mean():+7.1f} bps/筆  WR {(v>0).mean()*100:4.1f}%  中位 {v.median():+6.1f}  "
              f"最差 {v.min():+7.1f}  最好 {v.max():+7.1f}")
    print(f"  trailing 出場理由: {dict(d['why'].value_counts())}")
    print(f"\n  配對差 cap4h − trail: {d['diff'].mean():+.1f} bps  95%CI [{lo:+.1f}, {hi:+.1f}]  兩半 {h1:+.1f}/{h2:+.1f}"
          f"  SE {d['diff'].std(ddof=1)/np.sqrt(n):.1f}")
    for side, lab in ((1, "LONG"), (-1, "SHORT")):
        x = d[d.side == side]
        print(f"    {lab:5} n={len(x):3d}  trail {x.trail.mean():+6.1f}  cap4h {x.cap4h.mean():+6.1f}  差 {x['diff'].mean():+6.1f}")
    c1 = d["diff"].mean() >= 22; c2 = lo > 0; c3 = np.sign(h1) == np.sign(h2) and h1 > 0
    verdict = "GO" if (c1 and c2 and c3) else "NO-GO"
    print(f"\n  (1) 差 ≥ +22: {'過' if c1 else '不過'}  (2) CI 不含零: {'過' if c2 else '不過'}  (3) 兩半同號: {'過' if c3 else '不過'}"
          f"   ==> {verdict}")
    # slot effect (informational): how many signals were skipped under one-slot with each policy
    OUT.write_text(json.dumps({"n": n, "trail_mean": float(d.trail.mean()), "cap4h_mean": float(d.cap4h.mean()),
                               "diff": float(d["diff"].mean()), "ci": [lo, hi], "halves": [float(h1), float(h2)],
                               "reasons": {k: int(v) for k, v in d["why"].value_counts().items()},
                               "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
