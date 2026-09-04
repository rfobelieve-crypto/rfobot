# -*- coding: utf-8 -*-
"""trailing-vol 閘門的 1σ 突破 —— 不重疊窗口的 OOS（預註冊 2026-09-05）

**來源**：`flow_breakout_gate.py`（§1.14c）裡，TRAILVOL 是**對照臂**，卻在
2026-07-09 → 09-04 的 51 天上出 +36 bps/筆、10/10 幣正、兩半同號（日區塊 CI
含零）。那是看過資料後的觀察，在那個窗口不能變成假設。它的嫌疑很具體：
生存層同期讀到 ADX TRENDING 8/9 幣、breakout PAID 7/9——**那段本來就對突破
友善**。所以唯一乾淨的做法是拿到**沒看過、跨多個 regime** 的資料上驗。

**資料**：`sweep_failure/.cache/*_1h.csv`，2024-02-15 起。**窗口 2024-02-15 →
2026-07-08 23:00 UTC，與 §1.14c 完全不重疊**（後者從 07-09 00:00 起）。
同一批 10 幣（BTC ETH BNB XRP ADA DOGE LINK SUI UNI AAVE）。

**從 1m 版翻譯到 1h 版，跑前寫死（是翻譯不是調參）**：
  σ_1h(i−1) = 過去 24 根小時 log 報酬的 std（trailing-only）
  σ_4h      = σ_1h × 2（√4；1m 版是 tv × √240，同一個量的尺度換算）
  閘門      σ_1h(i−1) > 其 trailing 168 根第 80 百分位（同 LB、同 Q）
  中心 c    = bar i 的 open；多在 c(1+σ_4h)，空在 c(1−σ_4h)，k = 1
  觸發      bar i..i+3 的 high/low，先碰到哪邊進哪邊，成交價＝觸發價
  **同一根 bar 兩邊都碰到**：順序不可知 → 保守假設「進場後立刻被打回中心」，
            gross = −σ_4h（1m 版沒這個問題；這裡吃的是對策略**不利**的假設）
  停損      進場**之後**的 bar，low ≤ c（多）／high ≥ c（空）→ 以 c 出場
            進場那根 bar 本身看不到內部順序 → 不判停損（**對策略有利**的盲點，
            與上一條相抵；兩者都寫在這裡，讀結果時要記得）
  平倉      bar i+3 收盤
  成本      10 bps/邊（同 §1.14c）；另報零成本

**三臂**：TRAILVOL（假設）／RANDOM 20%（固定種子，對照）／ALL（無條件）。

**判準（凍結）**：GO 必須同時
  (1) TRAILVOL 淨 bps > 0，日區塊 bootstrap 95% CI 不含零（10 幣合併）
  (2) TRAILVOL − RANDOM > 0 且 CI 不含零
  (4) 10 幣裡淨為正的 ≥ 7
  (5) 前後兩半同號
  (6) **逐季**：淨為正的季 ≥ 75%（regime 穩健性——這條是為 §1.14c 那個嫌疑加的）
  任一不過 = NO-GO。不調 k、不換窗、不換 Q。

**預測（寫死）**：先驗四六——四成它是 7–9 月的 regime 假象（(6) 會殺它），
六成突破在高波動 regime 的延續是真的但被成本吃到貼零（(1) CI 含零）。
若六條全過，那是今天挖到的第一個「結構型、不需要流資料、成本付得起」的策略。

Run: python research/vol_breakout_oos.py
Out: research/results/vol_breakout_oos.json
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "vol_breakout_oos.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "LINK", "SUI", "UNI", "AAVE"]
END = int(dt.datetime(2026, 7, 9, tzinfo=dt.timezone.utc).timestamp())   # exclusive
K, W, LB, Q, COST_SIDE = 1.0, 4, 168, 0.80, 10.0


def load(sym):
    t, o, h, l, c = [], [], [], [], []
    with open(CACHE / f"{sym}USDT_1h.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            ts = int(r["time"])
            if ts >= END:
                break
            t.append(ts); o.append(float(r["open"])); h.append(float(r["high"]))
            l.append(float(r["low"])); c.append(float(r["close"]))
    return (np.array(t), np.array(o), np.array(h), np.array(l), np.array(c))


def run_coin(sym, rng):
    t, o, h, l, c = load(sym)
    r = np.diff(np.log(c), prepend=np.nan)
    s1 = pd.Series(r).rolling(24, min_periods=12).std().values
    thr = pd.Series(s1).rolling(LB, min_periods=LB // 2).quantile(Q).values
    n = len(c)
    fire = {"TRAILVOL": np.zeros(n, bool), "RANDOM": rng.random(n) < 0.20,
            "ALL": np.ones(n, bool)}
    fire["TRAILVOL"][1:] = (s1[:-1] > thr[:-1]) & np.isfinite(thr[:-1])
    out = {a: [] for a in fire}
    for i in range(LB + 1, n - W):
        sig = s1[i - 1] * 2.0
        if not np.isfinite(sig) or sig <= 0:
            continue
        cen = o[i]; up = cen * (1 + K * sig); dn = cen * (1 - K * sig)
        res = None
        for j in range(i, i + W):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                res = (-sig * 1e4, "both"); break
            if hu or hd:
                side = 1 if hu else -1
                entry = up if hu else dn
                exit_px = c[i + W - 1]; reason = "time"
                for k in range(j + 1, i + W):
                    if (side > 0 and l[k] <= cen) or (side < 0 and h[k] >= cen):
                        exit_px = cen; reason = "stop"; break
                res = (side * (exit_px / entry - 1) * 1e4, reason); break
        if res is None:
            continue
        gross, reason = res
        day = t[i] // 86400
        qtr = f"{dt.datetime.utcfromtimestamp(t[i]).year}Q{(dt.datetime.utcfromtimestamp(t[i]).month-1)//3+1}"
        for a in fire:
            if fire[a][i]:
                out[a].append({"day": int(day), "q": qtr, "gross": gross,
                               "net": gross - 2 * COST_SIDE, "reason": reason})
    return {a: pd.DataFrame(v) for a, v in out.items()}


def dblock(df, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    g = {d: v["net"].values for d, v in df.groupby("day")}; days = np.array(list(g))
    bs = [np.concatenate([g[d] for d in rng.choice(days, len(days))]).mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def diff_ci(a, b, B=2000, seed=8):
    rng = np.random.default_rng(seed)
    da = {d: v["net"].values for d, v in a.groupby("day")}
    db = {d: v["net"].values for d, v in b.groupby("day")}
    days = np.array(sorted(set(da) | set(db))); bs = []
    for _ in range(B):
        p = rng.choice(days, len(days))
        bs.append(np.concatenate([da[d] for d in p if d in da]).mean()
                  - np.concatenate([db[d] for d in p if d in db]).mean())
    return float(np.mean(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rng = np.random.default_rng(20260905)
    print("=" * 100)
    print(f"  trailing-vol 閘門 1σ 突破 · OOS 2024-02-15 → 2026-07-08 · 1h bar · 成本 {COST_SIDE:.0f} bps/邊")
    print("=" * 100)
    pooled = {a: [] for a in ("TRAILVOL", "RANDOM", "ALL")}; per = {}
    print(f"  {'幣':6}" + "".join(f"{a:>20}" for a in pooled) + "   (n / 淨 bps/筆)")
    for sym in COINS:
        res = run_coin(sym, rng)
        per[sym] = {a: {"n": int(len(d)), "net": float(d["net"].mean()) if len(d) else None}
                    for a, d in res.items()}
        for a, d in res.items():
            d = d.copy(); d["sym"] = sym; pooled[a].append(d)
        print(f"  {sym:6}" + "".join(f"{per[sym][a]['n']:>9d}/{(per[sym][a]['net'] or 0):>+9.1f}" for a in pooled))
    P = {a: pd.concat(v, ignore_index=True) for a, v in pooled.items()}
    print(); summ = {}
    for a, d in P.items():
        lo, hi = dblock(d); d2 = d.sort_values("day"); half = len(d) // 2
        qs = d.groupby("q")["net"].mean()
        summ[a] = {"n": int(len(d)), "net": float(d["net"].mean()), "gross": float(d["gross"].mean()),
                   "ci": [lo, hi], "wr": float((d["net"] > 0).mean()),
                   "halves": [float(d2["net"].iloc[:half].mean()), float(d2["net"].iloc[half:].mean())],
                   "coins_pos": int(sum(1 for s in COINS if (per[s][a]["net"] or 0) > 0)),
                   "q_pos": int((qs > 0).sum()), "q_n": int(len(qs)),
                   "quarters": {k: float(v) for k, v in qs.items()},
                   "reasons": {k: int(v) for k, v in d["reason"].value_counts().items()}}
        s = summ[a]
        print(f"  {a:9} n={s['n']:6d}  淨 {s['net']:+6.1f} [{lo:+.1f}, {hi:+.1f}]  毛 {s['gross']:+6.1f}"
              f"  WR {s['wr']*100:4.1f}%  兩半 {s['halves'][0]:+.1f}/{s['halves'][1]:+.1f}"
              f"  幣正 {s['coins_pos']}/10  季正 {s['q_pos']}/{s['q_n']}  出場 {s['reasons']}")
    print("\n  逐季（TRAILVOL 淨 bps）: " + "  ".join(f"{k}:{v:+.0f}" for k, v in summ["TRAILVOL"]["quarters"].items()))
    d_rd = diff_ci(P["TRAILVOL"], P["RANDOM"])
    print(f"  TRAILVOL − RANDOM: {d_rd[0]:+.1f} [{d_rd[1]:+.1f}, {d_rd[2]:+.1f}]")
    T = summ["TRAILVOL"]
    c1 = T["net"] > 0 and T["ci"][0] > 0
    c2 = d_rd[0] > 0 and d_rd[1] > 0
    c4 = T["coins_pos"] >= 7
    c5 = np.sign(T["halves"][0]) == np.sign(T["halves"][1]) and T["halves"][0] > 0
    c6 = T["q_pos"] / T["q_n"] >= 0.75
    verdict = "GO" if all((c1, c2, c4, c5, c6)) else "NO-GO"
    print(f"\n  (1) {'過' if c1 else '不過'} (2) {'過' if c2 else '不過'} (4) {'過' if c4 else '不過'}"
          f" (5) {'過' if c5 else '不過'} (6) {'過' if c6 else '不過'}   ==> {verdict}")
    OUT.write_text(json.dumps({"summary": summ, "per_coin": per, "diff_random": d_rd, "verdict": verdict,
                               "c": [bool(x) for x in (c1, c2, c4, c5, c6)]}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
