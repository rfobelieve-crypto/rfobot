# -*- coding: utf-8 -*-
"""道哥「主力資金沙盤」的交集測試：存量 × 流量同向極端 → 延續？（預註冊 2026-09-05）

原文與翻譯在 docs/orderflow_daoge.md。這裡只做那份文件末節的預註冊草案，
一個字不改：

  Y（存量）  Coinglass 期貨聚合 CVD 的 delta：agg_taker_buy − agg_taker_sell，
             用 V7 特徵建構器凍結的 _zscore（rolling 24h, min 4）。
  Z（流量）  flow_bars_1m 的平均單筆成交額 volume_usd/trade_count，逐分鐘算、
             每小時取最大、rolling 168h z-score（trailing-only）。
  X          該小時報酬（1h K 線）。
  時間對齊   同 V7：Coinglass 標籤 t 屬於 bar t；forward 從 bar t 收盤起算。
  條件桶     JOINT = Z_z > 1 ∧ |Y_z| > 1，方向 = sign(Y)。
             JOINT_Q（圓圈壓境）= JOINT ∧ sign(X) = sign(Y)。
  對照       UNCOND：所有小時都跟 sign(Y)；Y_ONLY：|Y_z|>1；Z_ONLY：Z_z>1、
             方向 = sign(X)（Z 沒有方向，用戰果所在側）。
  標籤       forward 1h / 4h 收盤報酬 × 方向（同向為正）。
  判準       JOINT 命中率 − UNCOND ≥ 3pp ∧ 日區塊 bootstrap 95% CI 離零 ∧
             前後兩半同號，至少一個 horizon。**跑前用 prereg_power 報 MDE**：
             條件桶 n≈250 → 命中率 SE≈3pp → MDE≈9pp，3pp 的門檻本身功效不足，
             所以真正能下結論的是 CI 那條；3pp 只是方向要求。
  先驗       1h/4h 我押反轉（JOINT 命中率 < 50%）；條件化後不確定。

Run: python research/daoge_sandbox_test.py
Out: research/results/daoge_sandbox_test.json
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

OUT = ROOT / "research" / "results" / "daoge_sandbox_test.json"
ZWIN, ZMIN = 24, 4          # frozen with feature_builder_live._zscore
ZWIN_FLOW = 168
THR = 1.0


def zscore(s, win=ZWIN, mn=ZMIN):
    m = s.rolling(win, min_periods=mn).mean()
    sd = s.rolling(win, min_periods=mn).std().replace(0, np.nan)
    return (s - m) / sd


def load():
    cg = pd.read_parquet(ROOT / "market_data/raw_data/cg_futures_cvd_agg_1h.parquet")
    cg.index = pd.to_datetime(cg.index, utc=True)
    y = (cg["agg_taker_buy_vol"] - cg["agg_taker_sell_vol"]).astype(float)
    Y = zscore(y).rename("Y")
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT window_start, volume_usd, trade_count FROM flow_bars_1m "
                        "WHERE canonical_symbol='BTC-USD' ORDER BY window_start")
            f = pd.DataFrame(cur.fetchall())
    finally:
        conn.close()
    ws = f["window_start"]
    f["t"] = pd.to_datetime(ws, unit="ms", utc=True) if np.issubdtype(ws.dtype, np.number) else pd.to_datetime(ws, utc=True)
    f["avg"] = f["volume_usd"].astype(float) / f["trade_count"].astype(float).replace(0, np.nan)
    zh = f.set_index("t")["avg"].resample("1h").max()
    Z = zscore(zh, ZWIN_FLOW, ZWIN_FLOW // 2).rename("Z")
    rows = list(csv.DictReader(open(ROOT / "research/sweep_failure/.cache/BTCUSDT_1h.csv", newline="")))
    k = pd.DataFrame({"close": [float(r["close"]) for r in rows]},
                     index=pd.to_datetime([int(r["time"]) for r in rows], unit="s", utc=True))
    k["X"] = k["close"].pct_change()
    k["f1"] = k["close"].shift(-1) / k["close"] - 1
    k["f4"] = k["close"].shift(-4) / k["close"] - 1
    d = k.join(Y, how="inner").join(Z, how="inner").dropna()
    d["day"] = (d.index.view("int64") // 86_400_000_000_000).astype(int)
    return d


def dblock(v, days, B=3000, seed=9):
    rng = np.random.default_rng(seed); g = {}
    for x, dd in zip(v, days):
        g.setdefault(dd, []).append(x)
    ks = np.array(list(g))
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    d = load()
    print("=" * 96)
    print(f"  道哥沙盤交集測試 · BTC 1h · n={len(d)} 小時 ({d.index[0].date()} → {d.index[-1].date()})")
    print("  Y=Coinglass 期貨 CVD delta z(24h)  Z=平均單筆成交額 z(168h)  條件 |Y|>1 ∧ Z>1")
    print("=" * 96)
    sY = np.sign(d["Y"]); sX = np.sign(d["X"])
    arms = {
        "UNCOND (跟 sign Y)": (np.ones(len(d), bool), sY),
        "Y_ONLY |Y|>1":        ((d["Y"].abs() > THR).values, sY),
        "Z_ONLY Z>1 (跟 X)":   ((d["Z"] > THR).values, sX),
        "JOINT |Y|>1 ∧ Z>1":   (((d["Y"].abs() > THR) & (d["Z"] > THR)).values, sY),
        "JOINT_Q ∧ signX=signY": (((d["Y"].abs() > THR) & (d["Z"] > THR) & (sX == sY)).values, sY),
    }
    res = {}
    print(f"  {'臂':24}{'h':>3}{'n':>6}{'命中率':>8}{'均值bps':>9}{'CI(bps)':>18}{'兩半命中':>14}")
    for name, (mask, sg) in arms.items():
        res[name] = {}
        for h in ("f1", "f4"):
            sub = d[mask]; r = (sg[mask] * sub[h]).values * 1e4
            if len(r) < 20:
                continue
            hit = (r > 0).mean(); lo, hi = dblock(r, sub["day"].values)
            half = len(r) // 2
            h1, h2 = (r[:half] > 0).mean(), (r[half:] > 0).mean()
            res[name][h] = {"n": int(len(r)), "hit": float(hit), "mean": float(r.mean()),
                            "ci": [lo, hi], "halves": [float(h1), float(h2)]}
            print(f"  {name:24}{h[1:]:>2}h{len(r):>6d}{hit*100:>7.1f}%{r.mean():>+9.1f}"
                  f"  [{lo:+6.1f},{hi:+6.1f}]   {h1*100:5.1f}%/{h2*100:5.1f}%")
    # pre-registered criterion: JOINT vs UNCOND
    print()
    verdict = "NO-GO"; detail = []
    for h in ("f1", "f4"):
        J = res["JOINT |Y|>1 ∧ Z>1"].get(h); U = res["UNCOND (跟 sign Y)"].get(h)
        if not J or not U:
            continue
        # bootstrap the hit-rate difference by day
        mJ = arms["JOINT |Y|>1 ∧ Z>1"][0]; rJ = (sY[mJ] * d[mJ][h]).values; dJ = d[mJ]["day"].values
        rU = (sY * d[h]).values; dU = d["day"].values
        rng = np.random.default_rng(11); gJ, gU = {}, {}
        for x, dd in zip(rJ, dJ): gJ.setdefault(dd, []).append(x > 0)
        for x, dU_ in zip(rU, dU): gU.setdefault(dU_, []).append(x > 0)
        days = np.array(sorted(set(gU)))
        diffs = []
        for _ in range(2000):
            p = rng.choice(days, len(days))
            a = np.concatenate([gJ[dd] for dd in p if dd in gJ]); b = np.concatenate([gU[dd] for dd in p])
            diffs.append(a.mean() - b.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5]); dm = J["hit"] - U["hit"]
        c1 = dm >= 0.03; c2 = lo > 0; c3 = np.sign(J["halves"][0] - 0.5) == np.sign(J["halves"][1] - 0.5)
        ok = c1 and c2 and c3
        se = np.std(diffs)
        detail.append({"h": h, "diff_pp": dm * 100, "ci_pp": [lo * 100, hi * 100], "se_pp": se * 100,
                       "mde_pp": 2.802 * se * 100, "pass": bool(ok)})
        print(f"  JOINT − UNCOND @{h[1:]}h: {dm*100:+.1f}pp  CI [{lo*100:+.1f}, {hi*100:+.1f}]  "
              f"SE {se*100:.1f}pp (MDE {2.802*se*100:.1f}pp)  → (1){'過' if c1 else '不過'} (2){'過' if c2 else '不過'} (3){'過' if c3 else '不過'}")
        if ok:
            verdict = "GO"
    # the mirror reading (reversal) is just hit<50%: say it plainly
    print(f"\n  ==> {verdict}")
    OUT.write_text(json.dumps({"n": int(len(d)), "arms": res, "criterion": detail, "verdict": verdict},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
