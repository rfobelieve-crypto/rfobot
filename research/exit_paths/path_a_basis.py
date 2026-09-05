# -*- coding: utf-8 -*-
"""路徑 A 的 A0 前置檢查：跨場館基差與領先落後（2026-09-05）

A0 不需要訊號分鐘，只需要兩條 mid 序列，所以**現在就能算**——如果基差的日內
標準差 > 6 bps，PREREG 說 A.2.2 之後全部作廢、改走「在 Lighter 價格上重跑 G1」。
早知道早改路線。

兩個對照場館分別判（PREREG 修正 2）：
  **Binance 現貨** —— §1.15 訊號的原生資料，決定「訊號能不能直接搬過去」
  **Bitget 永續** —— 產品端真正下單的地方，決定「換場館執行要不要重算訊號」

領先落後：對 250 ms 取樣的 Δmid 做互相關，掃 ±5 秒（±20 格）。PREREG：
若 Lighter 系統性領先或落後 > 500 ms，A.2.2 的 t0 要修正。

**這支對「資料還不夠一天」是誠實的**：不足 24 小時就印「初判」，不寫判決。

Run: python research/exit_paths/path_a_basis.py
Out: research/results/path_a_basis.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "research" / "exit_paths" / "logs" / "lighter"
OUT = ROOT / "research" / "results" / "path_a_basis.json"
THR_BPS, LAG_MS = 6.0, 500.0
STEP_MS = 250


def load():
    """回傳 (ts_ms, mid_lighter, mid_binance, mid_bitget|nan)。舊格式沒有 bitget 欄。"""
    rows = []
    for f in sorted(LOGS.glob("*.csv")):
        with open(f, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "l_bid" not in rd.fieldnames:
                continue
            has_g = "g_bid" in rd.fieldnames
            for r in rd:
                try:
                    lb, la = float(r["l_bid"]), float(r["l_ask"])
                    bb, ba = float(r["b_bid"]), float(r["b_ask"])
                    g = ((float(r["g_bid"]) + float(r["g_ask"])) / 2
                         if has_g and r.get("g_bid") else np.nan)
                    rows.append((int(r["ts_ms"]), (lb + la) / 2, (bb + ba) / 2, g))
                except (TypeError, ValueError):
                    continue
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows, float)
    return a[np.argsort(a[:, 0])]      # 按時間排，不按檔名——輪替後的新檔檔名
                                       # 反而排在舊檔前面（'.' < '_'）


def lead_lag(a, b, max_lag=20):
    """argmax_k corr(Δa(t), Δb(t−k))。k>0 = b 落後 a（a 領先）。"""
    da, db = np.diff(a), np.diff(b)
    best, bl = 0.0, 0
    for k in range(-max_lag, max_lag + 1):
        x = da[max_lag:-max_lag] if max_lag else da
        y = db[max_lag - k: len(db) - max_lag - k] if max_lag else db
        n = min(len(x), len(y))
        if n < 100:
            continue
        x2, y2 = x[:n], y[:n]
        if x2.std() == 0 or y2.std() == 0:
            continue
        c = float(np.corrcoef(x2, y2)[0, 1])
        if abs(c) > abs(best):
            best, bl = c, k
    return best, bl * STEP_MS


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    d = load()
    if len(d) < 500:
        print("樣本不足（<500 列），先讓錄製器跑一陣子"); return 0
    hours = (d[-1, 0] - d[0, 0]) / 3.6e6
    print("=" * 96)
    print(f"  路徑 A · A0 前置檢查｜{len(d):,} 列 × 250 ms ≈ {hours:.2f} 小時"
          f"｜門檻：基差日內 σ ≤ {THR_BPS} bps")
    print("=" * 96)
    res = {"rows": int(len(d)), "hours": round(hours, 3), "thr_bps": THR_BPS, "venues": {}}
    for name, col in (("Binance 現貨", 2), ("Bitget 永續", 3)):
        m = np.isfinite(d[:, col])
        if m.sum() < 500:
            print(f"  {name:<12} 樣本 {m.sum()} 列 —— 不足，略過（Bitget 腿 2026-09-05 才加）")
            continue
        L, O = d[m, 1], d[m, col]
        bas = (L - O) / ((L + O) / 2) * 1e4
        c, lag = lead_lag(L, O)
        res["venues"][name] = {"rows": int(m.sum()), "mean_bps": float(bas.mean()),
                               "sd_bps": float(bas.std()), "p1_p99": [float(np.percentile(bas, 1)),
                               float(np.percentile(bas, 99))], "corr": c, "lag_ms": lag,
                               "pass_a0": bool(bas.std() <= THR_BPS), "pass_lag": bool(abs(lag) <= LAG_MS)}
        v = res["venues"][name]
        print(f"  {name:<12} 基差均值 {v['mean_bps']:+7.2f} bps   σ {v['sd_bps']:5.2f} bps"
              f"   1–99% [{v['p1_p99'][0]:+.1f},{v['p1_p99'][1]:+.1f}]"
              f"   Δmid 相關 {c:+.3f} @ {lag:+.0f} ms")
        print(f"  {'':<12} A0（σ ≤ {THR_BPS}）：{'過' if v['pass_a0'] else '不過'}"
              f"   領先落後（|lag| ≤ {LAG_MS:.0f} ms）：{'過' if v['pass_lag'] else '不過 → t0 要修正'}")
    if hours < 24:
        res["status"] = "初判（不足 24 小時，不是判決）"
        print(f"\n  ⏳ 只有 {hours:.2f} 小時，**這是初判不是判決**——PREREG 要的是日內 σ，"
              f"至少要一個完整交易日（含亞洲/歐洲/美國時段各自的波動）")
    else:
        res["status"] = "判決"
        allp = all(v["pass_a0"] for v in res["venues"].values())
        print(f"\n  ==> A0 {'PASS，進 A.2.2' if allp else 'REJECT，改走「在 Lighter 價格上重跑 G1」'}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
