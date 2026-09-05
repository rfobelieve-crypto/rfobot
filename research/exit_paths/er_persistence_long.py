# -*- coding: utf-8 -*-
"""ER 持續性的長視野版——同一套判準，換資料源換功效（2026-09-05）

`er_persistence_horizons.py` 在 1h / 4h 上給出乾淨的否定（CI 含零、逐幣不同號），
**但 24h 只有 n=384、CI 半寬 0.16，72h 直接跑不動**——因為它吃的是撤單資料的
時間範圍（44 天）。

**ER 只需要價格，不需要撤單資料。** 手上有 9 個幣、2.5 年的 1 小時 K 線
（`research/sweep_failure/.cache/*_1h.csv`），足以把 24h / 72h / 168h 的功效補起來。

**這是換資料源不是換判準**（要能被稽核）：
  - P1（CI 離零 ∧ |ρ| ≥ 0.10）、P2（≥8 幣同號）**一字不改**
  - 換的理由是**功效**，不是因為看到 24h 的點估計是 +0.04 才去找更多資料
    ——若當時 24h 是負的，一樣要補這個測試
  - **解析度不同要明講**：這裡的 ER 用小時收盤算，路徑長比分鐘級粗，所以
    ER 的絕對值會偏高。trailing 與 forward 用同一種解析度，比值的意義不變，
    但**不可與分鐘版的 ρ 直接比大小**
  - 對照關 P3 改成：**4h 這一格要與分鐘版同號且同樣不顯著**（分鐘版 −0.0176）。
    重現不了就是儀器壞了，長視野的數字一律不採信

Run: python research/exit_paths/er_persistence_long.py
Out: research/results/er_persistence_long.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "er_persistence_long.json"
HORIZONS_H = (4, 24, 72, 168)
RHO_FLOOR = 0.10


def er(p):
    if len(p) < 3:
        return np.nan
    path = np.abs(np.diff(p)).sum()
    return abs(p[-1] - p[0]) / path if path > 0 else np.nan


def dblock(v, days, B=2000, seed=17):
    rng = np.random.default_rng(seed)
    g = {}
    for x, dd in zip(v, days):
        if np.isfinite(x):
            g.setdefault(int(dd), []).append(x)
    ks = np.array(list(g))
    if len(ks) < 5:
        return float("nan"), float("nan")
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def rank_pairs(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rho = float(np.corrcoef(rx, ry)[0, 1])
    prod = (rx - rx.mean()) * (ry - ry.mean()) / (rx.std() * ry.std())
    return rho, prod


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 96)
    print("  ER 持續性 · 長視野（1h K 線，2.5 年）｜判準與分鐘版相同，只換資料源")
    print("=" * 96)
    data = {}
    for f in sorted(CACHE.glob("*USDT_1h.csv")):
        sym = f.name.split("USDT")[0]
        rows = list(csv.DictReader(open(f, newline="")))
        if len(rows) < 5000:
            continue
        t = np.array([int(r["time"]) for r in rows], np.int64)
        c = np.array([float(r["close"]) for r in rows], float)
        data[sym] = (t, c)
    if not data:
        print("  無 K 線快取"); return 0
    span = max((v[0][-1] - v[0][0]) / 86400 for v in data.values())
    print(f"  {len(data)} 個幣 · 每個約 {len(next(iter(data.values()))[1]):,} 根 · 跨度 {span:.0f} 天\n")

    res = {"coins": sorted(data), "rho_floor": RHO_FLOOR, "horizons": {}}
    print(f"  {'視野':>6}{'n':>8}{'rank 相關':>11}{'CI':>22}{'逐幣同號':>10}  判定")
    for H in HORIZONS_H:
        rows, per = [], {}
        for sym, (t, c) in data.items():
            v = []
            for i in range(H, len(c) - H, H):          # 不重疊
                a, b = er(c[i - H:i]), er(c[i:i + H])
                if np.isfinite(a) and np.isfinite(b):
                    v.append((a, b, t[i] // 86400))
            if len(v) < 50:
                continue
            arr = np.array(v, float)
            per[sym] = float(rank_pairs(arr[:, 0], arr[:, 1])[0])
            rows.append(arr)
        if not rows:
            print(f"  {H:>5}h  樣本不足"); continue
        A = np.vstack(rows)
        rho, prod = rank_pairs(A[:, 0], A[:, 1])
        lo, hi = dblock(prod, A[:, 2])
        same = sum(1 for r in per.values() if np.sign(r) == np.sign(rho))
        p1 = (lo * hi > 0) and abs(rho) >= RHO_FLOOR
        p2 = same >= 8
        res["horizons"][f"{H}h"] = {"n": int(len(A)), "rho": rho, "ci": [lo, hi],
                                    "same_sign": same, "coins": len(per),
                                    "per_coin": per, "P1": bool(p1), "P2": bool(p2)}
        print(f"  {H:>5}h{len(A):>8}{rho:>+11.4f}  [{lo:+7.4f},{hi:+7.4f}]{same:>6}/{len(per):<3}"
              f"  P1 {'過' if p1 else '不過'}  P2 {'過' if p2 else '不過'}")

    h4 = res["horizons"].get("4h", {})
    p3 = bool(h4) and (h4["ci"][0] * h4["ci"][1] <= 0)     # 4h 應同樣不顯著
    res["P3_control_ok"] = p3
    print(f"\n  P3 對照（4h 應與分鐘版同樣不顯著）：CI [{h4.get('ci',[float('nan')]*2)[0]:+.4f},"
          f"{h4.get('ci',[float('nan')]*2)[1]:+.4f}] → {'一致，儀器可信' if p3 else '**不一致，數字不採信**'}")
    passed = [k for k, v in res["horizons"].items() if v["P1"] and v["P2"]]
    res["verdict"] = ("儀器未通過對照，不下結論" if not p3 else
                      (f"左右軸畫得出來：{', '.join(passed)}" if passed else
                       "路徑形態在 4h–168h 上同樣不可預測 → 網格只能做上半排"))
    print(f"  ==> {res['verdict']}")

    # 功效聲明：每個視野實際能測到多小的效應
    print("\n  功效（各視野的 MDE，80%）：", end="")
    for k, v in res["horizons"].items():
        se = (v["ci"][1] - v["ci"][0]) / (2 * 1.96)
        print(f"  {k} ±{2.802*se:.3f}", end="")
        v["mde"] = float(2.802 * se)
    print()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
