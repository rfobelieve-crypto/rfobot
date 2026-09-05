# -*- coding: utf-8 -*-
"""網格二維矩陣的第二半：路徑形態在多長的視野上才可預測（預註冊，2026-09-05）

`er_regime_split.py` 已答出第一半：在 **30–60 分鐘**的視野上，擴張訊號分不出
路徑形態（E1）、對 ADX 沒有增量（E2）、而且 **ER 自己也沒有持續性**（E3，
rank 相關 −0.013、CI 含零）。連現任的 ADX 都分不出來（震盪 0.1328 vs 趨勢
0.1321，幾乎相同）。

**但那個視野是「擴張訊號」的尺度，不是網格的尺度。** 訊號峰值在 15–30 分鐘，
而網格持有庫存是數小時到數天，ADX(14) 在小時棒上刻畫的也是那個尺度。
**用 60 分鐘的結果去否定一個以天為單位的策略，是視野錯配。**

所以這支問同一個問題、換視野：**trailing ER 能不能預測 forward ER。**
如果連這個都不行，二維矩陣的左右軸在任何尺度上都畫不出來，網格的調節只能
做上半排（依 σ 調格距與部位），控制精度差一階——**那是一個明確的、可以據以
設計的結論，不是失敗。**

**視野由策略決定，不是由結果決定**（先寫再跑）：
  1h   對照組，應與 er_regime_split 的 E3 一致（−0.013、含零）
  4h   V7 的持有尺度、也是網格重錨的量級
  24h  網格一輪庫存周期的量級
  72h  週級，ADX(14) 在小時棒上的有效記憶長度

**判準（三條，全部事前寫死）**
  P1  rank 相關的日區塊 bootstrap 95% CI 離零，且 |ρ| ≥ 0.10
      （0.10 是「能拿來分桶」的下限；低於它分出來的兩桶會大量重疊）
  P2  ≥8/10 幣同號
  P3  對照組 1h 必須重現 er_regime_split 的結果（±0.02 內）——**重現不了
      就是這支儀器壞了，上面兩條一律不採信**（mistake.md 2026-08-11）
  任一視野同時過 P1、P2，且 P3 成立 ⇒ 左右軸在那個視野上畫得出來。

**先驗**：ER 的持續性應該隨視野拉長而上升（更長的窗把雜訊平均掉），所以
24h/72h 先驗偏過、1h 先驗不過。若**全部視野都不過**，結論是路徑形態不可預測，
網格只能做上半排。

Run: python research/exit_paths/er_persistence_horizons.py
Out: research/results/er_persistence_horizons.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from flow_grid_gate import load  # noqa: E402

OUT = ROOT / "research" / "results" / "er_persistence_horizons.json"
SYMS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
        "ADA-USD", "LINK-USD", "AVAX-USD", "SUI-USD", "UNI-USD", "BNB-USD", "AAVE-USD"]
EX = "binance"
HORIZONS_H = (1, 4, 24, 72)
RHO_FLOOR = 0.10
E3_REF = -0.013          # er_regime_split 的 60 分鐘結果，P3 的對照


def er(p):
    if len(p) < 3:
        return np.nan
    path = np.abs(np.diff(p)).sum()
    return abs(p[-1] - p[0]) / path if path > 0 else np.nan


def dblock(v, days, B=2000, seed=13):
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
    print("  ER 持續性 × 視野（網格二維矩陣的左右軸能不能畫）")
    print("=" * 96)
    mids = {}
    for s in SYMS:
        try:
            x = load(s, EX)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: {str(e)[:50]}"); continue
        if x is None or len(x) < 5000:
            continue
        mids[s] = (x["m"].values.astype(np.int64), x["mid_price"].astype(float).values)
    print(f"  可用標的 {len(mids)}：{', '.join(sorted(mids))}\n")

    res = {"horizons": {}, "rho_floor": RHO_FLOOR}
    print(f"  {'視野':>5}{'n':>8}{'rank 相關':>11}{'CI':>22}{'逐幣同號':>10}  判定")
    for H in HORIZONS_H:
        W = H * 60                      # 分鐘
        rows, per = [], {}
        for s, (ms, mid) in mids.items():
            v = []
            # 不重疊取樣：每個視野長度取一個點，避免相鄰樣本互相污染
            for i in range(W, len(mid) - W, W):
                a, b = er(mid[i - W:i]), er(mid[i:i + W])
                if np.isfinite(a) and np.isfinite(b):
                    v.append((a, b, ms[i] // 86_400_000))
            if len(v) < 30:
                continue
            arr = np.array(v, float)
            rho_s, _ = rank_pairs(arr[:, 0], arr[:, 1])
            per[s] = float(rho_s)
            rows.append(arr)
        if not rows:
            print(f"  {H:>4}h  樣本不足"); continue
        A = np.vstack(rows)
        rho, prod = rank_pairs(A[:, 0], A[:, 1])
        lo, hi = dblock(prod, A[:, 2])
        same = sum(1 for r in per.values() if np.sign(r) == np.sign(rho))
        p1 = (lo * hi > 0) and abs(rho) >= RHO_FLOOR
        p2 = same >= 8
        res["horizons"][f"{H}h"] = {"n": int(len(A)), "rho": rho, "ci": [lo, hi],
                                    "same_sign": same, "coins": len(per),
                                    "per_coin": per, "P1": bool(p1), "P2": bool(p2)}
        print(f"  {H:>4}h{len(A):>8}{rho:>+11.4f}  [{lo:+7.4f},{hi:+7.4f}]{same:>6}/{len(per):<3}"
              f"  P1 {'過' if p1 else '不過'}  P2 {'過' if p2 else '不過'}")

    h1 = res["horizons"].get("1h", {})
    p3 = abs(h1.get("rho", 99) - E3_REF) <= 0.02 if h1 else False
    res["P3_control_ok"] = bool(p3)
    print(f"\n  P3 對照（1h 應重現 er_regime_split 的 {E3_REF:+.3f}，容差 ±0.02）："
          f"實測 {h1.get('rho', float('nan')):+.4f} → {'重現，儀器可信' if p3 else '**重現不了，上面的數字一律不採信**'}")

    passed = [k for k, v in res["horizons"].items() if v["P1"] and v["P2"]]
    if not p3:
        verdict = "儀器未通過對照，不下結論"
    elif passed:
        verdict = f"左右軸畫得出來：{', '.join(passed)}"
    else:
        verdict = "路徑形態在所有測試視野上都不可預測 → 網格只能做上半排（依 σ 調格距與部位）"
    res["verdict"] = verdict
    print(f"  ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
