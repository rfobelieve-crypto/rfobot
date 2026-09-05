# -*- coding: utf-8 -*-
"""網格二維控制矩陣的前置檢查：擴張訊號落在哪一半排（預註冊，2026-09-05）

**要回答的問題**：網格是 short gamma，損益只吃 σ 沒有方向項——但 σ 有兩個相反的
作用。高 σ 且價格在區間內來回 ⇒ 穿越次數暴增（天堂）；高 σ 且單向穿透 ⇒ 庫存
單邊累積（地獄）。**單看 σ 會把這兩者混為一談**，而這正好解釋 §1.14b 為什麼
NO-GO：那個閘門只有上半排、沒有左右，把天堂和地獄一起擋掉了（震盪期≈不加閘門、
只在空頭期 +0.95%）。

路徑形態的便宜儀器是 Efficiency Ratio：`ER = |P_t − P_{t−n}| / Σ|ΔP_i|`。
低 ER＝路徑長、位移小＝震盪。**但這條線上已經有一個現任的路徑形態儀：ADX**
（§0.49，震盪 meanR +0.075 vs 趨勢 +0.016，CI 離零、8/9 幣），而 trend_z 就是
因為對 ADX 沒有增量而退役的。**所以 ER 必須證明它對 ADX 有增量，否則只是換名字。**

**訊號定義逐字沿用 §1.14b 凍結版**（不重寫，避免第二份實作）：
  shock   每分鐘撤單量 / rolling-60 中位（`cancel_playbook_watcher` 原式）
  shock_h 該小時 60 個分鐘 shock 的平均
  訊號    shock_h(i−1) ≥ shock_h 的 trailing 168 小時第 80 百分位（因果）

**判準（跑之前寫死）**
  E1 分離    訊號小時 vs 非訊號小時的 forward ER 均值差 |Δ| ≥ 0.03 ∧ 日區塊
             bootstrap CI 離零 ∧ ≥8/10 幣同號
  E2 增量    在**每一個 ADX 狀態內**，E1 的差仍然 CI 離零——否則 ER 的分離只是
             ADX 的翻版，二維矩陣退化成一維
  E3 現成解  trailing ER 對 forward ER 的 rank 相關，CI 離零；**若 E3 的量級
             壓過訊號的分離，正確結論是「直接用 ER，不需要流訊號來分左右」**
  全格報告   2×2 矩陣（訊號高低 × ER 高低）每格的 n 與 forward 波動，不挑格
  功效       10 幣 × ~44 天 × 24 小時 ≈ 10,000 小時，ER 的 sd ~0.15
             → 差的 SE ≈ 0.004。**這是少見的功效不成問題的測試。**

**先驗**：擴張訊號預測的是 |move| 的大小，不是形狀，所以 **E1 先驗是不分離或
只有微弱正向**（撤單先於方向性推動 ⇒ 可能略偏趨勢）。E3 先驗是強（ER 有持續性）。
若 E1 不過而 E3 過，結論是「二維矩陣成立，但左右軸要用 ER 自己量，流訊號只管上下」。

Run: python research/exit_paths/er_regime_split.py
Out: research/results/er_regime_split.json
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
sys.path.insert(0, str(ROOT / "research" / "lp_ladder"))
from flow_grid_gate import load  # noqa: E402
from grid_adx_gate import adx_labels  # noqa: E402

OUT = ROOT / "research" / "results" / "er_regime_split.json"
SYMS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
        "ADA-USD", "LINK-USD", "AVAX-USD", "SUI-USD", "UNI-USD", "BNB-USD", "AAVE-USD"]
EX = "binance"
P80_WIN = 168          # 小時
FWD = (30, 60)         # forward ER 的視窗（分鐘）


def er(p):
    """Efficiency Ratio：位移 / 路徑長。低 = 震盪。"""
    if len(p) < 3:
        return np.nan
    path = np.abs(np.diff(p)).sum()
    return abs(p[-1] - p[0]) / path if path > 0 else np.nan


def build(sym):
    x = load(sym, EX)
    if x is None or len(x) < 5000:
        return None
    tot = x["bc"].astype(float) + x["ac"].astype(float)
    base = tot.rolling(60, min_periods=30).median()
    x["shock"] = tot / base.replace(0, np.nan)          # 凍結原式
    x["hr"] = (x["m"] // 3_600_000).astype(np.int64)
    mid = x["mid_price"].astype(float).values
    hrs = sorted(x["hr"].unique())
    g = x.groupby("hr")
    sh = g["shock"].mean()
    idx = {h: i for i, h in enumerate(hrs)}
    rows = []
    arr_hr = x["hr"].values
    for h in hrs:
        i = idx[h]
        if i < P80_WIN + 1 or i >= len(hrs) - 2:
            continue
        prev = sh.iloc[i - 1]
        thr = sh.iloc[max(0, i - 1 - P80_WIN):i - 1].quantile(0.80)
        if not np.isfinite(prev) or not np.isfinite(thr):
            continue
        m0 = np.searchsorted(arr_hr, h)
        m1 = np.searchsorted(arr_hr, h, side="right")
        if m1 - m0 < 30:
            continue
        row = {"hr": h, "day": h // 24, "sig": int(prev >= thr),
               "trail_er": er(mid[max(0, m0 - 60):m0])}
        for w in FWD:
            row[f"er{w}"] = er(mid[m0:m0 + w])
            seg = mid[m0:m0 + w]
            row[f"vol{w}"] = float(np.std(np.diff(np.log(seg))) * 1e4) if len(seg) > 3 else np.nan
        rows.append(row)
    d = pd.DataFrame(rows).dropna(subset=["er60", "trail_er"])
    if len(d) < 500:
        return None
    # ADX（現任儀器）用小時 OHLC
    o = x.groupby("hr")["mid_price"].agg(["max", "min", "last"])
    st, _ = adx_labels(o["max"].astype(float).values, o["min"].astype(float).values,
                       o["last"].astype(float).values)
    smap = {h: st[i] for i, h in enumerate(hrs) if i < len(st)}
    d["adx"] = d["hr"].map(smap).fillna("")
    return d


def dblock(v, days, B=2000, seed=9):
    rng = np.random.default_rng(seed); g = {}
    for x, dd in zip(v, days):
        if np.isfinite(x):
            g.setdefault(dd, []).append(x)
    ks = np.array(list(g))
    if len(ks) < 5:
        return float("nan"), float("nan")
    out = [np.concatenate([g[dd] for dd in rng.choice(ks, len(ks))]).mean() for _ in range(B)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 100)
    print("  網格二維矩陣前置：擴張訊號 vs 路徑形態（ER），對照現任儀器 ADX")
    print("=" * 100)
    frames, per = {}, {}
    for s in SYMS:
        try:
            d = build(s)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: {str(e)[:60]}"); continue
        if d is None:
            continue
        frames[s] = d
        a, b = d.loc[d.sig == 1, "er60"], d.loc[d.sig == 0, "er60"]
        per[s] = {"n": len(d), "n_sig": int(d.sig.sum()),
                  "er_sig": float(a.mean()), "er_non": float(b.mean()),
                  "d": float(a.mean() - b.mean())}
        print(f"  {s:<10} n={len(d):<6} 訊號 {d.sig.sum():<5} "
              f"ER60 訊號 {a.mean():.3f} / 非訊號 {b.mean():.3f}  差 {a.mean()-b.mean():+.4f}")
    if not frames:
        print("  無資料"); return 0
    D = pd.concat(frames.values(), keys=frames.keys(), names=["sym"]).reset_index(level=0)
    res = {"per_coin": per, "n": int(len(D))}

    print(f"\n  ── E1 分離（合池 n={len(D)}）──")
    for w in FWD:
        a = D.loc[D.sig == 1, f"er{w}"]; b = D.loc[D.sig == 0, f"er{w}"]
        diff = a.mean() - b.mean()
        paired = np.concatenate([a.values - b.mean(), np.full(0, np.nan)])
        lo, hi = dblock(a.values - b.mean(), D.loc[D.sig == 1, "day"].values)
        res[f"e1_{w}"] = {"sig": float(a.mean()), "non": float(b.mean()), "diff": float(diff),
                          "ci": [lo, hi]}
        print(f"     ER{w}: 訊號 {a.mean():.4f}  非訊號 {b.mean():.4f}  差 {diff:+.4f}  CI [{lo:+.4f},{hi:+.4f}]")
    same = sum(1 for v in per.values() if np.sign(v["d"]) == np.sign(res["e1_60"]["diff"]))
    e1 = abs(res["e1_60"]["diff"]) >= 0.03 and res["e1_60"]["ci"][0] * res["e1_60"]["ci"][1] > 0 and same >= 8
    print(f"     逐幣同號 {same}/{len(per)}   E1（|差|≥0.03 ∧ CI 離零 ∧ ≥8 幣）：{'過' if e1 else '不過'}")

    print("\n  ── E2 對 ADX 的增量（現任儀器）──")
    e2 = True
    for st in ("RANGING", "TRENDING", "NEUTRAL"):
        sub = D[D.adx == st]
        if len(sub) < 300:
            print(f"     {st:<9} n={len(sub)} 太少，略過"); continue
        a = sub.loc[sub.sig == 1, "er60"]; b = sub.loc[sub.sig == 0, "er60"]
        if len(a) < 50 or len(b) < 50:
            continue
        lo, hi = dblock(a.values - b.mean(), sub.loc[sub.sig == 1, "day"].values)
        ok = lo * hi > 0
        e2 &= ok
        print(f"     {st:<9} n={len(sub):<6} ER60 訊號 {a.mean():.4f} / 非 {b.mean():.4f}"
              f"  差 {a.mean()-b.mean():+.4f}  CI [{lo:+.4f},{hi:+.4f}]  {'離零' if ok else '含零'}")
        res[f"e2_{st}"] = {"n": int(len(sub)), "diff": float(a.mean() - b.mean()), "ci": [lo, hi]}
    print(f"     ADX 本身：" + "  ".join(
        f"{st} ER60 {D.loc[D.adx==st,'er60'].mean():.4f} (n={int((D.adx==st).sum())})"
        for st in ("RANGING", "TRENDING", "NEUTRAL") if (D.adx == st).sum() > 0))

    print("\n  ── E3 trailing ER 的持續性（現成解）──")
    m = np.isfinite(D["trail_er"]) & np.isfinite(D["er60"])
    rx = pd.Series(D.loc[m, "trail_er"]).rank().values
    ry = pd.Series(D.loc[m, "er60"]).rank().values
    rho = float(np.corrcoef(rx, ry)[0, 1])
    prod = (rx - rx.mean()) * (ry - ry.mean()) / (rx.std() * ry.std())
    lo, hi = dblock(prod, D.loc[m, "day"].values)
    e3 = lo * hi > 0
    print(f"     rank 相關 {rho:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  {'離零' if e3 else '含零'}")
    res["e3"] = {"rho": rho, "ci": [lo, hi]}

    print("\n  ── 2×2 全格（訊號 × trailing ER 高低，門檻 = ER 中位）──")
    med = D["trail_er"].median()
    print(f"     {'':<14}{'低 ER（震盪）':>16}{'高 ER（單向）':>16}")
    cells = {}
    for si, sl in ((1, "訊號"), (0, "無訊號")):
        line = f"     {sl:<14}"
        for lowhi, lbl in ((True, "低"), (False, "高")):
            sub = D[(D.sig == si) & ((D.trail_er <= med) if lowhi else (D.trail_er > med))]
            cells[f"{sl}_{lbl}"] = {"n": int(len(sub)), "er60": float(sub["er60"].mean()),
                                    "vol60": float(sub["vol60"].mean())}
            line += f"  n={len(sub):<5} ER {sub['er60'].mean():.3f} vol {sub['vol60'].mean():.1f}"
        print(line)
    res["cells"] = cells
    verdict = ("二維矩陣成立且流訊號分得出左右" if (e1 and e2) else
               "二維矩陣成立，但左右軸要用 ER 自己量，流訊號只管上下" if e3 else
               "兩軸都分不出來")
    res.update({"E1": bool(e1), "E2": bool(e2), "E3": bool(e3), "verdict": verdict})
    print(f"\n  E1 {'過' if e1 else '不過'}  E2 {'過' if e2 else '不過'}  E3 {'過' if e3 else '不過'}"
          f"   ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
