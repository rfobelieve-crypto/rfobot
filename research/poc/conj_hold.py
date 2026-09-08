# -*- coding: utf-8 -*-
"""持有時間 × 停損的網格 —— 「後面浪費很多波段」是不是真的

使用者 2026-09-09：「我看歷史回測的圖……後面都浪費很多波段，你有試過
15 分鐘級別嗎」。

為什麼這個問題可能是關鍵，而不只是調參
    `conj_rescue.py` 的結論是「差一點點」：delay=3 毛 +0.1287、成本 0.12,
    淨 −0.0186。**成本是每筆固定的，優勢卻可能隨持有時間長大**——
    現行 HOLD=60 分是凍結定義裡拍板的，從來沒有在誠實錨點下掃過。
    若持有 4 小時的毛利是 0.30 ATR 而成本仍是 0.12，那就從負轉正。
    這不是「調參把它調到過」，這是**成本結構的比值問題**。

    TODO §1.03 掃過一次持有期曲線，但那是在**前視錨點**下跑的（§1.03b），
    整批作廢，必須重跑。

===========================================================================
判準（跑之前寫死）
===========================================================================
    網格   持有 ∈ {30, 60, 120, 240, 480, 960} 分
           停損 ∈ {1.0, 1.5, 2.0, 3.0, 無}
           進場固定 = 事件成立 + 3 分（誠實錨點下毛利最好的那格，
                     但**不是**為了本檔挑的——它在 conj_redef 就定了）
           成本    分腿（進場 7 / 時間出場 3 / 停損出場 10 bps）

    G1  某格淨值日聚類 CI 下緣 > 0 **且**逐幣 ≥6/9
    G2  **多重比較**：6 × 5 = 30 格，5% 之下期望約 1.5 格靠運氣過。
        所以單一格過關**不算數**，必須**它的鄰格也同向**
        （持有的上下一格、停損的上下一格，至少 3/4 個鄰格淨值為正）。
        這條是為了擋「掃出一格最好看的」——threshold-sweep 陷阱
        （mistake.md 2026-06-20）。
    G3  全格報告，不挑格。
    G4  **已知答案對照**：持有 60 / 停損 1.0 那一格必須重現 conj_redef
        delay=3 的毛 +0.1287（容差 0.005）。對不上代表機器寫錯。
    G5  任何過關的格子**只代表值得開一條自己的前瞻時鐘從零累積**，
        不得復活現行註冊（§0.92）。並且要報「毛利隨持有時間的形狀」——
        單調上升才支持「行情還在走」，先升後降代表有最佳點（更可疑）。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import event_census as ec  # noqa: E402
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W5 = 5
DELAY = 3
HOLDS = (30, 60, 120, 240, 480, 960)
STOPS = (1.0, 1.5, 2.0, 3.0, None)
FLOW = ("delta_ext", "vol_burst")
LEGS = (7.0, 3.0, 10.0)
REF_G4 = 0.1287
TOL = 0.005
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=1500):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (float("nan"),) * 3
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    rows = []
    for sym in ec.CORE9:
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "open", "high", "low", "close", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        op = b["open"].to_numpy(float)
        hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        cl = b["close"].to_numpy(float)
        at = b["atr_h14"].to_numpy(float)
        n = len(ts)
        dstr = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")

        cand, _t, _c, _a, _d = cr.ck.frozen_cand(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        pairs = [(int(m), "sweep")
                 for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in FLOW:
            v = cand.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pairs.append((int(m), nm))

        for a, mem in cr.groups_with_members(pairs):
            sig = {t for _, t in mem}
            if "sweep" not in sig or not (sig & set(FLOW)):
                continue
            m_sw = min(m for m, t in mem if t == "sweep")
            m_fl = min(m for m, t in mem if t in FLOW)
            rd = max(m_sw, m_fl)
            if rd < W5 or rd + DELAY + max(HOLDS) >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
            j0 = rd + DELAY
            ent = float(op[j0])
            row = dict(sym=sym, day=dstr[rd], entry=ent, atr=A)
            mx = max(HOLDS)
            adv = ((ent - lo[j0 + 1:j0 + mx + 1]) if d > 0
                   else (hi[j0 + 1:j0 + mx + 1] - ent)) / A
            for H in HOLDS:
                seg = adv[:H]
                for S in STOPS:
                    key = f"h{H}s{'x' if S is None else S}"
                    if S is None:
                        row[key] = float(d * (cl[j0 + H] - ent) / A)
                        row[key + "_st"] = False
                    else:
                        k = np.flatnonzero(seg >= S)
                        if len(k):
                            row[key] = -S
                            row[key + "_st"] = True
                        else:
                            row[key] = float(d * (cl[j0 + H] - ent) / A)
                            row[key + "_st"] = False
            rows.append(row)

    d = pd.DataFrame(rows)
    e = d.entry.to_numpy()
    A = d.atr.to_numpy()
    days = d.day.to_numpy()
    print("=== 持有 × 停損網格（誠實錨點，進場 = 成立 +3 分）===")
    print(f"母體 {len(d):,} 筆、{d.day.nunique()} 日、9 幣")
    print()

    g4 = float(np.mean(d["h60s1.0"]))
    ok4 = abs(g4 - REF_G4) < TOL
    print("=== G4 已知答案對照（持有 60／停損 1.0）===")
    print(f"  毛 {g4:+.4f}（conj_redef delay=3 {REF_G4:+.4f}）-> "
          + ("PASS" if ok4 else "**FAIL —— 機器寫錯，以下不解讀**"))
    if not ok4:
        return 1
    print()

    res = {"n": int(len(d)), "cells": {}}
    grid_net = {}
    print("=== G3 全格報告：每格「毛 / 淨 / 淨CI下 / 幣+」，不挑格 ===")
    hdr = "持有".rjust(6) + "".join(
        f"{('停損 ' + ('無' if S is None else str(S))):>26s}" for S in STOPS)
    print(hdr)
    for H in HOLDS:
        line = f"{H:5d}m"
        for S in STOPS:
            key = f"h{H}s{'x' if S is None else S}"
            g = d[key].to_numpy(float)
            st = d[key + "_st"].to_numpy(bool)
            nt = g - (LEGS[0] + np.where(st, LEGS[2], LEGS[1])) / 1e4 * e / A
            mg = float(np.mean(g))
            mn, ln, _ = day_ci(nt, days)
            per = d.assign(x=nt).groupby("sym").x.mean()
            pos = int((per > 0).sum())
            grid_net[(H, S)] = (mn, ln, pos)
            res["cells"][key] = dict(gross=mg, net=mn, net_lo=ln, coins_pos=pos,
                                     stop_rate=float(st.mean()))
            mark = "*" if (ln > 0 and pos >= 6) else " "
            line += f"{mg:+7.3f}/{mn:+6.3f}/{ln:+6.3f}/{pos}{mark}"
        print(line)
    print("  （* = 該格淨值 CI 下緣 > 0 且逐幣 ≥6/9）")
    print()

    print("=== G5 毛利隨持有時間的形狀（停損 1.0，看行情還在不在走）===")
    for H in HOLDS:
        g = float(np.mean(d[f"h{H}s1.0"]))
        print(f"  {H:4d} 分  毛 {g:+.4f}   每分鐘 {g/H*60:+.4f}/小時")
    print()

    print("=== G1/G2 判定（單格過關不算數，要鄰格同向）===")
    winners = [(H, S) for (H, S), (mn, ln, pos) in grid_net.items()
               if ln > 0 and pos >= 6]
    if not winners:
        print("  **沒有任何格子過 G1** -> 拉長持有救不回來。")
    for H, S in winners:
        hi_i = HOLDS.index(H)
        si = STOPS.index(S)
        nb = []
        for di in (-1, 1):
            if 0 <= hi_i + di < len(HOLDS):
                nb.append(grid_net[(HOLDS[hi_i + di], S)][0])
            if 0 <= si + di < len(STOPS):
                nb.append(grid_net[(H, STOPS[si + di])][0])
        same = sum(1 for x in nb if x > 0)
        verdict = "**過 G2（鄰格同向）**" if same >= 3 else f"G2 未過（鄰格只有 {same}/{len(nb)} 為正）"
        print(f"  持有 {H}m／停損 {S}：淨 {grid_net[(H,S)][0]:+.4f} "
              f"CI下 {grid_net[(H,S)][1]:+.4f} 幣 {grid_net[(H,S)][2]}/9 -> {verdict}")
    res["winners"] = [[h, s] for h, s in winners]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_hold.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print()
    print("written ->", OUT / "conj_hold.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
