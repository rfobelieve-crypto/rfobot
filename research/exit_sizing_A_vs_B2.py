# -*- coding: utf-8 -*-
"""A（trailing 樂透）vs B2（幅度停利工資）在同一個 MDD 預算下的可部署規模（預註冊 2026-09-05）

**問題**：§0.88f 顯示 B2 均值 −13 bps 但 WR 36→65%、中位 −42→+51。使用者選了
穩定形狀。要驗的是：**在 2x 槓桿＋ −20%/−30% kill switch 的世界裡，穩定形狀
能不能靠開大部位把均值差補回來**——這是 sizing 問題，不是出場問題。

**設計（寫死）**
  樣本   同 §0.88f 的 124 筆低頻 Strong，每筆 A 與 B2 的每名目淨報酬（bps）
  序列   (i) 配對序列：兩臂用同一批交易依時間順序複利（忽略單倉重疊，兩臂同樣忽略）
         (ii) 單倉序列：各臂依自己的持倉長度執行單倉制——上一筆沒平就跳過這筆
              （B2 平得快能多吃幾筆，這是真實的槽位效應；兩臂交易集不同，不配對）
  權益   每筆 equity × (1 + L × r)，L = 有效槓桿（名目／權益），與 NOTIONAL_LEV_MULT 同義
  預算   MDD 預算兩檔：**−15%**（Stage 3→4a 門檻精神）與 **−30%**（總損 DEMOTE）
  穩健   交易區塊 bootstrap（區塊 10 筆，2000 次）；**用 bootstrap 的 MDD p95 當預算
         約束**，不用單一實現路徑（實現路徑只是一次抽樣）
  L*     每臂在每檔預算下的最大 L，使 MDD p95 ≤ 預算；L 掃 0.25 步，上限 6
  判準   **B2 在 L*(B2) 的終值中位 > A 在 L*(A) 的終值中位，且差的 bootstrap 95% CI
         不含零，且兩檔預算皆成立** → 「sizing 翻轉」成立；否則 NO-GO。
         另報：每臂每筆 mean/sd、Kelly f*、在現行 L=2 下的 MDD p95 與終值。
  預測   B2 的 L* 明顯大於 A（左尾雖同為 −410 但頻率低得多）；能不能翻轉——五五開。
         單倉序列裡 B2 會多吃到交易，這一項會偏向 B2，另列不合併。

Run: python research/exit_sizing_A_vs_B2.py
Out: research/results/exit_sizing_A_vs_B2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from exit_giveback_family import bars, db, trail_leg, COST, K_MAG  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "exit_sizing_A_vs_B2.json"
BUDGETS = (0.15, 0.30)
LGRID = np.arange(0.25, 6.01, 0.25)
B, BLK = 2000, 10
CAP_H = 72


def trades():
    df = bars(); s, h = db()
    up_t = set(s[s.side > 0]["t"]); dn_t = set(s[s.side < 0]["t"])
    rows = []
    for _, r in s.iterrows():
        t_e = r["t"] + pd.Timedelta(hours=1)
        if t_e not in df.index or r["t"] not in h.index:
            continue
        i0 = df.index.get_loc(t_e)
        if i0 < 20 or i0 + CAP_H >= len(df):
            continue
        side = r["side"]; entry = df["open"].iloc[i0]
        opp = {x + pd.Timedelta(hours=1) for x in (dn_t if side > 0 else up_t)}
        a, ka, _ = trail_leg(df, i0, side, opp)
        tp = entry * (1 + side * K_MAG * float(h.loc[r["t"], "m"]))
        b, kb, _ = trail_leg(df, i0, side, opp, tp=tp)
        rows.append({"i0": i0, "kA": ka, "kB": kb, "A": (a - COST) / 1e4, "B2": (b - COST) / 1e4})
    return pd.DataFrame(rows)


def slot_seq(d, k_col, r_col):
    out, busy_until = [], -1
    for _, r in d.iterrows():
        if r["i0"] <= busy_until:
            continue
        out.append(r[r_col]); busy_until = r[k_col]
    return np.array(out)


def path_stats(r, L):
    eq = np.cumprod(np.maximum(1 + L * r, 0.0))
    pk = np.maximum.accumulate(eq)
    return eq[-1], float(((eq - pk) / pk).min())


def boot(r, L, rng):
    n = len(r); k = max(1, n // BLK)
    fin, mdd = np.empty(B), np.empty(B)
    for i in range(B):
        st = rng.integers(0, max(1, n - BLK), k)
        rr = np.concatenate([r[j:j + BLK] for j in st])[:n]
        fin[i], mdd[i] = path_stats(rr, L)
    return fin, mdd


def lstar(r, budget, rng):
    best = None
    for L in LGRID:
        fin, mdd = boot(r, L, rng)
        if np.percentile(mdd, 5) >= -budget:      # p95 of drawdown depth
            best = (float(L), fin, mdd)
        else:
            break
    return best


def main():
    d = trades(); n = len(d)
    rng = np.random.default_rng(20260905)
    print("=" * 96)
    print(f"  A trailing vs B2 幅度停利 — 同 MDD 預算下的可部署槓桿  (n={n} 配對交易)")
    print("=" * 96)
    res = {"n": n, "series": {}, "budgets": {}}
    for lab, r in (("A", d["A"].values), ("B2", d["B2"].values)):
        m, sd = r.mean(), r.std(ddof=1); kelly = m / sd**2 if sd else 0
        fin2, mdd2 = boot(r, 2.0, rng)
        res["series"][lab] = {"mean_bps": m * 1e4, "sd_bps": sd * 1e4, "kelly": kelly,
                              "L2_final_med": float(np.median(fin2)), "L2_mdd_p95": float(np.percentile(mdd2, 5))}
        print(f"  {lab:3} 每筆 mean {m*1e4:+6.1f} sd {sd*1e4:5.0f} bps  Kelly f*={kelly:4.1f}x   "
              f"L=2 現行: 終值中位 {np.median(fin2):.3f}  MDD p95 {np.percentile(mdd2,5):+.1%}")
    print()
    print(f"  {'預算':>6} {'臂':>3} {'L*':>5} {'終值中位':>9} {'終值 p5':>8} {'MDD p95':>8} {'年化≈':>7}")
    yrs = 5.0 / 12.0   # 04-03 → 09-04 ≈ 5 months
    flips = []
    for bud in BUDGETS:
        res["budgets"][str(bud)] = {}
        got = {}
        for lab, r in (("A", d["A"].values), ("B2", d["B2"].values)):
            L, fin, mdd = lstar(r, bud, rng) or (0.0, np.ones(B), np.zeros(B))
            got[lab] = (L, fin, mdd)
            res["budgets"][str(bud)][lab] = {"L": L, "final_med": float(np.median(fin)),
                                              "final_p5": float(np.percentile(fin, 5)),
                                              "mdd_p95": float(np.percentile(mdd, 5))}
            print(f"  {-bud:>6.0%} {lab:>3} {L:>5.2f} {np.median(fin):>9.3f} {np.percentile(fin,5):>8.3f} "
                  f"{np.percentile(mdd,5):>+8.1%} {np.median(fin)**(1/yrs)-1:>+7.0%}")
        diff = got["B2"][1] - got["A"][1]          # same bootstrap seed stream? no -- independent; use quantiles
        lo, hi = np.percentile(diff, [2.5, 97.5])
        ok = np.median(got["B2"][1]) > np.median(got["A"][1]) and lo > 0
        flips.append(ok)
        res["budgets"][str(bud)]["B2_minus_A_final"] = [float(diff.mean()), float(lo), float(hi)]
        print(f"         B2 − A 終值差 {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  → {'B2 贏' if ok else '未翻轉'}")
    # slot-enforced (informational)
    print("\n  單倉制序列（各臂自己的持倉長度；不配對，B2 平得快會多吃交易）:")
    for lab, kc in (("A", "kA"), ("B2", "kB")):
        r = slot_seq(d, kc, lab)
        fin, mdd = boot(r, 2.0, rng)
        print(f"    {lab:3} 交易數 {len(r):3d}  L=2 終值中位 {np.median(fin):.3f}  MDD p95 {np.percentile(mdd,5):+.1%}")
        res["series"][lab]["slot_n"] = int(len(r)); res["series"][lab]["slot_L2_final_med"] = float(np.median(fin))
    verdict = "GO（sizing 翻轉成立）" if all(flips) else "NO-GO（sizing 沒有翻轉均值差）"
    res["verdict"] = verdict
    print(f"\n  ==> {verdict}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
