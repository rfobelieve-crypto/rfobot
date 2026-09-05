# -*- coding: utf-8 -*-
"""B2 幅度停利 ＋ 只吃串內第一根 —— 單倉制配對驗證（預註冊 2026-09-05）

**為什麼**：§0.88g 單倉制下 B2 輸 A（1.087 vs 1.336），原因具體：B2 平得快、
槽位釋放快，多吃 23 筆，而多吃到的是 §0.80 量過的「串內第 2 根以後」（WR 40%
vs 第一根 63.5%）。§0.80 是獨立於出場的發現；把它接到 B2 上是**兩個獨立發現的
互補**，不是新參數。

**串的定義（逐字沿用 §0.80）**：同向 Strong 且與前一筆同向 Strong 相隔 ≤ 6h
視為同串；「第一根」＝ 過去 6h 內沒有同向 Strong 的那筆。

**四臂（全部單倉制、L=2；上一筆沒平就跳過）**
  A        trailing，全部訊號（現行）
  A+first  trailing，只吃第一根          ← 對照：濾網對 trailing 也有用嗎？
  B2       幅度停利，全部訊號
  B2+first 幅度停利，只吃第一根          ← 假設

**判準（寫死）**：GO 必須同時
  (1) B2+first 終值中位（bootstrap，L=2）> A **且** > A+first
      —— 要贏「也拿到濾網的 trailing」，否則只是濾網在作用
  (2) B2+first 的 MDD p95 ≤ A 的
  (3) B2+first − A+first 終值差的 bootstrap 95% CI 不含零
  (4) 前後兩半各自 B2+first > A+first（終值）
  任一不過 = NO-GO；(3) 若是唯一不過的，明寫「方向一致但未達功效」。
  另報同預算 L*（−15%／−30%）下四臂的終值，供 sizing 參考，不進判準。
**預測**：(1)(2) 過（濾網補 B2 的漏洞、B2 本來就低離散）；(3) 五五開（終值
bootstrap 很寬）；A+first 也會比 A 好——濾網本身有價值，那是 §0.80 的結論。

Run: python research/exit_b2_firstcluster.py
Out: research/results/exit_b2_firstcluster.json
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
from exit_sizing_A_vs_B2 import boot, lstar, BUDGETS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = ROOT / "research" / "results" / "exit_b2_firstcluster.json"
CLUSTER_H = 6
CAP_H = 72


def build():
    df = bars(); s, h = db()
    s = s.sort_values("t").reset_index(drop=True)
    # first-of-cluster: no same-direction Strong within the prior 6h
    first = []
    for i, r in s.iterrows():
        prev = s[(s.side == r.side) & (s.t < r.t) & (s.t >= r.t - pd.Timedelta(hours=CLUSTER_H))]
        first.append(len(prev) == 0)
    s["first"] = first
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
        rows.append({"t": r["t"], "i0": i0, "first": bool(r["first"]), "kA": ka, "kB": kb,
                     "A": (a - COST) / 1e4, "B2": (b - COST) / 1e4})
    return pd.DataFrame(rows)


def slot(d, kcol, rcol, only_first):
    out, busy = [], -1
    for _, r in d.iterrows():
        if only_first and not r["first"]:
            continue
        if r["i0"] <= busy:
            continue
        out.append(r[rcol]); busy = r[kcol]
    return np.array(out)


def main():
    d = build(); rng = np.random.default_rng(20260905)
    nf = int(d["first"].sum())
    print("=" * 96)
    print(f"  B2 ＋ 只吃串內第一根 · 單倉制 · 全部訊號 {len(d)} 筆，第一根 {nf} 筆（≤{CLUSTER_H}h 同向為同串）")
    print("=" * 96)
    ARMS = [("A", "kA", "A", False), ("A+first", "kA", "A", True),
            ("B2", "kB", "B2", False), ("B2+first", "kB", "B2", True)]
    R, fins = {}, {}
    print(f"  {'臂':9}{'交易':>5}{'每筆均值':>9}{'WR':>7}{'L=2 終值中位':>13}{'終值 p5':>9}{'MDD p95':>9}")
    for lab, kc, rc, of in ARMS:
        r = slot(d, kc, rc, of)
        fin, mdd = boot(r, 2.0, rng); fins[lab] = fin
        R[lab] = {"n": int(len(r)), "mean_bps": float(r.mean() * 1e4), "wr": float((r > 0).mean()),
                  "final_med": float(np.median(fin)), "final_p5": float(np.percentile(fin, 5)),
                  "mdd_p95": float(np.percentile(mdd, 5)), "r": r}
        print(f"  {lab:9}{len(r):>5d}{r.mean()*1e4:>+9.1f}{(r>0).mean()*100:>6.1f}%{np.median(fin):>13.3f}"
              f"{np.percentile(fin,5):>9.3f}{np.percentile(mdd,5):>+9.1%}")
    # halves on the hypothesis pair (by trade order)
    def halves(lab):
        r = R[lab]["r"]; h = len(r) // 2
        return float(np.prod(1 + 2 * r[:h])), float(np.prod(1 + 2 * r[h:]))
    hB, hA = halves("B2+first"), halves("A+first")
    diff = fins["B2+first"] - fins["A+first"]; lo, hi = np.percentile(diff, [2.5, 97.5])
    c1 = R["B2+first"]["final_med"] > R["A"]["final_med"] and R["B2+first"]["final_med"] > R["A+first"]["final_med"]
    c2 = R["B2+first"]["mdd_p95"] >= R["A"]["mdd_p95"]
    c3 = lo > 0
    c4 = hB[0] > hA[0] and hB[1] > hA[1]
    print(f"\n  B2+first − A+first 終值差 {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"  兩半終值  B2+first {hB[0]:.3f}/{hB[1]:.3f}   A+first {hA[0]:.3f}/{hA[1]:.3f}")
    print(f"\n  (1) 贏 A 與 A+first: {'過' if c1 else '不過'}  (2) MDD 不比 A 差: {'過' if c2 else '不過'}"
          f"  (3) CI 離零: {'過' if c3 else '不過'}  (4) 兩半同向: {'過' if c4 else '不過'}")
    verdict = "GO" if all((c1, c2, c3, c4)) else ("NO-GO（方向一致但未達功效）" if (c1 and c2 and c4) else "NO-GO")
    print(f"  ==> {verdict}")
    print("\n  同預算 L*（參考，不進判準）:")
    for bud in BUDGETS:
        line = f"    {-bud:>5.0%}  "
        for lab, *_ in ARMS:
            got = lstar(R[lab]["r"], bud, rng)
            L, fin = (got[0], got[1]) if got else (0.0, np.ones(10))
            line += f"{lab} L*={L:.2f} 終值 {np.median(fin):.3f}   "
        print(line)
    for lab in R: R[lab].pop("r")
    OUT.write_text(json.dumps({"arms": R, "diff": [float(diff.mean()), float(lo), float(hi)],
                               "halves": {"B2+first": hB, "A+first": hA}, "c": [bool(x) for x in (c1, c2, c3, c4)],
                               "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
