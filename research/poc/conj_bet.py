# -*- coding: utf-8 -*-
"""交會線的「下注面」數字 —— CI 之外還要報 P(優勢>0) 與回落分布

CLAUDE.md 核心原則 #10：「CI 不跨零」是宣稱發現的門檻，不是下注的門檻。
研究端問「這是不是我挑出來的」，資金端問「期望值為正嗎、活得下來嗎」。
兩者用不同的量，而這支腳本只產後者——**它不判決任何東西**，判決仍歸
`conj_wf.py`（樣本外）與 `conj_clock*.py`（前瞻）。

報三組，全部**樣本外優先**（原則 #9）：
    P(優勢 > 0)     同一組日聚類 bootstrap 直接數，不是從 CI 反推
    回落分布        重抽交易順序，p50 / p90 / p95 與 P(回落>50%)
    Sharpe / PF     逐筆淨值的年化夏普與獲利因子

母體 = S+D+V（現行規格），切半：前半樣本內、後半樣本外。
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
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import conj_backtest as cb  # noqa: E402

OUT = HERE / "data" / "results"
RNG = np.random.default_rng(20260909)
B = 4000
SLOTS = 3          # 與 conj_ab_sim 同一格
LEV = 2.0


def collect():
    rows = []
    for sym in cb.CORE9:
        tr, _ = cb.ledger(sym, arm="A")
        for t in tr:
            if t["sigk"] != "and":
                continue
            T = pd.Timestamp(t["anchor_ts"], unit="ms", tz="UTC")
            rows.append(dict(sym=sym, ts=t["anchor_ts"],
                             day=T.strftime("%Y-%m-%d"),
                             R=t["R"], Rn=t["R_net"],
                             atrp=t["atr"] / t["entry"]))
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def day_boot(x, days, b=B):
    """日聚類 bootstrap，回傳重抽均值的整條分布（不是只回 CI）。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    out = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        out[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return out


def dd_paths(d, b=1500):
    """回落分布：重抽**交易日順序**（保留同日成組），逐筆複利走一遍。

    名目 = LEV x 權益 / SLOTS，每筆報酬 = R_net x ATR% x 名目倍數。
    這是 conj_ab_sim 那個模擬的分布版：一條路徑不夠，要看它的散布。
    """
    uq = d.day.unique()
    grp = {k: g for k, g in d.groupby("day")}
    mdds = np.empty(b)
    for i in range(b):
        order = RNG.permutation(uq)
        eq, peak, mdd = 1.0, 1.0, 0.0
        for day in order:
            g = grp[day]
            for r, a in zip(g.Rn.to_numpy(), g.atrp.to_numpy()):
                eq *= 1.0 + (LEV / SLOTS) * r * a
                if eq <= 0:
                    eq = 1e-9
                peak = max(peak, eq)
                mdd = max(mdd, 1.0 - eq / peak)
        mdds[i] = mdd
    return mdds


def block(lab, d):
    x = d.Rn.to_numpy()
    bs = day_boot(x, d.day.to_numpy())
    p_pos = float((bs > 0).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    win = x > 0
    pf = (x[win].sum() / -x[~win].sum()) if (~win).any() and x[~win].sum() < 0 else float("nan")
    # 年化夏普：以實際跨越天數換算每年筆數
    days_span = max(1, (pd.to_datetime(d.day).max() - pd.to_datetime(d.day).min()).days)
    per_yr = len(d) / days_span * 365.0
    sharpe = float(x.mean() / x.std(ddof=1) * np.sqrt(per_yr / SLOTS)) if x.std() > 0 else float("nan")
    mdd = dd_paths(d)
    per = d.groupby("sym").Rn.mean()
    print(f"\n=== {lab}（n={len(d):,}，{days_span} 天）===")
    print(f"  每筆淨值      {x.mean():+.4f} ATR   CI [{lo:+.4f}, {hi:+.4f}]"
          f"   逐幣 {int((per>0).sum())}/9")
    print(f"  **P(優勢>0)   {p_pos*100:5.1f}%**   <- 下注面看這個，不是 CI")
    print(f"  勝率          {win.mean()*100:5.1f}%    獲利因子 {pf:.3f}"
          f"    年化夏普 {sharpe:+.2f}")
    print(f"  回落（{LEV:.0f}x／{SLOTS} 槽） p50 {np.percentile(mdd,50)*100:4.1f}%"
          f"  p90 {np.percentile(mdd,90)*100:4.1f}%"
          f"  p95 {np.percentile(mdd,95)*100:4.1f}%")
    print(f"                 P(回落>50%) {(mdd>0.5).mean()*100:4.1f}%"
          f"   P(回落>70%) {(mdd>0.7).mean()*100:4.1f}%")
    return dict(n=len(d), mean=float(x.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_edge_pos=p_pos, win_rate=float(win.mean()), pf=float(pf),
                sharpe=sharpe, coins_pos=int((per > 0).sum()),
                mdd_p50=float(np.percentile(mdd, 50)),
                mdd_p90=float(np.percentile(mdd, 90)),
                mdd_p95=float(np.percentile(mdd, 95)),
                p_mdd_over_50=float((mdd > 0.5).mean()),
                p_mdd_over_70=float((mdd > 0.7).mean()))


def main():
    d = collect()
    mid = d.ts.min() + (d.ts.max() - d.ts.min()) // 2
    res = {}
    # 樣本外先報（原則 #9）；樣本內只當對照
    res["oos"] = block("樣本外（後半）", d[d.ts > mid])
    res["is"] = block("（樣本內，後半的對照）", d[d.ts <= mid])
    res["all"] = block("全期（含樣本內，僅供對照）", d)
    r = res["oos"]["mean"] / res["is"]["mean"] if res["is"]["mean"] else float("nan")
    print(f"\n樣本外／樣本內 = {r*100:.0f}%   <- 過擬合程度的直接讀數")
    res["oos_over_is"] = float(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_bet.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwritten -> {OUT / 'conj_bet.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
