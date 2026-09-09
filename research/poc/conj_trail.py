# -*- coding: utf-8 -*-
"""「後面都浪費很多波段」—— 讓賺的跑，不要固定時間砍掉

使用者 2026-09-09：「後面都浪費很多波段」。`conj_hold.py` 已驗證毛利隨
持有單調上升（30m +0.117 -> 960m +0.211），也就是**行情確實還在走**，
固定時間出場把它切掉了。但單純拉長持有的代價是變異數（30 格全不過閘）。

移動停損是第三條路：**留住還在走的、砍掉轉向的**——它同時處理「浪費波段」
與「變異數」，因為它把左尾切掉而不是把右尾切掉。

===========================================================================
出場變體（進場固定 = 成立 +3 分開盤；單位 ATR；成本 限價兩腿 2/2/10 bps）
===========================================================================
    fix_H      固定持有 H 分（對照組，H ∈ {60, 240, 480}）
    trail_k    移動停損：從進場起追蹤最有利價，停損 = 極值 ∓ k×ATR
               k ∈ {1.0, 1.5, 2.0, 3.0}；上限持有 960 分（避免無限期）
    trail_k_be 同上，但**先到 1×ATR 獲利才開始移動**（之前用固定 k×ATR 停損）
               —— 這是「先保住本再讓它跑」的常見做法，一併報

判準（跑之前寫死）
    T1  全格報告，不挑格。每格報：淨值 / 日聚類 CI 下緣 / 逐幣 / 平均持有分鐘
        / 停損出場佔比。**平均持有必須報**——移動停損若把持有拉到 10 小時，
        它就繼承 `conj_hold` 的容量問題，不能假裝沒有。
    T2  過閘 = 淨值 CI 下緣 > 0 且逐幣 ≥6/9，且**鄰格（k 的上下一格）同向**。
    T3  **已知答案對照**：fix_60 必須重現 conj_hold 的 h60s1.0 那格
        （毛 +0.1286，容差 0.005）。對不上不解讀。
    T4  看過失敗後的搜尋 -> 過閘只代表值得開自己的前瞻時鐘（§0.92）。
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
W5, DELAY = 5, 3
FIX_H = (60, 240, 480)
TRAIL_K = (1.0, 1.5, 2.0, 3.0)
MAXHOLD = 960
LEGS = (2.0, 2.0, 10.0)
REF_T3 = 0.1286
TOL = 0.005
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=1500):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (float("nan"),) * 2
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5))


def trail_exit(fav, adv, cl_rel, k, be=False):
    """移動停損的逐分鐘模擬（全部以 ATR 為單位、相對進場）。

    fav[i]  = 到第 i 分鐘為止，順勢方向的**最有利**幅度（單調不減）
    adv[i]  = 第 i 分鐘逆勢方向的最大幅度（該分鐘的不利極值）
    cl_rel[i] = 第 i 分鐘收盤相對進場的順勢幅度
    回傳 (R, 分鐘數, 是否被停損)
    """
    n = len(adv)
    for i in range(n):
        # 停損水準（相對進場，順勢為正）：極值往回 k
        if be and fav[i] < 1.0:
            lvl = -k                      # 還沒到 1 ATR：固定停損在 -k
        else:
            lvl = fav[i] - k
        # 這一分鐘的不利極值是否觸及停損水準
        if -adv[i] <= lvl:
            return float(lvl), i + 1, True
    return float(cl_rel[n - 1]), n, False


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
        ds = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
        cand, _t, _c, _a, _d = cr.ck.frozen_cand(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        pr = [(int(m), "sweep")
              for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in ("delta_ext", "vol_burst"):
            v = cand.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pr.append((int(m), nm))
        for a, mem in cr.groups_with_members(pr):
            sg = {t for _, t in mem}
            if "sweep" not in sg or not (sg & {"delta_ext", "vol_burst"}):
                continue
            rd = max(min(m for m, t in mem if t == "sweep"),
                     min(m for m, t in mem if t in ("delta_ext", "vol_burst")))
            if rd < W5 or rd + DELAY + MAXHOLD >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
            j0 = rd + DELAY
            ent = float(op[j0])
            sl = slice(j0 + 1, j0 + MAXHOLD + 1)
            up = ((hi[sl] - ent) if d > 0 else (ent - lo[sl])) / A   # 順勢極值
            dn = ((ent - lo[sl]) if d > 0 else (hi[sl] - ent)) / A   # 逆勢極值
            clr = (d * (cl[sl] - ent)) / A
            fav = np.maximum.accumulate(np.maximum(up, 0.0))
            row = dict(sym=sym, day=ds[rd], entry=ent, atr=A)
            for H in FIX_H:
                k = np.flatnonzero(dn[:H] >= 1.0)
                if len(k):
                    row[f"fix{H}"], row[f"fix{H}_st"], row[f"fix{H}_m"] = \
                        -1.0, True, int(k[0]) + 1
                else:
                    row[f"fix{H}"], row[f"fix{H}_st"], row[f"fix{H}_m"] = \
                        float(clr[H - 1]), False, H
            for kk in TRAIL_K:
                for be in (False, True):
                    R, m, st = trail_exit(fav, dn, clr, kk, be)
                    tag = f"tr{kk}{'be' if be else ''}"
                    row[tag], row[tag + "_st"], row[tag + "_m"] = R, st, m
            rows.append(row)
    d = pd.DataFrame(rows)
    days = d.day.to_numpy()
    res = {"n": int(len(d))}
    print(f"=== 母體 {len(d):,} 筆（誠實錨點、進場 = 成立 +{DELAY} 分）===")
    print()
    g = float(np.mean(d.fix60))
    ok3 = abs(g - REF_T3) < TOL
    print(f"=== T3 已知答案對照：fix60 毛 {g:+.4f}"
          f"（conj_hold h60s1.0 {REF_T3:+.4f}）-> "
          + ("PASS" if ok3 else "**FAIL，不解讀**") + " ===")
    if not ok3:
        return 1
    print()

    def rep(tag):
        gg = d[tag].to_numpy(float)
        st = d[tag + "_st"].to_numpy(bool)
        nt = gg - (LEGS[0] + np.where(st, LEGS[2], LEGS[1])) / 1e4 \
            * d.entry.to_numpy() / d.atr.to_numpy()
        mn, ln = day_ci(nt, days)
        per = d.assign(x=nt).groupby("sym").x.mean()
        return dict(gross=float(np.mean(gg)), net=mn, net_lo=ln,
                    coins_pos=int((per > 0).sum()), stop=float(st.mean()),
                    hold=float(np.mean(d[tag + "_m"])))

    print("=== T1 全格（不挑格）===")
    print(f"{'出場':>12s} {'毛':>8s} {'淨':>9s} {'淨CI下':>9s} {'幣+':>5s} "
          f"{'停損%':>6s} {'平均持有(分)':>12s}")
    grid = {}
    for H in FIX_H:
        r = rep(f"fix{H}")
        grid[f"fix{H}"] = r
        print(f"{('固定 '+str(H)+'m'):>12s} {r['gross']:+8.4f} {r['net']:+9.4f} "
              f"{r['net_lo']:+9.4f} {r['coins_pos']:4d}/9 {r['stop']*100:5.0f}% "
              f"{r['hold']:12.0f}")
    for be in (False, True):
        for kk in TRAIL_K:
            tag = f"tr{kk}{'be' if be else ''}"
            r = rep(tag)
            grid[tag] = r
            lab = f"移動 {kk}" + ("＋保本" if be else "")
            ok = r["net_lo"] > 0 and r["coins_pos"] >= 6
            print(f"{lab:>12s} {r['gross']:+8.4f} {r['net']:+9.4f} "
                  f"{r['net_lo']:+9.4f} {r['coins_pos']:4d}/9 {r['stop']*100:5.0f}% "
                  f"{r['hold']:12.0f}" + ("  *" if ok else ""))
    res["grid"] = grid
    print()
    print("=== T2 判定（鄰格同向才算）===")
    win = [t for t, r in grid.items()
           if t.startswith("tr") and r["net_lo"] > 0 and r["coins_pos"] >= 6]
    if not win:
        print("  **沒有任何移動停損格過閘**")
    for t in win:
        base = t.replace("be", "")
        kk = float(base[2:])
        be = t.endswith("be")
        i = TRAIL_K.index(kk)
        nb = [grid[f"tr{TRAIL_K[i+di]}{'be' if be else ''}"]["net"]
              for di in (-1, 1) if 0 <= i + di < len(TRAIL_K)]
        same = sum(1 for x in nb if x > 0)
        print(f"  {t}：淨 {grid[t]['net']:+.4f} CI下 {grid[t]['net_lo']:+.4f} "
              f"幣 {grid[t]['coins_pos']}/9 平均持有 {grid[t]['hold']:.0f} 分 -> "
              + ("**鄰格同向**" if same == len(nb) else f"鄰格 {same}/{len(nb)}"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_trail.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print()
    print("written ->", OUT / "conj_trail.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
