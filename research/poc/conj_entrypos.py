# -*- coding: utf-8 -*-
"""「進場一下子就出場了」—— 進場點是不是坐在衝刺的末端

使用者 2026-09-09 看圖指出：某一筆進場 81920、停損 81366（1 ATR），
**出場價與停損價同一個數字**，而且進出場在圖上幾乎黏在一起。

這與 `conj_hold.py` 網格的形狀一致：停損放寬（1.0 → 2.0 → 3.0 → 無）
每一格都變好。兩者可能是同一個機制的兩面：

    交會事件依定義是一根**衝刺**（掃單 ∧ 極端流量）。
    在成立後 +3 分進場 = 追在衝刺末端。
    1 ATR 的停損擋不住那根**自然回踩**，於是先被掃掉、再看著它往原方向走。

旁證（已量到，不是推論）：`conj_rescue.py` 的 C3 量出「限價掛在
close(ready) 的成交率 97.8%」—— **回踩幾乎一定會來**。

===========================================================================
判準（跑之前寫死）
===========================================================================
E1  **先描述再處方**：停損被打到的時間分布。若中位數 ≤ 10 分鐘，
    「一下子就出場」成立；否則使用者看到的是個案，不得當成通則。
E2  進場點在事件區間裡的位置：`(entry − 區間低) / (區間高 − 區間低)`，
    區間取 [ready−5, ready+3]。順勢方向下若中位 ≥ 0.7，代表**追在極端**。
E3  回踩限價：限價 = close(ready) − k × ATR（順勢方向），k ∈
    {0, 0.10, 0.25, 0.50}，等 K ∈ {3, 5, 10} 分鐘。全格報告
    成交率 / 進場價改善 / 停損率 / 淨值與日聚類 CI / 逐幣。
    未成交的那批要一併報它們的市價毛利（逆選擇照妖鏡）。
E4  過閘 = 淨值 CI 下緣 > 0 且逐幣 ≥6/9。**單格不算數**，要 k 或 K 的
    鄰格同向（threshold-sweep 陷阱，mistake.md 2026-06-20）。
E5  這是在看過失敗之後提出的，任何過閘只代表**值得開自己的前瞻時鐘**，
    不得復活現行註冊（§0.92）。
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
W5, DELAY, HOLD = 5, 3, 60
FLOW = ("delta_ext", "vol_burst")
LEGS = (2.0, 2.0, 10.0)          # 限價兩腿（本檔測的都是限價進場）
KS = (0.0, 0.10, 0.25, 0.50)
KWAIT = (3, 5, 10)
STOP = 1.0
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


def main():
    base, lim = [], []
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
        for nm in FLOW:
            v = cand.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pr.append((int(m), nm))

        for a, mem in cr.groups_with_members(pr):
            sg = {t for _, t in mem}
            if "sweep" not in sg or not (sg & set(FLOW)):
                continue
            rd = max(min(m for m, t in mem if t == "sweep"),
                     min(m for m, t in mem if t in FLOW))
            if rd < W5 or rd + DELAY + max(KWAIT) + HOLD >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
            j0 = rd + DELAY
            ent = float(op[j0])
            end = j0 + HOLD
            adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
                   else (hi[j0 + 1:end + 1] - ent)) / A
            k = np.flatnonzero(adv >= STOP)
            stopped = bool(len(k))
            tstop = int(k[0]) + 1 if stopped else np.nan
            R = -STOP if stopped else float(d * (cl[end] - ent) / A)
            # E2 進場在區間裡的位置（順勢方向）
            seg_h = float(np.max(hi[rd - W5:j0 + 1]))
            seg_l = float(np.min(lo[rd - W5:j0 + 1]))
            rng = seg_h - seg_l
            pos = (ent - seg_l) / rng if rng > 0 else np.nan
            if d < 0:
                pos = 1 - pos
            base.append(dict(sym=sym, day=ds[rd], R=R, stopped=stopped,
                             tstop=tstop, pos=pos, entry=ent, atr=A))
            # E3 回踩限價
            L0 = float(cl[rd])
            for kk in KS:
                L = L0 - d * kk * A
                for K in KWAIT:
                    seg = (lo[rd + 1:rd + 1 + K] if d > 0 else hi[rd + 1:rd + 1 + K])
                    hk = np.flatnonzero(seg <= L) if d > 0 else np.flatnonzero(seg >= L)
                    if len(hk):
                        j = rd + 1 + int(hk[0])
                        e2 = j + HOLD
                        if e2 + 1 >= n:
                            continue
                        ad2 = ((L - lo[j + 1:e2 + 1]) if d > 0
                               else (hi[j + 1:e2 + 1] - L)) / A
                        st2 = bool((ad2 >= STOP).any())
                        R2 = -STOP if st2 else float(d * (cl[e2] - L) / A)
                        lim.append(dict(sym=sym, day=ds[j], k=kk, K=K, filled=True,
                                        R=R2, stopped=st2, entry=L, atr=A, R_mkt=R))
                    else:
                        lim.append(dict(sym=sym, day=ds[rd], k=kk, K=K, filled=False,
                                        R=np.nan, stopped=False, entry=L, atr=A,
                                        R_mkt=R))
    d = pd.DataFrame(base)
    dl = pd.DataFrame(lim)
    res = {"n": int(len(d))}

    print(f"=== 母體 {len(d):,} 筆（誠實錨點、進場 = 成立 +3 分、停損 1 ATR）===")
    print()
    print("=== E1 停損多快被打到（使用者：「進場一下子就出場了」）===")
    ts_ = d.tstop.dropna().to_numpy(float)
    print(f"  停損率 {d.stopped.mean()*100:.1f}%   n={len(ts_):,}")
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<2d} {np.percentile(ts_, q):6.0f} 分")
    med = float(np.median(ts_))
    print(f"  -> 中位 {med:.0f} 分 " + ("**「一下子就出場」成立**" if med <= 10
                                       else "（> 10 分，使用者看到的是個案）"))
    for w in (3, 5, 10, 20):
        print(f"    {w:2d} 分鐘內被停損：{np.mean(ts_ <= w)*100:5.1f}%"
              f"（佔全部交易 {np.mean(ts_ <= w)*d.stopped.mean()*100:5.1f}%）")
    res["stop_rate"] = float(d.stopped.mean())
    res["tstop_median"] = med
    print()
    print("=== E2 進場點在事件區間裡的位置（1.0 = 順勢方向的極端）===")
    pv = d.pos.dropna().to_numpy(float)
    print(f"  中位 {np.median(pv):.3f}   平均 {np.mean(pv):.3f}   "
          f"≥0.7 佔 {np.mean(pv >= 0.7)*100:.1f}%   ≥0.9 佔 {np.mean(pv >= 0.9)*100:.1f}%")
    print("  -> " + ("**追在極端**（中位 ≥ 0.7）" if np.median(pv) >= 0.7
                     else "沒有系統性追在極端"))
    res["pos_median"] = float(np.median(pv))
    # 停損 vs 沒停損的進場位置
    a1 = d[d.stopped].pos.median()
    a2 = d[~d.stopped].pos.median()
    print(f"  被停損那批 中位 {a1:.3f}   沒被停損 {a2:.3f}   差 {a1-a2:+.3f}")
    print()
    print("=== E3 回踩限價（限價 = close(ready) − k×ATR，等 K 分）===")
    print(f"{'k':>5s} {'K':>3s} {'成交率':>7s} {'進場改善(ATR)':>13s} {'停損率':>7s} "
          f"{'淨':>9s} {'淨CI下':>9s} {'幣+':>5s} {'未成交的市價毛':>14s}")
    win = []
    for kk in KS:
        for K in KWAIT:
            g = dl[(dl.k == kk) & (dl.K == K)]
            if len(g) < 100:
                continue
            f = g[g.filled]
            miss = g[~g.filled]
            if len(f) < 50:
                continue
            nt = f.R.to_numpy() - (LEGS[0] + np.where(f.stopped.to_numpy(bool),
                                                      LEGS[2], LEGS[1])) / 1e4 \
                * f.entry.to_numpy() / f.atr.to_numpy()
            mn, ln = day_ci(nt, f.day.to_numpy())
            per = f.assign(x=nt).groupby("sym").x.mean()
            pos_c = int((per > 0).sum())
            imp = kk  # 相對 close(ready) 的改善就是 k（成交才算）
            mm = float(miss.R_mkt.mean()) if len(miss) else np.nan
            ok = ln > 0 and pos_c >= 6
            if ok:
                win.append((kk, K, mn, ln, pos_c))
            print(f"{kk:5.2f} {K:3d} {f.filled.mean() if False else len(f)/len(g)*100:6.1f}% "
                  f"{imp:13.2f} {f.stopped.mean()*100:6.1f}% {mn:+9.4f} {ln:+9.4f} "
                  f"{pos_c:4d}/9 {mm:+14.4f}" + ("  *" if ok else ""))
            res.setdefault("limit", {})[f"k{kk}K{K}"] = dict(
                fill=float(len(f) / len(g)), stop=float(f.stopped.mean()),
                net=mn, net_lo=ln, coins_pos=pos_c, miss_mkt=mm)
    print()
    print("=== E4 判定（單格不算數，要鄰格同向）===")
    if not win:
        print("  **沒有任何格子過閘** -> 往回踩掛單救不回來（停損 1 ATR 下）。")
    else:
        for kk, K, mn, ln, pc in win:
            print(f"  k={kk} K={K}：淨 {mn:+.4f} CI下 {ln:+.4f} 幣 {pc}/9")
        print(f"  過閘 {len(win)}/{len(res.get('limit', {}))} 格"
              + ("（成片，可信）" if len(win) >= 3 else "（零星，當雜訊看）"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_entrypos.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_entrypos.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
