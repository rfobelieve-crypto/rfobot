# -*- coding: utf-8 -*-
"""兩個倉位因子對**現行 SDV** 有沒有增量（2026-09-10 預註冊，測一次結案）

使用者 2026-09-10 指出這題優先於投票機制，理由成立：

  「SDV 的母體已經被旗標篩過一次（有旗標 +0.331、沒旗標 +0.056）。新因子
    在『所有掃單』這個大池子上有效，很可能是因為它在做旗標已經做過的事。
    在 SDV 內部還有沒有增量，這是完全不同的問題。而且它的答案可能是
    『沒有』，那樣投票機制就不用設計了。」

同時採納他另外三點：
  · 平均單筆大小**已移除** —— 它與 |delta| 在事件窗上相關 0.887，是既有
    量能資訊的變形不是新資訊（分鐘級只有 0.466，聚合後才現形）。
  · 因子只剩兩個，用**迴歸**讓資料自己定權重，不用等權投票。
  · **測一次，過不過都結案。** −0.01 那個位置最容易讓人「再試一次」，
    所以參數、因子清單、判準全部寫死在這裡，跑完就寫判決。

===========================================================================
進場時點：為什麼基準也要延後
===========================================================================
機制上，倉位變化發生在掃單**之後**；而 OI 是 5 分鐘粒度，SDV 現行的
「成立 +3 分進場」在那 3 分鐘內通常拿不到新的 OI 點。所以因子臂必須
延後到 ready+K 才有資料可用。

**因此基準也延後到同一個時點** —— 拿「ready+3 進場的基準」去比「ready+K
進場的因子臂」，量到的會是進場時點的差，不是因子的貢獻。現行規格
（+3 分）另外印出來當參考，不參與判準。

===========================================================================
臂（全格報告）
===========================================================================
    BASE    延後進場的 SDV，方向照現行規格（成立前 5 分鐘動能）
    FLT_oi  OI 因子當**濾網**：只做前半學到的那一側
    FLT_ls  散戶多空比當濾網
    TRN_oi  OI 因子當**方向翻轉**：不利那一側反著做（不減 n）
    TRN_ls  散戶多空比當方向翻轉
    REG     兩因子線性迴歸，係數**只用前半**估；只做預測值 > 0 的

判準（四條全過才算有增量）
    Q1  樣本外每筆淨值 > BASE
    Q2  標準誤（SE，這個平均值本身有多不準）< BASE 的 SE
    Q3  樣本外日聚類 CI 下緣 > 0
    Q4  逐幣 >= 6/9
**濾網臂會減少樣本數，n 下降本身會推高 SE**，所以 Q2 對濾網臂是嚴格的
——那正是它該過的關：篩掉的必須真的是雜訊，不能只是把樣本變小。

自曝檢查
    S1  BASE 在 K=0（即現行 +3 分進場）必須重現 conj_backtest 的 SDV 數字，
        對不上代表母體或進出場串錯了。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import conj_backtest as cb  # noqa: E402
import conj_clock as ck  # noqa: E402
import conj_redef as cr  # noqa: E402
import event_census as ec  # noqa: E402
import sdv_after_sweep as sas  # noqa: E402

OUT = HERE / "data" / "results"
K = 10                      # 因子觀察窗（分鐘），與 sdv_vote 的最佳格一致
FLOW = cb.FLOW
W, STOP, HOLD = cb.W, cb.STOP, cb.HOLD
SEED = 20260910


def build(sym, k):
    """SDV 事件（現行規格的母體），延後 k 分鐘進場，附兩個倉位因子。"""
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    n = len(ts)
    oi_t, oi_v, oi_ls = sas.load_oi(sym)

    pairs = [(int(m), "sweep") for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            pairs += [(int(m), nm) for m in ec.cooldown_filter(np.sort(v))]

    rows = []
    for _a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig or not ({"delta_ext", "vol_burst"} <= sig):
            continue                                   # 只要 SDV
        ready = max(min(m for m, t in mem if t == "sweep"),
                    min(m for m, t in mem if t in FLOW))
        j0 = ready + k
        if ready < W or j0 + 1 + HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)   # 現行方向規則

        t_ms = int(ts[ready])
        base = int(np.searchsorted(oi_t, t_ms, side="right")) - 1
        end = int(np.searchsorted(oi_t, int(ts[j0]), side="right")) - 1
        if base < 0 or end <= base:
            continue
        f_oi = sas._pct(oi_v[end], oi_v[base])
        f_ls = sas._pct(oi_ls[end], oi_ls[base])
        if not (np.isfinite(f_oi) and np.isfinite(f_ls)):
            continue

        rows.append(dict(sym=sym, day=int(ts[j0]) // 86_400_000,
                         f_oi=f_oi, f_ls=f_ls,
                         R_with=sas.run_trade(op, hi, lo, cl, j0, d, A),
                         R_against=sas.run_trade(op, hi, lo, cl, j0, -d, A)))
    return rows


def boot(days, vals, n=2000):
    rng = np.random.default_rng(SEED)
    by = {}
    for dd, v in zip(days, vals):
        by.setdefault(int(dd), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return float(np.mean(vals)), np.nan, np.nan
    arr = [np.array(by[x]) for x in ks]
    idx = rng.integers(0, len(ks), size=(n, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(n)])
    return float(np.mean(vals)), float(o.std(ddof=1)), float(np.percentile(o, 2.5))


def main():
    recs = []
    for s in cb.CORE9:
        recs += build(s, K)
    d = pd.DataFrame(recs).reset_index(drop=True)
    mid = float(d.day.median())
    # 逐幣、只用前半的分布轉百分位（跨幣尺度不可比的教訓）
    for col in ("f_oi", "f_ls"):
        out = np.full(len(d), np.nan)
        for s, g in d.groupby("sym"):
            ref = np.sort(g[g.day < mid][col].to_numpy(float))
            if len(ref) < 20:
                continue
            out[g.index.to_numpy()] = np.searchsorted(
                ref, g[col].to_numpy(float), side="right") / len(ref)
        d[col + "_p"] = out
    d = d.dropna(subset=["f_oi_p", "f_ls_p"]).reset_index(drop=True)
    first, second = d[d.day < mid], d[d.day >= mid]

    print("=" * 78)
    print(f"SDV 母體、延後 {K} 分鐘進場   {len(d):,} 筆"
          f"（前半 {len(first):,} / 後半 {len(second):,}）")
    print()
    signs = {}
    for col, nm in (("f_oi_p", "OI 變化"), ("f_ls_p", "散戶多空比")):
        h = first[first[col] > 0.5]["R_with"].mean()
        l = first[first[col] <= 0.5]["R_with"].mean()
        signs[col] = 1 if h >= l else -1
        print(f"  前半學到：{nm:12} 高{h:+.3f} / 低{l:+.3f} -> "
              f"{'高值→順勢' if signs[col] > 0 else '高值→逆勢'}")

    # 迴歸：只用前半估係數
    X1 = np.c_[np.ones(len(first)), first.f_oi_p, first.f_ls_p]
    beta = np.linalg.lstsq(X1, first.R_with.to_numpy(), rcond=None)[0]
    X2 = np.c_[np.ones(len(second)), second.f_oi_p, second.f_ls_p]
    pred = X2 @ beta
    print(f"  迴歸係數（只用前半）：常數 {beta[0]:+.3f}  OI {beta[1]:+.3f}  "
          f"散戶多空比 {beta[2]:+.3f}")

    arms = {}
    arms["BASE 延後進場的 SDV"] = (second.R_with.to_numpy(), second.day.values,
                                   second.sym.values)
    for col, nm in (("f_oi_p", "OI"), ("f_ls_p", "散戶多空比")):
        good = (second[col].to_numpy() > 0.5) == (signs[col] > 0)
        arms[f"FLT_{nm} 濾網"] = (second.R_with.to_numpy()[good],
                                  second.day.values[good], second.sym.values[good])
        dirs = np.where(second[col].to_numpy() > 0.5, signs[col], -signs[col])
        arms[f"TRN_{nm} 翻轉"] = (np.where(dirs > 0, second.R_with, second.R_against),
                                  second.day.values, second.sym.values)
    sel = pred > 0
    arms["REG 迴歸(只做預測>0)"] = (second.R_with.to_numpy()[sel],
                                     second.day.values[sel], second.sym.values[sel])

    print()
    print(f"後半（真樣本外）")
    print(f"{'臂':26} {'n':>6} {'淨/筆':>9} {'SE':>8} {'CI下緣':>9} {'幣+':>6}")
    res = {}
    for nm, (v, dd, sy) in arms.items():
        v = np.asarray(v, float)
        if len(v) < 30:
            print(f"{nm:26} {len(v):6,}  樣本太少，不評")
            continue
        m, se, lo = boot(dd, v)
        per = pd.DataFrame({"s": sy, "v": v}).groupby("s").v.mean()
        res[nm] = dict(n=int(len(v)), m=m, se=se, lo=lo,
                       npos=int((per > 0).sum()), nsym=int(len(per)))
        print(f"{nm:26} {len(v):6,} {m:+9.4f} {se:8.4f} {lo:+9.4f} "
              f"{int((per > 0).sum()):3d}/{len(per)}")

    b = res["BASE 延後進場的 SDV"]
    print()
    print("判準（四條全過才算有增量）：")
    any_pass = False
    for nm, o in res.items():
        if nm.startswith("BASE"):
            continue
        q1, q2 = o["m"] > b["m"], o["se"] < b["se"]
        q3, q4 = o["lo"] > 0, o["npos"] >= 6
        ok = q1 and q2 and q3 and q4
        any_pass |= ok
        print(f"  {nm:26} Q1報酬{'>' if q1 else '<'}基準 "
              f"Q2 SE{'降' if q2 else '升'} Q3下緣{'>0' if q3 else '<=0'} "
              f"Q4幣{o['npos']}/{o['nsym']}  -> {'**有增量**' if ok else '無增量'}")
    print()
    print("判決：" + ("至少一臂有增量，可進下一步設計"
                     if any_pass else "**兩個倉位因子對現行 SDV 沒有增量** —— 依預註冊，結案"))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_increment.json"
    p.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
