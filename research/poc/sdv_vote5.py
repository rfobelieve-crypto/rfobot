# -*- coding: utf-8 -*-
"""五因子投票：把 delta 與 volume 從硬門檻降級成因子（2026-09-10 預註冊）

使用者 2026-09-10：「原本的 delta 跟 volume 也要一起加入因子行列」。

現行規格把 D（|delta| 5 分後向和 >= 滾動 30 日 p99）與 V（量能比 >= p99）
當**硬門檻**（S ∧ D ∧ V 才算數）。那把強度資訊二元化丟掉了：剛好差一點
的掃單與差很多的掃單被當成同一回事，而剛好超過的與遠遠超過的也是。
本檔把它們降級成**連續因子**，與倉位類因子平起平坐一起投票。

連帶：母體從 SDV 的 1,584 筆變成**所有掃單 9,251 筆**。

===========================================================================
修掉上一輪的一個缺陷：跨幣尺度
===========================================================================
`sdv_vote.py` 用「全部樣本的中位數」切每個因子。OI 變化與散戶多空比變化
是**變化率**，跨幣可比，沒事；但平均單筆大小是**絕對值** ——

    BTC 的平均單筆 ~0.005 顆   DOGE 的平均單筆 幾千顆

拿全體中位數去切它，切出來的是**幣種**不是強度。上一輪它「後半翻號、
被判不穩定」很可能是這個造成的，不是因子本身的問題。

**本檔所有因子一律逐幣標準化**：用**該幣前半**的分布把值轉成百分位
（0-1），前半算 ECDF、套用到全部。後半一個數字都不參與這個轉換。

===========================================================================
五個因子
===========================================================================
    f_oi     OI 變化率          倉位在增還是在減
    f_ls     散戶多空比變化率    散戶站哪一邊
    f_size   平均單筆大小        下單的是誰（volume / n_trades）
    f_delta  |delta| 5 分後向和  現行的 D，改成連續值
    f_vol    量能比              現行的 V，改成連續值

方向一律**只用前半學**（照 sdv_vote 的流程，不由我指定）：
把該幣前半的百分位以 0.5 切兩半，看哪一邊順勢比較賺 -> sign。

===========================================================================
判準（跑之前凍結）
===========================================================================
臂     五個單因子各一臂、VOTE5（五因子多數決）、B（全順勢）、C（全逆勢）
       另加 VOTE_STABLE：**只讓在前半內部就已經穩定的因子投票**
       （前半再切兩半，兩半 sign 一致才有投票權；這條規則只用前半資訊）

P1  VOTE5 或 VOTE_STABLE 的樣本外每筆淨值 > max(所有單因子, B, C)
P2  該臂日聚類 bootstrap CI 下緣 > 0
P3  該臂逐幣 >= 6/9
三條全過才是採用候選。**單因子不設門檻**（使用者已明說單因子 fail 正常），
但翻號者如實標出。

自曝檢查
    A1  f_delta / f_vol 的高百分位子集，應該大致重現 SDV 母體的量級 ——
        對不上代表我把現行門檻翻譯成連續值時翻錯了。
    A2  逐幣標準化後，每個因子在每個幣的高低兩組筆數應該接近各半 ——
        差太多代表 ECDF 沒套對。
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
import event_census as ec  # noqa: E402
import sdv_after_sweep as sas  # noqa: E402

OUT = HERE / "data" / "results"
KS = (5, 10, 15)
FACTORS = (("f_oi", "OI 變化"), ("f_ls", "散戶多空比"), ("f_size", "平均單筆大小"),
           ("f_delta", "主動量強度(D)"), ("f_vol", "量能強度(V)"))
STOP, HOLD, W = cb.STOP, cb.HOLD, cb.W
SEED = 20260910


def build_sym(sym, k):
    """一個幣的逐筆記錄，含五個原始因子值與順勢／逆勢報酬。"""
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close",
                                 "volume", "n_trades", "delta"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    ntr = np.nan_to_num(b["n_trades"].to_numpy(float), nan=0.0)
    dlt = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
    n = len(ts)
    oi_t, oi_v, oi_ls = sas.load_oi(sym)

    ev = pd.read_parquet(cb.EVENTS / f"{sym}.parquet",
                         columns=["t_sweep", "side"]).sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    ev_sd = ev["side"].to_numpy(object)

    rows = []
    for m in ec.cooldown_filter(np.sort(cand["sweep"])):
        m = int(m)
        j0 = m + k
        if m < W or j0 + 1 + HOLD >= n:
            continue
        A = float(at[m])
        if not np.isfinite(A) or A <= 0:
            continue
        t_ms = int(ts[m]) + 60_000
        i = int(np.searchsorted(ev_ts, t_ms))
        if i >= len(ev_ts) or ev_ts[i] != t_ms:
            continue
        d_s = 1.0 if str(ev_sd[i]) == "buyside" else -1.0

        base = int(np.searchsorted(oi_t, t_ms, side="right")) - 1
        end = int(np.searchsorted(oi_t, int(ts[j0]), side="right")) - 1
        if base < 0 or end <= base:
            continue
        f_oi = sas._pct(oi_v[end], oi_v[base])
        f_ls = sas._pct(oi_ls[end], oi_ls[base])
        seg = slice(m, j0 + 1)
        vs, ns = vol[seg].sum(), ntr[seg].sum()
        if ns <= 0 or vs <= 0:
            continue
        f_size = vs / ns
        f_delta = float(np.abs(dlt[seg].sum()))
        f_vol = float(vs)
        if not all(np.isfinite(x) for x in (f_oi, f_ls, f_size, f_delta, f_vol)):
            continue

        rows.append(dict(sym=sym, day=int(ts[j0]) // 86_400_000, j0=j0,
                         f_oi=f_oi, f_ls=f_ls, f_size=f_size,
                         f_delta=f_delta, f_vol=f_vol,
                         R_with=sas.run_trade(op, hi, lo, cl, j0, d_s, A),
                         R_against=sas.run_trade(op, hi, lo, cl, j0, -d_s, A)))
    return rows


def pctile_by_sym(d, col, mid):
    """逐幣：用**該幣前半**的分布，把值轉成 0-1 百分位。後半不參與。"""
    out = np.full(len(d), np.nan)
    for s, g in d.groupby("sym"):
        ref = g[g.day < mid][col].to_numpy(float)
        if len(ref) < 20:
            continue
        ref = np.sort(ref)
        out[g.index.to_numpy()] = np.searchsorted(ref, g[col].to_numpy(float),
                                                  side="right") / len(ref)
    return out


def learn_sign(first, col):
    hi = first[first[col] > 0.5]["R_with"].mean()
    lo = first[first[col] <= 0.5]["R_with"].mean()
    return (1 if hi >= lo else -1), float(hi), float(lo)


def stable_in_first(first, col):
    """前半再切兩半，兩半 sign 一致才給投票權（只用前半資訊）。"""
    q = first.day.median()
    a, b = first[first.day < q], first[first.day >= q]
    if len(a) < 50 or len(b) < 50:
        return False
    sa = learn_sign(a, col)[0]
    sb = learn_sign(b, col)[0]
    return sa == sb


def ci_lo(days, vals):
    rng = np.random.default_rng(SEED)
    by = {}
    for dd, v in zip(days, vals):
        by.setdefault(int(dd), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return np.nan
    arr = [np.array(by[x]) for x in ks]
    idx = rng.integers(0, len(ks), size=(2000, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(2000)])
    return float(np.percentile(o, 2.5))


def main():
    allout = {}
    for k in KS:
        recs = []
        for s in cb.CORE9:
            recs += build_sym(s, k)
        d = pd.DataFrame(recs).reset_index(drop=True)
        mid = float(d.day.median())
        for col, _ in FACTORS:
            d[col] = pctile_by_sym(d, col, mid)
        d = d.dropna(subset=[c for c, _ in FACTORS]).reset_index(drop=True)
        first, second = d[d.day < mid], d[d.day >= mid]

        print("=" * 78)
        print(f"K = {k} 分鐘   {len(d):,} 筆（前半 {len(first):,} / 後半 {len(second):,}）")
        print()
        print("只用前半學方向（逐幣標準化後）：")
        signs, stable = {}, {}
        for col, nm in FACTORS:
            sg, hi, lo = learn_sign(first, col)
            st = stable_in_first(first, col)
            signs[col], stable[col] = sg, st
            h2 = second[second[col] > 0.5]["R_with"].mean()
            l2 = second[second[col] <= 0.5]["R_with"].mean()
            post = 1 if h2 >= l2 else -1
            print(f"  {nm:14} 前半 高{hi:+.3f}/低{lo:+.3f} -> "
                  f"{'高值→順勢' if sg > 0 else '高值→逆勢'}"
                  f"   前半內部穩定 {'是' if st else '否'}"
                  f"   後半 高{h2:+.3f}/低{l2:+.3f}{'' if post == sg else '  ← 後半翻號'}")

        arms = {}
        for col, nm in FACTORS:
            dirs = np.where(second[col].to_numpy() > 0.5, signs[col], -signs[col])
            arms[nm] = np.where(dirs > 0, second.R_with, second.R_against)
        for lab, cols in (("VOTE5 五因子", [c for c, _ in FACTORS]),
                          ("VOTE_STABLE 只讓穩定的投",
                           [c for c, _ in FACTORS if stable[c]])):
            if not cols:
                continue
            v = np.zeros(len(second))
            for col in cols:
                v += np.where(second[col].to_numpy() > 0.5, signs[col], -signs[col])
            arms[f"{lab}({len(cols)})"] = np.where(v >= 0, second.R_with, second.R_against)
        arms["B 全部順勢"] = second.R_with.to_numpy()
        arms["C 全部逆勢"] = second.R_against.to_numpy()

        print()
        print(f"後半（真樣本外） n = {len(second):,}")
        print(f"{'臂':26} {'淨/筆':>9} {'CI下緣':>9} {'幣+':>6}")
        res = {}
        for nm, v in arms.items():
            v = np.asarray(v, float)
            per = pd.DataFrame({"sym": second.sym.values, "v": v}).groupby("sym").v.mean()
            lo = ci_lo(second.day.values, v)
            res[nm] = dict(m=float(v.mean()), lo=lo, npos=int((per > 0).sum()),
                           nsym=int(len(per)))
            print(f"{nm:26} {v.mean():+9.4f} {lo:+9.4f} {int((per > 0).sum()):3d}/{len(per)}")

        print()
        for nm in [x for x in res if x.startswith("VOTE")]:
            o = res[nm]
            others = [res[x]["m"] for x in res if not x.startswith("VOTE")]
            p1, p2, p3 = o["m"] > max(others), o["lo"] > 0, o["npos"] >= 6
            print(f"  {nm}: P1 {'PASS' if p1 else 'FAIL'} / P2 {'PASS' if p2 else 'FAIL'}"
                  f"（{o['lo']:+.4f}）/ P3 {'PASS' if p3 else 'FAIL'}（{o['npos']}/{o['nsym']}）"
                  f" -> {'採用候選' if (p1 and p2 and p3) else '不過'}")
        print()
        allout[str(k)] = res

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_vote5.json"
    p.write_text(json.dumps(allout, indent=2, default=float), encoding="utf-8")
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
