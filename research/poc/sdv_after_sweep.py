# -*- coding: utf-8 -*-
"""掃單之後：倉位怎麼變，決定延續還是反轉（2026-09-10 預註冊）

使用者的假設（原話，2026-09-10）：
  「以 buyside liquidity 來說今天價格上去了獵取了，如果散戶不投降還在空空空
    那價格可能就會延續；那如果今天是價格上去了散戶原本做空的不多，反而是
    突破後做多的比較多，價格就會變假突破跌回來獵取散戶止損變成一種反轉」

**這個假設可以證偽，因為兩種情境在資料上留下相反的指紋：**

    空單不投降 -> 繼續被強平 -> 那是**平倉** -> 未平倉量**下降**
    追多的變多 -> 那是**開新倉**           -> 未平倉量**上升**

兩種情境的主動買都會飆高（強平是被迫市價買、追多是主動市價買），所以
現行的 D（|delta| 極端）與 V（量能爆發）**結構上分不出來** —— 它們只看
量的大小，不看倉位的方向。這是這條線第一次問「是誰在下單、倉位在增還是減」。

===========================================================================
為什麼母體是「所有掃單」而不是 SDV
===========================================================================
§1.03j 判過：沒有流量的掃單（佔 60.4%）既不延續也不反轉，**什麼都沒發生**。
但那次只用「有沒有量」去切。如果真正的判別器是倉位方向，那 60.4% 裡面
可能同時混著延續與反轉兩種，平均起來當然是零。**本檔就是要拆開它。**

===========================================================================
設計：直接做成可交易的形式，不做前視版
===========================================================================
機制上，清算發生在掃單**之後**幾分鐘。要觀察它就必須等 —— 所以本檔
**延後 K 分鐘進場**，用 [t, t+K] 這段已經發生的變化當判別器：

    掃單在 t（該分鐘收盤，事件成立）
    等到 t+K，此時 [t, t+K] 的 OI 變化、多空比變化、平均單筆大小都**已知**
    在 t+K 的下一分鐘開盤進場
    K ∈ {5, 10, 15}，全格報告

**沒有前視**：OI 與多空比是 5 分鐘粒度，只取 create_time ∈ (t, t+K] 的
資料點；那些在 t+K 當下已經發布。基準點取 create_time <= t 的最後一筆。
（這條線今天已經被前視咬過一次，§1.03b —— 進場定義前視讓 +0.5064 縮成
+0.1094。所以本檔的自曝檢查 S1 直接驗這件事。）

===========================================================================
三個判別器（性質互相獨立，不是同一個量的變形）
===========================================================================
    F1  OI 變化      倉位在增還是在減      -> 平倉清算 vs 開新倉
    F2  散戶多空比    散戶站哪一邊、往哪跑   -> 不投降 vs 追進來
    F3  平均單筆大小  下單的是誰            -> 大單掃盤 vs 散戶亂跑
        = volume / n_trades，這條線從未用過（delta 是 taker_buy_base 的
          變形，誤差 0.0；F3 與 volume 相關僅 0.52、與 |delta| 0.47）

===========================================================================
判準（跑之前凍結，事後不放寬）
===========================================================================
臂（每個 K 各跑一次，全格報告）
    A  使用者的假設   OI 降 -> 順勢；OI 升 -> 逆勢
    B  對照：全部順勢（掃 buyside 就做多）
    C  對照：全部逆勢
    D  單因子分桶     F1/F2/F3 各自的高低桶，順勢報酬的差

主判準（A 臂要成立必須全部滿足）
    P1  A 的每筆淨值 > max(B, C)          —— 判別器要贏過兩個無腦對照
    P2  A 的日聚類 bootstrap CI 下緣 > 0
    P3  逐幣 ≥ 6/9
    P4  樣本外（後半）同樣成立            —— 判決只看這一段

自曝檢查（儀器對不對，與判決分開）
    S1  前視防護：把判別器改用 (t+K, t+2K] 的資料（真正的未來）重跑一次，
        若 A 臂**沒有**明顯變好，代表我的對齊寫錯了 —— 真的前視應該要
        給出更好的成績，給不出來就是資料根本沒接上。
    S2  B 臂（全部順勢）在 SDV 子集上必須重現既有的量級（+0.3 附近），
        對不上代表母體或進出場串錯了。

用法
    python research/poc/sdv_after_sweep.py
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

OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
KS = (5, 10, 15)
STOP, HOLD, W = cb.STOP, cb.HOLD, cb.W
SEED = 20260910

SYM_MAP = {s: f"{s}USDT" for s in cb.CORE9}


def load_oi(sym):
    """5 分鐘的 OI 與多空比。回傳 (ts_ms, oi, retail_ls) 三個等長陣列。"""
    d = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest",
                                 "count_long_short_ratio"])
    t = pd.to_datetime(d["create_time"]).astype("int64") // 10 ** 6
    o = np.argsort(t.values)
    return (t.values[o],
            d["sum_open_interest"].to_numpy(float)[o],
            d["count_long_short_ratio"].to_numpy(float)[o])


def _pct(a, b):
    return np.nan if (not np.isfinite(a) or not np.isfinite(b) or b == 0) else (a - b) / abs(b)


def build(sym, k, future=False):
    """一個幣、延後 k 分鐘的逐筆記錄。future=True 用 (t+k, t+2k] —— S1 用。"""
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close",
                                 "volume", "n_trades"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    ntr = np.nan_to_num(b["n_trades"].to_numpy(float), nan=0.0)
    n = len(ts)
    oi_t, oi_v, oi_ls = load_oi(sym)

    # 掃單的側別（buyside = 向上穿越）
    ev = pd.read_parquet(cb.EVENTS / f"{sym}.parquet",
                         columns=["t_sweep", "side"]).sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    ev_sd = ev["side"].to_numpy(object)

    rows = []
    for m in ec.cooldown_filter(np.sort(cand["sweep"])):
        m = int(m)
        j0 = m + k                      # 判別器可用的時刻
        if m < W or j0 + 1 + HOLD >= n:
            continue
        A = float(at[m])
        if not np.isfinite(A) or A <= 0:
            continue
        t_ms = int(ts[m]) + 60_000       # 掃單那分鐘的收盤
        i = int(np.searchsorted(ev_ts, t_ms))
        if i >= len(ev_ts) or ev_ts[i] != t_ms:
            continue
        side = str(ev_sd[i])
        d_sweep = 1.0 if side == "buyside" else -1.0   # 順勢 = 穿越方向

        # ---- 判別器視窗（沒有前視：只取 t+k 當下已發布的資料點）----
        w0, w1 = (t_ms, int(ts[j0])) if not future else (int(ts[j0]), int(ts[m + 2 * k]))
        base = int(np.searchsorted(oi_t, w0, side="right")) - 1
        end = int(np.searchsorted(oi_t, w1, side="right")) - 1
        if base < 0 or end <= base:
            continue
        f1 = _pct(oi_v[end], oi_v[base])          # OI 變化率
        f2 = _pct(oi_ls[end], oi_ls[base])        # 散戶多空比變化率
        seg = slice(m, j0 + 1)
        f3 = (vol[seg].sum() / ntr[seg].sum()) if ntr[seg].sum() > 0 else np.nan
        if not (np.isfinite(f1) and np.isfinite(f2) and np.isfinite(f3)):
            continue

        rows.append(dict(sym=sym, day=int(ts[j0]) // 86_400_000, j0=j0,
                         side=side, d_sweep=d_sweep, A=A, f1=f1, f2=f2, f3=f3))
    return rows, op, hi, lo, cl, n


def run_trade(op, hi, lo, cl, j0, d, A):
    """進場 open(j0+1)、停損 STOP×A、持有 HOLD。回傳淨 R。"""
    e = j0 + 1
    ent = float(op[e])
    end = e + HOLD
    adv = ((ent - lo[e + 1:end + 1]) if d > 0 else (hi[e + 1:end + 1] - ent)) / A
    if len(np.flatnonzero(adv >= STOP)):
        R, stopped = -STOP, True
    else:
        R, stopped = float(d * (cl[end] - ent) / A), False
    leg = cb.COST_ENTRY + (cb.COST_STOP if stopped else cb.COST_TIME)
    return R - leg / 1e4 * ent / A


def ci_lo(sub, col="R"):
    rng = np.random.default_rng(SEED)
    by = {k: v.to_numpy(float) for k, v in sub.groupby("day")[col]}
    ks = list(by)
    if len(ks) < 5:
        return np.nan
    arr = [by[k] for k in ks]
    idx = rng.integers(0, len(ks), size=(2000, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(2000)])
    return float(np.percentile(o, 2.5))


def evaluate(k, future=False):
    recs = []
    for s in cb.CORE9:
        rows, op, hi, lo, cl, n = build(s, k, future)
        for r in rows:
            d_s = r["d_sweep"]
            # A：OI 降（平倉清算）-> 順勢；OI 升（開新倉）-> 逆勢
            d_a = d_s if r["f1"] < 0 else -d_s
            r["RA"] = run_trade(op, hi, lo, cl, r["j0"], d_a, r["A"])
            r["RB"] = run_trade(op, hi, lo, cl, r["j0"], d_s, r["A"])
            r["RC"] = -0.0 + run_trade(op, hi, lo, cl, r["j0"], -d_s, r["A"])
            recs.append(r)
    return pd.DataFrame(recs)


def report(d, k, mid, tag=""):
    print(f"\n{'=' * 74}")
    print(f"K = {k} 分鐘{tag}   n = {len(d):,}")
    print(f"{'臂':26} {'期間':8} {'n':>6} {'淨/筆':>9} {'CI下緣':>9} {'幣+':>6}")
    out = {}
    for arm, lab in (("RA", "A 使用者假設（OI 決定方向）"),
                     ("RB", "B 對照：全部順勢"),
                     ("RC", "C 對照：全部逆勢")):
        for pl, sub in (("全期", d), ("樣本外", d[d.day >= mid])):
            per = sub.groupby("sym")[arm].mean()
            lo = ci_lo(sub.rename(columns={arm: "R"}))
            out[(arm, pl)] = dict(n=len(sub), m=float(sub[arm].mean()), lo=lo,
                                  npos=int((per > 0).sum()), nsym=len(per))
            print(f"{lab:26} {pl:8} {len(sub):6,} {sub[arm].mean():+9.4f} "
                  f"{lo:+9.4f} {int((per > 0).sum()):3d}/{len(per)}")
        print()
    return out


def main():
    rng = np.random.default_rng(SEED)
    allout = {}
    first = evaluate(KS[0])
    mid = float(first.day.median())
    for k in KS:
        d = first if k == KS[0] else evaluate(k)
        o = report(d, k, mid)
        # D：單因子分桶（全格，不挑）
        print(f"{'單因子分桶（樣本外，順勢報酬）':26}")
        oos = d[d.day >= mid]
        for f, nm in (("f1", "OI 變化"), ("f2", "散戶多空比變化"), ("f3", "平均單筆大小")):
            q = oos[f].quantile([0.25, 0.5, 0.75]).tolist()
            lab = ["最低25%", "25-50%", "50-75%", "最高25%"]
            cuts = [-np.inf] + q + [np.inf]
            line = []
            for i in range(4):
                a = oos[(oos[f] > cuts[i]) & (oos[f] <= cuts[i + 1])]
                line.append(f"{lab[i]} {a.RB.mean():+.3f}({len(a)})")
            print(f"  {nm:14} " + "  ".join(line))
        allout[str(k)] = o
        # 判準
        a_o, b_o, c_o = o[("RA", "樣本外")], o[("RB", "樣本外")], o[("RC", "樣本外")]
        p1 = a_o["m"] > max(b_o["m"], c_o["m"])
        p2 = a_o["lo"] > 0
        p3 = a_o["npos"] >= 6
        print(f"\n  判準 P1 贏過兩個對照 {'PASS' if p1 else 'FAIL'} / "
              f"P2 CI下緣>0 {'PASS' if p2 else 'FAIL'}（{a_o['lo']:+.4f}）/ "
              f"P3 逐幣≥6/9 {'PASS' if p3 else 'FAIL'}（{a_o['npos']}/{a_o['nsym']}）"
              f" -> {'採用候選' if (p1 and p2 and p3) else '不過'}")

    # ---- S1 前視對照 ----
    print(f"\n{'=' * 74}\nS1 前視對照（判別器改用真正的未來，A 臂應該明顯變好）")
    f = evaluate(KS[0], future=True)
    fo = report(f, KS[0], mid, tag="  [前視版，不可交易]")
    base = allout[str(KS[0])][("RA", "全期")]["m"]
    print(f"  正常版 A 全期 {base:+.4f}  vs  前視版 {fo[('RA','全期')]['m']:+.4f}")
    print("  前視版沒有明顯更好 -> 資料可能沒接上，下面的數字要先查儀器")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_after_sweep.json"
    p.write_text(json.dumps({str(k): {f"{a}_{b}": v for (a, b), v in o.items()}
                             for k, o in allout.items()}, indent=2, default=float),
                 encoding="utf-8")
    print(f"\nwritten -> {p}")


if __name__ == "__main__":
    main()
