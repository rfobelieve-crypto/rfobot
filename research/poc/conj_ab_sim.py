# -*- coding: utf-8 -*-
"""A/B 對照（方向資訊）＋ 1000U × 10x 資金曲線模擬

使用者 2026-09-09：「我分成兩個對照組一個有 coinglass，一個是現在的」
「跑一個模擬用 1000u 10 倍槓桿來做模擬測試」。

**先更正一件事**：Coinglass 的 parquet 是 **BTC-only、1 小時、4000 根**
（那是 V7 用的），對九幣分鐘級策略沒用。真正可用的同類資訊在
`research/poc/data/oi/`：**九幣、5 分鐘、2024-02 至今完整**，欄位包含
未平倉量、大戶帳戶/持倉多空比、全體帳戶多空比、主動買賣量比。
B 組用它，並在報告裡如實稱它為「Binance 期貨部位資料」而不是 Coinglass。

**因果性**：這份資料的一列不是 create_time 當下的快照，帶著整個 5 分鐘
區間的資訊（`event_census` 檔頭記載過）。所以一律只用
`create_time <= t − 5min` 的列 —— 與凍結定義同一條規矩。

===========================================================================
A/B 兩組（唯一差異是**方向判定**；事件、進場、出場、成本全部相同）
===========================================================================
    A   現行：impulse = sign(close(ready) − close(ready−5))
    B1  強平濾網：A 的方向，但**只在 ΔOI < 0 時交易**
        （OI 下降 = 有人在平倉/被強平；OI 上升 = 新開倉，不是強平）
    B2  大戶多空比定方向：sign(−Δ 大戶持倉多空比)
        （大戶多空比下降 = 大戶在減多或加空 -> 看空）
    B3  A ∧ B2 同向才進
    B4  強平濾網 ∧ 大戶同向（B1 ∧ B3）

判準（跑之前寫死）
    AB1 全格報告命中率／毛／淨／日聚類 CI／逐幣，不挑格
    AB2 過閘 = 樣本外淨值 CI 下緣 > 0 且逐幣 ≥6/9
    AB3 **命中率才是重點**：A 是 47.6%。若 B 沒把命中率抬起來，
        那 Coinglass/OI 這條資訊路就是無效的，不得用淨值的小幅改善蓋過去
    AB4 樣本外（後半）與樣本內（前半）分開報，樣本內不得單獨引用

===========================================================================
資金模擬（使用者指定 1000U × 10x）
===========================================================================
逐筆依真實時間排序、槽位管理、複利。單筆名目 = 權益 × 槓桿 / 槽數，
所以總曝險 ≤ 權益 × 槓桿。以每筆的真實 ATR% 換算價格變動 -> 損益。

**強平判定**：單筆最大不利幅度 MAE 若超過 1/槓桿（10x -> 10%），
在停損之前就先被強平，該筆損失整個保證金。這是 10 倍槓桿真正的風險，
不模擬它就是在畫一條假的資金曲線。

同時跑 2x（本專案現行規則 NOTIONAL_LEV_MULT）當對照——
CLAUDE.md 寫死「有效槓桿 2x 是 hard cap」，10x 是使用者指定的模擬情境，
**模擬不等於建議**。
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
import conj_causal as cc  # noqa: E402
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
W5, DELAY, HOLD, STOP = 5, 3, 480, 3.0
MIN_MS = 60_000
LEGS = (2.0, 2.0, 6.0)
ARMS = ("A", "B1", "B2", "B3", "B4")
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


def oi_features(sym, ts):
    """回傳 (dOI%, d大戶持倉多空比) —— 一律只用 create_time <= t−5min 的列。"""
    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest",
                                 "sum_toptrader_long_short_ratio"])
    oms = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10 ** 6).to_numpy()
    oiv = o["sum_open_interest"].to_numpy(float)
    tlr = o["sum_toptrader_long_short_ratio"].to_numpy(float)
    cut = ts - 5 * MIN_MS                      # 因果切點，與 event_census 同
    j_hi = np.searchsorted(oms, cut, side="right") - 1
    j_lo = np.searchsorted(oms, cut - W5 * MIN_MS, side="right") - 1
    ok = (j_lo >= 0) & (j_hi > j_lo)
    d_oi = np.full(len(ts), np.nan)
    d_lr = np.full(len(ts), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_oi[ok] = (oiv[j_hi[ok]] - oiv[j_lo[ok]]) / oiv[j_lo[ok]] * 100
        d_lr[ok] = tlr[j_hi[ok]] - tlr[j_lo[ok]]
    return d_oi, d_lr


def build():
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
        day = ts // 86_400_000
        d_oi, d_lr = oi_features(sym, ts)
        cand, _t, _c, _a, _d, q = ec.detect_all(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        caus = cc.causal_flags(q, day)
        pr = [(int(m), "sweep")
              for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in ("delta_ext", "vol_burst"):
            v = caus.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pr.append((int(m), nm))
        for a, mem in cr.groups_with_members(pr):
            sg = {t for _, t in mem}
            if "sweep" not in sg or not (sg & {"delta_ext", "vol_burst"}):
                continue
            if not ({"delta_ext", "vol_burst"} <= sg):
                continue                                   # 只做 S+D+V
            rd = max(min(m for m, t in mem if t == "sweep"),
                     min(m for m, t in mem if t in ("delta_ext", "vol_burst")))
            if rd < W5 or rd + DELAY + HOLD >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            dA = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
            doi, dlr = d_oi[rd], d_lr[rd]
            dB2 = float(np.sign(-dlr)) if np.isfinite(dlr) and dlr != 0 else np.nan
            j0 = rd + DELAY
            ent = float(op[j0])
            end = j0 + HOLD
            r = dict(sym=sym, ts=int(ts[j0]), entry=ent, atr=A,
                     atr_pct=A / ent,
                     day=pd.Timestamp(int(ts[rd]), unit="ms",
                                      tz="UTC").strftime("%Y-%m-%d"),
                     d0=int(ts[rd] // 86_400_000), doi=doi, dlr=dlr,
                     dA=dA, dB2=dB2)
            for arm in ARMS:
                if arm == "A":
                    dv, ok = dA, True
                elif arm == "B1":
                    dv, ok = dA, bool(np.isfinite(doi) and doi < 0)
                elif arm == "B2":
                    dv, ok = dB2, bool(np.isfinite(dB2))
                elif arm == "B3":
                    dv = dA
                    ok = bool(np.isfinite(dB2) and dB2 == dA)
                else:
                    dv = dA
                    ok = bool(np.isfinite(doi) and doi < 0 and
                              np.isfinite(dB2) and dB2 == dA)
                if not ok or not np.isfinite(dv):
                    r[f"R_{arm}"] = np.nan
                    r[f"st_{arm}"] = False
                    r[f"mae_{arm}"] = np.nan
                    r[f"ex_{arm}"] = np.nan
                    continue
                adv = ((ent - lo[j0 + 1:end + 1]) if dv > 0
                       else (hi[j0 + 1:end + 1] - ent)) / A
                k = np.flatnonzero(adv >= STOP)
                st = bool(len(k))
                mae = float(np.max(adv[:k[0] + 1])) if st else float(np.max(adv))
                r[f"R_{arm}"] = -STOP if st else float(dv * (cl[end] - ent) / A)
                r[f"st_{arm}"] = st
                r[f"mae_{arm}"] = mae
                r[f"ex_{arm}"] = int(ts[j0 + 1 + int(k[0])]) if st else int(ts[end])
            rows.append(r)
    return pd.DataFrame(rows)


def net_of(sub, arm):
    g = sub[f"R_{arm}"].to_numpy(float)
    st = sub[f"st_{arm}"].to_numpy(bool)
    return g - (LEGS[0] + np.where(st, LEGS[2], LEGS[1])) / 1e4 / sub.atr_pct.to_numpy()


def simulate(tr, arm, equity0=1000.0, lev=10.0, slots=3):
    """逐筆時間序、槽位、複利、含強平判定。"""
    t = tr[np.isfinite(tr[f"R_{arm}"])].sort_values("ts")
    eq = equity0
    free = [0] * slots
    curve, peak, mdd, nliq, ntr = [], equity0, 0.0, 0, 0
    for _, r in t.iterrows():
        k = int(np.argmin(free))
        if r.ts < free[k]:
            continue                                   # 槽全滿，跳過
        notion = eq * lev / slots
        margin = notion / lev
        # 強平：不利幅度超過 1/槓桿就先被清算，拿不到停損
        mae_pct = r[f"mae_{arm}"] * r.atr_pct
        if mae_pct >= 1.0 / lev:
            pnl = -margin
            nliq += 1
        else:
            pnl = notion * r[f"R_{arm}"] * r.atr_pct
        fee = notion * (LEGS[0] + (LEGS[2] if r[f"st_{arm}"] else LEGS[1])) / 1e4
        eq += pnl - fee
        ntr += 1
        free[k] = int(r[f"ex_{arm}"])
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)
        curve.append((int(r.ts), eq))
        if eq <= 0:
            break
    return dict(final=eq, ret_pct=(eq / equity0 - 1) * 100, mdd=mdd,
                n=ntr, n_liq=nliq, ruin=bool(eq <= 0), curve=curve)


def main():
    d = build()
    res = {}
    print(f"=== 母體 S+D+V {len(d):,} 筆（停損 {STOP} ATR／持有 {HOLD} 分）===")
    print(f"  OI 可得 {np.isfinite(d.doi).mean()*100:.1f}%   "
          f"大戶多空比可得 {np.isfinite(d.dlr).mean()*100:.1f}%")
    print(f"  ΔOI < 0（強平特徵）佔 {np.mean(d.doi < 0)*100:.1f}%   "
          f"大戶方向與價格方向一致 {np.nanmean(d.dB2 == d.dA)*100:.1f}%")
    print()
    d0, d1 = int(d.d0.min()), int(d.d0.max())
    mid = d0 + (d1 - d0) // 2
    for lab, sub in (("全期", d), ("前半（樣本內）", d[d.d0 <= mid - 1]),
                     ("後半（樣本外）", d[d.d0 > mid])):
        print(f"=== {lab} ===")
        print(f"{'臂':>4s} {'n':>6s} {'命中率':>7s} {'毛':>9s} {'淨':>9s} "
              f"{'CI下':>9s} {'幣+':>5s}")
        for arm in ARMS:
            s = sub[np.isfinite(sub[f"R_{arm}"])]
            if len(s) < 50:
                print(f"{arm:>4s} {len(s):6d}   樣本不足")
                continue
            nt = net_of(s, arm)
            mn, ln = day_ci(nt, s.day.to_numpy())
            per = s.assign(x=nt).groupby("sym").x.mean()
            hit = float(np.mean(s[f"R_{arm}"] > 0))
            print(f"{arm:>4s} {len(s):6d} {hit*100:6.1f}% "
                  f"{float(np.mean(s[f'R_{arm}'])):+9.4f} {mn:+9.4f} {ln:+9.4f} "
                  f"{int((per>0).sum()):4d}/9"
                  + ("  *" if ln > 0 and (per > 0).sum() >= 6 else ""))
            res.setdefault(lab, {})[arm] = dict(n=int(len(s)), hit=hit,
                                                net=mn, net_lo=ln,
                                                coins=int((per > 0).sum()))
        print()

    print("=== 資金曲線模擬：1000 USDT ===")
    print(f"{'臂':>4s} {'槓桿':>5s} {'槽':>3s} {'筆數':>6s} {'期末':>11s} "
          f"{'報酬':>10s} {'最大回落':>9s} {'強平次數':>8s}")
    for arm in ARMS:
        for lev, slots in ((10.0, 3), (10.0, 1), (2.0, 3)):
            s = simulate(d, arm, 1000.0, lev, slots)
            if s["n"] < 20:
                continue
            print(f"{arm:>4s} {lev:4.0f}x {slots:3d} {s['n']:6d} "
                  f"{s['final']:11,.0f} {s['ret_pct']:+9.1f}% {s['mdd']:8.1f}% "
                  f"{s['n_liq']:8d}" + ("   **爆倉**" if s["ruin"] else ""))
            res.setdefault("sim", {})[f"{arm}_lev{lev:.0f}_slot{slots}"] = {
                k: v for k, v in s.items() if k != "curve"}
    print()
    print("  強平判定：單筆最大不利幅度 ≥ 1/槓桿（10x -> 10% 價格）即在停損前")
    print("  被清算，損失整個保證金。不模擬這件事就是畫一條假的資金曲線。")
    print(f"  參考：1 ATR ≈ {d.atr_pct.median()*100:.3f}% 價格，"
          f"停損 {STOP} ATR ≈ {d.atr_pct.median()*STOP*100:.2f}%，"
          f"10x 下 = 權益的 {d.atr_pct.median()*STOP*10*100:.1f}%／筆")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_ab_sim.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print()
    print("written ->", OUT / "conj_ab_sim.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
