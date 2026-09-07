# -*- coding: utf-8 -*-
"""橋樑檢定 — 我們標的「價格幾何」事件，伴不伴隨真實的被迫成交？

使用者 2026-09-06 提出的未確認前提：
    系統標的是**價格軌跡的幾何**（影線刺穿前低、收不收回）。
    機制假設講的是**成交的來源**（停損被觸發、清算引擎被觸發）。
    兩者不是同一件事，中間的橋樑從來沒被驗證過。

    如果一半的事件沒有實質被迫成交，那 POC 距離的解釋力當然被稀釋
    ——不是假設錯，是母體混了。而 OI accel 跑出平坦結果時，
    「機制不成立」與「事件裡真清算佔比太低」在現行設計下分不出來。

五個欄位，掃單當下 ±5 分鐘：
    volume_spike  該窗成交量 / 前 30 日同時段同長度窗的中位數
    oi_drop_pct   該窗未平倉量變化（%），負值 = 部位被銷毀
    liq_usd       該窗清算名目（只有 BTC/ETH、159 天、且完整率 23.6%）
    delta_sweep   該窗主動成交淨額，**依掃單方向取號**（>0 = 攻擊方向與
                  穿越方向一致）。taker 標記是撮合引擎原生的,不是 tick rule
                  推論:2026-09-06 對 aggTrades is_buyer_maker 驗過 1,440 分鐘,
                  最大相對誤差 4.8e-16、相關 1.0000000000。
    absorption    （報告時**不翻號**：AUC>0.5 代表事件的 λ 比隨機時刻**高**，
                  也就是 book 更薄、吸收更**弱**）
                  該窗「每單位主動成交量推動多少價格」= 逐分鐘 Δprice 對
                  delta 的**迴歸斜率**（Kyle λ），再乘 std(delta)/ATR 無量綱化。
                  **低 = 吸收強**（大量主動單打進去價格不太動 = 有人在接）。
                  用斜率不用 |Δp|/|delta| 比值:比值在 delta→0 時炸掉,而
                  delta 過零正是買賣勢均力敵——有意義的狀態,不是雜訊。
                  比值版並列為敏感度（absorption_ratio）。

**這是刻畫不是特徵。** 窗口含 t_sweep 之後的資料，所以它回答「這個事件
當時發生了什麼」，不能直接拿來當濾網——真要當濾網必須改成只用事前資訊，
否則是前視。

**兩個對照組，因為第一個有日層級的混淆**（使用者 2026-09-07 指出）：

  XDAY     同幣、同一個 UTC 小時、**不同天**。控制了幣別與日內節律，
           **沒有控制日**。而事件已知集中在高波動日（6 天供 97% 的變異，
           §1.00 Stage 5），所以事件抽自波動日、對照抽自全部日——五個欄位
           的判別度裡都混了「波動日 vs 平常日」。
  SAMEDAY  同幣、**同一天**、非事件分鐘（避開任何事件 ±30 分鐘，免得對照
           被事件餘波污染）。日層級的東西全部被消掉，剩下的才是事件本身。

  **兩個都報，差額就是日層級混淆的大小**——那個數字本身要記錄。
  分離度用 AUC：0.5 = 掃單事件跟對照在該欄位上完全一樣。

資料覆蓋（2026-09-06 量過，寫在這裡免得結果被誤讀）
    oi_drop_pct   九幣、全歷史、5 分鐘粒度            完整
    volume_spike  九幣、全歷史、1 分鐘                完整
    liq_usd       BTC/ETH、2026-03-31 起、**完整率 23.6%**，
                  且漏失隨強度惡化（大時段 2.3%）——只能當交叉檢查，
                  不能當主證據。用它檢驗前兩個是不是被迫流的好代理。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
EVENTS = HERE / "data" / "events"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results"
MIN_MS = 60_000
WIN_MIN = 5                     # +-5 minutes
BASE_DAYS = 30
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
RNG = np.random.default_rng(20260906)
GUARD_MIN = 30          # SAMEDAY 對照必須離任何事件 >30 分鐘


def window_volume(cum, pos, t, ts0):
    """Sum of 1m volume over [t-5min, t+5min], via a prefix-sum lookup."""
    lo = (t - WIN_MIN * MIN_MS - ts0) // MIN_MS
    hi = (t + WIN_MIN * MIN_MS - ts0) // MIN_MS
    lo = np.clip(lo, 0, len(cum) - 1)
    hi = np.clip(hi, 0, len(cum) - 1)
    return cum[hi] - cum[lo]


def build_for(sym, anchors, kd=None):
    """anchors: int64 ms array.  kd: +1 buyside / -1 sellside（決定 delta 取號）。"""
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "volume", "close", "delta", "atr_h14"])
    ts0 = int(b["ts"].iloc[0])
    v = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    cum = np.concatenate([[0.0], np.cumsum(v)])

    win = window_volume(cum, None, anchors, ts0)
    # baseline: the same 11-minute window on each of the previous 30 days
    base = np.empty((BASE_DAYS, len(anchors)))
    for k in range(1, BASE_DAYS + 1):
        base[k - 1] = window_volume(cum, None, anchors - k * 86_400_000, ts0)
    med = np.median(base, axis=0)
    spike = np.where(med > 0, win / med, np.nan)

    # --- delta 與 absorption（Kyle λ）------------------------------------
    ts = b["ts"].to_numpy(np.int64)
    dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
    cl = b["close"].to_numpy(float)
    atr = b.set_index("ts")["atr_h14"]
    cdl = np.concatenate([[0.0], np.cumsum(dl)])
    lo_i = np.clip((anchors - WIN_MIN * MIN_MS - ts0) // MIN_MS, 0, len(cdl) - 1)
    hi_i = np.clip((anchors + WIN_MIN * MIN_MS - ts0) // MIN_MS, 0, len(cdl) - 1)
    dsum = cdl[hi_i] - cdl[lo_i]
    sign = 1.0 if kd is None else np.asarray(kd, float)
    a_at = anchors_atr = np.array([atr.get(int(t), np.nan) for t in anchors])

    lam = np.full(len(anchors), np.nan)
    rat = np.full(len(anchors), np.nan)
    for i in range(len(anchors)):
        lo, hi = int(lo_i[i]), int(hi_i[i])
        if hi - lo < 6 or not np.isfinite(a_at[i]) or a_at[i] <= 0:
            continue
        dp = np.diff(cl[lo:hi + 1])
        dd = dl[lo + 1:hi + 1]
        good = np.isfinite(dp) & np.isfinite(dd)
        if good.sum() < 6 or np.std(dd[good]) == 0:
            continue
        slope = np.polyfit(dd[good], dp[good], 1)[0]
        lam[i] = slope * np.std(dd[good]) / a_at[i]
        tot = np.abs(dd[good]).sum()
        rat[i] = (abs(cl[hi] - cl[lo]) / tot / a_at[i]) if tot > 0 else np.nan

    o = pd.read_parquet(OI / f"{sym}.parquet",
                        columns=["create_time", "sum_open_interest"])
    o["ms"] = (pd.to_datetime(o["create_time"], utc=True).astype("int64") // 10**6)
    ots = o["ms"].to_numpy(np.int64)
    oi = o["sum_open_interest"].to_numpy(float)
    i_lo = np.searchsorted(ots, anchors - WIN_MIN * MIN_MS, side="right") - 1
    i_hi = np.searchsorted(ots, anchors + WIN_MIN * MIN_MS, side="right") - 1
    ok = (i_lo >= 0) & (i_hi >= 0) & (i_hi > i_lo)
    oi_pct = np.full(len(anchors), np.nan)
    oi_pct[ok] = (oi[i_hi[ok]] - oi[i_lo[ok]]) / oi[i_lo[ok]] * 100.0
    return pd.DataFrame(dict(sym=sym, anchor=anchors, win_volume=win,
                             volume_spike=spike, oi_drop_pct=oi_pct,
                             delta_sweep=sign * dsum, absorption=lam,
                             absorption_ratio=rat))


def liq_for(anchors_by_sym):
    """liq_total_usd in the window.  BTC/ETH only, 2026-03-31 onward."""
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    c = get_db_conn()
    d = pd.read_sql("SELECT canonical_symbol s, window_start w, liq_total_usd u "
                    "FROM liquidation_1m", c)
    c.close()
    d["sym"] = d["s"].str.replace("-USD", "", regex=False)
    out = {}
    for sym, anchors in anchors_by_sym.items():
        g = d[d.sym == sym]
        if g.empty:
            continue
        w = g["w"].to_numpy(np.int64)
        u = g["u"].to_numpy(float)
        order = np.argsort(w)
        w, u = w[order], u[order]
        cum = np.concatenate([[0.0], np.cumsum(u)])
        lo = np.searchsorted(w, anchors - WIN_MIN * MIN_MS, side="left")
        hi = np.searchsorted(w, anchors + WIN_MIN * MIN_MS, side="right")
        covered = (anchors >= w.min() + 86_400_000) & (anchors <= w.max())
        vals = np.where(covered, cum[hi] - cum[lo], np.nan)
        out[sym] = vals
    return out


def auc(pos, neg):
    """P(random event > random control), NaN-safe."""
    p = pos[np.isfinite(pos)]
    n = neg[np.isfinite(neg)]
    if len(p) < 20 or len(n) < 20:
        return np.nan
    allv = np.concatenate([p, n])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=",".join(CORE9))
    a = ap.parse_args()
    syms = [s for s in (x.strip().upper() for x in a.syms.split(",")) if s
            and (OI / f"{s}.parquet").exists()]
    print("coins with OI on disk:", ", ".join(syms), "\n")

    ev_frames, ct_frames, anchors_ev, anchors_ct = [], [], {}, {}
    sameday_frames, sameday_miss, anchors_sd = [], 0, {}
    for s in syms:
        ev = pd.read_parquet(EVENTS / f"{s}.parquet", columns=["t_sweep", "side"])
        t = ev["t_sweep"].to_numpy(np.int64)
        # control: same coin, same UTC hour-of-day, a different day
        offs = RNG.integers(1, 300, len(t)) * 86_400_000 * RNG.choice([-1, 1], len(t))
        lo = pd.read_parquet(BARS / f"{s}.parquet", columns=["ts"])["ts"]
        c = np.clip(t + offs, int(lo.iloc[0]) + 31 * 86_400_000, int(lo.iloc[-1]))
        anchors_ev[s], anchors_ct[s] = t, c
        kd = np.where(ev["side"].to_numpy() == "buyside", 1.0, -1.0)
        e = build_for(s, t, kd)
        e["side"] = ev["side"].to_numpy()
        e["kind"] = "event"
        k = build_for(s, c, kd)
        k["side"] = ev["side"].to_numpy()
        k["kind"] = "control"
        # --- SAMEDAY：同幣、同一天、離任何事件 >GUARD 分鐘的分鐘 ---------
        day0 = (t // 86_400_000) * 86_400_000
        sd = np.full(len(t), -1, dtype=np.int64)
        ev_by_day = {}
        for x in t:
            ev_by_day.setdefault(int((x // 86_400_000) * 86_400_000), []).append(int(x))
        miss = 0
        for i, d0 in enumerate(day0):
            same = np.array(ev_by_day[int(d0)], dtype=np.int64)
            cand = d0 + np.arange(0, 1440) * MIN_MS
            far = (np.abs(cand[:, None] - same[None, :]).min(axis=1)
                   > GUARD_MIN * MIN_MS)
            cand = cand[far]
            if len(cand) == 0:
                miss += 1
                continue
            sd[i] = cand[RNG.integers(0, len(cand))]
        ok = sd > 0
        sdd = build_for(s, sd[ok], kd[ok])
        sdd["side"] = ev["side"].to_numpy()[ok]
        sdd["kind"] = "control_sameday"
        anchors_sd[s] = sd[ok]
        sameday_frames.append(sdd)
        sameday_miss += miss
        ev_frames.append(e)
        ct_frames.append(k)

    liq_e = liq_for(anchors_ev)
    liq_c = liq_for(anchors_ct)
    liq_s = liq_for(anchors_sd)
    for f in ev_frames:
        s = f["sym"].iloc[0]
        f["liq_usd"] = liq_e.get(s, np.full(len(f), np.nan))
    for f in ct_frames:
        s = f["sym"].iloc[0]
        f["liq_usd"] = liq_c.get(s, np.full(len(f), np.nan))
    for f in sameday_frames:
        s = f["sym"].iloc[0]
        f["liq_usd"] = liq_s.get(s, np.full(len(f), np.nan))
    E = pd.concat(ev_frames, ignore_index=True)
    C = pd.concat(ct_frames, ignore_index=True)
    S = pd.concat(sameday_frames, ignore_index=True)
    print(f"SAMEDAY 對照 n={len(S):,}（{sameday_miss} 筆事件當天找不到合格分鐘）")
    print()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([E, C] + sameday_frames, ignore_index=True).to_parquet(
        OUT / "bridge.parquet", index=False)

    res = {"n_events": int(len(E)), "n_controls": int(len(C)), "coins": syms}
    print(f"events={len(E):,}  controls={len(C):,}\n")
    print("=== 分布：事件 vs 對照（同幣、同 UTC 小時、不同天）===\n")
    print(f"{'欄位':14s} {'組':8s} {'n':>6s} {'q10':>9s} {'q25':>9s} "
          f"{'中位':>9s} {'q75':>9s} {'q90':>9s}")
    for col in ("volume_spike", "oi_drop_pct", "liq_usd", "delta_sweep",
                "absorption"):
        for name, D in (("event", E), ("control", C)):
            x = D[col].dropna()
            if len(x) < 20:
                print(f"{col:14s} {name:8s} {len(x):6d}   (資料不足)")
                continue
            q = x.quantile([.1, .25, .5, .75, .9])
            print(f"{col:14s} {name:8s} {len(x):6,d} " +
                  " ".join(f"{q.iloc[i]:9.3f}" for i in range(5)))
        print()

    print("=== 分離度 AUC（0.5 = 掃單事件與隨機時刻在該欄位上完全一樣）===\n")
    # oi_drop:部位銷毀是負值,取負號讓「更多銷毀」= 更大。
    # absorption(=Kyle λ)**不翻號**:λ 高 = 每單位主動量推動更多價格 = book 薄
    # = 吸收**弱**。第一版把它跟 oi_drop 一起翻,算術對但標籤反,會讓讀者
    # 把 0.186 讀成「事件吸收更強」——實際是 AUC 0.814「事件吸收更弱」。
    FLIP = {"oi_drop_pct"}
    for col in ("volume_spike", "oi_drop_pct", "liq_usd", "delta_sweep",
                "absorption"):
        pe = -E[col].to_numpy(float) if col in FLIP else E[col].to_numpy(float)
        pc = -C[col].to_numpy(float) if col in FLIP else C[col].to_numpy(float)
        ps = -S[col].to_numpy(float) if col in FLIP else S[col].to_numpy(float)
        v = auc(pe, pc)
        vs_ = auc(pe, ps)
        res[f"auc_{col}"] = v
        res[f"auc_{col}_sameday"] = vs_
        res[f"day_confound_{col}"] = (v - vs_) if np.isfinite(v) and np.isfinite(vs_) else None
        f1 = f"{v:9.4f}" if np.isfinite(v) else "      n/a"
        f2 = f"{vs_:11.4f}" if np.isfinite(vs_) else "        n/a"
        f3 = f"{v - vs_:+8.4f}" if np.isfinite(v) and np.isfinite(vs_) else "     n/a"
        print(f"  {col:14s} {f1} {f2} {f3}")
        for side in ("sellside", "buyside"):
            a1 = auc(pe[(E.side == side).to_numpy()], pc[(C.side == side).to_numpy()])
            a2 = auc(pe[(E.side == side).to_numpy()], ps[(S.side == side).to_numpy()])
            res[f"auc_{col}_{side}"] = a1
            res[f"auc_{col}_{side}_sameday"] = a2
            g1 = f"{a1:9.4f}" if np.isfinite(a1) else "      n/a"
            g2 = f"{a2:11.4f}" if np.isfinite(a2) else "        n/a"
            g3 = f"{a1 - a2:+8.4f}" if np.isfinite(a1) and np.isfinite(a2) else "     n/a"
            print(f"      {side:8s} {g1} {g2} {g3}")

    print("\n=== 「沒有被迫流足跡」的事件佔多少 ===\n")
    for th in (1.0, 1.25, 1.5, 2.0):
        f = float((E.volume_spike.dropna() < th).mean())
        fc = float((C.volume_spike.dropna() < th).mean())
        print(f"  volume_spike < {th:<5.2f}  事件 {f*100:5.1f}%   對照 {fc*100:5.1f}%")
        res[f"frac_spike_lt_{th}"] = f
    fe = float((E.oi_drop_pct.dropna() >= 0).mean())
    fc = float((C.oi_drop_pct.dropna() >= 0).mean())
    print(f"  oi_drop_pct >= 0（沒有部位銷毀）  事件 {fe*100:5.1f}%   對照 {fc*100:5.1f}%")
    res["frac_no_oi_drop"] = fe

    print("\n=== 代理效度：在看得到清算的地方，前兩欄追不追得上 liq_usd ===\n")
    sub = E.dropna(subset=["liq_usd"])
    if len(sub) > 100:
        from scipy.stats import spearmanr
        for col in ("volume_spike", "oi_drop_pct"):
            m = sub.dropna(subset=[col])
            x = -m[col] if col == "oi_drop_pct" else m[col]
            rho = spearmanr(x, m.liq_usd).correlation
            res[f"spearman_{col}_vs_liq"] = float(rho)
            print(f"  spearman({col}, liq_usd) = {rho:+.4f}   n={len(m):,}")
        print(f"  （覆蓋：{len(sub):,} / {len(E):,} 筆事件 = {len(sub)/len(E)*100:.1f}%）")
    else:
        print("  重疊樣本不足")

    (OUT / "bridge.json").write_text(json.dumps(res, indent=2, default=float),
                                     encoding="utf-8")
    print("\nwritten ->", OUT / "bridge.json")


if __name__ == "__main__":
    main()
