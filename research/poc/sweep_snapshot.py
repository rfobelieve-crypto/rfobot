# -*- coding: utf-8 -*-
"""每一筆掃單的訂單流快照表（2026-09-10 建，基礎建設非研究）

使用者 2026-09-10：「不能針對每一次觸發 sweep 的時候把他們的訂單流值都
記起來然後分析好幾百筆，什麼訂單流組合對我的 sweep 事件最有利嗎」

前半句該做，而且獨立於後半句的統計爭議：現在每測一個假設都要從頭重算
事件、重抓因子，一輪十幾分鐘。這張表把所有拿得到的值一次算好，之後任何
假設都是查表。

===========================================================================
欄位命名就是前視防護
===========================================================================
    pre_*       事件當下**已經知道**的（只用 t 或更早的資料）
    post{K}_*   事件後 K 分鐘才知道的 —— 要用它就必須把進場延到 t+K
    y_*         結果（標籤），任何進場決策都不可以讀它

這條線 2026-09-09 才被前視咬過一次（§1.03b：錨點用群內最早那一分鐘，
22.3% 的單下在事件成立之前，+0.5064 誠實化後只剩 +0.1094）。所以不靠
記憶防前視，靠命名 —— 看到 post/y 開頭就知道它不能出現在進場條件裡。

===========================================================================
每一列 = 一筆掃單（冷卻後）
===========================================================================
身分     sym / ts / side / level / level_age_d / is_sdv / sig
情境     pre_atr / pre_close / pre_mom5（成立前 5 分鐘動能，現行方向規則）
流量     {pre,post5,post10,post15}_{vol,delta,ntrades,avgsize}
         vol=成交量  delta=主動買賣差額  ntrades=成交筆數  avgsize=vol/ntrades
倉位     pre_oi / pre_ls_retail / pre_ls_top_acct / pre_ls_top_pos / pre_taker_ls
         post{5,10,15}_ 同五項的**變化率**
標籤     y_with_d{0,5,10,15} / y_against_d{...}   順勢／逆勢，各延遲進場一次
         y_mfe / y_mae                            進場後最大有利／不利（ATR）

倉位五項的來源是 data/oi 的 5 分鐘表：
    sum_open_interest                 未平倉量
    count_long_short_ratio            全體帳戶多空比（帳戶數加權 ≈ 散戶）
    count_toptrader_long_short_ratio  大戶帳戶多空比
    sum_toptrader_long_short_ratio    大戶持倉多空比
    sum_taker_long_short_vol_ratio    主動買賣量比
後三項**這條線從未用過**；大戶與散戶的差值正是「賺散戶止損的錢」這個
前提的直接度量。

用法
    python research/poc/sweep_snapshot.py            # 建表
    python research/poc/sweep_snapshot.py --check    # 只跑正確性檢查
出：research/poc/data/sweep_snapshot.parquet
"""
from __future__ import annotations

import argparse
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

OUT = HERE / "data" / "sweep_snapshot.parquet"
OI_DIR = HERE / "data" / "oi"
WINS = (5, 10, 15)
DELAYS = (0, 5, 10, 15)
HOLD, STOP, W = cb.HOLD, cb.STOP, cb.W
OI_COLS = [("sum_open_interest", "oi"),
           ("count_long_short_ratio", "ls_retail"),
           ("count_toptrader_long_short_ratio", "ls_top_acct"),
           ("sum_toptrader_long_short_ratio", "ls_top_pos"),
           ("sum_taker_long_short_vol_ratio", "taker_ls")]


def load_oi(sym):
    d = pd.read_parquet(OI_DIR / f"{sym}.parquet",
                        columns=["create_time"] + [c for c, _ in OI_COLS])
    t = (pd.to_datetime(d["create_time"]).astype("int64") // 10 ** 6).to_numpy()
    o = np.argsort(t)
    return t[o], {short: d[c].to_numpy(float)[o] for c, short in OI_COLS}


def _rate(a, b):
    return np.nan if (not np.isfinite(a) or not np.isfinite(b) or b == 0) else (a - b) / abs(b)


def build(sym):
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
    oi_t, oi_c = load_oi(sym)

    ev = pd.read_parquet(cb.EVENTS / f"{sym}.parquet",
                         columns=["level_id", "t_sweep", "side", "sweep_lvl"]
                         ).sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    lv = pd.read_parquet(cb.LEVELS / f"{sym}.parquet",
                         columns=["level_id", "hour_ts"]).set_index("level_id")

    # SDV 事件的成立分鐘（用來標 is_sdv）
    pairs = [(int(m), "sweep") for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in cb.FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            pairs += [(int(m), nm) for m in ec.cooldown_filter(np.sort(v))]
    sdv_sweep = {}
    for _a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig:
            continue
        m_sw = min(m for m, t in mem if t == "sweep")
        sdv_sweep[m_sw] = "+".join(sorted(sig))

    rows = []
    for m in ec.cooldown_filter(np.sort(cand["sweep"])):
        m = int(m)
        if m < W + 10 or m + max(DELAYS) + 1 + HOLD >= n:
            continue
        A = float(at[m])
        if not np.isfinite(A) or A <= 0:
            continue
        t_ms = int(ts[m]) + 60_000                 # 掃單那分鐘的收盤
        i = int(np.searchsorted(ev_ts, t_ms))
        if i >= len(ev_ts) or ev_ts[i] != t_ms:
            continue
        r_ev = ev.iloc[i]
        side = str(r_ev["side"])
        d_s = 1.0 if side == "buyside" else -1.0
        lid = r_ev["level_id"]
        age = ((t_ms - int(lv.loc[lid, "hour_ts"])) / 86_400_000
               if lid in lv.index else np.nan)

        row = dict(sym=sym, ts=int(ts[m]), side=side,
                   level=float(r_ev["sweep_lvl"]), level_age_d=age,
                   is_sdv=bool("delta_ext" in sdv_sweep.get(m, "")
                               and "vol_burst" in sdv_sweep.get(m, "")),
                   sig=sdv_sweep.get(m, "sweep"),
                   pre_atr=A, pre_close=float(cl[m]),
                   pre_mom5=float((cl[m] - cl[m - W]) / A))

        # ---- 流量：事件前 10 分鐘（已知） + 事件後各窗（要延後才知）----
        seg = slice(m - 10, m + 1)
        v0, t0 = vol[seg].sum(), ntr[seg].sum()
        row.update(pre_vol=float(v0), pre_delta=float(dlt[seg].sum()),
                   pre_ntrades=float(t0),
                   pre_avgsize=float(v0 / t0) if t0 > 0 else np.nan)
        for k in WINS:
            sg = slice(m, m + k + 1)
            v1, t1 = vol[sg].sum(), ntr[sg].sum()
            row.update({f"post{k}_vol": float(v1),
                        f"post{k}_delta": float(dlt[sg].sum()),
                        f"post{k}_ntrades": float(t1),
                        f"post{k}_avgsize": float(v1 / t1) if t1 > 0 else np.nan})

        # ---- 倉位：base 取 create_time <= 事件時刻的最後一筆 ----
        base = int(np.searchsorted(oi_t, t_ms, side="right")) - 1
        if base < 0:
            continue
        for _, short in OI_COLS:
            row[f"pre_{short}"] = float(oi_c[short][base])
        for k in WINS:
            end = int(np.searchsorted(oi_t, int(ts[m + k]), side="right")) - 1
            for _, short in OI_COLS:
                row[f"post{k}_{short}"] = (_rate(oi_c[short][end], oi_c[short][base])
                                           if end > base else np.nan)

        # ---- 標籤 ----
        for dly in DELAYS:
            e = m + dly + 1
            ent = float(op[e])
            end_i = e + HOLD
            for sgn, nm2 in ((d_s, "with"), (-d_s, "against")):
                adv = ((ent - lo[e + 1:end_i + 1]) if sgn > 0
                       else (hi[e + 1:end_i + 1] - ent)) / A
                if len(np.flatnonzero(adv >= STOP)):
                    R, stopped = -STOP, True
                else:
                    R, stopped = float(sgn * (cl[end_i] - ent) / A), False
                leg = cb.COST_ENTRY + (cb.COST_STOP if stopped else cb.COST_TIME)
                row[f"y_{nm2}_d{dly}"] = R - leg / 1e4 * ent / A
                if nm2 == "with":
                    row[f"y_stopped_d{dly}"] = bool(stopped)
            if dly == 0:
                fav = ((hi[e + 1:end_i + 1] - ent) if d_s > 0
                       else (ent - lo[e + 1:end_i + 1])) / A
                unf = ((ent - lo[e + 1:end_i + 1]) if d_s > 0
                       else (hi[e + 1:end_i + 1] - ent)) / A
                row["y_mfe"] = float(np.nanmax(fav)) if len(fav) else np.nan
                row["y_mae"] = float(np.nanmax(unf)) if len(unf) else np.nan
        rows.append(row)
    return rows


def checks(d):
    """正確性檢查 —— 不是判準，是「這張表有沒有建錯」。"""
    print("\n" + "=" * 72)
    print("正確性檢查")
    ok = True

    # C1 SDV 子集的順勢報酬（延遲 3 不在網格裡，用 d0 與既有值比量級）
    s = d[d.is_sdv]
    print(f"  C1 SDV 子集 n={len(s):,}（既有 ledger 為 1,584）"
          f"  順勢 d0 每筆 {s.y_with_d0.mean():+.4f}")
    if abs(len(s) - 1584) > 60:
        print("     ** 筆數與 ledger 差太多，母體可能串錯 **")
        ok = False

    # C2 前視防護：post 欄位不得與 pre 欄位完全相同（那代表窗沒推進去）
    same = [k for k in ("vol", "delta", "ntrades")
            if np.allclose(d[f"post5_{k}"].fillna(0), d[f"pre_{k}"].fillna(0))]
    print(f"  C2 post5 與 pre 是否雷同：{same if same else '無（正常）'}")
    ok &= not same

    # C3 停損有沒有真的生效。
    #
    # 第一版寫成「順勢+逆勢 的平均應該為正（停損截斷左尾）」—— 那個性質
    # 是對的，但**只對毛報酬成立**；表裡的 y_* 是扣成本後的，兩腿各扣一次
    # 就被拉回零附近（實測 −0.0013，而毛的是 +0.0522）。檢查自己寫錯了，
    # 不是表寫錯（factor-research 最後一條：自己剛寫的儀器最危險）。
    # 改成直接看停損率 —— 現行規格量過是 18~19%。
    sr = float(d.y_stopped_d0.mean())
    print(f"  C3 停損率 {sr*100:.1f}%（現行規格量過 18~19%）")
    if not (0.10 <= sr <= 0.30):
        print("     ** 停損率離譜，停損邏輯可能沒生效 **")
        ok = False

    # C4 倉位欄位的覆蓋率
    cov = {short: float(d[f"post10_{short}"].notna().mean())
           for _, short in OI_COLS}
    print(f"  C4 倉位欄位 post10 覆蓋率：" +
          "  ".join(f"{k} {v*100:.0f}%" for k, v in cov.items()))
    if min(cov.values()) < 0.8:
        print("     ** 有欄位覆蓋率低於 80%，OI 表可能有缺口 **")
        ok = False

    print(f"  -> {'全部通過' if ok else '**有檢查沒過，先修表再用**'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check and OUT.exists():
        checks(pd.read_parquet(OUT))
        return
    rows = []
    for s in cb.CORE9:
        r = build(s)
        rows += r
        print(f"  {s:5} {len(r):6,} 筆掃單")
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)
    print(f"\n{len(d):,} 列 × {len(d.columns)} 欄  -> {OUT}")
    print(f"  其中 SDV {int(d.is_sdv.sum()):,} 筆"
          f"（{d.is_sdv.mean()*100:.1f}%）")
    checks(d)


if __name__ == "__main__":
    main()
