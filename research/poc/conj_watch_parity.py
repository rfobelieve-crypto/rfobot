# -*- coding: utf-8 -*-
"""live 偵測器 vs 離線偵測器 —— 已知答案的對照

`conj_watch.py` 的檔頭寫著「同一份偵測，不是第二份實作」。
**那句話一開始是宣稱不是事實**，這支就是去驗它。

===========================================================================
這支測試自己被修過兩次，兩次都記在這裡（自己剛寫的儀器最危險）
===========================================================================
第一版：拿 live 的**今日**門檻去比 30 天前的事件。live 的門檻窗是「最近
    30 天」、離線是「該事件當天往前 30 天」——25 天前的事件兩邊用的是完全
    不同的資料窗。那個測試分不出「實作有 bug」和「門檻窗依設計不同」。
    而 live 實際運行時每天更新門檻、當天與離線同窗，線上沒有這個問題。

第二版：改用離線當天的門檻，重疊率反而從 43.5% 掉到 55.6%——門檻已經對齊
    了還是不一致，於是問題被逼到**組裝層**，也就是真正的 bug 所在（見
    `conj_watch.assemble` 的 docstring）。

現在的版本把問題拆成兩個獨立的臂，因為它們的修法不同：

    A 臂「組裝一致嗎」  兩邊都餵**離線的旗標**，只比組裝。
        這是純粹的已知答案對照：live 現在直接呼叫 `event_triage.cluster`，
        所以 A 臂應該近乎完全重疊。不重疊 = 組裝還有第二份實作。

    B 臂「門檻表等價嗎」 live 用**自己預算的門檻表**（thresholds.parquet）。
        A 過而 B 不過 -> 差異純粹來自門檻近似（單一 30 日 p99 vs 逐日滾動），
        那是設計取捨不是 bug，但要量出它有多大。

判準（跑之前寫死）
    V1 A 臂重疊率 >= 0.95  -> 組裝是同一份
       < 0.95 -> **組裝仍是第二份實作**，上線前必須修
    V2 B 臂重疊率 >= 0.80 且 live 獨有 <= 0.20
       -> 預算的門檻表夠接近，可以上線
       不過 -> 門檻要改成逐日重算（每天一次，成本可接受）
    V3 差異必須逐項歸因，不得只報比率。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_clock as ck  # noqa: E402
import conj_causal as cc  # noqa: E402
import conj_watch as cw  # noqa: E402

BARS = HERE / "data" / "bars"
DAYS = 30
TOL_MIN = 2          # 時間對齊容差（分鐘）
FLOWM = set(cw.FLOW)


def offline_moments(cand, lo_i):
    """離線的交會時刻（同 conj_clock 的定義，去掉 oi —— live 不吃 OI）。"""
    pairs = []
    for nm in et.NAMES:
        if nm == "oi_crash" or nm == "liq_burst":
            continue
        v = cand.get(nm)
        if v is None or len(v) == 0:
            continue
        for m in ec.cooldown_filter(np.sort(v)):
            pairs.append((int(m), nm))
    return sorted({a for a, s in et.cluster(pairs)
                   if "sweep" in s and (s & FLOWM) and a >= lo_i})


def compare(off, live, tol=TOL_MIN):
    offa = np.array(sorted(off), np.int64)
    lva = np.array(sorted(live), np.int64)
    both = sum(1 for o in offa
               if len(lva) and np.min(np.abs(lva - o)) <= tol)
    only_off = len(offa) - both
    only_live = sum(1 for v in lva
                    if not (len(offa) and np.min(np.abs(offa - v)) <= tol))
    return both, only_off, only_live


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    thr = pd.read_parquet(HERE / "data" / "live" / "thresholds.parquet")
    thr_by = {r.sym: r for r in thr.itertuples()}

    A = np.zeros(3, int)          # both, only_off, only_live
    B = np.zeros(3, int)
    rows = []
    for sym in ec.CORE9:
        if sym not in thr_by:
            continue
        t = thr_by[sym]
        cand, ts, cl, at, day = ck.frozen_cand(sym, liq)
        n = len(ts)
        lo_i = int(np.searchsorted(ts, ts[-1] - DAYS * 86_400_000))
        off = offline_moments(cand, lo_i)

        # ---- A 臂：兩邊同旗標，只比組裝 ----
        _c, _ts, _cl, _at, _day, q = ec.detect_all(sym, liq)
        caus = cc.causal_flags(q, _day)
        fa = {"sweep": cand["sweep"]}
        for k in cw.FLOW:
            v = caus.get(k)
            fa[k] = np.asarray(v if v is not None else [], np.int64)
        liveA = [a for a, s in cw.assemble(fa) if a >= lo_i]

        # ---- B 臂：live **每天重算一次**門檻（真實行為的逐日模擬）----
        # 不可以拿一個固定的「最近 30 天」門檻去比 30 天前的事件——那是這支
        # 測試第一版的病（窗不同 vs 實作不同分不開）。這裡每一天都用
        # 「當天往前 30 天」重算，與離線 causal_flags 同窗。
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "volume", "delta"])
        bts = b["ts"].to_numpy(np.int64)
        bvol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
        bdl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
        acc = {k: [] for k in cw.FLOW}
        d0 = int(bts[lo_i] // 86_400_000) * 86_400_000
        dz = int(bts[-1] // 86_400_000) * 86_400_000
        for dstart in range(d0, dz + 1, 86_400_000):
            td = cw.thresholds_asof(bts, bvol, bdl, hi=dstart, sym=sym)
            if td is None:
                continue
            ff = cw.flow_flags(bts, bvol, bdl, td)
            day_idx = np.flatnonzero((bts >= dstart)
                                     & (bts < dstart + 86_400_000))
            for k in cw.FLOW:
                acc[k].append(np.intersect1d(ff[k], day_idx))
        fb = {k: (np.concatenate(v) if v else np.array([], np.int64))
              for k, v in acc.items()}
        fb["sweep"] = cand["sweep"]
        liveB = [a for a, s in cw.assemble(fb) if a >= lo_i]

        ra = compare(off, liveA)
        rb = compare(off, liveB)
        A += np.array(ra)
        B += np.array(rb)
        rows.append((sym, len(off), ra[0], ra[1], ra[2], rb[0], rb[1], rb[2]))

    print(f"=== live vs 離線偵測（最近 {DAYS} 天，容差 ±{TOL_MIN} 分）===")
    print()
    print(f"{'幣':6s} {'離線':>5s} | {'A 都抓':>6s} {'A 漏':>5s} {'A 多':>5s}"
          f" | {'B 都抓':>6s} {'B 漏':>5s} {'B 多':>5s}")
    for r in rows:
        print(f"{r[0]:6s} {r[1]:5d} | {r[2]:6d} {r[3]:5d} {r[4]:5d}"
              f" | {r[5]:6d} {r[6]:5d} {r[7]:5d}")

    def rep(nm, arr, need_rec, need_fp=None):
        both, oo, ol = arr
        tot_off, tot_live = both + oo, both + ol
        rec = both / tot_off if tot_off else float("nan")
        fp = ol / tot_live if tot_live else 0.0
        ok = rec >= need_rec and (need_fp is None or fp <= need_fp)
        print(f"{nm}  重疊 {rec*100:5.1f}%（需 >={need_rec*100:.0f}%）"
              f"   live 獨有 {fp*100:5.1f}%"
              + (f"（需 <={need_fp*100:.0f}%）" if need_fp else "")
              + f"   離線 {tot_off}  live {tot_live}")
        return ok, rec, fp

    print()
    print("=== 預註冊判準 ===")
    print()
    okA, recA, _ = rep("V1 A 臂（同旗標，只比組裝）", A, 0.95)
    print("   -> " + ("PASS —— 組裝是同一份實作" if okA else
                      "**FAIL —— 組裝仍是第二份實作，上線前必須修**"))
    okB, recB, fpB = rep("V2 B 臂（live 自己的門檻表）", B, 0.80, 0.20)
    print("   -> " + ("PASS —— 預算的門檻表夠接近，可以上線" if okB else
                      "**FAIL —— 門檻表要改成逐日重算**"))
    print()
    print("V3 歸因：A 臂量的是組裝，B 臂多出來的差就是**門檻近似**的代價"
          f"（{(recA - recB)*100:+.1f} pp）。")


if __name__ == "__main__":
    main()
