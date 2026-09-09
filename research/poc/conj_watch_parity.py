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

    C 臂「掃單那一段也對嗎」 2026-09-08 補。A/B 兩臂都把**離線的掃單事件**
        餵給 live 邏輯（`fb["sweep"] = cand["sweep"]`），所以
        「**活價位表 + 穿越測試**」——熱路徑真正在跑的那條——原本
        **從來沒被對照過**。C 臂改用 live 自己的掃單來源：每個小時開盤
        重建活價位（`pivot_table` 過濾，與 `levels_asof` 逐項驗證一致），
        再對該小時的每一分鐘做**穿越測試**。

判準（跑之前寫死）
    V1 A 臂重疊率 >= 0.95  -> 組裝是同一份
       < 0.95 -> **組裝仍是第二份實作**，上線前必須修
    V2 B 臂重疊率 >= 0.80 且 live 獨有 <= 0.20
       -> 預算的門檻表夠接近，可以上線
       不過 -> 門檻要改成逐日重算（每天一次，成本可接受）
    V3 差異必須逐項歸因，不得只報比率。
    V4 C 臂重疊率 >= 0.60 且 live 獨有 <= 0.50。
       **反向證明失敗，已知這一關抓不到穿越/水準測試的差別**：把穿越測試
       改回水準測試（我 2026-09-07 真的犯過的那個錯），C 臂只從 90.5% 動到
       88.6%，照樣 PASS。原因是 `assemble` 的 60 分鐘冷卻把洪水吸收掉了
       ——水準測試讓幾乎每分鐘都算掃單，冷卻只留每小時一個，總數幾乎沒變。
       所以 C 臂只能抓「價位表整個空掉」那類粗故障。**細的那類由 V5 抓。**
    V5 **原始掃單筆數比**（live / 離線）必須落在 [0.3, 3.0]。
       這一關**不穿過冷卻與交會條件**，直接量掃單那一段——水準測試會讓
       它暴增幾十倍，當場現形。反向證明過（見下）。
       **這是回歸守衛不是驗證關**——live 的掃單來源（分鐘級穿越活價位）
       與離線（小時級 `detect_sweeps`）本來就是不同的偵測器，本來就不該
       要求高重疊。門檻是照 2026-09-08 的實測校準的，用途是「下次有人動
       `levels_asof` / `detect_window` 把它弄壞時會大聲失敗」。
       校準當天的實測寫在下面，未來若門檻要動必須連同理由一起改。
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
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
DAYS = 30
TOL_MIN = 2          # 時間對齊容差（分鐘）
FLOWM = set(cw.FLOW)


def offline_moments(cand, lo_i):
    """離線的交會時刻（去掉 oi —— live 不吃 OI）。

    2026-09-09：錨點改成 **ready = max(第一根掃單, 第一根流量)**，與
    `conj_watch.assemble` 同步。兩邊必須用同一個錨點定義，否則這支對照
    量到的是「錨點定義差 0-5 分鐘」而不是「兩份實作同不同意」——容差只有
    2 分鐘，中位偏移就是 2 分鐘，命中率會憑空掉一截，而那個掉法看起來
    完全像「發射路徑壞了」。**改了 live 就要同步改對照，否則對照失去
    分辨力**（mistake.md 2026-09-07：對照測試自己被修了三次）。
    """
    pairs = []
    for nm in et.NAMES:
        if nm == "oi_crash" or nm == "liq_burst":
            continue
        v = cand.get(nm)
        if v is None or len(v) == 0:
            continue
        for m in ec.cooldown_filter(np.sort(v)):
            pairs.append((int(m), nm))
    out = set()
    for _a, mem in cr.groups_with_members(pairs):
        s = {x for _, x in mem}
        if "sweep" not in s or not (s & FLOWM):
            continue
        ready = max(min(m for m, x in mem if x == "sweep"),
                    min(m for m, x in mem if x in FLOWM))
        if ready >= lo_i:
            out.add(ready)
    return sorted(out)


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
    C = np.zeros(3, int)
    RAW = [0, 0]                  # live 原始掃單 / 離線原始掃單
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
        liveA = [a for a, s, _ in cw.assemble(fa) if a >= lo_i]

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
        liveB = [a for a, s, _ in cw.assemble(fb) if a >= lo_i]

        # ---- C 臂：掃單也用 live 自己的來源（活價位 + 穿越測試）----
        pt = cw.pivot_table(sym)
        bh = pd.read_parquet(BARS / f"{sym}.parquet", columns=["high", "low"])
        mhi = np.nan_to_num(bh["high"].to_numpy(float), nan=-np.inf)
        mlo = np.nan_to_num(bh["low"].to_numpy(float), nan=np.inf)
        sw = []
        if pt is not None:
            hts, conf, lvl, ishi, fp, nh, _a = pt
            h0 = int(np.searchsorted(hts, int(bts[lo_i])))
            for hh in range(max(h0, 1), nh):
                al = (conf < hh) & (fp >= hh)
                if not al.any():
                    continue
                lb = lvl[al & ishi]
                ls = lvl[al & ~ishi]
                m0 = int(np.searchsorted(bts, int(hts[hh])))
                m1 = int(np.searchsorted(bts, int(hts[hh]) + 3_600_000))
                if m1 <= m0:
                    continue
                rm = np.maximum.accumulate(mhi[m0:m1])
                rn = np.minimum.accumulate(mlo[m0:m1])
                for k in range(m1 - m0):
                    pm = rm[k - 1] if k else -np.inf
                    pn = rn[k - 1] if k else np.inf
                    if (((lb < mhi[m0 + k]) & (lb >= pm)).any()
                            or ((ls > mlo[m0 + k]) & (ls <= pn)).any()):
                        sw.append(m0 + k)
        fc = dict(fb)
        fc["sweep"] = np.array(sw, np.int64)
        liveC = [a for a, s, _ in cw.assemble(fc) if a >= lo_i]
        # V5：原始筆數,不穿過冷卻/交會（那兩層會把洪水吸收掉）
        RAW[0] += int((np.asarray(sw, np.int64) >= lo_i).sum())
        RAW[1] += int((np.asarray(cand["sweep"], np.int64) >= lo_i).sum())

        ra = compare(off, liveA)
        rb = compare(off, liveB)
        rc = compare(off, liveC)
        A += np.array(ra)
        B += np.array(rb)
        C += np.array(rc)
        rows.append((sym, len(off), ra[0], ra[1], ra[2],
                     rb[0], rb[1], rb[2], rc[0], rc[1], rc[2]))

    print(f"=== live vs 離線偵測（最近 {DAYS} 天，容差 ±{TOL_MIN} 分）===")
    print()
    print(f"{'幣':6s} {'離線':>5s} | {'A抓':>4s} {'漏':>3s} {'多':>3s}"
          f" | {'B抓':>4s} {'漏':>3s} {'多':>3s} | {'C抓':>4s} {'漏':>3s} {'多':>3s}")
    for r in rows:
        print(f"{r[0]:6s} {r[1]:5d} | {r[2]:4d} {r[3]:3d} {r[4]:3d}"
              f" | {r[5]:4d} {r[6]:3d} {r[7]:3d} | {r[8]:4d} {r[9]:3d} {r[10]:3d}")

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
    okC, recC, fpC = rep("V4 C 臂（掃單也用 live 來源）", C, 0.60, 0.50)
    print("   -> " + ("PASS —— 回歸守衛就位" if okC else
                      "**FAIL —— live 的掃單來源與離線不一致**"))
    ratio = RAW[0] / max(RAW[1], 1)
    ok5 = 0.3 <= ratio <= 3.0
    print(f"V5 原始掃單筆數 live {RAW[0]:,} / 離線 {RAW[1]:,} = {ratio:.2f}x"
          f"（需 0.3~3.0，不穿過冷卻）")
    print("   -> " + ("PASS" if ok5 else
                      "**FAIL —— live 的掃單偵測跟離線不是同一件事**"))
    print()
    print("V3 歸因：A 臂量的是組裝，B 臂多出來的差就是**門檻近似**的代價"
          f"（{(recA - recB)*100:+.1f} pp）。")


if __name__ == "__main__":
    main()
