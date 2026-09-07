# -*- coding: utf-8 -*-
"""發射路徑的已知答案對照 —— 把真實發生過的交會時刻餵回去，要求它發得出

為什麼需要這支
    交會事件約**每幣每 2.5 天一次**，所以 `conj_watch.py` 正常跑一輪的結果
    永遠是 `events=0`。那個 0 同時代表兩件事：

        (a) 這一分鐘剛好沒有事件（正常）
        (b) 發射路徑壞了，永遠不會發（bug）

    畫面上長得一模一樣。這正是 [[mistake 2026-08-26 SELECT 了卻沒 emit]]
    的形狀——**唯一看得見這條路徑的時刻，正是沒人在看的時刻**。所以驗收
    不能等「哪天剛好有事件時再看一眼」，必須用不依賴那個條件的方式驗。

做法
    從離線偵測取每個幣最近 30 天內**真實發生過**的交會時刻，把該時刻前
    LOOKBACK_MIN 根 1 分鐘 K 當成「當下」餵給 `detect_window`（也就是熱路徑
    真正在跑的那顆），要求它把該時刻發出來。

    不寫 DB。這裡驗的是偵測到寫入列之間那一段，DB 寫入本身由
    `INSERT IGNORE` 與 UNIQUE KEY 保證。

判準（跑之前寫死）
    I1 命中率 >= 0.80 -> 發射路徑會動
       低於 -> **發射路徑在真實事件上發不出來**，上線前必須修
    I2 反向證明：把 flow 門檻調到不可能達到（x100），命中率必須掉到 0。
       沒掉到 0 -> 這支測試自己是壞的（它在量別的東西），結果不解讀。
    I3 逐幣全格報告，不挑幣。
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
import conj_clock as ck  # noqa: E402
import conj_watch as cw  # noqa: E402
from conj_watch_parity import offline_moments  # noqa: E402

BARS = HERE / "data" / "bars"
DAYS = 30
MAX_PER_SYM = 8          # 每幣最多注入幾個，夠判定就好


def run(levels, thr_by, liq, blind=False):
    """blind=True 時把門檻乘 100（反向證明：命中率必須掉到 0）。"""
    hit = miss = 0
    rows = []
    for sym in ec.CORE9:
        if sym not in thr_by:
            continue
        t = thr_by[sym]
        if blind:
            t = cw.Thr(t.sym, t.thr_delta * 100, t.thr_vol * 100,
                       t.vol_base_tod, t.atr)
        cand, ts, cl, at, day = ck.frozen_cand(sym, liq)
        lo_i = int(np.searchsorted(ts, ts[-1] - DAYS * 86_400_000))
        moments = offline_moments(cand, lo_i)[-MAX_PER_SYM:]

        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "high", "low", "close",
                                     "volume", "delta"])
        bts = b["ts"].to_numpy(np.int64)
        hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        cls = b["close"].to_numpy(float)
        vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
        dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)

        h = m = 0
        thr_cache = {}
        for a in moments:
            # **當天**的門檻，不是今天的。live 每天重算一次，拿今天的門檻去比
            # 30 天前的事件是拿不同資料窗互比 —— 這支測試的第一版就是這樣，
            # 分不出「實作有 bug」和「門檻窗依設計不同」。
            dstart = int(bts[a]) // 86_400_000 * 86_400_000
            if dstart not in thr_cache:
                thr_cache[dstart] = cw.thresholds_asof(
                    bts, vol, dl, hi=dstart, sym=sym, atr=float(t.atr))
            td = thr_cache[dstart]
            if td is None:
                continue
            if blind:
                td = cw.Thr(td.sym, td.thr_delta * 100, td.thr_vol * 100,
                            td.vol_base_tod, td.atr)
            # 「當下」= 錨點之後 MERGE_GAP 分（流量可能比掃單晚到，
            # 熱路徑要等那幾根收盤才組得出這個時刻）
            end = a + cw.MERGE_GAP + 1
            s0 = end - cw.LOOKBACK_MIN
            if s0 < 0 or end > len(bts):
                continue
            sl = slice(s0, end)
            # **當時**還活著的價位，不是今天的。今天的表裡沒有這個事件掃掉的
            # 那個價位（它已經被標記為穿越過），拿它重播歷史必然是 0 命中。
            # 切在**事件所屬那根小時 K 的開盤**。切在事件當下是不對的：
            # 該事件掃掉的價位在那根小時 K 裡就已被標記為穿越，表裡沒有它，
            # 於是永遠對不到。production 的表最多舊一小時，行為與此一致。
            gl = pd.DataFrame(cw.levels_asof(
                sym, hi_ts=int(bts[a]) // 3_600_000 * 3_600_000))
            if not len(gl):
                m += 1
                continue
            ev = cw.detect_window(sym, bts[sl], hi[sl], lo[sl], cls[sl],
                                  vol[sl], dl[sl], td, gl, 0)
            got = any(abs(e[1] - int(bts[a])) <= 2 * 60_000 for e in ev)
            h, m = h + int(got), m + int(not got)
        hit, miss = hit + h, miss + m
        rows.append((sym, h + m, h, m))
    return hit, miss, rows


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    levels = pd.read_parquet(HERE / "data" / "live" / "levels.parquet")
    thr = pd.read_parquet(HERE / "data" / "live" / "thresholds.parquet")
    thr_by = {r.sym: r for r in thr.itertuples()}

    hit, miss, rows = run(levels, thr_by, liq)
    print("=== 發射路徑注入測試（真實歷史交會時刻 -> detect_window）===")
    print()
    print(f"{'幣':6s} {'注入':>5s} {'發出':>5s} {'漏':>5s}")
    for sym, n, h, m in rows:
        print(f"{sym:6s} {n:5d} {h:5d} {m:5d}")
    tot = hit + miss
    rate = hit / tot if tot else float("nan")
    print()
    print("=== 預註冊判準 ===")
    print()
    ok1 = rate >= 0.80
    print(f"I1 命中率 {rate*100:.1f}%（{hit}/{tot}，需 >=80%）-> "
          + ("PASS —— 發射路徑會動" if ok1 else
             "**FAIL —— 真實事件發不出來，上線前必須修**"))

    bh, bm, _ = run(levels, thr_by, liq, blind=True)
    ok2 = bh == 0
    print(f"I2 反向證明（門檻 x100）命中 {bh}/{bh+bm}，需為 0 -> "
          + ("PASS —— 這支測試量的是它宣稱在量的東西" if ok2 else
             "**FAIL —— 測試自己是壞的，I1 不解讀**"))


if __name__ == "__main__":
    main()
