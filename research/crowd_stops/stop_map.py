# -*- coding: utf-8 -*-
"""群眾止損地圖 + A3' 體檢（2026-09-10）

目標（使用者 2026-09-10）：「不是要知道他們用什麼策略，而是分析出他們策略
的止損——也就是散戶止損在哪——用我們的 SDV 來做進場」。

===========================================================================
為什麼這不是 §1.03i 清算位密度的重做
===========================================================================
§1.03i（2026-09-09）測過「OI 推導的清算位密度」：**構造全過、交易全不過**，
死因是 **K3 混淆對照** —— 改用 ATR% 分同樣的五分位，單調性 −0.800，
而密度只有 +0.100。那一點點結構用波動度解釋比用密度解釋好八倍。

止損位在三件事上跟清算位不同，而這三件剛好都對著上面那個死因：

    觸發條件   清算看保證金率（槓桿的函數）；止損看價格碰到規則算出的位置
    怎麼得到   清算位要**推估**（OI + 假設槓桿分布，模型輸出）；
               止損位**可以算**（SuperTrend(10,3.0) 今天在哪，誰算都一樣）
    分布形狀   清算是連續質量、實測非零率 100%（所以天生像波動度）；
               止損是離散價位，每幣每週期最多 3 個，多數時候 0 或 1

還有一件只有止損有的：**方向**。多單止損在下、空單止損在上，所以
「前方 vs 後方」的安慰劑檢定做得出來；清算質量沒有這個結構。

**先驗是負的**：最接近的前例失敗了。所以 ATR 混淆從「次要對照」升成
主關卡，而且放在最前面。

===========================================================================
地圖的定義（凍結，零自由參數）
===========================================================================
三個原型，**指標本身就是止損**，用 `detail=True` 從凍結函式投影：

    Donchian(20)        止損 = 對側通道帶（多頭 lo20、空頭 hi20）
    SuperTrend(10,3.0)  止損 = 那條線（多頭下軌、空頭上軌）
    PSAR(0.02,0.2)      止損 = SAR 值

三個週期 **1h / 4h / 1d**（群眾實際在看的那三個；4h 與 1d 由 1h 重採樣，
不引入新資料）。日界用 **UTC** —— 這裡要模擬的是群眾看到的圖，而
Binance / TradingView 的加密日線就是 UTC；這是慣例不是被調出來的參數。
（判決用的日界仍然是 UTC+8，§1.03m，兩者用途不同。）

所以每個幣、每個時刻最多 **3 x 3 = 9** 個止損價位。**不加權**（每個原型
算一個），因為每個原型後面跟著多少錢是不可知的；用近期損益當人氣代理的
加權版只當敏感度，不當主版本。

**只收未被穿的止損**：已經被穿的止損不是待觸發的燃料
（tests/test_stop_detail_parity.py 檔頭有這條規則的來由）。

===========================================================================
主變數（事件層）
===========================================================================
對每一筆掃單，方向 d = sign(pre_mom5)（SDV 的方向規則）：

    ahead   前方 0 ~ 1.5 ATR 內的止損檔數
            d>0 -> (P, P+1.5A]；d<0 -> [P-1.5A, P)
    behind  後方同距離的止損檔數   <- **安慰劑**，不該有預測力
    nearest 最近那個前方止損的距離（ATR）

幾何上前方的止損必然是**反向部位**的止損（多單止損在現價之下、空單在之上），
所以方向性是構造保證的，不需要另外篩。`V1` 會去驗證這件事確實成立。

===========================================================================
A3' 三道體檢（任一不過這條線就結束，在寫任何檢定之前）
===========================================================================
    G1  corr(ahead, ATR 分位) 的 |r| < 0.5        <- §1.03i 的死因
    G2  ahead 的分布要有對比度：沒有任何一個值佔 > 90%
    G3  地圖 vs 真實清算（九幣小時級 180 天）：止損密集的小時，
        實際清算金額是否較高。這是**答案已知**的驗證
        —— 清算是已發生的事實，止損是算出來的預測，兩者不同源。

自曝檢查
    V1  前方止損的部位方向必須與幾何一致（現價之上的止損 pos=-1）
        違反率必須 < 1%
    V2  快照筆數 9,262、SDV 1,584
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from research.crowd_battery import pos_breakout  # noqa: E402
from research.harness import asof  # noqa: E402
from research.crowd_battery2 import pos_supertrend  # noqa: E402
from research.crowd_battery3 import pos_psar  # noqa: E402

FROZEN = HERE / "frozen"
SNAP = ROOT / "research" / "poc" / "data" / "sweep_snapshot.parquet"
LIQ = ROOT / "research" / "poc" / "data" / "liq"
OUT = HERE / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
ARCH = {"donchian": pos_breakout, "supertrend": pos_supertrend,
        "psar": pos_psar}
TFS = {"1h": 1, "4h": 4, "1d": 24}      # 以 1h 為單位重採樣
REACH = 1.5                             # 前/後方各看 1.5 ATR
SEED = 20260910


def load_1h(sym):
    rows = []
    with open(FROZEN / f"{sym}USDT_1h.csv", newline="",
              encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) < 6:
                continue
            rows.append((int(float(x[0])) * 1000, float(x[1]), float(x[2]),
                         float(x[3]), float(x[4]), float(x[5])))
    return rows


def resample(bars, k):
    """1h -> k 小時。對齊到 UTC 的 k 小時邊界（1d = UTC 日界）。"""
    if k == 1:
        return bars
    out = []
    step = k * 3_600_000
    cur = None
    for t, o, h, l, c, v in bars:
        b0 = (t // step) * step
        if cur is None or cur[0] != b0:
            if cur is not None:
                out.append(tuple(cur))
            cur = [b0, o, h, l, c, v]
        else:
            cur[2] = max(cur[2], h)
            cur[3] = min(cur[3], l)
            cur[4] = c
            cur[5] += v
    if cur is not None:
        out.append(tuple(cur))
    return out


def stop_series(sym):
    """回傳 {(tf, arch): (close_ms[], pos[], stop[])}，close_ms = 該 bar
    的**收盤時刻**（= 這個止損價被知道的時刻，嚴格早於它才能用）。"""
    b1 = load_1h(sym)
    out = {}
    for tf, k in TFS.items():
        bars = resample(b1, k)
        step = k * 3_600_000
        cms = np.array([b[0] + step for b in bars], np.int64)
        for an, fn in ARCH.items():
            det = fn(bars, detail=True)
            pos = np.array([d["pos"] for d in det], np.int8)
            stp = np.array([np.nan if d["stop"] is None else d["stop"]
                            for d in det], float)
            out[(tf, an)] = (cms, pos, stp)
    return out


def build_events():
    d = pd.read_parquet(SNAP, columns=["sym", "ts", "side", "level", "is_sdv",
                                       "pre_atr", "pre_close", "pre_mom5",
                                       "y_with_d5", "y_against_d5"])
    d = d[d.sym.isin(CORE9)].reset_index(drop=True)
    rows = []
    v1_bad = v1_tot = 0
    for sym, g in d.groupby("sym", sort=False):
        ss = stop_series(sym)
        ts = g.ts.to_numpy(np.int64)
        P = g.pre_close.to_numpy(float)
        A = g.pre_atr.to_numpy(float)
        dd = np.sign(g.pre_mom5.to_numpy(float))
        dd[dd == 0] = 1.0
        ahead = np.zeros(len(g), int)
        behind = np.zeros(len(g), int)
        near = np.full(len(g), np.nan)
        for (tf, an), (cms, pos, stp) in ss.items():
            jj, ok = asof(ts, cms)          # 嚴格早於，見 harness.asof
            p_, s_ = pos[jj], stp[jj]
            live = ok & (p_ != 0) & np.isfinite(s_)
            # 只收未被穿的止損：多單止損要在現價之下、空單在之上
            live &= np.where(p_ == 1, s_ <= P, s_ >= P)
            up = dd > 0
            dist = np.where(up, (s_ - P) / A, (P - s_) / A)    # 前方為正
            inA = live & (dist > 0) & (dist <= REACH)
            inB = live & (dist < 0) & (dist >= -REACH)
            ahead += inA.astype(int)
            behind += inB.astype(int)
            near = np.where(inA & (np.isnan(near) | (dist < near)), dist, near)
            v1_tot += int(inA.sum())
            v1_bad += int((inA & (p_ == np.where(up, 1, -1))).sum())
        rows.append(pd.DataFrame(dict(
            sym=sym, ts=ts, is_sdv=g.is_sdv.to_numpy(),
            atr_pct=A / P, d=dd, ahead=ahead, behind=behind, nearest=near,
            y_with=g.y_with_d5.to_numpy(float),
            y_against=g.y_against_d5.to_numpy(float))))
    e = pd.concat(rows, ignore_index=True)
    return e, v1_bad, v1_tot


def liq_hourly():
    out = []
    for sym in CORE9:
        p = LIQ / f"{sym}.parquet"
        if not p.exists():
            continue
        q = pd.read_parquet(p)
        t = q["time"].astype("int64")
        t = t * 1000 if t.max() < 1e12 else t
        out.append(pd.DataFrame(dict(sym=sym, hour=(t // 3_600_000),
                                     liq=q["total_usd"].astype(float))))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    print("建止損地圖（3 原型 x 3 週期，九幣）…")
    e, v1_bad, v1_tot = build_events()
    OUT.mkdir(parents=True, exist_ok=True)
    e.to_parquet(OUT / "stop_map_events.parquet", index=False)

    res = {}
    print()
    print("V2 自曝：事件 %d（應 9,262），SDV %d（應 1,584）  %s"
          % (len(e), int(e.is_sdv.sum()),
             "PASS" if len(e) == 9262 and int(e.is_sdv.sum()) == 1584
             else "**FAIL**"))
    v1r = v1_bad / max(v1_tot, 1)
    print("V1 自曝：前方止損的部位方向違反幾何 %d/%d = %.3f%%  %s"
          % (v1_bad, v1_tot, 100 * v1r, "PASS" if v1r < 0.01 else "**FAIL**"))
    res["V2"] = dict(n=int(len(e)), n_sdv=int(e.is_sdv.sum()))
    res["V1"] = dict(bad=v1_bad, tot=v1_tot, rate=float(v1r))

    print("=" * 74)
    print("G2 對比度：前方止損檔數的分布（0 ~ 9）")
    vc = e.ahead.value_counts(normalize=True).sort_index()
    for k, v in vc.items():
        print("   ahead=%d  %5.1f%%  (n=%d)" % (k, 100 * v, int((e.ahead == k).sum())))
    g2 = float(vc.max()) < 0.90
    print("   最大佔比 %.1f%%  ->  %s" % (100 * vc.max(),
                                      "PASS" if g2 else "**FAIL 沒有對比度**"))
    print("   後方（安慰劑）平均 %.2f vs 前方 %.2f"
          % (e.behind.mean(), e.ahead.mean()))
    res["G2"] = dict(dist={int(k): float(v) for k, v in vc.items()},
                     max_share=float(vc.max()), pass_=bool(g2),
                     mean_ahead=float(e.ahead.mean()),
                     mean_behind=float(e.behind.mean()))

    print("=" * 74)
    print("G1 混淆：前方止損檔數 vs ATR 分位（§1.03i 的死因）")
    e["atr_q"] = e.groupby("sym").atr_pct.rank(pct=True)
    r_all = float(e.ahead.corr(e.atr_q))
    rs_all = float(e.ahead.corr(e.atr_q, method="spearman"))
    print("   全體  pearson %+.3f   spearman %+.3f" % (r_all, rs_all))
    per = {}
    for sym, g in e.groupby("sym"):
        per[sym] = float(g.ahead.corr(g.atr_q, method="spearman"))
    print("   逐幣 spearman：" + "  ".join("%s %+.2f" % (k, v)
                                        for k, v in per.items()))
    worst = max(abs(v) for v in per.values())
    g1 = abs(rs_all) < 0.5 and worst < 0.5
    print("   最大絕對值 %.3f  ->  %s" % (worst,
                                     "PASS" if g1 else "**FAIL 這就是波動度的替身**"))
    res["G1"] = dict(pearson=r_all, spearman=rs_all, per_sym=per,
                     worst=float(worst), pass_=bool(g1))

    print("=" * 74)
    print("G3 地圖 vs 真實清算（答案已知的驗證；九幣小時級 180 天）")
    lq = liq_hourly()
    if lq is None:
        print("   清算資料缺 -> 跳過")
    else:
        # 每個幣、每小時：那個小時內發生的掃單事件的前方止損檔數總和
        e["hour"] = e.ts // 3_600_000
        agg = e.groupby(["sym", "hour"], as_index=False).ahead.sum()
        m = agg.merge(lq, on=["sym", "hour"], how="inner")
        print("   可比對小時 %d（有掃單且有清算資料）" % len(m))
        if len(m) > 50:
            m["lq_rank"] = m.groupby("sym").liq.rank(pct=True)
            rr = float(m.ahead.corr(m.lq_rank, method="spearman"))
            per3 = {k: float(g.ahead.corr(g.lq_rank, method="spearman"))
                    for k, g in m.groupby("sym")}
            npos = sum(1 for v in per3.values() if v > 0)
            print("   spearman(前方止損檔數, 該小時清算金額分位) = %+.3f" % rr)
            print("   逐幣：" + "  ".join("%s %+.2f" % (k, v)
                                       for k, v in per3.items())
                  + "   -> %d/%d 為正" % (npos, len(per3)))
            print("   分組：")
            for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 9)):
                s = m[(m.ahead >= lo) & (m.ahead <= hi)]
                if len(s):
                    print("     ahead %s  n=%5d  清算分位中位 %.3f"
                          % (("%d" % lo) if lo == hi else "%d+" % lo,
                             len(s), s.lq_rank.median()))
            g3 = rr > 0 and npos >= 6
            print("   -> %s" % ("PASS（地圖對得上真實清算）" if g3
                               else "**FAIL 地圖跟真實清算對不上**"))
            res["G3"] = dict(spearman=rr, per_sym=per3, n_hours=int(len(m)),
                             n_pos=npos, pass_=bool(g3))

    p = OUT / "stop_map_a3.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print("\nwritten -> " + str(p))


if __name__ == "__main__":
    main()
