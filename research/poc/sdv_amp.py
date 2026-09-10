# -*- coding: utf-8 -*-
"""擁擠程度 -> 掃單後的**幅度**（方向無關）（2026-09-10 預註冊）

===========================================================================
為什麼放掉方向
===========================================================================
使用者 2026-09-10：「我覺得不要糾結方向這個問題」。有證據支持，而且是
這個專案三度確認過的：

    B1-B4 四個判別臂         樣本外全滅（§1.03e）
    有號 CVD 當方向來源       與五分鐘動能 96-97% 相同（§1.03j）
    SDV 勝率                  49.0%（樣本外 47.9%）—— 方向接近硬幣
    撤單流 TEST B             撤單強度 -> |未來報酬| **四個 horizon 全過、
                              兩半一致**（h5 +0.115 [+0.096,+0.134]）

最後那一行是這個專案唯一反覆站得住的東西：**幅度可以預測，方向不行。**
它在 2026-07-29 被使用者否決，理由是「系統沒有任何旋鈕接得住波動預測
（固定 2x sizing、3xATR 停損）」。

**但 SDV 有那個旋鈕。** 停損固定 3 ATR、持有固定 480 分、方向用動能，
所以它的期望值直接是**移動幅度**的函數：幅度大 -> 右尾更肥 -> 在同樣
49% 勝率下期望值更高。幅度預測在這裡不是 sizing，是**進場選擇**。

而 §1.04 的診斷已經量出 SDV 的損益就住在尾部（最好的 5% 貢獻總報酬
138%、中位數交易 -0.1376 ATR）。所以「讓尾部更常出現」正是唯一對得上
這個形狀的改善方向。

===========================================================================
假設（符號事前寫死，單尾）
===========================================================================
    H：掃單發生時**市場越擁擠**（不分哪一側），事後的**絕對移動越大**。

機制（標明是推論）：擁擠 = 同向部位多 = 價格一動就有人被迫平倉 =
級聯更長。§1.05 今天量到強制平倉發生在有槓桿順勢重倉的那一側，
所以擁擠程度本身（不分側）才是燃料的總量。

===========================================================================
定義（凍結）
===========================================================================
標的（方向無關，**三個，因為幅度有兩種而它們要的載具相反**）
    A  amp  = |y_with_d5|          **淨位移** —— SDV 要的
    B  rng  = y_mfe + |y_mae|      總擺幅
    C  chop = rng - amp            **擺幅超出位移的部分** —— 網格要的
                                   高 = 來回走了很多但沒走遠
    尾部 = 該標的 > 該幣自己的 p95（逐幣算，避免跨幣尺度）

    使用者 2026-09-10：「如果是波動就要用網格的方式了」。這句逼出上面的
    A/C 分裂：**高波動對 SDV 與對網格的意義相反**。SDV 要的是走得遠
    （位移），網格要的是走得多但不走遠（震盪）—— 趨勢會讓網格的庫存
    越積越虧。所以「擁擠預測幅度」必須拆開問「預測的是哪一種幅度」，
    否則結論會被套到錯的載具上。

    限制註記：停損把 mae 截在 -3 ATR，所以 chop 在被停損的交易上被低估。

變數（三個，全部方向無關、全部嚴格早於事件）
    c_retail   |z(散戶帳戶多空比)|      偏離常態的程度
    c_toppos   |z(大戶部位多空比)|
    c_oi       OI 名目的過去 30 天分位   規模型擁擠
    z = (值 - 過去 30 天中位) / 過去 30 天 IQR，rolling 且 shift(1)

母體
    主   全部掃單 9,262
    次   SDV 1,584

統計量（**不用均值**：SDV 的均值檢定 MDE = 0.746 ATR，而所有測過的
因子效應量 <= 0.287，那條路已證明走不通。厚尾上數東西比平均東西便宜。）
    尾部事件落在「高擁擠半邊」的比例，**二項單尾**；虛無比例 = 該半邊
    的樣本占比。

===========================================================================
判準（凍結，跑之前 commit）
===========================================================================
    R1  主母體、三變數 x 三標的共九格，至少一格：二項單尾 p < 0.05 / 9
    R2  **ATR 增量關（本節的核心）**
        (a) ATR 分位自己切的集中度 = 波動基準線
        (b) **在 ATR 五分位之內**再按擁擠切，集中度仍須顯著
        若 (b) 消失 -> 結論是「這只是波動度」。那不是新發現：
        波動可預測這件事 2026-07-29 就測過並被否決，**不得當成新東西**。
    R3  逐幣 >= 6/9 同號
    R4  置換：幣內打亂擁擠序列、跑完整流程、取三格最佳，p < 0.05
    R5  經濟代價：若通過，必須同時報「只做高擁擠那半」之後的
        事件率、以及 SDV 子母體的每筆淨值變化。事件率已經是問題之一
        （1.70 筆/天），砍半要講清楚。
    R1..R4 全過 -> 候選；任一不過 -> 記錄結案，寫明是「測過輸了」
    還是「測不動」。

**處置分流（事前寫死，不得看到結果才決定要去哪）**

    擁擠 -> amp  且過 R2   -> SDV 的進場濾網
    擁擠 -> chop 且過 R2   -> **路由到網格**，不是 SDV。產品端 jarvis
                              已有 grid bot，所以這是既有載具不是新工程
    只有 ATR 有效、擁擠無增量 -> **不是新發現**。波動可預測 2026-07-29
                              就測過並被使用者否決；可記「ATR 本身可以
                              當網格的開關」，但那是工程決定不是研究發現
    三個標的都不過          -> 結案

===========================================================================
自曝檢查
===========================================================================
    S1  母體 9,262 / SDV 1,584
    S2  我算的 as-of 散戶多空比必須對上快照的 pre_ls_retail（答案已知），
        中位絕對誤差 < 1e-9
    S3  corr(擁擠, ATR 分位) 要印出來 —— 它決定 R2 有沒有分辨力
    S4  z 的中位要貼零（否則 |z| 會被水平偏移污染）
    S5  尾部佔比必須 ~5%（p95 的定義檢查）
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

SNAP = HERE / "data" / "sweep_snapshot.parquet"
OI = HERE / "data" / "oi"
OUT = HERE / "data" / "results" / "sdv_amp.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
WIN = 30 * 24 * 12                 # 30 天的 5 分鐘根數
SEED = 20260910
NPERM = 400
VARS = ("c_retail", "c_toppos", "c_oi")


def load_oi(sym):
    q = pd.read_parquet(OI / f"{sym}.parquet")
    t = q["create_time"]
    if np.issubdtype(t.dtype, np.datetime64):
        ms = t.astype("int64") // 10 ** 6
    else:
        ms = t.astype("int64")
        ms = ms * 1000 if ms.max() < 1e12 else ms
    return q.assign(_ms=np.asarray(ms)).sort_values("_ms").reset_index(drop=True)


def _z(v):
    r = v.shift(1).rolling(WIN, min_periods=WIN // 4)
    iqr = (r.quantile(0.75) - r.quantile(0.25)).replace(0.0, np.nan)
    return (v - r.median()) / iqr


def build():
    d = pd.read_parquet(SNAP, columns=["sym", "ts", "is_sdv", "pre_atr",
                                       "pre_close", "pre_ls_retail",
                                       "y_with_d5", "y_mfe", "y_mae"])
    d = d[d.sym.isin(CORE9)].reset_index(drop=True)
    frames, s2 = [], []
    for sym, g in d.groupby("sym", sort=False):
        q = load_oi(sym)
        zr = _z(q["count_long_short_ratio"].astype(float)).to_numpy()
        zt = _z(q["sum_toptrader_long_short_ratio"].astype(float)).to_numpy()
        oiv = q["sum_open_interest_value"].astype(float)
        qoi = oiv.shift(1).rolling(WIN, min_periods=WIN // 4).rank(pct=True)
        qoi = qoi.to_numpy()
        oms = q["_ms"].to_numpy(np.int64)
        j = np.searchsorted(oms, g.ts.to_numpy(np.int64), side="left") - 1
        ok = j >= 0
        jj = np.clip(j, 0, len(oms) - 1)
        s2.append(np.abs(q["count_long_short_ratio"].to_numpy(float)[jj]
                         - g.pre_ls_retail.to_numpy(float))[ok])
        frames.append(pd.DataFrame(dict(
            sym=sym, ts=g.ts.to_numpy(), is_sdv=g.is_sdv.to_numpy(),
            amp=np.abs(g.y_with_d5.to_numpy(float)),
            rng=g.y_mfe.to_numpy(float) + np.abs(g.y_mae.to_numpy(float)),
            chop=(g.y_mfe.to_numpy(float) + np.abs(g.y_mae.to_numpy(float))
                  - np.abs(g.y_with_d5.to_numpy(float))),
            atr_pct=(g.pre_atr / g.pre_close).to_numpy(float),
            z_retail=np.where(ok, zr[jj], np.nan),
            z_toppos=np.where(ok, zt[jj], np.nan),
            c_retail=np.abs(np.where(ok, zr[jj], np.nan)),
            c_toppos=np.abs(np.where(ok, zt[jj], np.nan)),
            c_oi=np.where(ok, qoi[jj], np.nan))))
    return pd.concat(frames, ignore_index=True), np.concatenate(s2)


def hi_half(sub, var):
    """幣內按中位切高半邊。"""
    v = sub[var].to_numpy(float)
    ok = np.isfinite(v)
    hi = np.zeros(len(sub), bool)
    syms = sub.sym.to_numpy()
    for s in np.unique(syms):
        m = (syms == s) & ok
        if m.sum() >= 20:
            hi |= m & (v > np.nanmedian(v[m]))
    return hi, ok


def binom_p(k, n, p0):
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
                     for i in range(k, n + 1)))


TARGETS = (("amp", "淨位移"), ("rng", "總擺幅"), ("chop", "震盪"))


def tail_test(sub, var, tail, strata=None, ycol="amp"):
    """尾部是否集中在高擁擠半邊。strata 給定時，在每層內各自切中位。"""
    v = sub[var].to_numpy(float)
    ok = np.isfinite(v)
    hi = np.zeros(len(sub), bool)
    syms = sub.sym.to_numpy()
    keys = (syms if strata is None
            else np.char.add(syms.astype(str), strata.astype(str)))
    for kk in np.unique(keys):
        m = (keys == kk) & ok
        if m.sum() >= 20:
            hi |= m & (v > np.nanmedian(v[m]))
    t = tail & ok
    n, k = int(t.sum()), int((t & hi).sum())
    if n < 20:
        return None
    p0 = float((hi & ok).sum() / max(ok.sum(), 1))
    per = {}
    for s in np.unique(syms):
        m = (syms == s) & t
        if m.sum() >= 3:
            per[s] = float((m & hi).sum() / m.sum() - p0)
    return dict(n_tail=n, k_hi=k, share=k / n, p0=p0, p=binom_p(k, n, p0),
                n_pos=sum(1 for x in per.values() if x > 0), n_sym=len(per),
                per_sym=per,
                amp_hi=float(np.nanmean(sub[ycol].to_numpy()[hi & ok])),
                amp_lo=float(np.nanmean(sub[ycol].to_numpy()[(~hi) & ok])))


def main():
    e, s2 = build()
    res = {}
    ok1 = len(e) == 9262 and int(e.is_sdv.sum()) == 1584
    print("S1 母體 %d（應 9,262）  SDV %d（應 1,584）  %s"
          % (len(e), int(e.is_sdv.sum()), "PASS" if ok1 else "**FAIL**"))
    # nanmedian 不是 median：來源有一列本身是 NaN（AVAX），用 median 會讓
    # 整個自曝檢查回傳 nan —— 一個看起來像壞掉、其實是好的結果。
    s2m = float(np.nanmedian(s2))
    print("S2 as-of 散戶多空比 vs 快照：中位絕對誤差 %.3g（有限 %d/%d、p99 %.3g）"
          "  %s" % (s2m, int(np.isfinite(s2).sum()), len(s2),
                    float(np.nanpercentile(s2, 99)),
                    "PASS" if s2m < 1e-9 else "**FAIL**"))
    for nm in ("retail", "toppos"):
        z = e["z_" + nm]
        print("S4 z_%-7s 中位 %+.3f  IQR %.3f  有效 %.0f%%"
              % (nm, z.median(), z.quantile(.75) - z.quantile(.25),
                 100 * z.notna().mean()))
    e["atr_q"] = e.groupby("sym").atr_pct.rank(pct=True)
    s3 = {}
    for v in VARS:
        r = float(e[v].corr(e.atr_q, method="spearman"))
        s3[v] = r
        print("S3 corr(%s, ATR 分位) spearman %+.3f" % (v, r))
    res["S"] = dict(n=int(len(e)), n_sdv=int(e.is_sdv.sum()),
                    s2_med=float(np.median(s2)), s3=s3)

    cells = {}
    for pop, sub0 in (("全掃單", e), ("SDV", e[e.is_sdv])):
        sub = sub0.reset_index(drop=True)
        aq = pd.qcut(sub.atr_q, 5, labels=False, duplicates="drop").to_numpy()
        for tgt, tlab in TARGETS:
            th = sub.groupby("sym")[tgt].transform(lambda x: x.quantile(0.95))
            tail = (sub[tgt] > th).to_numpy()
            print("\n" + "=" * 76)
            print("%s x %s  n=%d  尾部 %d（%.1f%%，S5 應 ~5%%）"
                  % (pop, tlab, len(sub), int(tail.sum()), 100 * tail.mean()))
            for v in VARS:
                r = tail_test(sub, v, tail, ycol=tgt)
                r2 = tail_test(sub, v, tail, strata=aq, ycol=tgt)
                if r is None:
                    continue
                cells["%s|%s|%s" % (pop, tgt, v)] = dict(plain=r, within_atr=r2)
                print("  %-9s 尾部落高半邊 %5.1f%%（虛無 %.1f%%）p=%.4f"
                      "  逐幣 %d/%d  標的均值 %.3f vs %.3f"
                      % (v, 100 * r["share"], 100 * r["p0"], r["p"],
                         r["n_pos"], r["n_sym"], r["amp_hi"], r["amp_lo"]))
                if r2:
                    print("  %-9s   **ATR 五分位內** %5.1f%%  p=%.4f  逐幣 %d/%d"
                          % ("", 100 * r2["share"], r2["p"],
                             r2["n_pos"], r2["n_sym"]))
            ra = tail_test(sub, "atr_q", tail, ycol=tgt)
            if ra:
                cells["%s|%s|ATR基準線" % (pop, tgt)] = dict(plain=ra)
                print("  %-9s 尾部落高半邊 %5.1f%%  p=%.4f  逐幣 %d/%d"
                      "   <- 波動基準線"
                      % ("ATR", 100 * ra["share"], ra["p"],
                         ra["n_pos"], ra["n_sym"]))
    res["cells"] = cells

    print("\n" + "=" * 76)
    rng = np.random.default_rng(SEED)
    sub = e.reset_index(drop=True)
    TAILS = {}
    for tg, _ in TARGETS:
        th = sub.groupby("sym")[tg].transform(lambda x: x.quantile(0.95))
        TAILS[tg] = (sub[tg] > th).to_numpy()
    real = max(c["plain"]["share"] for k, c in cells.items()
               if k.startswith("全掃單|") and not k.endswith("ATR基準線"))
    vals, hits = [], 0
    for _ in range(NPERM):
        w = sub.copy()
        for v in VARS:
            x = w[v].to_numpy(float).copy()
            for s, idx in w.groupby("sym").indices.items():
                x[idx] = rng.permutation(x[idx])
            w[v] = x
        b = max((tail_test(w, v, TAILS[tg], ycol=tg) or {"share": -1})["share"]
                for v in VARS for tg, _lab in TARGETS)
        vals.append(b)
        hits += (b >= real)
    p4 = (hits + 1) / (NPERM + 1)
    print("R4 置換 %d 輪：隨機最佳中位 %.3f、p95 %.3f；真實 %.3f"
          % (NPERM, float(np.median(vals)), float(np.percentile(vals, 95)), real))
    print("   **p = %.4f**  %s" % (p4, "PASS" if p4 < 0.05 else "**FAIL**"))
    res["R4"] = dict(p=float(p4), real=float(real), med=float(np.median(vals)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print("\nwritten -> " + str(OUT))


if __name__ == "__main__":
    main()
