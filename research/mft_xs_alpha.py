# -*- coding: utf-8 -*-
"""MFT 第一測：他那個 1 小時橫斷面 alpha 的**毛利**（2026-09-11）

===========================================================================
為什麼現在做這個
===========================================================================
使用者 2026-09-11：「可以量一下 MFT」。

MFT 這條線**成本側已經量完、門檻已知**（`gate0_xs_turnover.py`）：

    每小時換手 52%  ->  吃單 4.55 bps/小時（= 109 bps/天）、掛單 1.52 bps/小時

**缺的是毛利。** 這支補那一半。`Gate 0` 的 (c) 問的就是
「可實現的邊際 vs 成本，比值多少」——我們有分母了，現在量分子。

===========================================================================
規格（抄他的，逐項標明哪一項是我們改的）
===========================================================================
〈A Real HFT/MFT Alpha〉（docs/external_reading.md 第 7 則）：

    舊檔 = 上一個快照（1 分鐘前）這個價位上有超過 $100；新檔 = 沒有
    深度聚合到 5 bps 與 10 bps，各自算失衡 (bid−ask)/(bid+ask)
    **新舊兩個失衡符號相反**，所以合成訊號 = new − old
    宇宙滾動前 50（市值）、權重＝特徵的橫斷面 z、縮放到總槓桿 1
    每小時再平衡，標的是 1h 收盤對收盤報酬
    宣稱：各自 >2 Sharpe、合起來 >3 Sortino（**原始訊號，未扣成本**）

**我們與他不同的四處，全部寫出來**：

| 項目 | 他 | 我們 | 後果 |
|---|---|---|---|
| 宇宙 | 滾動前 50 | **11 個**（`orderbook_snapshots_1m` 有的） | 11 個名字的橫斷面 z 很吵，功效差 |
| 場館 | 未明說 | Binance L20 | |
| 標的報酬 | 「1h 收盤對收盤」 | **mid 對 mid** | 故意的：成交價在薄簿口上自帶負自相關，會偽裝成反轉（mistake.md 2026-09-11）|
| $100 門檻 | 他的數 | 照抄，**未經我們驗證** | 與 `hl_mid.OLD_LEVEL_USD` 同一個未驗常數 |

**特徵建構直接 import `gate0_xs_turnover`**，不重寫——那支是這個特徵的
唯一實作，而 C1 會驗證它的凍結輸出沒有被我今天的改動移動。

===========================================================================
自曝檢查（跑之前就寫死，答案已知）
===========================================================================
C1  **凍結對照**：重算每小時換手，必須重現 `gate0_xs_turnover.json` 的
    `turnover_mean`（容差 0.005）。我今天為了留 new/old 而改過 `build()`，
    這一關就是驗那個改動是 additive 的。
C2  **機制關**：他說「新舊兩個失衡符號相反」。量 `corr(new, old)`
    與兩者各自的 IC。**這一關不判過不過，它問的是「它在做我以為的事嗎」**
    （mistake.md 2026-09-09 第三條）。相反才是他的機制；同號代表他的
    代理在我們這份資料上抓到的不是同一件事。
C3  **零成本對照**：淨值必須 ≤ 毛利。反了就是成本算式壞了
    （`factor-research.md` 第 10 條，5 秒鐘擋掉整類錯誤）。
C4  **功效先算**：用**判決會用的那台機器**（日聚類 bootstrap）算每小時毛利的
    SE，再跟 4.55 / 1.52 bps 的門檻比。**SE ≥ 門檻 ⇒ 這個設計不能做決定**
    （mistake.md 2026-09-04 / 09-06），那就報「無效判決」而不是 FAIL。

===========================================================================
報告紀律
===========================================================================
核心原則 9：**樣本外先講**。這支是**單一樣本、沒有樣本外切分**——
所以它**不得被當成「發現」**，只能回答一個更窄的問題：
**「毛利有沒有大到值得繼續？」** 那是 Gate 0 的問題，不是 alpha 的判決。
要升級成判決必須再走 `.claude/rules/factor-research.md` 的十道。

    python research/mft_xs_alpha.py
    python research/mft_xs_alpha.py --days 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import gate0_xs_turnover as XS                   # noqa: E402

OUT = ROOT / "research" / "results" / "mft_xs_alpha.json"
FROZEN = ROOT / "research" / "results" / "gate0_xs_turnover.json"
TOL_C1 = 0.005
# 費率在 main() 裡按 --venue 決定（見 XS.fees_for 的說明：訊號量在哪、
# 單會送去哪，是兩件事，而這裡要用的是**後者**）。模組層先給佔位值，
# 讓它在被指定之前就不可能被誤用成一個看起來合理的預設。
TAKER = MAKER = float("nan")
FEE_SRC = "（未指定場館）"
RNG = np.random.default_rng(20260911)


def day_boot(x, days, b=4000):
    """日聚類 bootstrap。回傳 (均值, SE, CI下, CI上, P(>0))。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (np.nan,) * 5
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return (float(x.mean()), float(r.std(ddof=1)),
            float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)),
            float((r > 0).mean()))


def port_returns(f, col):
    """權重 x 下一小時 mid 報酬。**不重疊區間**（h=1 所以天然不重疊）。"""
    feat = f.pivot_table(index="minute", columns="sym", values=col)
    mid = f.pivot_table(index="minute", columns="sym", values="mid")
    w = XS.weights_from(feat)                     # 橫斷面 z -> 總槓桿 1
    fwd = mid.shift(-1) / mid - 1.0               # 這一格到下一格（1 小時）
    common = w.index.intersection(fwd.index)
    w, fwd = w.loc[common], fwd.loc[common]
    both = w.notna() & fwd.notna()
    r = (w.where(both) * fwd.where(both)).sum(axis=1, min_count=1)
    ic = [float(pd.Series(feat.loc[t]).corr(pd.Series(fwd.loc[t]), method="spearman"))
          for t in common]
    return r.dropna() * 1e4, pd.Series(ic, index=common).dropna(), w


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--venue", default="bitget",
                    choices=sorted(XS.VENUE_FEES),
                    help="**執行**場館的費率（不是訊號來源的場館）")
    ap.add_argument("--rebate", type=float, default=None,
                    help="返佣比例，覆蓋場館預設（0.5 = 打五折）")
    a = ap.parse_args()
    global TAKER, MAKER, FEE_SRC
    TAKER, MAKER, FEE_SRC = XS.fees_for(a.venue, a.rebate)
    print("費率：%s  taker %.2f / maker %.2f bps 每邊" % (a.venue, TAKER, MAKER))
    print("      %s" % FEE_SRC)
    print("      **訊號量自 Binance 簿口，這是執行場館的費率 —— 兩者刻意分開**")
    print()

    print("抓快照（%d 天）…" % a.days)
    d = XS.load_pairs(a.days)
    f = XS.build(d)
    if f is None or not len(f):
        print("沒有可用的再平衡點，停。")
        return 2
    f["day"] = pd.to_datetime(f.minute * 60000, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), days=a.days,
               rebalances=int(f.minute.nunique()), symbols=int(f.sym.nunique()),
               rows=int(len(f)), venue=a.venue, fee_src=FEE_SRC,
               thresholds=dict(taker_bps_h=TAKER, maker_bps_h=MAKER))
    print("再平衡 %d 次、%d 標的、%d 列\n"
          % (res["rebalances"], res["symbols"], res["rows"]))

    # ---------- C1 凍結對照 ----------
    # **這一關要跑在它的基準當初那份資料窗上，不是跑在本次的 --days 上。**
    # 換手是資料相依的量，拿 120 天的換手去比 14 天算出來的凍結值，必然不符
    # ——而它印出來的診斷會是「你動到了凍結的算術」，那是錯的歸因，
    # 會讓人以為程式壞了而不是窗變了（mistake.md 2026-09-10：把 sha 釘樁
    # 釘在一份滾動的資料窗上，那條守衛在下一次刷新就必紅，而且紅的原因
    # 跟它要保護的東西完全無關）。
    #
    # 正確的形式：守衛比的是「同一份輸入 -> 同樣的輸出」。所以窗不同時
    # 另外抓一份基準窗的資料來比，主分析照樣用使用者要的窗。
    print("=== C1 凍結對照（驗算術沒被改動；跑在基準自己的資料窗上）===")
    c1 = True
    if FROZEN.exists():
        fr = json.loads(FROZEN.read_text(encoding="utf-8"))
        ref_days = int(fr.get("days") or a.days)
        if ref_days == a.days:
            fc = f
        else:
            print("  本次窗 %d 天 ≠ 基準窗 %d 天 -> 另抓一份基準窗來比"
                  % (a.days, ref_days))
            fc = XS.build(XS.load_pairs(ref_days))
        for bnd in XS.BANDS:
            _, dw = XS.turnover(fc, "combo%d" % bnd)
            got = float(dw.mean())
            ref = ((fr.get("arms") or {}).get("%dbps" % bnd) or {}).get("turnover_mean")
            ok = ref is None or abs(got - ref) < TOL_C1
            c1 &= bool(ok)
            print("  %2d bps  換手本次 %.4f  凍結 %s  %s"
                  % (bnd, got, ("%.4f" % ref) if ref is not None else "—",
                     "OK" if ok else "**不符**"))
    else:
        print("  找不到凍結檔，跳過（**這一關因此沒有保護力**）")
        c1 = False
    res["C1"] = bool(c1)
    if not c1:
        print("\n**C1 未過 —— 我的改動動到了凍結的算術，以下不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        return 2
    print("  -> PASS\n")

    # ---------- C2 機制關：新舊符號相反嗎 ----------
    print("=== C2 機制關：他說「新舊兩個失衡符號相反」===")
    print("%-8s %12s %12s %12s %12s"
          % ("帶", "corr(新,舊)", "IC 新", "IC 舊", "IC 合成"))
    mech = {}
    for bnd in XS.BANDS:
        c = float(f["new%d" % bnd].corr(f["old%d" % bnd]))
        ics = {}
        for nm in ("new", "old", "combo"):
            _, ic, _ = port_returns(f, "%s%d" % (nm, bnd))
            ics[nm] = float(ic.mean()) if len(ic) else np.nan
        mech["%dbps" % bnd] = dict(corr_new_old=c, **{"ic_" + k: v for k, v in ics.items()})
        print("%-8s %12.3f %12.4f %12.4f %12.4f"
              % ("%d bps" % bnd, c, ics["new"], ics["old"], ics["combo"]))
    res["C2_mechanism"] = mech
    print("  他的機制要求 corr(新,舊) **為負**、且兩者 IC 反號。")
    print("  **這一關不判過不過** —— 它回答「它在做我以為的事嗎」。\n")

    # ---------- 毛利 / 淨值 / 功效 ----------
    print("=== 毛利、淨值、功效（每小時 bps）===")
    print("%-10s %9s %8s %9s %9s %8s %10s %10s"
          % ("訊號", "毛利", "SE", "CI下", "CI上", "P(>0)", "淨(吃單)", "淨(掛單)"))
    # 2026-09-11 第二輪：**把 new / old 也當成完整的臂**。第一輪只對 combo
    # 算了組合報酬，而 C2 量到 `new` 單獨的 |IC| 是 combo 的兩倍（0.0291 vs
    # 0.0149，兩個帶一致）。只報 IC 不報組合報酬，等於「量到它比較好卻沒有
    # 量它值多少錢」—— 而那個錢才是 Gate 0 的分子。
    arms = {}
    cols = ["%s%d" % (nm, bnd) for bnd in XS.BANDS
            for nm in ("new", "old", "combo")]
    for col in cols:
        r, ic, w = port_returns(f, col)
        dmap = f.drop_duplicates("minute").set_index("minute")["day"]
        dys = dmap.reindex(r.index).values
        m, se, lo, hi, ppos = day_boot(r.values, dys)
        _, dw = XS.turnover(f, col)
        tk = float(dw.mean()) * 2 * TAKER       # 單邊換手 x 兩邊 x 費率
        mk = float(dw.mean()) * 2 * MAKER
        arms[col] = dict(
            turnover=float(dw.mean()),
            gross_bps_h=m, se_bps_h=se, ci_lo=lo, ci_hi=hi, p_pos=ppos,
            ic_mean=float(ic.mean()), n_reb=int(len(r)),
            cost_taker_bps_h=tk, cost_maker_bps_h=mk,
            net_taker=m - tk, net_maker=m - mk,
            inconclusive_by_design=bool(se >= MAKER))
        print("%-10s %+9.3f %8.3f %+9.3f %+9.3f %7.1f%% %+10.3f %+10.3f"
              % (col, m, se, lo, hi, 100 * ppos, m - tk, m - mk))
    res["arms"] = arms

    # ---------- C3 零成本對照 ----------
    c3 = all(v["net_taker"] <= v["gross_bps_h"] + 1e-9
             and v["net_maker"] <= v["gross_bps_h"] + 1e-9 for v in arms.values())
    res["C3"] = bool(c3)
    print("\n=== C3 零成本對照 ===")
    print("  淨值 <= 毛利：%s" % ("OK" if c3 else "**壞了——成本算式有問題**"))

    # ---------- C4 功效 ----------
    print("\n=== C4 功效（SE vs 門檻）===")
    for k, v in arms.items():
        verdict = ("**無效判決（設計上測不動）**" if v["se_bps_h"] >= MAKER
                   else "有測量能力")
        print("  %-8s SE %.3f bps/h  vs 掛單門檻 %.2f / 吃單門檻 %.2f  -> %s"
              % (k, v["se_bps_h"], MAKER, TAKER, verdict))

    # ---------- 符號必須在樣本外決定 ----------
    # 第二輪跑出一個反直覺的東西：`new` 的 |IC| 最大（0.0291）但組合報酬最差
    # （−0.747）。原因是**符號**——IC 是負的，而權重是做多高 z，所以 IC 越負
    # 組合越虧。也就是說 `new` 是最強的訊號，只是方向要反過來。
    #
    # **但「在同一份樣本上挑符號」是白送一個自由度**（mistake.md 2026-09-09：
    # 事後找到的維度通過多少檢查都不算數）。所以這一關照那條規矩做：
    # **只用前半挑符號與臂，再看後半**，並且**報「前半選到什麼」**，
    # 不只報「我選的那個在後半如何」。
    print("")
    print("=== 符號與臂**只用前半挑**，後半才是樣本外 ===")
    halves = {}
    mins = sorted(f.minute.unique())
    cut = mins[len(mins) // 2]
    f1, f2 = f[f.minute < cut], f[f.minute >= cut]
    print("  前半 %d 次再平衡 / 後半 %d 次（切點 %s）"
          % (f1.minute.nunique(), f2.minute.nunique(),
             pd.to_datetime(cut * 60000, unit="ms", utc=True).strftime("%m-%d %H:%M")))
    print("  %-10s %11s %11s %8s" % ("臂", "前半毛(帶符號)", "後半毛(同符號)", "符號"))
    pick = None
    for col in cols:
        r1, _, _ = port_returns(f1, col)
        r2, _, _ = port_returns(f2, col)
        if not len(r1) or not len(r2):
            continue
        sgn = 1.0 if r1.mean() >= 0 else -1.0      # 符號由**前半**決定
        g1, g2 = sgn * r1.mean(), sgn * r2.mean()
        halves[col] = dict(sign=sgn, first=float(g1), second=float(g2))
        print("  %-10s %+11.3f %+11.3f %8s" % (col, g1, g2, "+" if sgn > 0 else "−"))
        if pick is None or g1 > halves[pick]["first"]:
            pick = col
    if pick:
        h = halves[pick]
        print("")
        print("  >> **只用前半，程序會挑 `%s`（符號 %s）**：前半 %+.3f -> 後半 %+.3f bps/h"
              % (pick, "+" if h["sign"] > 0 else "−", h["first"], h["second"]))
        print("  >> 後半 vs 掛單門檻 %.2f bps/h -> **%s**"
              % (MAKER, "越過" if h["second"] > MAKER else "不過"))
        res["oos_sign_pick"] = dict(arm=pick, **h)
    # 整張前後半表也存下來：網站的圖要讀它，而**手抄數字會漂**
    # （mistake.md 2026-08-26：把既有數字搬到新地方顯示 = 第二份實作）。
    res["halves"] = halves

    best = max(arms, key=lambda k: arms[k]["net_maker"])
    v = arms[best]
    print("")
    print("=== 哪一臂最好（**這是事後挑的，要標明**）===")
    print("  %s：毛 %+.3f、掛單淨 %+.3f bps/h、SE %.3f、換手 %.3f"
          % (best, v["gross_bps_h"], v["net_maker"], v["se_bps_h"],
             v["turnover"]))
    print("  **臂是事後挑的，所以這不是判決**（mistake.md 2026-09-09：事後找到")
    print("  的維度通過多少一致性檢查都不算數）。它只回答「值不值得繼續」。")
    print("  要變成判決必須**只用前半挑臂**，再看後半。")
    res["best_arm_posthoc"] = best

    print("\n=== 讀法（核心原則 9）===")
    print("  **這不是 alpha 的判決，是 Gate 0 的分子。** 單一樣本、沒有樣本外")
    print("  切分，所以它只能回答「毛利有沒有大到值得繼續」。")
    print("  要升級成判決必須走 factor-research.md 的十道。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
