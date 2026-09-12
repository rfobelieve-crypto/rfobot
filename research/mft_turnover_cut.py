# -*- coding: utf-8 -*-
"""R4：砍換手（2026-09-12）—— 而 R1 把這一題重新寫了一次

===========================================================================
R1 之後，R4 的問法變了
===========================================================================
原本 R4 是「把換手砍下來，讓手續費變小」。R1（TODO §1.27）量完之後，
**兩條執行路線的成本都線性隨 Σ|Δw| 走**：

    吃單淨/量 = 毛利/量 − 3.00                   （Bitget 返佣後）
    掛單淨/量 = 毛利/量 − 被動代價(7.1~7.6) − 1.00

所以「淨值為正」可以整理成**一個跟規模無關的比值**（每單位成交量賺幾 bps）：

    吃單要活：毛利 / Σ|Δw|  >  **3.0 bps**
    掛單要活：毛利 / Σ|Δw|  >  **約 8.5 bps**（被動代價 ~7.5 + 掛單費 1.0）

**現況（60 天、new5、符號 −1）：+0.52 bps 每單位成交量。** 離 3.0 差 5.7 倍。

> 這幾個數字在 2026-09-12 當天被修正過一次：§1.27 第一版把「每**有效期**的
> 毛利」減掉「每**全部期**攤的手續費」，兩個不同的分母，費被低估約 1.2~1.5 倍，
> 被動代價因此被高估到 7.5~9.6。改成一律用**每單位成交量**之後（費率本來
> 就是按成交量收的，跟有幾期能評分無關）落在 **7.1~7.6**。見 §1.27 的更正框。

於是 R4 不是「降成本」，是**把每單位換手的毛利拉上去**。抑制換手只有在
**毛利掉得比換手慢**的時候才有用 —— 那是本支要量的唯一一件事。
這個比值（bps per unit of turnover）是本支的主指標，不是淨值本身，
因為淨值還混著「這個窗的訊號剛好多強」。

===========================================================================
旋鈕（全部是既有實作，本支一行都不重寫）
===========================================================================
`gate0_xs_turnover` 是這些旋鈕的擁有者，而 §1.24 已經量過它們的**成本側**
（那支檔頭寫著「刻意只算成本，不算損益」）。本支補的正是缺的那一半。

  · EMA `weights_from(p, halflife=h)`：先平滑特徵再做橫斷面 z
  · 部分再平衡 `partial_rebalance(w, λ)`：每期只往目標走 λ 的比例

> **原本登記的是「不交易帶」`apply_band`，而 C2 當場抓到那個旋鈕是病的**
> （2026-09-12）：抑制後重新縮放 -> 縮放乘到每個名字上，「不交易」的也動了，
> 換手幾乎沒降（0.826 -> 0.770）且不單調；不重新縮放 -> 棘輪，Σ|w| 爬到
> **1.313**、換手反而**升到 0.92**。兩種寫法都量不到「少交易」。
> 換成部分再平衡不是放寬判準 —— C2 照舊，換的是旋鈕。理由與構造上的
> 兩個保證寫在 `partial_rebalance` 的 docstring。

第三個是使用者登記的構造（**不是抑制旋鈕，是換構造**，所以分開標）：

  · `tanh(時序z(特徵, 24h) / 3)` —— 用**時序** z 分數取代橫斷面 z，
    再用 tanh 壓尾。它同時做兩件事（換正規化軸 + 壓極值），
    所以它跟上面兩個不可比，表上會分區。

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
C1 **自曝**：`無抑制` 那一列的毛利必須逐格等於 §1.23 的式子
   （同一份資料重算，不是讀凍結檔）。不等於就是我接錯了，以下不解讀。
C2 **自曝**：每一族之內，Σ|Δw| 必須隨抑制強度單調下降。
   不單調 = 旋鈕接錯了。
C3 **誠實挑選**：用**前半**挑一個旋鈕設定（按「毛利/Σ|Δw|」最大），
   **報前半挑到什麼**，再看它在後半的值。不得報「我挑的那個在後半如何」
   （mistake.md 2026-09-09：唯一算數的驗證是把挑選程序放進樣本外）。
C4 **過關門檻**：被挑到的那一格在**後半**的「毛利/Σ|Δw|」CI 下緣 > 3.0。
   同時照核心原則 10 一併報 `P(比值 > 3.0)`。
MDE 先印再報結果：若 SE ≥ (3.0 − 現況) 的一半，**這個設計分辨不出來**，
   那就是 inconclusive by design，不是 FAIL（mistake.md 2026-09-04）。

    python research/mft_turnover_cut.py --days 120 --col new5 --sign -1
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
from research import mft_fill_conditional as FC                # noqa: E402

OUT = ROOT / "research" / "results" / "mft_turnover_cut.json"
RNG = np.random.default_rng(20260912)
# 部分再平衡的 λ（1 = 原始）。取代不交易帶 —— 那個旋鈕本身是病的，
# 理由寫在 gate0_xs_turnover.partial_rebalance 的 docstring。
LAMBDAS = (0.75, 0.5, 0.25, 0.1)
HALFLIVES = (1, 2, 4, 8, 24)


def ts_tanh_weights(p: pd.DataFrame, win: int = 24, div: float = 3.0):
    """使用者登記的構造：tanh(時序z(特徵, win) / div)，再縮到總槓桿 1。

    **與橫斷面 z 是不同的估計量**（正規化的軸不同），所以它在表上分區，
    不跟抑制旋鈕並列比較。`min_periods=win` -> 前 win 列是 NaN，
    那是 trailing-only 的代價，不是 bug。
    """
    mu = p.rolling(win, min_periods=win).mean()
    sd = p.rolling(win, min_periods=win).std().replace(0, np.nan)
    z = (p - mu) / sd
    w = np.tanh(z / div)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)


def boot_ratio(num, den, days, b=3000):
    """日聚類 bootstrap 的「總毛利 / 總換手」—— 比值要整體重抽，不可逐項平均。

    逐期算比值再平均會被換手很小的那幾期炸掉（分母趨零）。所以重抽之後
    **先各自加總再相除**，那才是「每單位換手賺多少」的定義。
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    days = np.asarray(days)          # 日期是字串，**不要一起轉 float**
    ok = np.isfinite(num) & np.isfinite(den)
    num, den, days = num[ok], den[ok], days[ok]
    if len(num) < 20 or den.sum() <= 0:
        return (np.nan,) * 5
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        pick = RNG.integers(0, len(uq), len(uq))
        sel = np.concatenate([ix[k] for k in pick])
        d = den[sel].sum()
        r[i] = num[sel].sum() / d if d > 0 else np.nan
    pt = num.sum() / den.sum()
    return (float(pt), float(np.nanstd(r, ddof=1)),
            float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5)),
            float(np.mean(r > 3.0)))


def arm_stats(w, fwd, days_all):
    """回傳逐期的 (毛利貢獻, 換手) 兩條序列 —— 比值的分子與分母。"""
    both = w.notna() & fwd.notna()
    g = (w.where(both) * fwd.where(both)).sum(axis=1, min_count=1) * 1e4
    # Σ|Δw|（雙邊），成本的乘數。**NaN 權重當 0（空倉）**，因為 §1.23 的
    # 估計量就是這樣算毛利的（NaN 那一格貢獻 0）—— 分子說空倉、分母說還抱著
    # 會是兩個故事。`fillna(0)` 之後「回到空倉」才會被算成一筆真的交易。
    dw = w.fillna(0.0).diff().abs().sum(axis=1)
    idx = g.dropna().index.intersection(dw.dropna().index)
    return g.loc[idx], dw.loc[idx], days_all.loc[idx]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--col", default="new5")
    ap.add_argument("--sign", type=float, default=-1.0,
                    help="§1.23b 誠實挑選挑到 new5 的符號是 -1")
    ap.add_argument("--venue", default="bitget")
    ap.add_argument("--rebate", type=float, default=None)
    a = ap.parse_args()

    tk, mk, src = XS.fees_for(a.venue, a.rebate)
    NEED_TAKER = tk                     # 毛利/換手 要超過的門檻（吃單）
    print("=== R4：砍換手 —— 但真正要拉的是「每單位換手的毛利」 ===")
    print("場館 %s（%s）：吃單 %.2f / 掛單 %.2f bps 每邊" % (a.venue, src, tk, mk))
    print("吃單要活的門檻：毛利 / Σ|Δw| > **%.2f bps**\n" % NEED_TAKER)

    d = XS.load_pairs(a.days)
    f = XS.build(d)
    if f is None or not len(f):
        print("沒有再平衡點，停。")
        return 2
    feat = f.pivot_table(index="minute", columns="sym", values=a.col)
    midp = f.pivot_table(index="minute", columns="sym", values="mid")
    base = XS.weights_from(feat) * float(a.sign)
    midp = midp.reindex(index=base.index, columns=base.columns)
    fwd = midp.shift(-1) / midp - 1.0
    days_all = pd.to_datetime(pd.Series(base.index) * 60000, unit="ms", utc=True) \
                 .dt.strftime("%Y-%m-%d")
    days_all.index = base.index
    print("再平衡 %d 次、%d 標的、臂 %s、符號 %+.0f\n"
          % (len(base), len(base.columns), a.col, a.sign))

    # ── 組臂 ───────────────────────────────────────────────────────
    arms = [("無抑制", "base", base)]
    for lm in LAMBDAS:
        arms.append(("部分再平衡 λ=%.2f" % lm, "partial",
                     XS.partial_rebalance(base, lm)))
    for h in HALFLIVES:
        arms.append(("EMA 半衰期 %dh" % h, "ema",
                     XS.weights_from(feat, halflife=h) * float(a.sign)))
    arms.append(("tanh(時序z24/3)", "tanh",
                 ts_tanh_weights(feat) * float(a.sign)))

    res = {"asof": time.strftime("%Y-%m-%d %H:%M:%S"), "days": a.days,
           "col": a.col, "sign": float(a.sign), "venue": a.venue,
           "need_ratio_taker": NEED_TAKER, "arms": {}}

    # ---------- C1 自曝 ----------
    print("=== C1 自曝：『無抑制』必須逐格重現 §1.23 的毛利 ===")
    g0, dw0, dy0 = arm_stats(base, fwd, days_all)
    both = base.notna() & fwd.notna()
    ref = (base.where(both) * fwd.where(both)).sum(axis=1, min_count=1).dropna() * 1e4
    al = ref.align(g0, join="inner")
    worst = float((al[0] - al[1]).abs().max()) if len(al[0]) else np.inf
    ok1 = len(g0) == len(ref) and worst < 1e-9
    print("  期數 %d vs %d、逐期最大差 %.3e -> %s"
          % (len(g0), len(ref), worst, "PASS" if ok1 else "**FAIL**"))
    res["C1"] = bool(ok1)
    if not ok1:
        print("\n**C1 未過 —— 接錯了，以下不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 2

    # ---------- MDE 先印 ----------
    pt0, se0, lo0, hi0, pp0 = boot_ratio(g0, dw0, dy0)
    gap = NEED_TAKER - pt0
    print("\n=== MDE 先講（不是跑完才補）===")
    print("  現況 毛利/Σ|Δw| = %.3f bps（CI [%.3f, %.3f]、SE %.3f）"
          % (pt0, lo0, hi0, se0))
    print("  要到 %.2f 還差 %.3f bps" % (NEED_TAKER, gap))
    decidable = se0 < gap / 2.0
    print("  SE %.3f vs 缺口的一半 %.3f -> %s"
          % (se0, gap / 2.0,
             "這個設計分辨得出來" if decidable
             else "**SE 太大：就算某一格看起來到了，也分辨不出真假**"))
    res["mde"] = dict(now=pt0, se=se0, need=NEED_TAKER, gap=gap,
                      decidable=bool(decidable))

    # ---------- 全格報告 ----------
    print("\n=== 全格報告（不挑）===")
    print("  「@槓桿1」= 除以 Σ|w|。部分再平衡會讓帳本縮小（凸組合的後果），")
    print("  而**常數倍的槓桿對比值是恆等的**（分子分母同時乘），所以比較要看")
    print("  比值；但絕對數要換算到同一個槓桿才讀得懂。")
    print("%-18s %8s %9s %10s %9s %9s %9s %7s %9s %9s"
          % ("設定", "Σ|Δw|", "毛利", "毛利/換手", "CI下", "CI上",
             "P(>%.1f)" % NEED_TAKER, "Σ|w|", "毛利@槓1", "換手@槓1"))
    series = {}
    for name, fam, w in arms:
        g, dw, dy = arm_stats(w, fwd, days_all)
        if not len(g):
            continue
        pt, se, lo, hi, pp = boot_ratio(g, dw, dy)
        # **總槓桿要印出來。** 不交易帶不再重新縮放（見 apply_band 的註解），
        # 所以 Σ|w| 可能漂 —— 漂多少必須看得見，不然等於把一個已知的
        # 失效模式藏起來。1.0 附近才代表各臂的毛利可比。
        lev = float(w.abs().sum(axis=1).replace(0, np.nan).mean())
        series[name] = (g, dw, dy, fam)
        res["arms"][name] = dict(family=fam, turnover=float(dw.mean()),
                                 gross=float(g.mean()), ratio=pt, se=se,
                                 ci_lo=lo, ci_hi=hi, p_above=pp,
                                 gross_leverage=lev, n=int(len(g)),
                                 gross_at_lev1=float(g.mean() / lev) if lev else None,
                                 turnover_at_lev1=float(dw.mean() / lev) if lev else None)
        print("%-18s %8.4f %+9.4f %10.3f %+9.3f %+9.3f %8.1f%% %7.3f %+9.4f %9.4f"
              % (name, dw.mean(), g.mean(), pt, lo, hi, pp * 100, lev,
                 g.mean() / lev if lev else np.nan,
                 dw.mean() / lev if lev else np.nan))

    # ---------- C2 自曝：族內單調 ----------
    print("\n=== C2 自曝：每一族之內 Σ|Δw| 必須隨抑制強度單調下降 ===")
    ok2 = True
    for fam, labels in (("partial", ["無抑制"]
                         + ["部分再平衡 λ=%.2f" % lm for lm in LAMBDAS]),
                        ("ema", ["無抑制"]
                         + ["EMA 半衰期 %dh" % h for h in HALFLIVES])):
        vals = [res["arms"][x]["turnover"] for x in labels if x in res["arms"]]
        mono = all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
        ok2 = ok2 and mono
        print("  %-6s %s -> %s" % (fam, " > ".join("%.4f" % v for v in vals),
                                   "PASS" if mono else "**FAIL**"))
    res["C2"] = bool(ok2)
    if not ok2:
        print("\n**C2 未過 —— 旋鈕接錯了，以下不解讀。**")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return 2

    # ---------- C3 誠實挑選：只用前半挑，報前半挑到什麼 ----------
    print("\n=== C3 誠實挑選：**只用前半挑**，然後看後半 ===")
    idx = list(base.index)
    cut = idx[len(idx) // 2]
    print("  切點 %s（前半 %d 期 / 後半 %d 期）"
          % (pd.to_datetime(cut * 60000, unit="ms", utc=True)
             .strftime("%Y-%m-%d %H:%M"),
             sum(1 for x in idx if x < cut), sum(1 for x in idx if x >= cut)))
    rows = []
    for name, (g, dw, dy, fam) in series.items():
        m1 = g.index < cut
        m2 = ~m1
        r1 = boot_ratio(g[m1], dw[m1], dy[m1])
        r2 = boot_ratio(g[m2], dw[m2], dy[m2])
        rows.append((name, fam, r1, r2))
        res["arms"][name]["first_ratio"] = r1[0]
        res["arms"][name]["second_ratio"] = r2[0]
        res["arms"][name]["second_ci_lo"] = r2[2]
        res["arms"][name]["second_p_above"] = r2[4]
    print("%-18s %11s %11s %11s %9s"
          % ("設定", "前半 比值", "後半 比值", "後半 CI下", "後半P(>門檻)"))
    for name, fam, r1, r2 in rows:
        print("%-18s %11.3f %11.3f %11.3f %8.1f%%"
              % (name, r1[0], r2[0], r2[2], r2[4] * 100))
    # 前半挑（只看前半的點估計），全族一起排 —— tanh 也在候選裡，但它是
    # 換構造不是抑制，所以另外印一行說明它被挑到或沒被挑到。
    pick = max(rows, key=lambda x: (x[2][0] if np.isfinite(x[2][0]) else -1e9))
    pname, pfam, pr1, pr2 = pick
    print("\n  **前半會挑到：%s**（前半 %.3f）" % (pname, pr1[0]))
    print("  它在後半：比值 %.3f、CI [%.3f, %.3f]、P(>%.2f) = %.1f%%"
          % (pr2[0], pr2[2], pr2[3], NEED_TAKER, pr2[4] * 100))
    passed = np.isfinite(pr2[2]) and pr2[2] > NEED_TAKER
    print("  C4 門檻（後半 CI 下緣 > %.2f）-> %s"
          % (NEED_TAKER, "PASS" if passed else "**不過**"))
    # **MDE 要算在被決定的那個東西上。** 上面那一關只算了「無抑制」，
    # 而抑制越強換手越小、比值越吵 —— 用基準臂的 SE 去宣告「這個設計分辨
    # 得出來」，是拿一個比較穩的量去替一個比較吵的量作保（同族：
    # mistake.md 2026-09-06「SE 要用判決那台機器跑出來」）。
    gap2 = NEED_TAKER - pr2[0]
    dec2 = np.isfinite(pr2[1]) and pr2[1] < abs(gap2) / 2.0
    print("  **被挑到那一格自己的 MDE**：後半 SE %.3f、距門檻 %.3f -> %s"
          % (pr2[1], gap2,
             "分辨得出來" if dec2 else
             "**分辨不出來（inconclusive by design，不是 FAIL）**"))
    res["pick_decidable"] = bool(dec2)
    res["pick"] = dict(name=pname, family=pfam, first=pr1[0], second=pr2[0],
                       second_ci_lo=pr2[2], second_ci_hi=pr2[3],
                       second_p_above=pr2[4], C4=bool(passed))

    # ---------- 掛單那一側（用 R1 的模擬器，不重寫） ----------
    print("\n=== 被挑到那一格的掛單側（沿用 R1 的成交模擬器）===")
    book = FC.load_book(a.days)
    by = FC.prep_book(book)
    wpick = dict(arms)[pname] if False else None
    for nm, fam, w in arms:
        if nm == pname:
            wpick = w
            break
    if wpick is not None:
        r = FC.simulate(wpick, fwd, midp, by, 1.0)
        gm = float(r["port"].mean())
        tno = r["dwabs_all"]
        print("  掛單 δ=1 bps：條件毛利 %+.4f、成交率 %.1f%%、淨 %+.4f bps/h"
              % (gm, r["fill_rate"] * 100, gm - mk * tno * r["fill_rate"]))
        print("  （對照吃單淨 %+.4f）"
              % (float(series[pname][0].mean()) - tk * float(series[pname][1].mean())))
        res["pick"]["maker_gross_d1"] = gm
        res["pick"]["maker_fill_rate"] = r["fill_rate"]

    print("\n=== 讀法 ===")
    print("  主指標是**毛利 / Σ|Δw|**，不是淨值 —— 淨值還混著「這個窗的訊號")
    print("  剛好多強」，而比值是「這個訊號每做一單位換手值多少錢」，")
    # 掛單門檻 = 被動代價 + 掛單費，而被動代價要從 §1.27 的判決檔讀，
    # **不要寫死**（它在 2026-09-12 被更正過一次，寫死的數字會過期）。
    fcj = ROOT / "research" / "results" / "mft_fill_conditional.json"
    pen = None
    try:
        pen = json.loads(fcj.read_text(encoding="utf-8")) \
            .get("passive_penalty_bps_per_unit_volume")
    except Exception:
        pass
    need_mk = (float(pen) + mk) if pen is not None else None
    print("  可以直接對門檻 %.2f（吃單）與 %s（掛單）比。"
          % (NEED_TAKER,
             ("%.2f" % need_mk) if need_mk else "（§1.27 的判決檔還沒產生）"))
    if need_mk:
        res["need_ratio_maker"] = need_mk
    print("  抑制換手只有在**毛利掉得比換手慢**時才有用；比值沒有上升，")
    print("  就代表這條訊號的資訊本來就住在高頻那一段，平滑掉的是訊號不是雜訊。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("\nwritten -> %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
