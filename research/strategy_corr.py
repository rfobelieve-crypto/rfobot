# -*- coding: utf-8 -*-
"""三條策略線是不是同一個賭注：相關矩陣與組合層分解（2026-09-10 預註冊）

**為什麼做這個**：這個專案講過一句話 ——「三條線其實是同一門生意，所以
沒有分散」—— 但那句話**從來沒有被量過**。所有判決都是單策略的
（這條線過不過），沒有任何數字回答「它對整體的邊際貢獻是多少」。

這不是新研究線，是把既有的三本帳放到同一根時間軸上算一次相關。

===========================================================================
資料來源與它們各自的誠實標籤（跑之前寫死，不事後補）
===========================================================================
    V7      `tracked_signals` 的 Strong 訊號，每筆 = sign(方向) x 4h 報酬
            **毛的**（不含成本）。而且 2026-04-03 起 Strong 的定義從
            「13~18% 的 bar」換成「rolling top 5%」—— 窗內含這條語意分界，
            所以另跑一次 04-03 之後的子窗當敏感度。
    SDV     `conj_backtest.ledger` 的 sigk=="and"，每筆 R_net（ATR 單位、
            **已扣成本**）。
    舊線    `sweep_core.backtest_symbol` 的 R（災難停損單位、含 SLIP）。
            **這條線 2026-09-07 判為交易設計不可執行**（成交價拿不到）。
            所以它在這裡是**訊號層序列不是損益序列**，只能回答「它跟別人
            同不同步」，不能回答「把它加進組合會賺多少」。
    套利    只錄不做、且在另一個 repo，沒有交易序列 -> 不納入。
    撤單流  方向性判決 FAIL，沒有交易 -> 不納入。

===========================================================================
方法（凍結）
===========================================================================
    窗口     三條線都有資料的交集（V7 起點是綁的：2025-11-16）
    日界     UTC+8（與 §1.03m 的統一決定一致）
    歸屬     **進場日**。敏感度另報出場日。
             理由：要問的是「同一天的市場讓它們同時賺或同時賠嗎」，
             那是進場時點的事；持有 4~8 小時，兩者多半同日。
    序列     每天把當日所有交易的報酬**相加**（不是平均）—— 組合看的是
             當天總損益，一天開五筆就是五筆的曝險。
    沒交易   記 0，不是 NaN。空手的那天對組合的貢獻就是 0。
    相關     Pearson 與 Spearman 各報，且分兩種樣本：
               (a) 窗內全部日曆日（含 0）  <- 組合層要看的
               (b) 兩邊當天都有交易的日    <- 訊號層同不同步
             兩者會不一樣，而不一樣本身是資訊：(a) 低而 (b) 高 =
             「很少同時開火，但同時開火時同方向」。
    CI       對「日」重抽 2000 次（日就是獨立單位，不需要再分塊）
    分散     等風險權重 w_i 正比 1/sigma_i，
             分散比 DR = sum(w_i sigma_i) / sigma_portfolio
             DR = 1 -> 完全同一個賭注；DR = sqrt(3) = 1.732 -> 三條完全獨立

===========================================================================
自曝檢查（答案已知，對不上就是我這支腳本錯，不是結論）
===========================================================================
    S1  SDV 全歷史交易數必須是 1,584（現行 ledger 的凍結值）
    S2  舊線九幣合計必須是 7,083（sweep_core docstring 釘住的值）
    S3  每條線自己的日序列總和，必須等於它窗內逐筆報酬的總和（歸屬不漏筆）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "poc"))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

TZ_MS = 8 * 3600 * 1000
SEED = 20260910
NBOOT = 2000
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
DECODE_TOP5 = pd.Timestamp("2026-04-03")     # Strong 定義換成 top-5% 的那天
OUT = ROOT / "research" / "results" / "strategy_corr.json"
NAMES = ("V7", "SDV", "OLD")                 # OLD = 流動性獵取舊線


def day8(ms):
    return (np.asarray(ms, np.int64) + TZ_MS) // 86_400_000


def load_v7():
    from shared.db import get_db_conn
    c = get_db_conn()
    cur = c.cursor()
    cur.execute("select signal_time, direction, actual_return_4h "
                "from tracked_signals where strength='Strong' "
                "and actual_return_4h is not null order by signal_time")
    rows = cur.fetchall()
    c.close()
    d = pd.DataFrame(rows)
    ms = d["signal_time"].astype("datetime64[ns]").astype("int64") // 10 ** 6
    # 訊號是 bar 標籤，開火在 label+1h（mistake.md 2026-07-28）
    d["entry_ms"] = ms + 3_600_000
    d["exit_ms"] = d["entry_ms"] + 4 * 3_600_000
    d["r"] = np.where(d["direction"] == "UP", 1.0, -1.0) * d["actual_return_4h"]
    return d[["entry_ms", "exit_ms", "r"]].copy()


def load_sdv():
    import conj_backtest as cb
    rows = []
    for s in CORE9:
        trs, _ = cb.ledger(s)
        rows += [dict(entry_ms=t["entry_ts"], exit_ms=t["exit_ts"], r=t["R_net"])
                 for t in trs if t["sigk"] == "and"]
    return pd.DataFrame(rows)


def load_old():
    import sweep_core as SC
    cache = ROOT / "research" / "sweep_failure" / ".cache"
    rows = []
    for s in CORE9:
        p = cache / (s + "USDT_1h.csv")
        if not p.exists():
            print("  ** missing " + p.name)
            continue
        for t in SC.backtest_symbol(SC.load_csv(str(p))):
            rows.append(dict(entry_ms=int(t[0]), exit_ms=int(t[1]), r=float(t[2])))
    return pd.DataFrame(rows)


def daily(df, days_index, on="entry_ms", val="r"):
    g = df.assign(_d=day8(df[on])).groupby("_d")[val].sum()
    return g.reindex(days_index, fill_value=0.0)


def boot_corr(x, y, n=NBOOT, method="pearson"):
    rng = np.random.default_rng(SEED)
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    m = len(xs)
    if m < 10:
        return float("nan"), float("nan")
    idx = rng.integers(0, m, size=(n, m))
    out = np.empty(n)
    for i in range(n):
        a, b = xs[idx[i]], ys[idx[i]]
        if method == "spearman":
            a = pd.Series(a).rank().to_numpy()
            b = pd.Series(b).rank().to_numpy()
        if a.std() == 0 or b.std() == 0:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(a, b)[0, 1]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def block(books, days, anchor, label, res):
    S = pd.DataFrame({k: daily(v, days, anchor) for k, v in books.items()})
    ACT = pd.DataFrame({k: daily(v.assign(one=1.0), days, anchor, "one")
                        for k, v in books.items()})
    lo, hi = days[0], days[-1]
    s3 = True
    for k, v in books.items():
        m = (day8(v[anchor]) >= lo) & (day8(v[anchor]) <= hi)
        if abs(float(S[k].sum()) - float(v.loc[m, "r"].sum())) > 1e-6:
            s3 = False

    print("\n" + "=" * 76)
    print("歸屬 = " + label + "    S3 逐筆總和守恆 "
          + ("PASS" if s3 else "**FAIL**"))
    print("%-6s%9s%9s%8s%11s%10s" % ("線", "窗內筆數", "有交易日", "開火率",
                                     "日均", "日sigma"))
    for k in books:
        n = int(ACT[k].sum())
        nd = int((ACT[k] > 0).sum())
        print("%-6s%9s%9s%7.0f%%%+11.4f%10.4f"
              % (k, format(n, ","), format(nd, ","), 100.0 * nd / len(days),
                 S[k].mean(), S[k].std()))

    print("\n相關（Pearson；全日含 0 共 %d 天）" % len(days))
    cell = {}
    for a, b in (("V7", "SDV"), ("V7", "OLD"), ("SDV", "OLD")):
        pa = float(S[a].corr(S[b]))
        sp = float(S[a].corr(S[b], method="spearman"))
        ci = boot_corr(S[a], S[b])
        both = (ACT[a] > 0) & (ACT[b] > 0)
        nb = int(both.sum())
        pb = float(S.loc[both, a].corr(S.loc[both, b])) if nb > 9 else float("nan")
        cib = boot_corr(S.loc[both, a], S.loc[both, b]) if nb > 9 else (float("nan"),) * 2
        cell[a + "|" + b] = dict(pearson_all=pa, spearman_all=sp, ci_all=list(ci),
                                 n_both=nb, pearson_both=pb, ci_both=list(cib))
        print("  %-4s~%-4s 全日 r=%+.3f CI[%+.3f,%+.3f] rho=%+.3f   |   "
              "同開火 %3d 天 r=%+.3f CI[%+.3f,%+.3f]"
              % (a, b, pa, ci[0], ci[1], sp, nb, pb, cib[0], cib[1]))

    sig = S.std()
    w = (1.0 / sig) / (1.0 / sig).sum()
    port = (S * w).sum(axis=1)
    dr = float((w * sig).sum() / port.std())
    print("\n等風險權重 " + ", ".join("%s %.2f" % (k, w[k]) for k in w.index))
    print("  分散比 DR = %.3f   （1.000 = 完全同一個賭注；1.732 = 三條完全獨立）"
          % dr)
    cov = S.cov()
    mrc = {k: float(w[k] * float((cov.loc[k] * w).sum()) / port.var()) for k in S}
    print("  風險貢獻：" + ", ".join("%s %.0f%%" % (k, 100 * v)
                                 for k, v in mrc.items()))
    res[label] = dict(corr=cell, dr=dr, risk_contrib=mrc,
                      weights={k: float(w[k]) for k in w.index},
                      daily_mean={k: float(S[k].mean()) for k in S},
                      daily_std={k: float(S[k].std()) for k in S},
                      fire_rate={k: float((ACT[k] > 0).mean()) for k in S},
                      s3_pass=bool(s3))
    return S, ACT


def main():
    print("載入三本帳…")
    books = {"V7": load_v7(), "SDV": load_sdv(), "OLD": load_old()}
    for k, v in books.items():
        print("  %-4s %7s 筆   %s ~ %s"
              % (k, format(len(v), ","),
                 pd.to_datetime(v.entry_ms.min(), unit="ms").date(),
                 pd.to_datetime(v.entry_ms.max(), unit="ms").date()))

    n_sdv, n_old = len(books["SDV"]), len(books["OLD"])
    ok1, ok2 = n_sdv == 1584, n_old == 7083
    print()
    print("S1 SDV 全歷史 %s（應為 1,584）  %s"
          % (format(n_sdv, ","), "PASS" if ok1 else "**FAIL**"))
    print("S2 舊線九幣 %s（應為 7,083）  %s"
          % (format(n_old, ","), "PASS" if ok2 else "**FAIL**"))

    lo = max(int(day8([v.entry_ms.min()])[0]) for v in books.values())
    hi = min(int(day8([v.entry_ms.max()])[0]) for v in books.values())
    days = np.arange(lo, hi + 1)
    print("\n交集窗口 %s ~ %s（%d 天，UTC+8 日界）"
          % (pd.to_datetime(lo * 86400000 - TZ_MS, unit="ms").date(),
             pd.to_datetime(hi * 86400000 - TZ_MS, unit="ms").date(), len(days)))

    res = dict(window_days=int(len(days)), S1_sdv_n=n_sdv, S2_old_n=n_old,
               S1_pass=bool(ok1), S2_pass=bool(ok2),
               window=[str(pd.to_datetime(lo * 86400000 - TZ_MS, unit="ms").date()),
                       str(pd.to_datetime(hi * 86400000 - TZ_MS, unit="ms").date())])

    block(books, days, "entry_ms", "進場日", res)
    block(books, days, "exit_ms", "出場日", res)

    cut = int(day8([int(DECODE_TOP5.value // 10 ** 6)])[0])
    sub = days[days >= cut]
    if len(sub) > 30:
        print("\n" + "=" * 76)
        print("敏感度：只用 2026-04-03 之後（V7 Strong 同一種 top-5% 定義），%d 天"
              % len(sub))
        S = pd.DataFrame({k: daily(v, sub, "entry_ms") for k, v in books.items()})
        sens = {}
        for a, b in (("V7", "SDV"), ("V7", "OLD"), ("SDV", "OLD")):
            r = float(S[a].corr(S[b]))
            ci = boot_corr(S[a], S[b])
            sens[a + "|" + b] = dict(pearson=r, ci=list(ci))
            print("  %-4s~%-4s r=%+.3f CI[%+.3f,%+.3f]" % (a, b, r, ci[0], ci[1]))
        sig = S.std()
        w = (1.0 / sig) / (1.0 / sig).sum()
        dr2 = float((w * sig).sum() / (S * w).sum(axis=1).std())
        print("  分散比 DR = %.3f" % dr2)
        res["post_top5"] = dict(days=int(len(sub)), dr=dr2, corr=sens)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print("\nwritten -> " + str(OUT))


if __name__ == "__main__":
    main()
