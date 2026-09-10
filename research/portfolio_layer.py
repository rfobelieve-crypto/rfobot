# -*- coding: utf-8 -*-
"""組合層：N 條線的相關矩陣、分散比、組合層 CI、逐線邊際貢獻（2026-09-10）

§1.04 已經對三條線做過這件事，但寫死三條。這支把它通用化，並且補上
§1.04 沒有的兩樣東西：**組合層的信賴區間**與**逐線的邊際貢獻**（留一法）。

===========================================================================
要回答的問題，以及它為什麼值得問
===========================================================================
多條弱訊號疊加是業界的實際做法，而它的數學是：N 條**獨立**且各有真實
優勢的線，組合的 t 值大約隨 sqrt(N) 成長。**所以單獨測不出來的東西，
疊起來可能測得出來。**

這正好對上這個專案的處境：一堆線的點估計為正但落在尺的解析度以下
（SDV +0.1833 但 CI 下緣 −0.132、vol_burst、liq_burst、POC…），
它們被標成「測不動」而不是「是零」，而這兩件事在組合框架下的待遇不同。

**但 sqrt(N) 只在獨立時成立。** 而這裡每一條線都建在同一批掃單事件、
同一批幣、同一批日子上。所以第一件事不是疊加，是**量相關矩陣**：
若多數兩兩相關在 0.6 以上，sqrt(N) 就縮成 sqrt(2)，這條路沒有想的那麼寬。

===========================================================================
三個誠實標籤（寫在這裡，報告裡也要出現）
===========================================================================
1. **母體重疊不是 bug，是這批線的本質。** 下面的線有好幾條是同一批
   9,262 筆掃單的不同切法（S∧D∧V / S∧D / S∧V / 零旗標 / 全體）。
   它們**不可以同時下注**——那是同一筆錢押在同一個事件上好幾次。
   相關矩陣的用途是量「它們有多像」，不是建一個真的要跑的組合。
2. **單位與成本都不一致。** V7 是毛的報酬比例、SDV 是已扣成本的 ATR、
   舊線是訊號層的災難停損單位。所以**組合層的均值不是可交易的宣稱**。
   而且成本不會被分散化解決——每條線各付自己的那一份，疊十條付十次。
3. **分散化降低變異數，不創造均值。** 十條期望值為零的線疊起來，
   得到的是期望值為零、變異數較小的一條線。所以「點估計為正但測不動」
   跟「量出來是零」必須分開看，前者才是候選池。

===========================================================================
線的來源（全部來自既有產物，沒有任何新假設）
===========================================================================
    V7          tracked_signals 的 Strong（凍結在 tests/fixtures）
    OLD         掃單失敗反轉（凍結在 tests/fixtures；**交易設計已判不可執行**，
                這裡只是訊號層序列）
    SDV         S∧D∧V，conj_backtest.ledger 的 sigk=="and"
    S+D         只有 delta_ext（sigk=="d"）
    S+V         只有 vol_burst（sigk=="v"）
    SWEEP_all   全部 9,262 筆掃單，動能方向（快照 y_with_d5）
    NOFLOW      沒有任何流量旗標的掃單（快照 sig=="sweep"，6,251 筆）

輸出 results/portfolio_layer.json。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research", ROOT / "research" / "poc"):
    sys.path.insert(0, str(p))

from research.harness import boot_corr, boot_days, day8, write_report  # noqa: E402

FIX = ROOT / "research" / "tests" / "fixtures"
SNAP = ROOT / "research" / "poc" / "data" / "sweep_snapshot.parquet"
OUT = ROOT / "research" / "results" / "portfolio_layer.json"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


# ─────────────────────────── 線 ───────────────────────────

def _fixture(name):
    return pd.read_parquet(FIX / f"book_{name}.parquet")[["sym", "entry_ms", "r"]]


def _sdv_arms():
    import conj_backtest as cb
    rows = []
    for s in CORE9:
        trs, _ = cb.ledger(s)
        rows += [dict(sym=s, entry_ms=t["entry_ts"], r=t["R_net"],
                      k=t["sigk"]) for t in trs]
    d = pd.DataFrame(rows)
    return {"SDV": d[d.k == "and"], "S+D": d[d.k == "d"], "S+V": d[d.k == "v"]}


def _snapshot_arms():
    d = pd.read_parquet(SNAP, columns=["sym", "ts", "sig", "y_with_d5"])
    d = d[d.sym.isin(CORE9)].rename(columns={"ts": "entry_ms", "y_with_d5": "r"})
    return {"SWEEP_all": d, "NOFLOW": d[d.sig == "sweep"]}


def lines():
    out = {"V7": _fixture("v7"), "OLD": _fixture("old")}
    out.update(_sdv_arms())
    out.update(_snapshot_arms())
    return {k: v[["sym", "entry_ms", "r"]].reset_index(drop=True)
            for k, v in out.items()}


# ─────────────────────── 組合層 ───────────────────────

def daily(df, days):
    g = df.assign(_d=day8(df.entry_ms)).groupby("_d").r.sum()
    return g.reindex(days, fill_value=0.0)


def main():
    L = lines()
    lo = max(int(day8([v.entry_ms.min()])[0]) for v in L.values())
    hi = min(int(day8([v.entry_ms.max()])[0]) for v in L.values())
    days = np.arange(lo, hi + 1)
    S = pd.DataFrame({k: daily(v, days) for k, v in L.items()})
    names = list(S.columns)

    print("窗口 %s ~ %s（%d 天，UTC+8；交集由 V7 的起點決定）"
          % (pd.to_datetime(lo * 86400000 - 8 * 3600000, unit="ms").date(),
             pd.to_datetime(hi * 86400000 - 8 * 3600000, unit="ms").date(),
             len(days)))
    print()
    print("%-10s %7s %8s %9s %9s %8s" % ("線", "窗內筆數", "有交易日",
                                         "日均", "日sigma", "開火率"))
    per_line = {}
    for k in names:
        sub = L[k]
        m = (day8(sub.entry_ms) >= lo) & (day8(sub.entry_ms) <= hi)
        n = int(m.sum())
        nd = int((S[k] != 0).sum())
        per_line[k] = dict(n=n, n_days=nd, mean=float(S[k].mean()),
                           std=float(S[k].std()))
        print("%-10s %7s %8s %+9.4f %9.4f %7.0f%%"
              % (k, format(n, ","), format(nd, ","), S[k].mean(), S[k].std(),
                 100 * nd / len(days)))

    print()
    print("相關矩陣（Pearson，窗內全部日曆日含 0）")
    print("%-10s" % "" + "".join("%9s" % k[:8] for k in names))
    C = S.corr()
    for a in names:
        print("%-10s" % a[:9] + "".join(
            ("%9s" % "—") if a == b else ("%+9.2f" % C.loc[a, b])
            for b in names))

    off = [C.loc[a, b] for i, a in enumerate(names) for b in names[i + 1:]]
    print()
    print("兩兩相關：中位 %+.2f、最大 %+.2f、最小 %+.2f、>=0.6 的比例 %.0f%%"
          % (np.median(off), max(off), min(off),
             100 * np.mean([abs(x) >= 0.6 for x in off])))

    # 等波動權重（單位不同，必須先各自標準化）
    Z = S / S.std()
    w = pd.Series(1.0 / len(names), index=names)
    port = (Z * w).sum(axis=1)
    dr = float(1.0 / port.std())
    n_eff = dr ** 2
    print()
    print("分散比 DR = %.3f   （1 = 同一個賭注；sqrt(%d) = %.3f = 完全獨立）"
          % (dr, len(names), np.sqrt(len(names))))
    print("**有效獨立線數 = DR^2 = %.2f**（名目 %d 條）" % (n_eff, len(names)))

    print()
    print("組合層（等波動權重；**不是可交易的宣稱**，見檔頭標籤 2）")
    pm, pse, plo, pp = boot_days(days, port.to_numpy())
    print("  日均 %+.4f  SE %.4f  CI下緣 %+.4f  P(>0) %.0f%%"
          % (pm, pse, plo, 100 * pp))

    print()
    print("逐線邊際貢獻（留一法：拿掉它，組合的 t 值變多少）")
    def tval(cols):
        z = (S[cols] / S[cols].std())
        p = (z * (1.0 / len(cols))).sum(axis=1)
        m, se, _l, _p = boot_days(days, p.to_numpy())
        return (m / se) if se and np.isfinite(se) and se > 0 else np.nan
    t_all = tval(names)
    marg = {}
    print("  全體 t = %+.3f" % t_all)
    for k in names:
        rest = [x for x in names if x != k]
        t_wo = tval(rest)
        marg[k] = float(t_all - t_wo)
        print("    拿掉 %-10s t -> %+.3f   邊際 %+.3f" % (k, t_wo, t_all - t_wo))

    res = dict(window=[str(pd.to_datetime(lo * 86400000 - 8 * 3600000,
                                          unit="ms").date()),
                       str(pd.to_datetime(hi * 86400000 - 8 * 3600000,
                                          unit="ms").date())],
               days=int(len(days)), lines=per_line,
               corr={a: {b: float(C.loc[a, b]) for b in names} for a in names},
               corr_offdiag=dict(median=float(np.median(off)),
                                 max=float(max(off)), min=float(min(off)),
                                 frac_ge_06=float(np.mean(
                                     [abs(x) >= 0.6 for x in off]))),
               dr=dr, n_effective=float(n_eff),
               portfolio=dict(mean=pm, se=pse, ci_lo=plo, p_pos=pp,
                              t=float(t_all)),
               marginal_t=marg,
               caveats=["母體重疊：好幾條是同一批 9,262 筆掃單的不同切法，"
                        "不可同時下注",
                        "單位與成本不一致（V7 毛、SDV 淨、OLD 訊號層），"
                        "組合均值不是可交易的宣稱",
                        "成本不會被分散化解決：每條線各付自己那一份"])
    p = write_report(OUT, res)
    print("\nwritten -> " + str(p))


if __name__ == "__main__":
    main()
