# -*- coding: utf-8 -*-
"""SDV 的問題到底在哪：描述性診斷，不是判決（2026-09-10）

使用者問「SDV 的問題在哪裡，是市場結構嗎」。這支不測任何假設、不挑任何
子集，只把三件可以直接量的事量出來，好讓那個問題有數字可以回答：

  A  報酬的形狀  —— 去掉最極端的 x% 之後還剩多少。
                   如果去尾就轉負，代表均值是由少數尾部事件決定的，
                   那麼「估得準」需要的樣本量遠大於「均值是多少」本身。
  B  解析度      —— 日聚類 bootstrap 的 SE，以及在這個 SE 之下
                   「把母體切兩半比較」能偵測到的最小差異（MDE）。
                   拿它對照已經測過的那些因子的效應量。
  C  清算資料的可用範圍 —— 九幣小時級 2026-03-11 起。算出落在這個窗裡的
                   SDV 事件數，以及那個 n 下的 MDE。這決定「連環爆」
                   這個假設現在到底測不測得動。

**沒有任何判準，因為沒有在判任何東西。** 這是為了回答「問題在哪」而做的
量測；任何要拿去改規格的東西都要另外預註冊。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import conj_backtest as cb  # noqa: E402
import conj_clock as ck  # noqa: E402

SEED = 20260910
NBOOT = 4000
OUT = HERE / "data" / "results" / "sdv_diagnose.json"
LIQ_T0 = pd.Timestamp("2026-03-11", tz="UTC")


def boot(days, vals, n=NBOOT):
    """日聚類 bootstrap：回傳 (均值, SE, CI 下緣, P(均值>0))。"""
    rng = np.random.default_rng(SEED)
    by = {}
    for d, v in zip(days, vals):
        by.setdefault(int(d), []).append(v)
    ks = list(by)
    arr = [np.array(by[k]) for k in ks]
    idx = rng.integers(0, len(ks), size=(n, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean()
                  for i in range(n)])
    return (float(np.mean(vals)), float(o.std(ddof=1)),
            float(np.percentile(o, 2.5)), float((o > 0).mean()))


def main():
    rows = []
    for s in cb.CORE9:
        trs, _ = cb.ledger(s)
        for t in trs:
            if t["sigk"] != "and":
                continue
            rows.append(dict(sym=s, ts=t["entry_ts"], Rn=t["R_net"],
                             stopped=t["stopped"],
                             fwd=t["entry_ts"] >= ck.FREEZE_MS))
    d = pd.DataFrame(rows)
    d["day"] = d.ts // 86_400_000
    mid = float(d.day.median())
    oos = d[d.day >= mid].reset_index(drop=True)
    res = {}

    print("SDV 全歷史 %d 筆，後半（樣本外）%d 筆" % (len(d), len(oos)))
    print("=" * 72)
    print("A 報酬的形狀：去掉最極端的 x% 之後剩下什麼")
    print("%8s %7s %10s %10s %10s" % ("去尾", "n", "全期均值", "後半均值", "後半中位"))
    shape = {}
    for q in (0.0, 0.01, 0.02, 0.05, 0.10):
        def trim(x):
            if q == 0:
                return x
            hi = np.quantile(x.Rn, 1 - q)
            lo = np.quantile(x.Rn, q)
            return x[(x.Rn <= hi) & (x.Rn >= lo)]
        ta, to = trim(d), trim(oos)
        shape["%.0f%%" % (100 * q)] = dict(n_all=len(ta), mean_all=float(ta.Rn.mean()),
                                          n_oos=len(to), mean_oos=float(to.Rn.mean()),
                                          median_oos=float(to.Rn.median()))
        print("%7.0f%% %7d %+10.4f %+10.4f %+10.4f"
              % (100 * q, len(ta), ta.Rn.mean(), to.Rn.mean(), to.Rn.median()))
    res["A_shape"] = shape

    win = d[d.Rn > 0]
    top = d.nlargest(max(1, int(0.05 * len(d))), "Rn")
    print()
    print("  勝率 %.1f%%（後半 %.1f%%）  停損率 %.1f%%"
          % (100 * (d.Rn > 0).mean(), 100 * (oos.Rn > 0).mean(),
             100 * d.stopped.mean()))
    print("  最好的 5%%（%d 筆）貢獻了總報酬的 %.0f%%"
          % (len(top), 100 * top.Rn.sum() / d.Rn.sum()))
    print("  贏家平均 %+.3f / 輸家平均 %+.3f  -> 賠率 %.2f"
          % (win.Rn.mean(), d[d.Rn <= 0].Rn.mean(),
             abs(win.Rn.mean() / d[d.Rn <= 0].Rn.mean())))
    res["A_extra"] = dict(wr_all=float((d.Rn > 0).mean()),
                          wr_oos=float((oos.Rn > 0).mean()),
                          stop_rate=float(d.stopped.mean()),
                          top5_share=float(top.Rn.sum() / d.Rn.sum()))

    print("=" * 72)
    print("B 解析度：SE 與「切兩半比較」的最小可偵測差異")
    lab = {}
    for name, sub in (("全期", d), ("後半（樣本外）", oos)):
        m, se, lo, p = boot(sub.day.values, sub.Rn.to_numpy())
        # 兩個等分桶比較：各 n/2，差值 SE = sqrt(2) x SE(n/2) = 2 x SE(n)
        mde = 1.96 * 2 * se
        lab[name] = dict(n=int(len(sub)), mean=m, se=se, ci_lo=lo, p_pos=p,
                         mde_split=float(mde))
        print("  %-14s n=%4d  均值 %+.4f  SE %.4f  CI下緣 %+.4f  "
              "P(>0) %.0f%%   切兩半 MDE %.3f"
              % (name, len(sub), m, se, lo, 100 * p, mde))
    res["B_resolution"] = lab

    print()
    print("  對照：已經測過的那些因子，它們的分格差是多少")
    print("    OI 崩最兇 10%%          +0.4087 vs 中間 +0.1219  ->  差 0.287")
    print("    分歧變化（置換後作廢）  +0.0772                  ->  差 ~0.08")
    print("    四種池子 / 樞紐尺度     全部落在 0.0x 量級")

    print("=" * 72)
    print("C 清算資料的窗口：九幣小時級，2026-03-11 起")
    t0 = int(LIQ_T0.value // 10 ** 6)
    inw = d[d.ts >= t0]
    m, se, lo, p = boot(inw.day.values, inw.Rn.to_numpy())
    mde = 1.96 * 2 * se
    print("  窗內 SDV 事件 %d 筆（全歷史的 %.0f%%），逐幣 %s"
          % (len(inw), 100 * len(inw) / len(d),
             "/".join(str(int(v)) for v in inw.groupby("sym").size())))
    print("  均值 %+.4f  SE %.4f  ->  **切兩半 MDE %.3f**" % (m, se, mde))
    print("  也就是：要在這個窗裡看出「連環爆 vs 沒連環爆」的差別，")
    print("  那個差必須大於 %.2f ATR 才偵測得到。" % mde)
    print("  （OI 崩那一格量到的差是 0.287 —— %s）"
          % ("測得動" if 0.287 > mde else "**測不動**"))
    res["C_liq_window"] = dict(n=int(len(inw)), mean=m, se=se, ci_lo=lo,
                               p_pos=p, mde_split=float(mde),
                               per_sym={k: int(v) for k, v in
                                        inw.groupby("sym").size().items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print("\nwritten -> " + str(OUT))


if __name__ == "__main__":
    main()
