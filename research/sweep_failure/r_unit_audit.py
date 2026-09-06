# -*- coding: utf-8 -*-
"""主線儀器檢查 — R 這把尺會不會隨事件本身變粗，因而掩蓋結果的集中度？

起因（2026-09-06，POC 管線的副產品）
    凍結引擎算 R 用 `A = atr14[j]`，而 bar j **就是掃單那根**。它的真實區間
    依定義偏大（價格剛突破一個極值），所以：

        risk = DIS x A        停損隨事件自己變寬
        R    = PnL / risk     劇烈事件的結果被自己的分母壓縮

    這**不是前視**（成交在 j 之後，a[j] 當時已知），所以引擎沒有錯。問題是
    別的：**如果 R 的分母隨事件變粗，那麼「哪幾天貢獻了大部分變異」這件事
    在 R 單位下會看不見。** 而這條線的每一個判決都是均值型的 dR / meanR。

    用不含事件那根的分母（atr14[j-1]）重算同一批交易，比兩件事：
      1. 單一最大日佔總變異的比例
      2. 逐日聚類的 meanR 標準誤

    **這支不重判任何東西。** 它只回答「既有判決的 CI 有沒有被這把尺弄窄」。

唯讀研究碼。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_core as sc  # noqa: E402

CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
CACHE = HERE / ".cache"
OUT = HERE.parent / "results"
B = 2000
RNG = np.random.default_rng(20260906)


def backtest_with_atr(bars, atr):
    """sweep_core.backtest_symbol 的逐字複製，只把 ATR 來源換成參數。

    改的只有 `A = atr[j]` 這一行的來源；其餘規則一個字不動。
    """
    n = len(bars)
    h = [b[sc.H] for b in bars]
    l = [b[sc.L] for b in bars]
    c = [b[sc.C] for b in bars]
    trades = []
    last_exit = -1
    for e in sc.detect_sweeps(bars):
        j, lvl = e["j"], e["level"]
        A = atr[j]
        if A is None or A == 0:
            continue
        kd = 1 if e["kind"] == "buy" else -1
        d = -kd
        fill = None
        for f in range(j + 1, min(j + 1 + sc.W, n)):
            if kd == 1 and l[f] <= lvl:
                fill = f
                break
            if kd == -1 and h[f] >= lvl:
                fill = f
                break
        if fill is None or fill <= last_exit or fill + 1 >= n:
            continue
        entry = lvl + d * sc.SLIP * A
        risk = sc.DIS * A
        stop = entry - d * risk
        R = None
        exitbar = min(fill + sc.HOLD, n - 1)
        for k in range(fill + 1, min(fill + sc.HOLD + 1, n)):
            if d == 1 and l[k] <= stop:
                R, exitbar = -1.0 - sc.SLIP / sc.DIS, k
                break
            if d == -1 and h[k] >= stop:
                R, exitbar = -1.0 - sc.SLIP / sc.DIS, k
                break
        if R is None:
            ex = c[exitbar] - d * sc.SLIP * A
            R = d * (ex - entry) / risk
        trades.append((bars[fill][0], R, A))
        last_exit = exitbar
    return trades


def shifted(atr):
    """ATR 的前一根：不含事件那根自己的真實區間。"""
    return [None] + list(atr[:-1])


def var_share_top_day(df, col):
    d = df.dropna(subset=[col])
    tot = float(((d[col] - d[col].mean()) ** 2).sum())
    by = ((d[col] - d[col].mean()) ** 2).groupby(d["day"]).sum().sort_values(ascending=False)
    return float(by.iloc[0] / tot), str(by.index[0]), int((d["day"] == by.index[0]).sum())


def day_cluster_se(df, col, b=B):
    d = df.dropna(subset=[col])
    days = d["day"].to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uniq))]
    x = d[col].to_numpy(float)
    reps = np.empty(b)
    for i in range(b):
        pick = RNG.integers(0, len(uniq), len(uniq))
        reps[i] = x[np.concatenate([idx[k] for k in pick])].mean()
    return float(np.std(reps, ddof=1))


def main():
    rows = []
    ratios = []
    for s in CORE9:
        bars = sc.load_csv(str(CACHE / f"{s}USDT_1h.csv"))
        a = sc.atr14(bars)
        a_prev = shifted(a)
        t_now = {t: (R, A) for t, R, A in backtest_with_atr(bars, a)}
        t_prev = {t: (R, A) for t, R, A in backtest_with_atr(bars, a_prev)}
        for t in sorted(set(t_now) & set(t_prev)):
            Rn, An = t_now[t]
            Rp, Ap = t_prev[t]
            rows.append(dict(sym=s, ts=t,
                             day=pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%d"),
                             R_frozen=Rn, R_prevatr=Rp,
                             atr_j=An, atr_jm1=Ap))
            if Ap:
                ratios.append(An / Ap)
    df = pd.DataFrame(rows)
    ra = np.array(ratios)

    print(f"共同交易 n={len(df):,}  幣={df.sym.nunique()}  UTC 日={df.day.nunique():,}\n")
    print("事件那根把 ATR 撐大多少（atr[j] / atr[j-1]）:")
    print(f"  中位={np.median(ra):.4f}  q75={np.percentile(ra,75):.4f} "
          f"q90={np.percentile(ra,90):.4f}  q99={np.percentile(ra,99):.4f} "
          f"max={ra.max():.3f}   >1 的比例={np.mean(ra>1)*100:.1f}%\n")

    out = {"n": int(len(df)), "n_days": int(df.day.nunique()),
           "atr_inflation": dict(median=float(np.median(ra)),
                                 q90=float(np.percentile(ra, 90)),
                                 q99=float(np.percentile(ra, 99)),
                                 frac_gt_1=float(np.mean(ra > 1)))}
    print(f"{'尺':14s} {'meanR':>9s} {'std':>8s} {'最大日佔變異':>13s} "
          f"{'那天':>12s} {'筆數':>5s} {'日聚類SE':>9s} {'t':>7s}")
    for col, name in (("R_frozen", "凍結(atr[j])"), ("R_prevatr", "不含事件那根")):
        share, day, k = var_share_top_day(df, col)
        se = day_cluster_se(df, col)
        m = float(df[col].mean())
        out[col] = dict(mean=m, std=float(df[col].std()), top_day_var_share=share,
                        top_day=day, top_day_n=k, day_cluster_se=se,
                        t=m / se if se else float("nan"))
        print(f"{name:14s} {m:9.4f} {df[col].std():8.4f} {share*100:12.2f}% "
              f"{day:>12s} {k:5d} {se:9.4f} {m/se if se else np.nan:7.2f}")

    print("\n前 5 大日各自佔總變異的比例:")
    for col, name in (("R_frozen", "凍結"), ("R_prevatr", "不含事件那根")):
        d = df.dropna(subset=[col])
        by = ((d[col] - d[col].mean()) ** 2).groupby(d["day"]).sum()
        by = (by / by.sum()).sort_values(ascending=False).head(5)
        print(f"  {name:12s} " + "  ".join(f"{i}={v*100:.2f}%" for i, v in by.items()))
        out.setdefault("top5_days", {})[col] = {str(i): float(v) for i, v in by.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r_unit_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    df.to_csv(OUT / "r_unit_audit_trades.csv", index=False)
    print("\nwritten ->", OUT / "r_unit_audit.json")


if __name__ == "__main__":
    main()
