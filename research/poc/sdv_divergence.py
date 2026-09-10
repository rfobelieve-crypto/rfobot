# -*- coding: utf-8 -*-
"""大戶與散戶的分歧，能不能預測掃單後的方向（2026-09-10 預註冊，測一次結案）

使用者的策略前提（原話 2026-09-10）：「我們賺得就是散戶止損的錢」。

**這個前提從來沒有被直接量過。** 而快照表裡剛好有兩組多空比，一組是
帳戶數加權（帳戶多、單小 ≈ 散戶），一組是大戶專屬：

    ls_retail     count_long_short_ratio            全體帳戶多空比
    ls_top_acct   count_toptrader_long_short_ratio  大戶帳戶多空比
    ls_top_pos    sum_toptrader_long_short_ratio    大戶持倉多空比

**兩者的差就是「大戶和散戶站在對立面」的直接度量**，而那正是「賺散戶
止損的錢」的前提。這條線從來沒用過這三個欄位。

===========================================================================
兩個分歧因子，兩種可用性
===========================================================================
    div_pos = ls_top_pos  − ls_retail      大戶持倉 vs 散戶
    div_acct = ls_top_acct − ls_retail     大戶帳戶 vs 散戶

**關鍵優勢：分歧的「水準」是 pre_ 欄位 —— 事件當下就知道。**
所以它不必延後進場，可以直接掛在現行規格上（成立 +3 分進場不變）。
分歧的「變化」是 post_ 欄位，那個要延後，一併測但分開報。

===========================================================================
方向：一樣不由我指定
===========================================================================
使用者的假設方向是「大戶比散戶更多頭 ＋ 向上掃 -> 延續」。但今天已經
證明過一次他的直覺方向可能是反的（§1.03n，OI 那題），所以**方向由前半
資料學**，同時把「使用者假設的方向」當對照臂一起報。兩者不同的話，
那個差異本身就是結論。

因子一律**逐幣、只用前半的分布**轉百分位（跨幣尺度不可比的教訓：平均
單筆大小上一輪翻號，很可能就是全體中位數切出來的是幣種而非強度）。

===========================================================================
判準（跑之前凍結，四條全過才算有用）
===========================================================================
    Q1  樣本外每筆淨值 > 同母體的基準（全部順勢）
    Q2  標準誤（這個平均值本身有多不準）< 基準的
    Q3  樣本外日聚類 CI 下緣 > 0
    Q4  逐幣 >= 6/9

母體分開報：**所有掃單**（9,262）與 **SDV 子集**（1,584）。
§1.03n 已知倉位因子在 SDV 內部零分辨力（旗標已吸收），所以 SDV 那欄
若也是零分辨力，是**預期中**的一致，不是新的失敗。

**測一次結案。** 不過就寫判決，不再換統計量重測同一題。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNAP = HERE / "data" / "sweep_snapshot.parquet"
OUT = HERE / "data" / "results"
SEED = 20260910
WINS = (5, 10, 15)


def boot(days, vals, n=2000):
    rng = np.random.default_rng(SEED)
    by = {}
    for d, v in zip(days, vals):
        by.setdefault(int(d), []).append(v)
    ks = list(by)
    if len(ks) < 5:
        return float(np.mean(vals)), np.nan, np.nan
    arr = [np.array(by[x]) for x in ks]
    idx = rng.integers(0, len(ks), size=(n, len(ks)))
    o = np.array([np.concatenate([arr[j] for j in idx[i]]).mean() for i in range(n)])
    return float(np.mean(vals)), float(o.std(ddof=1)), float(np.percentile(o, 2.5))


def pctile(d, col, mid):
    """逐幣、只用前半的分布轉百分位。"""
    out = np.full(len(d), np.nan)
    for s, g in d.groupby("sym"):
        ref = np.sort(g[g.day < mid][col].dropna().to_numpy(float))
        if len(ref) < 20:
            continue
        out[g.index.to_numpy()] = np.searchsorted(
            ref, g[col].to_numpy(float), side="right") / len(ref)
    return out


def evaluate(d, col, ycol, label, mid, res):
    first, second = d[d.day < mid], d[d.day >= mid]
    if len(second) < 100:
        return
    h = first[first[col] > 0.5][ycol].mean()
    l = first[first[col] <= 0.5][ycol].mean()
    sign = 1 if h >= l else -1

    base_m, base_se, base_lo = boot(second.day.values, second[ycol].to_numpy())
    per_b = second.groupby("sym")[ycol].mean()

    dirs = np.where(second[col].to_numpy() > 0.5, sign, -sign)
    yv = np.where(dirs > 0, second[ycol].to_numpy(),
                  -second[ycol].to_numpy() - 2 * 0.0)   # 逆勢用既有欄位
    ag = second[ycol.replace("with", "against")].to_numpy()
    yv = np.where(dirs > 0, second[ycol].to_numpy(), ag)
    m, se, lo = boot(second.day.values, yv)
    per = pd.DataFrame({"s": second.sym.values, "v": yv}).groupby("s").v.mean()

    # 使用者假設的方向（高分歧＝大戶比散戶多頭 -> 順勢），固定 +1
    yu = np.where(second[col].to_numpy() > 0.5, second[ycol].to_numpy(), ag)
    mu, seu, lou = boot(second.day.values, yu)

    q1, q2, q3, q4 = m > base_m, se < base_se, lo > 0, int((per > 0).sum()) >= 6
    print(f"{label:34} {len(second):6,} {m:+8.4f} {se:7.4f} {lo:+8.4f} "
          f"{int((per > 0).sum()):3d}/{len(per):<3} "
          f"{'高→順勢' if sign > 0 else '高→逆勢'}"
          f"{'  (與假設同向)' if sign > 0 else '  ← 與假設反向'}")
    res[label] = dict(n=int(len(second)), m=m, se=se, lo=lo,
                      npos=int((per > 0).sum()), nsym=int(len(per)),
                      sign=int(sign), base_m=base_m, base_se=base_se,
                      user_m=mu, pass_all=bool(q1 and q2 and q3 and q4))
    return dict(base=(base_m, base_se, base_lo, int((per_b > 0).sum()), len(per_b)))


def main():
    d = pd.read_parquet(SNAP)
    d["day"] = d.ts // 86_400_000
    d = d.reset_index(drop=True)
    d["div_pos"] = d.pre_ls_top_pos - d.pre_ls_retail
    d["div_acct"] = d.pre_ls_top_acct - d.pre_ls_retail
    for k in WINS:
        d[f"divchg_pos{k}"] = d[f"post{k}_ls_top_pos"] - d[f"post{k}_ls_retail"]

    res = {}
    for mname, sub in (("所有掃單", d), ("SDV 子集", d[d.is_sdv])):
        sub = sub.reset_index(drop=True)
        mid = float(sub.day.median())
        for c in ("div_pos", "div_acct") + tuple(f"divchg_pos{k}" for k in WINS):
            sub[c + "_p"] = pctile(sub, c, mid)
        second = sub[sub.day >= mid]
        bm, bse, blo = boot(second.day.values, second.y_with_d0.to_numpy())
        perb = second.groupby("sym").y_with_d0.mean()

        print("=" * 92)
        print(f"母體：{mname}   全部 {len(sub):,} 筆（樣本外 {len(second):,}）")
        print(f"{'臂':34} {'n':>6} {'淨/筆':>8} {'SE':>7} {'CI下緣':>8} {'幣+':>6}  前半學到的方向")
        print(f"{'基準（全部順勢，不用因子）':34} {len(second):6,} {bm:+8.4f} {bse:7.4f} "
              f"{blo:+8.4f} {int((perb > 0).sum()):3d}/{len(perb)}")

        # 水準（pre，事件當下已知 -> 不必延後進場）
        for c, nm in (("div_pos_p", "分歧·大戶持倉−散戶（當下已知）"),
                      ("div_acct_p", "分歧·大戶帳戶−散戶（當下已知）")):
            s2 = sub.dropna(subset=[c]).reset_index(drop=True)
            evaluate(s2, c, "y_with_d0", f"{nm}", float(s2.day.median()), res)
        # 變化（post，要延後 K 分鐘進場）
        for k in WINS:
            c = f"divchg_pos{k}_p"
            s2 = sub.dropna(subset=[c]).reset_index(drop=True)
            evaluate(s2, c, f"y_with_d{k}", f"分歧變化 {k} 分（需延後進場）",
                     float(s2.day.median()), res)

        print()
        print("  判準（Q1報酬>基準 ∧ Q2 SE<基準 ∧ Q3下緣>0 ∧ Q4逐幣≥6/9）：")
        for k, v in list(res.items()):
            if k.startswith("_"):
                continue
            q = (v["m"] > v["base_m"], v["se"] < v["base_se"],
                 v["lo"] > 0, v["npos"] >= 6)
            print(f"    {k:34} Q1{'✓' if q[0] else '✗'} Q2{'✓' if q[1] else '✗'} "
                  f"Q3{'✓' if q[2] else '✗'} Q4{'✓' if q[3] else '✗'}"
                  f"  -> {'**有用**' if all(q) else '不過'}")
        res = {f"{mname}·{k}": v for k, v in res.items()}
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_divergence.json"
    p.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
