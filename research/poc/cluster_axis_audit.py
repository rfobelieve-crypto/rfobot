# -*- coding: utf-8 -*-
"""日聚類夠不夠？—— 這條線的聚集同時發生在**價格軸**上

`engine_audit.py`（2026-09-07）量到：**25.5%** 的掃單事件，在 ±24 小時內有
另一個層級落在 0.25 ATR 之內。也就是說同一個價位常常被掃兩次以上。

這條線所有的 CI 都是**日聚類** bootstrap。日聚類處理的是「同一天的多筆交易
共享當天的衝擊」。但兩筆「同一個價位、相隔三天」的交易呢？日聚類把它們當成
兩個獨立的群，而它們其實是同一個結構被打兩次。

    若價格軸的聚類是實質的，**這條線每一個 CI 都偏窄**，
    包含 Gate F 的判準（n>=1400 ∧ 日聚類 CI 下緣>0）。
    這不是一個角落的問題，是地基。

前例：`sweep_forward` 當初就是因為量到 VIF=2.95（九個相關的幣同時吃到同一個
衝擊）才把 iid CI 換成日聚類。那次處理的是**跨幣**，這次是**跨時間、同價位**。

作法
    定義三種群：
      D  日：UTC 日（現行）
      L  價位群：同幣、層級價位在 0.25 ATR 之內、且掃單時間相隔 <= 7 天
                （用單向掃描把鏈接起來的事件併成一群）
      B  兩者：D 與 L 的聯集群（把任一種相連的都併在一起，最保守）
    對同一批交易（凍結引擎的全部成交）各跑一次 bootstrap，比 SE。

判準（跑之前寫死）
    C1 價格軸聚類是不是實質的
        SE(L) / SE(D) >= 1.20  ->  **是**，現行 CI 偏窄，這條線的每一個
        CI 都要用更保守的群重算，並在判決檔標注
        < 1.20  ->  日聚類已經夠，價格軸的冗餘不構成實質相依
    C2 最保守的版本要一起報
        SE(B) / SE(D) 一律印出來，不論 C1 結果。
    C3 已知答案的對照
        把層級價位隨機重排（**保持每個群的大小分布不變**）再算 SE(L)。
        對照的 SE 比值必須接近 1.0；若隨機分群也讓 SE 膨脹，代表膨脹來自
        「分群本身」而不是價格軸的相依，C1 不解讀。

**本檔不改任何判準，只量 CI 該有多寬。** 若 C1 成立，後續要做的是把既有
判決檔的 CI 重算並標注，不是放寬門檻。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SF = HERE.parents[0] / "sweep_failure"
sys.path.insert(0, str(SF))
import sweep_core as sc  # noqa: E402

CACHE = SF / ".cache"
OUT = HERE / "data" / "results"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
NEAR_ATR = 0.25
LINK_DAYS = 7
B = 2000
RNG = np.random.default_rng(20260907)


def collect():
    rows = []
    for sym in CORE9:
        bars = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
        for t in sc.backtest_symbol(bars, detail=True):
            rows.append(dict(sym=sym, R=float(t["R"]), level=float(t["level"]),
                             atr=float(t["atr"]),
                             ts=int(t["fill_ts"]) * 1000))
    d = pd.DataFrame(rows).sort_values(["sym", "ts"]).reset_index(drop=True)
    d["day"] = (d.ts // 86_400_000).astype(np.int64)
    return d


def level_groups(d, shuffle=False):
    """同幣、層級在 0.25 ATR 之內、時間相隔 <= LINK_DAYS 天 -> 併成一群。"""
    gid = np.full(len(d), -1, dtype=np.int64)
    nxt = 0
    for sym, idx in d.groupby("sym").groups.items():
        idx = np.asarray(idx)
        lv = d.level.to_numpy()[idx].copy()
        at = d.atr.to_numpy()[idx]
        ts = d.ts.to_numpy()[idx]
        if shuffle:
            # C3 對照：層級價位隨機重排（同幣內），時間結構不動
            lv = RNG.permutation(lv)
        # 單向掃描：按時間排序，與**還在時窗內**的既有群比價位
        order = np.argsort(ts)
        heads = []          # [(gid, level, atr, last_ts)]
        for k in order:
            g = -1
            for hi in range(len(heads) - 1, -1, -1):
                gg, hl, ha, hts = heads[hi]
                if ts[k] - hts > LINK_DAYS * 86_400_000:
                    continue
                if abs(lv[k] - hl) / max(ha, 1e-12) <= NEAR_ATR:
                    g = gg
                    heads[hi] = (gg, hl, ha, ts[k])
                    break
            if g < 0:
                g = nxt
                nxt += 1
                heads.append((g, lv[k], at[k], ts[k]))
            gid[idx[k]] = g
    return gid


def boot_se(x, groups):
    uq, inv = np.unique(groups, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(B)
    for i in range(B):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(np.std(reps, ddof=1)), len(uq)


def union_groups(a, b):
    """把 a、b 任一種相連的併在一起（union-find）。"""
    parent = {}

    def find(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def uni(u, v):
        ru, rv = find(("a", u)), find(("b", v))
        if ru != rv:
            parent[ru] = rv

    for u, v in zip(a, b):
        uni(u, v)
    return np.array([hash(find(("a", u))) for u in a])


def main():
    d = collect()
    x = d.R.to_numpy(float)
    print(f"凍結引擎成交 {len(d):,} 筆、{d.day.nunique():,} 個 UTC 日")
    print(f"meanR {x.mean():+.4f}")
    print()

    se_d, n_d = boot_se(x, d.day.to_numpy())
    gl = level_groups(d)
    se_l, n_l = boot_se(x, gl)
    gb = union_groups(d.day.to_numpy(), gl)
    se_b, n_b = boot_se(x, gb)

    print(f"{'群':22s} {'群數':>8s} {'SE':>9s} {'CI95 半寬':>10s} {'相對日聚類':>10s}")
    for name, se, nn in (("D 日（現行）", se_d, n_d),
                         ("L 價位群", se_l, n_l),
                         ("B 兩者聯集（最保守）", se_b, n_b)):
        print(f"{name:22s} {nn:8,d} {se:9.5f} {1.96*se:10.5f} "
              f"{se/se_d:10.2f}x")
    print()

    r1 = se_l / se_d
    v1 = ("**價格軸聚類是實質的 — 現行 CI 偏窄，全線 CI 要重算**"
          if r1 >= 1.20 else "日聚類已經夠")
    print(f"C1 SE(L)/SE(D) = {r1:.2f}x  -> {v1}")
    print(f"C2 SE(B)/SE(D) = {se_b/se_d:.2f}x（最保守版本，一律並報）")

    gs = level_groups(d, shuffle=True)
    se_s, n_s = boot_se(x, gs)
    r3 = se_s / se_d
    v3 = ("PASS（隨機分群不膨脹）" if r3 < 1.10
          else "**FAIL — 分群本身就會膨脹 SE，C1 不解讀**")
    print(f"C3 對照（層級隨機重排）SE 比 {r3:.2f}x，群數 {n_s:,}  -> {v3}")

    res = dict(n=int(len(d)), mean=float(x.mean()),
               se_day=se_d, se_level=se_l, se_both=se_b, se_shuffle=se_s,
               n_day=n_d, n_level=n_l, n_both=n_b, n_shuffle=n_s,
               ratio_level=r1, ratio_both=se_b / se_d, ratio_shuffle=r3,
               C1=v1, C3=v3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cluster_axis_audit.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "cluster_axis_audit.json")
    print()
    print("**若 C1 成立，後續是把既有判決檔的 CI 重算並標注，不是放寬門檻。**")


if __name__ == "__main__":
    main()
