# -*- coding: utf-8 -*-
"""交會事件的方向該由誰決定：價格動能，還是主動量本身

===========================================================================
為什麼問這個
===========================================================================
使用者（2026-09-08）：「兩個欄位就夠了嗎，不覺得定義太簡單了嗎」。

盤過之後，「加資料」這個方向被三次判決擋住（WQ101 75 個 alpha、86 個新特徵
6 個家族、21 個流動性代理，全部 ensemble A/B 不過；CLAUDE.md 因此寫死
「預設不再跑同源特徵 A/B」）。但同一次盤點抓到一件**不是加資料、是把
現有資料用滿**的事：

    ad = back_sum(np.abs(dl))     # <- 取絕對值，符號被扔掉

`delta_ext` 只問「單邊有多極端」，不問**往哪一邊**。而交易方向是另外用
`imp = sign(close(t) - close(t-5m))`（5 分鐘價格動能）決定的——一個更粗的
代理。**手上就有真正的方向資訊，卻用價格去猜它。**

機制假設（寫在看數字之前）：
    大量單邊主動買、但價格沒漲 = **有人在吃**（賣方有承接能力）
    -> 那不是強制流打穿薄簿的形狀，是強制流撞到牆的形狀
    -> 兩者不一致時不該進場

===========================================================================
三個臂（事前指定，不是搜出來的）
===========================================================================
    P  價格動能定方向   imp = sign(close(a) - close(a-5m))      <- 現行，對照組
    F  主動量定方向     imf = sign(Σ delta over [a-4, a])
    A  兩者一致才進場   方向 = 共同方向；不一致 -> 不交易

    另報 X = 不一致那一格（用 P 的方向），**全格報告不挑格**。

其餘一律照現行規格：進場 = 錨點 +2 分開盤、停損 1.0 ATR、持有 60 分、
成本 10 bps（= bps/1e4/ATR%，ATR 單位）。母體 = 現行註冊的
`掃單 ∧ (delta_ext ∨ vol_burst)`（NO-OI）。

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
N1  **已知答案對照，必須先過。** P 臂必須重現現行規格的 +0.2275
    （容差 ±0.02）。對不上代表機器寫錯了，以下不解讀。
N2  F 臂 vs P 臂：若 F 明顯較好，代表現行用價格猜方向是浪費。
    「明顯」＝差值的日聚類 bootstrap CI 不含零。
N3  A 臂（一致才進）：扣成本後日聚類 CI 下緣 > 0 **且**逐幣 ≥ 6/9。
N4  全格報告 P / F / A / X 四格與兩半穩定性，不挑格。
N5  **這是診斷，不是換註冊。** 現行時鐘註冊在 P 臂上；看過本表再改方向
    定義，就是 §0.92 判掉 C/D 的同一件事。要改必須另開時鐘從零起算。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402
import conj_clock as ck  # noqa: E402
import conj_causal as cc  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W = 5
HOLD = 60
DELAY = 2
STOP = 1.0
BPS = 10
REF_P = 0.2275
TOL = 0.02
RNG = np.random.default_rng(20260908)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return (float("nan"),) * 2
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == m)[0] for m in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[m] for m in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5))


def main():
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows, apc = [], {}
    for sym in ec.CORE9:
        cand, ts, cl, at, day = ck.frozen_cand(sym, liq)
        _c, _t, _cl, _at, _d, q = ec.detect_all(sym, liq)
        caus = cc.causal_flags(q, _d)
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["open", "high", "low", "delta"])
        op = b["open"].to_numpy(float)
        hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
        n = len(ts)
        apc[sym] = float(np.nanmedian(at / np.where(cl > 0, cl, np.nan)))

        pairs = [(int(m), "sweep")
                 for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in ("delta_ext", "vol_burst"):
            v = caus.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pairs.append((int(m), nm))

        for a, sig in et.cluster(pairs):
            if "sweep" not in sig or not (sig & {"delta_ext", "vol_burst"}):
                continue
            if a < W or a + DELAY + HOLD >= n:
                continue
            A = float(at[a])
            if not np.isfinite(A) or A <= 0:
                continue
            imp_p = float(np.sign(cl[a] - cl[a - W]) or 1.0)
            imp_f = float(np.sign(dl[a - W + 1:a + 1].sum()) or 1.0)
            j0 = a + DELAY
            ent = float(op[j0])
            end = j0 + HOLD

            def score(d):
                adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
                       else (hi[j0 + 1:end + 1] - ent)) / A
                if (adv >= STOP).any():
                    return -STOP
                return float(d * (cl[end] - ent) / A)

            rows.append(dict(
                sym=sym,
                day=pd.Timestamp(int(ts[a]), unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d"),
                both=bool({"delta_ext", "vol_burst"} <= sig),
                agree=bool(imp_p == imp_f),
                P=score(imp_p), F=score(imp_f)))

    d = pd.DataFrame(rows)
    w = d.groupby("sym").size()
    apw = float(sum(w[s] * apc[s] for s in w.index) / w.sum())
    cost = BPS / 1e4 / apw
    days = d.day.to_numpy()
    half = len(d) // 2

    print("=== 交會事件的方向：價格動能 vs 主動量本身 ===")
    print(f"母體 {len(d):,} 筆（現行註冊定義）、{d.day.nunique()} 日、9 幣   "
          f"ATR% {apw*100:.3f}%   成本 {cost:.4f} ATR")
    print(f"兩者方向一致的比例 {d.agree.mean()*100:.1f}%")
    print()

    mP, lP = day_ci(d.P.to_numpy(), days)
    ok1 = abs(mP - REF_P) < TOL
    print("=== N1 已知答案對照 ===")
    print(f"  P 臂 {mP:+.4f}（規格 {REF_P:+.4f}，容差 {TOL}）-> "
          + ("PASS" if ok1 else "**FAIL —— 機器寫錯了，以下不解讀**"))
    print()

    print("=== N4 全格報告（不挑格）===")
    print(f"{'臂':>16s} {'n':>7s} {'毛利':>9s} {'CI下緣':>9s} {'淨':>9s} "
          f"{'幣+':>5s} {'前半':>8s} {'後半':>8s}")
    res = {"n": int(len(d)), "cost": cost, "agree_rate": float(d.agree.mean()),
           "N1": bool(ok1), "arms": {}}

    def report(lab, sub, col):
        m, l = day_ci(sub[col].to_numpy(), sub.day.to_numpy())
        per = sub.groupby("sym")[col].mean() - cost
        npos = int((per > 0).sum())
        v = sub[col].to_numpy()
        h = len(v) // 2
        h1, h2 = float(np.nanmean(v[:h])), float(np.nanmean(v[h:]))
        print(f"{lab:>16s} {len(sub):7,d} {m:+9.4f} {l:+9.4f} {m-cost:+9.4f} "
              f"{npos:4d}/9 {h1:+8.4f} {h2:+8.4f}")
        res["arms"][lab] = dict(n=int(len(sub)), mean=m, ci_lo=l,
                                net=m - cost, net_lo=l - cost,
                                coins_pos=npos, half1=h1, half2=h2)
        return m, l, npos, h1, h2

    report("P 價格定方向", d, "P")
    report("F 主動量定方向", d, "F")
    mA, lA, nA, a1, a2 = report("A 一致才進", d[d.agree], "P")
    report("X 不一致（P向）", d[~d.agree], "P")
    report("X 不一致（F向）", d[~d.agree], "F")

    # N2：F - P 的差，逐事件配對，日聚類
    diff = (d.F - d.P).to_numpy()
    mD, lD = day_ci(diff, days)
    _, uD = day_ci(-diff, days)
    print()
    print(f"N2 F − P 的配對差 {mD:+.4f}   日聚類 CI 下緣 {lD:+.4f}")
    ok2 = ok1 and lD > 0
    print("   -> " + ("**F 明顯較好——用價格猜方向是浪費**" if ok2 else
                      "F 沒有明顯較好（CI 含零或為負）"))

    ok3 = ok1 and (lA - cost) > 0 and nA >= 6
    print(f"N3 A 臂扣成本 CI 下緣 {lA-cost:+.4f} > 0 且逐幣 {nA}/9 >= 6 ? -> "
          + ("**PASS**" if ok3 else "FAIL"))
    print(f"   兩半 {a1:+.4f} / {a2:+.4f} "
          + ("同號" if np.sign(a1) == np.sign(a2) else "**異號**"))
    print()
    print("N5 **這是診斷不是換註冊**：現行時鐘註冊在 P 臂上。看過本表再改"
          "方向定義，就是 §0.92 判掉 C/D 的同一件事——要改必須另開時鐘從零。")

    res.update(N2=bool(ok2), N3=bool(ok3), diff_FP=mD, diff_lo=lD)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flow_direction.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "flow_direction.json")


if __name__ == "__main__":
    main()
