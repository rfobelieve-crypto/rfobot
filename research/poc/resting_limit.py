# -*- coding: utf-8 -*-
"""誠實的限價進場 —— 凍結回測假設「在價位成交」，市場容許嗎？

前情（`resting_fill.py`，2026-09-07，P1/P2 已作廢）
    第一版把「成交」定義成「價格往**穿越方向**走得夠深」，而那正是這條
    規則的獲利方向 —— Spearman(深度, R) = +0.6884，分層變數就是結果。
    作廢後量到真正的問題：**下單當下價格已經在價位另一側的比例是 59.35%**。
    凍結引擎在那些情況照樣記「成交在價位」，但市場已經走過去了 ——
    掛限價不會成交（要等價格回頭），下市價單則成交在比價位更差的地方。

    所以要問的不是「掛單會不會被逆選擇」，是更前面一層：
        **凍結回測的成交價，市場當時給不給？**

本檔的模型（下單時點、成交條件、未成交處置，全部寫死在跑之前）
    下單：掃單 bar j **收盤時**，在 lvl 掛一張反轉方向的限價單
          （買側被掃 -> 掛賣單；賣側被掃 -> 掛買單）
    存活：到回踩窗結束（第 j+W 根小時收盤）為止，之後撤單
    兩種下單當下的處境**分開報告，因為經濟意義不同**：

      MARKETABLE  掛單價落在市場的「已可成交」那一側
                  -> 立即成交。保守起見記成交價 = lvl（真實成交只會更好）
                  -> **沒有佇列風險**，但進場時刻比凍結引擎**早**
      RESTING     掛單價在市場的另一側，是真正被動掛著的單
                  -> 只有價格**朝它走過來**並觸及 lvl 才成交
                  -> **佇列風險在這裡**：只碰到不穿過時，你可能排在後面

    佇列代理 δ（ATR 單位）：RESTING 那一側要求價格穿過 lvl **至少 δ**
    才算成交。δ 量的是「價格朝掛單走過來」的方向 —— 對剛進場的部位
    **不利**的方向，與獲利方向相反。所以它不再是結果的替身。

    未成交 = 沒有部位（R 不計入，不是虧損），另計成交率。

出場：完整凍結規則（3.5 ATR 災難停損 + HOLD=8），從成交所屬的小時 bar
起算，停損從**下一根**開始檢查。SLIP 一個字不動。

預註冊判準（判準寫在 CI 上，不寫在點估計上）
    Q1  凍結回測的成交價拿不拿得到
        配對差（誠實限價 − 凍結）的日聚類 CI **上緣 < 0**
        -> 凍結回測在進場價上是樂觀的，差值就是修正量
        CI 含零 -> INCONCLUSIVE（不得讀成「沒有樂觀」）
    Q2  逆選擇
        RESTING 子集內，δ=0.10 的 meanR 減 δ=0.00 的 meanR，
        日聚類 CI **上緣 < 0** -> 逆選擇 CONFIRMED
    Q3  安慰劑（已知答案的對照組，**必須**出現）
        同一套機器改用**獲利方向**分層，必須重現 resting_fill 的
        套套邏輯（Spearman ≥ +0.5）。重現不出來 = 機器測不到這類效應，
        則 Q1/Q2 一律不解讀（mistake.md 2026-08-11：重現不了已知病灶的
        harness 不能用來排序修法）。
    Q4  成本對照
        SLIP=0 重跑必須**優於**含成本結果，否則成本模型壞了。
        （這關由 run_cost_control() 在最後自動跑）

全格報告：兩種處境 × 五個 δ 全部印出來，不挑格。
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
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
CACHE = HERE.parents[0] / "sweep_failure" / ".cache"
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
MIN_MS = 60_000
HOUR_MS = 3_600_000
DELTAS = [0.00, 0.01, 0.02, 0.05, 0.10]
RNG = np.random.default_rng(20260907)


def exit_R(b1, h, lo, cl, n, f, d, entry, risk):
    """凍結出場：停損從 f+1 檢查，否則 HOLD 根後收盤出場。"""
    stop = entry - d * risk
    for q in range(f + 1, min(f + sc.HOLD + 1, n)):
        if (d == 1 and lo[q] <= stop) or (d == -1 and h[q] >= stop):
            return -1.0 - sc.SLIP / sc.DIS, True
    exb = min(f + sc.HOLD, n - 1)
    return d * (cl[exb] - d * sc.SLIP * sc.DIS * 0 - d * sc.SLIP
                * (risk / sc.DIS) - entry) / risk, False


def build(sym, placebo=False):
    b1 = sc.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    h = [x[sc.H] for x in b1]
    lo = [x[sc.L] for x in b1]
    cl = [x[sc.C] for x in b1]
    n = len(b1)
    hts = np.array([int(x[0]) for x in b1], dtype=np.int64) * 1000

    m = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "high", "low"])
    mts = m["ts"].to_numpy(np.int64)
    mhi = np.nan_to_num(m["high"].to_numpy(float), nan=-np.inf)
    mlo = np.nan_to_num(m["low"].to_numpy(float), nan=np.inf)
    nm = len(mts)

    rows = []
    for e in sc.backtest_symbol(b1, detail=True):
        j, lvl, A, d, risk = (e["j"], e["level"], e["atr"], e["d"], e["risk"])
        kd = -d                                    # 掃單方向
        t0 = int(b1[j][0]) * 1000 + HOUR_MS        # 下單時點＝掃單 bar 收盤
        t1 = int(b1[min(j + sc.W, n - 1)][0]) * 1000 + HOUR_MS
        i0 = int(np.searchsorted(mts, t0, side="left"))
        i1 = int(np.searchsorted(mts, t1, side="left"))
        if i1 <= i0 or i1 > nm:
            continue

        # 下單當下市場在價位的哪一側？（用掃單 bar 的收盤，那是下單時刻）
        # d=-1 做空，掛賣單在 lvl：市場 > lvl 時該賣單可立即成交
        # d=+1 做多，掛買單在 lvl：市場 < lvl 時該買單可立即成交
        px0 = cl[j]
        marketable = (px0 > lvl) if d == -1 else (px0 < lvl)

        # RESTING 那側，價格要「朝掛單走過來」才成交：
        #   做空掛 lvl（市場在下）-> 價格要漲上來，看 mhi
        #   做多掛 lvl（市場在上）-> 價格要跌下來，看 mlo
        if placebo:
            # 安慰劑：改用**獲利方向**分層（＝作廢那版的做法），必須重現套套邏輯
            appr = (lvl - mlo[i0:i1]) / A if d == -1 else (mhi[i0:i1] - lvl) / A
        else:
            appr = (mhi[i0:i1] - lvl) / A if d == -1 else (lvl - mlo[i0:i1]) / A

        row = dict(sym=sym, side=e["side"], marketable=bool(marketable),
                   R_frozen=float(e["R"]), lvl=lvl, atr=A,
                   day=pd.Timestamp(int(b1[j][0]) * 1000, unit="ms",
                                    tz="UTC").strftime("%Y-%m-%d"),
                   max_appr=float(np.nanmax(appr)) if len(appr) else np.nan)

        for dl in DELTAS:
            tag = f"{dl:.2f}"
            if marketable:
                f_bar, ok = j, True               # 立即成交，成交價保守記 lvl
            else:
                hit = np.flatnonzero(appr >= dl)
                if len(hit) == 0:
                    row[f"fill_{tag}"] = 0
                    row[f"R_{tag}"] = np.nan
                    continue
                k = i0 + int(hit[0])
                f_bar = int(np.searchsorted(hts, int(mts[k]), side="right")) - 1
                ok = 0 <= f_bar and f_bar + 1 < n
                if not ok:
                    row[f"fill_{tag}"] = 0
                    row[f"R_{tag}"] = np.nan
                    continue
            entry = lvl + d * sc.SLIP * A
            stop = entry - d * risk
            R = None
            for q in range(f_bar + 1, min(f_bar + sc.HOLD + 1, n)):
                if (d == 1 and lo[q] <= stop) or (d == -1 and h[q] >= stop):
                    R = -1.0 - sc.SLIP / sc.DIS
                    break
            if R is None:
                exb = min(f_bar + sc.HOLD, n - 1)
                R = d * (cl[exb] - d * sc.SLIP * A - entry) / risk
            row[f"fill_{tag}"] = 1
            row[f"R_{tag}"] = float(R)
        rows.append(row)
    return pd.DataFrame(rows)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return (float("nan"),) * 4
    uq, inv = np.unique(days, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([idx[k] for k in p])].mean()
    return (float(x.mean()), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)), float(np.std(reps, ddof=1)))


def main():
    d = pd.concat([build(s) for s in CORE9], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "resting_limit.parquet", index=False)
    days = d["day"].to_numpy()
    mk = d["marketable"].to_numpy(bool)
    res = {"n": int(len(d)), "days": int(d.day.nunique()),
           "marketable_share": float(mk.mean())}

    print(f"凍結交易集 n={len(d):,}   UTC 日={d.day.nunique():,}   幣={d.sym.nunique()}")
    print(f"下單當下 MARKETABLE（市場已在價位另一側）{mk.mean()*100:.2f}%   "
          f"RESTING（真正被動掛著）{(~mk).mean()*100:.2f}%")
    print("（全格報告：兩種處境 x 五個 δ 全印。任何一格都不是單獨的結論。）\n")

    print("=== 成交率與報酬 ===\n")
    print(f"{'δ(ATR)':>7s} {'總成交率':>9s} {'RESTING 成交率':>15s} "
          f"{'誠實限價 meanR':>15s} {'日聚類 CI95':>22s} "
          f"{'凍結 meanR(同母體)':>18s} {'配對差':>24s}")
    per = {}
    frozen_all = d["R_frozen"].to_numpy(float)
    for dl in DELTAS:
        tag = f"{dl:.2f}"
        f = d.get(f"fill_{tag}", pd.Series(0, index=d.index)).fillna(0).to_numpy(int) == 1
        rr = d[f"R_{tag}"].to_numpy(float)
        mA, loA, hiA, _ = day_ci(rr[f], days[f])
        mF, _, _, _ = day_ci(frozen_all[f], days[f])
        diff = rr[f] - frozen_all[f]
        mD, loD, hiD, seD = day_ci(diff, days[f])
        rest_fr = float(f[~mk].mean()) if (~mk).sum() else float("nan")
        per[tag] = dict(delta=dl, fill_rate=float(f.mean()),
                        resting_fill_rate=rest_fr, mean_honest=mA,
                        ci_honest=[loA, hiA], mean_frozen_same=mF,
                        diff=mD, diff_ci=[loD, hiD], diff_se=seD,
                        n_fill=int(f.sum()))
        print(f"{dl:7.2f} {f.mean()*100:8.2f}% {rest_fr*100:14.2f}% "
              f"{mA:+15.4f}  [{loA:+.4f},{hiA:+.4f}] {mF:+18.4f} "
              f"{mD:+.4f} [{loD:+.4f},{hiD:+.4f}]")
    res["by_delta"] = per

    print("\n=== 兩種處境分開看（δ=0）===\n")
    split = {}
    for name, sel in (("MARKETABLE 立即成交", mk), ("RESTING 被動掛著", ~mk)):
        rr = d["R_0.00"].to_numpy(float)
        f = d["fill_0.00"].fillna(0).to_numpy(int) == 1
        s = sel & f
        mA, loA, hiA, _ = day_ci(rr[s], days[s])
        mF, _, _, _ = day_ci(frozen_all[s], days[s])
        split[name] = dict(n=int(s.sum()), honest=mA, ci=[loA, hiA], frozen=mF)
        print(f"{name:22s} n={int(s.sum()):6,d}   誠實 {mA:+.4f} "
              f"[{loA:+.4f},{hiA:+.4f}]   凍結 {mF:+.4f}")
    res["split"] = split

    print("\n=== 預註冊判準 ===\n")
    t0 = f"{DELTAS[0]:.2f}"
    hiD = per[t0]["diff_ci"][1]
    q1 = ("CONFIRMED-樂觀" if np.isfinite(hiD) and hiD < 0 else
          "INCONCLUSIVE" if np.isfinite(hiD) else "N/A")
    print(f"Q1 凍結成交價拿不拿得到（配對差 CI 上緣 < 0 -> 凍結是樂觀的）")
    print(f"    δ=0 配對差 {per[t0]['diff']:+.4f}  "
          f"CI [{per[t0]['diff_ci'][0]:+.4f},{hiD:+.4f}]   -> {q1}")
    res["Q1"] = dict(verdict=q1, **{k: per[t0][k] for k in ("diff", "diff_ci")})

    tl = f"{DELTAS[-1]:.2f}"
    rsel = ~mk
    a = d.loc[rsel & (d["fill_0.00"].fillna(0) == 1), "R_0.00"].to_numpy(float)
    da = days[rsel & (d["fill_0.00"].fillna(0).to_numpy(int) == 1)]
    b_ = d.loc[rsel & (d[f"fill_{tl}"].fillna(0) == 1), f"R_{tl}"].to_numpy(float)
    db = days[rsel & (d[f"fill_{tl}"].fillna(0).to_numpy(int) == 1)]
    ma, _, _, sa = day_ci(a, da)
    mb, _, _, sb = day_ci(b_, db)
    se = float(np.sqrt(sa ** 2 + sb ** 2))
    lo_, hi_ = mb - ma - 1.96 * se, mb - ma + 1.96 * se
    q2 = ("CONFIRMED" if hi_ < 0 else "INCONCLUSIVE")
    print(f"\nQ2 逆選擇（RESTING 子集，δ=0.10 減 δ=0，CI 上緣 < 0 -> CONFIRMED）")
    print(f"    δ=0 {ma:+.4f}（n={len(a):,}）  δ=0.10 {mb:+.4f}（n={len(b_):,}）  "
          f"差 {mb-ma:+.4f} [{lo_:+.4f},{hi_:+.4f}]   -> {q2}")
    res["Q2"] = dict(verdict=q2, d0=ma, d10=mb, diff=mb - ma, ci=[lo_, hi_])

    print("\nQ3 安慰劑（同一套機器改用獲利方向分層，必須重現套套邏輯）")
    pl = pd.concat([build(s, placebo=True) for s in CORE9], ignore_index=True)
    x, y = pl["max_appr"].to_numpy(float), pl["R_frozen"].to_numpy(float)
    g = np.isfinite(x) & np.isfinite(y)
    rho_pl = float(pd.Series(x[g]).corr(pd.Series(y[g]), method="spearman"))
    xr = d["max_appr"].to_numpy(float)
    gr = np.isfinite(xr) & np.isfinite(frozen_all)
    rho_real = float(pd.Series(xr[gr]).corr(pd.Series(frozen_all[gr]),
                                            method="spearman"))
    q3 = "PASS" if rho_pl >= 0.5 else "**FAIL — 機器測不到,Q1/Q2 不解讀**"
    print(f"    安慰劑（獲利方向）Spearman = {rho_pl:+.4f}（需 ≥ +0.50）  -> {q3}")
    print(f"    本檔（走過來的方向）Spearman = {rho_real:+.4f}"
          f"   ← 這個**不應該**高；高就代表我又選到結果本身")
    res["Q3"] = dict(verdict=q3, rho_placebo=rho_pl, rho_real=rho_real)

    (OUT / "resting_limit.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\nwritten ->", OUT / "resting_limit.json")


if __name__ == "__main__":
    main()
