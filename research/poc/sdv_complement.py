# -*- coding: utf-8 -*-
"""SDV 的兩個追問 —— 換 CVD 有沒有用；沒有流量的掃單會不會反轉

使用者 2026-09-09（原話）：
  「現在只有 delta 跟成交量，如果 delta 換成 cvd 會不會更好，
    所以我要做的是抓到大的延續行情，然後如果延續的訊號沒出來
    不就代表會反轉嗎」

===========================================================================
先講一件不用測就知道的事（寫在這裡免得下次又問）
===========================================================================
**五分鐘的 CVD 變化，跟現在的 delta 五分鐘和，是同一個東西。**
CVD = cumsum(delta)，所以 CVD[t] − CVD[t−5] ≡ sum(delta[t−4..t])。
「把 delta 換成 CVD」如果窗口不變，換的只是名字。

真正不同的有三種，本檔測其中兩種（第三種需要簿口資料，沒有）：
    C1  **窗口拉長**：15 / 30 / 60 分鐘的 CVD 變化當觸發，取代 5 分鐘。
        先驗有利——[[project_orderflow_edge_verdict]] 量到「瞬時流在 4h
        零邊際價值、edge 在積分流」。但那是 V7 的方向問題，不是這裡的
        觸發問題，先驗不因為換了問法就自動成立。
    C2  **帶符號**：現在的 D 是 |delta|，**對方向是瞎的**。改用有號的
        CVD 變化當**方向來源**（取代五分鐘動能），這是這條線最大的洞
        ——樣本外勝率 47.9%、四個判別臂全滅。
    C3  CVD 背離（價格新高但 CVD 沒有）—— 需要更長的參考窗與樞紐配對，
        **本檔不做**，明寫沒查。

===========================================================================
第二題才是重點：「延續訊號沒出來 = 會反轉」嗎
===========================================================================
邏輯上不成立——**沒有 A 不等於 B，因為還有第三種：什麼都沒發生。**
但這個直覺可以直接測，而且我們手上剛好有那個母體：

    SDV        掃單 ∧ delta_ext ∧ vol_burst      -> 已知：順勢，樣本外 +0.1833
    互補母體    掃單 ∧ 沒有任何流量旗標            -> **從來沒測過**

**注意這不是舊線那筆死掉的交易。** 舊線（§1.02）死在成交假設——它掛限價
在被掃的價位上，而市場當時距離那個價位中位 42.6 bps。本檔的互補母體用
**SDV 同一套可執行進場**（ready+3 的開盤、市價），所以它繼承的是 SDV 的
執行假設，不是舊線的。舊線的訊號有效性本來就沒有被推翻。

===========================================================================
判準（跑之前寫死，事後不放寬 —— 核心原則 #8）
===========================================================================
    Q1  **三種結局全格報告，不挑**：互補母體同時用「順勢」與「逆勢」兩個
        方向各跑一次，並列印。**這一關沒有門檻，它的作用是擋掉
        「沒有 A 就是 B」這個推論**——如果兩邊都貼零，答案就是「第三種」。
    Q2  逆勢要成立：日聚類 bootstrap CI 下緣 > 0 ∧ 逐幣 ≥6/9。
        （與 SDV 判準同一把尺，不另設寬鬆版）
    Q3  **符號必須真的翻**：逆勢的淨值 > 順勢的淨值，且兩者差的 CI 下緣 > 0。
        只有「逆勢為正」不夠——那可能只是整個母體都在漲。
    Q4  **真樣本外**：前半／後半分開報，判決看後半。沒有參數要挑，所以
        不需要選擇窗，但仍然要看它在兩半是不是同號。
    Q5  C1／C2 的每一格全格報告，**不挑最好的那格**；C2 若要取代現行方向，
        判準同 Q2 且必須在後半也成立。

    過關只代表值得**開一條自己的前瞻紀錄**，不得直接改現行規格（§0.92）。
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
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import event_census as ec  # noqa: E402
import conj_redef as cr  # noqa: E402
import conj_backtest as cb  # noqa: E402
import conj_clock as ck  # noqa: E402

OUT = HERE / "data" / "results"
RNG = np.random.default_rng(20260909)
CVD_WINDOWS = (5, 15, 30, 60)     # C1：全格報告，不挑


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 40:
        return float("nan"), float("nan"), float("nan")
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5)), float((r > 0).mean())


def trade(op, hi, lo, cl, at, n, rd, d):
    """與 conj_backtest 完全同一套出場：3 ATR 停損、480 分持有、同一組成本。"""
    j0 = rd + cb.DELAY
    if rd < cb.W or j0 + cb.HOLD >= n:
        return None
    A = at[rd]
    if not (A > 0) or not np.isfinite(A):
        return None
    ent = op[j0]
    sp = ent - d * cb.STOP * A
    for k in range(j0 + 1, j0 + cb.HOLD + 1):
        if (lo[k] <= sp) if d > 0 else (hi[k] >= sp):
            cost = (cb.COST_ENTRY + cb.COST_STOP) / 1e4 * ent / A
            return -cb.STOP - cost
    cost = (cb.COST_ENTRY + cb.COST_TIME) / 1e4 * ent / A
    return float(d * (cl[j0 + cb.HOLD] - ent) / A) - cost


def build(sym):
    """回傳三批：SDV、互補（掃單但零流量）、以及每筆的 CVD 特徵。"""
    liq = cb._empty_liq()
    cand, ts, cl, at, _ = ck.frozen_cand(sym, liq)
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["open", "high", "low", "delta"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
    n = len(ts)
    cvd = np.concatenate([[0.0], np.cumsum(dl)])      # cvd[i+1] = sum(dl[:i+1])

    sw = [int(m) for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    pairs = [(m, "sweep") for m in sw]
    flow_min = set()
    for nm in cb.FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))
                flow_min.add(int(m))

    rows = []
    flowm = set(cb.FLOW)
    for _a, mem in cr.groups_with_members(pairs):
        s = {x for _, x in mem}
        if "sweep" not in s:
            continue
        sw0 = min(m for m, x in mem if x == "sweep")
        if s & flowm:
            rd = max(sw0, min(m for m, x in mem if x in flowm))
            pop = "SDV" if {"delta_ext", "vol_burst"} <= s else "部分流量"
        else:
            rd = sw0                      # 沒有流量,成立時刻就是掃單那分鐘
            pop = "互補（零流量）"
        if rd < max(CVD_WINDOWS) or rd + cb.DELAY + cb.HOLD >= n:
            continue
        A = at[rd]
        if not (A > 0) or not np.isfinite(A):
            continue
        mom = 1.0 if cl[rd] > cl[rd - cb.W] else -1.0
        r_with = trade(op, hi, lo, cl, at, n, rd, mom)
        r_against = trade(op, hi, lo, cl, at, n, rd, -mom)
        if r_with is None or r_against is None:
            continue
        rec = dict(sym=sym, ts=int(ts[rd]), pop=pop,
                   day=pd.Timestamp(int(ts[rd]), unit="ms", tz="UTC")
                        .strftime("%Y-%m-%d"),
                   r_with=r_with, r_against=r_against, mom=mom)
        # C1/C2：各窗口的**有號** CVD 變化，換算成該幣的量單位（除以 ATR 無意義,
        # 用同窗口的 |delta| 標準化,讓幣別可比）
        for w in CVD_WINDOWS:
            ch = cvd[rd + 1] - cvd[rd + 1 - w]
            base = np.abs(dl[max(0, rd + 1 - w):rd + 1]).sum()
            rec[f"cvd{w}"] = float(ch / base) if base > 0 else 0.0
        rows.append(rec)
    return rows


def report(lab, d, col, need_ci=True):
    m, lo_, p = day_ci(d[col].to_numpy(), d.day.to_numpy())
    per = d.groupby("sym")[col].mean()
    ok = (lo_ > 0) and int((per > 0).sum()) >= 6
    print(f"  {lab:<22s} n={len(d):5d}  淨 {m:+.4f}  CI下 {lo_:+.4f}  "
          f"P(>0) {p*100:4.1f}%  逐幣 {int((per>0).sum())}/9"
          + ("  **過閘**" if (ok and need_ci) else ""))
    return dict(n=len(d), mean=m, ci_lo=lo_, p_pos=p,
                coins=int((per > 0).sum()), pass_=bool(ok))


def main():
    rows = []
    for sym in ec.CORE9:
        rows.extend(build(sym))
    d = pd.DataFrame(rows)
    mid = d.ts.min() + (d.ts.max() - d.ts.min()) // 2
    res = {}

    print(f"\n母體切分（掃單經冷卻後全部 {len(d):,} 筆）")
    for p, g in d.groupby("pop"):
        print(f"  {p:<16s} {len(g):6d}  ({len(g)/len(d)*100:4.1f}%)")

    comp = d[d.pop == "互補（零流量）"]
    sdv = d[d.pop == "SDV"]

    print("\n=== Q1 三種結局全格報告：互補母體（掃單但零流量）===")
    print("  「沒有延續訊號 = 會反轉」這個推論，要成立必須逆勢那行明顯為正、")
    print("  順勢那行明顯為負。兩行都貼零 -> 答案是第三種：什麼都沒發生。")
    res["comp_with"] = report("互補·順勢", comp, "r_with")
    res["comp_against"] = report("互補·逆勢", comp, "r_against")
    print("  對照：")
    res["sdv_with"] = report("SDV·順勢（現行）", sdv, "r_with")
    res["sdv_against"] = report("SDV·逆勢", sdv, "r_against")

    print("\n=== Q3 符號有沒有真的翻（逆勢 − 順勢，同一批交易配對）===")
    for lab, g in (("互補", comp), ("SDV", sdv)):
        diff = g.r_against.to_numpy() - g.r_with.to_numpy()
        m, lo_, p = day_ci(diff, g.day.to_numpy())
        print(f"  {lab:<6s} 逆勢−順勢 {m:+.4f}  CI下 {lo_:+.4f}  P(>0) {p*100:4.1f}%"
              + ("  **翻了**" if lo_ > 0 else ""))
        res[f"flip_{lab}"] = dict(mean=m, ci_lo=lo_, p_pos=p)

    print("\n=== Q4 真樣本外（前半／後半分開，判決看後半）===")
    for half, g0 in (("前半", d[d.ts <= mid]), ("後半", d[d.ts > mid])):
        c = g0[g0.pop == "互補（零流量）"]
        print(f"  [{half}]")
        if len(c) >= 40:
            res[f"comp_against_{half}"] = report("  互補·逆勢", c, "r_against")
            res[f"comp_with_{half}"] = report("  互補·順勢", c, "r_with")

    print("\n=== C1/C2 CVD：窗口 × 用法（全格，不挑）===")
    print("  註：5 分鐘 CVD 變化 ≡ 現行 delta 五分鐘和,列出只為對照。")
    print("  C2 用法 = 拿**有號** CVD 變化當方向,取代五分鐘動能。")
    for w in CVD_WINDOWS:
        col = f"cvd{w}"
        s2 = sdv.copy()
        # 方向改由 CVD 符號決定：與動能同號就用 r_with,反號就用 r_against
        agree = np.sign(s2[col].to_numpy()) == s2.mom.to_numpy()
        s2["r_cvd"] = np.where(agree, s2.r_with, s2.r_against)
        m, lo_, p = day_ci(s2.r_cvd.to_numpy(), s2.day.to_numpy())
        per = s2.groupby("sym").r_cvd.mean()
        agree_pct = float(agree.mean()) * 100
        print(f"  C2 w={w:>2d}m  淨 {m:+.4f}  CI下 {lo_:+.4f}  "
              f"P(>0) {p*100:4.1f}%  逐幣 {int((per>0).sum())}/9  "
              f"（與動能同向 {agree_pct:.0f}%）")
        res[f"C2_w{w}"] = dict(mean=m, ci_lo=lo_, p_pos=p,
                               coins=int((per > 0).sum()), agree_pct=agree_pct)
    print("  C3（CVD 背離）**沒查** —— 需要更長參考窗與樞紐配對,本檔不做。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sdv_complement.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwritten -> {OUT / 'sdv_complement.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
