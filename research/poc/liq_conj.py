# -*- coding: utf-8 -*-
"""H5 —— 樞紐 × 清算密度：密集流動性會不會像強迫流一樣讓行情延續

使用者 2026-09-09：「我之前給的 OI 推導清算位指標，**在 swing 高低點如果有
密集流動性就會像強迫流一樣延續行情**，這個有驗證過嗎」。

前半（清算位模型本身是否有效）由 `liq_density2.py` 的 H1~H3 回答。
**後半——也就是使用者真正問的那一句——從來沒有測過**，本檔第一次測。

`liq_density.py` 測的是「掃過的密度能不能預測真實 OI 銷毀」（構造驗證），
**沒有條件在樞紐位置上**，也沒有接到報酬。本檔把兩者接起來。

===========================================================================
設計
===========================================================================
母體    交會事件 S+D+V（與現行規格同一批），每筆帶「被掃的價位」
密度    用 `liq_density2.build(probes=...)` 在**事件成立時刻**查詢
        被掃價位 ±0.1% 鄰域的模型清算質量。
        帳本狀態只由該時刻**之前**的資料建成 -> 因果。
標準化  逐幣、逐年取百分位排名（絕對質量在不同幣、不同時期不可比）
標籤    現行規格的每筆淨報酬（進場 = 成立 +3 分、停損 3 ATR、持有 480 分、
        成本 Bitget 返佣後 1/1/3 bps）—— **不另訂一套**

===========================================================================
判準（跑之前寫死，事後不放寬）
===========================================================================
    K0  **前置**：`liq_density2` 的 H1 與 H2 都要過。沒過則本檔的密度是
        一個未經驗證的量，K1~K3 一律不解讀。
    K1  主判準：高密度組（前 1/3）減低密度組（後 1/3）的每筆淨報酬，
        日聚類 bootstrap CI **下緣 > 0** 且逐幣 ≥6/9
        -> 「樞紐 × 密集清算流動性 -> 延續」成立
    K2  **單調性**：五分位全格報告。若只有極端一格好、中間亂跳，
        那是雜訊不是效應（mistake.md 2026-09-09 的時段教訓）。
    K3  **混淆對照（必跑）**：密度可能只是「波動大／量大」的替身。
        用 ATR% 與同期成交量各自做同樣的五分位切分並列報告。
        若那兩個也給出同樣大小的分格差，密度就沒有獨立資訊。
    K4  **真·樣本外**：只用前半資料決定門檻（第幾分位算「高」），
        套到後半。**這是今天翻掉時段維度與 C 臂門檻的那一關**，
        不做這一關的任何結論都不算數。
    K5  過關只代表值得**開一條自己的前瞻紀錄**，不得直接改現行規格（§0.92）。
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
import conj_backtest as cb  # noqa: E402
import liq_density2 as L2  # noqa: E402

OUT = HERE / "data" / "results"
HL = L2.HL_MAIN
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return (float("nan"),) * 2
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5))


def collect():
    rows = []
    for sym in ec.CORE9:
        tr, _ = cb.ledger(sym, arm="A")
        ev = [t for t in tr if t["sigk"] == "and" and t["level"]]
        if not ev:
            continue
        probes = [(int(t["anchor_ts"]), float(t["level"])) for t in ev]
        r = L2.build(sym, HL, probes=probes)
        if r is None or len(r) < 5:
            continue
        dens = r[4]
        for t, dv in zip(ev, dens):
            T = pd.Timestamp(t["anchor_ts"], unit="ms", tz="UTC")
            rows.append(dict(sym=sym, ts=t["anchor_ts"], day=T.strftime("%Y-%m-%d"),
                             yr=T.year, dens=float(dv), R=t["R"], Rn=t["R_net"],
                             atrp=t["atr"] / t["entry"]))
    return pd.DataFrame(rows)


def rank_within(d, col):
    return d.groupby(["sym", "yr"])[col].rank(pct=True)


def report(lab, d, key, nq=5):
    print(f"\n=== {lab}：逐五分位（全格，不挑）===")
    print(f"{'分位':>8s} {'n':>6s} {'淨':>9s} {'CI下':>9s} {'幣+':>5s}")
    qs = pd.qcut(d[key], nq, labels=False, duplicates="drop")
    out = []
    for q in sorted(pd.unique(qs.dropna())):
        s = d[qs == q]
        if len(s) < 40:
            continue
        m, l = day_ci(s.Rn.to_numpy(), s.day.to_numpy())
        per = s.groupby("sym").Rn.mean()
        print(f"{int(q)+1:>8d} {len(s):6d} {m:+9.4f} {l:+9.4f} "
              f"{int((per > 0).sum()):4d}/9")
        out.append(m)
    if len(out) >= 3:
        rho = pd.Series(out).corr(pd.Series(range(len(out))), method="spearman")
        print(f"  單調性(Spearman) {rho:+.3f}")
    hi = d[qs == max(pd.unique(qs.dropna()))]
    lo = d[qs == min(pd.unique(qs.dropna()))]
    if len(hi) >= 40 and len(lo) >= 40:
        diff = np.concatenate([hi.Rn.to_numpy(), -lo.Rn.to_numpy()])
        dd = np.concatenate([hi.day.to_numpy(), lo.day.to_numpy()])
        m, l = day_ci(diff, dd)
        ph = hi.groupby("sym").Rn.mean(); pl = lo.groupby("sym").Rn.mean()
        cm = sum(1 for s in ph.index if s in pl.index and ph[s] > pl[s])
        print(f"  高 − 低 = {m:+.4f}  CI下 {l:+.4f}  逐幣高>低 {cm}/{len(ph)}"
              + ("  **過閘**" if l > 0 and cm >= 6 else ""))
        return dict(diff=m, ci_lo=l, coins=cm)
    return None


def main():
    j = OUT / "liq_density2.json"
    if not j.exists():
        print("K0 未過：liq_density2.json 不在（構造驗證還沒跑完）。停。")
        return 1
    v = json.loads(j.read_text(encoding="utf-8"))
    if not (v.get("H1", {}).get("passed") and v.get("H2", {}).get("passed")):
        print("=== K0 前置未過 ===")
        print(f"  H1 {v.get('H1')}")
        print(f"  H2 {v.get('H2')}")
        print("  -> 密度是未經驗證的量，K1~K3 一律不解讀。停。")
        return 1
    print(f"K0 PASS（H1 係數 {v['H1']['coef']:+.4f}、H2 置換 p={v['H2']['p']:.4f}）")
    d = collect()
    d = d[np.isfinite(d.dens)]
    print(f"\n母體 {len(d):,} 筆交會事件（S+D+V 且有價位）"
          f"，密度非零佔 {(d.dens > 0).mean()*100:.1f}%")
    res = {}
    d["r_dens"] = rank_within(d, "dens")
    d["r_atr"] = rank_within(d, "atrp")
    res["K1K2"] = report("K1/K2 清算密度", d, "r_dens")
    res["K3_atr"] = report("K3 混淆對照：ATR%（不是密度）", d, "r_atr")
    mid = d.ts.min() + (d.ts.max() - d.ts.min()) // 2
    tr, te = d[d.ts <= mid], d[d.ts > mid]
    print("\n=== K4 真·樣本外：只用前半決定門檻 ===")
    if len(tr) < 150 or len(te) < 150:
        print("  樣本不足")
    else:
        best, bv = None, -9e9
        for q in (0.5, 0.6, 0.7, 0.8):
            s = tr[tr.r_dens >= q]
            if len(s) < 60:
                continue
            m = float(s.Rn.mean())
            if m > bv:
                bv, best = m, q
        print(f"  前半選到的門檻：密度百分位 >= {best}")
        for lab, s in (("前半（選門檻用的）", tr[tr.r_dens >= best]),
                       ("後半（真·樣本外）", te[te.r_dens >= best]),
                       ("後半·不設門檻對照", te)):
            if len(s) < 40:
                print(f"  {lab:>20s} 樣本不足"); continue
            m, l = day_ci(s.Rn.to_numpy(), s.day.to_numpy())
            per = s.groupby("sym").Rn.mean()
            print(f"  {lab:>20s} n={len(s):5d} 淨 {m:+.4f} CI下 {l:+.4f} "
                  f"{int((per > 0).sum())}/9"
                  + ("  **過閘**" if l > 0 and (per > 0).sum() >= 6 else ""))
        res["K4_threshold"] = best
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "liq_conj.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwritten -> {OUT / 'liq_conj.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
