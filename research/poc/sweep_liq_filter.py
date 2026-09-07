# -*- coding: utf-8 -*-
"""凍結引擎的交易，該不該分成「有清算流」與「沒有清算流」兩堆？

使用者 2026-09-07（看完回測圖之後）：
    「你每個獵取高低點 swing 都進場啊，這不是我們要的。我要的是高低點配合
     OI 推導的清算位，兩者都發生才行。不然遇到連續清算的時候價格就會一直
     延續。如果清算位不多的高低點，是不是反轉的可能性比較高？」

這個假設與 `TRIAGE.md` 的分流結果是同一件事的兩面。分流用的是「事後延續」
的符號（正 = 延續，負 = 反轉）：

    只有掃單        5m -0.033   15m -0.026   60m +0.002   <- 反轉
    掃單 + 強制流   5m +0.187   15m +0.218   60m +0.278   <- 強烈延續

而凍結引擎是一條**反轉**策略（掃買側流動性 -> 做空）。所以伴隨清算流的
掃單，引擎站在錯的一邊；沒有清算流的才是它該做的。**本檔用引擎的真實
交易（R 單位）直接量這個差距**，不是用事件研究的 ATR 單位轉述。

===========================================================================
紀律聲明（先寫，因為這正是這條線翻過船的地方）
===========================================================================
這個濾網是從**已經看過的資料**長出來的。把它加進凍結規則就是 §0.92 變體 B
那個陷阱——事後加一條看似合理的濾網，剛好濾掉表現差的那批，把 FAIL 變成
PASS，而濾掉的正是死最快的交易（存活偏誤）。

所以本檔**不改凍結規則**。它只做三件事：
  1. 量出兩堆的差距，判準寫在跑之前；
  2. 結果進**顯示層**（回測檢視器標記哪些交易屬於哪一堆）；
  3. 若差距成立，開**前瞻時鐘**另行註冊，達標前不動任何規則。

分組變數的時點（無前視，這是本檔唯一會致命的地方）
    清算流事件必須落在**掃單 bar j 之內**（[bar j 開盤, bar j 收盤]）。
    成交發生在 j+1..j+W 的某一根，**嚴格晚於** bar j 收盤，所以「這根掃單
    bar 有沒有伴隨清算流」在成交當下是已知的。這與凍結引擎自己的 `pierce`
    同一個時點（sweep_core 註解：Known at the sweep bar, i.e. strictly
    before the fill）。**不得**用成交 bar 或之後的流事件——那就是
    mistake.md 2026-09-03 那個 +0.52R 假發現的形狀。

流事件的定義沿用因果門檻版（`conj_causal.causal_flags`）：滾動 30 日分位、
逐日更新、只用嚴格更早的分鐘。不是全樣本 p99（那有前視，且無法往前跑）。

===========================================================================
判準（跑之前寫死，寫在 CI 上不寫在點估計上）
===========================================================================
    F1 濾網有沒有價值（主判準）
        meanR（無清算流） − meanR（有清算流）的日聚類 bootstrap
        CI **下緣 > 0**  -> 兩堆確實不同，濾網有價值
        CI 含零 -> INCONCLUSIVE，不得宣稱濾網有效
        CI 上緣 < 0 -> 反向（有清算流反而更好），一律停手查儀器
    F2 剩下那堆自己站不站得住
        「無清算流」那堆的 meanR 日聚類 CI **下緣 > 0**
        （凍結全體是 +0.0363；濾完必須至少不比它差且自己顯著）
    F3 濾掉的比例與逐幣
        全格報告兩堆的 n、佔比、逐幣 meanR。若濾網濾掉 >70% 的交易，
        那不是濾網是換策略，要另行註冊。
    F4 反向對照（**必須**跑，否則 F1 不解讀）
        把分組變數換成**與清算無關但同樣稀有**的量——這裡用「掃單 bar 是
        當日第幾根」的同分位隨機標記，維持兩堆的樣本比例不變。
        對照組的 F1 差距 CI **必須含零**。若對照也顯著，代表差距來自
        「把樣本切成兩堆」本身，不是清算流。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "sweep_failure"))
import sweep_core as sc  # noqa: E402
import event_census as ec  # noqa: E402
import conj_causal as cc  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
FLOW = ("delta_ext", "vol_burst", "oi_crash")
HOUR_MS = 3_600_000
RNG = np.random.default_rng(20260907)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 30:
        return (float("nan"),) * 4
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    reps = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        reps[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return (float(x.mean()), float(np.percentile(reps, 2.5)),
            float(np.percentile(reps, 97.5)), float(np.std(reps, ddof=1)))


def diff_ci(a, da, b_, db, n=2000):
    """兩組獨立的日聚類 bootstrap 差值。"""
    def boot(x, dy):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        x, dy = x[ok], np.asarray(dy)[ok]
        uq, inv = np.unique(dy, return_inverse=True)
        ix = [np.where(inv == k)[0] for k in range(len(uq))]
        out = np.empty(n)
        for i in range(n):
            p = RNG.integers(0, len(uq), len(uq))
            out[i] = x[np.concatenate([ix[k] for k in p])].mean()
        return out
    d = boot(a, da) - boot(b_, db)
    return (float(np.nanmean(a) - np.nanmean(b_)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    for sym in ec.CORE9:
        _cand, mts, _cl, _at, day, q = ec.detect_all(sym, liq)
        caus = cc.causal_flags(q, day)
        flow_min = np.sort(np.concatenate(
            [caus.get(k, np.array([], np.int64)) for k in FLOW]
        )) if any(len(caus.get(k, [])) for k in FLOW) else np.array([], np.int64)
        flow_ts = mts[flow_min] if len(flow_min) else np.array([], np.int64)

        mb = pd.read_parquet(BARS / f"{sym}.parquet",
                             columns=["ts", "high", "low"])
        bts = mb["ts"].to_numpy(np.int64)
        bhi = np.nan_to_num(mb["high"].to_numpy(float), nan=-np.inf)
        blo = np.nan_to_num(mb["low"].to_numpy(float), nan=np.inf)

        b1 = sc.load_csv(str(HERE.parents[0] / "sweep_failure" / ".cache"
                             / f"{sym}USDT_1h.csv"))
        for t in sc.backtest_symbol(b1, detail=True):
            h0 = int(t["sweep_ts"]) * 1000          # 1h 快取的 time 是秒
            h1 = h0 + HOUR_MS
            # 寬窗：掃單 bar 整根小時（第一版用的，保留並列報告）
            w0 = int(np.searchsorted(flow_ts, h0, side="left"))
            w1 = int(np.searchsorted(flow_ts, h1, side="left"))

            # 窄窗：真正的**穿越分鐘** +-5 分鐘。
            # 第一版只有寬窗，60.2% 的掃單 bar 都算「有流」——而定義出 +0.278
            # 的那個效應用的是 +-5 分鐘、只佔 8.7% 的時刻。用一小時分組等於把
            # 交會與非交會混在一起再問有沒有差。這是儀器與假設不匹配，不是結果。
            a = int(np.searchsorted(bts, h0, side="left"))
            z = int(np.searchsorted(bts, h1, side="left"))
            lvl = float(t["level"])
            pm = -1
            if z > a:
                seg = (bhi[a:z] > lvl) if t["kind"] == "buy" else (blo[a:z] < lvl)
                nz = np.flatnonzero(seg)
                if len(nz):
                    pm = int(bts[a + int(nz[0])])
            if pm > 0:
                n0 = int(np.searchsorted(flow_ts, pm - 5 * 60_000, side="left"))
                n1 = int(np.searchsorted(flow_ts, pm + 5 * 60_000, side="right"))
                n_near = n1 - n0
            else:
                n_near = -1                      # 找不到穿越分鐘（分鐘資料缺）
            rows.append(dict(
                sym=sym, R=float(t["R"]), side=t["side"],
                pierce=float(t["pierce"]), stopped=bool(t["stopped"]),
                has_flow=bool(w1 > w0), n_flow=int(w1 - w0),
                n_near=int(n_near), pierce_ts=pm,
                fill_ts=int(t["fill_ts"]) * 1000,
                day=pd.Timestamp(int(t["sweep_ts"]) * 1000, unit="ms",
                                 tz="UTC").strftime("%Y-%m-%d")))

    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "sweep_liq_filter.parquet", index=False)

    yes = d[d.has_flow]
    no = d[~d.has_flow]
    print("=== 凍結引擎的交易，依「掃單 bar 是否伴隨清算流」分兩堆 ===")
    print(f"（清算流 = 因果門檻下的 delta_ext / vol_burst / oi_crash，"
          f"落在掃單 bar 之內，嚴格早於成交）")
    print()
    print(f"{'組':16s} {'n':>7s} {'佔比':>7s} {'meanR':>9s} "
          f"{'日聚類 CI95':>24s} {'逐幣為正':>9s}")
    res = {"n_total": int(len(d))}
    for name, g in (("有清算流", yes), ("無清算流", no), ("全體（凍結）", d)):
        m, lo, hi, se = day_ci(g.R.to_numpy(), g.day.to_numpy())
        pc = g.groupby("sym").R.mean()
        pos = int((pc > 0).sum())
        res[name] = dict(n=int(len(g)), share=len(g) / len(d), mean=m,
                         ci=[lo, hi], se=se, coins_pos=pos)
        print(f"{name:16s} {len(g):7,d} {len(g)/len(d)*100:6.2f}% {m:+9.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}] {pos:>7d}/9")

    print()
    print("=== 預註冊判準 ===")
    print()
    md, dlo, dhi = diff_ci(no.R.to_numpy(), no.day.to_numpy(),
                           yes.R.to_numpy(), yes.day.to_numpy())
    v1 = ("PASS" if dlo > 0 else
          "**反向 — 停手查儀器**" if dhi < 0 else "INCONCLUSIVE")
    print(f"F1 濾網價值：無 − 有 = {md:+.4f}  CI [{dlo:+.4f}, {dhi:+.4f}]  -> {v1}")
    res["F1"] = dict(diff=md, ci=[dlo, dhi], verdict=v1)

    m_no = res["無清算流"]
    v2 = "PASS" if m_no["ci"][0] > 0 else "INCONCLUSIVE"
    print(f"F2 剩下那堆自己站得住：meanR {m_no['mean']:+.4f} "
          f"CI [{m_no['ci'][0]:+.4f}, {m_no['ci'][1]:+.4f}]  -> {v2}"
          f"   （凍結全體 {res['全體（凍結）']['mean']:+.4f}）")
    res["F2"] = dict(verdict=v2)

    drop = len(yes) / len(d)
    v3 = "PASS" if drop <= 0.70 else "**濾掉太多，這是換策略不是濾網**"
    print(f"F3 濾掉比例 {drop*100:.1f}%  -> {v3}")
    res["F3"] = dict(drop_share=drop, verdict=v3)

    # F4 反向對照：同樣的樣本比例，但用與清算無關的隨機標記
    rnd = RNG.random(len(d)) < drop
    a4, b4 = d.R.to_numpy()[~rnd], d.R.to_numpy()[rnd]
    da4, db4 = d.day.to_numpy()[~rnd], d.day.to_numpy()[rnd]
    m4, l4, h4 = diff_ci(a4, da4, b4, db4)
    v4 = ("PASS（含零）" if l4 <= 0 <= h4
          else "**FAIL — 切兩堆本身就會產生差距，F1 不解讀**")
    print(f"F4 反向對照（同比例隨機標記）：{m4:+.4f} "
          f"CI [{l4:+.4f}, {h4:+.4f}]  -> {v4}")
    res["F4"] = dict(diff=m4, ci=[l4, h4], verdict=v4)

    print()
    print("=== 窄窗（穿越分鐘 ±5 分）：依清算流事件個數分級，全格報告 ===")
    print()
    ok = d[d.n_near >= 0]
    print(f"  可定位穿越分鐘的交易 {len(ok):,} / {len(d):,}")
    print()
    print(f"{'流事件數':>8s} {'n':>7s} {'佔比':>7s} {'meanR':>9s} "
          f"{'日聚類 CI95':>24s} {'逐幣為正':>9s}")
    buckets = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 10**9, "3+")]
    grade = {}
    for lo_k, hi_k, lab in buckets:
        g = ok[(ok.n_near >= lo_k) & (ok.n_near <= hi_k)]
        if len(g) < 30:
            print(f"{lab:>8s} {len(g):7,d}  (n<30)")
            continue
        m, lo, hi, _ = day_ci(g.R.to_numpy(), g.day.to_numpy())
        pc = g.groupby("sym").R.mean()
        grade[lab] = dict(n=int(len(g)), share=len(g) / len(ok), mean=m,
                          ci=[lo, hi], coins_pos=int((pc > 0).sum()))
        print(f"{lab:>8s} {len(g):7,d} {len(g)/len(ok)*100:6.2f}% {m:+9.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}] {int((pc>0).sum()):>7d}/9")
    res["graded_near"] = grade
    if "0" in grade and "3+" in grade:
        z0 = ok[ok.n_near == 0]
        z3 = ok[ok.n_near >= 3]
        mg, glo, ghi = diff_ci(z0.R.to_numpy(), z0.day.to_numpy(),
                               z3.R.to_numpy(), z3.day.to_numpy())
        vg = ("PASS" if glo > 0 else
              "**反向**" if ghi < 0 else "INCONCLUSIVE")
        print()
        print(f"  極端對比（0 個 − 3+ 個）：{mg:+.4f} "
              f"CI [{glo:+.4f}, {ghi:+.4f}]  -> {vg}")
        res["graded_extreme"] = dict(diff=mg, ci=[glo, ghi], verdict=vg)

    print()
    print("=== 逐幣（meanR）===")
    print()
    t = d.groupby(["sym", "has_flow"]).R.agg(["count", "mean"]).unstack()
    print(t.to_string(float_format=lambda x: f"{x:+.4f}"))

    (OUT / "sweep_liq_filter.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "sweep_liq_filter.json")
    print()
    print("**不改凍結規則。** 結果只進顯示層；若 F1 成立，另開前瞻時鐘註冊。")


if __name__ == "__main__":
    main()
