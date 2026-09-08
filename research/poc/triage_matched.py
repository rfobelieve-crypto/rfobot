# -*- coding: utf-8 -*-
"""交會格的 +0.31 是真的，還是對「事前移動幅度」的選擇？

`event_triage.py`（2026-09-07）分流結果：
    只有掃單      n=4,416    60m  -0.006 ~ +0.006（前 15 分鐘還是負的）
    掃單+強制流   n=2,988    60m  **+0.3088 / MDE 0.1003**，五個 horizon 全中
    只有強制流    n=26,874   60m  -0.017

**這個數字大到不能直接相信。** 同一條線既有的量級是 sweep 60m +0.0709，
交會格是它的 4~7 倍 —— 而 mistake.md 的規矩是「跟同一條線既有量級差一個
數量級，先當儀器壞掉」（2026-09-03 那個 +0.52R 的假發現就是這樣擋下的）。

最可能的機制，而且它只打擊交會格
    要讓三四種**極端分位**事件同時開火，事前 5 分鐘必然有一段劇烈移動。
    而標籤 `impulse x dPrice / ATR` 量的是**延續**，一段暴力級聯的前 5 分鐘
    與後 5 分鐘本來就同號。純掃單格不要求極端流量，所以不吃這個條件。
    旁證：S+O+V+D 最大單日佔變異 20.50%、S+V+D 14.27% —— 集中在級聯日。

作法（沿用 `absorb_matched.py` 那套，已在本專案驗過）
    對每個事件時刻 t，在**同一天、非任何事件、離任何事件 >30 分鐘**的分鐘裡，
    挑一個**事前 5 分鐘移動幅度 |close(m)-close(m-5)|/ATR 最接近**的（caliper
    ±20%）當對照。配對之後兩邊「剛剛動了多少」一樣，剩下的差異就不能再用
    「級聯本來就會延續」解釋。

    對照的方向定義與事件完全相同（`impulse = sign(close(m)-close(m-5))`），
    所以比較的是同一個量，不是事件的延續 vs 對照的原始報酬。

判準（跑之前寫死，寫在 CI 上不寫在點估計上）
    M1 交會格活不活
        配對差（事件 - 對照）在 **60m** 的日聚類 CI **下緣 > 0**
        -> 交會的延續**不是**幅度選擇，SURVIVES
        CI 含零 -> **ARTIFACT-SUSPECT**：原始 +0.3088 不得引用為效應
        CI 上緣 < 0 -> 反向，更嚴重，一律停手查儀器
    M2 三條路要一起報，不挑格
        只有掃單 / 掃單+強制流 / 只有強制流 三條都跑同一套配對，全格報告。
    M3 配對品質關
        配對成功率 < 50% -> INCONCLUSIVE-MATCH（結論不可下）；
        配對後兩邊事前移動幅度的相對差中位 > 5% -> 同上。
    M4 已知答案的對照組（**必須**出現，否則 M1/M2 不解讀）
        「只有強制流」那條原始效應貼零（-0.017），配對後**應該仍然貼零**。
        若配對本身會製造效應，這一格會被它照出來。

**這是判決前的儀器關，不是判決。** 通過只代表「不是幅度選擇」，
不代表交會格可交易——那需要它自己的預註冊（含成本與容量）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_census as ec  # noqa: E402
import event_triage as et  # noqa: E402

W = 5
GUARD_MIN = 30
CALIPER = 0.20
TAUS = [5, 15, 30, 60, 240]
FLOW = {"oi_crash", "vol_burst", "delta_ext", "liq_burst"}
OUT = HERE / "data" / "results"
RNG = np.random.default_rng(20260907)


def lane_of(sig):
    if sig == frozenset({"sweep"}):
        return "只有掃單"
    if "sweep" in sig and (sig & FLOW):
        return "掃單+強制流"
    if "sweep" not in sig and (sig & FLOW):
        return "只有強制流"
    return None


def main():
    sys.path.insert(0, str(HERE.parents[1]))
    from shared.db import get_db_conn
    conn = get_db_conn()
    liq = pd.read_sql("SELECT canonical_symbol s, window_start w, "
                      "liq_total_usd u FROM liquidation_1m", conn)
    conn.close()
    liq["sym"] = liq["s"].str.replace("-USD", "", regex=False)

    rows = []
    tried = matched = 0
    for sym in ec.CORE9:
        cand, ts, cl, at, day, _q = ec.detect_all(sym, liq)
        a, b = collect_symbol(sym, cand, ts, cl, at, rows)
        tried += a
        matched += b

    d = pd.DataFrame(rows)
    report(d, tried, matched, OUT / "triage_matched")


def collect_symbol(sym, cand, ts, cl, at, rows):
    """一個幣的配對與計分。抽出來共用 —— 因果門檻版（`conj_causal.py`）
    必須走**同一套配對**，否則兩份實作會安靜地不同意（mistake.md 2026-08-26）。
    內容逐行來自原本 main() 的迴圈，行為未變。"""
    tried = matched = 0
    if True:
        n = len(ts)
        pairs = []
        for nm in et.NAMES:
            v = cand.get(nm)
            if v is None or len(v) == 0:
                continue
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))
        moments = et.cluster(pairs)
        ev_minutes = np.array(sorted({a for a, _ in moments}), dtype=np.int64)
        if len(ev_minutes) == 0:
            return tried, matched

        # 事前 5 分鐘移動幅度（全序列，向量化）—— 只用 m 之前的資訊
        idx = np.arange(n)
        prev = np.clip(idx - W, 0, n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            premove = np.abs(cl - cl[prev]) / np.where(at > 0, at, np.nan)
        sgn = np.sign(cl - cl[prev])
        sgn[sgn == 0] = 1.0

        by_day = {}
        for a in ev_minutes:
            by_day.setdefault(int(ts[a] // 86_400_000), []).append(int(a))

        for a, sig in moments:
            lane = lane_of(sig)
            if lane is None or a < W or a + max(TAUS) >= n:
                continue
            tried += 1
            d0 = int(ts[a] // 86_400_000)
            same = np.array(by_day[d0], dtype=np.int64)
            lo_i = int(np.searchsorted(ts, d0 * 86_400_000))
            hi_i = int(np.searchsorted(ts, (d0 + 1) * 86_400_000)) - 1
            c = np.arange(max(lo_i, W), min(hi_i, n - max(TAUS) - 1) + 1)
            if len(c) == 0:
                continue
            far = np.abs(c[:, None] - same[None, :]).min(axis=1) > GUARD_MIN
            c = c[far]
            if len(c) == 0:
                continue
            me = premove[a]
            if not np.isfinite(me) or me <= 0:
                continue
            ok = np.isfinite(premove[c]) & (np.abs(premove[c] - me) <= CALIPER * me)
            c = c[ok]
            if len(c) == 0:
                continue
            j = int(c[np.argmin(np.abs(premove[c] - me))])
            matched += 1
            row = dict(sym=sym, lane=lane,
                       # 2026-09-08 附加：完整簽名。`lane` 把
                       # 「掃單+delta」與「掃單+delta+vol」壓成同一格，
                       # 而互斥分解顯示兩者差 0.39 ATR（TODO §1.03）。
                       # 要另開「且」變體的時鐘就需要這一欄。
                       # **純附加**：既有消費者一律只讀 lane，行為未變
                       # （附加後 conj_clock --insample 必須逐位重現
                       # +0.3485，那是這次改動的已知答案對照）。
                       sig="+".join(sorted(sig)),
                       day=pd.Timestamp(int(ts[a]), unit="ms",
                                        tz="UTC").strftime("%Y-%m-%d"),
                       # 2026-09-07 附加：容量關要算的是**同一個持有窗內的
                       # 重疊**，不是同日總量。少了毫秒時戳就只能用日總量近似，
                       # 而那個近似對真正的問題免疫（級聯會同時打九個幣）。
                       ts_ms=int(ts[a]),
                       pre_e=float(me), pre_c=float(premove[j]))
            for tau in TAUS:
                ae = float(at[a])
                ac = float(at[j])
                row[f"e{tau}"] = (float(sgn[a] * (cl[a + tau] - cl[a]) / ae)
                                  if ae > 0 else np.nan)
                row[f"c{tau}"] = (float(sgn[j] * (cl[j + tau] - cl[j]) / ac)
                                  if ac > 0 else np.nan)
            rows.append(row)

    return tried, matched


def report(d, tried, matched, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(str(stem) + ".parquet", index=False)
    rate = matched / tried if tried else 0.0
    relgap = float((np.abs(d.pre_e - d.pre_c) / d.pre_e).median())
    print(f"配對 {matched:,} / {tried:,} = {rate*100:.1f}%   "
          f"（caliper ±{int(CALIPER*100)}%、同日、離任何事件 >{GUARD_MIN} 分）")
    print(f"配對品質：事前移動 |dP|/ATR 事件中位 {d.pre_e.median():.4f}  "
          f"對照中位 {d.pre_c.median():.4f}  相對差中位 {relgap*100:.2f}%")
    m3 = (rate >= 0.50) and (relgap <= 0.05)
    print(f"M3 配對品質關 -> {'PASS' if m3 else '**FAIL — INCONCLUSIVE-MATCH，以下不解讀**'}\n")

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

    print("=== 配對後：事件 vs 同日同幅度對照（全格報告）===\n")
    print(f"{'路':13s} {'n':>7s} {'τ':>5s} {'事件':>9s} {'對照':>9s} "
          f"{'差':>9s} {'日聚類 CI95':>24s}")
    res = {"matched": matched, "tried": tried, "rate": rate,
           "relgap_median": relgap, "M3": bool(m3), "lanes": {}}
    for lane in ("只有掃單", "掃單+強制流", "只有強制流"):
        g = d[d.lane == lane]
        if len(g) < 30:
            print(f"{lane:13s} (n<30)")
            continue
        res["lanes"][lane] = {}
        for tau in TAUS:
            e = g[f"e{tau}"].to_numpy(float)
            c = g[f"c{tau}"].to_numpy(float)
            me, _, _, _ = day_ci(e, g.day.to_numpy())
            mc, _, _, _ = day_ci(c, g.day.to_numpy())
            md, lo, hi, se = day_ci(e - c, g.day.to_numpy())
            res["lanes"][lane][str(tau)] = dict(n=int(len(g)), event=me,
                                                control=mc, diff=md,
                                                ci=[lo, hi], se=se)
            print(f"{lane if tau == TAUS[0] else '':13s} "
                  f"{len(g) if tau == TAUS[0] else '':>7} {tau:>4}m "
                  f"{me:+9.4f} {mc:+9.4f} {md:+9.4f}  [{lo:+.4f},{hi:+.4f}]")
        print()

    print("=== 預註冊判準 ===\n")
    key = "掃單+強制流"
    if key in res["lanes"]:
        c60 = res["lanes"][key]["60"]
        v = ("SURVIVES" if c60["ci"][0] > 0 else
             "**反向 — 停手查儀器**" if c60["ci"][1] < 0 else
             "**ARTIFACT-SUSPECT**")
        print(f"M1 交會格 60m 配對差 {c60['diff']:+.4f}  "
              f"CI [{c60['ci'][0]:+.4f},{c60['ci'][1]:+.4f}]  -> {v}")
        print(f"   （原始未配對值 +0.3088；配對後事件端 {c60['event']:+.4f}、"
              f"對照端 {c60['control']:+.4f}）")
        res["M1"] = dict(verdict=v, **c60)
    if "只有強制流" in res["lanes"]:
        f60 = res["lanes"]["只有強制流"]["60"]
        ok4 = abs(f60["diff"]) < 0.05
        print(f"\nM4 已知答案對照組：只有強制流 60m 配對差 {f60['diff']:+.4f} "
              f"（應仍貼零，|差| < 0.05）-> {'PASS' if ok4 else '**FAIL — 配對本身在製造效應**'}")
        res["M4"] = dict(diff=f60["diff"], passed=bool(ok4))

    Path(str(stem) + ".json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print()
    print("written ->", str(stem) + ".json")
    return res


if __name__ == "__main__":
    main()
