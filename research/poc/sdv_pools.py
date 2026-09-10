# -*- coding: utf-8 -*-
"""SDV 換用四種流動性池子，不只 swing pivot（2026-09-10 預註冊）

SDV 的價位表**只有 swing pivot 一種**（1h、PIVOT=10）。而 `events.py`
的檔頭自己寫著前瞻影子帳本畫四種池子（session 2,088 / pdh_pdl 1,215 /
swing 1,076 / pwh_pwl 224），**其中只有 swing 是 pivot**。

舊線 2026-07-29 量過（`research/sweep_failure/level_types.py`）：
**PDH/PDL 比 swing 好**（[[project_variant_b_liquidity]]）。那條線後來
死在成交假設（§1.02），但**訊號有效性沒有被推翻** —— 死的是交易設計，
而 SDV 用的是完全不同的進場（市價、成立 +3 分），不繼承那個假設。

**這是今天唯一一個會讓事件變多的方向**：其他測試都是在現有 1,584 筆裡
找更好的子集，這個是補上一類本來就該在的價位。

===========================================================================
池子定義（照抄 level_types.py，一個字不改）
===========================================================================
    swing     現行：1h K 的 PIVOT=10 樞紐高低（對照組）
    pdh_pdl   前一日的高／低，新的一天第一分鐘起生效
    session   Asia 00-08 / London 07-16 / New York 12-21（UTC 時段），
              **完成的**時段其高低各成一個池
    pwh_pwl   前一 ISO 週的高／低

日界：pdh_pdl 與 pwh_pwl 用 **UTC+8**（與 2026-09-10 的統一決定一致，
§1.03m/o）。session 維持 **UTC 時段** —— 那是市場的交易時段，不是日界，
改了就不是那三個市場了。

穿越判定與現行完全相同：盤中價穿過 **價位 ± 2 ticks**；一個價位被穿過
一次就消耗。流量條件（D/V）、併窗、冷卻、進出場全部沿用現行規格。

===========================================================================
判準（跑之前凍結，每種池子分開計分、全格報告）
===========================================================================
    R1  該池子的 SDV 樣本外每筆淨值 > 0，且日聚類 CI 下緣 > 0
    R2  逐幣 >= 6/9
    R3  事件數比現行 swing 多（否則沒有「讓機會變多」的意義）
    R1 ∧ R2 ∧ R3 -> 該池子可納入候選

**不做「四種混在一起看總分」** —— 混了之後變好不知道是誰的功勞，變差
也不知道該剔除誰（§0.92 的教訓）。同理不挑最好的那一種：全格報告，
每一種各自對照現行 swing。

自曝檢查
    S1  swing 那一格必須重現現行 SDV 的數字（1,584 筆、樣本外 +0.19 附近），
        對不上代表我另寫的穿越判定與 events.py 不一致。
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
import conj_redef as cr  # noqa: E402
import event_census as ec  # noqa: E402

OUT = HERE / "data" / "results"
TZ_MS = 8 * 3600 * 1000              # UTC+8，與 §1.03m 一致
SESSIONS = (("asia", 0, 8), ("london", 7, 16), ("ny", 12, 21))   # UTC 時段
K_TICKS = 2
W, DELAY, STOP, HOLD = cb.W, cb.DELAY, cb.STOP, cb.HOLD
FLOW = cb.FLOW
SEED = 20260910


def pools_time_based(ts, hi, lo, kind):
    """回傳 [(ready_ms, price, side)]，全部在 ready 時刻已完全可知。"""
    out = []
    if kind in ("pdh_pdl", "pwh_pwl"):
        if kind == "pdh_pdl":
            key = (ts + TZ_MS) // 86_400_000
        else:                                   # ISO 週：以 UTC+8 的週一為界
            key = ((ts + TZ_MS) // 86_400_000 + 3) // 7
        df = pd.DataFrame({"k": key, "h": hi, "l": lo})
        g = df.groupby("k").agg(h=("h", "max"), l=("l", "min"))
        ks = g.index.to_numpy()
        for i in range(len(ks) - 1):
            # 第 i 段結束 -> 下一段的第一分鐘起生效
            start = int(ts[np.searchsorted(key, ks[i + 1])])
            out.append((start, float(g.h.iloc[i]), "buyside"))
            out.append((start, float(g.l.iloc[i]), "sellside"))
    else:                                       # session
        h = ((ts + 0) // 3_600_000) % 24        # UTC 小時
        day = ts // 86_400_000
        for nm, s, e in SESSIONS:
            m = (h >= s) & (h < e)
            if not m.any():
                continue
            df = pd.DataFrame({"d": day[m], "h": hi[m], "l": lo[m]})
            g = df.groupby("d").agg(h=("h", "max"), l=("l", "min"))
            for d0, r in g.iterrows():
                # 完成的時段：其結束時刻 = 該日 e 點
                end = int(d0) * 86_400_000 + e * 3_600_000
                out.append((end, float(r.h), "buyside"))
                out.append((end, float(r.l), "sellside"))
    return sorted(out)


def sweeps_from_pools(pools, ts, hi, lo, tick):
    """每個池子第一次被穿過（±2 ticks）的分鐘索引。消耗一次即失效。"""
    res = []
    for ready_ms, px, side in pools:
        i0 = int(np.searchsorted(ts, ready_ms, side="left"))
        if i0 >= len(ts):
            continue
        thr = px + K_TICKS * tick if side == "buyside" else px - K_TICKS * tick
        arr = hi[i0:] if side == "buyside" else lo[i0:]
        j = np.flatnonzero(arr > thr) if side == "buyside" else np.flatnonzero(arr < thr)
        if len(j):
            res.append((i0 + int(j[0]), px, side))
    return sorted(res)


def run_pool(sym, kind):
    cand, ts, cl, at, _ = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close", "tick_size"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    tick = float(b["tick_size"].iloc[0])
    n = len(ts)

    if kind == "swing":
        sw = [(int(m), np.nan, "") for m in np.sort(cand["sweep"])]
    else:
        sw = sweeps_from_pools(pools_time_based(ts, hi, lo, kind), ts, hi, lo, tick)
    sw_idx = ec.cooldown_filter(np.array([x[0] for x in sw], np.int64)) if sw else []

    pairs = [(int(m), "sweep") for m in sw_idx]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            pairs += [(int(m), nm) for m in ec.cooldown_filter(np.sort(v))]

    rows = []
    for _a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig or not ({"delta_ext", "vol_burst"} <= sig):
            continue
        ready = max(min(m for m, t in mem if t == "sweep"),
                    min(m for m, t in mem if t in FLOW))
        if ready < W or ready + DELAY + HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)
        j0 = ready + DELAY
        ent = float(op[j0])
        end = j0 + HOLD
        adv = ((ent - lo[j0 + 1:end + 1]) if d > 0 else (hi[j0 + 1:end + 1] - ent)) / A
        if len(np.flatnonzero(adv >= STOP)):
            R, stopped = -STOP, True
        else:
            R, stopped = float(d * (cl[end] - ent) / A), False
        leg = cb.COST_ENTRY + (cb.COST_STOP if stopped else cb.COST_TIME)
        rows.append(dict(sym=sym, day=int(ts[ready]) // 86_400_000,
                         Rn=R - leg / 1e4 * ent / A))
    return rows, len(sw_idx)


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


def main():
    res = {}
    print(f"{'池子':12} {'掃單':>8} {'SDV':>7} {'期間':8} {'淨/筆':>9} "
          f"{'SE':>7} {'CI下緣':>9} {'幣+':>6}")
    base_n = None
    for kind in ("swing", "pdh_pdl", "session", "pwh_pwl"):
        rows, nsw = [], 0
        for s in cb.CORE9:
            r, k = run_pool(s, kind)
            rows += r
            nsw += k
        d = pd.DataFrame(rows)
        if d.empty:
            print(f"{kind:12} {nsw:8,} {0:7,}  無事件")
            continue
        mid = float(d.day.median())
        line = {}
        for lab, sub in (("全期", d), ("樣本外", d[d.day >= mid])):
            m, se, lo = boot(sub.day.values, sub.Rn.to_numpy())
            per = sub.groupby("sym").Rn.mean()
            line[lab] = dict(n=int(len(sub)), m=m, se=se, lo=lo,
                             npos=int((per > 0).sum()), nsym=int(len(per)))
            print(f"{kind if lab == '全期' else '':12} "
                  f"{nsw if lab == '全期' else '':>8} "
                  f"{len(d) if lab == '全期' else '':>7} {lab:8} "
                  f"{m:+9.4f} {se:7.4f} {lo:+9.4f} "
                  f"{int((per > 0).sum()):3d}/{len(per)}")
        line["n_sweeps"] = nsw
        res[kind] = line
        if kind == "swing":
            base_n = len(d)
        print()

    print("=" * 76)
    print(f"S1 自曝：swing 那格 SDV {res['swing']['全期']['n']:,} 筆"
          f"（現行 ledger 為 1,584）"
          f"  {'PASS' if abs(res['swing']['全期']['n'] - 1584) <= 60 else '**FAIL，穿越判定與 events.py 不一致**'}")
    print()
    print("判準（R1 樣本外淨>0 且 CI 下緣>0 ∧ R2 逐幣≥6/9 ∧ R3 事件比 swing 多）：")
    for kind, v in res.items():
        if kind == "swing":
            continue
        o = v["樣本外"]
        r1 = o["m"] > 0 and o["lo"] > 0
        r2 = o["npos"] >= 6
        r3 = v["全期"]["n"] > base_n
        print(f"  {kind:10} R1{'✓' if r1 else '✗'} R2{'✓' if r2 else '✗'} "
              f"R3{'✓' if r3 else '✗'}（{v['全期']['n']:,} vs swing {base_n:,}）"
              f"  -> {'**可納入候選**' if (r1 and r2 and r3) else '不過'}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sdv_pools.json"
    p.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten -> {p}")


if __name__ == "__main__":
    main()
