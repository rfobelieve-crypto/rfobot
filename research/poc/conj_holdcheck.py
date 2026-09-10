# -*- coding: utf-8 -*-
"""SDV 的 480 分鐘持有：賺的是訊號，還是單純的曝險？（2026-09-10 預註冊）

使用者 2026-09-10：「如果是 5 分鐘級別持倉 8 小時可能就要思考一下了」。

**這個批評指出一個真實的不對稱，而且它從沒被測過。** SDV 是 5 分鐘尺度的
微觀事件（掃單穿越 ＋ 5 分鐘 delta/量的 p99），持倉卻是 480 分鐘 —— 差 96 倍。

目前支持 480 分的唯一證據是 §1.03d 的「毛利隨持有單調上升，樣本外單調性
+0.907」。**那只證明「持越久賺越多」，沒有排除競爭解釋**：

    賺的可能不是那個掃單，而是「在這個市場曝險 8 小時」本身。

單調上升對兩種解釋都成立 —— 曝險越久，任何有正漂移的東西都賺越多。
要分辨它們，需要的是**對照組**，不是更多的持有長度網格。

===========================================================================
設計（跑之前凍結，事後不放寬）
===========================================================================

三個臂，**唯一差別是進場時點怎麼選**。幣、持有長度、停損 3 ATR、成本模型、
方向規則全部相同 —— 相同才問得出「差別來自哪裡」。

    A  SDV        現行規格：事件成立 +3 分開盤進場
    B1 隨機時點   同幣、同一天、隨機分鐘進場，**方向仍用同一條動能規則**
                  （close[t] > close[t-5] 就做多）
    B2 隨機時點   同幣、同一天、隨機分鐘、**方向也隨機**

為什麼要兩個對照臂（它們回答不同問題）：

    A − B1  = 「掃單＋流量這個事件」本身值多少
              （已經扣掉了「動能方向」的貢獻，因為 B1 也用動能方向）
    A − B2  = 「事件 ＋ 動能方向」合起來值多少
    B1 − B2 = 「動能方向」單獨值多少
    B2      = 純曝險基準。**無成本下它應該貼近零**

**配對**：每一筆 A 配 K 個同幣同日的隨機時點，取那 K 個的平均當對照值。
同日配對消掉「哪一天」的影響（市場整體漂移、波動狀態），剩下的才是時點。

判準（凍結）
    P1  A − B1 的日聚類 bootstrap CI 下緣 > 0
    P2  且逐幣 ≥ 6/9 同號
    P1 ∧ P2 成立 -> SDV 事件本身有超出動能方向的價值
    任一不成立   -> 在這個持有長度上，事件的貢獻無法與「隨機進場＋動能」分開

**核心產出不是過不過，是這張表**：A − B1 隨持有長度怎麼變。

    若差距在 15-60 分鐘就見頂、之後平掉
        -> 訊號的價值集中在前段，後面 420 分鐘是純曝險
        -> 「5 分鐘訊號配 8 小時持倉」這個質疑成立
    若差距隨持有長度持續擴大
        -> 事件確實啟動了一段長行情，480 分鐘有理由

自曝檢查（儀器對不對，跟判決分開）
    S1  A 臂在 HOLD=480 的池化毛利必須重現 conj_backtest.ledger 的值
        （已知答案對照 —— 這是新寫的儀器，factor-research 最後一條）
    S2  B2 在零成本下應貼近零。**顯著非零就先查儀器，不要解讀**
        （可能是隨機時點抽樣有偏、或資料有系統性漂移）

用法
    python research/poc/conj_holdcheck.py            # 全九幣
    python research/poc/conj_holdcheck.py --k 10     # 每筆配 10 個對照
"""
from __future__ import annotations

import argparse
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

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
FLOW = cb.FLOW
W, DELAY, STOP = cb.W, cb.DELAY, cb.STOP
# 持有長度網格。480 是現行規格；往下鋪到 15 分鐘，因為問題正是
# 「這個 5 分鐘事件的價值在哪一段持有長度上就用完了」。
HOLDS = (15, 30, 60, 120, 240, 480)
SEED = 20260910          # 凍結：對照時點的抽樣種子


def _exit_R(op, hi, lo, cl, j0, d, A, hold):
    """進場 open(j0)、停損 STOP×A、持有 hold 分。回傳 (R, stopped, ent)。"""
    ent = float(op[j0])
    end = j0 + hold
    adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
           else (hi[j0 + 1:end + 1] - ent)) / A
    hit = np.flatnonzero(adv >= STOP)
    if len(hit):
        return -STOP, True, ent
    return float(d * (cl[end] - ent) / A), False, ent


def _cost(ent, A, stopped):
    leg = cb.COST_ENTRY + (cb.COST_STOP if stopped else cb.COST_TIME)
    return leg / 1e4 * ent / A


def build(sym, k, rng):
    """回傳這個幣的逐筆長表：一列 = 一筆 SDV × 一個持有長度 × 一個臂。"""
    cand, ts, cl, at, _day = ck.frozen_cand(sym, cb._empty_liq())
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    n = len(ts)
    day = pd.to_datetime(ts, unit="ms").floor("D")
    # 每一天的分鐘索引範圍 —— 對照時點只在「同一天」裡抽
    day_i = pd.Series(np.arange(n)).groupby(day.values).agg(["min", "max"])

    sweeps = ec.cooldown_filter(np.sort(cand["sweep"]))
    pairs = [(int(m), "sweep") for m in sweeps]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))

    rows = []
    maxhold = max(HOLDS)
    for a, mem in cr.groups_with_members(pairs):
        sig = {t for _, t in mem}
        if "sweep" not in sig or not (sig & set(FLOW)):
            continue
        if not ({"delta_ext", "vol_burst"} <= sig):     # 只看 SDV（三者齊發）
            continue
        m_sw = min(m for m, t in mem if t == "sweep")
        m_fl = min(m for m, t in mem if t in FLOW)
        ready = max(m_sw, m_fl)
        if ready < W or ready + DELAY + maxhold >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        d = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)
        j0 = ready + DELAY
        dd = day[j0]
        if dd not in day_i.index:
            continue
        i0, i1 = int(day_i.loc[dd, "min"]), int(day_i.loc[dd, "max"])
        # 對照時點：同幣同日、隨機分鐘。必須留得下最長持有期，
        # 且要有 W 根前置（動能規則要看 close[t-W]）。
        lo_ok, hi_ok = max(i0, W), min(i1, n - maxhold - 2)
        if hi_ok <= lo_ok:
            continue
        picks = rng.integers(lo_ok, hi_ok + 1, size=k)

        for hold in HOLDS:
            R, st, ent = _exit_R(op, hi, lo, cl, j0, d, A, hold)
            rows.append(dict(sym=sym, day=dd, arm="A", hold=hold,
                             R=R, R_net=R - _cost(ent, A, st)))
            for arm in ("B1", "B2"):
                rs, rn = [], []
                for pi, p in enumerate(picks):
                    p = int(p)
                    Ap = float(at[p])
                    if not np.isfinite(Ap) or Ap <= 0:
                        continue
                    if arm == "B1":
                        dp = float(np.sign(cl[p] - cl[p - W]) or 1.0)
                    else:
                        # B2 的方向：用種子決定的固定序列，不隨 hold 變動,
                        # 否則同一個對照時點在不同持有長度上會拿到不同方向,
                        # 那就不是同一個對照了。
                        dp = 1.0 if ((p + pi) % 2 == 0) else -1.0
                    R2, st2, ent2 = _exit_R(op, hi, lo, cl, p, dp, Ap, hold)
                    rs.append(R2)
                    rn.append(R2 - _cost(ent2, Ap, st2))
                if rs:
                    rows.append(dict(sym=sym, day=dd, arm=arm, hold=hold,
                                     R=float(np.mean(rs)),
                                     R_net=float(np.mean(rn))))
    return pd.DataFrame(rows)


def day_boot(vals_by_day, n=2000, rng=None):
    """日聚類 bootstrap：重抽「日」，不是重抽「筆」。"""
    rng = rng or np.random.default_rng(7)
    days = list(vals_by_day.keys())
    if len(days) < 5:
        return (np.nan, np.nan)
    arrs = [np.asarray(vals_by_day[d], float) for d in days]
    out = np.empty(n)
    idx = rng.integers(0, len(days), size=(n, len(days)))
    for i in range(n):
        out[i] = np.concatenate([arrs[j] for j in idx[i]]).mean()
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5,
                    help="每筆 SDV 配幾個同日隨機對照（預設 5）")
    ap.add_argument("--syms", default="")
    a = ap.parse_args()
    syms = a.syms.split(",") if a.syms else list(cb.CORE9)
    rng = np.random.default_rng(SEED)

    parts = [build(s, a.k, rng) for s in syms]
    d = pd.concat(parts, ignore_index=True)
    if d.empty:
        sys.exit("沒有樣本")

    # ---- S1 已知答案對照：A 臂 @480 必須重現 ledger 的池化毛利 ----
    ref = []
    for s in syms:
        tr, _ = cb.ledger(s)
        ref += [t["R"] for t in tr if t["sigk"] == "and"]
    mine = d[(d.arm == "A") & (d.hold == 480)].R
    print("=" * 74)
    print(f"S1 已知答案對照（新儀器必驗）")
    print(f"   ledger  n={len(ref):5}  毛 {np.mean(ref):+.4f}")
    print(f"   本檔    n={len(mine):5}  毛 {mine.mean():+.4f}"
          f"   差 {abs(mine.mean()-np.mean(ref)):.4f}")
    ok1 = abs(mine.mean() - np.mean(ref)) < 0.02
    print(f"   {'PASS' if ok1 else '**FAIL —— 儀器對不上，下面的數字全部不可用**'}")

    # ---- S2 B2 在零成本下應貼近零 ----
    b2 = d[(d.arm == "B2") & (d.hold == 480)]
    g = {k: v.tolist() for k, v in b2.groupby("day").R}
    ci = day_boot(g)
    print(f"\nS2 純曝險基準 B2 @480（零成本）  {b2.R.mean():+.4f}"
          f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"   {'PASS（含零，符合預期）' if ci[0] <= 0 <= ci[1] else '**CI 不含零 —— 先查儀器，不要解讀**'}")

    # ---- 核心表 ----
    print("\n" + "=" * 74)
    print("每筆毛 R（ATR），逐持有長度")
    print(f"{'持有':>5} {'A=SDV':>9} {'B1=隨機時點':>12} {'B2=純曝險':>11}"
          f" {'A−B1':>9} {'CI 下緣':>9} {'幣+':>5}")
    res = {}
    for h in HOLDS:
        sub = d[d.hold == h]
        A = sub[sub.arm == "A"].set_index(["sym", "day"]).R
        B1 = sub[sub.arm == "B1"].set_index(["sym", "day"]).R
        B2 = sub[sub.arm == "B2"].R.mean()
        j = pd.DataFrame({"A": A}).join(pd.DataFrame({"B1": B1}), how="inner")
        j["diff"] = j.A - j.B1
        jr = j.reset_index()
        ci = day_boot({k: v.tolist() for k, v in jr.groupby("day")["diff"]})
        per = jr.groupby("sym")["diff"].mean()
        npos = int((per > 0).sum())
        print(f"{h:5d} {A.mean():+9.4f} {B1.mean():+12.4f} {B2:+11.4f}"
              f" {j['diff'].mean():+9.4f} {ci[0]:+9.4f} {npos:3d}/{len(per)}")
        res[h] = dict(A=float(A.mean()), B1=float(B1.mean()), B2=float(B2),
                      diff=float(j["diff"].mean()), ci_lo=ci[0], ci_hi=ci[1],
                      n_pos=npos, n_sym=int(len(per)), n=int(len(j)))

    h = 480
    r = res[h]
    p1 = r["ci_lo"] > 0
    p2 = r["n_pos"] >= 6
    print("\n" + "=" * 74)
    print(f"判準 @480   P1 CI 下緣 > 0 : {'PASS' if p1 else 'FAIL'}"
          f" ({r['ci_lo']:+.4f})")
    print(f"            P2 逐幣 ≥ 6/9  : {'PASS' if p2 else 'FAIL'}"
          f" ({r['n_pos']}/{r['n_sym']})")
    verdict = ("SDV 事件本身有超出動能方向的價值" if (p1 and p2)
               else "在此持有長度上，事件的貢獻與「隨機進場＋動能」分不開")
    print(f"            -> {verdict}")
    # 價值在哪一段用完
    best = max(HOLDS, key=lambda x: res[x]["diff"])
    print(f"\nA−B1 最大的持有長度 = {best} 分（差 {res[best]['diff']:+.4f}）")
    print(f"480 分相對它 {res[480]['diff'] - res[best]['diff']:+.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "conj_holdcheck.json"
    p.write_text(json.dumps(dict(k=a.k, seed=SEED, s1_pass=bool(ok1),
                                 holds=res), indent=2), encoding="utf-8")
    print(f"\nwritten -> {p}")


if __name__ == "__main__":
    main()
