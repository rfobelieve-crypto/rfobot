# -*- coding: utf-8 -*-
"""回測引擎自審 —— 使用者質疑「量到的只有那麼少嗎，引擎是不是有問題」

使用者 2026-09-09：「bps 是多小的單位你知道嗎，我一單進場才 0.0 幾%，
能量到的只有那麼少嗎，你的回測系統是不是要有問題，看一下 github 上面的
開源回測系統是怎麼建的」。

這個質疑成立：如果強制平倉級聯真的會走 1~2%，而我只量到 **0.13%**
（0.13 ATR × ATR% 0.985%），那不是「edge 很薄」，是**引擎沒抓到那段行情**。
在繼續推論之前先把引擎本身驗一次。

===========================================================================
對照開源回測引擎（backtrader / vectorbt / freqtrade）的標準做法，逐項自查
===========================================================================
    (1) 進場成交價    標準：訊號在 bar t 成立 -> 用 bar t+1 的 open 成交。
                      本引擎：open(ready+delay)。**一致。**
    (2) 停損成交價    標準：bar 的 low 穿過停損 -> 以停損價成交，
                      **但若 open 已跳空穿過，應以 open 成交**（更差）。
                      本引擎：一律以停損價成交 -> **樂觀**。V3 量它多樂觀。
    (3) 同根 bar 內同時觸及停損與續走：無法從 OHLC 判定先後，
                      標準做法是**保守假設先觸停損**。本引擎：是（先掃停損）。
    (4) 部位重疊      標準：portfolio 層排程。本引擎：逐事件獨立計分、
                      容量另外量（`conj_rescue`）。這是刻意的**分離**，
                      不是遺漏——但引用「每筆 R」時不得同時宣稱「可全吃」。
    (5) 手續費/滑價   分腿 bps 模型。已知缺陷：進場腿未被實盤驗證（§1.03d）。

===========================================================================
判準（跑之前寫死）
===========================================================================
V1  **隨機進場對照（最重要）**：同一台計分機器，把事件時刻換成**同幣同日
    的隨機分鐘**（避開事件 ±60 分），方向沿用同一條 impulse 規則。
    -> 期望 ≈ 0。若 |淨| > 0.02 ATR 或 CI 不含零，**機器本身有偏差**，
    上面所有結論一律作廢。
V2  **獨立重寫對照**：拿 BTC 用一支「逐筆 for 迴圈、不向量化」的樸素實作
    重算同一組交易，與主引擎逐筆比對。差異 > 1e-9 即為實作 bug。
V3  **停損跳空**：統計停損那根 bar 的 open 是否已穿過停損價。若比例高，
    「以停損價成交」就是系統性高估，要報出高估多少。
V4  **行情本身有多大**（直接回答使用者的量級質疑）：事件後 60/480 分鐘，
    (a) 絕對移動 |Δ|/price 的中位與 p90
    (b) 順勢移動 impulse×Δ/price 的中位與平均
    (c) 命中率（順勢移動 > 0 的比例）
    若 (a) 很大但 (b) 很小 -> **行情有走，但方向抓錯或被停損切掉**，
    那才是真正的病灶，而不是「edge 很薄」。
V5  **上界對照**：同一批事件，若能事後選對方向（|Δ| 的絕對值），
    每筆值多少？這是這批事件的理論天花板，用來判斷 0.13 ATR 是
    「訊號弱」還是「方向規則差」。
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
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W5, DELAY, HOLD, STOP = 5, 3, 60, 1.0
FLOW = ("delta_ext", "vol_burst")
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=1500):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (float("nan"),) * 2
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5))


def load(sym):
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "open", "high", "low", "close", "atr_h14"])
    return (b["ts"].to_numpy(np.int64), b["open"].to_numpy(float),
            np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf),
            np.nan_to_num(b["low"].to_numpy(float), nan=np.inf),
            b["close"].to_numpy(float), b["atr_h14"].to_numpy(float))


def events_of(sym):
    cand, _t, _c, _a, _d = cr.ck.frozen_cand(sym, pd.DataFrame(
        {"s": [], "w": [], "u": [], "sym": []}))
    pr = [(int(m), "sweep")
          for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pr.append((int(m), nm))
    out = []
    for a, mem in cr.groups_with_members(pr):
        sg = {t for _, t in mem}
        if "sweep" not in sg or not (sg & set(FLOW)):
            continue
        out.append(max(min(m for m, t in mem if t == "sweep"),
                       min(m for m, t in mem if t in FLOW)))
    return sorted(set(out))


def trade(op, hi, lo, cl, at, n, rd, delay=DELAY, hold=HOLD, stop=STOP):
    """主引擎（向量化）。回傳 dict 或 None。"""
    if rd < W5 or rd + delay + hold >= n:
        return None
    A = float(at[rd])
    if not np.isfinite(A) or A <= 0:
        return None
    d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
    j0 = rd + delay
    ent = float(op[j0])
    end = j0 + hold
    adv = ((ent - lo[j0 + 1:end + 1]) if d > 0 else (hi[j0 + 1:end + 1] - ent)) / A
    k = np.flatnonzero(adv >= stop)
    if len(k):
        jx = j0 + 1 + int(k[0])
        return dict(R=-stop, stopped=True, jx=jx, entry=ent, atr=A, d=d,
                    stop_px=ent - d * stop * A)
    return dict(R=float(d * (cl[end] - ent) / A), stopped=False, jx=end,
                entry=ent, atr=A, d=d, stop_px=ent - d * stop * A)


def trade_naive(op, hi, lo, cl, at, n, rd):
    """V2 獨立重寫：逐分鐘 for 迴圈，不用任何向量化，語意照抄規則書。"""
    if rd < W5 or rd + DELAY + HOLD >= n:
        return None
    A = at[rd]
    if not (A > 0) or not np.isfinite(A):
        return None
    d = 1.0 if cl[rd] > cl[rd - W5] else (-1.0 if cl[rd] < cl[rd - W5] else 1.0)
    j0 = rd + DELAY
    ent = op[j0]
    sp = ent - d * STOP * A
    for i in range(j0 + 1, j0 + HOLD + 1):
        if d > 0:
            if lo[i] <= sp:
                return -STOP
        else:
            if hi[i] >= sp:
                return -STOP
    return float(d * (cl[j0 + HOLD] - ent) / A)


def main():
    res = {}
    real, rand, naive_pairs, gap, moves = [], [], [], [], []
    for sym in ec.CORE9:
        ts, op, hi, lo, cl, at = load(sym)
        n = len(ts)
        ds = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
        ev = events_of(sym)
        evset = np.array(ev, np.int64)
        for rd in ev:
            t = trade(op, hi, lo, cl, at, n, rd)
            if not t:
                continue
            real.append(dict(sym=sym, day=ds[rd], **{k: t[k] for k in
                                                     ("R", "stopped", "entry", "atr")}))
            if t["stopped"]:
                # V3 停損那根的 open 是否已穿過停損價
                o = op[t["jx"]]
                through = (o <= t["stop_px"]) if t["d"] > 0 else (o >= t["stop_px"])
                slip = ((t["stop_px"] - o) if t["d"] > 0 else (o - t["stop_px"])) / t["atr"]
                gap.append(dict(through=bool(through), slip=float(slip)))
            # V4 行情本身
            j0 = rd + DELAY
            for H in (60, 480):
                if j0 + H < n:
                    dd = (cl[j0 + H] - op[j0]) / op[j0]
                    moves.append(dict(H=H, absmove=abs(dd), dirmove=t["d"] * dd,
                                      sym=sym, day=ds[rd]))
            if sym == "BTC":
                nv = trade_naive(op, hi, lo, cl, at, n, rd)
                naive_pairs.append((t["R"], nv))
        # V1 隨機對照：同幣、同一批日子、避開事件 ±60 分
        for rd in ev:
            d0 = ts[rd] // 86_400_000
            loi = int(np.searchsorted(ts, d0 * 86_400_000))
            hii = int(np.searchsorted(ts, (d0 + 1) * 86_400_000)) - 1
            if hii - loi < 200:
                continue
            for _ in range(6):
                c = int(RNG.integers(loi + W5, max(loi + W5 + 1, hii - HOLD - DELAY)))
                if np.abs(evset - c).min() <= 60:
                    continue
                t = trade(op, hi, lo, cl, at, n, c)
                if t:
                    rand.append(dict(sym=sym, day=ds[c], R=t["R"],
                                     stopped=t["stopped"], entry=t["entry"],
                                     atr=t["atr"]))
                break

    dr = pd.DataFrame(real)
    dq = pd.DataFrame(rand)
    print(f"=== 母體：事件 {len(dr):,} 筆 ／ 隨機對照 {len(dq):,} 筆 ===")
    print()

    print("=== V2 獨立重寫對照（BTC，逐筆比對）===")
    a = np.array([x for x, _ in naive_pairs], float)
    b = np.array([y for _, y in naive_pairs], float)
    dmax = float(np.nanmax(np.abs(a - b))) if len(a) else float("nan")
    print(f"  n={len(a):,}   最大逐筆差異 {dmax:.2e}  -> "
          + ("PASS（兩份實作同意）" if dmax < 1e-9 else "**FAIL —— 實作有 bug**"))
    res["V2_max_diff"] = dmax
    print()

    print("=== V1 隨機進場對照（機器本身有沒有偏差）===")
    mr, lr = day_ci(dr.R.to_numpy(), dr.day.to_numpy())
    mq, lq = day_ci(dq.R.to_numpy(), dq.day.to_numpy())
    hiq = mq + (mq - lq)
    print(f"  事件      每筆 {mr:+.4f} ATR   CI下 {lr:+.4f}")
    print(f"  隨機分鐘  每筆 {mq:+.4f} ATR   CI下 {lq:+.4f}   CI上 ~{hiq:+.4f}")
    ok1 = abs(mq) <= 0.02
    print(f"  -> " + ("PASS（隨機 ≈ 0，機器沒有偏差）" if ok1 else
                      "**FAIL —— 隨機也賺/賠，機器有偏差，上面結論作廢**"))
    print(f"  事件相對隨機的超額 {mr - mq:+.4f} ATR")
    res["V1_random"] = mq
    res["V1_event"] = mr
    print()

    print("=== V3 停損成交價有多樂觀（開源引擎會用 open 成交）===")
    dg = pd.DataFrame(gap)
    if len(dg):
        th = float(dg.through.mean())
        sl = float(dg[dg.through].slip.mean()) if dg.through.any() else 0.0
        print(f"  停損筆 {len(dg):,}   其中 open 已穿過停損 {th*100:.1f}%")
        print(f"  這些筆若改用 open 成交，平均多虧 {sl:.4f} ATR")
        print(f"  對全體每筆的高估 ≈ {th*sl*len(dg)/len(dr):.4f} ATR")
        res["V3_gap_through"] = th
        res["V3_overstate"] = float(th * sl * len(dg) / len(dr))
    print()

    print("=== V4 行情本身有多大（直接回答量級質疑）===")
    dm = pd.DataFrame(moves)
    for H in (60, 480):
        g = dm[dm.H == H]
        print(f"  進場後 {H} 分鐘（n={len(g):,}）")
        print(f"    絕對移動 |Δ|/price   中位 {g.absmove.median()*100:.3f}%"
              f"   p90 {g.absmove.quantile(.9)*100:.3f}%"
              f"   平均 {g.absmove.mean()*100:.3f}%")
        print(f"    順勢移動 impulse×Δ   中位 {g.dirmove.median()*100:+.3f}%"
              f"   平均 {g.dirmove.mean()*100:+.3f}%"
              f"   命中率 {(g.dirmove>0).mean()*100:.1f}%")
        res.setdefault("V4", {})[H] = dict(
            abs_median=float(g.absmove.median()), abs_mean=float(g.absmove.mean()),
            dir_mean=float(g.dirmove.mean()), hit=float((g.dirmove > 0).mean()))
    print()
    print("=== V5 上界：事後選對方向的話（這批事件的理論天花板）===")
    for H in (60, 480):
        g = dm[dm.H == H]
        print(f"  {H} 分鐘：每筆 {g.absmove.mean()*100:+.3f}%"
              f"（順勢實得 {g.dirmove.mean()*100:+.3f}%"
              f" = 天花板的 {g.dirmove.mean()/g.absmove.mean()*100:.0f}%）")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_engine_audit.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print()
    print("written ->", OUT / "conj_engine_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
