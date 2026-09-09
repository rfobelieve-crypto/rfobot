# -*- coding: utf-8 -*-
"""SDV 出場：早點放棄輸家 —— 一次，測完就關

===========================================================================
為什麼開這一次（前提已查證，不是憑印象）
===========================================================================
舊線（掃單失敗，§0.97）的出場家族累計 **13 次檢定**，通過的只有兩個：
`hold_4` 與 `fail_fast`，而且**形狀一致 —— 它們砍的是輸家的時間，不是
贏家的空間**。TODO 白紙黑字寫著「不得再對出場做第三輪」，但那條規矩的
範圍是**那條線**；SDV 是不同母體，所以技術上可以開一次。

**開一次，就是一次。** 測完不論結果都關，不得對任何門檻做掃描。

===========================================================================
測什麼 —— 兩個機制，地位不同
===========================================================================
**A. `fail_fast`（註冊候選，零參數）**
舊線的原始定義逐字照抄：
    「price closes back THROUGH the swept level against us
      (the retest thesis is dead). **Mechanism, not a number.**」
    `through = (close < lvl) if d == +1 else (close > lvl)`

同一條判斷式在兩條線上都成立，只是語意鏡像：舊線賭掃單失敗、價格收回，
價格繼續穿出去就是論點死；SDV 賭突破延續，價格**收回價位的錯誤那一側**
就是論點死。**這一臂沒有任何可調的數字**，所以沒有東西可以過擬合。

**B. MAE 早出（探索性，不得採用）**
使用者提的形狀：進場後 N 分鐘內若逆行超過 X ATR 就出。
**(N, X) = (60, 1.0)**，理由是這兩個數字**都已經存在於這條線的歷史**
（60 是被換掉的舊持有、1.0 是被換掉的舊停損），**不是看了本測試的結果
才挑的**。只測這一格，不掃。

B 過關也**不得**採用——它有兩個自由度，而本檔只花一發。要用必須另開
一份自己的預註冊。這條寫在這裡，事後不放寬。

===========================================================================
判準（跑之前寫死 —— 核心原則 #8）
===========================================================================
    E1  **主判準**：真樣本外（後半）配對差（同一批進場、只換出場）的
        日聚類 CI **下緣 > 0** ∧ 逐幣 ≥6/9。
    E2  全格報告：前半／後半／全期都印，判決只看後半。
    E3  **水準另外算**：配對設計對差值有效、對水準無效
        （mistake.md 2026-09-07）。所以除了差值，另報兩套規則各自在
        完整規則下的每筆淨值，不得從配對欄去推水準。
    E4  **它砍的到底是不是輸家**（這一關才對得上開這次的理由）：
        報「被提早出場的那些交易，在**原規則**下的最終結局」。
        若被砍掉的裡面贏家佔比 ≈ 母體贏家佔比，那它只是縮短持有時間，
        **不是「早點放棄輸家」**，就算 E1 過了也只能記成「縮短持有」。
    E5  cost=0 對照必跑（mistake.md 2026-07-28）：含成本結果 ≥ 零成本
        結果就是成本模型壞了。
    E6  過關只代表值得開一條自己的前瞻紀錄，不得直接改現行規格（§0.92）。
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
MAE_N = 60        # 分鐘。舊持有值，不是掃出來的
MAE_X = 1.0       # ATR。舊停損值，不是掃出來的


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 40:
        return (float("nan"),) * 3
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5)), float((r > 0).mean())


def walk(op, hi, lo, cl, n, rd, d, A, lvl, rule, cost=True):
    """一筆交易，可選的提早出場規則。回傳 (R_net, 出場種類, 出場後第幾分鐘)。

    出場種類：'stop' / 'time' / 'early'
    停損與持有與現行規格完全相同；`rule` 只**新增**一個提早離場的機會，
    不會取消停損（所以最壞情況不變，接得回 sizing 與 kill switch）。
    """
    j0 = rd + cb.DELAY
    ent = op[j0]
    sp = ent - d * cb.STOP * A
    ce, ct, cs = (cb.COST_ENTRY, cb.COST_TIME, cb.COST_STOP) if cost else (0., 0., 0.)
    for k in range(j0 + 1, j0 + cb.HOLD + 1):
        # 停損永遠先判（盤中觸價，早於任何收盤規則）
        if (lo[k] <= sp) if d > 0 else (hi[k] >= sp):
            return -cb.STOP - (ce + cs) / 1e4 * ent / A, "stop", k - j0
        if rule == "failfast":
            # 逐字照抄舊線 V6：收盤收回價位的錯誤那一側 -> 論點死
            if (cl[k] < lvl) if d > 0 else (cl[k] > lvl):
                px = cl[k]
                return (float(d * (px - ent) / A) - (ce + ct) / 1e4 * ent / A,
                        "early", k - j0)
        elif rule == "mae":
            if k - j0 <= MAE_N:
                adv = (ent - lo[k]) / A if d > 0 else (hi[k] - ent) / A
                if adv >= MAE_X:
                    px = ent - d * MAE_X * A
                    return (-MAE_X - (ce + cs) / 1e4 * ent / A, "early", k - j0)
    px = cl[j0 + cb.HOLD]
    return float(d * (px - ent) / A) - (ce + ct) / 1e4 * ent / A, "time", cb.HOLD


def build(sym):
    liq = cb._empty_liq()
    cand, ts, cl, at, _ = ck.frozen_cand(sym, liq)
    b = pd.read_parquet(cb.BARS / f"{sym}.parquet",
                        columns=["open", "high", "low", "close"])
    op = b["open"].to_numpy(float)
    hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
    lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
    cls = b["close"].to_numpy(float)
    n = len(ts)

    ev = pd.read_parquet(cb.EVENTS / f"{sym}.parquet",
                         columns=["t_sweep", "sweep_lvl"]).sort_values("t_sweep")
    ev_ts = ev["t_sweep"].to_numpy(np.int64)
    ev_lv = ev["sweep_lvl"].to_numpy(float)

    pairs = [(int(m), "sweep")
             for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
    for nm in cb.FLOW:
        v = cand.get(nm)
        if v is not None and len(v):
            for m in ec.cooldown_filter(np.sort(v)):
                pairs.append((int(m), nm))

    rows = []
    flowm = set(cb.FLOW)
    for _a, mem in cr.groups_with_members(pairs):
        s = {x for _, x in mem}
        if "sweep" not in s or not (s & flowm):
            continue
        if not ({"delta_ext", "vol_burst"} <= s):
            continue                       # 只跑 SDV（三者齊發）
        m_sw = min(m for m, x in mem if x == "sweep")
        ready = max(m_sw, min(m for m, x in mem if x in flowm))
        if ready < cb.W or ready + cb.DELAY + cb.HOLD >= n:
            continue
        A = float(at[ready])
        if not np.isfinite(A) or A <= 0:
            continue
        # 被掃的價位：取掃單那一分鐘對應的事件列
        j = int(np.searchsorted(ev_ts, int(ts[m_sw]), "right")) - 1
        if j < 0 or abs(int(ev_ts[j]) - int(ts[m_sw])) > 60_000:
            continue                       # 對不上就跳過，不猜
        lvl = float(ev_lv[j])
        if not np.isfinite(lvl) or lvl <= 0:
            continue
        d = 1.0 if cl[ready] > cl[ready - cb.W] else -1.0
        base, bk, _ = walk(op, hi, lo, cls, n, ready, d, A, lvl, None)
        ff, fk, fn = walk(op, hi, lo, cls, n, ready, d, A, lvl, "failfast")
        ma, mk, mn = walk(op, hi, lo, cls, n, ready, d, A, lvl, "mae")
        b0, _, _ = walk(op, hi, lo, cls, n, ready, d, A, lvl, None, cost=False)
        f0, _, _ = walk(op, hi, lo, cls, n, ready, d, A, lvl, "failfast", cost=False)
        rows.append(dict(sym=sym, ts=int(ts[ready]),
                         day=pd.Timestamp(int(ts[ready]), unit="ms", tz="UTC")
                              .strftime("%Y-%m-%d"),
                         base=base, ff=ff, mae=ma,
                         base0=b0, ff0=f0,
                         ff_early=(fk == "early"), mae_early=(mk == "early"),
                         ff_min=fn if fk == "early" else np.nan,
                         base_kind=bk, base_win=base > 0))
    return rows


def rep(lab, d, col):
    m, lo_, p = day_ci(d[col].to_numpy(), d.day.to_numpy())
    per = d.groupby("sym")[col].mean()
    ok = lo_ > 0 and int((per > 0).sum()) >= 6
    print(f"  {lab:<26s} n={len(d):5d}  {m:+.4f}  CI下 {lo_:+.4f}  "
          f"P(>0) {p*100:5.1f}%  逐幣 {int((per>0).sum())}/9"
          + ("  **過閘**" if ok else ""))
    return dict(n=len(d), mean=m, ci_lo=lo_, p_pos=p,
                coins=int((per > 0).sum()), passed=bool(ok))


def main():
    rows = []
    for sym in ec.CORE9:
        rows.extend(build(sym))
    d = pd.DataFrame(rows)
    d["d_ff"] = d.ff - d.base
    d["d_mae"] = d.mae - d.base
    mid = d.ts.min() + (d.ts.max() - d.ts.min()) // 2
    res = {"n": len(d)}
    print(f"\nSDV 母體 {len(d):,} 筆（能對到被掃價位的）")

    print("\n=== E1／E2 配對差（同一批進場，只換出場）===")
    for half, g in (("後半（真樣本外·判決）", d[d.ts > mid]),
                    ("前半（樣本內·對照）", d[d.ts <= mid]),
                    ("全期（對照）", d)):
        print(f"  [{half}]")
        res[f"ff_{half}"] = rep("A fail_fast − 現行", g, "d_ff")
        res[f"mae_{half}"] = rep("B MAE(60分,1.0ATR) − 現行", g, "d_mae")

    print("\n=== E3 水準（完整規則各自算，不可從配對欄推）===")
    o = d[d.ts > mid]
    for lab, c in (("現行（3ATR/480）", "base"), ("＋fail_fast", "ff"),
                   ("＋MAE 早出", "mae")):
        rep(lab, o, c)

    print("\n=== E4 它砍的是不是輸家 ===")
    base_wr = d.base_win.mean()
    for lab, flag in (("fail_fast", "ff_early"), ("MAE", "mae_early")):
        e = d[d[flag]]
        if not len(e):
            print(f"  {lab}: 從未觸發"); continue
        wr = e.base_win.mean()
        print(f"  {lab:<10s} 觸發 {len(e):5d} 筆（{len(e)/len(d)*100:4.1f}%）；"
              f"這些交易**在原規則下**的贏家佔比 {wr*100:5.1f}%"
              f"  vs 母體 {base_wr*100:5.1f}%"
              + ("  <- 砍到的多半是輸家" if wr < base_wr - 0.05
                 else "  <- **與母體差不多：這只是縮短持有,不是放棄輸家**"))
        res[f"E4_{lab}"] = dict(n=len(e), frac=len(e)/len(d),
                                win_rate=float(wr), base_win_rate=float(base_wr))
    if d.ff_early.any():
        print(f"  fail_fast 觸發時距進場中位 {d.ff_min.median():.0f} 分鐘")

    print("\n=== E5 cost=0 對照（含成本必須 ≤ 零成本）===")
    for lab, c, c0 in (("現行", "base", "base0"), ("fail_fast", "ff", "ff0")):
        a, z = d[c].mean(), d[c0].mean()
        print(f"  {lab:<10s} 含成本 {a:+.4f}  零成本 {z:+.4f}"
              + ("  OK" if a <= z else "  **成本模型壞了**"))
        res[f"E5_{lab}"] = dict(with_cost=float(a), zero_cost=float(z),
                                ok=bool(a <= z))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sdv_failfast.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwritten -> {OUT / 'sdv_failfast.json'}")
    print("\n本檔測完即關。不得對任何門檻做掃描，不得開第二輪（E6）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
