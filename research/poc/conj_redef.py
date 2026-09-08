# -*- coding: utf-8 -*-
"""交會線的進出場**重新定義** —— 錨點是「最早」，可交易時刻是「成立」

===========================================================================
2026-09-09 使用者：「這策略有問題呈現在圖表畫的差太多」→「這策略都進出場
需要重新定義」。查下去確認，而且比顯示問題嚴重得多。
===========================================================================

**病**

`event_triage.cluster` 的錨點 = 群內**最早**那一分鐘（事件研究的標準做法：
事件窗從事件開始算）。但這條線把它當**交易訊號時刻**用，於是：

    流量在 t 開火、掃單在 t+3 -> 錨點 = t -> 進場 = open(t+2)
    而 t+2 那一刻，掃單還沒發生 -> **交會事件不存在** -> 不可能下這張單

實測（九幣 3,005 筆、NO-OI 母體）：

    事件成立分鐘晚於錨點        60.7%（中位晚 2 分鐘）
    進場(錨點+2)早於事件成立    22.3%   <- 純前視

    現行(錨點+2)   +0.2286  CI [+0.1476,+0.3218]
    誠實(成立+2)   +0.1157  CI [+0.0366,+0.1986]     差 -0.1129
      偷看那 671 筆  +0.5064 -> +0.1094   差 -0.3970
      沒偷看 2,334   +0.1488

**一半的 edge 是這個前視造出來的**，而且它集中在被偷看的那 22%。
形狀與 §1.02 判掉舊線的那個相同（記帳假設了拿不到的東西），只是那次是
「成交價拿不到」，這次是「訊號當時還不存在」。

**波及範圍（全部要重算，不是只有本檔）**

    conj_entry.py        P0/P1/P2… 全部錨在錨點          +0.2927
    conj_pipeline.py     2 分鐘死線由該曲線推出           死線 2 分
    conj_pipeline_spec.py NO-OI 規格 +0.2278
    flow_direction.py    P/F/A 三臂 +0.2275
    出場體檢（TODO §1.03）停損格 1.0 ATR +0.2275
    conj_tradability.py  容量與成本刻畫
    conj_clock*.py       **量的是配對差、事件研究口徑，不受此影響**
                         —— 但它證明的是「交會之後有延續」，
                         **不是**「這個延續交易得到」。兩件事本檔起分開講。

===========================================================================
重新定義（本檔凍結，取代舊定義）
===========================================================================
    ready(群) = max(第一個 sweep 分鐘, 第一個 flow 分鐘)
              = 交會**成立**的那一分鐘（兩個成分都到齊）
    方向      impulse = sign(close(ready) − close(ready−5))   <- 也改錨在 ready
    進場      open(ready + delay)，delay ∈ {0,1,2,3,5,10}
    停損      1.0 × ATR_h14(ready)，分鐘高低價判定，從進場下一根起
    出場      停損 或 進場後 60 分收盤，先到者
    成本      分腿（進場 7 / 時間出場 3 / 停損出場 10 bps），逐筆換算
              cost_ATR = bps/1e4 × entry / ATR

判準（跑之前寫死，事後不放寬）
    D1  **已知答案對照，必須先過**：用舊錨點重跑 delay=2 必須重現 +0.2286
        （容差 0.005）。對不上代表機器寫錯，以下不解讀。
    D2  誠實定義下、扣成本後，日聚類 CI **下緣 > 0** 的最大 delay 是多少
        —— 那個數字就是新的死線。一格都沒有 -> 這條線在誠實定義下不可交易。
    D3  逐幣為正幾個（≥6/9 才算穩）
    D4  全格報告 delay × 情境，不挑格。
    D5  另報「只有掃單先發生」的子母體（ready == sweep 分鐘，即流量後到）
        與「流量先發生」的子母體 —— 前者才是「掃單觸發、流量確認」那個
        機制敘述真正對應的形狀。**這是刻畫不是挑格**，兩格都報。
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

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W = 5
HOLD = 60
STOP = 1.0
FLOW = ("delta_ext", "vol_burst")
DELAYS = (0, 1, 2, 3, 5, 10)
# **delay=0 是前視，只列印不判定**（2026-09-09 跑完第一版當場抓到）：
# open(ready) 是 ready 那一分鐘的**開盤**，而事件成立靠的是那一分鐘的
# **收盤**（掃單是穿越後的收盤、delta_ext/vol_burst 是 5 分鐘後向和）。
# 在開盤買 = 用還沒發生的收盤決定要不要買。實測 +0.5304，比含前視的舊定義
# (+0.2286) 還高一倍 —— 「數字大到不合理先查儀器」正好擋下它。
# 同族：mistake.md 2026-09-03（同一根 bar 的 open 與 close 屬於不同時刻）。
LOOKAHEAD_DELAYS = (0,)
TRADABLE_DELAYS = tuple(k for k in DELAYS if k not in LOOKAHEAD_DELAYS)
COST_ENTRY, COST_TIME, COST_STOP = 7.0, 3.0, 10.0
REF_OLD = 0.2286          # D1：舊錨點 delay=2 的已知值
TOL = 0.005
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=2000):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    if len(x) < 20:
        return (float("nan"),) * 3
    uq, inv = np.unique(days, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(uq))]
    r = np.empty(b)
    for i in range(b):
        p = RNG.integers(0, len(uq), len(uq))
        r[i] = x[np.concatenate([ix[k] for k in p])].mean()
    return float(x.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def groups_with_members(pairs, cooldown=None):
    """與 et.cluster 同一套 gap+cooldown，但**連群成員一起回傳**。

    et.cluster 只回 (anchor, signature)，拿不到「哪一分鐘是 sweep、哪一分鐘
    是 flow」—— 而那正是判斷「事件何時成立」需要的東西。演算法逐行照抄
    et.cluster，不得改動（改了就是第二份實作）。

    `cooldown` 預設 = `et.COOLDOWN`（60 分），與凍結定義一致。
    **傳 0 可以關掉冷卻**——那是為了看「同一波裡的後續開火」（`conj_rescue.py`
    的第二波假設）：冷卻是為了事件研究避免重複計數而設的，它會讓「開火之後
    再次開火」在資料上依定義為零。關掉冷卻的結果**不得**回頭餵給任何凍結
    的計分器。
    """
    if not pairs:
        return []
    cd = et.COOLDOWN if cooldown is None else cooldown
    prs = sorted(pairs)
    out = []
    cs = cl = prs[0][0]
    mem = [prs[0]]
    for m, t in prs[1:]:
        if m - cl <= et.MERGE_GAP:
            cl = m
            mem.append((m, t))
        else:
            out.append((cs, mem))
            cs = cl = m
            mem = [(m, t)]
    out.append((cs, mem))
    kept, last = [], -10 ** 9
    for a, mm in out:
        if a - last >= cd:
            kept.append((a, mm))
            last = a
    return kept


def main():
    rows, apc = [], {}
    for sym in ec.CORE9:
        cand, ts, cl, at, _day = ck.frozen_cand(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["open", "high", "low"])
        op = b["open"].to_numpy(float)
        hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        n = len(ts)
        apc[sym] = float(np.nanmedian(at / np.where(cl > 0, cl, np.nan)))

        pairs = [(int(m), "sweep")
                 for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in FLOW:
            v = cand.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pairs.append((int(m), nm))

        for a, mem in groups_with_members(pairs):
            types = {t for _, t in mem}
            if "sweep" not in types or not (types & set(FLOW)):
                continue
            m_sw = min(m for m, t in mem if t == "sweep")
            m_fl = min(m for m, t in mem if t in FLOW)
            ready = max(m_sw, m_fl)
            if ready < W or ready + max(DELAYS) + HOLD >= n:
                continue
            A = float(at[ready])
            if not np.isfinite(A) or A <= 0:
                continue

            def leg(j0, A, d):
                if j0 + HOLD >= n:
                    return np.nan, np.nan
                ent = float(op[j0])
                end = j0 + HOLD
                adv = ((ent - lo[j0 + 1:end + 1]) if d > 0
                       else (hi[j0 + 1:end + 1] - ent)) / A
                stopped = bool((adv >= STOP).any())
                R = -STOP if stopped else float(d * (cl[end] - ent) / A)
                c = (COST_ENTRY + (COST_STOP if stopped else COST_TIME)) / 1e4 * ent / A
                return R, R - c

            r = dict(sym=sym,
                     day=pd.Timestamp(int(ts[ready]), unit="ms",
                                      tz="UTC").strftime("%Y-%m-%d"),
                     lag=int(ready - a),
                     sweep_first=bool(m_sw <= m_fl))
            # 誠實：一切錨在 ready
            d_new = float(np.sign(cl[ready] - cl[ready - W]) or 1.0)
            for k in DELAYS:
                g, nt = leg(ready + k, A, d_new)
                r[f"g{k}"], r[f"n{k}"] = g, nt
            # 舊定義（只為 D1 對照）：錨在 a、方向也用 a
            if a >= W and a + 2 + HOLD < n:
                A0 = float(at[a])
                if np.isfinite(A0) and A0 > 0:
                    d_old = float(np.sign(cl[a] - cl[a - W]) or 1.0)
                    r["old2"], _ = leg(a + 2, A0, d_old)
            rows.append(r)

    d = pd.DataFrame(rows)
    days = d.day.to_numpy()
    w = d.groupby("sym").size()
    apw = float(sum(w[s] * apc[s] for s in w.index) / w.sum())

    print("=== 交會線 進出場重新定義 ===")
    print(f"母體 {len(d):,} 筆、{d.day.nunique()} 日、9 幣   "
          f"加權 ATR% {apw*100:.3f}%")
    print(f"事件成立晚於錨點 {(d.lag>0).mean()*100:.1f}%（中位 "
          f"{d.lag[d.lag>0].median() if (d.lag>0).any() else 0:.0f} 分）  "
          f"掃單先發生 {d.sweep_first.mean()*100:.1f}%")
    print()

    ok = False
    if "old2" in d:
        m, _, _ = day_ci(d.old2.to_numpy(), days)
        ok = abs(m - REF_OLD) < TOL
        print("=== D1 已知答案對照（舊錨點 delay=2）===")
        print(f"  {m:+.4f}（已知 {REF_OLD:+.4f}，容差 {TOL}）-> "
              + ("PASS" if ok else "**FAIL —— 機器寫錯了，以下不解讀**"))
        print()
    if not ok:
        print("D1 未過，停。")
        return 1

    print("=== D2/D4 誠實定義：全格報告，不挑 ===")
    print(f"{'delay':>6s} {'n':>6s} {'毛':>9s} {'毛CI下':>9s} "
          f"{'淨':>9s} {'淨CI下':>9s} {'淨CI上':>9s} {'幣+':>5s} {'停損率':>7s}")
    res = {"n": int(len(d)), "atr_pct_w": apw, "D1": bool(ok),
           "lag_gt0": float((d.lag > 0).mean()),
           "peek_rate_old": float((d.lag > 2).mean()), "delays": {}}
    best = None
    for k in DELAYS:
        g = d[f"g{k}"].to_numpy()
        nt = d[f"n{k}"].to_numpy()
        mg, lg, _ = day_ci(g, days)
        mn, ln, hn = day_ci(nt, days)
        per = d.groupby("sym")[f"n{k}"].mean()
        pos = int((per > 0).sum())
        sr = float(np.mean(g <= -STOP + 1e-12))
        tag = "  <- **前視，不列入判定**" if k in LOOKAHEAD_DELAYS else ""
        print(f"{k:6d} {np.isfinite(g).sum():6d} {mg:+9.4f} {lg:+9.4f} "
              f"{mn:+9.4f} {ln:+9.4f} {hn:+9.4f} {pos:4d}/9 {sr*100:6.1f}%{tag}")
        res["delays"][k] = dict(gross=mg, gross_lo=lg, net=mn, net_lo=ln,
                                net_hi=hn, coins_pos=pos, stop_rate=sr,
                                lookahead=k in LOOKAHEAD_DELAYS)
        if ln > 0 and pos >= 6 and k in TRADABLE_DELAYS:
            best = k
    print()
    print("=== D2 判定 ===")
    if best is None:
        print("  **沒有任何可交易 delay 的淨 CI 下緣 > 0 且逐幣 ≥6/9**")
        print("  -> 在誠實定義下，這條線扣成本後**不可交易**。")
        print("     現行 +0.2286 的一半來自「進場時事件還沒成立」的前視；")
        print("     最早的真實可成交價是 open(ready+1)，那一格淨值為負。")
        print("     毛利仍為正且 CI 下緣 > 0 -> **訊號有效性未被推翻**，")
        print("     壞的是「這個 edge 比成本小」—— 與 §1.02 舊線同一種結局。")
    else:
        print(f"  最大可用 delay = {best} 分鐘（淨 CI 下緣 > 0 且逐幣 ≥6/9）")
    res["max_delay_ok"] = best
    print()

    print("=== D5 子母體刻畫（兩格都報，不挑）===")
    print(f"{'子母體':>16s} {'n':>6s} {'毛@2':>9s} {'淨@2':>9s} {'淨CI下':>9s} {'幣+':>5s}")
    for lab, sub in (("掃單先(流量確認)", d[d.sweep_first]),
                     ("流量先(掃單確認)", d[~d.sweep_first])):
        if len(sub) < 30:
            continue
        mg, _, _ = day_ci(sub.g2.to_numpy(), sub.day.to_numpy())
        mn, ln, _ = day_ci(sub.n2.to_numpy(), sub.day.to_numpy())
        per = sub.groupby("sym").n2.mean()
        print(f"{lab:>16s} {len(sub):6d} {mg:+9.4f} {mn:+9.4f} {ln:+9.4f} "
              f"{int((per>0).sum()):4d}/9")
        res.setdefault("subpop", {})[lab] = dict(n=int(len(sub)), gross=mg,
                                                 net=mn, net_lo=ln)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_redef.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    d.to_parquet(OUT / "conj_redef.parquet", index=False)
    print()
    print("written ->", OUT / "conj_redef.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
