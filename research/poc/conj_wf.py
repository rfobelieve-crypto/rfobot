# -*- coding: utf-8 -*-
"""交會線的 walk-forward 樣本外驗證 —— 不開時鐘，用既有 928 天資料判

使用者 2026-09-09：「不要再掛預註冊了，你沒辦法用回測的方式嗎，還是你少了
哪些資料」。

**問對了，而且我先前用錯工具。** 我掃了出場參數（停損 × 持有）之後說「要開
新時鐘從零累積」，但參數搜尋的標準解法是 **walk-forward 樣本外**，不是等
四個月。這個專案自己的規則就寫著：「任何階段：strategy sweep 必須留 OOS
hold-out，禁全資料 fit」（CLAUDE.md）。

**資料也沒缺**：本線定義（sweep ∧ delta_ext ∨ vol_burst）全部來自 1 分鐘
K 線，九幣 928 天完整。OI 因粒度 5 分鐘本來就被排除在 NO-OI 母體外，
清算只有 BTC/ETH 也沒進定義。**缺的是樣本外驗證，不是資料。**

===========================================================================
兩種樣本外，都做，全格報告
===========================================================================
A  **前後半切**（最直觀）：用前半資料選參數，套到後半。
   後半從沒被看過 -> 它的績效就是誠實的樣本外估計。

B  **Walk-forward 滾動**（更接近真實運作）：
   訓練窗 T 天 -> 隔離 E 天（部位最長 960 分，1 天足夠）-> 測試窗 M 天
   -> 往前滾。每一折**只用該折訓練窗**選參數，套到測試窗。
   把所有測試窗的交易池起來 = 樣本外總績效。

參數網格（刻意小，避免選擇雜訊）
   停損 ∈ {2, 3, 5, 8, 無}   持有 ∈ {240, 480, 960}   共 15 格
   進場延遲固定 3 分（那是死線約束不是自由參數，`conj_stopvar` P3 量過）
   選參準則：訓練窗的**淨值平均**（同時報「淨值 CI 下緣」選法當敏感度）

成本：Bitget 標準 maker 2 bps / taker 6 bps -> (2, 2, 6) 分腿

母體：ALL（3,006）與 S+D+V（1,586）兩者都報。
S+D+V 的**事件定義**本身已有前瞻時鐘（`conj_clock_and`，凍結 2026-09-08），
所以那一半不是今天挑的；本檔驗的是**出場參數**那一半。

===========================================================================
判準（跑之前寫死）
===========================================================================
W1  樣本外淨值日聚類 CI 下緣 > 0 且逐幣 ≥6/9 -> **樣本外站得住**
W2  樣本外 / 樣本內 的比值要報。< 50% -> 參數是過擬合的，即使 W1 過也要標明
W3  **選到的參數在各折之間穩不穩**要報。若每折都跳 -> 「最佳參數」是雜訊，
    應改用固定參數（見 W4）
W4  **固定參數對照**（不做任何選擇）：停損 5 / 持有 480 全期套用，
    與「每折重選」比較。若固定參數的樣本外不比重選差 -> 參數選擇沒有價值，
    直接用固定的，自由度更少更可信
W5  **已知答案對照**：全樣本套固定參數必須重現先前結果
    （S+D+V、停損5、持有480、(2,2,6) -> 淨 +0.2948，容差 0.005）
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
import conj_causal as cc  # noqa: E402
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W5_, DELAY = 5, 3
STOPS = (2.0, 3.0, 5.0, 8.0, None)
HOLDS = (240, 480, 960)
LEGS = (2.0, 2.0, 6.0)          # Bitget 標準 maker/taker
EMBARGO_D = 1
REF_W5 = 0.2948
TOL = 0.005
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


def build():
    rows = []
    for sym in ec.CORE9:
        b = pd.read_parquet(BARS / f"{sym}.parquet",
                            columns=["ts", "open", "high", "low", "close", "atr_h14"])
        ts = b["ts"].to_numpy(np.int64)
        op = b["open"].to_numpy(float)
        hi = np.nan_to_num(b["high"].to_numpy(float), nan=-np.inf)
        lo = np.nan_to_num(b["low"].to_numpy(float), nan=np.inf)
        cl = b["close"].to_numpy(float)
        at = b["atr_h14"].to_numpy(float)
        n = len(ts)
        day = ts // 86_400_000
        cand, _t, _c, _a, _d, q = ec.detect_all(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        caus = cc.causal_flags(q, day)
        pr = [(int(m), "sweep")
              for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in ("delta_ext", "vol_burst"):
            v = caus.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pr.append((int(m), nm))
        for a, mem in cr.groups_with_members(pr):
            sg = {t for _, t in mem}
            if "sweep" not in sg or not (sg & {"delta_ext", "vol_burst"}):
                continue
            rd = max(min(m for m, t in mem if t == "sweep"),
                     min(m for m, t in mem if t in ("delta_ext", "vol_burst")))
            if rd < W5_ or rd + DELAY + max(HOLDS) >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            d = float(np.sign(cl[rd] - cl[rd - W5_]) or 1.0)
            j0 = rd + DELAY
            ent = float(op[j0])
            adv = ((ent - lo[j0 + 1:j0 + max(HOLDS) + 1]) if d > 0
                   else (hi[j0 + 1:j0 + max(HOLDS) + 1] - ent)) / A
            r = dict(sym=sym, d0=int(ts[rd] // 86_400_000), entry=ent, atr=A,
                     day=pd.Timestamp(int(ts[rd]), unit="ms",
                                      tz="UTC").strftime("%Y-%m-%d"),
                     both=bool({"delta_ext", "vol_burst"} <= sg))
            for H in HOLDS:
                for S in STOPS:
                    k = f"h{H}s{'x' if S is None else S}"
                    if S is not None and (adv[:H] >= S).any():
                        r[k], r[k + "_st"] = -S, True
                    else:
                        r[k], r[k + "_st"] = float(d * (cl[j0 + H] - ent) / A), False
            rows.append(r)
    return pd.DataFrame(rows)


def net(sub, key):
    g = sub[key].to_numpy(float)
    st = sub[key + "_st"].to_numpy(bool)
    return g - (LEGS[0] + np.where(st, LEGS[2], LEGS[1])) / 1e4 \
        * sub.entry.to_numpy() / sub.atr.to_numpy()


COMBOS = [(H, S) for H in HOLDS for S in STOPS]


def key_of(H, S):
    return f"h{H}s{'x' if S is None else S}"


def pick(train, by="mean"):
    best, bv = None, -1e9
    for H, S in COMBOS:
        k = key_of(H, S)
        if len(train) < 50:
            continue
        nt = net(train, k)
        v = float(np.mean(nt)) if by == "mean" else day_ci(nt, train.day.to_numpy())[1]
        if np.isfinite(v) and v > bv:
            bv, best = v, (H, S)
    return best


def report(lab, nts, days, syms):
    if len(nts) < 30:
        print(f"{lab:>22s}  樣本不足")
        return None
    mn, ln = day_ci(nts, days)
    per = pd.DataFrame({"s": syms, "x": nts}).groupby("s").x.mean()
    pc = int((per > 0).sum())
    print(f"{lab:>22s} {len(nts):6d} {mn:+9.4f} {ln:+9.4f} {pc:4d}/9"
          + ("  *" if ln > 0 and pc >= 6 else ""))
    return dict(n=int(len(nts)), net=mn, net_lo=ln, coins_pos=pc)


def main():
    d = build()
    res = {}
    print(f"=== 母體 ALL {len(d):,} 筆 ／ S+D+V {int(d.both.sum()):,} 筆"
          f"（{d.day.nunique()} 日）===")
    print()
    # W5 已知答案
    sub = d[d.both]
    nt = net(sub, key_of(480, 5.0))
    m5 = float(np.mean(nt))
    ok5 = abs(m5 - REF_W5) < TOL
    print(f"=== W5 已知答案對照：S+D+V/停損5/持有480 淨 {m5:+.4f}"
          f"（先前 {REF_W5:+.4f}）-> " + ("PASS" if ok5 else "**FAIL，不解讀**") + " ===")
    if not ok5:
        return 1
    print()

    for pop, dd in (("ALL", d), ("S+D+V", d[d.both])):
        print(f"########## 母體 {pop}（n={len(dd):,}）##########")
        d0, d1 = int(dd.d0.min()), int(dd.d0.max())
        mid = d0 + (d1 - d0) // 2
        tr = dd[dd.d0 <= mid - EMBARGO_D]
        te = dd[dd.d0 > mid]
        print(f"\n=== A 前後半切（訓練 {len(tr):,} 筆 / 測試 {len(te):,} 筆）===")
        print(f"{'':>22s} {'n':>6s} {'淨':>9s} {'CI下':>9s} {'幣+':>5s}")
        sel = pick(tr)
        print(f"  前半選到的參數：持有 {sel[0]}m／停損 "
              f"{'無' if sel[1] is None else sel[1]}")
        k = key_of(*sel)
        r_is = report("前半（樣本內）", net(tr, k), tr.day.to_numpy(), tr.sym.to_numpy())
        r_oos = report("後半（樣本外）", net(te, k), te.day.to_numpy(), te.sym.to_numpy())
        res.setdefault(pop, {})["split"] = dict(param=[sel[0], sel[1]],
                                                is_=r_is, oos=r_oos)
        if r_is and r_oos and r_is["net"] > 0:
            print(f"  W2 樣本外/樣本內 = {r_oos['net']/r_is['net']*100:.0f}%")

        # B walk-forward
        print(f"\n=== B Walk-forward 滾動（訓練 365 日 / 測試 120 日 / 隔離 1 日）===")
        oos_nt, oos_dy, oos_sy, params = [], [], [], []
        s = d0 + 365
        while s + 120 <= d1:
            trw = dd[(dd.d0 >= s - 365) & (dd.d0 <= s - EMBARGO_D)]
            tew = dd[(dd.d0 > s) & (dd.d0 <= s + 120)]
            if len(trw) >= 100 and len(tew) >= 20:
                p = pick(trw)
                kk = key_of(*p)
                params.append(p)
                oos_nt.append(net(tew, kk))
                oos_dy.append(tew.day.to_numpy())
                oos_sy.append(tew.sym.to_numpy())
            s += 120
        if oos_nt:
            print(f"{'':>22s} {'n':>6s} {'淨':>9s} {'CI下':>9s} {'幣+':>5s}")
            r_wf = report("樣本外池化", np.concatenate(oos_nt),
                          np.concatenate(oos_dy), np.concatenate(oos_sy))
            res[pop]["wf"] = dict(folds=len(params), oos=r_wf,
                                  params=[[a, b] for a, b in params])
            print(f"  W3 各折選到的參數：" +
                  " | ".join(f"{a}m/{'無' if b is None else b}" for a, b in params))
            uniq = len(set(params))
            print(f"     不同組合數 {uniq}/{len(params)} -> "
                  + ("穩定" if uniq <= 2 else "**跳動，最佳參數是雜訊**"))
            # W4 固定參數對照
            fx = key_of(480, 5.0)
            fnt, fdy, fsy = [], [], []
            s = d0 + 365
            while s + 120 <= d1:
                tew = dd[(dd.d0 > s) & (dd.d0 <= s + 120)]
                if len(tew) >= 20:
                    fnt.append(net(tew, fx))
                    fdy.append(tew.day.to_numpy())
                    fsy.append(tew.sym.to_numpy())
                s += 120
            r_fx = report("固定 480m/5ATR", np.concatenate(fnt),
                          np.concatenate(fdy), np.concatenate(fsy))
            res[pop]["fixed"] = r_fx
            if r_wf and r_fx:
                print(f"  W4 固定參數 {'不輸' if r_fx['net'] >= r_wf['net'] else '輸給'}"
                      f"每折重選 -> "
                      + ("**用固定參數即可，自由度更少**"
                         if r_fx["net"] >= r_wf["net"] else "參數選擇有價值"))
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_wf.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print("written ->", OUT / "conj_wf.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
