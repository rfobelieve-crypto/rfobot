# -*- coding: utf-8 -*-
"""停損放哪裡 ＋ 抱 8 小時的話 2 分鐘死線還重要嗎

使用者 2026-09-09 兩個問題：
    (a)「8 小時哪有那麼快」—— 指出整條線的**內部矛盾**：工程前提是
       「2 分鐘死線」（`conj_pipeline.py`），但 `conj_hold.py` 過閘的格子
       要抱 240~960 分鐘。**如果要抱 8 小時，2 分鐘的進場精度不該重要。**
       這不是抬槓，它決定整個工程難題還在不在。
    (b)「止損如果改放在訊號出來的那根 K 棒的最高點跟最低點呢」——
       結構停損取代波動停損。直接對上「進場一下子就出場了」：
       1 ATR 是固定寬度，訊號棒的高低點是**這次事件自己的**寬度。

===========================================================================
量什麼（判準跑之前寫死）
===========================================================================
單位一律 **ATR**（不是 R）。停損距離會因變體而不同，若用 R 當單位，
「被停損 = −1R」會把不同寬度的停損畫成同樣的損失，那是把風險藏起來。
所以停損被打到時記 **−(停損距離/ATR)**，可比較。

停損變體（進場固定 = 成立 +3 分開盤）
    atr1   進場 ∓ 1.0 × ATR          <- 現行
    atr2   進場 ∓ 2.0 × ATR
    bar    **訊號棒**（成立那一分鐘 ready）的低/高點      <- 使用者提的
    win    事件窗 [ready−5, ready] 的低/高點
    ent    [ready, 進場棒] 的低/高點（含進場前的走勢）
    none   無停損

持有  60 / 240 / 480 分
成本  限價兩腿 (2, 2, 10) bps —— 成交率已在 conj_rescue C3 量到 97.8%

    P1  全格報告 6 停損 × 3 持有，不挑格；並報每個變體的**停損距離分布**
        （中位幾個 ATR）與停損率 —— 沒有這兩個數字就看不懂為什麼。
    P2  過閘 = 淨值日聚類 CI 下緣 > 0 且逐幣 ≥6/9；**單格不算數**，
        鄰格（持有的上下一格）要同向。
    P3  **死線檢定**（回答 (a)）：在持有 480 分、停損取 P1 最好的那個變體下，
        掃進場延遲 1/3/5/10/15/30/60 分。
        若淨值從 delay 1 到 delay 30 的衰減 < 20%，
        -> **2 分鐘死線在長持有下不成立**，`conj_pipeline` 的工程結論要改寫。
        這一格是**描述**不是判決：它不改變任何閘門，只說明工程難度。
    P4  這是看過失敗後的搜尋，任何過閘只代表值得開自己的前瞻時鐘（§0.92）。
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
W5, DELAY = 5, 3
HOLDS = (60, 240, 480)
VARS = ("atr1", "atr2", "bar", "win", "ent", "none")
LEGS = (2.0, 2.0, 10.0)
DEADLINE_DELAYS = (1, 3, 5, 10, 15, 30, 60)
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


def collect(delay, holds):
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
        ds = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
        cand, _t, _c, _a, _d = cr.ck.frozen_cand(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        pr = [(int(m), "sweep")
              for m in ec.cooldown_filter(np.sort(cand["sweep"]))]
        for nm in ("delta_ext", "vol_burst"):
            v = cand.get(nm)
            if v is not None and len(v):
                for m in ec.cooldown_filter(np.sort(v)):
                    pr.append((int(m), nm))
        for a, mem in cr.groups_with_members(pr):
            sg = {t for _, t in mem}
            if "sweep" not in sg or not (sg & {"delta_ext", "vol_burst"}):
                continue
            rd = max(min(m for m, t in mem if t == "sweep"),
                     min(m for m, t in mem if t in ("delta_ext", "vol_burst")))
            if rd < W5 or rd + delay + max(holds) >= n:
                continue
            A = float(at[rd])
            if not np.isfinite(A) or A <= 0:
                continue
            d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
            j0 = rd + delay
            ent = float(op[j0])
            # 停損價（順勢方向的反向）
            sp = {
                "atr1": ent - d * 1.0 * A,
                "atr2": ent - d * 2.0 * A,
                "bar": (lo[rd] if d > 0 else hi[rd]),
                "win": (float(np.min(lo[rd - W5:rd + 1])) if d > 0
                        else float(np.max(hi[rd - W5:rd + 1]))),
                "ent": (float(np.min(lo[rd:j0 + 1])) if d > 0
                        else float(np.max(hi[rd:j0 + 1]))),
                "none": None,
            }
            mx = max(holds)
            hseg = hi[j0 + 1:j0 + mx + 1]
            lseg = lo[j0 + 1:j0 + mx + 1]
            row = dict(sym=sym, day=ds[rd], entry=ent, atr=A)
            for v in VARS:
                s = sp[v]
                if s is None:
                    dist = np.nan
                else:
                    dist = (ent - s) / A if d > 0 else (s - ent) / A
                    if not np.isfinite(dist) or dist <= 0:
                        # 停損落在進場的錯邊（訊號棒低點已被穿過）-> 記為
                        # 不可用，不用 fallback 到別的規則（那會偷偷換規則）
                        dist = np.nan
                row[v + "_d"] = dist
                for H in holds:
                    key = f"{v}_h{H}"
                    if s is None or not np.isfinite(dist):
                        if s is None:
                            row[key] = float(d * (cl[j0 + H] - ent) / A)
                            row[key + "_st"] = False
                        else:
                            row[key] = np.nan
                            row[key + "_st"] = False
                        continue
                    adv = ((ent - lseg[:H]) if d > 0 else (hseg[:H] - ent)) / A
                    if (adv >= dist).any():
                        row[key] = -dist
                        row[key + "_st"] = True
                    else:
                        row[key] = float(d * (cl[j0 + H] - ent) / A)
                        row[key + "_st"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def net_col(d, key):
    g = d[key].to_numpy(float)
    st = d[key + "_st"].to_numpy(bool)
    return g - (LEGS[0] + np.where(st, LEGS[2], LEGS[1])) / 1e4 \
        * d.entry.to_numpy() / d.atr.to_numpy()


def main():
    d = collect(DELAY, HOLDS)
    days = d.day.to_numpy()
    res = {"n": int(len(d))}
    print(f"=== 母體 {len(d):,} 筆（誠實錨點、進場 = 成立 +{DELAY} 分、"
          f"成本 限價兩腿）===")
    print()
    print("=== P1a 各停損變體的**距離**與可用性（沒有這個看不懂下面）===")
    print(f"{'變體':>6s} {'距離中位(ATR)':>13s} {'p10':>7s} {'p90':>7s} {'不可用':>7s}")
    for v in VARS:
        if v == "none":
            print(f"{v:>6s} {'—':>13s} {'—':>7s} {'—':>7s} {'—':>7s}")
            continue
        x = d[v + "_d"].to_numpy(float)
        bad = float(np.mean(~np.isfinite(x)))
        f = x[np.isfinite(x)]
        print(f"{v:>6s} {np.median(f):13.3f} {np.percentile(f,10):7.3f} "
              f"{np.percentile(f,90):7.3f} {bad*100:6.1f}%")
        res.setdefault("dist", {})[v] = dict(median=float(np.median(f)),
                                             unusable=bad)
    print("  （「不可用」= 停損落在進場的錯邊，也就是訊號棒的低/高點已經被"
          "穿過；這種筆不交易，**不 fallback 到別的規則**）")
    print()
    print("=== P1b 全格：淨值 / 淨CI下 / 幣+ / 停損率（* = 過閘）===")
    print(f"{'變體':>6s}" + "".join(f"{('持有 '+str(H)+'m'):>28s}" for H in HOLDS))
    grid = {}
    for v in VARS:
        line = f"{v:>6s}"
        for H in HOLDS:
            key = f"{v}_h{H}"
            sub = d[np.isfinite(d[key].to_numpy(float))]
            if len(sub) < 100:
                line += f"{'—':>28s}"
                continue
            nt = net_col(sub, key)
            mn, ln = day_ci(nt, sub.day.to_numpy())
            per = sub.assign(x=nt).groupby("sym").x.mean()
            pc = int((per > 0).sum())
            sr = float(sub[key + "_st"].mean())
            grid[(v, H)] = (mn, ln, pc)
            ok = ln > 0 and pc >= 6
            line += f"{mn:+8.4f}/{ln:+8.4f}/{pc}/{sr*100:3.0f}%{'*' if ok else ' '}"
            res.setdefault("grid", {})[key] = dict(net=mn, net_lo=ln,
                                                   coins_pos=pc, stop=sr,
                                                   n=int(len(sub)))
        print(line)
    print()
    print("=== P2 判定（鄰格同向才算）===")
    win = [(v, H) for (v, H), (mn, ln, pc) in grid.items() if ln > 0 and pc >= 6]
    if not win:
        print("  **沒有任何格子過閘**")
    best = None
    for v, H in sorted(win, key=lambda x: -grid[x][1]):
        i = HOLDS.index(H)
        nb = [grid[(v, HOLDS[i + di])][0] for di in (-1, 1)
              if 0 <= i + di < len(HOLDS) and (v, HOLDS[i + di]) in grid]
        same = sum(1 for x in nb if x > 0)
        okn = same == len(nb) and len(nb) > 0
        print(f"  {v} / 持有 {H}m：淨 {grid[(v,H)][0]:+.4f} "
              f"CI下 {grid[(v,H)][1]:+.4f} 幣 {grid[(v,H)][2]}/9 -> "
              + ("**鄰格同向**" if okn else f"鄰格 {same}/{len(nb)}"))
        if okn and best is None:
            best = v
    res["winners"] = [[v, H] for v, H in win]

    # ---- P3 死線檢定 ----
    bv = best or "atr2"
    print()
    print(f"=== P3 死線檢定：持有 480 分、停損 {bv}，掃進場延遲 ===")
    print("    （回答「8 小時哪有那麼快」—— 若長持有下延遲不重要，"
          "2 分鐘死線就不成立）")
    print(f"{'delay':>6s} {'n':>6s} {'淨':>9s} {'淨CI下':>9s} {'幣+':>5s} {'相對delay1':>11s}")
    base_net = None
    for k in DEADLINE_DELAYS:
        dd = collect(k, (480,))
        key = f"{bv}_h480"
        sub = dd[np.isfinite(dd[key].to_numpy(float))]
        if len(sub) < 100:
            continue
        nt = net_col(sub, key)
        mn, ln = day_ci(nt, sub.day.to_numpy())
        per = sub.assign(x=nt).groupby("sym").x.mean()
        pc = int((per > 0).sum())
        if base_net is None:
            base_net = mn
        rel = mn / base_net * 100 if base_net else np.nan
        print(f"{k:6d} {len(sub):6d} {mn:+9.4f} {ln:+9.4f} {pc:4d}/9 {rel:10.0f}%")
        res.setdefault("deadline", {})[k] = dict(net=mn, net_lo=ln, coins_pos=pc)
    dl = res.get("deadline", {})
    if 1 in dl and 30 in dl and dl[1]["net"] > 0:
        dec = 1 - dl[30]["net"] / dl[1]["net"]
        print(f"  delay 1 -> 30 分的衰減 {dec*100:.0f}%  -> "
              + ("**2 分鐘死線在長持有下不成立**（工程難度大幅下降）"
                 if dec < 0.20 else "死線仍然成立"))
        res["deadline_decay_1_to_30"] = float(dec)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_stopvar.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print()
    print("written ->", OUT / "conj_stopvar.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
