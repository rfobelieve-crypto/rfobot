# -*- coding: utf-8 -*-
"""交會線的三條救援路徑 —— 使用者 2026-09-09 指定，全部預註冊後才跑

背景：`conj_redef.py` 判出誠實錨點下扣成本不可交易（delay 1~10 淨值
−0.043 ~ −0.060、逐幣 2-3/9）。毛利仍為正且 CI 下緣離零（+0.034 ~ +0.052），
所以死因是**優勢比成本小**，不是沒有訊號。使用者提出三條路，本檔全部量：

    一、讓自己知道得早一點     -> R1 短窗偵測
    二、不追第一波（第二波）    -> R2 再次開火
    三、讓成本變小             -> R3 損益兩平成本

使用者自己講明了第一條的天花板，這裡照抄不打折：
「五分鐘的總和，你不可能在五分鐘還沒過完之前就知道。所以能省的是計算和
等待的時間，省不掉窗口本身。」
-> 所以 R1 量的**不是**「同一個事件早知道」，是「**有多少事件其實在更短
的窗上就已經爆表**」。那是換窗、換母體，不是加速。A3 就是為此設的。

===========================================================================
判準（跑之前寫死，事後不放寬；三條各自獨立，不得互相補位）
===========================================================================
R1 短窗偵測（窗 w ∈ {1,2,3} 分，各自的滾動 30 日 p99，因果門檻同一顆）
    A1  某個 w 在 **delay=1** 的淨值日聚類 CI 下緣 > 0 **且**逐幣 ≥6/9
        -> 這條路活
    A2  全格報告 w × delay，不挑格
    A3  **必報**：短窗事件與現行 5 分窗事件的重疊率與時間差。
        重疊率 < 50% -> 這是**另一個母體**不是「同一件事早知道」，
        必須明講，且它需要自己的前瞻時鐘，不得沿用現行註冊。

R2 第二波
    B1  **先報再次開火機率**：現行定義有 60 分鐘冷卻，所以「開火後再次
        開火」依定義為零 —— 必須把冷卻關掉才看得見。報 (0,60] 分鐘內
        再次出現交會時刻 / 純流量時刻的機率與間隔分布。
        機率 < 20% -> 沒有第二波，這條路直接結束（B2 不解讀）。
    B2  交易第二波：在第二個 ready + 1 分進場，停損／持有／成本全部照現行。
        淨值日聚類 CI 下緣 > 0 且逐幣 ≥6/9 -> 這條路活。
    B3  全格報告：第一波 / 第二波 / 只有第二波存在時的第一波，不挑格。

R3 成本
    C1  對每個 delay 算**損益兩平總成本**（淨值 CI 下緣 = 0 的那個 bps），
        並列出具名情境的實際成本，報差距。
    C2  兩平成本 > 現行情境成本 -> 降成本這條路有空間；否則沒有。
    C3  **掛限價進場要另外驗成交率**（`resting_limit.py` 判掉舊線的那一關）：
        本檔只算「如果成交得到，成本降到 X 會怎樣」，**不宣稱掛得到**。
        延續交易掛限價是逆風方向，成交率必須另測——這裡明寫為未量測。

**共通紀律**：本檔是在看過失敗之後提出的新假設，任何一格通過都**只代表
值得開一條自己的前瞻時鐘從零累積**，不得用來復活現行註冊（§0.92 判掉
C/D 變體的同一件事）。
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
import conj_causal as cc  # noqa: E402
import conj_redef as cr  # noqa: E402

BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "results"
W5 = 5
HOLD = 60
STOP = 1.0
FLOW = ("delta_ext", "vol_burst")
DELAYS = (1, 2, 3, 5)
SHORT_W = (1, 2, 3)
COST_ENTRY, COST_TIME, COST_STOP = 7.0, 3.0, 10.0
SCEN = {                       # (進場, 時間出場, 停損出場) bps
    "A 現行（市價進場）": (7.0, 3.0, 10.0),
    "B 全 taker": (7.0, 6.0, 10.0),
    "C 限價進場": (2.0, 3.0, 10.0),
    "D 限價兩腿": (2.0, 2.0, 10.0),
    "E 零費率場館": (0.0, 0.0, 3.0),
}
RNG = np.random.default_rng(20260909)


def day_ci(x, days, b=1500):
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


def back_sum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[i + 1] - c[np.clip(i + 1 - w, 0, len(x))]


def flow_q(sym, w):
    """窗長 w 的兩個流量量值 —— 與 `event_census.detect_all` 同一個配方，
    只換窗長（5 -> w）。名字沿用 delta_ext / vol_burst，這樣可以直接餵進
    `conj_causal.causal_flags` 的同一組滾動門檻，不另寫一份門檻程式。"""
    b = pd.read_parquet(BARS / f"{sym}.parquet",
                        columns=["ts", "volume", "delta"])
    vol = np.nan_to_num(b["volume"].to_numpy(float), nan=0.0)
    dl = np.nan_to_num(b["delta"].to_numpy(float), nan=0.0)
    n = len(vol)
    ad = back_sum(np.abs(dl), w)
    vw = back_sum(vol, w)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for kd in range(1, 31):
        sh = kd * 1440
        acc[sh:] += vw[:-sh]
        cnt[sh:] += 1
    base = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return {"delta_ext": ad,
            "vol_burst": np.where(base > 0, vw / base, np.nan)}


def ready_of(mem, flow_names=FLOW):
    m_sw = min(m for m, t in mem if t == "sweep")
    m_fl = min(m for m, t in mem if t in flow_names)
    return max(m_sw, m_fl), m_sw, m_fl


def score(op, hi, lo, cl, at, n, ready, j0, d=None):
    """一筆的毛利與停損旗標。與 conj_redef/conj_backtest 同一套算術。"""
    if j0 + HOLD >= n or ready < W5:
        return None
    A = float(at[ready])
    if not np.isfinite(A) or A <= 0:
        return None
    if d is None:
        d = float(np.sign(cl[ready] - cl[ready - W5]) or 1.0)
    ent = float(op[j0])
    end = j0 + HOLD
    adv = ((ent - lo[j0 + 1:end + 1]) if d > 0 else (hi[j0 + 1:end + 1] - ent)) / A
    stopped = bool((adv >= STOP).any())
    R = -STOP if stopped else float(d * (cl[end] - ent) / A)
    return dict(R=R, stopped=stopped, entry=ent, atr=A)


def net_of(g, ent, atr, stopped, legs):
    """向量化：`stopped` 是布林陣列，不能用 Python 的三元式（會拿整個陣列
    去做真值判斷）。用 np.where 逐筆選出場那條腿。"""
    e, t_, s = legs
    exit_bps = np.where(np.asarray(stopped, bool), s, t_)
    return g - (e + exit_bps) / 1e4 * ent / atr


def main():
    rows5, rowsS, rowsW2, refire = [], [], [], []
    limit_rows = {K: [] for K in (1, 2, 3, 5)}
    apc = {}
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
        apc[sym] = float(np.nanmedian(at / np.where(cl > 0, cl, np.nan)))
        dstr = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")

        cand5, _t, _c, _a, _d = cr.ck.frozen_cand(sym, pd.DataFrame(
            {"s": [], "w": [], "u": [], "sym": []}))
        sw_min = ec.cooldown_filter(np.sort(cand5["sweep"]))
        base_pairs = [(int(m), "sweep") for m in sw_min]

        def pairs_for(cand):
            pr = list(base_pairs)
            for nm in FLOW:
                v = cand.get(nm)
                if v is not None and len(v):
                    for m in ec.cooldown_filter(np.sort(v)):
                        pr.append((int(m), nm))
            return pr

        # ---- 現行 5 分窗（對照，也是 R2 的第一波） ----
        p5 = pairs_for(cand5)
        g5 = cr.groups_with_members(p5)
        ready5 = []
        for a, mem in g5:
            sig = {t for _, t in mem}
            if "sweep" not in sig or not (sig & set(FLOW)):
                continue
            rd, _, _ = ready_of(mem)
            ready5.append(rd)
            for k in DELAYS:
                r = score(op, hi, lo, cl, at, n, rd, rd + k)
                if r:
                    rows5.append(dict(sym=sym, day=dstr[rd], delay=k, **r))
            # ---- C3 掛限價：限價 = close(ready)，從 ready+1 起等 K 分鐘 ----
            A = float(at[rd])
            if rd >= W5 and np.isfinite(A) and A > 0:
                d = float(np.sign(cl[rd] - cl[rd - W5]) or 1.0)
                L = float(cl[rd])
                mkt = score(op, hi, lo, cl, at, n, rd, rd + 1)
                for K in limit_rows:
                    if rd + K + 1 + HOLD >= n:
                        continue
                    seg = (lo[rd + 1:rd + 1 + K] if d > 0 else hi[rd + 1:rd + 1 + K])
                    hitk = np.flatnonzero(seg <= L) if d > 0 else np.flatnonzero(seg >= L)
                    if len(hitk):
                        j = rd + 1 + int(hitk[0])
                        end = j + HOLD
                        adv = ((L - lo[j + 1:end + 1]) if d > 0
                               else (hi[j + 1:end + 1] - L)) / A
                        st = bool((adv >= STOP).any())
                        R = -STOP if st else float(d * (cl[end] - L) / A)
                        limit_rows[K].append(dict(
                            sym=sym, day=dstr[j], filled=True, R=R, stopped=st,
                            entry=L, atr=A, R_mkt=mkt["R"] if mkt else np.nan))
                    else:
                        limit_rows[K].append(dict(
                            sym=sym, day=dstr[rd], filled=False, R=np.nan,
                            stopped=False, entry=L, atr=A,
                            R_mkt=mkt["R"] if mkt else np.nan))
        ready5 = np.array(sorted(set(ready5)), np.int64)

        # ---- R1 短窗 ----
        for w in SHORT_W:
            q = flow_q(sym, w)
            cw = cc.causal_flags(q, day)
            pw = pairs_for(cw)
            for a, mem in cr.groups_with_members(pw):
                sig = {t for _, t in mem}
                if "sweep" not in sig or not (sig & set(FLOW)):
                    continue
                rd, _, _ = ready_of(mem)
                # 與 5 分窗最近的那個事件差幾分（A3）
                gap = np.nan
                if len(ready5):
                    j = int(np.argmin(np.abs(ready5 - rd)))
                    gap = float(rd - ready5[j])
                for k in DELAYS:
                    r = score(op, hi, lo, cl, at, n, rd, rd + k)
                    if r:
                        rowsS.append(dict(sym=sym, day=dstr[rd], w=w, delay=k,
                                          gap=gap, **r))

        # ---- R2 第二波：關掉冷卻才看得見 ----
        g0 = cr.groups_with_members(p5, cooldown=0)
        moments = []
        for a, mem in g0:
            sig = {t for _, t in mem}
            rd, m_sw, m_fl = ready_of(mem) if ("sweep" in sig and (sig & set(FLOW))) \
                else (a, None, None)
            moments.append((a, sig, rd))
        m_a = np.array([m[0] for m in moments], np.int64)
        for rd1 in ready5:
            j = int(np.searchsorted(m_a, rd1 + 1))
            nxt_conj = nxt_flow = None
            while j < len(moments) and m_a[j] <= rd1 + 60:
                a2, sig2, rd2 = moments[j]
                if nxt_flow is None and (sig2 & set(FLOW)):
                    nxt_flow = a2
                if nxt_conj is None and "sweep" in sig2 and (sig2 & set(FLOW)):
                    nxt_conj = rd2
                j += 1
            refire.append(dict(sym=sym, day=dstr[rd1],
                               conj=np.nan if nxt_conj is None else nxt_conj - rd1,
                               flow=np.nan if nxt_flow is None else nxt_flow - rd1))
            if nxt_conj is not None:
                r = score(op, hi, lo, cl, at, n, nxt_conj, nxt_conj + 1)
                if r:
                    rowsW2.append(dict(sym=sym, day=dstr[nxt_conj], kind="conj",
                                       gap=int(nxt_conj - rd1), **r))
            if nxt_flow is not None:
                r = score(op, hi, lo, cl, at, n, nxt_flow, nxt_flow + 1)
                if r:
                    rowsW2.append(dict(sym=sym, day=dstr[nxt_flow], kind="flow",
                                       gap=int(nxt_flow - rd1), **r))

    d5 = pd.DataFrame(rows5)
    dS = pd.DataFrame(rowsS)
    d2 = pd.DataFrame(rowsW2)
    dR = pd.DataFrame(refire)
    w = d5[d5.delay == 1].groupby("sym").size()
    apw = float(sum(w[s] * apc[s] for s in w.index) / w.sum())
    res = {"atr_pct_w": apw}

    def rep(df, legs=(COST_ENTRY, COST_TIME, COST_STOP)):
        if len(df) < 30:
            return None
        nt = net_of(df.R.to_numpy(), df.entry.to_numpy(), df.atr.to_numpy(),
                    df.stopped.to_numpy(), legs)
        mg, lg, _ = day_ci(df.R.to_numpy(), df.day.to_numpy())
        mn, ln, hn = day_ci(nt, df.day.to_numpy())
        per = df.assign(nt=nt).groupby("sym").nt.mean()
        return dict(n=int(len(df)), gross=mg, gross_lo=lg, net=mn, net_lo=ln,
                    net_hi=hn, coins_pos=int((per > 0).sum()),
                    stop=float(df.stopped.mean()))

    print("=== 對照：現行 5 分窗（誠實錨點）===")
    print(f"{'delay':>6s} {'n':>6s} {'毛':>9s} {'毛CI下':>9s} {'淨':>9s} "
          f"{'淨CI下':>9s} {'幣+':>5s}")
    for k in DELAYS:
        r = rep(d5[d5.delay == k])
        print(f"{k:6d} {r['n']:6d} {r['gross']:+9.4f} {r['gross_lo']:+9.4f} "
              f"{r['net']:+9.4f} {r['net_lo']:+9.4f} {r['coins_pos']:4d}/9")
        res.setdefault("base", {})[k] = r

    print()
    print("=== R1 短窗偵測（A2 全格，不挑）===")
    print(f"{'窗':>4s} {'delay':>6s} {'n':>6s} {'毛':>9s} {'淨':>9s} "
          f"{'淨CI下':>9s} {'幣+':>5s}")
    a1 = None
    for ww in SHORT_W:
        for k in DELAYS:
            r = rep(dS[(dS.w == ww) & (dS.delay == k)])
            if not r:
                continue
            print(f"{ww:4d} {k:6d} {r['n']:6d} {r['gross']:+9.4f} "
                  f"{r['net']:+9.4f} {r['net_lo']:+9.4f} {r['coins_pos']:4d}/9")
            res.setdefault("short", {})[f"w{ww}d{k}"] = r
            if k == 1 and r["net_lo"] > 0 and r["coins_pos"] >= 6:
                a1 = ww
    print()
    print("=== A3 短窗事件 vs 5 分窗事件（重疊率必報）===")
    for ww in SHORT_W:
        g = dS[(dS.w == ww) & (dS.delay == 1)]
        if not len(g):
            continue
        near = np.abs(g.gap.to_numpy(float))
        ov = float(np.mean(near <= 5))
        earlier = float(np.mean(g.gap.to_numpy(float) < 0))
        print(f"  {ww} 分窗  n={len(g):5d}（5 分窗 {len(d5[d5.delay==1]):,}）"
              f"  ±5 分內對得上 {ov*100:5.1f}%"
              f"  比 5 分窗早 {earlier*100:5.1f}%"
              f"  時間差中位 {np.nanmedian(g.gap):+.0f} 分")
        res.setdefault("overlap", {})[ww] = dict(n=int(len(g)), overlap=ov,
                                                 earlier=earlier)
        if ov < 0.5:
            print(f"      -> **重疊 < 50%：這是另一個母體，不是「同一件事早知道」**")
    print(f"  A1 -> " + (f"**窗 {a1} 過閘**" if a1 else
                         "**沒有任何短窗在 delay=1 過閘**"))

    print()
    print("=== R2 第二波（B1 先報機率）===")
    for c in ("conj", "flow"):
        v = dR[c].to_numpy(float)
        p = float(np.mean(np.isfinite(v)))
        med = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
        lab = "再次交會" if c == "conj" else "再次流量"
        print(f"  {lab}  60 分鐘內再次開火 {p*100:5.1f}%   間隔中位 {med:.0f} 分")
        res.setdefault("refire", {})[c] = dict(p=p, median_gap=med)
    pc = res["refire"]["conj"]["p"]
    if pc < 0.20 and res["refire"]["flow"]["p"] < 0.20:
        print("  -> **B1 未過（< 20%）：沒有第二波，B2 不解讀**")
    print(f"{'第二波':>8s} {'n':>6s} {'毛':>9s} {'淨':>9s} {'淨CI下':>9s} {'幣+':>5s}")
    for c in ("conj", "flow"):
        r = rep(d2[d2.kind == c])
        if not r:
            continue
        lab = "交會" if c == "conj" else "流量"
        print(f"{lab:>8s} {r['n']:6d} {r['gross']:+9.4f} {r['net']:+9.4f} "
              f"{r['net_lo']:+9.4f} {r['coins_pos']:4d}/9")
        res.setdefault("wave2", {})[c] = r

    print()
    print("=== R3 損益兩平成本（C1）===")
    print(f"{'delay':>6s} {'毛CI下(ATR)':>12s} {'兩平總成本(bps)':>16s}")
    for k in DELAYS:
        r = res["base"][k]
        bps = r["gross_lo"] * apw * 1e4
        print(f"{k:6d} {r['gross_lo']:+12.4f} {bps:16.1f}")
        res["base"][k]["breakeven_bps"] = bps
    print()
    print("  具名情境的實際成本（混合停損率後）與判定：")
    print(f"{'情境':>18s} {'實際bps':>9s} {'delay3 淨':>10s} {'淨CI下':>9s} {'幣+':>5s}")
    d3 = d5[d5.delay == 3]
    for lab, legs in SCEN.items():
        r = rep(d3, legs)
        eff = legs[0] + (1 - r["stop"]) * legs[1] + r["stop"] * legs[2]
        print(f"{lab:>18s} {eff:9.1f} {r['net']:+10.4f} {r['net_lo']:+9.4f} "
              f"{r['coins_pos']:4d}/9"
              + ("  <- **過閘**" if r["net_lo"] > 0 and r["coins_pos"] >= 6 else ""))
        res.setdefault("scen", {})[lab] = dict(eff_bps=eff, **r)
    print()
    print("=== C3 掛限價到底成不成交（限價 = close(ready)，等 K 分鐘）===")
    print("  延續交易掛限價是**逆風方向**：價格往你要的方向跑，你的單就沒被吃到。")
    print("  這一關正是 resting_limit.py 判掉舊線的那一關，不測就是用敘述代替數字。")
    print(f"{'等K分':>6s} {'成交率':>7s} {'毛|成交':>9s} {'淨|成交':>9s} "
          f"{'淨CI下':>9s} {'幣+':>5s} {'未成交那批的毛(市價)':>20s}")
    for K in (1, 2, 3, 5):
        f = pd.DataFrame(limit_rows[K])
        if len(f) < 30:
            continue
        fill = f[f.filled]
        r = rep(fill, SCEN["D 限價兩腿"]) if len(fill) >= 30 else None
        miss = f[~f.filled]
        mm = float(miss.R_mkt.mean()) if len(miss) else float("nan")
        if r:
            print(f"{K:6d} {f.filled.mean()*100:6.1f}% {r['gross']:+9.4f} "
                  f"{r['net']:+9.4f} {r['net_lo']:+9.4f} {r['coins_pos']:4d}/9 "
                  f"{mm:+20.4f}")
            res.setdefault("limit", {})[K] = dict(fill_rate=float(f.filled.mean()),
                                                  miss_gross_mkt=mm, **r)
    print("  判準 C3：成交率 ≥ 60% **且** 成交那批扣限價成本後 CI 下緣 > 0。")
    print("  「未成交那批的毛(市價)」是逆選擇的照妖鏡：它若明顯**高於**成交那批，")
    print("  代表限價專門漏掉會賺的那些 —— 那正是舊線死掉的形狀。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conj_rescue.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print()
    print("written ->", OUT / "conj_rescue.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
