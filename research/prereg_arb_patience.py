# -*- coding: utf-8 -*-
"""§1.24【預註冊】用耐心換速度：只做「已經活了 k 分鐘」的機會（2026-09-11）

===========================================================================
來源與使用者決定
===========================================================================
Quant Arb〈Ultimate Crypto Arbitrage Guide〉：

> 「已經存在幾秒的機會，很可能再存在幾秒；剛出現的很可能被人搶走。
> **如果你在延遲上跟不上，就去交易比較慢的那些機會。**」

使用者 2026-09-11：「我覺得可以做這個，**我們只求穩不求快**」。

我們已經量過入場券（`gate0_arb_persistence.py`）：
P(再活 1 分鐘 | 已活 k) = **0.596 / 0.821 / 0.916 / 0.982**（k=1/3/8/21），
8/8 配對單調，幣內打亂的對照組平坦。**年齡確實預測存活。**

===========================================================================
**動手前要先說清楚的衝突**（這決定了主要指標是什麼）
===========================================================================
§1.20 量到：§0.75 的綁束**不是深度，是事件數**（8 個配對裡 4 個的
收斂事件/年 = 0，全家族 $225/年）。而**年齡濾網會讓事件數更少**。

所以在現有 8 個配對上，「耐心」在數學上是**減法**：
每筆可能變好，但筆數一定變少，而我們缺的正是筆數。

=> **主要指標必須是 `$/年`，不是每筆 bps。**
   用每筆 bps 當主要指標，這個測試幾乎保證會「通過」而且毫無意義——
   因為條件在「活得久」上依定義會選到偏離較大的那些
   （mistake.md 2026-09-09：事後條件化的選擇效應）。

**耐心真正會贏的情境是「機會比資金多」**，那需要先加寬宇宙（§1.20 的下一步、
也是他〈How to level up your arb game〉第 5 條）。所以這支的用途是
**先知道耐心的代價與報酬各是多少**，再決定加寬之後要不要配上它。

===========================================================================
口徑（全部向凍結的計分器借，不自己發明）
===========================================================================
* band：`arb_premium_verdict.json` 的 `sides.{sell,buy}.full.band_bps`
* 事件偵測與收斂定義：**直接呼叫** `arblib.premium_verdict.convergence(
  rows, band, with_starts=True)` —— 它回傳每一段的起點與持續分鐘數
* 成本：`gate0_arb_capacity.json` 的 `band/2 − net@size$`
  （= 掛單進/吃單出、不跨場館、在金額最大的 size 上的來回成本）
* 取樣規模 size$：同一份 §1.20 的 `size_dollar`

**臂**：k ∈ {0, 1, 2, 3, 5, 8, 13, 21} 分鐘。k=0 就是現況。
一段偏離「在 k 可交易」的定義是**它活得比 k 久**（用凍結計分器回傳的分鐘數）。

    進場時的可捕獲 = |dev(起點+k)| − band × CONV_RETURN_FRAC
    每筆淨 = 可捕獲 − 成本
    $/年  = 事件數/年 × size$ × 每筆淨 / 1e4

===========================================================================
自曝檢查（跑之前寫死）
===========================================================================
C1  **對齊關**：凍結計分器回傳的每一個起點，我自己重算的 |dev| 必須 ≥ band
    （那正是它用來開一段的條件）。任何一個不符 = 我的中線重算跟它不一致，
    **以下全部不解讀**。
C2  **k=0 的可捕獲均值必須 ≈ band/2**（凍結毛利模型用的就是半個帶）。
    差太多代表我的「可捕獲」定義跟毛利模型不是同一件事。
C3  **選擇效應要被看見**：報「k 臂的進場 |dev|」與「k 臂的事件數」。
    每筆變好但事件數掉得更凶 -> $/年 變差，這支必須印出來，不准只報每筆。
C4  **打亂對照**：幣內打亂 premium 序列之後重跑。打亂破壞時序，所以
    「年齡 -> 更好」在對照組上應該消失。沒消失 = 我的 k 臂在量別的東西。

===========================================================================
門檻
===========================================================================
**跑之前先只算 SE**（`--se-only`），再把門檻寫進 `THRESHOLDS` 並 commit。
SE 要用**判決會用的那台機器**算（日聚類 bootstrap、重抽「天」之後在組內
重算兩臂的 $/天再相減）——不是手寫解析近似（mistake.md 2026-09-06）。

    python research/prereg_arb_patience.py --se-only
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import arb_home as AH                          # noqa: E402
AH.add_to_path()

OUT = ROOT / "research" / "results" / "prereg_arb_patience.json"
CAP = ROOT / "research" / "results" / "gate0_arb_capacity.json"
AGES = (0, 1, 2, 3, 5, 8, 13, 21)
SEED = 20260911
BOOT = 4000

# 門檻**刻意留空**，等 SE 算出來再寫（見檔頭）。
THRESHOLDS = None


def rows_for(pid, sub):
    """逐分鐘列，欄位名與 premium_verdict.load() 對齊（prem/ts）。"""
    import glob
    base = AH.LOGS / sub
    out = []
    for fp in [Path(x) for x in sorted(glob.glob(str(base) + "*.old"))] + [base]:
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        if not {"minute_ts", "premium_mean_bps"}.issubset(d.columns):
            continue
        d = d[["minute_ts", "premium_mean_bps"]].dropna()
        out.append(d)
    if not out:
        return None
    d = pd.concat(out, ignore_index=True).sort_values("minute_ts")
    d = d.drop_duplicates("minute_ts")
    return [{"ts": int(a), "prem": float(b)}
            for a, b in zip(d.minute_ts, d.premium_mean_bps)]


def episodes(rows, band):
    """**呼叫凍結的計分器**拿每段的起點與持續分鐘，不自己偵測。"""
    from arblib import premium_verdict as PV
    c = PV.convergence(rows, band, with_starts=True)
    return c, PV


def walk(rows, band, starts, PV, ages=AGES):
    """對每一段，算各年齡臂的進場 |dev| 與是否仍在帶上。

    中線的重算必須與 `convergence` 一致 —— C1 就是驗這件事。
    """
    ts = np.array([r["ts"] for r in rows])
    pr = np.array([r["prem"] for r in rows])
    idx = {int(t): i for i, t in enumerate(ts)}
    W = PV.MIDLINE_WIN
    recs, c1_bad = [], 0
    for s_ts, dur in starts:
        i = idx.get(int(s_ts))
        if i is None or i < W:
            continue
        mid = st.median(pr[i - W:i].tolist())
        dev0 = pr[i] - mid
        if abs(dev0) < band - 1e-9:            # C1：必須 >= band
            c1_bad += 1
            continue
        sign = 1 if dev0 > 0 else -1
        rec = dict(ts=int(s_ts), dur=dur, band=band, dev0=abs(dev0))
        for k in ages:
            j = i + k
            if j >= len(pr) or (dur is not None and dur < k):
                rec["dev%d" % k] = np.nan
                rec["alive%d" % k] = 0
                continue
            devk = sign * (pr[j] - mid)
            rec["dev%d" % k] = devk
            rec["alive%d" % k] = int(devk >= band * PV.CONV_RETURN_FRAC)
        recs.append(rec)
    return pd.DataFrame(recs), c1_bad


def arm_table(df, cost_bps, band, PV, size_usd, days):
    """每個年齡臂的事件數／每筆淨／$天。"""
    out = {}
    for k in AGES:
        d = df[df["alive%d" % k] == 1]
        if not len(d):
            out[k] = dict(n=0, capt=np.nan, net=np.nan, usd_day=0.0,
                          dev_entry=np.nan)
            continue
        capt = (d["dev%d" % k] - band * PV.CONV_RETURN_FRAC).clip(lower=0)
        net = capt - cost_bps
        out[k] = dict(n=int(len(d)), capt=float(capt.mean()),
                      net=float(net.mean()),
                      usd_day=float(len(d) / days * size_usd * net.mean() / 1e4),
                      dev_entry=float(d["dev%d" % k].mean()))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--se-only", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="C4 打亂對照")
    a = ap.parse_args()

    pv = json.loads((AH.RESULTS / "arb_premium_verdict.json")
                    .read_text(encoding="utf-8"))
    cap = json.loads(CAP.read_text(encoding="utf-8")) if CAP.exists() else {"pairs": {}}
    sub = {"SNDK": "minutes.csv"}
    for p in ("NBIS", "ANTH", "BTC", "HYPE", "ZEC", "NEAR", "GOLD_LL", "NVDA_LL"):
        sub[p] = "%s/minutes.csv" % p

    res = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"), ages=list(AGES),
               shuffled=bool(a.shuffle), pairs={})
    rng = np.random.default_rng(SEED)
    c1_total, per_pair, detail = 0, {}, []

    print("=== 逐配對：事件數 / 進場 |dev| / 每筆淨 / $天 ===")
    print("（成本 = band/2 − §1.20 的 net@size$；size$ 同一份）\n")
    for pid, s in sub.items():
        rows = rows_for(pid, s)
        if not rows or len(rows) < 100:
            continue
        sd = ((pv.get("pairs") or {}).get(pid) or {}).get("sides") or {}
        capp = (cap.get("pairs") or {}).get(pid) or {}
        size = capp.get("size_dollar")
        net_at = capp.get("net_at_dollar")
        # **只跑一側**（band 較大的那一側）。第一版對兩側各跑一次
        # `convergence(rows, band)`，而那支用的是 `abs(dev) >= band`
        # —— 它本來就同時抓兩個方向，所以跑兩次等於**重複計數同一批偏離**，
        # 合計 $/年 因此被放大。成本模型 (`family_specs`) 選的也是
        # `max(..., key=band_bps)` 那一側，兩邊必須用同一側才配得上。
        bands = {L: (((sd.get(L) or {}).get("full") or {}).get("band_bps") or 0)
                 for L in ("sell", "buy")}
        for side in [max(bands, key=lambda L: bands[L])]:
            band = bands[side] or None
            if not band:
                continue
            r = rows
            if a.shuffle:
                pr = np.array([x["prem"] for x in r])
                rng.shuffle(pr)
                r = [{"ts": x["ts"], "prem": float(p)} for x, p in zip(r, pr)]
            c, PV = episodes(r, band)
            if not c.get("episodes") or not c.get("starts"):
                continue
            df, bad = walk(r, band, c["starts"], PV)
            c1_total += bad
            if not len(df):
                continue
            days = max((r[-1]["ts"] - r[0]["ts"]) / 86400, 1e-9)
            if size and net_at is not None:
                cost = band / 2.0 - net_at
            else:
                size, cost = 1000.0, band / 2.0     # 沒有 §1.20 數字時保守
            # **來回成本不可能是負的。** 第一版因為拿「較大那一側的 net」
            # 去配「另一側的 band/2」而跑出 −1.48 / −0.47 —— 那是
            # mistake.md 2026-09-07 的形狀（一個在 A 構造下算的量拿去配 B）。
            # 物理上說不通的數字要在原地擋下來，不要讓它流進合計。
            if cost <= 0:
                print("%-12s **成本 %.2f <= 0,跳過**（band 與 net 不同側？）"
                      % (pid + ":" + side, cost))
                continue
            t = arm_table(df, cost, band, PV, size, days)
            # 逐段明細留著：SE 要用判決會用的那台機器算（日聚類 bootstrap）
            df2 = df.copy()
            df2["day"] = pd.to_datetime(df2.ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
            df2["_pair"] = pid + ":" + side
            df2["_size"] = size
            df2["_cost"] = cost
            df2["_band"] = band
            detail.append(df2)
            per_pair["%s:%s" % (pid, side)] = dict(
                band=band, cost_bps=round(cost, 3), size_usd=size,
                days=round(days, 2), arms={str(k): v for k, v in t.items()})
            print("%-12s band %6.2f  成本 %6.2f  size $%-7s %.1f 天"
                  % (pid + ":" + side, band, cost, int(size), days))
            print("   %-5s" % "k" + "".join("%9s" % ("k=%d" % k) for k in AGES))
            print("   %-5s" % "筆數" + "".join("%9d" % t[k]["n"] for k in AGES))
            print("   %-5s" % "進場dev" + "".join(
                ("%9.2f" % t[k]["dev_entry"]) if np.isfinite(t[k]["dev_entry"])
                else "%9s" % "—" for k in AGES))
            print("   %-5s" % "每筆淨" + "".join(
                ("%+9.2f" % t[k]["net"]) if np.isfinite(t[k]["net"])
                else "%9s" % "—" for k in AGES))
            print("   %-5s" % "$天" + "".join("%+9.3f" % t[k]["usd_day"]
                                              for k in AGES))
            print()
    res["pairs"] = per_pair
    res["C1_misaligned_starts"] = c1_total
    print("=== C1 對齊關 ===")
    print("  凍結計分器的起點裡，我重算 |dev| < band 的有 **%d** 個 -> %s"
          % (c1_total, "PASS" if c1_total == 0 else "**FAIL，以下不解讀**"))
    if c1_total:
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        return 2

    # ---------- 合計：$天 逐臂 ----------
    print("\n=== 全家族合計 $/天（**主要指標**）===")
    tot = {k: sum((v["arms"][str(k)]["usd_day"] or 0.0)
                  for v in per_pair.values()) for k in AGES}
    ntot = {k: sum(v["arms"][str(k)]["n"] for v in per_pair.values()) for k in AGES}
    print("   %-8s" % "k" + "".join("%10s" % ("k=%d" % k) for k in AGES))
    print("   %-8s" % "事件數" + "".join("%10d" % ntot[k] for k in AGES))
    print("   %-8s" % "$/天" + "".join("%+10.3f" % tot[k] for k in AGES))
    print("   %-8s" % "$/年" + "".join("%+10.1f" % (tot[k] * 365) for k in AGES))
    res["total_usd_day"] = {str(k): tot[k] for k in AGES}
    res["total_n"] = {str(k): ntot[k] for k in AGES}

    # ---------- C3 已知答案：k=0 就是 §1.20 的構造,必須重現它的 $/年 ----------
    print("")
    print("=== C3 k=0 必須重現 §1.20 的 $/年（它就是同一個構造）===")
    print("%-12s %12s %12s %8s" % ("配對:側", "本支 k=0", "§1.20", "比值"))
    c3_bad = 0
    for key, v in per_pair.items():
        pid = key.split(":")[0]
        ref = ((cap.get("pairs") or {}).get(pid) or {}).get("usd_per_year_best")
        mine = v["arms"]["0"]["usd_day"] * 365
        if not ref:
            print("%-12s %12.1f %12s %8s" % (key, mine, "—", "—"))
            continue
        rt = mine / ref
        ok = 0.5 <= rt <= 2.0
        c3_bad += 0 if ok else 1
        print("%-12s %12.1f %12.1f %8.2f%s"
              % (key, mine, ref, rt, "" if ok else "  **差距 >2x**"))
    res["C3_bad"] = c3_bad
    print("  -> %s（容差 0.5x–2x：兩邊的事件計數來源不同，不求相等求同量級）"
          % ("PASS" if c3_bad == 0 else "**%d 個超出容差**" % c3_bad))

    best = max(AGES, key=lambda k: tot[k])
    print("\n  最好的臂：**k=%d**（$%.1f/年 vs k=0 的 $%.1f/年）"
          % (best, tot[best] * 365, tot[0] * 365))
    print("  事件數 k=%d 是 k=0 的 %.1f%%" % (best, 100 * ntot[best] / max(ntot[0], 1)))

    # ---------- C2 ----------
    c2 = []
    for key, v in per_pair.items():
        a0 = v["arms"]["0"]
        if np.isfinite(a0["capt"] or np.nan):
            c2.append(a0["capt"] / (v["band"] / 2.0))
    print("\n=== C2 k=0 的可捕獲 / (band/2) 必須 ≈ 1 ===")
    if c2:
        print("  中位 %.3f（逐側 %d 個）-> %s"
              % (np.median(c2), len(c2),
                 "PASS" if 0.7 <= np.median(c2) <= 1.4 else "**偏離，定義不一致**"))
        res["C2_capt_over_halfband"] = float(np.median(c2))

    if a.se_only:
        print("")
        print("=== SE（判決會用的那台機器：日聚類 bootstrap，重抽「天」）===")
        D = pd.concat(detail, ignore_index=True)
        days_u = sorted(D.day.unique())
        by_day = {d: g for d, g in D.groupby("day")}
        nday = len(days_u)

        def usd_tot(g, k):
            """一組列在臂 k 上的總美元（除以天數才是 $/天）。"""
            d = g[g["alive%d" % k] == 1]
            if not len(d):
                return 0.0
            capt = (d["dev%d" % k] - d["_band"] * 0.5).clip(lower=0)
            return float((d["_size"] * (capt - d["_cost"]) / 1e4).sum())

        base = np.array([usd_tot(D, k) for k in AGES]) / nday
        boot = np.empty((BOOT, len(AGES)))
        for b_ in range(BOOT):
            pick = rng.integers(0, nday, nday)
            g = pd.concat([by_day[days_u[x]] for x in pick], ignore_index=True)
            boot[b_] = [usd_tot(g, k) for k in AGES]
        boot /= nday
        print("  %d 天、%d 段；重抽 %d 次" % (nday, len(D), BOOT))
        print("  %-8s %10s %10s %10s %10s %10s"
              % ("k", "$/年", "SE", "CI下", "CI上", "P(>k=0)"))
        se = {}
        for m, k in enumerate(AGES):
            diff = boot[:, m] - boot[:, 0]
            se[str(k)] = dict(
                usd_year=float(base[m] * 365),
                se_usd_year=float(boot[:, m].std(ddof=1) * 365),
                ci_lo=float(np.percentile(boot[:, m], 2.5) * 365),
                ci_hi=float(np.percentile(boot[:, m], 97.5) * 365),
                se_diff_usd_year=float(diff.std(ddof=1) * 365),
                p_beats_k0=float((diff > 0).mean()))
            v = se[str(k)]
            print("  %-8s %10.1f %10.1f %10.1f %10.1f %9.1f%%"
                  % ("k=%d" % k, v["usd_year"], v["se_usd_year"],
                     v["ci_lo"], v["ci_hi"], 100 * v["p_beats_k0"]))
        res["SE"] = se
        bk = max(AGES, key=lambda k: se[str(k)]["usd_year"])
        sd_ = se[str(bk)]["se_diff_usd_year"]
        gain = se[str(bk)]["usd_year"] - se["0"]["usd_year"]
        print("")
        print("  最好的臂 k=%d：比 k=0 多 $%.0f/年，而**那個差的 SE 是 $%.0f/年**"
              % (bk, gain, sd_))
        print("  -> %s"
              % ("**SE >= 效應 —— 這個設計分辨不出來**（mistake.md 2026-09-04）"
                 if sd_ >= abs(gain) else
                 "效應大於 SE，有測量能力；門檻可以寫在 CI 下緣上"))
        res["power_ok"] = bool(sd_ < abs(gain))

        big = D.groupby("_pair").apply(lambda g: usd_tot(g, 0)).sort_values(ascending=False)
        print("")
        print("  逐配對 k=0 的總美元（看集中度）：")
        for kk, vv in big.items():
            print("    %-12s %10.2f" % (kk, vv))
        drop = big.index[0]
        D2 = D[D["_pair"] != drop]
        b0 = usd_tot(D2, 0) / nday * 365
        bb = usd_tot(D2, bk) / nday * 365
        print("  **去掉最大的 %s 之後**：k=0 $%.1f/年 -> k=%d $%.1f/年（%s）"
              % (drop, b0, bk, bb, "仍然變好" if bb > b0 else "**反過來了**"))
        res["ex_largest"] = dict(dropped=drop, k0=b0, kbest=bb, k=bk)

    print("\n=== 讀法 ===")
    print("  **這支不下判決**（門檻還沒凍結）。它要先讓我們看見**代價**：")
    print("  每筆變好是依定義的（條件在活得久上會選到偏離大的），")
    print("  所以唯一有資訊的是 **$/年** 與 **事件數掉多少**。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print("written -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
