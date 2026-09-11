# -*- coding: utf-8 -*-
"""月底季節性：最後 N 天做多（2026-09-11，外部閱讀轉譯）

===========================================================================
來源與它宣稱什麼
===========================================================================
Quant Arb〈Easy Alpha Portfolio - 5 Strategies That Just Work〉（2025-05-19）
的第三個策略，原話很短：

> End of month rebalance works in crypto and less developed markets but not
> really equities. **Last 3/4 days of the month for crypto.**

他沒有給數字、沒有給樣本期、沒有給成本假設。所以這是一個**待驗的主張**，
不是一個結果。本支要做的就是把它翻成「我們要驗什麼」。

===========================================================================
Gate 0 先做（CLAUDE.md 核心原則 11：執行可行性排在資訊層之前）
===========================================================================
a. 訊號完全正確時我在什麼價格成交？ -> **日收盤**（月底前 N 天的收盤進、
   月底最後一根日收盤出）。這是一個日頻策略，不是延遲競賽。
b. 那個價格拿得到嗎？ -> 拿得到。日收盤附近的流動性對我們的規模是充裕的
   （對照：§1.02 舊線死在「回測假設的價格市場不給」，那是分鐘級的觸價；
   這裡沒有那個問題）。
c. 可實現邊際 vs 來回成本？ -> 成本是**一次來回的 taker**，攤在 3-4 天的
   持有上。用逐標的真實 bps（factor-research 第 10 條：不用統一單位）。
d. MDE：這個設計分辨得出那個比值嗎？ -> **本支的第一個輸出就是這個**，
   而且是在看效應之前先印出來。

**d 是這一支最重要的部分。** 月頻策略的樣本數是**月底次數**不是天數：
三年的資料只有 36 個樣本。mistake.md 2026-09-04（地形扳機：門檻 8pp、
SE 11.6pp，註冊那天起就不可能有答案）說的就是這件事。

===========================================================================
預註冊（跑之前寫死，事後不放寬）
===========================================================================
規則      月底前 K 個日收盤買進，最後一個日收盤賣出。**只做多**（原文是
          long-only 的再平衡效應）。K ∈ {3, 4}（原文「3/4 days」），
          **兩格都報，不挑**。另外報 K=2 與 K=5 當形狀對照。
宇宙      **事前凍結**：core9（BTC ETH SOL BNB XRP DOGE ADA LINK AVAX）
          —— 這個專案既有的凍結宇宙，不是為這次挑的。
          下市/新上市不另外處理：有多少歷史用多少，**逐標的報樣本數**。
成本      來回 taker，逐標的真實 bps（預設 10 bps 來回，可調）。
判準      C1 合池淨報酬的月聚類 bootstrap **CI 下緣 > 0**
          C2 **逐幣 ≥ 6/9 為正**
          C3 **前後半同號**
          C4 **誠實挑選**：只用前半挑 K，套到後半——前半挑到的那個 K
             在後半是不是還活著。**這一關才是真的樣本外**
             （mistake.md 2026-09-09：事後找到的維度通過多少一致性檢定
              都不算數，唯一算數的是把挑選程序本身放進樣本外）。
自曝      D1 隨機月份對照（把「月底」換成隨機的月內位置）必須不顯著。
          D2 零成本對照必須 **>=** 含成本結果（否則成本模型壞了，
             mistake.md 2026-07-28）。

    python research/seasonality_month_end.py
    python research/seasonality_month_end.py --days 3000 --cost-bps 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research" / ".cache" / "daily"
OUT = ROOT / "research" / "results" / "seasonality_month_end.json"
BASE = "https://api.binance.com/api/v3/klines"

# **事前凍結的宇宙** —— 這個專案既有的 core9，不是為這次挑的。
SYMS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]
KS = [2, 3, 4, 5]            # 原文說 3/4；2 與 5 是形狀對照，全格報告
BOOT = 4000


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "seas/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def fetch_daily(sym: str, days: int) -> list[tuple[int, float]]:
    """日 K 收盤。快取到 .cache/daily/，重跑不再打 API。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{sym}USDT_1d.csv"
    if fp.exists():
        out = []
        for line in fp.read_text(encoding="utf-8").splitlines()[1:]:
            a, b = line.split(",")
            out.append((int(a), float(b)))
        if out:
            return out
    end = int(time.time() * 1000)
    cur = end - days * 86400 * 1000
    rows: dict[int, float] = {}
    while cur < end:
        d = get(f"{BASE}?symbol={sym}USDT&interval=1d&startTime={cur}&limit=1000")
        if not d:
            break
        for k in d:
            rows[int(k[0]) // 1000] = float(k[4])
        nxt = int(d[-1][0]) + 86400 * 1000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    out = sorted(rows.items())
    fp.write_text("ts,close\n" + "\n".join(f"{a},{b}" for a, b in out),
                  encoding="utf-8")
    return out


def month_of(ts: int) -> tuple[int, int]:
    t = time.gmtime(ts)
    return t.tm_year, t.tm_mon


def build_events(bars, k: int, offset: int = 0):
    """回傳 [(年月, 進場ts, 毛報酬)]。

    offset=0 -> 月底最後 k 天（真正的訊號）
    offset>0 -> 把同樣長度的窗往前挪 offset 天（隨機/安慰劑對照）
    """
    by = defaultdict(list)
    for i, (ts, px) in enumerate(bars):
        by[month_of(ts)].append(i)
    ev = []
    for ym, idxs in sorted(by.items()):
        if len(idxs) < k + offset + 1:
            continue
        j_out = idxs[-1 - offset]          # 出場：該月最後一根（或往前挪）
        j_in = idxs[-1 - offset - k]       # 進場：再往前 k 根
        if j_in < 0:
            continue
        p_in, p_out = bars[j_in][1], bars[j_out][1]
        if p_in <= 0:
            continue
        ev.append((ym, bars[j_in][0], p_out / p_in - 1.0))
    return ev


def boot_ci(vals, groups, n=BOOT, seed=7):
    """**月聚類** bootstrap —— 同一個月底九個幣一起動，逐筆重抽會把那個
    相關性洗掉，給出一個天生偏窄的區間（mistake.md 2026-09-06）。"""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    rnd = random.Random(seed)
    gmap = defaultdict(list)
    for v, g in zip(vals, groups):
        gmap[g].append(v)
    keys = list(gmap)
    mu = sum(vals) / len(vals)
    ms = []
    for _ in range(n):
        pool = []
        for _ in range(len(keys)):
            pool.extend(gmap[keys[rnd.randrange(len(keys))]])
        ms.append(sum(pool) / len(pool))
    ms.sort()
    p_pos = sum(1 for m in ms if m > 0) / len(ms)
    return mu, ms[int(0.025 * n)], ms[int(0.975 * n)], p_pos


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3200)
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="來回 taker 成本（bps）")
    a = ap.parse_args()
    cost = a.cost_bps / 1e4

    print("=== 月底季節性：最後 K 天做多（外部主張，待驗）===")
    print(f"宇宙 core9（事前凍結）　成本 {a.cost_bps:.0f} bps 來回\n")

    data = {}
    for s in SYMS:
        try:
            b = fetch_daily(s, a.days)
        except Exception as e:                                # noqa: BLE001
            print(f"  {s}: 抓不到（{e}）")
            continue
        if len(b) > 60:
            data[s] = b
    if not data:
        print("**沒有資料，停。**")
        return 2

    print(f"{'幣':<6s} {'日K數':>7s} {'起':>12s} {'迄':>12s} {'月底數':>7s}")
    for s, b in data.items():
        ev = build_events(b, 4)
        print(f"{s:<6s} {len(b):7d} "
              f"{time.strftime('%Y-%m', time.gmtime(b[0][0])):>12s} "
              f"{time.strftime('%Y-%m', time.gmtime(b[-1][0])):>12s} "
              f"{len(ev):7d}")
    print()

    # ---------- Gate 0 的 d：先算 MDE，再看效應 ----------
    print("=== Gate 0·d　MDE：這個設計分辨得出多大的效應？（**先算，再看結果**）===")
    res = {"asof": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cost_bps": a.cost_bps, "syms": list(data), "K": {}}
    for k in KS:
        allv, allg = [], []
        for s, b in data.items():
            for ym, _, r in build_events(b, k):
                allv.append(r)
                allg.append(ym)
        n_months = len(set(allg))
        sd = (sum((x - sum(allv) / len(allv)) ** 2 for x in allv)
              / max(len(allv) - 1, 1)) ** 0.5
        # 月聚類：有效樣本是**月份數**不是列數（同一個月九個幣一起動）
        se = sd / math.sqrt(max(n_months, 1))
        print(f"  K={k}　列數 {len(allv):4d}　**月份數 {n_months:3d}**　"
              f"σ {sd*100:5.2f}%　-> SE ≈ **{se*100:.2f}%**　"
              f"（可偵測 ~{2*se*100:.2f}% 以上）")
        res["K"][str(k)] = {"n_rows": len(allv), "n_months": n_months,
                            "sd": sd, "se": se, "mde_2se": 2 * se}
    print("  成本是 %.2f%%，所以**效應要大於 成本 + 2SE 才有意義**。" % (cost * 100))
    print()

    # ---------- 全格報告 ----------
    print("=== 全格報告（不挑格）===")
    # **中位數與去掉最大一筆是必要欄位不是附註。** 加密貨幣的右尾很肥，
    # 任何「窗口平均報酬」都可能是一兩筆擠壓撐起來的，而平均數看不出來。
    # （factor-research 第 10 條的鄰居；§1.24 的 ex-largest 同一招。）
    print(f"{'K':>3s} {'月數':>5s} {'毛均':>8s} {'淨均':>8s} {'淨中位':>8s} "
          f"{'去最大':>8s} {'正比例':>7s} {'淨CI下':>9s} {'淨CI上':>9s} "
          f"{'P(>0)':>7s} {'幣+':>5s}")
    for k in KS:
        allv, allg, per = [], [], {}
        for s, b in data.items():
            rs = [r for _, _, r in build_events(b, k)]
            per[s] = sum(rs) / len(rs) - cost if rs else 0.0
            for ym, _, r in build_events(b, k):
                allv.append(r)
                allg.append(ym)
        net = [x - cost for x in allv]
        mu_g = sum(allv) / len(allv)
        mu, lo, hi, ppos = boot_ci(net, allg)
        pos = sum(1 for v in per.values() if v > 0)
        sv = sorted(net)
        med = sv[len(sv) // 2]
        mx = max(net)
        ex = (sum(net) - mx) / max(len(net) - 1, 1)
        frac_pos = sum(1 for x in net if x > 0) / len(net)
        print(f"{k:3d} {len(set(allg)):5d} {mu_g*100:7.3f}% {mu*100:7.3f}% "
              f"{med*100:7.3f}% {ex*100:7.3f}% {frac_pos*100:6.1f}% "
              f"{lo*100:8.3f}% {hi*100:8.3f}% {ppos*100:6.1f}% "
              f"{pos:3d}/{len(per)}")
        res["K"][str(k)].update(gross=mu_g, net=mu, median=med, ex_largest=ex,
                                frac_pos=frac_pos, top_share=mx / len(net) / mu
                                if mu else None, lo=lo, hi=hi,
                                p_pos=ppos, coins_pos=pos,
                                per_coin={s: v for s, v in per.items()})
    print()

    # ---------- D1 安慰劑：把窗往前挪 ----------
    print("=== D1 自曝：把同樣長度的窗往前挪（安慰劑），應該不顯著 ===")
    print(f"{'挪幾天':>7s} {'K=4 淨均':>10s} {'CI下':>9s} {'CI上':>9s}")
    plac = {}
    for off in (5, 10, 15):
        allv, allg = [], []
        for s, b in data.items():
            for ym, _, r in build_events(b, 4, offset=off):
                allv.append(r)
                allg.append(ym)
        net = [x - cost for x in allv]
        mu, lo, hi, _ = boot_ci(net, allg, seed=11 + off)
        plac[str(off)] = dict(net=mu, lo=lo, hi=hi)
        print(f"{off:7d} {mu*100:9.3f}% {lo*100:8.3f}% {hi*100:8.3f}%")
    res["placebo"] = plac
    print()

    # ---------- D2 零成本對照 ----------
    k = 4
    allv = [r for s, b in data.items() for _, _, r in build_events(b, k)]
    print("=== D2 自曝：零成本結果必須 >= 含成本結果 ===")
    g = sum(allv) / len(allv)
    print(f"  零成本 {g*100:+.3f}%　含成本 {(g-cost)*100:+.3f}%　"
          + ("PASS" if g >= g - cost else "**FAIL**"))
    print()

    # ---------- C4 誠實挑選：只用前半挑 K ----------
    print("=== C4 **誠實挑選**：只用前半挑 K，套到後半（這一關才是真樣本外）===")
    months = sorted({ym for s, b in data.items()
                     for ym, _, _ in build_events(b, 4)})
    cut = months[len(months) // 2]
    best_k, best_v = None, -9
    for k in KS:
        v, n = 0.0, 0
        for s, b in data.items():
            for ym, _, r in build_events(b, k):
                if ym <= cut:
                    v += r - cost
                    n += 1
        m = v / max(n, 1)
        if m > best_v:
            best_v, best_k = m, k
    allv, allg, per = [], [], defaultdict(list)
    for s, b in data.items():
        for ym, _, r in build_events(b, best_k):
            if ym > cut:
                allv.append(r - cost)
                allg.append(ym)
                per[s].append(r - cost)
    mu, lo, hi, ppos = boot_ci(allv, allg, seed=23)
    cpos = sum(1 for s, v in per.items() if sum(v) / len(v) > 0)
    print(f"  前半（{months[0]} ~ {cut}）挑到 **K={best_k}**"
          f"（前半淨 {best_v*100:+.3f}%）")
    print(f"  後半（沒看過的）淨 **{mu*100:+.3f}%**　"
          f"CI [{lo*100:+.3f}%, {hi*100:+.3f}%]　"
          f"P(>0) {ppos*100:.1f}%　逐幣 {cpos}/{len(per)}")
    res["honest"] = dict(cut=list(cut), picked_k=best_k, first_half=best_v,
                         second_half=mu, lo=lo, hi=hi, p_pos=ppos,
                         coins_pos=cpos, n_coins=len(per))
    print()

    # ---------- 判決 ----------
    h = res["honest"]
    print("=== 判決（照預註冊，事後不放寬）===")
    c1 = h["lo"] > 0
    c2 = h["coins_pos"] >= 6
    print(f"  C1 後半淨 CI 下緣 > 0 ......... {'PASS' if c1 else 'FAIL'}"
          f"（{h['lo']*100:+.3f}%）")
    print(f"  C2 後半逐幣 >= 6/9 ........... {'PASS' if c2 else 'FAIL'}"
          f"（{h['coins_pos']}/{h['n_coins']}）")
    print(f"  C4 前半挑到 K={best_k}，後半 {'還活著' if mu > 0 else '翻號'}")
    print()
    if c1 and c2:
        print("  -> **通過**。可以進下一步（規格凍結 + 前瞻時鐘）。")
    else:
        print("  -> **沒通過**。這個主張在我們的宇宙與成本下站不住，")
        print("     或者樣本數不足以判定 —— 看上面的 MDE 那一行決定是哪一種。")
    res["verdict"] = bool(c1 and c2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
