# -*- coding: utf-8 -*-
"""路徑 X：擁擠度橫截面多空（PREREG_X_crosssec.md，同批 commit）

做空 funding z-score 最高的 3 個幣、做多最低的 3 個幣，等金額美元中性，週再平衡。
資料是 2026-09-05 為了路徑 B 拉的 Binance 永續 funding + 8h K 線（19 幣、365 天）。

判準全部在 PREREG 裡，這支只算不改。關鍵是 C6：把排序換成隨機重複 500 次，
真實策略要落在隨機分布的 p95 以上——沒有這一關，任何正數都可能只是
「等額多空 19 個幣」這個結構本身的產物。

Run: python research/exit_paths/crosssec_crowding.py
Out: research/results/crosssec_crowding.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "research" / "results" / "funding_raw_cache.json"
OUT = ROOT / "research" / "results" / "crosssec_crowding.json"

BUCKET_MS = 8 * 3600 * 1000
Z_WIN, HOLD, K = 63, 21, 3          # 21 天 z 窗、7 天持有、每側 3 名
COSTS = {"maker": 4.0, "taker": 8.0, "zero": 0.0}
VENUE, DAYS = "binance", 365


def load():
    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    F, P, syms = {}, {}, []
    for key, val in raw.items():
        v, s, kind, d = key.split(":")
        if v != VENUE or d != str(DAYS) or not val:
            continue
        g = {}
        for t, x in val:
            b = (t // BUCKET_MS) * BUCKET_MS
            if kind == "f":
                g[b] = g.get(b, 0.0) + x
            else:
                g[b] = x                      # 8h 棒，一桶一根
        (F if kind == "f" else P)[s] = g
    syms = sorted(set(F) & set(P))
    buckets = sorted(set.intersection(*[set(F[s]) & set(P[s]) for s in syms]))
    return syms, buckets, F, P


def rank_ic(x, y):
    if len(x) < 4:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def run(syms, buckets, F, P, order_fn, cost_bps):
    """order_fn(crowd_vector) -> 排序索引（小到大）。回傳每週的 total 與診斷。"""
    weeks, ics, terc, prev = [], [], {0: [], 1: [], 2: []}, set()
    per_sym = {s: 0.0 for s in syms}
    for i in range(Z_WIN, len(buckets) - HOLD, HOLD):
        t = buckets[i]
        crowd, fut, carry, ok = [], [], [], []
        for s in syms:
            hist = [F[s][buckets[j]] for j in range(i - Z_WIN, i + 1)]
            sd = np.std(hist)
            if sd == 0:
                continue
            crowd.append((hist[-1] - np.mean(hist)) / sd)
            fut.append(P[s][buckets[i + HOLD]] / P[s][t] - 1)
            carry.append(sum(F[s][buckets[j]] for j in range(i + 1, i + HOLD + 1)))
            ok.append(s)
        if len(ok) < 3 * K:
            continue
        crowd, fut, carry = np.array(crowd), np.array(fut), np.array(carry)
        idx = order_fn(crowd)
        lo, hi = idx[:K], idx[-K:]                    # 低擁擠做多、高擁擠做空
        price = fut[lo].mean() - fut[hi].mean()
        cry = carry[hi].mean() - carry[lo].mean()     # 做空高 funding 幣＝收租
        names = {ok[j] for j in np.concatenate([lo, hi])}
        turn = len(names - prev) + len(prev - names)
        prev = names
        cost = turn * cost_bps / 1e4
        tot = price + cry - cost
        weeks.append((tot, price, cry, cost))
        ics.append(rank_ic(crowd, fut))
        for j, s in enumerate(ok):                    # C5：逐幣貢獻
            if j in lo:
                per_sym[s] += fut[j] / K
            elif j in hi:
                per_sym[s] += -fut[j] / K
        q = np.argsort(np.argsort(crowd)) * 3 // len(crowd)
        for b in (0, 1, 2):
            terc[b] += list(fut[q == b])
    return np.array(weeks), np.array(ics, float), terc, per_sym


def boot(v, B=4000, seed=7):
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    syms, buckets, F, P = load()
    print("=" * 100)
    print(f"  路徑 X 擁擠度橫截面多空（預註冊）｜{len(syms)} 幣 × {len(buckets)} 桶"
          f"｜每側 {K} 名、{HOLD//3} 天再平衡、z 窗 {Z_WIN//3} 天")
    print("=" * 100)
    res = {"syms": syms, "buckets": len(buckets)}
    asc = lambda c: np.argsort(c)
    for lbl, c in COSTS.items():
        w, ics, terc, per_sym = run(syms, buckets, F, P, asc, c)
        tot = w[:, 0]
        lo, hi = boot(tot)
        res[lbl] = {"weeks": len(tot), "mean_pct": float(tot.mean() * 100),
                    "ci_pct": [lo * 100, hi * 100], "ann_pct": float(tot.mean() * 52 * 100),
                    "price_pct": float(w[:, 1].mean() * 100), "carry_pct": float(w[:, 2].mean() * 100),
                    "cost_pct": float(w[:, 3].mean() * 100), "win": float((tot > 0).mean())}
        r = res[lbl]
        print(f"  [{lbl:<5}] {len(tot)} 週  週均 {r['mean_pct']:+.3f}%  CI [{lo*100:+.3f},{hi*100:+.3f}]"
              f"  年化 {r['ann_pct']:+.1f}%  勝週 {r['win']:.0%}"
              f"   (價格 {r['price_pct']:+.3f} + 租金 {r['carry_pct']:+.3f} − 成本 {r['cost_pct']:.3f})")
    w, ics, terc, per_sym = run(syms, buckets, F, P, asc, COSTS["maker"])
    tot = w[:, 0]; half = len(tot) // 2
    h1, h2 = tot[:half].mean(), tot[half:].mean()
    ic_ok = np.isfinite(ics)
    ic_lo, ic_hi = boot(ics[ic_ok])
    print(f"\n  C3 橫截面 rank IC：平均 {np.nanmean(ics):+.4f}  CI [{ic_lo:+.4f},{ic_hi:+.4f}]  ({ic_ok.sum()} 週)")
    print(f"  C4 三層（低擁擠→高擁擠）下一週報酬："
          + "  ".join(f"{np.mean(terc[b])*100:+.2f}% (n={len(terc[b])})" for b in (0, 1, 2)))
    top = sorted(per_sym.items(), key=lambda kv: -abs(kv[1]))[:3]
    tot_abs = sum(abs(v) for v in per_sym.values()) or 1
    print(f"  C5 貢獻最大三幣：" + "  ".join(f"{s} {v*100:+.1f}%（占 {abs(v)/tot_abs:.0%}）" for s, v in top))
    rng = np.random.default_rng(20260905)
    rand = []
    for _ in range(500):
        w2, _, _, _ = run(syms, buckets, F, P, lambda c: rng.permutation(len(c)), COSTS["maker"])
        rand.append(w2[:, 0].mean())
    rand = np.array(rand); pct = float((rand < tot.mean()).mean())
    print(f"  C6 隨機排序對照（500 次）：週均中位 {np.median(rand)*100:+.3f}%  "
          f"p95 {np.percentile(rand,95)*100:+.3f}%  → 真實策略落在第 {pct*100:.0f} 百分位")
    c1 = res["maker"]["ci_pct"][0] > 0
    c2 = h1 > 0 and h2 > 0
    c3 = (ic_lo > 0) or (ic_hi < 0)
    c5 = abs(top[0][1]) / tot_abs <= 0.40
    c6 = pct >= 0.95
    verdict = "存活（進下一階段）" if (c1 and c2 and c3 and c5 and c6) else (
        "有訊號但沒經濟價值" if c3 and not c1 else "NO-GO")
    print(f"\n  C1 經濟 CI 下界>0: {'過' if c1 else '不過'}  C2 兩半皆正: {'過' if c2 else '不過'}"
          f"  C3 IC 離零: {'過' if c3 else '不過'}  C5 非單幣: {'過' if c5 else '不過'}"
          f"  C6 勝過隨機 p95: {'過' if c6 else '不過'}")
    print(f"  兩半 {h1*100:+.3f}% / {h2*100:+.3f}%")
    print(f"  ==> {verdict}")
    se = tot.std() / np.sqrt(len(tot))
    print(f"  功效：週 total SE {se*100:.3f}%，MDE(80%) {2.802*se*100:.3f}%/週 = 年化 {2.802*se*52*100:.0f}%"
          f" —— 只測得出很大的效應；貼近零一律讀成 INCONCLUSIVE")
    res.update({"halves_pct": [h1 * 100, h2 * 100], "ic_mean": float(np.nanmean(ics)),
                "ic_ci": [ic_lo, ic_hi], "tercile_pct": [float(np.mean(terc[b]) * 100) for b in (0, 1, 2)],
                "random_pct": pct, "verdict": verdict,
                "mde_ann_pct": float(2.802 * se * 52 * 100),
                "bars": {"C1": c1, "C2": c2, "C3": c3, "C5": c5, "C6": c6}})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
