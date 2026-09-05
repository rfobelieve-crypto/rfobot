# -*- coding: utf-8 -*-
"""路徑 B Step 1：跨所 funding carry 的被動基準（PREREG_B_funding.md，同批 commit）

**這支只算被動基準**（永遠持有、每 8 小時按 diff 符號重新對齊），不做任何條件
策略。它存在的唯一理由是規格 B.9 的停止條件：**maker 層年化 < 5% 就整條線停止**
——連天花板都不夠高，條件版本不必做。

含 basis：`total = funding − basis 損益 − 換邊成本`。只報 funding 是這類策略
最常見的自欺（規格 B.3.3 原話），所以兩個都印，但判準只看 total。

資料：各所公開 REST 的 funding 結算歷史 + 8 小時 K 線收盤（算 basis）。
8h 桶邊界 UTC 00/08/16；桶內的實際結算值**加總**成 8h 等效率，不插值。

Run: python research/exit_paths/funding_baseline.py [--days 365]
Out: research/results/funding_baseline.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "funding_baseline.json"
CACHE = ROOT / "research" / "results" / "funding_raw_cache.json"

UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "AVAX", "SUI",
            "LTC", "TRX", "DOT", "NEAR", "APT", "UNI", "AAVE", "ARB", "OP", "PEPE"]
VENUES = ["binance", "bybit", "okx", "bitget"]
COSTS = {"maker": 4.0, "taker": 12.0, "zero": 0.0}      # 來回 bps
BUCKET_MS = 8 * 3600 * 1000
S = requests.Session()


def get(url, **kw):
    for i in range(3):
        try:
            r = S.get(url, timeout=25, **kw)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1 + i)
    return {}


# ── funding 結算歷史（回傳 [(ts_ms, rate)]，rate 是該次結算的實際費率） ──────
def f_binance(sym, start, end):
    out, t = [], start
    while t < end:
        d = get("https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": f"{sym}USDT", "startTime": t, "limit": 1000})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        t = int(d[-1]["fundingTime"]) + 1
        if len(d) < 1000:
            break
    return out


def f_bybit(sym, start, end):
    out, e = [], end
    while e > start:
        d = get("https://api.bybit.com/v5/market/funding/history",
                params={"category": "linear", "symbol": f"{sym}USDT", "endTime": e, "limit": 200})
        lst = ((d.get("result") or {}).get("list")) or []
        if not lst:
            break
        out += [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
        e = min(int(x["fundingRateTimestamp"]) for x in lst) - 1
        if len(lst) < 200:
            break
    return out


def f_okx(sym, start, end):
    out, e = [], end
    while e > start:
        d = get("https://www.okx.com/api/v5/public/funding-rate-history",
                params={"instId": f"{sym}-USDT-SWAP", "before": "", "after": e, "limit": 100})
        lst = d.get("data") or []
        if not lst:
            break
        out += [(int(x["fundingTime"]), float(x["realizedRate"] or x["fundingRate"])) for x in lst]
        e = min(int(x["fundingTime"]) for x in lst) - 1
        if len(lst) < 100:
            break
    return out


def f_bitget(sym, start, end):
    out = []
    for page in range(1, 12):
        d = get("https://api.bitget.com/api/v2/mix/market/history-fund-rate",
                params={"symbol": f"{sym}USDT", "productType": "USDT-FUTURES",
                        "pageSize": 100, "pageNo": page})
        lst = d.get("data") or []
        if not lst:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in lst]
        if len(lst) < 100:
            break
    return [(t, r) for t, r in out if t >= start]


# ── 8 小時收盤（算 basis） ──────────────────────────────────────────────
def k_binance(sym, start):
    out, t = [], start
    while True:
        d = get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": f"{sym}USDT", "interval": "8h", "startTime": t, "limit": 1000})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x[0]), float(x[4])) for x in d]
        t = int(d[-1][0]) + 1
        if len(d) < 1000:
            break
    return out


def k_bybit(sym, start):
    out, e = [], int(time.time() * 1000)
    while e > start:
        d = get("https://api.bybit.com/v5/market/kline",
                params={"category": "linear", "symbol": f"{sym}USDT", "interval": "480",
                        "end": e, "limit": 1000})
        lst = ((d.get("result") or {}).get("list")) or []
        if not lst:
            break
        out += [(int(x[0]), float(x[4])) for x in lst]
        e = min(int(x[0]) for x in lst) - 1
        if len(lst) < 1000:
            break
    return out


def k_okx(sym, start):
    out, e = [], int(time.time() * 1000)
    while e > start:
        d = get("https://www.okx.com/api/v5/market/history-candles",
                params={"instId": f"{sym}-USDT-SWAP", "bar": "8H", "after": e, "limit": 100})
        lst = d.get("data") or []
        if not lst:
            break
        out += [(int(x[0]), float(x[4])) for x in lst]
        e = min(int(x[0]) for x in lst) - 1
        if len(lst) < 100:
            break
    return out


def k_bitget(sym, start):
    out, e = [], int(time.time() * 1000)
    while e > start:
        d = get("https://api.bitget.com/api/v2/mix/market/history-candles",
                params={"symbol": f"{sym}USDT", "productType": "USDT-FUTURES", "granularity": "8H",
                        "endTime": e, "limit": 200})
        lst = d.get("data") or []
        if not lst:
            break
        out += [(int(x[0]), float(x[4])) for x in lst]
        e = min(int(x[0]) for x in lst) - 1
        if len(lst) < 200:
            break
    return out


FUND = {"binance": f_binance, "bybit": f_bybit, "okx": f_okx, "bitget": f_bitget}
KLINE = {"binance": k_binance, "bybit": k_bybit, "okx": k_okx, "bitget": k_bitget}


def bucketise(pairs, start, end, how="sum"):
    """把 (ts, value) 塞進 8 小時桶。funding 用加總（8h 等效率），價格用桶內最後一筆。"""
    g = {}
    for t, v in pairs:
        if t < start or t > end:
            continue
        b = (t // BUCKET_MS) * BUCKET_MS
        if how == "sum":
            g[b] = g.get(b, 0.0) + v
        else:
            if b not in g or t >= g[b][0]:
                g[b] = (t, v)
    return g if how == "sum" else {k: v for k, (_, v) in g.items()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    end = int(time.time() * 1000); start = end - a.days * 86400 * 1000

    raw = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for v in VENUES:
        for s in UNIVERSE:
            for kind, fn in (("f", FUND[v]), ("k", KLINE[v])):
                key = f"{v}:{s}:{kind}:{a.days}"
                if key in raw:
                    continue
                try:
                    raw[key] = fn(s, start, end) if kind == "f" else fn(s, start)
                except Exception as e:  # noqa: BLE001
                    print(f"  [WARN] {key}: {str(e)[:70]}", flush=True); raw[key] = []
                print(f"  {key}: {len(raw[key])}", flush=True)
    CACHE.write_text(json.dumps(raw), encoding="utf-8")

    F = {(v, s): bucketise(raw.get(f"{v}:{s}:f:{a.days}") or [], start, end, "sum")
         for v in VENUES for s in UNIVERSE}
    P = {(v, s): bucketise(raw.get(f"{v}:{s}:k:{a.days}") or [], start, end, "last")
         for v in VENUES for s in UNIVERSE}

    print("=" * 108)
    print(f"  路徑 B Step 1：被動基準（{a.days} 天，8h 桶，含 basis）｜停止條件：maker 層年化 < 5% 整條停")
    print("=" * 108)
    res = {"days": a.days, "pairs": {}}
    print(f"  {'場館對':<18}{'幣數':>5}{'桶數':>7}{'|diff|中位':>11}{'換邊/年':>8}"
          f"{'funding年化':>12}{'basis年化':>11}{'total(maker)':>13}{'MDD':>8}{'Sharpe':>8}")
    for A, B in itertools.combinations(VENUES, 2):
        per_coin, allpnl = {}, []
        for s in UNIVERSE:
            fa, fb, pa, pb = F[(A, s)], F[(B, s)], P[(A, s)], P[(B, s)]
            buckets = sorted(set(fa) & set(fb) & set(pa) & set(pb))
            if len(buckets) < 100:
                continue
            rows, prev_d = [], 0
            for i in range(1, len(buckets)):
                b0, b1 = buckets[i - 1], buckets[i]
                if b1 - b0 != BUCKET_MS:
                    prev_d = 0; continue
                diff = fa[b1] - fb[b1]
                d = 1 if diff > 0 else (-1 if diff < 0 else 0)
                ra = pa[b1] / pa[b0] - 1; rb = pb[b1] / pb[b0] - 1
                y = abs(diff) * 1e4                        # bps
                bas = -d * (ra - rb) * 1e4                 # bps
                flip = 1 if d != prev_d else 0
                prev_d = d
                rows.append((y, bas, flip))
            if len(rows) < 100:
                continue
            arr = np.array(rows, float)
            per_coin[s] = arr
            allpnl.append(arr)
        if not allpnl:
            continue
        A_ = np.vstack(allpnl)
        y, bas, flips = A_[:, 0], A_[:, 1], A_[:, 2]
        n_coins, n_b = len(per_coin), len(A_) // max(n_coins, 1)
        out = {"coins": n_coins, "buckets": int(len(A_))}
        for lbl, c in COSTS.items():
            pnl = (y + bas - flips * c) / 1e4
            eq = np.cumsum(pnl)
            ann = float(pnl.mean() * 1095)
            mdd = float((eq - np.maximum.accumulate(eq)).min())
            shp = float(pnl.mean() / pnl.std() * np.sqrt(1095)) if pnl.std() > 0 else 0.0
            out[lbl] = {"ann": ann, "mdd": mdd, "sharpe": shp}
        out.update(fund_ann=float(y.mean() / 1e4 * 1095), basis_ann=float(bas.mean() / 1e4 * 1095),
                   diff_med=float(np.median(y)), flips_yr=float(flips.mean() * 1095))
        res["pairs"][f"{A}-{B}"] = out
        m = out["maker"]
        print(f"  {A+'-'+B:<18}{n_coins:>5}{len(A_):>7}{out['diff_med']:>11.2f}{out['flips_yr']:>8.0f}"
              f"{out['fund_ann']:>+12.2%}{out['basis_ann']:>+11.2%}{m['ann']:>+13.2%}{m['mdd']:>+8.2%}{m['sharpe']:>8.2f}")

    best = max(res["pairs"].items(), key=lambda kv: kv[1]["maker"]["ann"]) if res["pairs"] else (None, None)
    gate = bool(best[1] and best[1]["maker"]["ann"] >= 0.05)
    res["best_pair"] = best[0]
    res["gate_step1_maker_ann_ge_5pct"] = gate
    res["verdict"] = ("Step 1 過門檻——可以做 G1" if gate else
                      "Step 1 未過（maker 層年化 < 5%）——依 PREREG B.9 整條線停止")
    print(f"\n  最佳對：{best[0]}  maker 年化 {best[1]['maker']['ann']:+.2%}" if best[0] else "\n  無有效場館對")
    print(f"  {'✅' if gate else '❌'} 停止條件：maker 層年化 ≥ 5%")
    print(f"  → {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
