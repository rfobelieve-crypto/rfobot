# -*- coding: utf-8 -*-
"""路徑 B2：perp DEX（Hyperliquid）對 CEX 的 funding carry 被動基準
（PREREG_B2_perpdex.md，同批 commit，commit 之後才看標籤）

兩個口徑都跑，這是本檔的防移動門柱設計：
  (a) B 口徑：每 8 小時按 diff 符號重對齊（跟已判死的 B 一字不差）
  (b) B2 口徑：|diff| ≥ θ_in 進場、最短持有 7 天、|diff| < θ_in/2 或滿 21 天出場
**只有 (b) 好看 ⇒ 那是持有期的功勞不是場館的功勞，不得記為新發現。**

主流組與長尾組分開判。長尾組用規則取（HL ∩ Bitget 永續 ∧ ≥180 天歷史），
全取不篩選。容量（HL 頂檔深度）與存續期分佈是獨立的兩關，不是附註。

Run: python research/exit_paths/perpdex_funding.py [--days 365]
Out: research/results/perpdex_funding.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "perpdex_funding.json"
CACHE = ROOT / "research" / "results" / "perpdex_cache.json"

HL = "https://api.hyperliquid.xyz/info"
BUCKET_MS = 8 * 3600 * 1000
MAJORS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC"]
FEE = {"maker": {"hl": 2.5, "cex": 2.0}, "taker": {"hl": 5.0, "cex": 5.5}, "zero": {"hl": 0.0, "cex": 0.0}}
THETA_IN = 1.0            # bps/8h，事前定，不掃
MIN_HOLD, MAX_HOLD = 21, 63       # 7 天 / 21 天（8h 桶）
MIN_HIST_DAYS = 180
S = requests.Session()


def hl_post(body, tries=3):
    for i in range(tries):
        try:
            r = S.post(HL, json=body, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 + i)
    return None


def get(url, **kw):
    for i in range(3):
        try:
            r = S.get(url, timeout=25, **kw)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 + i)
    return {}


def hl_funding(coin, start_ms, end_ms):
    out, t = [], start_ms
    while t < end_ms:
        d = hl_post({"type": "fundingHistory", "coin": coin, "startTime": t})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x["time"]), float(x["fundingRate"])) for x in d]
        nt = int(d[-1]["time"]) + 1
        if nt <= t:
            break
        t = nt
        if len(d) < 500:
            break
    return out


def hl_candles(coin, start_ms, end_ms):
    """4h K 線（HL 沒有 8H），桶內取最後一根 —— 與 CEX 側同一個慣例。"""
    out, t = [], start_ms
    while t < end_ms:
        d = hl_post({"type": "candleSnapshot",
                     "req": {"coin": coin, "interval": "4h", "startTime": t, "endTime": end_ms}})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x["t"]), float(x["c"])) for x in d]
        nt = int(d[-1]["t"]) + 1
        if nt <= t:
            break
        t = nt
        if len(d) < 500:
            break
    return out


def binance_symbols():
    d = get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    return {x["baseAsset"] for x in (d.get("symbols") or [])
            if x.get("quoteAsset") == "USDT" and x.get("status") == "TRADING"}


def binance_funding(sym, start_ms, end_ms):
    out, t = [], start_ms
    while t < end_ms:
        d = get("https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": f"{sym}USDT", "startTime": t, "limit": 1000})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        t = int(d[-1]["fundingTime"]) + 1
        if len(d) < 1000:
            break
    return out


def binance_candles(sym, start_ms):
    out, t = [], start_ms
    while True:
        d = get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": f"{sym}USDT", "interval": "4h", "startTime": t, "limit": 1000})
        if not isinstance(d, list) or not d:
            break
        out += [(int(x[0]), float(x[4])) for x in d]
        t = int(d[-1][0]) + 1
        if len(d) < 1000:
            break
    return out


def bitget_symbols():
    d = get("https://api.bitget.com/api/v2/mix/market/tickers",
            params={"productType": "usdt-futures"})
    return {x["symbol"][:-4] for x in (d.get("data") or []) if x.get("symbol", "").endswith("USDT")}


def bitget_funding(sym, start_ms):
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
    return [(t, r) for t, r in out if t >= start_ms]


def bitget_candles(sym, start_ms):
    out, e = [], int(time.time() * 1000)
    while e > start_ms:
        d = get("https://api.bitget.com/api/v2/mix/market/history-candles",
                params={"symbol": f"{sym}USDT", "productType": "usdt-futures",
                        "granularity": "4H", "endTime": e, "limit": 200})
        lst = d.get("data") or []
        if not lst:
            break
        out += [(int(x[0]), float(x[4])) for x in lst]
        ne = min(int(x[0]) for x in lst) - 1
        if ne >= e:
            break
        e = ne
        if len(lst) < 200:
            break
    return out


def bucketise(pairs, how="sum"):
    g = {}
    for t, v in pairs:
        b = (t // BUCKET_MS) * BUCKET_MS
        if how == "sum":
            g[b] = g.get(b, 0.0) + v
        else:
            if b not in g or t >= g[b][0]:
                g[b] = (t, v)
    return g if how == "sum" else {k: v for k, (_, v) in g.items()}


def series(fh, fc, ph, pc):
    """對齊後回傳 (buckets, diff_bps, retHL, retCEX)。"""
    bs = sorted(set(fh) & set(fc) & set(ph) & set(pc))
    rows = []
    for i in range(1, len(bs)):
        b0, b1 = bs[i - 1], bs[i]
        if b1 - b0 != BUCKET_MS:
            continue
        rows.append((b1, (fh[b1] - fc[b1]) * 1e4,
                     ph[b1] / ph[b0] - 1, pc[b1] / pc[b0] - 1))
    return rows


def run_a(rows, rt_bps):
    """B 口徑：每桶按符號重對齊。"""
    pnl, prev = [], 0
    for _, diff, rh, rc in rows:
        d = 1 if diff > 0 else (-1 if diff < 0 else 0)
        y = abs(diff)
        bas = -d * (rh - rc) * 1e4
        cost = rt_bps if d != prev else 0.0
        prev = d
        pnl.append((y + bas - cost) / 1e4)
    return np.array(pnl)


def run_b(rows, rt_bps):
    """B2 口徑：遲滯 + 最短持有 7 天。回傳 (逐桶損益, 持有天數清單)。"""
    pnl = np.zeros(len(rows)); pos, held, holds = 0, 0, []
    for i, (_, diff, rh, rc) in enumerate(rows):
        if pos == 0:
            if abs(diff) >= THETA_IN:
                pos = 1 if diff > 0 else -1
                held = 0
                pnl[i] -= rt_bps / 1e4          # 進場付一次來回（保守：進場即認列）
        if pos != 0:
            held += 1
            pnl[i] += (pos * diff - pos * (rh - rc) * 1e4) / 1e4
            if held >= MIN_HOLD and (abs(diff) < THETA_IN / 2 or held >= MAX_HOLD):
                holds.append(held * 8 / 24)
                pos, held = 0, 0
    return pnl, holds


def dblock(v, B=2000, seed=5, block=21):
    if len(v) < block * 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = len(v) // block
    out = []
    for _ in range(B):
        idx = rng.integers(0, nb, nb)
        out.append(np.concatenate([v[j * block:(j + 1) * block] for j in idx]).mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--max-tail", type=int, default=0)   # 0 = 全取（預註冊）
    # 主判定用哪個 CEX 腿：**在看到任何結果之前宣告**用 binance，理由與結果無關
    # ——它有 365 天 funding 歷史，Bitget 只有 90 天，而 90 天的窗口撐不起年化的
    # 主張。Bitget 作為敏感度另跑一次。
    ap.add_argument("--cex", default="binance", choices=["binance", "bitget"])
    a = ap.parse_args()
    end = int(time.time() * 1000); start = end - a.days * 86400 * 1000

    meta = hl_post({"type": "meta"}) or {}
    hl_uni = [x["name"] for x in meta.get("universe", [])]
    bg = binance_symbols() if a.cex == "binance" else bitget_symbols()
    F_FUND = binance_funding if a.cex == "binance" else (lambda sm, st, en: bitget_funding(sm, st))
    F_CAND = binance_candles if a.cex == "binance" else bitget_candles
    print("=" * 104)
    print(f"  路徑 B2：Hyperliquid × {a.cex} funding carry 被動基準（預註冊）｜{a.days} 天")
    print("=" * 104)
    print(f"  HL universe {len(hl_uni)}｜{a.cex} USDT 永續 {len(bg)}｜CEX 腿 = {a.cex}（主判定，事前宣告）")

    def norm(c):
        for p in ("kk", "k"):
            if c.startswith(p) and c[len(p):] in bg:
                return c[len(p):]
        return c

    majors = [c for c in MAJORS if c in hl_uni and c in bg]
    # **全取不篩選**（PREREG）。第一版用 universe 順序截斷到 25 個，那是未註冊的
    # 篩選，而且剛好排除了 HYPE / XPL 這些上市較晚的標的——也就是主張所在的那些。
    tail = [c for c in hl_uni if c not in MAJORS and (c in bg or norm(c) in bg)]
    if a.max_tail and a.max_tail < len(tail):
        print(f"  [WARN] 長尾組被 --max-tail 截斷成 {a.max_tail}/{len(tail)}"
              f" —— 這不是預註冊的一部分，只該用於冒煙測試")
        tail = tail[:a.max_tail]
    print(f"  主流組 {len(majors)}：{majors}")
    print(f"  長尾組 {len(tail)}（HL ∩ {a.cex}，全取不篩選，本輪上限 {a.max_tail}）")

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    def fetch(coin):
        k = f"{coin}:{a.days}:{a.cex}"
        if k in cache:
            return cache[k]
        bsym = coin if coin in bg else norm(coin)
        v = {"fh": hl_funding(coin, start, end), "ph": hl_candles(coin, start, end),
             "fc": F_FUND(bsym, start, end), "pc": F_CAND(bsym, start)}
        cache[k] = v
        return v

    res = {"days": a.days, "groups": {}}
    for gname, coins in (("主流組", majors), ("長尾組", tail)):
        print(f"\n  ── {gname} ──")
        print(f"  {'幣':<10}{'桶':>6}{'|diff|中位':>11}{'(a)年化':>10}{'(b)年化':>10}"
              f"{'(b)持有天中位':>13}{'HL頂檔$':>11}")
        rowsA, rowsB, holds_all, per = [], [], [], {}
        for c in coins:
            v = fetch(c)
            if not (v["fh"] and v["fc"] and v["ph"] and v["pc"]):
                continue
            rows = series(bucketise(v["fh"]), bucketise(v["fc"]),
                          bucketise(v["ph"], "last"), bucketise(v["pc"], "last"))
            if len(rows) < 200:
                continue
            rt_m = 2 * (FEE["maker"]["hl"] + FEE["maker"]["cex"])
            pa = run_a(rows, rt_m)
            pb, holds = run_b(rows, rt_m)
            # 容量＝10 bps 內的累積名目（雙側取小），不是最佳價位單一檔——
            # 單一檔在薄簿上會給出 $0/$3 這種物理上說不通的數字（2026-09-05 抓到）
            book = hl_post({"type": "l2Book", "coin": c}) or {}
            lv = (book.get("levels") or [[], []])
            top = 0.0
            try:
                sides = []
                for si, sgn in ((0, +1), (1, -1)):
                    lvls = [(float(x["px"]), float(x["sz"])) for x in lv[si]]
                    if not lvls:
                        sides.append(0.0); continue
                    best = lvls[0][0]
                    lim = best * (1 - sgn * 10 / 1e4)
                    sides.append(sum(px * sz for px, sz in lvls
                                     if (sgn > 0 and px >= lim) or (sgn < 0 and px <= lim)))
                top = min(sides)
            except Exception:  # noqa: BLE001
                pass
            per[c] = {"buckets": len(rows), "diff_med": float(np.median([abs(r[1]) for r in rows])),
                      "a_ann": float(pa.mean() * 1095), "b_ann": float(pb.mean() * 1095),
                      "hold_med": float(np.median(holds)) if holds else float("nan"),
                      "n_trades": len(holds), "top_usd": top}
            p = per[c]
            print(f"  {c:<10}{len(rows):>6}{p['diff_med']:>11.2f}{p['a_ann']:>+10.1%}{p['b_ann']:>+10.1%}"
                  f"{p['hold_med']:>13.1f}{top:>11,.0f}")
            rowsA.append(pa); rowsB.append(pb); holds_all += holds
        if not rowsA:
            print("  無有效標的"); continue
        A = np.concatenate(rowsA); Bv = np.concatenate(rowsB)
        g = {"coins": len(per), "per_coin": per}
        for lbl in ("maker", "taker", "zero"):
            rt = 2 * (FEE[lbl]["hl"] + FEE[lbl]["cex"])
            pa = np.concatenate([run_a(series(bucketise(fetch(c)["fh"]), bucketise(fetch(c)["fc"]),
                                              bucketise(fetch(c)["ph"], "last"), bucketise(fetch(c)["pc"], "last")), rt)
                                 for c in per])
            pb = np.concatenate([run_b(series(bucketise(fetch(c)["fh"]), bucketise(fetch(c)["fc"]),
                                              bucketise(fetch(c)["ph"], "last"), bucketise(fetch(c)["pc"], "last")), rt)[0]
                                 for c in per])
            lo_a, hi_a = dblock(pa); lo_b, hi_b = dblock(pb)
            g[lbl] = {"a_ann": float(pa.mean() * 1095), "a_ci": [lo_a * 1095, hi_a * 1095],
                      "b_ann": float(pb.mean() * 1095), "b_ci": [lo_b * 1095, hi_b * 1095]}
            print(f"    [{lbl:<5} 來回 {rt:.1f} bps]  (a) B 口徑 {g[lbl]['a_ann']:+7.1%}"
                  f"  CI [{lo_a*1095:+.1%},{hi_a*1095:+.1%}]   (b) B2 口徑 {g[lbl]['b_ann']:+7.1%}"
                  f"  CI [{lo_b*1095:+.1%},{hi_b*1095:+.1%}]")
        if holds_all:
            h = np.array(holds_all)
            print(f"    持有天數：中位 {np.median(h):.1f}  p90 {np.percentile(h,90):.1f}  交易數 {len(h)}")
            g["holds"] = {"median_d": float(np.median(h)), "p90_d": float(np.percentile(h, 90)), "n": len(h)}
        thin = [c for c, p in per.items() if p["top_usd"] < 50_000]
        g["thin_capacity"] = thin
        print(f"    零錢容量（HL 頂檔 < $50k）：{len(thin)}/{len(per)} 個 {thin[:8]}")
        gate = g["maker"]["b_ann"] >= 0.05
        g["gate_5pct"] = bool(gate)
        print(f"    {'✅' if gate else '❌'} 停止條件：(b) maker 年化 ≥ 5%"
              f" → {'往下做 G1' if gate else '本組永久結案'}")
        res["groups"][gname] = g
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
