# -*- coding: utf-8 -*-
"""路徑 C 的前置檢查：強平方向約定的價格對齊證明（2026-09-05）

**要擋的錯**：`liq_events.side` 的語意是我在錄製器裡人工對齊的——OKX 的 `side`
是強平單自己的方向，Bybit v5 的 `S` 是被平倉部位的方向，我把後者翻轉過來。
**如果翻錯了，主假設（空單被強平＝被迫買入＝價格上漲）整個反過來**，而且要等
累積到 80 個級聯事件、跑完整條分析才會發現。

**已知答案的對照**：被迫買入必然把價格往上推，這是物理不是統計。所以：
  side=BUY （空單被強平）的分鐘，該分鐘報酬應該**顯著為正**
  side=SELL（多單被強平）的分鐘，該分鐘報酬應該**顯著為負**
兩者若同號或反過來，就是約定錯了或時間戳錯了——**先修儀器，不要解讀**
（mistake.md 2026-07-29：新寫的診斷要先在答案已知的資料上跑一次）。

順帶量兩件事，都是 PREREG C 之後要用的：
  (a) 兩所對同一個幣的強平時間戳有沒有系統性偏移（跨所對帳）
  (b) 單邊佔比 ≥80% 的分鐘佔多少（C.2.1 的事件定義要用這個條件）

Run: python research/exit_paths/path_c_align.py [--hours 6]
Out: research/results/path_c_align.json
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
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research" / "results" / "path_c_align.json"
DOM = 0.80          # 單邊佔比門檻（C.2.1）


def klines(sym, start_ms, end_ms):
    """Binance 期貨 1m K 線（REST 可用；WS 才是被擋的那個）。"""
    try:
        d = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": f"{sym}USDT", "interval": "1m",
                                 "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                         timeout=20).json()
        if not isinstance(d, list):
            return {}
        return {int(x[0]): (float(x[1]), float(x[4])) for x in d}   # open, close
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--hours", type=float, default=6.0)
    a = ap.parse_args()
    since = int((time.time() - a.hours * 3600) * 1000)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, venue, side, FLOOR(ts_event/60000)*60000 m, "
                "SUM(notional_usd) usd, COUNT(*) n FROM liq_events "
                "WHERE ts_event >= %s GROUP BY 1,2,3,4", (since,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        print("尚無強平資料"); return 0

    agg = {}
    for r in rows:
        k = (r["symbol"], int(r["m"]))
        d = agg.setdefault(k, {"BUY": 0.0, "SELL": 0.0, "venues": set(), "n": 0})
        d[r["side"]] = d.get(r["side"], 0.0) + float(r["usd"])
        d["venues"].add(r["venue"]); d["n"] += int(r["n"])

    print("=" * 96)
    print(f"  路徑 C 前置：強平方向約定的價格對齊證明｜近 {a.hours:g} 小時"
          f"｜{len(rows)} 組、{len(agg)} 個(幣,分鐘)")
    print("=" * 96)

    syms = sorted({k[0] for k in agg})
    lo, hi = min(k[1] for k in agg), max(k[1] for k in agg) + 60000
    kl = {}
    for s in syms:
        k = klines(s, lo - 60000, hi + 60000)
        if k:
            kl[s] = k
    print(f"  取得 1m K 線的幣：{len(kl)}/{len(syms)}")

    buy, sell, dom_n = [], [], 0
    for (s, m), d in agg.items():
        tot = d["BUY"] + d["SELL"]
        if tot <= 0 or s not in kl or m not in kl[s]:
            continue
        o, c = kl[s][m]
        ret = (c / o - 1) * 1e4                      # 該分鐘的報酬 bps
        share = max(d["BUY"], d["SELL"]) / tot
        if share < DOM:
            continue
        dom_n += 1
        (buy if d["BUY"] > d["SELL"] else sell).append((ret, tot))

    def rep(tag, v, expect):
        if not v:
            print(f"  {tag:<26} 無樣本"); return None
        r = np.array([x[0] for x in v])
        wr = float((np.sign(r) == expect).mean())
        se = r.std() / np.sqrt(len(r)) if len(r) > 1 else float("nan")
        print(f"  {tag:<26} n={len(r):<4} 該分鐘報酬均值 {r.mean():+7.2f} bps"
              f"  中位 {np.median(r):+7.2f}  ±SE {se:.2f}  方向符合率 {wr:.0%}")
        return {"n": int(len(r)), "mean_bps": float(r.mean()), "median_bps": float(np.median(r)),
                "se_bps": float(se), "sign_match": wr}

    print(f"  單邊佔比 ≥{DOM:.0%} 的 (幣,分鐘)：{dom_n}\n")
    b = rep("BUY（空單被強平）", buy, +1)
    s_ = rep("SELL（多單被強平）", sell, -1)

    ok = None
    if b and s_ and b["n"] >= 5 and s_["n"] >= 5:
        gap = b["mean_bps"] - s_["mean_bps"]
        se = np.hypot(b["se_bps"], s_["se_bps"])
        ok = bool(gap > 0 and gap > 2 * se)
        print(f"\n  BUY − SELL = {gap:+.2f} bps（±{se:.2f}）"
              f"  → 方向約定 {'正確（被迫買入推高價格）' if ok else '**存疑，先修儀器不要解讀**'}")
        if gap < 0:
            print("  ⚠ 差為負＝約定翻反了：liq_recorder 的 Bybit/OKX side 對齊要重看")
    else:
        print("\n  ⏳ 樣本不足以判斷方向約定（兩側各需 ≥5 個單邊分鐘），讓錄製器多跑幾小時")

    both = [k for k, d in agg.items() if len(d["venues"]) > 1]
    print(f"  跨所同時看到的 (幣,分鐘)：{len(both)}  —— 兩所對帳要等樣本更多")
    res = {"hours": a.hours, "groups": len(rows), "cells": len(agg), "dominant_cells": dom_n,
           "buy": b, "sell": s_, "convention_ok": ok, "cross_venue_cells": len(both),
           "symbols_with_klines": len(kl)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
