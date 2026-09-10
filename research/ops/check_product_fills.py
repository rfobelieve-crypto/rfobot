# -*- coding: utf-8 -*-
"""產品端還有沒有在成交（2026-09-10）

**為什麼需要這條**：2026-09-10 拉磨坊的逐筆資料時發現，產品端三個 bot
（磨坊 grid、獵取 raid、V7）的最後一筆 live 成交都在 **2026-08-30/31**，
**十一天零成交**。而當時 freshness board 是 `0 red / 44 tracked`，
`v7 export pipe` 那一列綠著、旗標寫 `HTTP 200, 8 events`。

**管子是通的，裡面沒有東西流過去。** 既有守衛盯的是「端點回不回 200」，
而該問的是「最近有沒有新的成交」。這跟 mistake.md 2026-09-01 同一個形狀：
freshness 監測的是心跳，而這裡沒有心臟。

判準（刻意寬，因為單一策略閒置是正常的）：

    磨坊有月磨損預算閘（燒到上限就停到下月 1 號 UTC），所以它單獨沉默
    不代表壞了。**紅的條件是「全部來源都超過 SILENT_H 小時沒有成交」**
    —— 一條線閒置是策略，三條線同時閒置不是。

    另外記錄每個來源各自的沉默時數，**即使整體是綠的也印出來**，
    因為「某一條悄悄停了」是這次真正發生的事。

輸出 results/product_fills_last.json 給 freshness 的 json_flag 讀。
不改產品端、不送任何單 —— 純唯讀。

**產品端該補的那一半（研究端做不到）**：匯出只有 fills 與 v7 兩種，
沒有「策略還在不在跑」的狀態端點。所以這條守衛只能從「有沒有成交」
反推，分不出「被手動停了」「交易所連線斷了」「帳戶沒錢了」。
那需要產品端加一個 status 匯出。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

SILENT_H = 72.0          # 全部來源都超過這個小時數沒成交 -> 紅
OUT = ROOT / "research" / "results" / "product_fills_last.json"
SRCS = ("grid", "raid", "v7")


def main():
    import mill_ingest as mi

    rows = mi.fetch()
    if rows is None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(
            ok=False, reason="匯出拉不到（token / uid / URL 或服務問題）",
            asof=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False,
            indent=2), encoding="utf-8")
        print("product fills: RED  匯出拉不到")
        return 1

    now_ms = int(time.time() * 1000)
    per = {}
    for src in SRCS:
        g = [f for f in rows if f.get("src") == src and f.get("mode") == "live"]
        if not g:
            per[src] = dict(n=0, last_ms=None, silent_h=None, reliable=0)
            continue
        last = max(int(f["t"]) for f in g)
        per[src] = dict(n=len(g), last_ms=last,
                        silent_h=round((now_ms - last) / 3_600_000, 1),
                        reliable=sum(1 for f in g if f.get("cid_known")))

    live = [v["silent_h"] for v in per.values() if v["silent_h"] is not None]
    ok = bool(live) and min(live) <= SILENT_H
    worst = max(live) if live else None

    import datetime as dt
    print("%-6s %7s %9s %10s %s" % ("來源", "live 筆", "歸屬可靠", "沉默(時)",
                                    "最後一筆 (UTC)"))
    for src in SRCS:
        v = per[src]
        print("%-6s %7d %9d %10s %s"
              % (src, v["n"], v["reliable"],
                 "—" if v["silent_h"] is None else "%.1f" % v["silent_h"],
                 "—" if not v["last_ms"] else
                 dt.datetime.utcfromtimestamp(v["last_ms"] / 1000)
                 .strftime("%Y-%m-%d %H:%M")))

    reason = ("全部來源沉默超過 %.0fh（最久 %.1fh）" % (SILENT_H, worst)
              if not ok else
              "最近一筆 %.1fh 前" % min(live))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        ok=ok, reason=reason, silent_threshold_h=SILENT_H, per_src=per,
        asof=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2,
        ensure_ascii=False), encoding="utf-8")
    print()
    print("product fills: %s  %s" % ("OK" if ok else "RED", reason))
    print("written -> " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
