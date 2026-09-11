# -*- coding: utf-8 -*-
"""§1.19 Gate 0 —— HL 的執行可行性，量測（2026-09-11）

===========================================================================
為什麼這支要排在任何 alpha 工作之前
===========================================================================
`docs/common_cause_scan.md` 假說 1：把 37 個已結案判決橫著讀，最大的一群
有六個成員，全部是「**資訊層過、執行層死**」，而且我們每次都在最後才發現。
CLAUDE.md 核心原則 11 因此訂立 Gate 0：**執行可行性排在資訊層之前**。

這支就是 HL 的 Gate 0。它回答一個數字：**每來回至少要賺幾 bps 才不賠。**
那個數字出來之後，它會在做研究之前就過濾掉大部分想法。

===========================================================================
量得到 vs 量不到（先講清楚）
===========================================================================
**量得到**：REST 往返延遲、WS 資料側延遲、費率級距、頂檔佇列筆數、價差。
**量不到**：**真實的下單到成交延遲**。那需要一個有資金的帳戶去送真單，
而本專案在 HL 上沒有帳戶。REST 往返是它的**下界**，不是它本身
——真實值還要加簽章、風控、撮合。**不得把下界當成實測值引用。**

    python research/hl/gate0.py
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "results" / "hl_gate0.json"
INFO = "https://api.hyperliquid.xyz/info"
MID_DIR = Path(r"D:\flowbot_data\hl\mid")
ZERO_ADDR = "0x" + "0" * 40          # 零成交量帳戶 = 我們的起始級距


def post(body, timeout=20):
    r = urllib.request.Request(INFO, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def rest_rtt(n=12):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        post({"type": "l2Book", "coin": "BTC"})
        ts.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.2)
    ts.sort()
    return dict(median_ms=round(statistics.median(ts)), p25=round(ts[len(ts)//4]),
                p75=round(ts[3*len(ts)//4]), min=round(ts[0]), max=round(ts[-1]), n=n)


def ws_latency(seconds=45):
    """HL 在訊息裡戳的時間 -> 我們收到。含網路與 HL 自己的推送間隔。"""
    import websocket
    lat, end = [], time.time() + seconds

    def on_open(ws):
        for c in ("BTC", "ETH", "SOL", "HYPE"):
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": "l2Book", "coin": c}}))

    def on_msg(ws, m):
        now = time.time() * 1000
        d = json.loads(m)
        if d.get("channel") == "l2Book":
            t = (d.get("data") or {}).get("time")
            if t:
                lat.append(now - int(t))
        if time.time() > end:
            ws.close()

    websocket.WebSocketApp("wss://api.hyperliquid.xyz/ws",
                           on_open=on_open, on_message=on_msg
                           ).run_forever(ping_interval=20)
    lat = sorted(x for x in lat if -5000 < x < 60000)
    if not lat:
        return dict(n=0)
    n = len(lat)
    return dict(median_ms=round(statistics.median(lat)), p25=round(lat[n//4]),
                p75=round(lat[3*n//4]), p95=round(lat[int(0.95*n)]), n=n)


def fees():
    d = post({"type": "userFees", "user": ZERO_ADDR})
    fs = d.get("feeSchedule", {})
    return dict(
        perp_taker_bps=round(1e4 * float(d["userCrossRate"]), 2),
        perp_maker_bps=round(1e4 * float(d["userAddRate"]), 2),
        spot_taker_bps=round(1e4 * float(fs.get("spotCross", 0)), 2),
        spot_maker_bps=round(1e4 * float(fs.get("spotAdd", 0)), 2),
        vip_tiers=[dict(ntl=float(t["ntlCutoff"]),
                        taker_bps=round(1e4 * float(t["cross"]), 2),
                        maker_bps=round(1e4 * float(t["add"]), 2))
                   for t in (fs.get("tiers", {}).get("vip") or [])[:4]],
        mm_rebate_tiers=[dict(maker_fraction=float(t["makerFractionCutoff"]),
                              maker_bps=round(1e4 * float(t["add"]), 3))
                         for t in (fs.get("tiers", {}).get("mm") or [])],
        referral_discount=d.get("activeReferralDiscount"),
    )


def book_stats():
    """從 hl_mid 的落盤算價差與頂檔佇列。**不另外發請求。**"""
    import pandas as pd
    fs = sorted(glob.glob(str(MID_DIR / "**" / "*.parquet"), recursive=True))
    if not fs:
        return dict(n_files=0)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    g = d.groupby("coin").agg(spread=("spread_bps", "median"),
                              bn=("bid_n", "median"), an=("ask_n", "median"),
                              bn5=("bid_n5", "median"), an5=("ask_n5", "median"))
    g["q"] = (g.bn + g.an) / 2
    g["q5"] = (g.bn5 + g.an5) / 2
    return dict(n_files=len(fs), rows=int(len(d)), coins=int(d.coin.nunique()),
                spread_bps_median=round(float(g.spread.median()), 3),
                spread_bps_p25=round(float(g.spread.quantile(.25)), 3),
                spread_bps_p75=round(float(g.spread.quantile(.75)), 3),
                top_queue_orders_median=round(float(g.q.median()), 1),
                top5_queue_orders_median=round(float(g.q5.median()), 1))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    r = dict(asof=time.strftime("%Y-%m-%d %H:%M:%S"),
             rest_rtt=rest_rtt(), ws_latency=ws_latency(), fees=fees(),
             book=book_stats())

    f, b = r["fees"], r["book"]
    tk, mk = f["perp_taker_bps"], f["perp_maker_bps"]
    r["roundtrip_cost_bps"] = dict(taker_taker=round(2 * tk, 2),
                                   maker_taker=round(mk + tk, 2),
                                   maker_maker=round(2 * mk, 2))
    sp = b.get("spread_bps_median")
    if sp is not None:
        r["pure_market_making_net_bps"] = round(sp - 2 * mk, 3)

    print("=== §1.19 Gate 0：HL 執行可行性 ===\n")
    print("延遲（**下界，不是真實下單到成交**）")
    print("  REST 往返    中位 %s ms（%s~%s）" % (r["rest_rtt"]["median_ms"],
                                             r["rest_rtt"]["min"], r["rest_rtt"]["max"]))
    print("  WS 資料側    中位 %s ms，p95 %s ms" % (r["ws_latency"].get("median_ms"),
                                               r["ws_latency"].get("p95")))
    print("  -> 知道簿口變化到能下單，反應往返約 %s ms"
          % (r["ws_latency"].get("median_ms", 0) + r["rest_rtt"]["median_ms"]))
    print("\n費率（零成交量帳戶 = 我們的級距）")
    print("  永續 taker %.1f bps / maker %.1f bps" % (tk, mk))
    print("  現貨 taker %.1f bps / maker %.1f bps"
          % (f["spot_taker_bps"], f["spot_maker_bps"]))
    print("  掛單返佣要 maker 量佔全所 %.1f%% 以上才開始（%s bps）"
          % (100 * f["mm_rebate_tiers"][0]["maker_fraction"],
             f["mm_rebate_tiers"][0]["maker_bps"]) if f["mm_rebate_tiers"] else "")
    print("\n來回成本")
    for k, v in r["roundtrip_cost_bps"].items():
        print("  %-14s %5.1f bps" % (k, v))
    print("\n簿口（前 40 名，%d 列）" % b.get("rows", 0))
    print("  價差中位 %.2f bps（p25 %.2f / p75 %.2f）"
          % (sp, b["spread_bps_p25"], b["spread_bps_p75"]))
    print("  頂檔掛單筆數中位 %.0f、前 5 檔 %.0f"
          % (b["top_queue_orders_median"], b["top5_queue_orders_median"]))
    print("\n**純賺價差（做市）淨值 = %.2f bps** -> %s"
          % (r["pure_market_making_net_bps"],
             "在這個費率級距上，流動幣做市是算術上做不到的"
             if r["pure_market_making_net_bps"] < 0 else "為正"))
    print("\n**Gate 0 的門檻**：任何 HL 策略每來回至少要賺")
    print("   %.1f bps（兩腿都 maker，但我們已量到掛單常不成交）" % r["roundtrip_cost_bps"]["maker_maker"])
    print("   %.1f bps（maker 進 taker 出，**現實路徑**）" % r["roundtrip_cost_bps"]["maker_taker"])
    print("   %.1f bps（兩腿都 taker）" % r["roundtrip_cost_bps"]["taker_taker"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten -> " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
