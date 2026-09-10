# -*- coding: utf-8 -*-
"""擴大幣種宇宙：選幣規則事前凍結（2026-09-10）

使用者 2026-09-10：「一直糾結在這九個幣身上，我覺得要往外拓」。

**為什麼這一步值得做**：SDV 的判決卡在信賴區間下緣跨零（樣本外 792 筆、
下緣 −0.136）。實測「隨機抽 k 個幣」的標準誤：

    3 幣 SE 0.2409   5 幣 0.1872   7 幣 0.1834   9 幣 0.1812

加幣是**唯一不用等時間就能增加樣本**的方法，而且新幣是**從未被看過的
真樣本外** —— 比再切一次歷史強得多。有效是強力證據，沒效也是重要資訊
（代表效應只存在於大幣）。

**SDV 跑的是 NO-OI 版本，所以新幣只需要 1 分鐘 K 線**（D 與 V 都只用
volume/delta）。Binance 的未平倉量歷史只留 30 天，幸好用不到。

===========================================================================
選幣規則（事前凍結，看到任何績效之前就定死）
===========================================================================
    1. Binance USDT **現貨**（與現有九幣同一個市場 —— 見下方修正）
    2. 2024-03 當時**已經有日線資料** = 當時已上市，必須覆蓋整段歷史

    **2026-09-10 修正**：第一版用的是**永續**清單（fapi），但
    `fetch_bars.py` 抓的是**現貨**（api/v3），而現有九幣的資料全是現貨。
    兩者標的不同（永續的 1000PEPE / 1000SHIB / 1000FLOKI / 1000BONK 在
    現貨叫 PEPE / SHIB / FLOKI / BONK），抓取直接 400。混市場比選錯幣
    更糟 —— 那會讓新舊幣的資料來自兩個不同的簿口。已改用現貨清單重凍。
    3. 排名依據 = **2024 年 3 月的日線成交額**（回測期起點當時就知道的
       資訊），不是現在的排名 —— 用現在的排名選就是拿未來資訊挑標的
    4. 取前 N 名，N 事前定為 **30**（含現有九幣）
    5. 穩定幣對、槓桿代幣、明顯的包裝資產排除（規則式，不逐個挑）

**已知並記錄的偏誤：存活者宇宙。** 期間被 Binance 下架的幣拿不到清單，
所以它們不在樣本裡。這對前 30 大的影響有限（下架的幾乎都是小幣），
但它是真的偏誤，不假裝沒有（factor-research §2）。

輸出 data/universe.json —— 一旦寫出就是凍結清單，之後不得增刪。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "universe.json"
CUTOFF_MS = 1707955200000        # 2024-02-15 00:00 UTC
RANK_FROM = "2024-03-01"
RANK_TO = "2024-04-01"
TOP_N = 30
EXCLUDE_HINT = ("USDC", "FDUSD", "TUSD", "BUSD", "DAI", "UP", "DOWN",
                "BULL", "BEAR", "WBTC", "WBETH", "BETH")
CORE9 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX"]


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "poc/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def main():
    print("抓 Binance USDT 現貨清單…")
    info = get("https://api.binance.com/api/v3/exchangeInfo")
    cands = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        base = s.get("baseAsset", "")
        if any(h in base for h in EXCLUDE_HINT):
            continue
        cands.append(base)
    print(f"  USDT 現貨交易對 {len(cands)} 個"
          f"（上市時間由『2024-03 有沒有日線』判斷，比 onboardDate 直接）")

    print(f"抓 {RANK_FROM} ~ {RANK_TO} 的日線成交額當排名依據（當時就知道的資訊）…")
    import datetime as dt
    t0 = int(dt.datetime.fromisoformat(RANK_FROM).replace(
        tzinfo=dt.timezone.utc).timestamp() * 1000)
    t1 = int(dt.datetime.fromisoformat(RANK_TO).replace(
        tzinfo=dt.timezone.utc).timestamp() * 1000)
    vols = {}
    for i, b in enumerate(cands):
        try:
            k = get(f"https://api.binance.com/api/v3/klines?symbol={b}USDT"
                    f"&interval=1d&startTime={t0}&endTime={t1}&limit=40")
            # 當時沒有日線 = 當時還沒上市（比 onboardDate 直接，且對現貨可用）
            if not k or len(k) < 25:
                continue
            vols[b] = sum(float(x[7]) for x in k)     # quote volume
        except Exception:
            continue
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(cands)}")
        time.sleep(0.04)

    rank = sorted(vols.items(), key=lambda x: -x[1])
    top = [b for b, _ in rank[:TOP_N]]
    missing = [c for c in CORE9 if c not in top]
    if missing:
        print(f"  注意：現有九幣裡 {missing} 不在前 {TOP_N}，仍然保留（它們是既有樣本）")
        top = top + missing

    print(f"\n凍結宇宙 {len(top)} 個幣：")
    for i, b in enumerate(top):
        tag = " (現有)" if b in CORE9 else ""
        print(f"  {i+1:2}. {b:8} {vols.get(b, 0)/1e9:8.2f} B{tag}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        frozen_at="2026-09-10",
        rule=dict(market="Binance USDT SPOT",
                  listed_by="has daily klines in 2024-03",
                  rank_by=f"quote volume {RANK_FROM}~{RANK_TO}",
                  top_n=TOP_N,
                  excluded_hints=list(EXCLUDE_HINT),
                  known_bias="survivorship: delisted symbols not obtainable"),
        core9=CORE9,
        universe=top,
        new=[b for b in top if b not in CORE9],
        rank_volume_usd={b: vols.get(b, 0) for b in top},
    ), indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}")
    print(f"新增 {len([b for b in top if b not in CORE9])} 個幣要抓 K 線")


if __name__ == "__main__":
    main()
