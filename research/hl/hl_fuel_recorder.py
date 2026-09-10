# -*- coding: utf-8 -*-
"""Hyperliquid「前方燃料圖」錄製器（2026-09-11）

===========================================================================
為什麼是這個
===========================================================================
2026-09-10 一整天，三條線都死在同一件事上：**我們在推估一個別的地方看得見
的東西**。

    §1.03i  用 OI + **假設的槓桿分布**推導清算位密度
            -> 構造全過、交易全不過，死在它其實是波動度的替身
    §1.05   用公開指標算群眾止損地圖
            -> 跟真實爆倉**反向排列**（spearman −0.182、逐幣 0/9）
    §1.06   用多空比推擁擠程度
            -> 九格最好的一格，隨機也有 5.2% 機率做到

三次問的都是同一個問題：**掃單之後前方有沒有燃料。**
三次都只能用代理，因為中心化交易所不公開部位。

Hyperliquid 公開。`clearinghouseState` 逐地址給出
`coin / szi / entryPx / leverage / **liquidationPx** / positionValue`。
那就是 Coinglass 用模型猜、而我們猜錯三次的那個量 —— 這裡直接讀。

**這支是錄製器不是檢定。** 它不下任何判決；判準要另外預註冊。

**錄四樣東西，而且只錄沒有歷史的那些**（2026-09-11 使用者：「能抓的就抓、
能錄的就錄」）。分類的原則是**可不可以事後補**：

    market   234 個幣的 OI / funding / premium / oracle vs mark / impactPxs
             -> **OI 沒有歷史端點**，不錄就永遠拿不到
    book     L2 簿口兩側各 20 檔
             -> 沒有歷史
    fuel     逐地址部位 -> **清算價**離現價多遠的名目直方圖（多空 x 全倉/逐倉）
             -> 沒有歷史。這是 Coinglass 用模型猜、我們猜錯三次的那個量
    orders   逐地址掛單，**含觸發單（止損/止盈）的 triggerPx**
             -> 沒有歷史。這是 §1.05 用 SuperTrend/PSAR/Donchian 代理、
                結果跟真實爆倉反向排列的那個量 —— 這裡是真的掛單

**可以事後補、今晚不錄的**：candleSnapshot（3 年以上）、fundingHistory
（400 天以上）、userFills（逐地址分頁）。它們有歷史，所以不急。

===========================================================================
實測的可行性（2026-09-11，開工前先量）
===========================================================================
    市場數      234 個永續
    總未平倉    $10.36B（BTC 2.87B / ETH 2.36B / HYPE 1.75B）
    覆蓋率      **242 個地址就涵蓋 6.98% 的總未平倉名目**
                （而且地址是從 40 個幣各 10 筆最近成交隨手撈的）
    部位        6,317 個，其中 60% 帶 liquidationPx
    速率        ~11.5 req/s、無金鑰、無費用
    歷史        **沒有**。`clearinghouseState` 只有當下 -> 只能往前累積

**橫截面救了樣本量**：234 市場 x 每小時 = 每天約 5,600 個幣-小時。
一個月約 17 萬。對照 SDV 是 930 天 1,584 筆 —— 這是整個專案第一次
樣本量不是限制條件。

===========================================================================
凍結的設計決定（在累積任何資料之前寫死）
===========================================================================
    分桶      清算價相對現價的**百分比距離**，固定邊界：
              0.5 / 1 / 2 / 3 / 5 / 7.5 / 10 / 15 / 20 / 30 / 50 / inf
              多單的清算價在現價**之下**、空單在**之上** —— 幾何保證，會驗
    方向      long / short 分開存。「前方燃料」對上漲行情是**空單**的清算價
    保證金模式 **cross / isolated 分開存**（2026-09-11 第一次快照後立刻加）。
              全倉的清算價是**帳戶級**的：一個資金充足的帳戶，它每個部位的
              清算價都遠在天邊（實測 BTC >50% 外有 $26.5M、1-2% 內只有 $905）。
              所以全倉的「距離」混了部位大小與帳戶健康度；**逐倉才是乾淨的
              逐部位清算價**。這個切分在累積任何歷史之前就加進 schema ——
              有了歷史再改就是滾動窗那個病（mistake.md 2026-09-10）。
    聚合量    名目（positionValue）與**部位數**都存
              —— 名目看錢、筆數看人，兩者在級聯裡的意義不同
    覆蓋率    每次快照都存「抽樣名目 / 該幣總未平倉名目」
              **這個數字必須跟著資料一起存**，否則事後無法判斷燃料圖是否完整
    地址宇宙  持續累積、只增不減（有部位的人遲早會成交而被看到）
              存成 addresses.json，**進 data manifest 的 append 類**

**2026-09-11 撤回原本「只存聚合」的決定（累積 1 小時後、歷史還很短時）**：
原本寫「不存逐地址明細（隱私與體積）」。但那個決定讓**結果變數無法重建** ——
清算事件在公開端點裡沒有旗標（實測：WS 成交帶只有 coin/side/px/sz/users，
25 個地址 60 天 32,905 筆成交裡 `dir` 零筆清算），唯一可得的判定是
**「部位在下一個快照消失，而期間價格穿過它的清算價」**，而那需要逐地址明細。

所以改成**逐部位存 parquet**（addr/coin/szi/entry/liq_px/value/lev），
直方圖改成從它推導 —— 一個真相源。體積約 0.3MB/小時壓縮後 = 7MB/天。

這個改動現在做，因為歷史只有 1 小時；有了幾週歷史再改 schema 就是
滾動窗那個病（mistake.md 2026-09-10）。

===========================================================================
跑法
===========================================================================
    python research/hl/hl_fuel_recorder.py --discover     只擴充地址宇宙
    python research/hl/hl_fuel_recorder.py                擴充 + 錄一次快照
    python research/hl/hl_fuel_recorder.py --max-addr 800 限制本輪查幾個地址

自報旗標寫到 results/hl_fuel_last.json 給 freshness 的 json_flag 讀。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
ADDR_FILE = DATA / "addresses.json"
SNAP_DIR = DATA / "snapshots"          # fuel 直方圖（監看用，可從 POS_DIR 推導）
POS_DIR = DATA / "positions"           # **逐部位明細，真相源**
MKT_DIR = DATA / "market"              # 每幣 OI/funding/premium
BOOK_DIR = DATA / "book"               # L2 兩側 20 檔
ORD_DIR = DATA / "orders"              # 掛單 + 觸發單
FLAG = ROOT / "research" / "results" / "hl_fuel_last.json"
API = "https://api.hyperliquid.xyz/info"

# 凍結的分桶邊界（清算價離現價的 % 距離）
BINS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, float("inf"))
DISCOVER_COINS = 60          # 每輪從 OI 最大的前 N 個幣撈地址
MAX_ADDR_DEFAULT = 1200      # 每輪查多少個地址（11.5 req/s -> 約 2 分鐘）
SLEEP = 0.03                 # 約 30 req/s 上限之下的保守間隔


def info(body, tries=3, timeout=30):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                API, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(0.5 * (i + 1))
    return None


def markets():
    """回傳 {coin: dict(mark, oi_usd, funding, oi)}。"""
    d = info({"type": "metaAndAssetCtxs"})
    if not (isinstance(d, list) and len(d) > 1):
        return {}
    out = {}
    for m, c in zip(d[0]["universe"], d[1]):
        try:
            mark = float(c["markPx"])
            oi = float(c["openInterest"])
            out[m["name"]] = dict(mark=mark, oi=oi, oi_usd=oi * mark,
                                  funding=float(c.get("funding") or 0),
                                  max_lev=m.get("maxLeverage"))
        except Exception:
            continue
    return out


def load_addrs():
    if ADDR_FILE.exists():
        try:
            return set(json.loads(ADDR_FILE.read_text(encoding="utf-8"))["addresses"])
        except Exception:
            pass
    return set()


def save_addrs(s):
    ADDR_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADDR_FILE.write_text(json.dumps(dict(
        n=len(s), updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        note="只增不減：有部位的人遲早會成交而被看到。data manifest 的 append 類。",
        addresses=sorted(s)), indent=0), encoding="utf-8")


def discover(mk, known):
    """從成交最近的幾個幣撈新地址。"""
    found = set()
    for coin in sorted(mk, key=lambda k: -mk[k]["oi_usd"])[:DISCOVER_COINS]:
        d = info({"type": "recentTrades", "coin": coin})
        if isinstance(d, list):
            for tr in d:
                for u in (tr.get("users") or []):
                    if isinstance(u, str) and u.startswith("0x") and len(u) == 42:
                        found.add(u.lower())
        time.sleep(SLEEP)
    return found - known, found


def bin_of(pct):
    for i, b in enumerate(BINS):
        if pct <= b:
            return i
    return len(BINS) - 1


def snapshot(mk, addrs, max_addr):
    """查地址的部位。**逐部位明細是真相源**，直方圖同時算出來供監看。"""
    agg = {}          # coin -> (side, lev_type) -> bin -> [notional, count]
    sampled = {}      # coin -> notional
    detail = []       # 逐部位：清算事件只能從它與下一個快照的差推出來
    geom_bad = 0
    n_pos = n_liq = n_ok = 0
    for a in list(addrs)[:max_addr]:
        d = info({"type": "clearinghouseState", "user": a}, tries=2)
        time.sleep(SLEEP)
        if not isinstance(d, dict):
            continue
        n_ok += 1
        for ap in d.get("assetPositions") or []:
            p = ap.get("position") or {}
            coin = p.get("coin")
            if coin not in mk:
                continue
            try:
                szi = float(p.get("szi") or 0)
                val = abs(float(p.get("positionValue") or 0))
            except Exception:
                continue
            if szi == 0 or val <= 0:
                continue
            n_pos += 1
            sampled[coin] = sampled.get(coin, 0.0) + val
            lq = p.get("liquidationPx")
            if lq in (None, "", "null"):
                continue
            try:
                lq = float(lq)
            except Exception:
                continue
            if lq <= 0:
                continue
            n_liq += 1
            mark = mk[coin]["mark"]
            try:
                ent = float(p.get("entryPx") or 0)
            except Exception:
                ent = 0.0
            side = "long" if szi > 0 else "short"
            lv = p.get("leverage") or {}
            lev_type = str(lv.get("type") or "?")
            try:
                lev_val = float(lv.get("value") or 0)
            except Exception:
                lev_val = 0.0
            # 幾何：多單的清算價必在現價之下、空單在之上
            if (side == "long" and lq >= mark) or (side == "short" and lq <= mark):
                geom_bad += 1
                continue
            pct = abs(lq - mark) / mark * 100.0
            k = agg.setdefault(coin, {}).setdefault((side, lev_type), {})
            cell = k.setdefault(bin_of(pct), [0.0, 0, 0.0])
            cell[0] += val
            cell[1] += 1
            cell[2] += val * lev_val          # 名目加權槓桿，事後可還原平均
            detail.append((a, coin, szi, ent, lq, val, lev_type, lev_val,
                           mark))
    return agg, sampled, dict(addrs_ok=n_ok, positions=n_pos, with_liq=n_liq,
                              geom_violations=geom_bad), detail


def rec_market(mk, ts):
    """234 個幣的市場狀態。OI 沒有歷史端點，所以這是唯一的來源。"""
    d = info({"type": "metaAndAssetCtxs"})
    rows = []
    if isinstance(d, list) and len(d) > 1:
        for m, c in zip(d[0]["universe"], d[1]):
            try:
                rows.append(dict(
                    ts=ts, coin=m["name"], max_lev=m.get("maxLeverage"),
                    mark=float(c["markPx"]), oracle=float(c["oraclePx"]),
                    mid=float(c["midPx"]) if c.get("midPx") else None,
                    oi=float(c["openInterest"]),
                    funding=float(c.get("funding") or 0),
                    premium=float(c["premium"]) if c.get("premium") else None,
                    day_ntl_vlm=float(c.get("dayNtlVlm") or 0),
                    impact_bid=(float(c["impactPxs"][0])
                                if c.get("impactPxs") else None),
                    impact_ask=(float(c["impactPxs"][1])
                                if c.get("impactPxs") else None)))
            except Exception:
                continue
    MKT_DIR.mkdir(parents=True, exist_ok=True)
    (MKT_DIR / (time.strftime("%Y%m%d_%H", time.gmtime(ts)) + ".json")).write_text(
        json.dumps(dict(ts=ts, rows=rows), ensure_ascii=False), encoding="utf-8")
    return len(rows)


def rec_book(mk, ts, n_coins):
    """L2 兩側各 20 檔。按 OI 取前 n_coins 個幣（全 234 個也只要 ~20 秒）。"""
    rows = []
    for coin in sorted(mk, key=lambda k: -mk[k]["oi_usd"])[:n_coins]:
        d = info({"type": "l2Book", "coin": coin}, tries=2)
        time.sleep(SLEEP)
        if not (isinstance(d, dict) and d.get("levels")):
            continue
        bid, ask = d["levels"][0], d["levels"][1]
        rows.append(dict(ts=ts, coin=coin,
                         bid=[[lv["px"], lv["sz"], lv["n"]] for lv in bid],
                         ask=[[lv["px"], lv["sz"], lv["n"]] for lv in ask]))
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    (BOOK_DIR / (time.strftime("%Y%m%d_%H", time.gmtime(ts)) + ".json")).write_text(
        json.dumps(dict(ts=ts, rows=rows), ensure_ascii=False), encoding="utf-8")
    return len(rows)


def rec_orders(mk, addrs, ts, max_addr):
    """逐地址掛單。觸發單（止損/止盈）**逐筆存**（它們很少、且獨一無二）；
    普通限價單聚合成距離直方圖（它們在 L2 簿口裡本來就看得見）。"""
    trig, agg = [], {}
    n_ord = n_trig = n_ok = 0
    for a in list(addrs)[:max_addr]:
        d = info({"type": "frontendOpenOrders", "user": a}, tries=2)
        time.sleep(SLEEP)
        if not isinstance(d, list):
            continue
        n_ok += 1
        for o in d:
            coin = o.get("coin")
            if coin not in mk:
                continue
            n_ord += 1
            mark = mk[coin]["mark"]
            try:
                sz = abs(float(o.get("sz") or 0) or float(o.get("origSz") or 0))
            except Exception:
                sz = 0.0
            if o.get("isTrigger"):
                n_trig += 1
                try:
                    tpx = float(o.get("triggerPx"))
                except Exception:
                    continue
                trig.append(dict(coin=coin, side=o.get("side"), sz=sz,
                                 trigger_px=tpx, limit_px=o.get("limitPx"),
                                 order_type=o.get("orderType"),
                                 reduce_only=bool(o.get("reduceOnly")),
                                 pos_tpsl=bool(o.get("isPositionTpsl")),
                                 dist_pct=(tpx - mark) / mark * 100.0,
                                 mark=mark))
                continue
            try:
                px = float(o.get("limitPx"))
            except Exception:
                continue
            pct = abs(px - mark) / mark * 100.0
            side = "bid" if (o.get("side") == "B") else "ask"
            cell = agg.setdefault(coin, {}).setdefault(side, {}).setdefault(
                bin_of(pct), [0.0, 0])
            cell[0] += sz * px
            cell[1] += 1
    rows = [dict(ts=ts, coin=c, side=s, bin=b, notional=v[0], n_ord=v[1])
            for c, sd in agg.items() for s, bb in sd.items()
            for b, v in bb.items()]
    ORD_DIR.mkdir(parents=True, exist_ok=True)
    (ORD_DIR / (time.strftime("%Y%m%d_%H", time.gmtime(ts)) + ".json")).write_text(
        json.dumps(dict(ts=ts, addrs_ok=n_ok, n_orders=n_ord, n_trigger=n_trig,
                        resting=rows, triggers=trig), ensure_ascii=False),
        encoding="utf-8")
    return dict(addrs_ok=n_ok, n_orders=n_ord, n_trigger=n_trig,
                resting_rows=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="只擴充地址宇宙")
    ap.add_argument("--max-addr", type=int, default=MAX_ADDR_DEFAULT)
    ap.add_argument("--book-coins", type=int, default=234)
    ap.add_argument("--tasks", default="market,book,fuel,orders")
    args = ap.parse_args()
    tasks = set(x.strip() for x in args.tasks.split(","))

    t0 = time.time()
    mk = markets()
    if not mk:
        FLAG.parent.mkdir(parents=True, exist_ok=True)
        FLAG.write_text(json.dumps(dict(ok=False, reason="metaAndAssetCtxs 拿不到",
                                        asof=time.strftime("%Y-%m-%d %H:%M:%S")),
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print("RED  metaAndAssetCtxs 拿不到")
        return 1
    tot_oi = sum(v["oi_usd"] for v in mk.values())
    print("市場 %d 個，總未平倉 $%.1fM" % (len(mk), tot_oi / 1e6))

    known = load_addrs()
    new, seen = discover(mk, known)
    known |= new
    save_addrs(known)
    print("地址宇宙 %d（本輪新增 %d，掃了 %d 個幣的最近成交）"
          % (len(known), len(new), DISCOVER_COINS))
    if args.discover:
        print("--discover：只擴充宇宙，不錄快照")
        return 0

    ts = int(time.time())
    did = {}
    if "market" in tasks:
        did["market"] = rec_market(mk, ts)
        print("market  -> %d 個幣的 OI/funding/premium" % did["market"])
    if "book" in tasks:
        did["book"] = rec_book(mk, ts, args.book_coins)
        print("book    -> %d 個幣的 L2（兩側各 20 檔）" % did["book"])
    if "orders" in tasks:
        did["orders"] = rec_orders(mk, known, ts, args.max_addr)
        print("orders  -> %d 筆掛單、**%d 筆觸發單(止損)**、聚合 %d 列"
              % (did["orders"]["n_orders"], did["orders"]["n_trigger"],
                 did["orders"]["resting_rows"]))
    if "fuel" not in tasks:
        FLAG.parent.mkdir(parents=True, exist_ok=True)
        FLAG.write_text(json.dumps(dict(
            ok=True, reason="tasks=%s（未含 fuel）" % args.tasks, did=did,
            asof=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False,
            indent=2), encoding="utf-8")
        return 0

    agg, sampled, stat, detail = snapshot(mk, known, args.max_addr)
    if detail:
        import pandas as pd
        POS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detail, columns=["addr", "coin", "szi", "entry_px",
                                      "liq_px", "value_usd", "lev_type",
                                      "lev_val", "mark"]).assign(ts=ts)             .to_parquet(POS_DIR / (time.strftime("%Y%m%d_%H",
                                                 time.gmtime(ts)) + ".parquet"),
                        index=False)
        print("positions -> %d 個部位明細（真相源，清算事件由相鄰快照差推出）"
              % len(detail))
    cov = sum(sampled.values()) / tot_oi if tot_oi else 0.0
    print("查 %d 個地址 -> %d 部位、%d 個有清算價、幾何違反 %d"
          % (stat["addrs_ok"], stat["positions"], stat["with_liq"],
             stat["geom_violations"]))
    print("覆蓋率 %.2f%% 的總未平倉名目" % (100 * cov))

    rows = []
    for coin, sides in agg.items():
        for (side, lev_type), bins in sides.items():
            for bi, (ntl, cnt, lvw) in bins.items():
                rows.append(dict(ts=ts, coin=coin, side=side,
                                 lev_type=lev_type,
                                 lev_wavg=(lvw / ntl if ntl else None), bin=bi,
                                 bin_hi_pct=(None if BINS[bi] == float("inf")
                                             else BINS[bi]),
                                 notional=ntl, n_pos=cnt,
                                 mark=mk[coin]["mark"],
                                 oi_usd=mk[coin]["oi_usd"],
                                 sampled_usd=sampled.get(coin, 0.0),
                                 funding=mk[coin]["funding"]))
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAP_DIR / (time.strftime("%Y%m%d_%H", time.gmtime(ts)) + ".json")
    out.write_text(json.dumps(dict(
        ts=ts, asof_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)),
        bins=[None if b == float("inf") else b for b in BINS],
        coverage_frac=cov, total_oi_usd=tot_oi, stat=stat,
        n_addresses=len(known), rows=rows), ensure_ascii=False), encoding="utf-8")
    print("快照 %d 列 -> %s" % (len(rows), out.name))

    ok = stat["with_liq"] > 100 and stat["geom_violations"] == 0 and cov > 0.01
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=bool(ok),
        reason=("覆蓋 %.2f%%、%d 個清算價、幾何違反 %d、地址 %d、%.0f 秒"
                % (100 * cov, stat["with_liq"], stat["geom_violations"],
                   len(known), time.time() - t0)),
        coverage_frac=cov, n_addresses=len(known), stat=stat, did=did,
        snapshots=len(list(SNAP_DIR.glob("*.json"))),
        asof=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False,
        indent=2), encoding="utf-8")
    print("hl fuel: %s  （快照累積 %d 個）"
          % ("OK" if ok else "RED", len(list(SNAP_DIR.glob("*.json")))))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
