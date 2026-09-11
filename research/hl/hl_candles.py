# -*- coding: utf-8 -*-
"""Hyperliquid 歷史 K 線下載（2026-09-11）

===========================================================================
這是 HL 上**唯一可以回填的歷史**
===========================================================================
逐地址部位、清算價、掛單簿、成交帶 —— 四樣都**沒有歷史端點**，停一小時就
永久少一小時。但 `candleSnapshot` 有歷史，所以價格這一層可以補。

機制關（§1.10 `prereg_fuel_mechanism.py`）需要 ATR 與後續位移，兩者都只要
價格。所以把這段抓下來，機制關的價格側就不必等三個月。

===========================================================================
凍結的決定
===========================================================================
    單次上限   **5000 根**（實測：要 730 天的 1h 只回 5003 根，最早
               2026-02-14）—— 所以必須用 endTime **往回翻頁**，
               不是把 startTime 設很早就好。第一版如果只發一次請求，
               會安靜地只拿到最近 7 個月並且看起來很正常。
    bar 標籤   用 **`t`（開盤時刻）** 當索引，與研究層全專案一致
               （label T 的 bar 在 T+interval 才完整）。`T` 一併存著，
               因為「同一根 bar 的不同欄位屬於不同時刻」這件事咬過
               （mistake.md 2026-09-03），留著才查得出來。
    時間單位   毫秒。**不寫死**，用 harness.to_ms 自動偵測
               （mistake.md 2026-04-12：同一個供應商不同端點單位不同）。
    欄位       t / T / o / h / l / c / v / n
               **`n` 是該根的成交筆數** —— CEX 的公開 K 線沒有這一欄，
               它讓「平均單筆大小」變成可算的，留著。
    停止條件   某一頁回 < 2 根、或最早時間沒有再往前推 -> 該幣抓完。
               「沒有再往前推」這條是必要的：端點對上古時期會一直回同一頁，
               只看根數會無限迴圈。
    落盤       parquet，一幣一檔，按 t 去重後排序。重跑是增量不是覆寫。

宇宙：**主場**（dex 為空）的全部永續。子場館與現貨先不抓 —— 部位那側也
沒在錄它們，抓了價格也配不成對。

    python research/hl/hl_candles.py --interval 1h
    python research/hl/hl_candles.py --interval 1h --coins BTC,ETH   （驗收）
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research.harness import to_ms          # noqa: E402  單位自動偵測，不寫死

INFO = "https://api.hyperliquid.xyz/info"
OUTDIR = HERE / "data" / "candles"
FLAG = ROOT / "research" / "results" / "hl_candles_last.json"
SLEEP = 0.5           # candleSnapshot 是**重端點**：0.15 讓 234 個幣裡
                      # 42 個吃到 429（2026-09-11 實測）。錄製器那邊的
                      # 0.15 不動 —— 它打的是 l2Book / clearinghouseState。
PAGE_MAX = 5000       # 端點單次上限，實測值
MS = dict(__import__("collections").OrderedDict())


def post(body, tries=6):
    """429 要**指數退避**，不是固定 1 秒。

    被限流是一個會持續一段時間的狀態，固定短退避只是繼續撞牆。
    2026-09-11 第一輪用 1 秒 x 3 次，42 個幣直接放棄。
    """
    for i in range(tries):
        try:
            r = urllib.request.Request(
                INFO, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(r, timeout=30).read())
        except Exception as e:
            if i == tries - 1:
                raise
            is429 = "429" in str(e)
            time.sleep(min(20.0, (2.0 ** i) * (2.0 if is429 else 0.5)))
    return None


def main_coins():
    """主場永續的幣別清單。"""
    meta, ctxs = post({"type": "metaAndAssetCtxs"})
    return [u["name"] for u in meta["universe"] if not u.get("isDelisted")]


def interval_ms(iv):
    n = int("".join(c for c in iv if c.isdigit()))
    u = "".join(c for c in iv if c.isalpha())
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[u]


def fetch_coin(coin, interval, stop_ms=None):
    """往回翻頁抓到底。回傳 DataFrame（可能為空）。"""
    step = interval_ms(interval)
    end = int(time.time() * 1000)
    out, earliest = [], None
    for _page in range(200):                 # 硬上限，防端點行為改變造成無限迴圈
        r = post({"type": "candleSnapshot",
                  "req": {"coin": coin, "interval": interval,
                          "startTime": end - PAGE_MAX * step, "endTime": end}})
        if not r or len(r) < 2:
            break
        out.extend(r)
        t0 = min(int(x["t"]) for x in r)
        # **沒有再往前推就停**：端點對上古時期會一直回同一頁，
        # 只看「根數夠不夠」會無限迴圈。
        if earliest is not None and t0 >= earliest:
            break
        earliest = t0
        if stop_ms and t0 <= stop_ms:
            break
        end = t0 - 1
        time.sleep(SLEEP)
    if not out:
        return pd.DataFrame()
    d = pd.DataFrame(out)
    for c in ("t", "T", "n"):
        if c in d:
            d[c] = d[c].astype("int64")
    for c in ("o", "h", "l", "c", "v"):
        if c in d:
            d[c] = d[c].astype(float)
    d["t"] = d["t"].map(to_ms)               # 單位自動偵測
    return (d.drop_duplicates(subset=["t"], keep="last")
             .sort_values("t").reset_index(drop=True))


def merge_save(coin, interval, d):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / ("%s_%s.parquet" % (coin, interval))
    if p.exists() and len(d):
        d = (pd.concat([pd.read_parquet(p), d], ignore_index=True)
               .drop_duplicates(subset=["t"], keep="last")
               .sort_values("t").reset_index(drop=True))
    if len(d):
        d.to_parquet(p, index=False)
    return p, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--coins", default=None, help="逗號分隔；預設主場全部")
    ap.add_argument("--missing", action="store_true",
                    help="只抓還沒有 parquet 的幣（429 之後補跑用）")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    coins = (a.coins.split(",") if a.coins else main_coins())
    if a.missing:
        have = {Path(f).name.rsplit("_", 1)[0]
                for f in glob.glob(str(OUTDIR / ("*_%s.parquet" % a.interval)))}
        skipped = len([c for c in coins if c in have])
        coins = [c for c in coins if c not in have]
        print("--missing：已有 %d 個，這輪補 %d 個" % (skipped, len(coins)))
    print("抓 %d 個幣的 %s K 線 -> %s" % (len(coins), a.interval, OUTDIR))
    rows, spans, fails = 0, [], []
    t0 = time.time()
    for i, c in enumerate(coins, 1):
        try:
            d = fetch_coin(c, a.interval)
            if not len(d):
                fails.append(c)
                continue
            p, n = merge_save(c, a.interval, d)
            rows += n
            spans.append((c, n,
                          time.strftime("%Y-%m-%d", time.gmtime(d.t.min()/1000)),
                          time.strftime("%Y-%m-%d", time.gmtime(d.t.max()/1000))))
            if i <= 5 or i % 40 == 0:
                print("  [%3d/%d] %-10s %5d 根  %s ~ %s"
                      % (i, len(coins), c, n, spans[-1][2], spans[-1][3]))
        except Exception as e:
            fails.append("%s:%s" % (c, e))
        time.sleep(SLEEP)

    dur = time.time() - t0
    print("完成：%d 幣、%d 根、%.0f 秒、失敗 %d" % (len(spans), rows, dur, len(fails)))
    if fails:
        print("失敗清單（前 10）:", fails[:10])
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=bool(spans) and len(fails) <= max(3, 0.05 * len(coins)),
        interval=a.interval, coins_ok=len(spans), coins_fail=len(fails),
        rows=rows, dur_sec=round(dur),
        earliest=min((s[2] for s in spans), default=None),
        latest=max((s[3] for s in spans), default=None),
        reason="HL candleSnapshot 單次上限 5000 根，已往回翻頁到底",
        asof=time.strftime("%Y-%m-%d %H:%M:%S")),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("flag -> " + str(FLAG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
