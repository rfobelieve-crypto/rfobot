# -*- coding: utf-8 -*-
"""路徑 A 的錄製器：Lighter 永續 L2 × Binance 現貨 L1（2026-09-05 起）

PREREG：`research/exit_paths/PREREG_A_lighter.md`。這支只負責**錄**，不做任何
判斷——判準在 PREREG 裡，跑分析是另一支程式。

**為什麼是 WebSocket**：Lighter 的深層 REST（orderBookOrders / orderBookDetails）
擋在 AWS 人機驗證後面，**不繞過機器人偵測**；`wss://mainnet.zklighter.elliot.ai/
stream` 是它文件裡的公開頻道，先送一筆完整快照（`subscribed/order_book`，
實測 1,681 檔買 / 1,037 檔賣）再送增量（`update/order_book`，每秒約 13 筆，
size=0 表示該價位消失）。

**為什麼對照場館是 Binance 現貨**：§1.15 分鐘級系統的價格序列來自
`orderbook_snapshots_1m`（`exchange='binance'`），規格書寫的 Bitget 是誤記。
另外 Binance **期貨** WS 從這台機器連得上但不吐資料（用 aggTrade 當對照組驗過），
所以也只能用現貨。

**取樣 250 ms**：判準要的是 300 ms 延遲成本，所以落點在 250/500 ms 兩側，
分析時報上下界；PREREG 寫死「上下界跨越門檻就判 INCONCLUSIVE，不取有利端」。

輸出：`research/exit_paths/logs/lighter/YYYY-MM-DD.csv`（每日一檔）
     `research/results/lighter_last.json`（freshness 旗標）

Run: python research/exit_paths/lighter_recorder.py
"""
from __future__ import annotations

import csv
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "research" / "exit_paths" / "logs" / "lighter"
FLAG = ROOT / "research" / "results" / "lighter_last.json"
L_URL = "wss://mainnet.zklighter.elliot.ai/stream"
B_URL = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
# 2026-09-05：加 Bitget 永續。A0 的判準是「訊號能不能轉移到 Lighter」，
# 對照場館要兩個：**Binance 現貨**（§1.15 訊號的原生資料）與 **Bitget 永續**
# （產品端真正下單的地方）。兩個基差都算，PREREG 的 6 bps 門檻對兩者分別判。
G_URL = "wss://ws.bitget.com/v2/ws/public"
MARKET_ID = 1                      # Lighter BTC 永續
SAMPLE_MS = 250
BPS = (1, 5, 10)                   # 累積名目的深度帶

COLS = (["ts_ms", "l_upd_us", "l_bid", "l_ask", "l_bid_sz", "l_ask_sz"]
        + [f"l_cum_bid_{b}bps" for b in BPS] + [f"l_cum_ask_{b}bps" for b in BPS]
        + ["b_bid", "b_ask", "b_bid_sz", "b_ask_sz", "g_bid", "g_ask", "g_bid_sz", "g_ask_sz"])

book = {"bids": {}, "asks": {}, "upd_us": 0, "ready": False}
binance = {"bid": None, "ask": None, "bid_sz": None, "ask_sz": None, "ts": 0.0}
bitget = {"bid": None, "ask": None, "bid_sz": None, "ask_sz": None, "ts": 0.0}
lock = threading.Lock()
stat = {"rows": 0, "l_msgs": 0, "b_msgs": 0, "g_msgs": 0, "day": None, "writer": None, "fh": None}


def flag(ok: bool, reason: str):
    FLAG.write_text(json.dumps({
        "ok": bool(ok), "reason": reason,
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_today": stat["rows"], "l_msgs": stat["l_msgs"], "b_msgs": stat["b_msgs"],
        "g_msgs": stat["g_msgs"],
    }, ensure_ascii=False), encoding="utf-8")


def on_lighter(ws, raw):
    try:
        m = json.loads(raw)
        t = m.get("type", "")
        ob = m.get("order_book") or {}
        if not ob:
            return
        snap = t.startswith("subscribed")
        with lock:
            if snap:
                book["bids"].clear(); book["asks"].clear()
            for side, key in (("bids", "bids"), ("asks", "asks")):
                d = book[side]
                for lv in (ob.get(key) or []):
                    p, s = float(lv["price"]), float(lv["size"])
                    if s <= 0:
                        d.pop(p, None)
                    else:
                        d[p] = s
            book["upd_us"] = int(m.get("last_updated_at") or 0)
            book["ready"] = bool(book["bids"] and book["asks"])
        stat["l_msgs"] += 1
    except Exception as e:  # noqa: BLE001
        print("[WARN] lighter parse:", str(e)[:120], flush=True)


def on_binance(ws, raw):
    try:
        m = json.loads(raw)
        binance.update(bid=float(m["b"]), ask=float(m["a"]),
                       bid_sz=float(m["B"]), ask_sz=float(m["A"]), ts=time.time())
        stat["b_msgs"] += 1
    except Exception as e:  # noqa: BLE001
        print("[WARN] binance parse:", str(e)[:120], flush=True)


def on_bitget(ws, raw):
    try:
        m = json.loads(raw)
        for d in (m.get("data") or []):
            b, a = (d.get("bids") or [None])[0], (d.get("asks") or [None])[0]
            if not b or not a:
                continue
            bitget.update(bid=float(b[0]), ask=float(a[0]),
                          bid_sz=float(b[1]), ask_sz=float(a[1]), ts=time.time())
            stat["g_msgs"] += 1
    except Exception as e:  # noqa: BLE001
        print("[WARN] bitget parse:", str(e)[:120], flush=True)


def run_ws(name, url, sub, handler):
    back = 1
    while True:
        try:
            ws = websocket.WebSocketApp(
                url, on_open=(lambda w: w.send(json.dumps(sub))) if sub else None,
                on_message=handler,
                on_error=lambda w, e: print(f"[WARN] {name}:", str(e)[:110], flush=True),
                on_close=lambda w, c, m: print(f"[WARN] {name} closed {c}", flush=True))
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {name} run_forever:", str(e)[:120], flush=True)
        if name == "lighter":
            with lock:
                book["ready"] = False        # 斷線後的舊簿不可信，等新快照
        time.sleep(back); back = min(back * 2, 60)


def cum(levels, best, sign):
    """best 的 ±b bps 內的累積名目（USD）。sign=+1 買側往下、−1 賣側往上。"""
    out = []
    for b in BPS:
        lim = best * (1 - sign * b / 1e4)
        tot = 0.0
        for p, s in levels.items():
            if (sign > 0 and p >= lim) or (sign < 0 and p <= lim):
                tot += p * s
        out.append(round(tot, 2))
    return out


def writer_loop():
    nxt = time.time()
    while True:
        nxt += SAMPLE_MS / 1000.0
        time.sleep(max(0.0, nxt - time.time()))
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        if day != stat["day"]:
            if stat["fh"]:
                stat["fh"].close()
            OUTDIR.mkdir(parents=True, exist_ok=True)
            p = OUTDIR / f"{day}.csv"
            # 表頭守衛：欄位變過就把舊檔輪替走，不要把新格式的列append 進舊表頭
            # （2026-08-29 的同族——那次是輪替之後下游計數器從零重數，這次是
            #  不輪替導致兩種格式混在同一個檔案，兩個都不會報錯）。
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as fh0:
                        hdr = (fh0.readline() or "").strip().split(",")
                except Exception:  # noqa: BLE001
                    hdr = []
                if hdr != COLS:
                    alt = p.with_name(f"{day}_schema{len(hdr)}.csv")
                    i = 1
                    while alt.exists():
                        alt = p.with_name(f"{day}_schema{len(hdr)}_{i}.csv"); i += 1
                    p.rename(alt)
                    print(f"[INFO] schema changed ({len(hdr)} -> {len(COLS)} cols); rotated to {alt.name}",
                          flush=True)
            new = not p.exists()
            stat["fh"] = open(p, "a", newline="", encoding="utf-8")
            stat["writer"] = csv.writer(stat["fh"])
            if new:
                stat["writer"].writerow(COLS)
            stat["day"], stat["rows"] = day, 0
        with lock:
            if not book["ready"] or binance["bid"] is None or bitget["bid"] is None:
                continue
            bids, asks = dict(book["bids"]), dict(book["asks"])
            upd = book["upd_us"]
        lb, la = max(bids), min(asks)
        if lb >= la:                          # 交叉簿：丟掉，不要寫進去
            continue
        row = ([int(time.time() * 1000), upd, lb, la, bids[lb], asks[la]]
               + cum(bids, lb, +1) + cum(asks, la, -1)
               + [binance["bid"], binance["ask"], binance["bid_sz"], binance["ask_sz"]]
               + [bitget["bid"], bitget["ask"], bitget["bid_sz"], bitget["ask_sz"]])
        stat["writer"].writerow(row); stat["rows"] += 1
        if stat["rows"] % 240 == 0:           # 每分鐘落盤 + 更新旗標
            stat["fh"].flush()
            sb = time.time() - binance["ts"]; sg = time.time() - bitget["ts"]
            flag(sb < 60 and sg < 60,
                 f"rows_today={stat['rows']} binance_stale={sb:.0f}s bitget_stale={sg:.0f}s")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("lighter_recorder: starting", flush=True)
    flag(True, "starting")
    threading.Thread(target=run_ws, args=("lighter", L_URL,
        {"type": "subscribe", "channel": f"order_book/{MARKET_ID}"}, on_lighter), daemon=True).start()
    threading.Thread(target=run_ws, args=("binance", B_URL, None, on_binance), daemon=True).start()
    threading.Thread(target=run_ws, args=("bitget", G_URL,
        {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "books1",
                                      "instId": "BTCUSDT"}]}, on_bitget), daemon=True).start()
    threading.Thread(target=writer_loop, daemon=True).start()
    while True:
        time.sleep(60)
        if stat["l_msgs"] == 0 or stat["b_msgs"] == 0 or stat["g_msgs"] == 0:
            flag(False, f"no frames: lighter={stat['l_msgs']} binance={stat['b_msgs']} bitget={stat['g_msgs']}")


if __name__ == "__main__":
    main()
