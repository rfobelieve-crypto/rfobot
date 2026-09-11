# -*- coding: utf-8 -*-
"""Hyperliquid 全市場成交帶（常駐 WS，2026-09-11）

===========================================================================
為什麼要錄，以及最大的價值不是成交本身
===========================================================================
`recentTrades` 只給 10 筆，所以 REST 拿不到 tape。WebSocket 可以：
實測訂閱 **234 個幣全部成功、零錯誤**，151.9 筆/秒 = 1,313 萬筆/天、
原始 3.4GB/天（parquet 壓縮後約 300-400MB/天）。

**最大的價值是每一筆成交都帶雙方地址。** 小時快照的覆蓋率受限於「我知道
幾個地址」（實測 242 -> 1,167 個地址時覆蓋 7% -> 17%），而**成交帶裡每個
有部位的人遲早都會出現**。所以 tape 會把地址宇宙從抽樣收斂到接近完整，
覆蓋率跟著走 —— 那比 tape 本身的時序精度更重要。

第二個價值：**清算事件的時點**。清算在公開端點沒有旗標（實測 WS 成交帶
只有 coin/side/px/sz/users、userFills 的 dir 零筆清算），只能靠
「部位在下一個快照消失且價格穿過清算價」推定 —— 那是**小時解析度**。
有 tape 就能把那一小時縮到秒。

===========================================================================
凍結的設計決定
===========================================================================
    存放      **D 槽**（`HL_TAPE_DIR`，預設 D:\\flowbot_data\\hl\\trades）
              C 只有 88GB 且放著 repo；D 有 712GB。10GB/月放 D
    輪替      UTC 小時，parquet
    欄位      ts / coin / side / px / sz / tid / a0 / a1
    地址      寫**自己的** addresses_tape.json，不碰小時錄製器那份
              —— 兩個寫入者共用一個檔是 mistake.md 2026-04-19 那個坑
              （read-then-merge 容易漏，分檔根本不會有這個問題）
    斷線      自動重連，每次重連記一筆；**旗標帶上重連次數**
    判準      freshness 看旗標的 asof 與 ok，不看行程在不在
              （mistake.md 2026-07-28：判斷交易系統死活永遠看資料新鮮度）

**不訂閱現貨**（326 個市場）與其他 10 個 perpDex：先把主場的量跑穩，
擴充是另一個決定，而且要先知道主場一天多少 GB。

===========================================================================
跑法
===========================================================================
    python research/hl/hl_tape.py                  常駐
    python research/hl/hl_tape.py --seconds 120    跑一段就停（驗收用）
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAPE_DIR = Path(os.environ.get("HL_TAPE_DIR", r"D:\flowbot_data\hl\trades"))
ADDR_TAPE = HERE / "data" / "addresses_tape.json"
FLAG = ROOT / "research" / "results" / "hl_tape_last.json"
WS_URL = "wss://api.hyperliquid.xyz/ws"
INFO_URL = "https://api.hyperliquid.xyz/info"
FLUSH_SEC = 300            # 每 5 分鐘落盤一次（斷電最多損失 5 分鐘）

_buf = []
_addrs = set()
_lock = threading.Lock()
_stat = dict(trades=0, reconnects=0, flushes=0, rows_written=0,
             started=time.time(), last_trade_ms=0)


def coins():
    req = urllib.request.Request(
        INFO_URL, data=json.dumps({"type": "meta"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return [m["name"] for m in json.loads(r.read().decode())["universe"]]


def load_tape_addrs():
    if ADDR_TAPE.exists():
        try:
            return set(json.loads(ADDR_TAPE.read_text(encoding="utf-8"))["addresses"])
        except Exception:
            pass
    return set()


def flush():
    """落盤：成交按 UTC 小時分檔 append，地址整份重寫（只增不減）。"""
    with _lock:
        rows, _buf[:] = list(_buf), []
        addrs = set(_addrs)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ts", "coin", "side", "px", "sz",
                                         "tid", "a0", "a1"])
        for hr, part in df.groupby(df.ts // 3_600_000):
            d = TAPE_DIR / time.strftime("%Y%m%d", time.gmtime(hr * 3600))
            d.mkdir(parents=True, exist_ok=True)
            p = d / (time.strftime("%H", time.gmtime(hr * 3600)) + ".parquet")
            if p.exists():
                part = pd.concat([pd.read_parquet(p), part], ignore_index=True)
            # **按 tid 去重**：WS 訂閱時會先送每個幣的近期成交快照，
            # 所以每次重連都會重播一批（也因此第一次 flush 會產生很多舊日期
            # 的檔案 —— 死掉的市場最後一筆成交在 2024 年，那是真資料不是 bug）。
            part = part.drop_duplicates(subset=["tid"], keep="first")
            part.to_parquet(p, index=False)
        _stat["rows_written"] += len(rows)
    ADDR_TAPE.parent.mkdir(parents=True, exist_ok=True)
    ADDR_TAPE.write_text(json.dumps(dict(
        n=len(addrs), updated=time.strftime("%Y-%m-%d %H:%M:%S"),
        note="成交帶自己的地址檔。**不碰小時錄製器那份** —— 兩個寫入者共用"
             "一個檔是 mistake.md 2026-04-19 那個坑；分檔就不會有。",
        addresses=sorted(addrs)), indent=0), encoding="utf-8")
    _stat["flushes"] += 1
    write_flag()


def write_flag():
    up = time.time() - _stat["started"]
    lag = ((time.time() * 1000 - _stat["last_trade_ms"]) / 1000
           if _stat["last_trade_ms"] else None)
    ok = bool(_stat["trades"] > 0 and (lag is None or lag < 120))
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(dict(
        ok=ok,
        reason=("成交 %d 筆（%.1f/秒）、地址 %d、重連 %d、最後一筆 %s 秒前"
                % (_stat["trades"], _stat["trades"] / max(up, 1), len(_addrs),
                   _stat["reconnects"],
                   "—" if lag is None else "%.0f" % lag)),
        trades=_stat["trades"], addresses=len(_addrs),
        reconnects=_stat["reconnects"], rows_written=_stat["rows_written"],
        uptime_sec=round(up), tape_dir=str(TAPE_DIR),
        asof=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False,
        indent=2), encoding="utf-8")


def run(seconds=None):
    import websocket
    cs = coins()
    print("訂閱 %d 個幣的成交帶 -> %s" % (len(cs), TAPE_DIR))
    _addrs.update(load_tape_addrs())
    print("既有地址 %d" % len(_addrs))
    stop_at = (time.time() + seconds) if seconds else None

    def on_open(ws):
        for c in cs:
            ws.send(json.dumps({"method": "subscribe",
                                "subscription": {"type": "trades", "coin": c}}))

    def on_msg(ws, m):
        try:
            d = json.loads(m)
        except Exception:
            return
        if d.get("channel") != "trades":
            return
        with _lock:
            for t in d.get("data", []):
                us = t.get("users") or [None, None]
                _buf.append((int(t["time"]), t["coin"], t.get("side"),
                             float(t["px"]), float(t["sz"]), int(t["tid"]),
                             us[0] if len(us) > 0 else None,
                             us[1] if len(us) > 1 else None))
                for u in us:
                    if isinstance(u, str) and len(u) == 42:
                        _addrs.add(u.lower())
                _stat["trades"] += 1
                _stat["last_trade_ms"] = int(t["time"])
        if stop_at and time.time() > stop_at:
            ws.close()

    last = time.time()
    while True:
        ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_msg)
        ws.run_forever(ping_interval=20, ping_timeout=10)
        flush()
        if stop_at and time.time() > stop_at:
            break
        _stat["reconnects"] += 1
        print("WS 斷線，第 %d 次重連" % _stat["reconnects"])
        time.sleep(min(30, 2 ** min(_stat["reconnects"], 5)))
        last = last  # noqa

    flush()
    up = time.time() - _stat["started"]
    print("停止：成交 %d 筆（%.1f/秒）、寫入 %d 列、地址 %d、重連 %d、%.0f 秒"
          % (_stat["trades"], _stat["trades"] / max(up, 1),
             _stat["rows_written"], len(_addrs), _stat["reconnects"], up))


def ticker():
    while True:
        time.sleep(FLUSH_SEC)
        try:
            flush()
        except Exception as e:
            print("flush 失敗:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=None,
                    help="跑幾秒就停（驗收用；預設常駐）")
    a = ap.parse_args()
    if not a.seconds:
        threading.Thread(target=ticker, daemon=True).start()
    run(a.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
