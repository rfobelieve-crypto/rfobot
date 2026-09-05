# -*- coding: utf-8 -*-
"""路徑 C 的 C0：交易所強平推送地面真值錄製器（OKX + Bybit，2026-09-05 起）

**為什麼要自己錄**：Coinglass 本專案的方案對清算歷史只開放 1 小時
（1m/5m/15m 實測 403），而一個級聯是幾分鐘的事件——1 小時桶會把它抹平。
交易所自己的清算推送是逐筆、免費、無第三方延遲的地面真值，但**只有即時流、
沒有歷史**。所以這條線的時鐘從今天開始走。

**場館選擇是被機器決定的，不是偏好**：Binance 期貨 WS（`fstream`）從這台機器
連得上但**不吐任何資料**——用 `btcusdt@aggTrade` 當對照組（必定有流量）驗證，
也是零幀，所以是連線被擋不是清算稀疏（Binance 現貨 WS 正常）。改用
**OKX `liquidation-orders`（一次訂閱涵蓋全部 SWAP）+ Bybit `allLiquidation`
（逐標的訂閱，20 幣宇宙事前凍結）**。兩所併錄比單一來源硬：一所斷線時另一所
還在，而且可以互相對帳。

**單位（先修，別事後查）**：OKX 的 `sz` 是**張數**不是幣數，名目要乘
`ctVal`；幣本位合約（BTC-USD-SWAP，`ctValCcy=USD`）的一張直接就是 100 美元、
**不再乘價格**。Bybit 線性合約的 `v` 已經是幣數。兩所單位不同而欄位名字很像，
這正是 2026-09-03 把 Bitget 深度記成萬分之一的那個坑。

寫入 MySQL `liq_events`（每筆一列，`symbol` 一律正規化成幣種代號如 BTC）。每輪 flush 另寫
`research/results/liq_last.json {ok, reason, asof}` 給 freshness board——
mtime 規則分不出「今天沒有清算，合法」與「WS 斷了，是 bug」，兩者留下
同一種空白（mistake.md 2026-09-03）。

方向約定（寫死，之後不要重推）：`side` 是**強平單自己的方向**。
`SELL` = 多單被強平（被迫賣出）→ 價格下跌級聯。
`BUY`  = 空單被強平（被迫買回）→ 價格上漲級聯 = 主假設那一側。

Run: python research/exit_paths/liq_recorder.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import websocket

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.db import get_db_conn  # noqa: E402

FLAG = ROOT / "research" / "results" / "liq_last.json"
OKX_URL = "wss://ws.okx.com:8443/ws/v5/public"
BYBIT_URL = "wss://stream.bybit.com/v5/public/linear"
# 宇宙事前凍結（30 日永續成交量前段，2026-09-05 定，不隨結果調整）
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "AVAX", "SUI",
            "LTC", "TRX", "DOT", "NEAR", "APT", "UNI", "AAVE", "ARB", "OP", "PEPE"]
# Bybit 的部分標的帶乘數前綴；**逐一訂閱不是批次**——實測一個壞主題
# （allLiquidation.PEPEUSDT）會讓整批 20 個訂閱一起被拒，而且只回一行
# success:false，看起來就像「市場很安靜」。
BYBIT_ALIAS = {"PEPE": "1000PEPE"}
FLUSH_N, FLUSH_S = 40, 30.0

DDL = """
CREATE TABLE IF NOT EXISTS liq_events (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts_event BIGINT NOT NULL,
  venue VARCHAR(12) NOT NULL,
  symbol VARCHAR(24) NOT NULL,
  side VARCHAR(8) NOT NULL,
  price DECIMAL(24,10) NOT NULL,
  qty DECIMAL(28,10) NOT NULL,
  notional_usd DECIMAL(24,6) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY ix_sym_ts (symbol, ts_event),
  KEY ix_venue (venue, ts_event),
  KEY ix_ts (ts_event)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

CTVAL = {}                          # instId -> (ctVal, ctValCcy)


def load_okx_ctval():
    """張數 → 名目 的換算表。抓不到就讓錄製器不要啟動：寧可沒有資料，
    也不要一整批單位錯掉的資料（錯的資料比沒有資料貴）。"""
    r = requests.get("https://www.okx.com/api/v5/public/instruments",
                     params={"instType": "SWAP"}, timeout=20).json()
    for x in r.get("data") or []:
        CTVAL[x["instId"]] = (float(x.get("ctVal") or 0), str(x.get("ctValCcy") or ""))
    if len(CTVAL) < 50:
        raise RuntimeError(f"okx ctVal map too small ({len(CTVAL)}) — refusing to record wrong units")
    print(f"okx ctVal map: {len(CTVAL)} instruments", flush=True)


def base_of(sym: str) -> str:
    s = sym.upper()
    if "-" in s:
        return s.split("-")[0]
    for pre in ("1000000", "10000", "1000"):
        if s.startswith(pre) and len(s) > len(pre) + 3:
            s = s[len(pre):]
            break
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            return s[: -len(q)]
    return s


buf, lock = [], threading.Lock()
stat = {"n": 0, "last_flush": 0.0, "last_msg": 0.0}


def flag(ok: bool, reason: str):
    FLAG.write_text(json.dumps({
        "ok": bool(ok), "reason": reason,
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_total": stat["n"],
    }, ensure_ascii=False), encoding="utf-8")


def flush(force=False):
    with lock:
        if not buf or (not force and len(buf) < FLUSH_N and time.time() - stat["last_flush"] < FLUSH_S):
            return
        rows, buf[:] = list(buf), []
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO liq_events (ts_event,venue,symbol,side,price,qty,notional_usd) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)", rows)
            conn.commit()
        finally:
            conn.close()
        stat["n"] += len(rows); stat["last_flush"] = time.time()
        flag(True, f"flushed {len(rows)}")
        print(f"[{datetime.now():%H:%M:%S}] +{len(rows)} rows (total {stat['n']})", flush=True)
    except Exception as e:  # noqa: BLE001
        with lock:
            buf[:0] = rows                      # put them back, do not lose events
        flag(False, f"db flush failed: {e}")
        print("[WARN] flush failed:", e, flush=True)


def _push(ts, venue, sym, side, px, qty, notional):
    """qty 一律存**幣數**（notional/price），不存場館原生單位——OKX 給的是張數，
    Bybit 給的是幣數，混在同一欄會讓下游的 z-score 跨所不可比。"""
    if px <= 0 or qty <= 0 or notional <= 0:
        return
    with lock:
        buf.append((int(ts), venue, base_of(sym), side, px, notional / px, notional))


def on_okx(ws, raw):
    """liquidation-orders: data[].details[] = {side, sz, bkPx, ts}. `side` 是強平單
    自己的方向，與 Bybit 對齊後統一成 BUY/SELL。"""
    stat["last_msg"] = time.time()
    try:
        m = json.loads(raw)
        for d in (m.get("data") or []):
            inst = d.get("instId", "")
            ct, ccy = CTVAL.get(inst, (0.0, ""))
            if ct <= 0:
                continue                      # 不認識的合約寧可丟掉，不要猜單位
            for x in (d.get("details") or []):
                px, sz = float(x.get("bkPx") or 0), float(x.get("sz") or 0)
                qty = sz * ct                 # 幣數（幣本位時是計價幣金額）
                notional = qty if ccy in ("USD", "USDT", "USDC") else qty * px
                _push(x.get("ts") or int(time.time() * 1000), "okx", inst,
                      str(x.get("side", "")).upper(), px, sz, notional)
    except Exception as e:  # noqa: BLE001
        print("[WARN] okx parse:", e, flush=True)
    flush()


def on_bybit(ws, raw):
    """allLiquidation.<sym>: data[] = {T, s, S, v, p}. Bybit v5 的 `S` 是**被平倉
    部位的方向**（Buy = 多單被平），與 OKX 的「強平單方向」相反，所以在這裡翻轉
    成統一約定：SELL = 多單被強平。"""
    stat["last_msg"] = time.time()
    try:
        m = json.loads(raw)
        for x in (m.get("data") or []):
            side = "SELL" if str(x.get("S", "")).upper().startswith("B") else "BUY"
            px, v = float(x.get("p") or 0), float(x.get("v") or 0)
            _push(x.get("T") or int(time.time() * 1000), "bybit", x.get("s", ""),
                  side, px, v, px * v)
    except Exception as e:  # noqa: BLE001
        print("[WARN] bybit parse:", e, flush=True)
    flush()


def run_ws(name, url, subs, handler):
    """subs: 要送出的訂閱訊息清單（逐一送，一個被拒不影響其他）。"""
    def _open(w):
        for m in subs:
            w.send(json.dumps(m)); time.sleep(0.05)

    def _wrap(w, raw):
        if '"success":false' in raw or '"event":"error"' in raw:
            print(f"[WARN] {name} subscribe rejected: {raw[:160]}", flush=True)
            return
        if '"success":true' in raw or '"event":"subscribe"' in raw:
            stat["subs"] = stat.get("subs", 0) + 1
            print(f"[OK] {name} subscribed ({stat['subs']}): {raw[:110]}", flush=True)
            return
        handler(w, raw)

    back = 1
    while True:
        try:
            ws = websocket.WebSocketApp(
                url, on_open=_open, on_message=_wrap,
                on_error=lambda w, e: print(f"[WARN] {name} ws:", str(e)[:110], flush=True),
                on_close=lambda w, c, m: print(f"[WARN] {name} closed {c}", flush=True))
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {name} run_forever:", e, flush=True)
        flush(force=True)
        time.sleep(back); back = min(back * 2, 60)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    finally:
        conn.close()
    load_okx_ctval()
    print(f"liq_recorder: table ready — okx(all SWAP) + bybit({len(UNIVERSE)} syms)", flush=True)
    flag(True, "starting")
    threading.Thread(target=run_ws, args=("okx", OKX_URL,
        [{"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}], on_okx),
        daemon=True).start()
    threading.Thread(target=run_ws, args=("bybit", BYBIT_URL,
        [{"op": "subscribe", "args": [f"allLiquidation.{BYBIT_ALIAS.get(s, s)}USDT"]} for s in UNIVERSE],
        on_bybit), daemon=True).start()
    while True:                     # heartbeat: flush on time even when quiet
        time.sleep(20)
        flush(force=True)
        quiet = time.time() - (stat["last_msg"] or time.time())
        if quiet > 1800:
            flag(False, f"no liquidation frames for {quiet/60:.0f} min")


if __name__ == "__main__":
    main()
