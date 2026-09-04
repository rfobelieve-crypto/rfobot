# -*- coding: utf-8 -*-
"""M1 — read the ACTUAL fee off fill receipts and confront fees.py with it.

Why this script places no orders
--------------------------------
The whole fee table rests on published schedules and one screenshot of a
referral page. M1 (LIVE_50U_SPEC) turns that into "what the venue actually
billed". That needs a fill, but it does NOT need automated order placement:
the operator puts on one minimum-size trade per venue by hand (OKX equity
perps $1.9-7.7, Bitget $5, Binance $5, HL $10) and this reads the receipt.

Placing orders is the operator's action. This file is read-only by
construction: every endpoint below is a query, and there is no signing code
for anything that creates or cancels an order.

What it reports per venue
  * n fills found in the window
  * fee actually charged, in bps of notional, split maker vs taker
  * the value fees.py currently believes, and the difference
  * a verdict line: MATCH (<0.5 bps) / DIFFERS -> fees.py must be corrected
    from the receipt, never the other way round

Credentials (read-only keys are enough; put them in flow_system/.env):
  HL_ACCOUNT_ADDRESS            HL needs no key at all - userFills is public
                                by address, so this one works immediately
  LIGHTER_ACCOUNT_INDEX         public account index
  OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE
  BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
  BINANCE_API_KEY / BINANCE_API_SECRET
A venue with no credentials is skipped with one line, never a crash.

Run: python research/arb/fee_receipts.py [--hours 48]
Out: research/results/arb_fee_receipts.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fees as FEES               # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "arb_fee_receipts.json"
TIMEOUT = 20
MATCH_BPS = 0.5


def env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    f = ROOT / ".env"
    if f.exists():
        for line in io.open(f, encoding="utf-8"):
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def bps(fee: float, notional: float) -> float | None:
    return None if not notional else fee / notional * 1e4


# ── venue readers: each returns [{ts, symbol, side, notional, fee, maker}] ──

def hl_fills(since_ms: int, dex: str = "") -> list:
    """Public: userFills by address. No API key, which is why HL is the one
    venue that can be checked the moment the operator has traded."""
    addr = env("HL_ACCOUNT_ADDRESS")
    if not addr:
        return []
    body = {"type": "userFills", "user": addr}
    r = requests.post("https://api.hyperliquid.xyz/info", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for f in r.json() or []:
        if int(f.get("time", 0)) < since_ms:
            continue
        px, sz = float(f.get("px", 0)), float(f.get("sz", 0))
        out.append({"ts": int(f["time"]), "symbol": f.get("coin"),
                    "side": f.get("side"), "notional": px * sz,
                    "fee": float(f.get("fee", 0)),
                    "maker": bool(f.get("crossed")) is False})
    return out


def okx_fills(since_ms: int) -> list:
    k, s, p = env("OKX_API_KEY"), env("OKX_API_SECRET"), env("OKX_API_PASSPHRASE")
    if not (k and s and p):
        return []
    path = "/api/v5/trade/fills-history?instType=SWAP&limit=100"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    sign = base64.b64encode(hmac.new(s.encode(), (ts + "GET" + path).encode(),
                                     hashlib.sha256).digest()).decode()
    h = {"OK-ACCESS-KEY": k, "OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts,
         "OK-ACCESS-PASSPHRASE": p}
    r = requests.get("https://www.okx.com" + path, headers=h, timeout=TIMEOUT)
    out = []
    for f in (r.json().get("data") or []):
        t = int(f.get("ts", 0))
        if t < since_ms:
            continue
        px, sz = float(f.get("fillPx", 0)), float(f.get("fillSz", 0))
        # OKX size is in contracts; ctVal folded in by the caller's notional
        out.append({"ts": t, "symbol": f.get("instId"), "side": f.get("side"),
                    "notional": px * sz, "fee": -float(f.get("fee", 0)),
                    "maker": f.get("execType") == "M"})
    return out


def bitget_fills(since_ms: int) -> list:
    k, s, p = env("BITGET_API_KEY"), env("BITGET_API_SECRET"), env("BITGET_API_PASSPHRASE")
    if not (k and s and p):
        return []
    path = "/api/v2/mix/order/fills?productType=USDT-FUTURES&limit=100"
    ts = str(int(time.time() * 1000))
    sign = base64.b64encode(hmac.new(s.encode(), (ts + "GET" + path).encode(),
                                     hashlib.sha256).digest()).decode()
    h = {"ACCESS-KEY": k, "ACCESS-SIGN": sign, "ACCESS-TIMESTAMP": ts,
         "ACCESS-PASSPHRASE": p, "locale": "en-US"}
    r = requests.get("https://api.bitget.com" + path, headers=h, timeout=TIMEOUT)
    out = []
    for f in ((r.json().get("data") or {}).get("fillList") or []):
        t = int(f.get("cTime", 0))
        if t < since_ms:
            continue
        px, sz = float(f.get("price", 0)), float(f.get("baseVolume", 0))
        fee = 0.0
        for d in (f.get("feeDetail") or []):
            fee += abs(float(d.get("totalFee", 0)))
        out.append({"ts": t, "symbol": f.get("symbol"), "side": f.get("side"),
                    "notional": px * sz, "fee": fee,
                    "maker": f.get("tradeScope") == "maker"})
    return out


def binance_fills(since_ms: int) -> list:
    k, s = env("BINANCE_API_KEY"), env("BINANCE_API_SECRET")
    if not (k and s):
        return []
    q = f"startTime={since_ms}&timestamp={int(time.time()*1000)}&limit=100"
    sig = hmac.new(s.encode(), q.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"https://fapi.binance.com/fapi/v1/userTrades?{q}&signature={sig}",
                     headers={"X-MBX-APIKEY": k}, timeout=TIMEOUT)
    out = []
    for f in (r.json() if isinstance(r.json(), list) else []):
        out.append({"ts": int(f["time"]), "symbol": f.get("symbol"),
                    "side": "buy" if f.get("buyer") else "sell",
                    "notional": float(f.get("quoteQty", 0)),
                    "fee": float(f.get("commission", 0)),
                    "maker": bool(f.get("maker"))})
    return out


READERS = {"HL": hl_fills, "okx": okx_fills, "bitget": bitget_fills,
           "binance": binance_fills}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48.0)
    a = ap.parse_args()
    since = int((time.time() - a.hours * 3600) * 1000)
    print("=" * 92)
    print(f"  M1 費率回執——過去 {a.hours:.0f} 小時的成交，帳單 vs fees.py（本檔只讀，不下單）")
    print("=" * 92)
    res, any_fill = {"asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                     "window_hours": a.hours, "venues": {}}, False
    for venue, fn in READERS.items():
        try:
            fills = fn(since)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {venue:<9} 讀取失敗：{type(e).__name__}: {e}")
            res["venues"][venue] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if not fills:
            have = venue == "HL" and env("HL_ACCOUNT_ADDRESS")
            why = "沒有成交" if have or venue != "HL" else "未設憑證"
            print(f"  {venue:<9} {why}（跳過）")
            res["venues"][venue] = {"n": 0, "reason": why}
            continue
        any_fill = True
        rows = {"maker": [], "taker": []}
        for f in fills:
            b = bps(f["fee"], f["notional"])
            if b is not None:
                rows["maker" if f["maker"] else "taker"].append(b)
        v = {"n": len(fills)}
        print(f"  {venue:<9} {len(fills)} 筆成交")
        for kind in ("taker", "maker"):
            xs = rows[kind]
            if not xs:
                continue
            actual = sum(xs) / len(xs)
            believed = FEES.fee_bps(venue, maker=(kind == "maker"))
            sched = FEES.fee_bps(venue, maker=(kind == "maker"), rebate=False)
            diff = actual - believed
            verdict = "MATCH" if abs(diff) <= MATCH_BPS else "DIFFERS → 以回執為準改 fees.py"
            v[kind] = {"n": len(xs), "actual_bps": round(actual, 3),
                       "fees_py_bps": round(believed, 3),
                       "schedule_bps": round(sched, 3),
                       "diff_bps": round(diff, 3), "verdict": verdict}
            print(f"      {kind:<6} n={len(xs):<3} 帳單 {actual:>7.3f} bps ｜ "
                  f"fees.py {believed:>6.3f}（費率表 {sched:>5.2f}）｜ "
                  f"差 {diff:>+7.3f} → {verdict}")
        res["venues"][venue] = v
    if not any_fill:
        print("\n  沒有任何成交可讀。M1 的做法：每所手動下一筆最小單"
              "（OKX 股票永續 $1.9–7.7、Bitget $5、Binance $5、HL $10），再跑這支。")
        print("  HL 只要 .env 有 HL_ACCOUNT_ADDRESS 就讀得到（userFills 是公開的），"
              "CEX 需要唯讀 API key。")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
