# -*- coding: utf-8 -*-
"""Bitget in-venue basis recorder — §0.91 (registered 2026-09-02).

Question (product side's request, 產品端請求_站內資金費收租_20260902.md):
spot long + perp short on the SAME venue collects funding. What does the
annualised distribution actually look like for BTC/ETH — median, and how
often does it flip negative (you pay instead)?

The only data anybody had was one sample on 2026-08-20 (BTC +10.9% /
ADA −19.5%). That is a point, not a distribution, and no spot adapter gets
written before the distribution is known.

TWO LAYERS, and the separation is the whole point:
  * BACKFILL (prior only, never a verdict): history-fund-rate reaches back
    months. Thresholds were frozen in TODO §0.91 BEFORE this was pulled —
    setting them after seeing it would be in-sample by construction.
  * FORWARD (the verdict): rows recorded from the registration day onward.

What is recorded every cycle, per symbol:
  funding rate + its interval (4h contracts exist — annualising with a
  hard-coded 8 would be wrong), mark/index/last/bid/ask so the premium in
  bps is reconstructible, plus volume and open interest.

Premium matters as much as the rate: this trade's real risk is not the
carry going to zero, it is entering at a wide premium and eating the
convergence on the perp leg while the spot gain sits in a different
margin pool (the 2026-08-31 uniMMR case the operator forwarded — spot and
perp are not one account unless the venue says so).

Run: python research/arb/basis_recorder.py            # one cycle
     python research/arb/basis_recorder.py --backfill # 180d prior, once
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "https://api.bitget.com"
PT = "USDT-FUTURES"
TIMEOUT = 20

# Verdict symbols vs context symbols. Only the first two decide anything
# (§0.91); the rest are recorded to see whether flipping negative is an
# altcoin trait, and they can never be swapped in afterwards.
VERDICT_SYMBOLS = ("BTCUSDT", "ETHUSDT")
CONTEXT_SYMBOLS = ("SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                   "LINKUSDT", "AVAXUSDT", "BNBUSDT", "LTCUSDT")
ALL_SYMBOLS = VERDICT_SYMBOLS + CONTEXT_SYMBOLS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _get(path: str, params: dict) -> list:
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "00000":
        raise RuntimeError(f"bitget {path}: code={j.get('code')} {j.get('msg')}")
    return j.get("data") or []


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def ensure_tables(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS basis_obs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts_received BIGINT NOT NULL,
            symbol VARCHAR(24) NOT NULL,
            is_verdict TINYINT NOT NULL DEFAULT 0,
            funding_rate DECIMAL(20,10) NULL,
            fund_interval_h INT NULL,
            funding_annual_pct DOUBLE NULL,
            last_px DOUBLE NULL, bid_px DOUBLE NULL, ask_px DOUBLE NULL,
            mark_px DOUBLE NULL, index_px DOUBLE NULL,
            premium_bp DOUBLE NULL, spread_bp DOUBLE NULL,
            volume_usdt DOUBLE NULL, open_interest DOUBLE NULL,
            KEY idx_sym_ts (symbol, ts_received)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS basis_funding_hist (
            symbol VARCHAR(24) NOT NULL,
            funding_time BIGINT NOT NULL,
            funding_rate DECIMAL(20,10) NOT NULL,
            fund_interval_h INT NULL,
            PRIMARY KEY (symbol, funding_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")


def cycle() -> int:
    tickers = {t["symbol"]: t for t in _get("/api/v2/mix/market/tickers",
                                            {"productType": PT})}
    contracts = {c["symbol"]: c for c in _get("/api/v2/mix/market/contracts",
                                              {"productType": PT})}
    now_ms = int(time.time() * 1000)
    rows = []
    for sym in ALL_SYMBOLS:
        t, c = tickers.get(sym), contracts.get(sym)
        if not t or not c or c.get("symbolStatus") != "normal":
            continue
        fr = _f(t.get("fundingRate"))
        fi = int(_f(c.get("fundInterval"), 8) or 8)
        mark, index = _f(t.get("markPrice")), _f(t.get("indexPrice"))
        bid, ask = _f(t.get("bidPr")), _f(t.get("askPr"))
        ann = (fr * (24.0 / fi) * 365.0 * 100.0) if fr is not None else None
        prem = ((mark - index) / index * 1e4) if (mark and index) else None
        spr = ((ask - bid) / bid * 1e4) if (bid and ask and bid > 0) else None
        rows.append((now_ms, sym, 1 if sym in VERDICT_SYMBOLS else 0,
                     fr, fi, ann, _f(t.get("lastPr")), bid, ask, mark, index,
                     prem, spr, _f(t.get("usdtVolume")),
                     _f(t.get("holdingAmount"))))
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            ensure_tables(cur)
            cur.executemany(
                "INSERT INTO basis_obs (ts_received, symbol, is_verdict, "
                "funding_rate, fund_interval_h, funding_annual_pct, last_px, "
                "bid_px, ask_px, mark_px, index_px, premium_bp, spread_bp, "
                "volume_usdt, open_interest) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        conn.commit()
    finally:
        conn.close()
    say = ", ".join(f"{r[1].replace('USDT','')} {r[5]:+.1f}%"
                    for r in rows if r[2] == 1 and r[5] is not None)
    print(f"basis_obs: {len(rows)} rows ({say})")
    return 0


def backfill(pages: int = 6) -> int:
    """Prior only — thresholds were frozen before this ran (TODO §0.91)."""
    from shared.db import get_db_conn
    conn = get_db_conn()
    total = 0
    try:
        with conn.cursor() as cur:
            ensure_tables(cur)
            contracts = {c["symbol"]: c for c in
                         _get("/api/v2/mix/market/contracts", {"productType": PT})}
            for sym in ALL_SYMBOLS:
                fi = int(_f((contracts.get(sym) or {}).get("fundInterval"), 8) or 8)
                seen = []
                for page in range(1, pages + 1):
                    try:
                        d = _get("/api/v2/mix/market/history-fund-rate",
                                 {"symbol": sym, "productType": PT,
                                  "pageSize": 100, "pageNo": page})
                    except Exception as e:
                        print(f"  {sym} page {page}: {e}")
                        break
                    if not d:
                        break
                    seen += [(sym, int(x["fundingTime"]),
                              float(x["fundingRate"]), fi) for x in d]
                    time.sleep(0.2)
                if seen:
                    cur.executemany(
                        "INSERT IGNORE INTO basis_funding_hist "
                        "(symbol, funding_time, funding_rate, fund_interval_h) "
                        "VALUES (%s,%s,%s,%s)", seen)
                    total += len(seen)
                    print(f"  {sym}: {len(seen)} settlements")
        conn.commit()
    finally:
        conn.close()
    print(f"basis_funding_hist: {total} rows (PRIOR ONLY — not the verdict)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    try:
        return backfill() if a.backfill else cycle()
    except Exception as e:                                  # never kill the bat
        print(f"basis_recorder failed (skip): {e!r}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
