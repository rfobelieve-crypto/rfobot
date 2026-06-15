"""
Backfill historical data for under-utilized CG / Deribit endpoints.

Three sources:
  1. CG ETF flow history  — daily BTC spot ETF net flow (IBIT, FBTC, etc)
  2. CG Fear & Greed index — daily sentiment (0-100)
  3. Deribit DVOL history  — hourly BTC implied vol index (paginated)

Output parquets land in market_data/raw_data/ and are picked up by
feature_builder_live.py (see the alt-source merge block added there).

All three are stored at NATIVE resolution:
  - ETF flow / F&G: daily, indexed at 00:00 UTC
  - DVOL:           hourly, indexed on the bar open

Downstream merge is handled via pandas.merge_asof (backward) in
feature_builder_live.py, so daily values forward-fill into every 1h bar.

Usage:
    python research/backfill_historical_alt.py
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
RAW = Path("market_data/raw_data")
CG_KEY = os.getenv("COINGLASS_API_KEY", "")
CG_BASE = "https://open-api-v4.coinglass.com/api"
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def _cg_headers() -> dict:
    return {"CG-API-KEY": CG_KEY, "accept": "application/json"}


# ── 1. CG ETF flow history ──────────────────────────────────────────

def backfill_etf_flow() -> None:
    """Fetch full BTC spot ETF daily flow history."""
    print("=" * 60)
    print("[1/3] CG ETF flow history (daily)")
    print("=" * 60)

    try:
        resp = requests.get(
            CG_BASE + "/etf/bitcoin/flow-history",
            headers=_cg_headers(), timeout=30,
        )
        body = resp.json()
        if body.get("code") not in ("0", 0):
            print(f"  API error: {body.get('msg')}")
            return
        data = body.get("data", [])
        if not data:
            print("  empty response")
            return
    except Exception as e:
        print(f"  fetch failed: {e}")
        return

    rows = []
    for entry in data:
        ts = entry.get("timestamp")
        if ts is None:
            continue
        row = {
            "dt": pd.to_datetime(int(ts), unit="ms", utc=True),
            "etf_net_flow_usd": float(entry.get("flow_usd", 0) or 0),
            "etf_btc_price": float(entry.get("price_usd", 0) or 0),
        }
        for t in entry.get("etf_flows", []) or []:
            name = (t.get("etf_ticker", "") or "").upper()
            flow = float(t.get("flow_usd", 0) or 0)
            if name == "IBIT":
                row["etf_flow_ibit"] = flow
            elif name == "FBTC":
                row["etf_flow_fbtc"] = flow
            elif name == "GBTC":
                row["etf_flow_gbtc"] = flow
        rows.append(row)

    df = pd.DataFrame(rows).set_index("dt").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    out = RAW / "cg_etf_flow_daily.parquet"
    df.to_parquet(out)
    print(f"  saved {len(df)} days, {df.index.min().date()} → {df.index.max().date()}")


# ── 2. CG Fear & Greed history ──────────────────────────────────────

def backfill_fear_greed() -> None:
    """Fetch full historical Fear & Greed index (daily)."""
    print("\n" + "=" * 60)
    print("[2/3] CG Fear & Greed index (daily)")
    print("=" * 60)

    try:
        resp = requests.get(
            CG_BASE + "/index/fear-greed-history",
            headers=_cg_headers(), params={"limit": 4000}, timeout=30,
        )
        body = resp.json()
        if body.get("code") not in ("0", 0):
            print(f"  API error: {body.get('msg')}")
            return
        d = body.get("data", {})
        if not isinstance(d, dict):
            print("  unexpected shape")
            return
        values = d.get("data_list", [])
        times = d.get("time_list", [])
        if not values or not times:
            print("  empty")
            return
    except Exception as e:
        print(f"  fetch failed: {e}")
        return

    df = pd.DataFrame({
        "dt": pd.to_datetime(times, unit="ms", utc=True),
        "fear_greed_value": [float(v) for v in values],
    }).set_index("dt").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    out = RAW / "cg_fear_greed_daily.parquet"
    df.to_parquet(out)
    print(f"  saved {len(df)} days, {df.index.min().date()} → {df.index.max().date()}")


# ── 3. Deribit DVOL history ─────────────────────────────────────────

def backfill_dvol(target_days: int = 250) -> None:
    """Fetch Deribit BTC DVOL hourly OHLC with pagination."""
    print("\n" + "=" * 60)
    print(f"[3/3] Deribit DVOL history (hourly, target {target_days}d)")
    print("=" * 60)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total_span_ms = target_days * 86400 * 1000
    start_ms = end_ms - total_span_ms

    all_rows: list[list] = []
    cursor_end = end_ms
    max_iter = 20

    for i in range(max_iter):
        try:
            resp = requests.get(
                DERIBIT_BASE + "/get_volatility_index_data",
                params={
                    "currency": "BTC",
                    "start_timestamp": start_ms,
                    "end_timestamp": cursor_end,
                    "resolution": "3600",
                },
                timeout=30,
            )
            body = resp.json()
        except Exception as e:
            print(f"  iter {i}: request failed: {e}")
            break

        result = body.get("result", {})
        rows = result.get("data", []) if isinstance(result, dict) else result
        if not rows:
            break

        all_rows.extend(rows)
        cont = result.get("continuation") if isinstance(result, dict) else None
        earliest = int(rows[0][0])
        print(f"  iter {i}: +{len(rows)} rows, earliest={pd.to_datetime(earliest, unit='ms', utc=True)}")

        if cont is None or earliest <= start_ms:
            break
        cursor_end = earliest - 1
        time.sleep(0.3)

    if not all_rows:
        print("  no data")
        return

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close"])
    df["dt"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("dt").sort_index().drop(columns=["ts"])
    df = df[~df.index.duplicated(keep="last")]
    df.columns = ["dvol_open", "dvol_high", "dvol_low", "dvol_close"]

    out = RAW / "deribit_dvol_1h.parquet"
    df.to_parquet(out)
    print(f"  saved {len(df)} hours, {df.index.min()} → {df.index.max()}")


def run_all() -> None:
    backfill_etf_flow()
    backfill_fear_greed()
    backfill_dvol(target_days=250)
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
