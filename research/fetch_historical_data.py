"""
Fetch ALL historical data needed for training from Coinglass + Binance APIs.
Saves to D:\flowbot_data\ with proper schema documentation.

Usage:
    set COINGLASS_API_KEY=xxx
    python research/fetch_historical_data.py

Data sources:
  - Coinglass API v4: OI, OI_agg, Liquidation, Long/Short, Global L/S, Funding, Taker
  - Binance Futures REST: 1h klines (OHLCV + quote_vol + taker volumes)
"""
from __future__ import annotations

import json
import os
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
CG_BASE = "https://open-api-v4.coinglass.com/api"
CG_API_KEY = os.environ.get("COINGLASS_API_KEY", "")
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

OUTPUT_DIR = Path("D:/flowbot_data/historical")
SCHEMA_FILE = OUTPUT_DIR / "SCHEMA.md"

# How far back to fetch (Coinglass limit varies by plan)
# Startup plan: up to ~6 months of 1h data (max 4000 per request)
CG_LIMIT = 4000

# Binance: fetch 1 year of 1h klines (max 1500 per request)
BINANCE_LIMIT = 1500

# ── Coinglass Endpoints ───────────────────────────────────────────────────
CG_ENDPOINTS = {
    "oi": {
        "path": "/futures/open-interest/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance BTCUSDT Open Interest (OHLC)",
        "columns": ["time", "open", "high", "low", "close"],
        "unit": "USD notional",
    },
    "oi_agg": {
        "path": "/futures/open-interest/aggregated-history",
        "params": {"symbol": "BTC"},
        "description": "All-exchange aggregated BTC Open Interest (OHLC)",
        "columns": ["time", "open", "high", "low", "close"],
        "unit": "USD notional",
    },
    "liquidation": {
        "path": "/futures/liquidation/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance BTCUSDT Liquidation volumes",
        "columns": ["time", "long_liquidation_usd", "short_liquidation_usd"],
        "unit": "USD",
    },
    "long_short": {
        "path": "/futures/top-long-short-account-ratio/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance top trader long/short account ratio",
        "columns": ["time", "top_account_long_percent", "top_account_short_percent",
                     "top_account_long_short_ratio"],
        "unit": "percent / ratio",
    },
    "global_ls": {
        "path": "/futures/global-long-short-account-ratio/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance global long/short account ratio",
        "columns": ["time", "global_account_long_percent", "global_account_short_percent",
                     "global_account_long_short_ratio"],
        "unit": "percent / ratio",
    },
    "funding": {
        "path": "/futures/funding-rate/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance BTCUSDT Funding Rate (OHLC)",
        "columns": ["time", "open", "high", "low", "close"],
        "unit": "rate (e.g. 0.0001 = 0.01%)",
    },
    "taker": {
        "path": "/futures/taker-buy-sell-volume/history",
        "params": {"exchange": "Binance", "symbol": "BTCUSDT"},
        "description": "Binance BTCUSDT taker buy/sell volume",
        "columns": ["time", "taker_buy_volume_usd", "taker_sell_volume_usd"],
        "unit": "USD",
    },
}


def fetch_coinglass(name: str, cfg: dict, interval: str = "1h") -> pd.DataFrame:
    """Fetch one Coinglass endpoint with pagination."""
    all_rows = []
    end_time = None  # Start from latest, go backward

    print(f"\n  Fetching {name} ({cfg['description']})...")

    for page in range(10):  # Max 10 pages = 40,000 bars
        params = {
            "interval": interval,
            "limit": CG_LIMIT,
            **cfg["params"],
        }
        if end_time:
            params["end_time"] = end_time

        headers = {"CG-API-KEY": CG_API_KEY}

        try:
            resp = requests.get(CG_BASE + cfg["path"], params=params,
                                headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ERROR: {e}")
            break

        if data.get("code") != "0":
            print(f"    API error: code={data.get('code')} msg={data.get('msg')}")
            break

        rows = data.get("data", [])
        if not rows:
            print(f"    No more data (page {page+1})")
            break

        all_rows.extend(rows)

        # Find earliest timestamp for next page
        timestamps = []
        for r in rows:
            t = r.get("t") or r.get("time") or r.get("createTime")
            if t:
                timestamps.append(int(t))
        if timestamps:
            earliest = min(timestamps)
            # Set end_time to 1ms before earliest to avoid overlap
            end_time = earliest - 1
            earliest_dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc)
            print(f"    Page {page+1}: {len(rows)} rows, earliest={earliest_dt}")
        else:
            break

        # If we got fewer than limit, we've reached the end
        if len(rows) < CG_LIMIT:
            print(f"    Reached end of data")
            break

        time.sleep(1.5)  # Rate limit

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Numeric conversion
    for col in df.columns:
        if col not in ("time", "t", "createTime"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp
    time_col = next((c for c in ["t", "time", "createTime"] if c in df.columns), None)
    if time_col:
        ts = df[time_col].astype(float)
        if ts.max() > 1e12:
            df["dt"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["dt"] = pd.to_datetime(ts, unit="s", utc=True)
        df = df.set_index("dt").sort_index()

    # Deduplicate
    df = df[~df.index.duplicated(keep="last")]

    print(f"    Total: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")
    return df


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """Fetch Binance Futures klines with backward pagination."""
    all_rows = []
    end_time = None

    print(f"\n  Fetching Binance {symbol} {interval} klines...")

    for page in range(20):  # Max 20 pages = 30,000 bars (~3.4 years)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": BINANCE_LIMIT,
        }
        if end_time:
            params["endTime"] = end_time

        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            print(f"    ERROR: {e}")
            break

        if not rows:
            print(f"    No more data (page {page+1})")
            break

        all_rows.extend(rows)

        # Find earliest open time for next page
        earliest_ts = min(int(r[0]) for r in rows)
        end_time = earliest_ts - 1
        earliest_dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc)
        print(f"    Page {page+1}: {len(rows)} rows, earliest={earliest_dt}")

        if len(rows) < BINANCE_LIMIT:
            print(f"    Reached end of data")
            break

        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "ts_open", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trade_count", "taker_buy_vol",
        "taker_buy_quote", "ignore",
    ])

    for c in ["open", "high", "low", "close", "volume",
              "taker_buy_vol", "taker_buy_quote", "quote_vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
    df["ts_open"] = pd.to_numeric(df["ts_open"])
    df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Drop incomplete current bar
    df = df.iloc[:-1]

    print(f"    Total: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")
    return df


def write_schema(results: dict):
    """Write schema documentation."""
    lines = [
        "# Historical Data Schema",
        "",
        f"Fetched: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Files",
        "",
    ]

    for name, info in results.items():
        lines.append(f"### {info['file']}")
        lines.append(f"- **Description**: {info['description']}")
        lines.append(f"- **Rows**: {info['rows']}")
        lines.append(f"- **Range**: {info['start']} ~ {info['end']}")
        lines.append(f"- **Interval**: 1h")
        lines.append(f"- **Index**: `dt` (UTC datetime)")
        lines.append(f"- **Columns**:")
        for col, dtype in info["dtypes"].items():
            unit = info.get("unit", "")
            lines.append(f"  - `{col}` ({dtype}) — {unit}")
        lines.append("")

    lines.extend([
        "## Data Sources",
        "",
        "### Coinglass API v4",
        "- Base URL: https://open-api-v4.coinglass.com/api",
        "- Auth: CG-API-KEY header",
        "- Rate limit: ~1 req/sec",
        "- Max rows per request: 4000 (Startup plan)",
        "",
        "### Binance Futures REST",
        "- URL: https://fapi.binance.com/fapi/v1/klines",
        "- No auth required",
        "- Max rows per request: 1500",
        "",
        "## Usage in Training",
        "",
        "These files are the raw inputs to `research/rebuild_1h_enhanced.py`",
        "which builds `research/ml_data/BTC_USD_1h_enhanced.parquet`.",
        "",
        "Feature mapping:",
        "- `cg_oi_1h.parquet` → cg_oi_close, cg_oi_delta, cg_oi_accel, pctchg_*, range_*, upper_shadow",
        "- `cg_oi_agg_1h.parquet` → cg_oi_agg_close, cg_oi_agg_delta, cg_oi_binance_share",
        "- `cg_liquidation_1h.parquet` → cg_liq_long, cg_liq_short, cg_liq_total, cg_liq_ratio",
        "- `cg_long_short_1h.parquet` → cg_ls_long_pct, cg_ls_short_pct, cg_ls_ratio",
        "- `cg_global_ls_1h.parquet` → cg_gls_long_pct, cg_gls_short_pct, cg_gls_ratio, cg_ls_divergence",
        "- `cg_funding_1h.parquet` → cg_funding_close, cg_funding_range",
        "- `cg_taker_1h.parquet` → cg_taker_buy, cg_taker_sell, cg_taker_delta, cg_taker_ratio",
        "- `binance_klines_1h.parquet` → OHLCV, quote_vol, taker volumes, return features",
        "",
    ])

    SCHEMA_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Schema written to {SCHEMA_FILE}")


def main():
    if not CG_API_KEY:
        print("ERROR: COINGLASS_API_KEY not set")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"API Key: {CG_API_KEY[:8]}...")

    results = {}

    # ── Coinglass endpoints ───────────────────────────────────────────
    for name, cfg in CG_ENDPOINTS.items():
        df = fetch_coinglass(name, cfg, interval="1h")
        if df.empty:
            print(f"  WARNING: {name} returned empty!")
            continue

        fname = f"cg_{name}_1h.parquet"
        fpath = OUTPUT_DIR / fname
        df.to_parquet(fpath)
        print(f"    Saved: {fpath} ({fpath.stat().st_size / 1024:.1f} KB)")

        results[name] = {
            "file": fname,
            "description": cfg["description"],
            "rows": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "dtypes": {c: str(df[c].dtype) for c in df.columns if c != "dt"},
            "unit": cfg.get("unit", ""),
        }

        time.sleep(1.5)

    # ── Binance klines ────────────────────────────────────────────────
    df = fetch_binance_klines()
    if not df.empty:
        fname = "binance_klines_1h.parquet"
        fpath = OUTPUT_DIR / fname
        df.to_parquet(fpath)
        print(f"    Saved: {fpath} ({fpath.stat().st_size / 1024:.1f} KB)")

        results["binance_klines"] = {
            "file": fname,
            "description": "Binance BTCUSDT Perpetual 1h klines (OHLCV + taker + quote_vol)",
            "rows": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "dtypes": {c: str(df[c].dtype) for c in df.columns if c != "dt"},
            "unit": "price=USD, volume=BTC, quote_vol=USD",
        }

    # ── Manifest ──────────────────────────────────────────────────────
    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "files": {k: {kk: vv for kk, vv in v.items() if kk != "dtypes"}
                  for k, v in results.items()},
        "total_files": len(results),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  Manifest: {manifest_path}")

    # ── Schema doc ────────────────────────────────────────────────────
    write_schema(results)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"  {info['file']:35s}  {info['rows']:>6d} rows  {info['start'][:10]} ~ {info['end'][:10]}")
    print(f"\n  Total: {len(results)} files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
