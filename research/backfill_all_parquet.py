"""
Unified backfill: Binance klines + all 14 CG endpoints -> raw_data parquet

Usage:
    python research/backfill_all_parquet.py          # CLI
    from research.backfill_all_parquet import run_backfill  # import
"""
import os, sys, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
RAW = Path("market_data/raw_data")
CG_KEY = os.getenv("COINGLASS_API_KEY", "")
CG_BASE = "https://open-api-v4.coinglass.com/api"


def cg_headers():
    return {"CG-API-KEY": CG_KEY, "accept": "application/json"}


def cg_fetch(path, params, name=""):
    url = f"{CG_BASE}{path}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=cg_headers(), timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") in ("0", 0):
                data = body.get("data", [])
                if isinstance(data, list):
                    return data
            print(f"  {name}: API error code={body.get('code')} msg={body.get('msg')}")
            return []
        except Exception as e:
            print(f"  {name}: attempt {attempt+1} failed: {e}")
            time.sleep(2 * (attempt + 1))
    return []


def to_1h_df(rows, time_col="time"):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in ("exchange", "symbol", "dt"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if time_col in df.columns:
        # Auto-detect timestamp unit: 10 digits = seconds, 13 digits = milliseconds
        sample_ts = df[time_col].dropna().iloc[0] if len(df) > 0 else 0
        unit = "s" if sample_ts < 1e12 else "ms"
        df["dt"] = pd.to_datetime(df[time_col], unit=unit, utc=True)
        df = df.set_index("dt").sort_index()
        df = df.drop(columns=[time_col], errors="ignore")
    return df


def merge_parquet(existing_path, new_df):
    if existing_path.exists() and not new_df.empty:
        old = pd.read_parquet(existing_path)
        for c in new_df.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new_df.columns:
                new_df[c] = np.nan
        combined = pd.concat([old, new_df[old.columns]])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()
    elif not new_df.empty:
        return new_df.sort_index()
    elif existing_path.exists():
        return pd.read_parquet(existing_path)
    return pd.DataFrame()


def backfill_klines():
    print("=" * 60)
    print("[1/3] Binance klines 1h backfill")
    print("=" * 60)

    klines_path = RAW / "binance_klines_1h.parquet"
    old_klines = pd.read_parquet(klines_path)
    last_ts = int(old_klines.index.max().timestamp() * 1000)
    print(f"  Existing: {len(old_klines)} bars, ends at {old_klines.index.max()}")

    all_new = []
    start_time = last_ts + 3600000
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while start_time < now_ms:
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines", params={
            "symbol": "BTCUSDT", "interval": "1h", "startTime": start_time, "limit": 1500
        }, timeout=15)
        rows = resp.json()
        if not rows:
            break
        df = pd.DataFrame(rows, columns=[
            "ts_open", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trade_count", "taker_buy_vol",
            "taker_buy_quote", "ignore"
        ])
        for c in ["open", "high", "low", "close", "volume",
                   "taker_buy_vol", "taker_buy_quote", "quote_vol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
        df["ts_open"] = pd.to_numeric(df["ts_open"])
        df["dt"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
        df = df.set_index("dt").sort_index()
        complete = df[df.index < pd.Timestamp.now(tz="UTC").floor("h")]
        all_new.append(complete)
        start_time = int(rows[-1][0]) + 3600000
        last = complete.index.max() if len(complete) else "N/A"
        print(f"  Fetched {len(complete)} bars up to {last}")
        time.sleep(0.2)

    if all_new:
        new_klines = pd.concat(all_new)
        old_klines["quote_vol"] = pd.to_numeric(old_klines["quote_vol"], errors="coerce")
        combined = pd.concat([old_klines, new_klines])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        for c in ["open", "high", "low", "close", "volume",
                   "taker_buy_vol", "taker_buy_quote", "quote_vol",
                   "trade_count", "ts_open", "close_time"]:
            if c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce")
        combined.to_parquet(klines_path)
        print(f"  Updated: {len(combined)} bars, {combined.index.min()} -> {combined.index.max()}")
    else:
        print("  No new klines to add")


def backfill_cg(endpoints, label):
    print("\n" + "=" * 60)
    print(f"[{label}] Coinglass endpoints")
    print("=" * 60)

    for name, cfg in endpoints.items():
        pq_file = RAW / f"cg_{name}_1h.parquet"
        rows = cg_fetch(cfg["path"], cfg["params"], name)
        if not rows:
            print(f"  {name}: FAILED")
            continue
        df = to_1h_df(rows)
        if df.empty:
            print(f"  {name}: empty")
            continue
        combined = merge_parquet(pq_file, df)
        combined.to_parquet(pq_file)
        print(f"  {name}: {len(combined)} bars, {combined.index.min().date()} -> {combined.index.max().date()}")
        time.sleep(0.8)


EXISTING_CG = {
    "oi":          {"path": "/futures/open-interest/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "oi_agg":      {"path": "/futures/open-interest/aggregated-history", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000}},
    "liquidation": {"path": "/futures/liquidation/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "long_short":  {"path": "/futures/top-long-short-account-ratio/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "global_ls":   {"path": "/futures/global-long-short-account-ratio/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "funding":     {"path": "/futures/funding-rate/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "taker":       {"path": "/futures/taker-buy-sell-volume/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
}

NEW_CG = {
    "coinbase_premium": {"path": "/coinbase-premium-index", "params": {"interval": "1h", "limit": 4000}},
    "bitfinex_margin":  {"path": "/bitfinex-margin-long-short", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000}},
    "top_ls_position":  {"path": "/futures/top-long-short-position-ratio/history", "params": {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "1h", "limit": 4000}},
    "futures_cvd_agg":  {"path": "/futures/aggregated-cvd/history", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000, "exchange_list": "Binance"}},
    "spot_cvd_agg":     {"path": "/spot/aggregated-cvd/history", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000, "exchange_list": "Binance"}},
    "liq_agg":          {"path": "/futures/liquidation/aggregated-history", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000, "exchange_list": "Binance"}},
    "oi_coin_margin":   {"path": "/futures/open-interest/aggregated-coin-margin-history", "params": {"symbol": "BTC", "interval": "1h", "limit": 4000, "exchange_list": "Binance"}},
}


DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def backfill_dvol():
    """Incremental backfill of Deribit BTC DVOL hourly OHLC."""
    print("\n" + "=" * 60)
    print("[4/4] Deribit DVOL 1h backfill")
    print("=" * 60)

    pq_path = RAW / "deribit_dvol_1h.parquet"
    if pq_path.exists():
        old = pd.read_parquet(pq_path)
        start_ms = int(old.index.max().timestamp() * 1000) + 1
        print(f"  Existing: {len(old)} bars, ends at {old.index.max()}")
    else:
        old = pd.DataFrame()
        start_ms = int((pd.Timestamp.now("UTC") - pd.Timedelta(days=250)).timestamp() * 1000)
        print("  No existing data, fetching 250 days")

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if start_ms >= end_ms:
        print("  Already up to date")
        return

    all_rows = []
    cursor_end = end_ms
    for i in range(20):
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
        earliest = int(rows[0][0])
        print(f"  iter {i}: +{len(rows)} rows, earliest={pd.to_datetime(earliest, unit='ms', utc=True)}")

        cont = result.get("continuation") if isinstance(result, dict) else None
        if cont is None or earliest <= start_ms:
            break
        cursor_end = earliest - 1
        time.sleep(0.3)

    if not all_rows:
        print("  No new data")
        return

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close"])
    df["dt"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("dt").sort_index().drop(columns=["ts"])
    df = df[~df.index.duplicated(keep="last")]
    df.columns = ["dvol_open", "dvol_high", "dvol_low", "dvol_close"]

    combined = merge_parquet(pq_path, df)
    combined.to_parquet(pq_path)
    print(f"  Updated: {len(combined)} bars, {combined.index.min()} -> {combined.index.max()}")


def run_backfill():
    """Run full backfill: Binance klines + all 14 CG endpoints + Deribit DVOL."""
    backfill_klines()
    backfill_cg(EXISTING_CG, "2/4 existing")
    backfill_cg(NEW_CG, "3/4 new")
    backfill_dvol()
    print("\n" + "=" * 60)
    print("DONE - all parquet files updated")
    print("=" * 60)


def is_stale(max_age_hours: float = 6.0) -> bool:
    """Check if any parquet file is older than max_age_hours."""
    now = pd.Timestamp.now(tz="UTC")
    for pq in RAW.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq)
            if df.empty:
                continue
            age = (now - df.index.max()).total_seconds() / 3600
            if age > max_age_hours:
                return True
        except Exception:
            continue
    # Also check klines exist
    if not (RAW / "binance_klines_1h.parquet").exists():
        return True
    return False


def ensure_fresh(max_age_hours: float = 6.0):
    """Run backfill only if data is stale. Safe to call before training."""
    if is_stale(max_age_hours):
        age = _get_age_hours()
        print(f"Parquet data is {age:.1f}h old (threshold: {max_age_hours}h). Running backfill...")
        run_backfill()
    else:
        age = _get_age_hours()
        print(f"Parquet data is fresh ({age:.1f}h old). Skipping backfill.")


def _get_age_hours() -> float:
    """Return the age of the STALEST parquet file."""
    now = pd.Timestamp.now(tz="UTC")
    max_age = 0.0
    for pq in RAW.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq)
            if df.empty:
                continue
            age = (now - df.index.max()).total_seconds() / 3600
            max_age = max(max_age, age)
        except Exception:
            continue
    return max_age if max_age > 0 else 999.0


if __name__ == "__main__":
    run_backfill()
