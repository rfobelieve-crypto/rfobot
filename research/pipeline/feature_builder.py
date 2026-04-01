"""
LAYER 2: feature_builder.py

Builds feature_bars_15m by merging three raw sources:
  1. flow_bars_15m  — taker flow (delta, CVD, volume)
  2. oi_snapshots   — open interest (ASOF join: last snapshot <= bar close)
  3. funding_rates  — funding rate  (ASOF join: last rate <= bar close)

Also fetches missing OHLCV from Binance klines for bars without price data.

No lookahead: every feature uses only data available at bar close.

Usage:
    python -m research.pipeline.feature_builder
    python -m research.pipeline.feature_builder --symbol BTC-USD
"""
from __future__ import annotations

import sys
import os
import time
import logging
import argparse

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

BAR_MS     = 15 * 60 * 1000
ZSCORE_WIN = 48          # 48 × 15m = 12 hours
BATCH_SIZE = 500

SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}


# ─── Raw loaders ──────────────────────────────────────────────────────────────

def _load_flow(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT window_start, window_end,
                       buy_notional_usd, sell_notional_usd,
                       delta_usd, volume_usd, trade_count, cvd_usd,
                       bar_open, bar_high, bar_low, bar_close, source
                FROM flow_bars_15m
                WHERE canonical_symbol = %s
                ORDER BY window_start
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df = df.rename(columns={"window_start": "bucket_15m"})
    df["delta_ratio"] = np.where(
        df["volume_usd"] > 0,
        df["delta_usd"] / df["volume_usd"],
        0.0,
    )
    return df


def _load_oi(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_exchange, oi_notional_usd
                FROM oi_snapshots
                WHERE canonical_symbol = %s AND exchange = 'binance'
                ORDER BY ts_exchange
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df = df.rename(columns={"oi_notional_usd": "oi_usd"})
    # Deduplicate by ts_exchange, keep last
    df = df.drop_duplicates("ts_exchange", keep="last")
    return df


def _load_funding(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_exchange, funding_rate
                FROM funding_rates
                WHERE canonical_symbol = %s AND exchange = 'binance'
                ORDER BY ts_exchange
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df = df.drop_duplicates("ts_exchange", keep="last")
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df


# ─── OHLCV fetch from Binance ─────────────────────────────────────────────────

def _fetch_ohlcv_binance(binance_sym: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 15m OHLCV klines from Binance USDT-M futures."""
    url    = "https://fapi.binance.com/fapi/v1/klines"
    limit  = 1500
    rows   = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            resp = requests.get(url, params={
                "symbol":    binance_sym,
                "interval":  "15m",
                "startTime": cursor,
                "endTime":   end_ms,
                "limit":     limit,
            }, timeout=15)
            resp.raise_for_status()
            page = resp.json()
        except Exception:
            logger.exception("OHLCV fetch failed for %s", binance_sym)
            break
        if not page:
            break
        rows.extend(page)
        last_open = int(page[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + BAR_MS
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["open_time", "open", "high", "low", "close"]].copy()
    df = df.rename(columns={"open_time": "bucket_15m"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["bucket_15m"] = df["bucket_15m"].astype(np.int64)
    return df


# ─── Core build logic ─────────────────────────────────────────────────────────

def build(symbol: str = "BTC-USD", fetch_ohlcv: bool = True) -> int:
    logger.info("Building feature_bars_15m for %s", symbol)

    flow_df = _load_flow(symbol)
    if flow_df.empty:
        logger.warning("No flow data for %s", symbol)
        return 0

    oi_df      = _load_oi(symbol)
    funding_df = _load_funding(symbol)

    # ── 1. OI: ASOF join (last OI snapshot <= bar close) ──────────────────────
    flow_df["window_end_ms"] = flow_df["bucket_15m"] + BAR_MS

    if not oi_df.empty:
        oi_df["oi_usd"] = oi_df["oi_usd"].astype(float)
        oi_df = oi_df.sort_values("ts_exchange")
        flow_df = flow_df.sort_values("window_end_ms")
        flow_df = pd.merge_asof(
            flow_df, oi_df,
            left_on="window_end_ms", right_on="ts_exchange",
            direction="backward",
        )
        flow_df["oi_usd"] = flow_df["oi_usd"].astype(float)
        flow_df["oi_change_usd"] = flow_df["oi_usd"].diff()
        flow_df["oi_change_pct"] = np.where(
            flow_df["oi_usd"].shift(1) > 0,
            flow_df["oi_change_usd"] / flow_df["oi_usd"].shift(1),
            np.nan,
        )
    else:
        flow_df["oi_usd"] = np.nan
        flow_df["oi_change_usd"] = np.nan
        flow_df["oi_change_pct"] = np.nan

    # ── 2. Funding: ASOF join + rolling zscore ────────────────────────────────
    if not funding_df.empty:
        funding_df = funding_df.sort_values("ts_exchange")
        flow_df = pd.merge_asof(
            flow_df, funding_df,
            left_on="window_end_ms", right_on="ts_exchange",
            direction="backward",
        )
        roll_mean = flow_df["funding_rate"].rolling(ZSCORE_WIN, min_periods=4).mean()
        roll_std  = flow_df["funding_rate"].rolling(ZSCORE_WIN, min_periods=4).std()
        flow_df["funding_zscore"] = np.where(
            roll_std > 0,
            (flow_df["funding_rate"] - roll_mean) / roll_std,
            0.0,
        )
    else:
        flow_df["funding_rate"]   = np.nan
        flow_df["funding_zscore"] = np.nan

    # ── 3. OHLCV: fetch missing from Binance ──────────────────────────────────
    # Reset index after merges so boolean mask aligns correctly
    flow_df = flow_df.reset_index(drop=True)
    if fetch_ohlcv:
        missing_mask = flow_df["bar_close"].isna()
        if missing_mask.any():
            miss_min = int(flow_df.loc[missing_mask, "bucket_15m"].min())
            miss_max = int(flow_df.loc[missing_mask, "bucket_15m"].max()) + BAR_MS
            binance_sym = SYMBOL_MAP.get(symbol)
            if binance_sym:
                logger.info("Fetching OHLCV for %d missing bars from Binance...",
                            missing_mask.sum())
                ohlcv = _fetch_ohlcv_binance(binance_sym, miss_min, miss_max)
                if not ohlcv.empty:
                    ohlcv_lookup = ohlcv.set_index("bucket_15m")
                    for col, src in [("bar_open","open"),("bar_high","high"),
                                     ("bar_low","low"),("bar_close","close")]:
                        flow_df[col] = flow_df["bucket_15m"].map(
                            ohlcv_lookup[src]
                        ).combine_first(flow_df[col])

    # ── 4. Write to feature_bars_15m ──────────────────────────────────────────
    flow_df = flow_df.sort_values("bucket_15m").reset_index(drop=True)
    now_ms  = int(time.time() * 1000)

    sql = """
        INSERT INTO feature_bars_15m
            (symbol, bucket_15m,
             buy_notional, sell_notional, delta_usd, volume_usd,
             delta_ratio, cvd_usd, trade_count,
             bar_open, bar_high, bar_low, bar_close,
             oi_usd, oi_change_usd, oi_change_pct,
             funding_rate, funding_zscore,
             data_source, computed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            buy_notional   = VALUES(buy_notional),
            sell_notional  = VALUES(sell_notional),
            delta_usd      = VALUES(delta_usd),
            volume_usd     = VALUES(volume_usd),
            delta_ratio    = VALUES(delta_ratio),
            cvd_usd        = VALUES(cvd_usd),
            trade_count    = VALUES(trade_count),
            bar_open       = VALUES(bar_open),
            bar_high       = VALUES(bar_high),
            bar_low        = VALUES(bar_low),
            bar_close      = VALUES(bar_close),
            oi_usd         = VALUES(oi_usd),
            oi_change_usd  = VALUES(oi_change_usd),
            oi_change_pct  = VALUES(oi_change_pct),
            funding_rate   = VALUES(funding_rate),
            funding_zscore = VALUES(funding_zscore),
            computed_at    = VALUES(computed_at)
    """

    def _v(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    total = 0
    for i in range(0, len(flow_df), BATCH_SIZE):
        chunk  = flow_df.iloc[i: i + BATCH_SIZE]
        params = []
        for r in chunk.itertuples():
            params.append((
                symbol, int(r.bucket_15m),
                _v(r.buy_notional_usd), _v(r.sell_notional_usd),
                _v(r.delta_usd), _v(r.volume_usd),
                _v(r.delta_ratio), _v(r.cvd_usd),
                int(r.trade_count) if r.trade_count and not (isinstance(r.trade_count, float) and np.isnan(r.trade_count)) else None,
                _v(getattr(r, "bar_open", None)), _v(getattr(r, "bar_high", None)),
                _v(getattr(r, "bar_low",  None)), _v(getattr(r, "bar_close", None)),
                _v(getattr(r, "oi_usd",   None)), _v(getattr(r, "oi_change_usd", None)),
                _v(getattr(r, "oi_change_pct", None)),
                _v(getattr(r, "funding_rate", None)),
                _v(getattr(r, "funding_zscore", None)),
                str(r.source) if hasattr(r, "source") and r.source else "unknown",
                now_ms,
            ))
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
            conn.commit()
            total += len(params)
        except Exception:
            logger.exception("Batch write failed at row %d", i)
        finally:
            conn.close()

    logger.info("feature_bars_15m: %d rows written for %s", total, symbol)
    return total


def run(symbols: list[str] | None = None):
    targets = symbols or ["BTC-USD", "ETH-USD"]
    for sym in targets:
        try:
            build(sym)
        except Exception:
            logger.exception("feature_builder failed for %s", sym)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="e.g. BTC-USD")
    args = ap.parse_args()
    run([args.symbol] if args.symbol else None)
