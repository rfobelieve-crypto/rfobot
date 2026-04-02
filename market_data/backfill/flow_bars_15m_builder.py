"""
Build historical flow_bars_15m from two sources:

  1. CSV backfill  — market_data/aggtrades_data/*.csv (Binance BTCUSD_PERP coin-M)
                     Each contract = $100 USD (inverse perp)
  2. DB rollup     — aggregate existing flow_bars_1m → 15m (recent live data)

Output: flow_bars_15m table (canonical_symbol=BTC-USD, exchange_scope depends on source)

Usage:
    python -m market_data.backfill.flow_bars_15m_builder            # both sources
    python -m market_data.backfill.flow_bars_15m_builder --csv-only
    python -m market_data.backfill.flow_bars_15m_builder --db-only
"""
from __future__ import annotations

import sys
import os
import glob
import time
import logging
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

CSV_DIR        = os.path.join(os.path.dirname(__file__), "../../market_data/aggtrades_data")
BAR_MS         = 15 * 60 * 1000          # 15 minutes in ms
CONTRACT_USD   = 100                      # BTCUSD_PERP: 1 contract = $100
CANONICAL      = "BTC-USD"
SCOPE_CSV      = "binance_coinm"          # source tag for CSV-built bars
SCOPE_LIVE     = "all"                    # source tag for live DB rollup
BATCH_SIZE     = 1000


# ─────────────────────────────────────────────────────────────────────────────
# CSV → flow_bars_15m
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
        dtype={"price": "float64", "quantity": "float64",
               "transact_time": "int64", "is_buyer_maker": "bool"},
    )


def build_from_csv() -> int:
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not files:
        logger.warning("No CSV files found in %s", CSV_DIR)
        return 0

    logger.info("Loading %d CSV files...", len(files))
    chunks = []
    for f in files:
        logger.info("  %s", os.path.basename(f))
        chunks.append(_load_csv(f))

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    logger.info("Loaded %d trades, aggregating to 15m bars...", len(df))

    # Assign to 15m bucket
    df["bucket"] = (df["transact_time"] // BAR_MS) * BAR_MS

    # notional_usd: BTCUSD_PERP = quantity (contracts) * $100
    df["notional"] = df["quantity"] * CONTRACT_USD
    df["buy_n"]    = df["notional"].where(~df["is_buyer_maker"], 0.0)
    df["sell_n"]   = df["notional"].where( df["is_buyer_maker"], 0.0)

    g = df.groupby("bucket")
    bars = pd.DataFrame({
        "window_start":      g["bucket"].first(),
        "bar_open":          g["price"].first(),
        "bar_high":          g["price"].max(),
        "bar_low":           g["price"].min(),
        "bar_close":         g["price"].last(),
        "buy_notional_usd":  g["buy_n"].sum(),
        "sell_notional_usd": g["sell_n"].sum(),
        "trade_count":       g["price"].count(),
    }).reset_index(drop=True)

    bars["window_end"]  = bars["window_start"] + BAR_MS
    bars["delta_usd"]   = bars["buy_notional_usd"] - bars["sell_notional_usd"]
    bars["volume_usd"]  = bars["buy_notional_usd"] + bars["sell_notional_usd"]
    bars["cvd_usd"]     = bars["delta_usd"].cumsum()
    bars = bars.sort_values("window_start").reset_index(drop=True)

    logger.info("Built %d 15m bars, writing to DB...", len(bars))
    return _upsert_bars(bars, CANONICAL, SCOPE_CSV, "csv_backfill")


# ─────────────────────────────────────────────────────────────────────────────
# flow_bars_1m → flow_bars_15m (DB rollup for live data)
# ─────────────────────────────────────────────────────────────────────────────

def build_from_db(symbol: str = CANONICAL) -> int:
    logger.info("Rolling up flow_bars_1m → flow_bars_15m for %s...", symbol)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Only rollup complete 15m buckets not yet in flow_bars_15m
            cur.execute("""
                SELECT
                    (window_start DIV %s) * %s  AS bucket,
                    SUM(buy_notional_usd)        AS buy_notional_usd,
                    SUM(sell_notional_usd)       AS sell_notional_usd,
                    SUM(delta_usd)               AS delta_usd,
                    SUM(volume_usd)              AS volume_usd,
                    SUM(trade_count)             AS trade_count,
                    MAX(cvd_usd)                 AS cvd_usd,
                    COUNT(*)                     AS bar_count
                FROM flow_bars_1m
                WHERE canonical_symbol = %s
                  AND exchange_scope   = %s
                  AND (window_start DIV %s) * %s NOT IN (
                      SELECT window_start FROM flow_bars_15m
                      WHERE canonical_symbol = %s AND exchange_scope = %s
                  )
                GROUP BY bucket
                HAVING bar_count = 15
                ORDER BY bucket
            """, (BAR_MS, BAR_MS, symbol, SCOPE_LIVE,
                  BAR_MS, BAR_MS, symbol, SCOPE_LIVE))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No new complete 15m buckets to rollup.")
        return 0

    bars = pd.DataFrame([dict(r) for r in rows])
    bars = bars.rename(columns={"bucket": "window_start"})
    bars["window_end"] = bars["window_start"] + BAR_MS
    bars["bar_open"]   = None
    bars["bar_high"]   = None
    bars["bar_low"]    = None
    bars["bar_close"]  = None

    logger.info("Rolling up %d buckets from flow_bars_1m", len(bars))
    return _upsert_bars(bars, symbol, SCOPE_LIVE, "live")


# ─────────────────────────────────────────────────────────────────────────────
# DB write
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_bars(bars: pd.DataFrame, canonical: str, scope: str, source: str) -> int:
    sql = """
        INSERT INTO flow_bars_15m
            (canonical_symbol, exchange_scope, window_start, window_end,
             buy_notional_usd, sell_notional_usd, delta_usd, volume_usd,
             trade_count, cvd_usd, bar_open, bar_high, bar_low, bar_close, source)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            buy_notional_usd  = VALUES(buy_notional_usd),
            sell_notional_usd = VALUES(sell_notional_usd),
            delta_usd         = VALUES(delta_usd),
            volume_usd        = VALUES(volume_usd),
            trade_count       = VALUES(trade_count),
            cvd_usd           = VALUES(cvd_usd),
            bar_open          = VALUES(bar_open),
            bar_high          = VALUES(bar_high),
            bar_low           = VALUES(bar_low),
            bar_close         = VALUES(bar_close)
    """
    total = 0
    for i in range(0, len(bars), BATCH_SIZE):
        chunk = bars.iloc[i: i + BATCH_SIZE]
        params = [
            (
                canonical, scope,
                int(r.window_start), int(r.window_end),
                float(r.buy_notional_usd), float(r.sell_notional_usd),
                float(r.delta_usd), float(r.volume_usd),
                int(r.trade_count), float(r.cvd_usd),
                float(r.bar_open)  if r.bar_open  is not None and pd.notna(r.bar_open)  else None,
                float(r.bar_high)  if r.bar_high  is not None and pd.notna(r.bar_high)  else None,
                float(r.bar_low)   if r.bar_low   is not None and pd.notna(r.bar_low)   else None,
                float(r.bar_close) if r.bar_close is not None and pd.notna(r.bar_close) else None,
                source,
            )
            for r in chunk.itertuples()
        ]
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
            conn.commit()
            total += len(params)
            logger.info("  Upserted %d / %d bars", total, len(bars))
        except Exception:
            logger.exception("Batch upsert failed at row %d", i)
        finally:
            conn.close()

    logger.info("Done: %d rows written to flow_bars_15m [%s %s]", total, canonical, scope)
    return total


# ─────────────────────────────────────────────────────────────────────────────

def run(csv: bool = True, db: bool = True):
    total = 0
    if csv:
        total += build_from_csv()
    if db:
        total += build_from_db()
    logger.info("flow_bars_15m build complete. Total rows written: %d", total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--db-only",  action="store_true")
    args = ap.parse_args()

    do_csv = not args.db_only
    do_db  = not args.csv_only
    run(csv=do_csv, db=do_db)
