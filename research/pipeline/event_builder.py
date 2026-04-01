"""
LAYER 3: event_builder.py

Constructs event_features from:
  - liquidity_events  (raw sweep events from TradingView)
  - feature_bars_15m  (clean aligned features, Layer 2)

Each event row includes:
  - Pre-event features at T-1, T-2, T-4 (bars strictly before trigger)
  - Interaction features
  - Post-event outcomes at T+1, T+2, T+4
  - Label (reversal / continuation / neutral)

Label logic (strict, no ambiguity):
  BSL: return_4bar <= -0.5% → reversal  |  >= +0.5% → continuation
  SSL: return_4bar >= +0.5% → reversal  |  <= -0.5% → continuation

Usage:
    python -m research.pipeline.event_builder
"""
from __future__ import annotations

import sys
import os
import time
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

BAR_MS           = 15 * 60 * 1000
LABEL_THRESHOLD  = 0.005   # 0.5%
CANONICAL_MAP    = {
    "BTCUSDT.P": "BTC-USD",
    "ETHUSDT.P": "ETH-USD",
    "BTCUSDT":   "BTC-USD",
    "ETHUSDT":   "ETH-USD",
    "BTC-USD":   "BTC-USD",
    "ETH-USD":   "ETH-USD",
}
SIDE_MAP = {"buy": "BSL", "sell": "SSL"}


# ─── Load raw events ──────────────────────────────────────────────────────────

def _load_events() -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT event_uuid, symbol, liquidity_side,
                       trigger_ts, entry_price
                FROM liquidity_events
                ORDER BY trigger_ts
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["canonical_symbol"] = df["symbol"].map(CANONICAL_MAP)
    df["event_side"]       = df["liquidity_side"].map(SIDE_MAP)
    # Use int64 explicitly — Windows numpy default int is int32 which overflows
    df["trigger_ts_ms"]  = df["trigger_ts"].astype("int64") * np.int64(1000)
    df["trigger_bucket"] = (df["trigger_ts_ms"] // np.int64(BAR_MS)) * np.int64(BAR_MS)
    df["entry_price"]      = df["entry_price"].astype(float)
    return df[df["canonical_symbol"].notna() & df["event_side"].notna()]


# ─── Load feature bars ────────────────────────────────────────────────────────

def _load_features(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT bucket_15m, delta_ratio, oi_change_pct,
                       funding_rate, funding_zscore,
                       bar_open, bar_close
                FROM feature_bars_15m
                WHERE symbol = %s
                ORDER BY bucket_15m
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.set_index("bucket_15m")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Feature extraction ───────────────────────────────────────────────────────

def _get_bar(feat: pd.DataFrame, bucket: int) -> dict:
    """Return feature bar dict, or empty dict if not found."""
    if bucket in feat.index:
        return feat.loc[bucket].to_dict()
    return {}


def _safe(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _label(event_side: str, return_4bar: float | None) -> str | None:
    if return_4bar is None:
        return None
    if event_side == "BSL":
        if return_4bar <= -LABEL_THRESHOLD:
            return "reversal"
        if return_4bar >= LABEL_THRESHOLD:
            return "continuation"
    elif event_side == "SSL":
        if return_4bar >= LABEL_THRESHOLD:
            return "reversal"
        if return_4bar <= -LABEL_THRESHOLD:
            return "continuation"
    return "neutral"


# ─── Main build ───────────────────────────────────────────────────────────────

def build() -> int:
    events = _load_events()
    if events.empty:
        logger.info("No events in liquidity_events.")
        return 0

    # Get already-processed event_uuids
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT event_uuid FROM event_features")
            done = {r["event_uuid"] for r in cur.fetchall()}
    finally:
        conn.close()

    new_events = events[~events["event_uuid"].isin(done)]
    if new_events.empty:
        logger.info("All %d events already processed.", len(events))
        return 0

    logger.info("Processing %d new events...", len(new_events))

    # Load features per symbol
    feat_cache: dict[str, pd.DataFrame] = {}
    rows_to_insert = []
    now_ms = int(time.time() * 1000)

    for _, ev in new_events.iterrows():
        sym    = ev["canonical_symbol"]
        side   = ev["event_side"]
        bucket = int(ev["trigger_bucket"])
        price  = ev["entry_price"]

        if sym not in feat_cache:
            feat_cache[sym] = _load_features(sym)
        feat = feat_cache[sym]

        if feat.empty:
            logger.warning("No feature_bars_15m data for %s, skipping event %s",
                           sym, ev["event_uuid"][:8])
            continue

        # Pre-event bars (strictly before trigger bucket)
        b1 = bucket - BAR_MS       # T-1
        b2 = bucket - 2 * BAR_MS   # T-2
        b4 = bucket - 4 * BAR_MS   # T-4

        f1 = _get_bar(feat, b1)
        f2 = _get_bar(feat, b2)
        f4 = _get_bar(feat, b4)

        delta_1bar   = _safe(f1, "delta_ratio")
        delta_2bar   = _safe(f2, "delta_ratio")
        delta_4bar   = _safe(f4, "delta_ratio")
        oi_chg_1bar  = _safe(f1, "oi_change_pct")
        oi_chg_2bar  = _safe(f2, "oi_change_pct")
        fund_rate    = _safe(f1, "funding_rate")
        fund_z       = _safe(f1, "funding_zscore")

        # Interaction features
        pressure = (
            delta_1bar * oi_chg_1bar
            if delta_1bar is not None and oi_chg_1bar is not None
            else None
        )

        # delta_price_divergence: sign(delta) × sign(bar_return)
        # if both positive or both negative → aligned (positive)
        # if opposite → diverging (negative)
        bar_ret_1 = None
        if f1.get("bar_open") and f1.get("bar_close"):
            open_, close_ = _safe(f1, "bar_open"), _safe(f1, "bar_close")
            if open_ and close_ and open_ > 0:
                bar_ret_1 = (close_ - open_) / open_
        delta_price_div = (
            np.sign(delta_1bar) * np.sign(bar_ret_1)
            if delta_1bar is not None and bar_ret_1 is not None
            else None
        )

        flow_accel = (
            delta_1bar - delta_2bar
            if delta_1bar is not None and delta_2bar is not None
            else None
        )

        # Post-event outcomes (returns relative to trigger_price)
        def _fwd_return(offset_bars: int) -> float | None:
            b = bucket + offset_bars * BAR_MS
            fb = _get_bar(feat, b)
            close_ = _safe(fb, "bar_close")
            if close_ is None or price <= 0:
                return None
            return (close_ - price) / price

        r1 = _fwd_return(1)
        r2 = _fwd_return(2)
        r4 = _fwd_return(4)

        label = _label(side, r4)

        rows_to_insert.append((
            ev["event_uuid"], sym, side,
            int(ev["trigger_ts_ms"]), float(price), bucket,
            delta_1bar, delta_2bar, delta_4bar,
            oi_chg_1bar, oi_chg_2bar,
            fund_rate, fund_z,
            pressure,
            float(delta_price_div) if delta_price_div is not None else None,
            float(flow_accel) if flow_accel is not None else None,
            r1, r2, r4,
            label, now_ms if label else None,
        ))

    if not rows_to_insert:
        logger.info("No rows to insert.")
        return 0

    sql = """
        INSERT INTO event_features (
            event_uuid, symbol, event_side,
            trigger_ts_ms, trigger_price, trigger_bucket_15m,
            delta_1bar, delta_2bar, delta_4bar,
            oi_change_1bar, oi_change_2bar,
            funding_rate, funding_zscore,
            pressure, delta_price_divergence, flow_acceleration,
            return_1bar, return_2bar, return_4bar,
            label, labeled_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            return_1bar = VALUES(return_1bar),
            return_2bar = VALUES(return_2bar),
            return_4bar = VALUES(return_4bar),
            label       = VALUES(label),
            labeled_at  = VALUES(labeled_at)
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows_to_insert)
        conn.commit()
        logger.info("event_features: %d rows inserted", len(rows_to_insert))
    finally:
        conn.close()

    return len(rows_to_insert)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build()
