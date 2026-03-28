"""
Feature extractor: compute event features from flow_bars_1m.

For each liquidity_event, extracts:
- Pre-sweep flow context (5 min before)
- Post-event flow windows (2h, 4h, 6h)
- Derived features: CVD slope, delta imbalance, price return, divergence, absorption
"""

import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

# Windows to compute (minutes after event)
POST_WINDOWS = {
    "2h": 120,
    "4h": 240,
    "6h": 360,
}

# Pre-sweep lookback (minutes before event)
PRE_LOOKBACK = 5


def extract_features(event: dict) -> dict:
    """
    Given a liquidity_event row (dict), compute all features.

    Returns a flat dict ready for DB insert.
    """
    trigger_ts = event["trigger_ts"]  # unix seconds
    entry_price = float(event["entry_price"])
    symbol = event.get("symbol", "BTC")
    # Map to canonical symbol
    canonical = "BTC-USD" if "BTC" in symbol.upper() else "ETH-USD"

    features = {
        "event_uuid": event["event_uuid"],
        "symbol": symbol,
        "liquidity_side": event["liquidity_side"],
        "entry_price": entry_price,
        "trigger_ts": trigger_ts,
        "session": event.get("session"),
    }

    # ── Pre-sweep flow ──
    pre = _query_flow_window(canonical, trigger_ts, -PRE_LOOKBACK, 0)
    features["pre_delta_usd"] = pre.get("delta_usd") if pre else None
    features["pre_volume_usd"] = pre.get("volume_usd") if pre else None
    features["pre_buy_sell_ratio"] = pre.get("buy_sell_ratio") if pre else None
    features["pre_cvd_usd"] = pre.get("cvd_usd") if pre else None
    features["pre_trade_count"] = pre.get("trade_count") if pre else None

    # ── Post-event flow windows ──
    for label, minutes in POST_WINDOWS.items():
        post = _query_flow_window(canonical, trigger_ts, 0, minutes)
        prefix = f"post_{label}_"
        features[prefix + "delta_usd"] = post.get("delta_usd") if post else None
        features[prefix + "volume_usd"] = post.get("volume_usd") if post else None
        features[prefix + "buy_sell_ratio"] = post.get("buy_sell_ratio") if post else None
        features[prefix + "cvd_usd"] = post.get("cvd_usd") if post else None
        features[prefix + "trade_count"] = post.get("trade_count") if post else None

    # ── Derived features ──
    features["cvd_slope_2h"] = _cvd_slope(features, "2h", 120)
    features["cvd_slope_4h"] = _cvd_slope(features, "4h", 240)
    features["delta_imbalance_2h"] = _delta_imbalance(features, "2h")
    features["delta_imbalance_4h"] = _delta_imbalance(features, "4h")

    # Price returns from flow_bars_1m (approximate: last bar's implied price)
    for label, minutes in POST_WINDOWS.items():
        ret = _price_return_from_bars(canonical, trigger_ts, minutes, entry_price)
        features[f"price_return_{label}"] = ret

    # Divergence: price moves one way, delta moves the other
    features["delta_divergence_2h"] = _detect_divergence(
        features.get("price_return_2h"), features.get("post_2h_delta_usd"))
    features["delta_divergence_4h"] = _detect_divergence(
        features.get("price_return_4h"), features.get("post_4h_delta_usd"))

    # Absorption: large delta but small price move
    features["absorption_detected"] = _detect_absorption(
        features.get("price_return_2h"), features.get("post_2h_delta_usd"),
        features.get("post_2h_volume_usd"))

    # ── Reserved fields (OI / liquidation / orderbook) ──
    for field in ("oi_change_2h", "oi_change_4h",
                  "liq_buy_usd_2h", "liq_sell_usd_2h",
                  "liq_buy_usd_4h", "liq_sell_usd_4h",
                  "orderbook_imbalance_2h", "orderbook_imbalance_4h"):
        features[field] = None

    # ── Actual outcome label (from liquidity_events) ──
    features["label"] = event.get("result_4h") or event.get("result_1h")

    return features


def _query_flow_window(canonical: str, trigger_ts: int,
                       offset_start_min: int, offset_end_min: int) -> dict | None:
    """
    Query flow_bars_1m for a window relative to trigger_ts.

    offset_start_min/offset_end_min: minutes relative to trigger_ts.
    Negative = before event, positive = after.
    """
    start_ms = (trigger_ts + offset_start_min * 60) * 1000
    end_ms = (trigger_ts + offset_end_min * 60) * 1000

    sql = """
    SELECT
        COUNT(*) AS bar_count,
        COALESCE(SUM(buy_notional_usd), 0) AS total_buy,
        COALESCE(SUM(sell_notional_usd), 0) AS total_sell,
        COALESCE(SUM(delta_usd), 0) AS total_delta,
        COALESCE(SUM(volume_usd), 0) AS total_volume,
        COALESCE(SUM(trade_count), 0) AS total_trades
    FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s
      AND window_start < %s
      AND exchange_scope = 'all'
    """

    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, start_ms, end_ms))
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to query flow window")
        return None

    if not row or row["bar_count"] == 0:
        return None

    total_buy = float(row["total_buy"])
    total_sell = float(row["total_sell"])
    total_volume = total_buy + total_sell

    # Get latest CVD in window
    cvd = _get_latest_cvd_in_range(canonical, start_ms, end_ms)

    return {
        "bar_count": row["bar_count"],
        "buy_usd": total_buy,
        "sell_usd": total_sell,
        "delta_usd": float(row["total_delta"]),
        "volume_usd": total_volume,
        "trade_count": int(row["total_trades"]),
        "cvd_usd": cvd,
        "buy_sell_ratio": round(total_buy / total_sell, 4) if total_sell > 0 else None,
    }


def _get_latest_cvd_in_range(canonical: str, start_ms: int, end_ms: int) -> float:
    sql = """
    SELECT cvd_usd FROM flow_bars_1m
    WHERE canonical_symbol = %s AND window_start >= %s AND window_start < %s
      AND exchange_scope = 'all'
    ORDER BY window_start DESC LIMIT 1
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, start_ms, end_ms))
                row = cur.fetchone()
                return float(row["cvd_usd"]) if row else 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0


def _cvd_slope(features: dict, window: str, minutes: int) -> float | None:
    """CVD change per minute over the window."""
    cvd = features.get(f"post_{window}_cvd_usd")
    pre_cvd = features.get("pre_cvd_usd")
    if cvd is None or pre_cvd is None or minutes == 0:
        return None
    return round((cvd - pre_cvd) / minutes, 10)


def _delta_imbalance(features: dict, window: str) -> float | None:
    """(buy - sell) / volume ratio. Range: -1 to +1."""
    delta = features.get(f"post_{window}_delta_usd")
    volume = features.get(f"post_{window}_volume_usd")
    if delta is None or volume is None or volume == 0:
        return None
    return round(delta / volume, 4)


def _price_return_from_bars(canonical: str, trigger_ts: int,
                            offset_minutes: int, entry_price: float) -> float | None:
    """
    Approximate price return by looking at the last flow bar's
    delta-implied direction. Falls back to liquidity_events return if available.

    For now, query the average price from normalized_trades around the target time.
    If no data, return None.
    """
    # Use a 1-minute window around the target time
    target_ms = (trigger_ts + offset_minutes * 60) * 1000
    window_ms = 60_000  # 1 minute

    sql = """
    SELECT AVG(price) AS avg_price
    FROM normalized_trades
    WHERE canonical_symbol = %s
      AND ts_exchange >= %s
      AND ts_exchange < %s
    """

    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, target_ms, target_ms + window_ms))
                row = cur.fetchone()
                if row and row["avg_price"] is not None:
                    avg_price = float(row["avg_price"])
                    return round((avg_price - entry_price) / entry_price * 100, 4)
        finally:
            conn.close()
    except Exception:
        pass

    return None


def _detect_divergence(price_return: float | None, delta: float | None) -> bool | None:
    """Price up + delta negative, or price down + delta positive."""
    if price_return is None or delta is None:
        return None
    if abs(price_return) < 0.1:  # too small to be meaningful
        return False
    return (price_return > 0 and delta < 0) or (price_return < 0 and delta > 0)


def _detect_absorption(price_return: float | None, delta: float | None,
                       volume: float | None) -> bool | None:
    """Large delta relative to volume, but small price move."""
    if price_return is None or delta is None or volume is None or volume == 0:
        return None
    imbalance = abs(delta / volume)
    # High imbalance (>0.3) but small price move (<0.3%)
    return imbalance > 0.3 and abs(price_return) < 0.3
