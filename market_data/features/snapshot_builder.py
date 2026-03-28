"""
Snapshot feature builder: extract features for a specific time window.

For each (event, snapshot_type), queries flow_bars_1m to compute:
- delta_value: net delta in the window
- cvd_change: CVD change from trigger to window end
- cvd_sign_flip: whether CVD flipped to reversal direction
- price_change_pct: % price change from entry
- reclaim_flag: price reclaimed sweep_ref level
- break_again_flag: price broke further past entry
"""

import time
import logging
from shared.db import get_db_conn

logger = logging.getLogger(__name__)


def build_snapshot(event: dict, snapshot_type: str, offset_sec: int) -> dict:
    """
    Build feature snapshot for a given event and time window.

    Args:
        event: row from event_registry
        snapshot_type: '15m', '1h', or '4h'
        offset_sec: window duration in seconds (900, 3600, 14400)

    Returns:
        dict ready for snapshot_repository.save_snapshot()
    """
    trigger_ts = int(event["trigger_ts"])
    entry_price = float(event["entry_price"])
    symbol = event["symbol"]
    liquidity_side = event["liquidity_side"].lower()
    sweep_ref_price = float(event["sweep_ref_price"]) if event.get("sweep_ref_price") else None

    # Map to canonical symbol
    canonical = "BTC-USD" if "BTC" in symbol.upper() else "ETH-USD"

    # Query flow_bars_1m for the window
    start_ms = trigger_ts * 1000
    end_ms = (trigger_ts + offset_sec) * 1000
    flow = _query_flow_window(canonical, start_ms, end_ms)

    # Delta value
    delta_value = flow["delta_usd"] if flow else None

    # CVD change: difference between last CVD in window and CVD at trigger
    cvd_at_trigger = _get_cvd_at(canonical, start_ms)
    cvd_at_end = _get_cvd_at(canonical, end_ms)
    cvd_change = None
    if cvd_at_trigger is not None and cvd_at_end is not None:
        cvd_change = cvd_at_end - cvd_at_trigger

    # CVD sign flip: did CVD change direction toward reversal?
    cvd_sign_flip = None
    if cvd_change is not None:
        if liquidity_side == "sell":
            cvd_sign_flip = cvd_change > 0  # SSL: positive CVD = reversal
        elif liquidity_side == "buy":
            cvd_sign_flip = cvd_change < 0  # BSL: negative CVD = reversal

    # Price change %
    price_change_pct = _get_price_change(canonical, trigger_ts, offset_sec, entry_price)

    # Reclaim flag: price moved back past sweep_ref_price
    reclaim_flag = None
    if sweep_ref_price and price_change_pct is not None:
        price_now = entry_price * (1 + price_change_pct / 100)
        if liquidity_side == "sell":
            reclaim_flag = price_now > sweep_ref_price
        elif liquidity_side == "buy":
            reclaim_flag = price_now < sweep_ref_price

    # Break again flag: price continued past entry in sweep direction
    break_again_flag = None
    if price_change_pct is not None:
        if liquidity_side == "sell":
            break_again_flag = price_change_pct < -0.1  # further down
        elif liquidity_side == "buy":
            break_again_flag = price_change_pct > 0.1   # further up

    # Label: only 4h gets ground truth from liquidity_events
    label = None
    if snapshot_type == "4h":
        label = _get_label(event["event_uuid"])

    # ── OI features ──
    oi = _compute_oi_features(canonical, trigger_ts, trigger_ts + offset_sec)

    # ── Funding rate at trigger time ──
    funding_rate = _get_funding_at(canonical, trigger_ts)

    # ── Liquidation data in window ──
    liq = _get_liquidation_window(canonical, trigger_ts, trigger_ts + offset_sec)

    return {
        "event_uuid": event["event_uuid"],
        "event_type": event.get("event_type"),
        "canonical_symbol": canonical,
        "liquidity_side": event["liquidity_side"],
        "trigger_price": entry_price,
        "trigger_ts": trigger_ts,
        "snapshot_type": snapshot_type,
        "snapshot_ts": int(time.time()),

        "delta_value": delta_value,
        "cvd_change": cvd_change,
        "cvd_sign_flip": cvd_sign_flip,
        "price_change_pct": price_change_pct,
        "reclaim_flag": reclaim_flag,
        "break_again_flag": break_again_flag,

        # OI
        "oi_baseline_okx": oi.get("oi_baseline_okx"),
        "oi_baseline_binance": oi.get("oi_baseline_binance"),
        "oi_snapshot_okx": oi.get("oi_snapshot_okx"),
        "oi_snapshot_binance": oi.get("oi_snapshot_binance"),
        "oi_change_okx": oi.get("oi_change_okx"),
        "oi_change_binance": oi.get("oi_change_binance"),
        "oi_change_okx_pct": oi.get("oi_change_okx_pct"),
        "oi_change_binance_pct": oi.get("oi_change_binance_pct"),
        "oi_change_total": oi.get("oi_change_total"),
        "oi_change_total_pct": oi.get("oi_change_total_pct"),

        # Funding rate
        "funding_rate": funding_rate,

        # Liquidations in window
        "liq_buy_usd":   liq.get("liq_buy_usd"),
        "liq_sell_usd":  liq.get("liq_sell_usd"),
        "liq_total_usd": liq.get("liq_total_usd"),
        "liq_count":     liq.get("liq_count"),

        "label": label,
    }


def _query_flow_window(canonical: str, start_ms: int, end_ms: int) -> dict | None:
    """Aggregate flow_bars_1m for a time range."""
    sql = """
    SELECT
        COALESCE(SUM(delta_usd), 0) AS delta_usd,
        COALESCE(SUM(volume_usd), 0) AS volume_usd,
        COALESCE(SUM(buy_notional_usd), 0) AS buy_usd,
        COALESCE(SUM(sell_notional_usd), 0) AS sell_usd,
        COUNT(*) AS bar_count
    FROM flow_bars_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s AND window_start < %s
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

    return {
        "delta_usd": float(row["delta_usd"]),
        "volume_usd": float(row["volume_usd"]),
        "buy_usd": float(row["buy_usd"]),
        "sell_usd": float(row["sell_usd"]),
        "bar_count": row["bar_count"],
    }


def _get_cvd_at(canonical: str, target_ms: int) -> float | None:
    """Get the CVD value at or just before a given timestamp."""
    sql = """
    SELECT cvd_usd FROM flow_bars_1m
    WHERE canonical_symbol = %s AND window_start <= %s AND exchange_scope = 'all'
    ORDER BY window_start DESC LIMIT 1
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, target_ms))
                row = cur.fetchone()
                return float(row["cvd_usd"]) if row else None
        finally:
            conn.close()
    except Exception:
        return None


def _get_price_change(canonical: str, trigger_ts: int,
                      offset_sec: int, entry_price: float) -> float | None:
    """Get % price change at the end of the window.

    Primary: normalized_trades (accurate, 3d retention).
    Fallback: delta/volume ratio from flow_bars_1m (90d retention).
    """
    target_ms = (trigger_ts + offset_sec) * 1000
    window_ms = 60_000  # ±1 minute

    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                # 1) Try normalized_trades first (works for events < 3 days)
                cur.execute("""
                    SELECT AVG(price) AS avg_price
                    FROM normalized_trades
                    WHERE canonical_symbol = %s
                      AND ts_exchange >= %s AND ts_exchange < %s
                """, (canonical, target_ms - window_ms, target_ms + window_ms))
                row = cur.fetchone()
                if row and row["avg_price"] is not None:
                    avg_price = float(row["avg_price"])
                    return round((avg_price - entry_price) / entry_price * 100, 4)

                # 2) Fallback: estimate from flow_bars_1m delta/volume ratio
                trigger_ms = trigger_ts * 1000
                cur.execute("""
                    SELECT COALESCE(SUM(delta_usd), 0) AS total_delta,
                           COALESCE(SUM(volume_usd), 0) AS total_vol
                    FROM flow_bars_1m
                    WHERE canonical_symbol = %s
                      AND exchange_scope = 'all'
                      AND window_start >= %s AND window_start < %s
                """, (canonical, trigger_ms, target_ms + 60_000))
                delta_row = cur.fetchone()
                if delta_row and float(delta_row["total_vol"] or 0) > 0:
                    total_delta = float(delta_row["total_delta"])
                    total_vol = float(delta_row["total_vol"])
                    estimated_pct = round(total_delta / total_vol * 100, 4)
                    return estimated_pct
        finally:
            conn.close()
    except Exception:
        pass
    return None


def _get_label(event_uuid: str) -> str | None:
    """Get ground truth label — check event_registry first, fallback to liquidity_events."""
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                # Prefer event_registry (unified lifecycle)
                cur.execute(
                    "SELECT result_4h, result_1h FROM event_registry WHERE event_uuid = %s LIMIT 1",
                    (event_uuid,),
                )
                row = cur.fetchone()
                if row:
                    label = row.get("result_4h") or row.get("result_1h")
                    if label:
                        return label

                # Fallback: liquidity_events (backward compat)
                cur.execute(
                    "SELECT result_4h, result_1h FROM liquidity_events WHERE event_uuid = %s LIMIT 1",
                    (event_uuid,),
                )
                row = cur.fetchone()
                if row:
                    return row.get("result_4h") or row.get("result_1h")
        finally:
            conn.close()
    except Exception:
        pass
    return None


def _get_oi_near(canonical: str, exchange: str, ts_unix: int) -> float | None:
    """Get the closest OI notional_usd near a given unix timestamp."""
    ts_ms = ts_unix * 1000
    # Search within ±5 minutes
    window_ms = 5 * 60 * 1000
    sql = """
    SELECT oi_notional_usd, ts_exchange,
           ABS(ts_exchange - %s) AS dist
    FROM oi_snapshots
    WHERE canonical_symbol = %s AND exchange = %s
      AND ts_exchange BETWEEN %s AND %s
    ORDER BY dist ASC
    LIMIT 1
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (ts_ms, canonical, exchange,
                                  ts_ms - window_ms, ts_ms + window_ms))
                row = cur.fetchone()
                if row:
                    return float(row["oi_notional_usd"])
        finally:
            conn.close()
    except Exception:
        logger.debug("OI lookup failed for %s %s @ %d", exchange, canonical, ts_unix)
    return None


def _compute_oi_features(canonical: str, trigger_ts: int, snapshot_ts: int) -> dict:
    """
    Compute OI baseline/snapshot/change for OKX + Binance.
    Returns dict with all oi_* fields (None if data missing).
    """
    result = {}

    total_baseline = 0.0
    total_snapshot = 0.0
    has_any = False

    for exch in ("okx", "binance"):
        baseline = _get_oi_near(canonical, exch, trigger_ts)
        snapshot = _get_oi_near(canonical, exch, snapshot_ts)

        result[f"oi_baseline_{exch}"] = baseline
        result[f"oi_snapshot_{exch}"] = snapshot

        if baseline is not None and snapshot is not None and baseline > 0:
            change = snapshot - baseline
            change_pct = round(change / baseline * 100, 4)
            result[f"oi_change_{exch}"] = round(change, 4)
            result[f"oi_change_{exch}_pct"] = change_pct
            total_baseline += baseline
            total_snapshot += snapshot
            has_any = True
        else:
            result[f"oi_change_{exch}"] = None
            result[f"oi_change_{exch}_pct"] = None
            if baseline is None or snapshot is None:
                logger.debug("OI missing for %s %s (baseline=%s, snapshot=%s)",
                             exch, canonical, baseline, snapshot)

    if has_any and total_baseline > 0:
        total_change = total_snapshot - total_baseline
        result["oi_change_total"] = round(total_change, 4)
        result["oi_change_total_pct"] = round(total_change / total_baseline * 100, 4)
    else:
        result["oi_change_total"] = None
        result["oi_change_total_pct"] = None

    return result


def _get_funding_at(canonical: str, ts_unix: int) -> float | None:
    """Get the most recent funding rate at or before a given unix timestamp."""
    ts_ms = ts_unix * 1000
    sql = """
    SELECT funding_rate
    FROM funding_rates
    WHERE canonical_symbol = %s AND ts_exchange <= %s
    ORDER BY ts_exchange DESC LIMIT 1
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, ts_ms))
                row = cur.fetchone()
                if row:
                    return float(row["funding_rate"])
        finally:
            conn.close()
    except Exception:
        logger.debug("Funding rate lookup failed for %s @ %d", canonical, ts_unix)
    return None


def _get_liquidation_window(canonical: str, start_ts: int, end_ts: int) -> dict:
    """Sum liquidation buckets in a time window (unix seconds)."""
    start_ms = start_ts * 1000
    end_ms   = end_ts   * 1000
    sql = """
    SELECT
        COALESCE(SUM(liq_buy_usd),   0) AS liq_buy_usd,
        COALESCE(SUM(liq_sell_usd),  0) AS liq_sell_usd,
        COALESCE(SUM(liq_total_usd), 0) AS liq_total_usd,
        COALESCE(SUM(liq_count),     0) AS liq_count
    FROM liquidation_1m
    WHERE canonical_symbol = %s
      AND window_start >= %s AND window_start < %s
    """
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (canonical, start_ms, end_ms))
                row = cur.fetchone()
                if row and float(row["liq_total_usd"]) > 0:
                    return {
                        "liq_buy_usd":   float(row["liq_buy_usd"]),
                        "liq_sell_usd":  float(row["liq_sell_usd"]),
                        "liq_total_usd": float(row["liq_total_usd"]),
                        "liq_count":     int(row["liq_count"]),
                    }
        finally:
            conn.close()
    except Exception:
        logger.debug("Liquidation window lookup failed for %s", canonical)
    return {
        "liq_buy_usd":   None,
        "liq_sell_usd":  None,
        "liq_total_usd": None,
        "liq_count":     None,
    }
