"""
CRUD for market_state_bars.
This table is the single source of truth for the visualization layer.
"""
from __future__ import annotations
import logging
import time
import pandas as pd
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

_UPSERT = """
INSERT INTO market_state_bars (
    symbol, timeframe, window_start, window_end,
    delta_usd, delta_ratio, delta_direction,
    volume_usd, rel_volume,
    cvd_change, cvd_slope, cvd_flip,
    oi_change_pct, oi_direction,
    funding_rate, liq_total_usd,
    rolling_mean, rolling_std, upper_band, lower_band, z_score,
    reversal_score, continuation_score, confidence,
    final_bias, risk_adj_score, signal, score_model,
    event_count, computed_at
) VALUES (
    %s,%s,%s,%s, %s,%s,%s, %s,%s, %s,%s,%s,
    %s,%s, %s,%s, %s,%s,%s,%s,%s,
    %s,%s,%s, %s,%s,%s,%s, %s,%s
)
ON DUPLICATE KEY UPDATE
    delta_usd=VALUES(delta_usd), delta_ratio=VALUES(delta_ratio),
    delta_direction=VALUES(delta_direction),
    volume_usd=VALUES(volume_usd), rel_volume=VALUES(rel_volume),
    cvd_change=VALUES(cvd_change), cvd_slope=VALUES(cvd_slope),
    cvd_flip=VALUES(cvd_flip),
    oi_change_pct=VALUES(oi_change_pct), oi_direction=VALUES(oi_direction),
    funding_rate=VALUES(funding_rate), liq_total_usd=VALUES(liq_total_usd),
    rolling_mean=VALUES(rolling_mean), rolling_std=VALUES(rolling_std),
    upper_band=VALUES(upper_band), lower_band=VALUES(lower_band),
    z_score=VALUES(z_score),
    reversal_score=VALUES(reversal_score),
    continuation_score=VALUES(continuation_score),
    confidence=VALUES(confidence), final_bias=VALUES(final_bias),
    risk_adj_score=VALUES(risk_adj_score), signal=VALUES(signal),
    score_model=VALUES(score_model),
    event_count=VALUES(event_count), computed_at=VALUES(computed_at)
"""


def upsert_bar(row: dict):
    """Insert or update one market_state_bars row."""
    params = (
        row["symbol"], row["timeframe"], row["window_start"], row["window_end"],
        row.get("delta_usd"), row.get("delta_ratio"), row.get("delta_direction"),
        row.get("volume_usd"), row.get("rel_volume"),
        row.get("cvd_change"), row.get("cvd_slope"), row.get("cvd_flip"),
        row.get("oi_change_pct"), row.get("oi_direction"),
        row.get("funding_rate"), row.get("liq_total_usd"),
        row.get("rolling_mean"), row.get("rolling_std"),
        row.get("upper_band"), row.get("lower_band"), row.get("z_score"),
        row.get("reversal_score"), row.get("continuation_score"), row.get("confidence"),
        row.get("final_bias"), row.get("risk_adj_score"),
        row.get("signal", 0), row.get("score_model"),
        row.get("event_count", 0), int(time.time()),
    )
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_UPSERT, params)
    except Exception:
        logger.exception("upsert_bar failed @ %s", row.get("window_start"))
    finally:
        conn.close()


def query_bars(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Load market_state_bars for the given range.

    Returns DataFrame sorted by window_start ASC with a 'timestamp' column
    in Asia/Taipei timezone.  Returns empty DataFrame on no data.
    """
    sql = """
    SELECT * FROM market_state_bars
    WHERE symbol=%s AND timeframe=%s
      AND window_start >= %s AND window_start <= %s
    ORDER BY window_start ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, timeframe, start_ms, end_ms))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = (
            pd.to_datetime(df["window_start"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Taipei")
        )
        return df
    except Exception:
        logger.exception("query_bars failed")
        return pd.DataFrame()
    finally:
        conn.close()
