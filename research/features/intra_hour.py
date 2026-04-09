"""
Intra-Hour Features from flow_bars_1m — 補充 Coinglass 看不到的分鐘級訂單流特徵。

利用 market_data 服務已在寫入的 1min flow bars，聚合成 H1 特徵：
  1. delta_flip_freq_1h    — 過去 1h 內 delta 方向翻轉次數（市場猶豫度）
  2. cvd_accel_15m         — 最近 15min CVD 加速度 vs 前 45min
  3. volume_front_load     — 前 30min 成交量佔 1h 比例（前重後輕 = 動能衰減）
  4. large_delta_ratio_1h  — 大單 delta 佔總 delta 比例（機構 vs 散戶）
  5. max_1m_delta_zscore   — 1h 內最大單分鐘 delta 的 z-score（爆量偵測）
  6. buy_sell_count_ratio  — 買單筆數 / 賣單筆數（筆數不平衡）

數據源：Railway MySQL flow_bars_1m_ml (舊) + flow_bars_1m (新)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_conn():
    """Get DB connection — try shared.db first, fallback to direct pymysql."""
    try:
        from shared.db import get_db_conn
        return get_db_conn()
    except Exception:
        pass
    # Fallback: direct connection (works on local Python 3.9)
    import pymysql
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # Try Railway external first, then local
    for host, port, user, pw, db in [
        (os.getenv("MYSQLHOST", "caboose.proxy.rlwy.net"),
         int(os.getenv("MYSQLPORT", "18766")),
         os.getenv("MYSQLUSER", "root"),
         os.getenv("MYSQLPASSWORD", os.getenv("MYSQL_PASSWORD", "")),
         os.getenv("MYSQLDATABASE", os.getenv("MYSQL_DB", "railway"))),
    ]:
        try:
            return pymysql.connect(
                host=host, port=port, user=user, password=pw, database=db,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception:
            continue
    raise ConnectionError("Cannot connect to MySQL")


def fetch_flow_bars_1h(lookback_hours: int = 500) -> pd.DataFrame:
    """
    從 Railway MySQL 讀取 BTC 1min flow bars，聚合成 1h。

    嘗試讀兩張表（舊 + 新），合併去重。
    Returns: DataFrame indexed by hour (UTC), columns = intra-hour features.
    """
    try:
        conn = _get_conn()
    except Exception as e:
        logger.warning("DB connection failed for intra-hour features: %s", e)
        return pd.DataFrame()

    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp() * 1000)

    all_rows = []
    try:
        with conn.cursor() as cur:
            # Old table (richer: has large_buy/sell, buy_count/sell_count)
            try:
                cur.execute("""
                    SELECT ts_open as window_start, delta, volume, delta_ratio,
                           buy_vol, sell_vol, large_buy_vol, large_sell_vol,
                           trade_count, buy_count, sell_count
                    FROM flow_bars_1m_ml
                    WHERE symbol = 'BTC-USD' AND ts_open >= %s
                    ORDER BY ts_open
                """, (cutoff_ms,))
                old_rows = cur.fetchall()
                if old_rows:
                    df_old = pd.DataFrame(old_rows)
                    df_old["source"] = "old"
                    all_rows.append(df_old)
                    logger.info("flow_bars_1m_ml: %d rows", len(df_old))
            except Exception as e:
                logger.debug("Old table read failed: %s", e)

            # New table
            try:
                cur.execute("""
                    SELECT window_start, delta_usd as delta, volume_usd as volume,
                           buy_notional_usd as buy_vol, sell_notional_usd as sell_vol,
                           trade_count, cvd_usd
                    FROM flow_bars_1m
                    WHERE canonical_symbol = 'BTC-USD' AND window_start >= %s
                    ORDER BY window_start
                """, (cutoff_ms,))
                new_rows = cur.fetchall()
                if new_rows:
                    df_new = pd.DataFrame(new_rows)
                    df_new["source"] = "new"
                    # New table doesn't have large_buy/sell, buy_count/sell_count
                    for col in ["large_buy_vol", "large_sell_vol", "buy_count", "sell_count", "delta_ratio"]:
                        if col not in df_new.columns:
                            df_new[col] = np.nan
                    all_rows.append(df_new)
                    logger.info("flow_bars_1m: %d rows", len(df_new))
            except Exception as e:
                logger.debug("New table read failed: %s", e)
    finally:
        conn.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    # Convert to float
    for col in ["delta", "volume", "buy_vol", "sell_vol", "large_buy_vol",
                 "large_sell_vol", "trade_count", "buy_count", "sell_count", "delta_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert timestamp to datetime
    df["dt"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    df = df.sort_values("dt").drop_duplicates(subset=["window_start"], keep="last")

    # Assign hour bucket
    df["hour"] = df["dt"].dt.floor("h")

    # ── Aggregate per hour ──
    result = _aggregate_hourly(df)
    return result


def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    從 1min bars 聚合成每小時的 intra-hour 特徵。
    """
    features = []

    for hour, group in df.groupby("hour"):
        if len(group) < 10:  # 至少 10 min bars
            continue

        delta = group["delta"].values
        volume = group["volume"].values
        n = len(group)

        feat = {"dt": hour}

        # 1. Delta Flip Frequency — 方向翻轉次數 / 總 bars
        #    高 = 市場猶豫，方向不斷切換
        #    低 = 方向一致，趨勢明確
        signs = np.sign(delta)
        signs_nonzero = signs[signs != 0]
        if len(signs_nonzero) > 1:
            flips = np.sum(signs_nonzero[1:] != signs_nonzero[:-1])
            feat["ih_delta_flip_freq"] = flips / len(signs_nonzero)
        else:
            feat["ih_delta_flip_freq"] = 0

        # 2. CVD Acceleration — 最近 15min CVD vs 前 45min CVD
        #    正 = 買方加速（最近更猛）
        #    負 = 賣方加速
        cvd = np.cumsum(delta)
        if n >= 15:
            recent_15 = cvd[-1] - cvd[-min(15, n)]
            earlier = cvd[-min(15, n)] - cvd[0] if n > 15 else 0
            total_vol = volume.sum()
            if total_vol > 0:
                feat["ih_cvd_accel_15m"] = (recent_15 - earlier) / total_vol
            else:
                feat["ih_cvd_accel_15m"] = 0
        else:
            feat["ih_cvd_accel_15m"] = 0

        # 3. Volume Front Load — 前 30min 成交量 / 整小時
        #    高 = 動能集中在前半（可能衰減）
        #    低 = 動能集中在後半（可能延續）
        half = n // 2
        if half > 0 and volume.sum() > 0:
            feat["ih_volume_front_load"] = volume[:half].sum() / volume.sum()
        else:
            feat["ih_volume_front_load"] = 0.5

        # 4. Large Delta Ratio — 大單 delta / 總 delta
        #    高 = 機構主導方向
        #    低 = 散戶為主
        large_buy = group["large_buy_vol"].values
        large_sell = group["large_sell_vol"].values
        if not np.all(np.isnan(large_buy)):
            lb = np.nansum(large_buy)
            ls = np.nansum(large_sell)
            total_d = abs(delta.sum())
            if total_d > 0:
                feat["ih_large_delta_ratio"] = (lb - ls) / total_d
            else:
                feat["ih_large_delta_ratio"] = 0
        else:
            feat["ih_large_delta_ratio"] = np.nan

        # 5. Max 1min Delta Z-score — 最大單分鐘 delta / 全小時 std
        #    高 = 有一分鐘爆量（sweep 或大單）
        delta_std = delta.std()
        if delta_std > 0:
            feat["ih_max_1m_delta_z"] = delta[np.argmax(np.abs(delta))] / delta_std
        else:
            feat["ih_max_1m_delta_z"] = 0

        # 6. Buy/Sell Count Ratio — 買單筆數 / 賣單筆數
        #    >1 = 買方筆數多（散戶看多）
        #    <1 = 賣方筆數多
        bc = group["buy_count"].values
        sc = group["sell_count"].values
        if not np.all(np.isnan(bc)):
            total_buy = np.nansum(bc)
            total_sell = np.nansum(sc)
            if total_sell > 0:
                feat["ih_buy_sell_count_ratio"] = total_buy / total_sell
            else:
                feat["ih_buy_sell_count_ratio"] = 1.0
        else:
            feat["ih_buy_sell_count_ratio"] = np.nan

        features.append(feat)

    if not features:
        return pd.DataFrame()

    result = pd.DataFrame(features)
    result["dt"] = pd.to_datetime(result["dt"], utc=True)
    result = result.set_index("dt").sort_index()

    # Replace inf/nan safety
    for col in result.columns:
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    return result
