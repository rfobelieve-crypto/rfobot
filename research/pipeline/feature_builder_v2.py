"""
LAYER 2 (v2): feature_builder_v2.py

Builds features_{5m,15m,1h} by resampling 1m sources and computing:
  - OHLCV + returns + realized vol
  - Signed flow + CVD (daily reset, Taipei tz) + z-score  (USDT-M source)
  - OI columns: all NaN (historical OI not available; collect realtime)
  - Funding rate / deviation / z-score  (ASOF join, Binance 8h)
  - Cross features: funding×CVD  (OI cross features set to NaN)
  - EMA(9/21) + MACD
  - Return & delta-ratio lags (1–10)
  - Forward-looking labels (up/down/neutral, 0.20% threshold)
  - bull_bear_score (0–100): 0.5×delta_ratio + 0.3×cvd + 0.2×funding

No lookahead: all features use data available at bar close.
Labels are forward-looking and intended for offline training only.

Usage:
    python -m research.pipeline.feature_builder_v2
    python -m research.pipeline.feature_builder_v2 --timeframe 5m
    python -m research.pipeline.feature_builder_v2 --timeframe 15m
    python -m research.pipeline.feature_builder_v2 --timeframe 1h
"""
from __future__ import annotations

import sys
import os
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.db import get_db_conn

logger = logging.getLogger(__name__)

LABEL_THRESHOLD = 0.002   # 0.20%
LARGE_THRESHOLD = 100_000.0
ZSCORE_WIN      = 60
BATCH           = 500

TF_CONFIG = {
    "5m":  {"resample": "5min",  "ms": 5  * 60 * 1000},
    "15m": {"resample": "15min", "ms": 15 * 60 * 1000},
    "1h":  {"resample": "1h",    "ms": 60 * 60 * 1000},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _zscore(s: pd.Series, window: int = ZSCORE_WIN) -> pd.Series:
    m   = s.rolling(window, min_periods=4).mean()
    std = s.rolling(window, min_periods=4).std()
    return pd.Series(np.where(std > 0, (s - m) / std, 0.0), index=s.index)


def _cvd_daily_reset(delta: pd.Series, tz: str = "Asia/Taipei") -> pd.Series:
    """Cumulative delta with daily reset at local midnight."""
    local_dates = delta.index.tz_convert(tz).date
    result = pd.Series(np.nan, index=delta.index, dtype=float)
    for d in pd.unique(local_dates):
        mask = local_dates == d
        result[mask] = delta[mask].fillna(0).cumsum().values
    return result


def _label(ret: pd.Series, threshold: float = LABEL_THRESHOLD) -> pd.Series:
    labels = pd.Series("neutral", index=ret.index)
    labels[ret >= threshold]  = "up"
    labels[ret <= -threshold] = "down"
    labels[ret.isna()]        = None
    return labels


def _compute_strength_labels(df: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
    """
    Compute 4-hour forward strength labels.  No lookahead bias.

    Window size is auto-derived from bar size:
      15m → 16 bars,  1h → 4 bars,  5m → 48 bars

    Algorithm (per bar t):
      1. future_high_4h = max(high[t+1 .. t+window])   via sliding_window_view
      2. future_low_4h  = min(low[t+1  .. t+window])
      3. up_move_4h     = (future_high - close) / close        (≥ 0)
      4. down_move_4h   = (future_low  - close) / close        (≤ 0)
      5. vol_4h_proxy   = rolling 14-bar ATR% (past-only)
      6. up_move_vol_adj   = up_move_4h / vol
      7. down_move_vol_adj = abs(down_move_4h) / vol
      8. strength_raw      = up_move_4h + down_move_4h  (net direction, + = bullish)
      9. strength_vol_adj  = up_move_vol_adj - down_move_vol_adj

    The last `window` rows will have NaN labels (insufficient future data).
    """
    window = max(1, int(4 * 3_600_000 // tf_ms))
    n      = len(df)

    h_arr = df["high"].values.astype(float)
    l_arr = df["low"].values.astype(float)

    future_high = np.full(n, np.nan)
    future_low  = np.full(n, np.nan)

    if n > window:
        # sliding_window_view(arr, W+1)[i] = arr[i : i+W+1]
        # future slice for bar t = arr[t+1 : t+W+1] = view[t][:, 1:]
        win_h = sliding_window_view(h_arr, window_shape=window + 1)  # (n-window, window+1)
        win_l = sliding_window_view(l_arr, window_shape=window + 1)
        valid = n - window
        future_high[:valid] = win_h[:, 1:].max(axis=1)
        future_low[:valid]  = win_l[:, 1:].min(axis=1)

    df = df.copy()
    df["future_high_4h"] = future_high
    df["future_low_4h"]  = future_low

    # ── Moves relative to current close ───────────────────────────────────────
    df["up_move_4h"]   = (df["future_high_4h"] - df["close"]) / df["close"]
    df["down_move_4h"] = (df["future_low_4h"]  - df["close"]) / df["close"]

    # ── Volatility proxy: ATR% rolling 14-bar (past-only, no lookahead) ──────
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["vol_4h_proxy"] = (tr / df["close"]).rolling(14, min_periods=4).mean()

    # ── Vol-adjusted moves ────────────────────────────────────────────────────
    safe_vol = df["vol_4h_proxy"].replace(0.0, np.nan)
    df["up_move_vol_adj"]   = df["up_move_4h"]          / safe_vol
    df["down_move_vol_adj"] = df["down_move_4h"].abs()   / safe_vol

    # ── Net strength ──────────────────────────────────────────────────────────
    # strength_raw > 0 → upside > downside (bullish potential)
    # strength_raw < 0 → downside > upside (bearish potential)
    df["strength_raw"]     = df["up_move_4h"] + df["down_move_4h"]
    df["strength_vol_adj"] = df["up_move_vol_adj"] - df["down_move_vol_adj"]

    return df


def _v(x):
    """Convert to float, return None for NaN."""
    if x is None:
        return None
    try:
        f = float(x)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _load_ohlcv(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_open, open, high, low, close, volume, quote_vol,
                       trade_count, taker_buy_vol, taker_buy_quote
                FROM ohlcv_1m WHERE symbol = %s ORDER BY ts_open
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts_open"] = df["ts_open"].astype(np.int64)
    for col in ["open","high","low","close","volume","quote_vol",
                "taker_buy_vol","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(int)
    idx = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    idx.name = None   # prevent "ts_open" index/column ambiguity
    df.index = idx
    return df


def _load_flow(symbol: str, exchange: str = "all") -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_open, buy_vol, sell_vol, delta, volume,
                       large_buy_vol, large_sell_vol,
                       trade_count, buy_count, sell_count
                FROM flow_bars_1m_ml WHERE symbol = %s AND exchange = %s ORDER BY ts_open
            """, (symbol, exchange))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts_open"] = df["ts_open"].astype(np.int64)
    for col in df.columns:
        if col != "ts_open":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    idx = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    idx.name = None
    df.index = idx
    return df


def _load_oi(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_exchange, oi_notional_usd
                FROM oi_snapshots WHERE canonical_symbol = %s AND exchange = 'binance'
                ORDER BY ts_exchange
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts_exchange"]      = df["ts_exchange"].astype(np.int64)
    df["oi_notional_usd"]  = pd.to_numeric(df["oi_notional_usd"], errors="coerce")
    df = df.drop_duplicates("ts_exchange").sort_values("ts_exchange")
    return df


def _load_funding(symbol: str) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts_exchange, funding_rate
                FROM funding_rates WHERE canonical_symbol = %s
                ORDER BY ts_exchange
            """, (symbol,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts_exchange"]  = df["ts_exchange"].astype(np.int64)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.drop_duplicates("ts_exchange").sort_values("ts_exchange")
    return df


# ─── Core builder ─────────────────────────────────────────────────────────────

def build(symbol: str = "BTC-USD", timeframe: str = "15m") -> int:
    cfg      = TF_CONFIG[timeframe]
    rule     = cfg["resample"]
    tf_ms    = cfg["ms"]
    table    = f"features_{timeframe}"

    logger.info("Building %s for %s (timeframe=%s)", table, symbol, timeframe)

    ohlcv_1m    = _load_ohlcv(symbol)
    flow_1m     = _load_flow(symbol, exchange="all")
    flow_bnc_1m = _load_flow(symbol, exchange="binance")
    flow_okx_1m = _load_flow(symbol, exchange="okx")
    fund_df     = _load_funding(symbol)

    if ohlcv_1m.empty:
        logger.warning("No ohlcv_1m data for %s", symbol)
        return 0

    # ── Resample OHLCV ────────────────────────────────────────────────────────
    ohlcv_tf = ohlcv_1m.resample(rule).agg(
        open        = ("open",          "first"),
        high        = ("high",          "max"),
        low         = ("low",           "min"),
        close       = ("close",         "last"),
        volume      = ("volume",        "sum"),
        quote_vol   = ("quote_vol",     "sum"),
        trade_count = ("trade_count",   "sum"),
        taker_buy_vol   = ("taker_buy_vol",   "sum"),
        taker_buy_quote = ("taker_buy_quote", "sum"),
    ).dropna(subset=["close"])

    # ── Resample Flow ─────────────────────────────────────────────────────────
    if not flow_1m.empty:
        flow_tf = flow_1m.resample(rule).agg(
            buy_vol        = ("buy_vol",       "sum"),
            sell_vol       = ("sell_vol",      "sum"),
            delta          = ("delta",         "sum"),
            volume_flow    = ("volume",        "sum"),
            large_buy_vol  = ("large_buy_vol", "sum"),
            large_sell_vol = ("large_sell_vol","sum"),
            flow_trades    = ("trade_count",   "sum"),
            buy_count      = ("buy_count",     "sum"),
            sell_count     = ("sell_count",    "sum"),
        )
        flow_tf["delta_ratio"] = np.where(
            flow_tf["volume_flow"] > 0,
            flow_tf["delta"] / flow_tf["volume_flow"],
            np.nan,
        )
        df = ohlcv_tf.join(flow_tf, how="left")
    else:
        logger.warning("No flow_bars_1m_ml data — flow features will be NaN")
        df = ohlcv_tf.copy()
        for col in ["buy_vol","sell_vol","delta","volume_flow","delta_ratio",
                    "large_buy_vol","large_sell_vol","flow_trades","buy_count","sell_count"]:
            df[col] = np.nan

    # ── Price features ────────────────────────────────────────────────────────
    df["return_1b"]        = df["close"].pct_change()
    df["realized_vol_20b"] = df["return_1b"].rolling(20, min_periods=4).std()

    # ── CVD daily reset ───────────────────────────────────────────────────────
    df["cvd"]        = _cvd_daily_reset(df["delta"].fillna(0))
    df["cvd_zscore"] = _zscore(df["cvd"])
    df["large_delta"] = df["large_buy_vol"] - df["large_sell_vol"]

    # window_end_ms = bar close time in ms (used for funding ASOF join)
    window_end_ms = pd.Series(
        (df.index.astype(np.int64) // 1_000_000 + tf_ms).astype(np.int64),
        index=df.index, name="window_end_ms",
    )

    # ── Cross-exchange divergence ─────────────────────────────────────────────
    # binance_delta_ratio and okx_delta_ratio resampled independently
    if not flow_bnc_1m.empty:
        bnc_tf = flow_bnc_1m.resample(rule).agg(
            bnc_buy_vol  = ("buy_vol",  "sum"),
            bnc_sell_vol = ("sell_vol", "sum"),
            bnc_volume   = ("volume",   "sum"),
        )
        df["bnc_delta_ratio"] = np.where(
            bnc_tf["bnc_volume"] > 0,
            (bnc_tf["bnc_buy_vol"] - bnc_tf["bnc_sell_vol"]) / bnc_tf["bnc_volume"],
            np.nan,
        )
    else:
        df["bnc_delta_ratio"] = np.nan

    if not flow_okx_1m.empty:
        okx_tf = flow_okx_1m.resample(rule).agg(
            okx_buy_vol  = ("buy_vol",  "sum"),
            okx_sell_vol = ("sell_vol", "sum"),
            okx_volume   = ("volume",   "sum"),
        )
        df["okx_delta_ratio"] = np.where(
            okx_tf["okx_volume"] > 0,
            (okx_tf["okx_buy_vol"] - okx_tf["okx_sell_vol"]) / okx_tf["okx_volume"],
            np.nan,
        )
    else:
        df["okx_delta_ratio"] = np.nan

    # divergence: positive = Binance more bullish than OKX, negative = OKX leads
    df["exchange_divergence"] = df["bnc_delta_ratio"] - df["okx_delta_ratio"]

    # ── OI features: NaN (historical OI not available; collect realtime) ────────
    for col in ["oi", "oi_delta", "oi_accel", "oi_divergence"]:
        df[col] = np.nan

    # ── Funding ASOF join ─────────────────────────────────────────────────────
    if not fund_df.empty:
        left = window_end_ms.to_frame().sort_values("window_end_ms")
        merged = pd.merge_asof(
            left,
            fund_df[["ts_exchange","funding_rate"]],
            left_on="window_end_ms", right_on="ts_exchange",
            direction="backward",
        )
        df["funding_rate"]      = merged["funding_rate"].values.astype(float)
        df["funding_deviation"] = (
            df["funding_rate"] - df["funding_rate"].rolling(ZSCORE_WIN, min_periods=4).mean()
        )
        df["funding_zscore"]    = _zscore(df["funding_rate"])
    else:
        for col in ["funding_rate","funding_deviation","funding_zscore"]:
            df[col] = np.nan

    # ── Cross features ────────────────────────────────────────────────────────
    df["cvd_x_oi_delta"] = np.nan   # OI not available
    df["funding_x_cvd"]  = df["funding_zscore"] * df["cvd_zscore"]
    df["cvd_oi_ratio"]   = np.nan   # OI not available


    # ── Lags ──────────────────────────────────────────────────────────────────
    for i in range(1, 11):
        df[f"return_lag_{i}"] = df["return_1b"].shift(i)
        df[f"delta_lag_{i}"]  = df["delta_ratio"].shift(i)

    # ── Forward labels (lookahead — training only) ────────────────────────────
    bars_5m  = max(1, 5  // (tf_ms // 60_000))
    bars_15m = max(1, 15 // (tf_ms // 60_000))
    bars_1h  = max(1, 60 // (tf_ms // 60_000))
    df["future_return_5m"]  = df["close"].shift(-bars_5m)  / df["close"] - 1
    df["future_return_15m"] = df["close"].shift(-bars_15m) / df["close"] - 1
    df["future_return_1h"]  = df["close"].shift(-bars_1h)  / df["close"] - 1
    df["label_5m"]  = _label(df["future_return_5m"])
    df["label_15m"] = _label(df["future_return_15m"])
    df["label_1h"]  = _label(df["future_return_1h"])

    # ── 4h Strength labels (lookahead — training only) ───────────────────────
    df = _compute_strength_labels(df, tf_ms)

    # ── bull_bear_score (0–100) ───────────────────────────────────────────────
    # Weights: 0.5 delta_ratio + 0.3 cvd + 0.2 funding  (OI removed)
    raw = (
        0.5 * df["delta_ratio"].fillna(0).clip(-1, 1)
      + 0.3 * np.tanh(df["cvd_zscore"].fillna(0) / 2)
      + 0.2 * np.tanh(-df["funding_zscore"].fillna(0) / 2)
    )
    df["bull_bear_score"] = ((raw + 1) / 2 * 100).clip(0, 100)

    # ── ts_open (ms) ──────────────────────────────────────────────────────────
    df["ts_open"] = (df.index.astype(np.int64) // 1_000_000).astype(np.int64)
    df = df.sort_values("ts_open").reset_index(drop=True)

    # ── Write to DB ───────────────────────────────────────────────────────────
    now_ms = int(time.time() * 1000)

    cols = [
        "symbol","ts_open",
        "open","high","low","close","volume","return_1b","realized_vol_20b",
        "buy_vol","sell_vol","delta","delta_ratio",
        "cvd","cvd_zscore","large_buy_vol","large_sell_vol","large_delta",
        "oi","oi_delta","oi_accel","oi_divergence",
        "funding_rate","funding_deviation","funding_zscore",
        "cvd_x_oi_delta","funding_x_cvd","cvd_oi_ratio",
        "bnc_delta_ratio","okx_delta_ratio","exchange_divergence",
        *[f"return_lag_{i}" for i in range(1, 11)],
        *[f"delta_lag_{i}"  for i in range(1, 11)],
        "future_return_5m","future_return_15m","future_return_1h",
        "label_5m","label_15m","label_1h",
        "future_high_4h","future_low_4h",
        "up_move_4h","down_move_4h","vol_4h_proxy",
        "up_move_vol_adj","down_move_vol_adj",
        "strength_raw","strength_vol_adj",
        "bull_bear_score","computed_at",
    ]

    placeholders = ",".join(["%s"] * len(cols))
    update_cols  = [c for c in cols if c not in ("symbol","ts_open")]
    updates      = ", ".join(f"{c} = VALUES({c})" for c in update_cols)
    sql = f"""
        INSERT INTO {table} ({",".join(cols)})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {updates}
    """

    label_cols = {"label_5m","label_15m","label_1h"}

    def _row(r, symbol: str, now_ms: int) -> tuple:
        row = [symbol, int(r.ts_open)]
        for c in cols[2:]:
            if c == "computed_at":
                row.append(now_ms)
            elif c in label_cols:
                v = getattr(r, c, None)
                row.append(str(v) if v and not (isinstance(v, float) and np.isnan(v)) else None)
            else:
                row.append(_v(getattr(r, c, None)))
        return tuple(row)

    total = 0
    for i in range(0, len(df), BATCH):
        chunk  = df.iloc[i: i + BATCH]
        params = [_row(r, symbol, now_ms) for r in chunk.itertuples()]
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

    logger.info("%s: %d rows written for %s", table, total, symbol)
    return total


def run(symbols: list[str] | None = None,
        timeframes: list[str] | None = None):
    targets = symbols    or ["BTC-USD"]
    tfs     = timeframes or ["5m", "15m", "1h"]
    for sym in targets:
        for tf in tfs:
            try:
                build(sym, tf)
            except Exception:
                logger.exception("feature_builder_v2 failed for %s %s", sym, tf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",    default=None)
    ap.add_argument("--timeframe", default=None, choices=["5m","15m","1h"])
    args = ap.parse_args()
    run(
        [args.symbol]    if args.symbol    else None,
        [args.timeframe] if args.timeframe else None,
    )
