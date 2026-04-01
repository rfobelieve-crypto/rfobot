"""
Build features from live API data (Binance klines + Coinglass).

Replicates the formulas from enhanced_features.py and inject_coinglass.py,
but operates on live DataFrames instead of parquet files.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ZSCORE_WIN = 96   # 24h
SHORT_WIN = 4     # 1h
MED_WIN = 16      # 4h
LONG_WIN = 96     # 24h


def _zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def build_live_features(klines: pd.DataFrame,
                        cg_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build feature DataFrame from live API data.

    Parameters:
        klines: Binance 15m klines (OHLCV + taker_buy_vol + trade_count)
        cg_data: dict from fetch_coinglass() (30m data, will be forward-filled to 15m)

    Returns:
        DataFrame indexed by datetime with all features.
    """
    df = klines[["open", "high", "low", "close", "volume",
                 "taker_buy_vol", "taker_buy_quote", "trade_count"]].copy()

    # ── Kline-derived features ──────────────────────────────────────────────
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change_pct"] = df["close"].pct_change()
    df["realized_vol_20b"] = df["log_return"].rolling(20, min_periods=4).std()
    df["return_skew"] = df["log_return"].rolling(LONG_WIN, min_periods=20).skew()
    df["return_kurtosis"] = df["log_return"].rolling(LONG_WIN, min_periods=20).apply(
        lambda x: x.kurtosis(), raw=False
    )

    # Taker delta
    taker_sell = df["volume"] - df["taker_buy_vol"]
    taker_delta = df["taker_buy_vol"] - taker_sell
    df["taker_delta_ratio"] = taker_delta / df["volume"].replace(0, np.nan)

    df["taker_delta_ma_4h"] = taker_delta.rolling(MED_WIN, min_periods=4).mean()
    df["taker_delta_std_4h"] = taker_delta.rolling(MED_WIN, min_periods=4).std()
    df["taker_delta_ma_24h"] = taker_delta.rolling(LONG_WIN, min_periods=4).mean()
    df["taker_delta_std_24h"] = taker_delta.rolling(LONG_WIN, min_periods=4).std()

    df["volume_ma_4h"] = df["volume"].rolling(MED_WIN, min_periods=4).mean()
    df["volume_ma_24h"] = df["volume"].rolling(LONG_WIN, min_periods=4).mean()

    # Return lags
    for lag in range(1, 11):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    # ── Coinglass features ──────────────────────────────────────────────────
    _inject_coinglass(df, cg_data)

    logger.info("Live features: %d bars x %d columns", len(df), len(df.columns))
    return df


def _inject_coinglass(df: pd.DataFrame, cg_data: dict[str, pd.DataFrame]):
    """Inject Coinglass features into df (in-place), forward-filling 30m to 15m."""

    def _merge_cg(cg_df: pd.DataFrame, cols: dict[str, str]):
        """Merge Coinglass df into main df using merge_asof + ffill."""
        if cg_df.empty:
            for target in cols.values():
                df[target] = np.nan
            return
        for src, target in cols.items():
            if src in cg_df.columns:
                merged = pd.merge_asof(
                    df[["close"]].reset_index(),
                    cg_df[[src]].reset_index().rename(columns={src: target}),
                    left_on="dt", right_on="dt", direction="backward"
                ).set_index("dt")
                df[target] = merged[target].ffill(limit=4)
            else:
                df[target] = np.nan

    # OI (per-exchange)
    oi = cg_data.get("oi", pd.DataFrame())
    _merge_cg(oi, {"c": "cg_oi_close"})
    if "cg_oi_close" in df.columns:
        df["cg_oi_delta"] = df["cg_oi_close"].diff()
        df["cg_oi_accel"] = df["cg_oi_delta"].diff()

    # OI aggregated
    oi_agg = cg_data.get("oi_agg", pd.DataFrame())
    _merge_cg(oi_agg, {"c": "cg_oi_agg_close"})
    if "cg_oi_agg_close" in df.columns:
        df["cg_oi_agg_delta"] = df["cg_oi_agg_close"].diff()
    if "cg_oi_close" in df.columns and "cg_oi_agg_close" in df.columns:
        df["cg_oi_binance_share"] = df["cg_oi_close"] / df["cg_oi_agg_close"].replace(0, np.nan)

    # Liquidation
    liq = cg_data.get("liquidation", pd.DataFrame())
    liq_cols = {}
    for src, tgt in [("longVolUsd", "cg_liq_long"), ("shortVolUsd", "cg_liq_short"),
                     ("volUsd", "cg_liq_total")]:
        liq_cols[src] = tgt
    _merge_cg(liq, liq_cols)
    if "cg_liq_long" in df.columns and "cg_liq_short" in df.columns:
        total = (df["cg_liq_long"] + df["cg_liq_short"]).replace(0, np.nan)
        df["cg_liq_ratio"] = df["cg_liq_long"] / total
        df["cg_liq_imbalance"] = (df["cg_liq_long"] - df["cg_liq_short"]) / total

    # Long/Short ratio
    ls = cg_data.get("long_short", pd.DataFrame())
    ls_cols = {}
    for src, tgt in [("longRatio", "cg_ls_long_pct"), ("shortRatio", "cg_ls_short_pct")]:
        ls_cols[src] = tgt
    _merge_cg(ls, ls_cols)
    if "cg_ls_long_pct" in df.columns and "cg_ls_short_pct" in df.columns:
        df["cg_ls_ratio"] = df["cg_ls_long_pct"] / df["cg_ls_short_pct"].replace(0, np.nan)

    # Funding rate
    funding = cg_data.get("funding", pd.DataFrame())
    _merge_cg(funding, {"c": "cg_funding_close"})
    if "h" in (funding.columns if not funding.empty else []) and "l" in (funding.columns if not funding.empty else []):
        _merge_cg(funding, {"h": "_fh", "l": "_fl"})
        df["cg_funding_range"] = df.get("_fh", 0) - df.get("_fl", 0)
        df.drop(columns=["_fh", "_fl"], errors="ignore", inplace=True)
    else:
        df["cg_funding_range"] = np.nan

    # Taker buy/sell
    taker = cg_data.get("taker", pd.DataFrame())
    taker_cols = {}
    for src, tgt in [("buyVolUsd", "cg_taker_buy"), ("sellVolUsd", "cg_taker_sell")]:
        taker_cols[src] = tgt
    _merge_cg(taker, taker_cols)
    if "cg_taker_buy" in df.columns and "cg_taker_sell" in df.columns:
        df["cg_taker_delta"] = df["cg_taker_buy"] - df["cg_taker_sell"]
        total = (df["cg_taker_buy"] + df["cg_taker_sell"]).replace(0, np.nan)
        df["cg_taker_ratio"] = df["cg_taker_buy"] / total

    # Z-scores
    for col in ["cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
                "cg_ls_ratio", "cg_taker_delta", "cg_funding_close"]:
        if col in df.columns:
            df[f"{col}_zscore"] = _zscore(df[col])

    # Cross features
    if "cg_liq_total" in df.columns and "cg_oi_close" in df.columns:
        df["cg_liq_x_oi"] = df["cg_liq_total"] / df["cg_oi_close"].replace(0, np.nan)
    if "cg_funding_close" in df.columns and "cg_ls_ratio" in df.columns:
        df["cg_crowding"] = df["cg_funding_close"] * df["cg_ls_ratio"]
    if "cg_taker_delta_zscore" in df.columns and "cg_oi_delta_zscore" in df.columns:
        df["cg_conviction"] = df["cg_taker_delta_zscore"] * df["cg_oi_delta_zscore"]
