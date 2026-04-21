"""
Shared pytest fixtures for the BTC prediction indicator test suite.

Provides synthetic data fixtures that mirror production data shapes
without requiring live API access or database connections.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    200 rows of synthetic BTC 1h OHLCV data.

    Price range: $65,000-$70,000 with realistic random walk.
    Volume: 500-2000 BTC per bar.
    Includes taker_buy_vol, taker_buy_quote, trade_count, quote_vol
    columns that the feature builder expects.
    """
    np.random.seed(42)
    n = 200
    base_time = pd.Timestamp("2026-04-01", tz="UTC")
    index = pd.date_range(base_time, periods=n, freq="1h")

    # Random walk price
    returns = np.random.normal(0, 0.003, n)
    close = 67_000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = close * (1 + np.random.normal(0, 0.001, n))

    volume = np.random.uniform(500, 2000, n)
    taker_buy_vol = volume * np.random.uniform(0.4, 0.6, n)
    trade_count = np.random.randint(5000, 50000, n).astype(float)
    quote_vol = volume * close

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "taker_buy_vol": taker_buy_vol,
        "taker_buy_quote": taker_buy_vol * close,
        "trade_count": trade_count,
        "quote_vol": quote_vol,
    }, index=index)
    df.index.name = "dt"
    return df


@pytest.fixture
def mock_db_conn():
    """
    Mock pymysql connection returning empty results.

    cursor() returns a context manager whose execute() is a no-op
    and fetchall()/fetchone() return empty list / None.
    """
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """
    200 rows with ~20 sample feature columns at realistic ranges.

    Includes kline-derived features, Coinglass z-scores, and
    regime indicators that the inference engine expects.
    """
    np.random.seed(42)
    n = 200
    base_time = pd.Timestamp("2026-04-01", tz="UTC")
    index = pd.date_range(base_time, periods=n, freq="1h")

    close = 67_000 * np.exp(np.cumsum(np.random.normal(0, 0.003, n)))

    df = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.001, n)),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.random.uniform(500, 2000, n),
        "log_return": np.random.normal(0, 0.003, n),
        "realized_vol_20b": np.abs(np.random.normal(0.005, 0.002, n)),
        "return_skew": np.random.normal(0, 0.5, n),
        "return_kurtosis": np.random.normal(3, 1, n),
        "taker_delta_ratio": np.random.normal(0, 0.1, n),
        "cg_oi_delta_zscore": np.random.normal(0, 1, n),
        "cg_funding_close_zscore": np.random.normal(0, 1, n),
        "cg_taker_delta_zscore": np.random.normal(0, 1, n),
        "cg_ls_ratio_zscore": np.random.normal(0, 1, n),
        "cg_ls_divergence_zscore": np.random.normal(0, 1, n),
        "is_trending_bull": np.random.choice([0.0, 1.0], n, p=[0.8, 0.2]),
        "is_trending_bear": np.random.choice([0.0, 1.0], n, p=[0.8, 0.2]),
        "vol_acceleration": np.random.uniform(0.5, 2.0, n),
        "impact_asymmetry": np.random.normal(0, 0.001, n),
        "absorption_score": np.random.normal(0, 1000, n),
    }, index=index)
    df.index.name = "dt"
    return df
