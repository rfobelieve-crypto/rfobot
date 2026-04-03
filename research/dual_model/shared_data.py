"""
Shared data loading, caching, and walk-forward splitting.

Central module for all dual-model research scripts.
Fetches Binance klines + Coinglass endpoints, builds features via
the live feature builder, and provides walk-forward cross-validation.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "research" / "dual_model" / ".cache"
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"


def ensure_dirs():
    """Create cache and results directories."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_cache_data(limit: int = 500, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch klines + CG data, build all features, cache to parquet.

    Parameters
    ----------
    limit : Number of 1h bars to fetch (max ~500 for CG endpoints).
    force_refresh : If True, re-fetch even if cache exists.

    Returns
    -------
    DataFrame with all features, indexed by datetime (UTC).
    """
    ensure_dirs()
    cache_path = CACHE_DIR / "features_all.parquet"

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        logger.info("Loaded cached features: %d bars x %d cols", len(df), len(df.columns))
        return df

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
    from indicator.feature_builder_live import build_live_features

    klines = fetch_binance_klines(limit=limit)
    cg_data = fetch_coinglass(interval="1h", limit=limit)

    features = build_live_features(klines, cg_data)

    # Attach raw close for label construction
    if "close" not in features.columns and "close" in klines.columns:
        features["close"] = klines["close"].reindex(features.index)

    features.to_parquet(cache_path)
    logger.info("Cached features: %d bars x %d cols → %s",
                len(features), len(features.columns), cache_path)
    return features


def walk_forward_splits(
    n_samples: int,
    initial_train: int = 288,
    test_size: int = 48,
    step: int = 48,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate walk-forward expanding-window train/test index splits.

    Parameters
    ----------
    n_samples : Total number of samples.
    initial_train : Minimum training window (bars). 288 = 12 days.
    test_size : Test window size (bars). 48 = 2 days.
    step : Step between folds (bars). 48 = 2 days.

    Returns
    -------
    List of (train_indices, test_indices) tuples.
    No overlap; test is always strictly after train.
    """
    splits = []
    train_end = initial_train
    while train_end + test_size <= n_samples:
        train_idx = list(range(train_end))
        test_idx = list(range(train_end, train_end + test_size))
        splits.append((train_idx, test_idx))
        train_end += step
    if not splits:
        logger.warning("Not enough data for walk-forward: n=%d, init=%d, test=%d",
                       n_samples, initial_train, test_size)
    else:
        logger.info("Walk-forward: %d folds, train=%d..%d, test=%d bars each",
                     len(splits), initial_train, train_end - step, test_size)
    return splits


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return columns that are valid model features (exclude targets, metadata)."""
    exclude = {
        "ts_open", "open", "high", "low", "close", "volume",
        "taker_buy_vol", "taker_buy_quote", "trade_count", "quote_vol",
        "close_time", "ignore",
        "y_return_4h", "y_dir", "y_abs_return", "y_vol_adj_abs",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("y_")]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_cache_data(force_refresh=True)
    print(f"Features loaded: {len(df)} bars x {len(df.columns)} columns")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    feats = get_available_features(df)
    print(f"Available features: {len(feats)}")

    splits = walk_forward_splits(len(df))
    for i, (tr, te) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr)} bars, test={len(te)} bars, "
              f"test range={df.index[te[0]]} → {df.index[te[-1]]}")
