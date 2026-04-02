"""
Bar runner: the orchestration loop.

Flow per iteration:
  get_pending_windows
    → assemble_features          (bar_generator layer)
    → upsert_bar                 (storage layer)

Can run as a blocking loop (daemon thread / standalone process)
or be called once for backfill.
"""
from __future__ import annotations
import logging
import time

from research.bar_generator.time_bars import get_pending_windows
from research.bar_generator.feature_assembler import assemble_features
from research.storage.market_state_repository import upsert_bar
from research.storage.schema import ensure_schema

logger = logging.getLogger(__name__)


def run_once(
    symbol: str,
    timeframe: str,
    lookback_days: int = 7,
    band_window: int = 20,
    band_n_sigma: float = 2.0,
) -> int:
    """
    Compute all pending bars for one (symbol, timeframe).
    Returns the number of bars successfully written.
    """
    lookback_seconds = lookback_days * 86_400
    windows = get_pending_windows(symbol, timeframe, lookback_seconds)

    if not windows:
        return 0

    logger.info("Computing %d bars  [%s %s]", len(windows), symbol, timeframe)
    count = 0

    for window_start_ms, window_end_ms in windows:
        try:
            features = assemble_features(symbol, timeframe, window_start_ms, window_end_ms)
            upsert_bar(features)
            count += 1

        except Exception:
            logger.exception("bar failed @ %s %s %d", symbol, timeframe, window_start_ms)

    logger.info("Done: %d/%d bars written  [%s %s]", count, len(windows), symbol, timeframe)
    return count


def run_loop(
    symbols: list[str],
    timeframes: list[str],
    lookback_days: int = 7,
    interval_seconds: int = 60,
    band_window: int = 20,
    band_n_sigma: float = 2.0,
):
    """
    Continuous loop: runs run_once for all (symbol, timeframe) pairs
    every `interval_seconds`.  Designed for daemon thread or process.
    """
    ensure_schema()
    logger.info(
        "Bar runner started  symbols=%s  timeframes=%s  interval=%ds",
        symbols, timeframes, interval_seconds,
    )
    while True:
        for symbol in symbols:
            for tf in timeframes:
                try:
                    run_once(symbol, tf, lookback_days, band_window, band_n_sigma)
                except Exception:
                    logger.exception("run_once error  %s %s", symbol, tf)
        time.sleep(interval_seconds)
