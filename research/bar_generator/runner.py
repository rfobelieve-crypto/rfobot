"""
Bar runner: the orchestration loop.

Flow per iteration:
  get_pending_windows
    → assemble_features          (bar_generator layer)
    → compute_score              (score_engine layer)
    → compute_bands              (features/statistics layer)
    → upsert_bar                 (storage layer)

Can run as a blocking loop (daemon thread / standalone process)
or be called once for backfill.
"""
from __future__ import annotations
import logging
import time
import pandas as pd

from research.bar_generator.time_bars import get_pending_windows
from research.bar_generator.feature_assembler import assemble_features
from research.score_engine.registry import get_model
from research.features.statistics import compute_bands
from research.storage.market_state_repository import upsert_bar, query_bars
from research.storage.schema import ensure_schema
from research.config.settings import TF_MS

logger = logging.getLogger(__name__)


def run_once(
    symbol: str,
    timeframe: str,
    lookback_days: int = 7,
    score_model: str = "rule_based",
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
    model = get_model(score_model)
    config = {"score_model": score_model, "band_window": band_window, "band_n_sigma": band_n_sigma}
    count = 0

    for window_start_ms, window_end_ms in windows:
        try:
            # 1. Feature extraction
            features = assemble_features(symbol, timeframe, window_start_ms, window_end_ms)

            # 2. Scoring
            score_out = model.compute_score(features, config)

            # 3. Dynamic bands (features/statistics, not scorer)
            band_stats = _bands_for_window(
                symbol, timeframe, window_start_ms,
                band_window, band_n_sigma,
                current_score=score_out.risk_adj_score,
            )

            # 4. Assemble full row
            row = {
                **features,
                "reversal_score":     score_out.reversal_score,
                "continuation_score": score_out.continuation_score,
                "confidence":         score_out.confidence,
                "final_bias":         score_out.final_bias,
                "risk_adj_score":     score_out.risk_adj_score,
                "signal":             score_out.signal,
                "score_model":        model.model_name,
                **band_stats,
            }

            upsert_bar(row)
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
    score_model: str = "rule_based",
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
                    run_once(symbol, tf, lookback_days, score_model, band_window, band_n_sigma)
                except Exception:
                    logger.exception("run_once error  %s %s", symbol, tf)
        time.sleep(interval_seconds)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bands_for_window(
    symbol: str,
    timeframe: str,
    current_window_start: int,
    band_window: int,
    n_sigma: float,
    current_score: float,
) -> dict:
    """
    Fetch historical risk_adj_score values, append current score,
    then call features/statistics.compute_bands().
    """
    tf_ms = TF_MS.get(timeframe, 3_600_000)
    lookback_start = current_window_start - band_window * tf_ms

    hist = query_bars(symbol, timeframe, lookback_start, current_window_start - 1)

    if hist.empty or "risk_adj_score" not in hist.columns:
        series = pd.Series([current_score])
    else:
        series = pd.concat([
            hist["risk_adj_score"].dropna(),
            pd.Series([current_score]),
        ], ignore_index=True)

    return compute_bands(series, window=band_window, n_sigma=n_sigma)
