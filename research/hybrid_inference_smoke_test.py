"""Smoke test for indicator/hybrid_inference.py — verify v9 load + LDC compute
work on cached features/klines.  Does NOT write to DB."""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from indicator.hybrid_inference import HybridInferenceEngine

FEATURES_CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache" / "features_all.parquet"
KLINES = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"


def main():
    features = pd.read_parquet(FEATURES_CACHE)
    if features.index.tz is not None:
        features.index = features.index.tz_convert("UTC").tz_localize(None)

    klines = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    logger.info("Features: %d bars, last %s", len(features), features.index[-1])
    logger.info("Klines:   %d bars, last %s", len(klines), klines.index[-1])

    engine = HybridInferenceEngine()
    if engine.long_model is None:
        logger.error("v9 model not loaded — run retrain_winrate_v9_weekly.py first")
        return 1

    p_long, p_short = engine.predict_v9(features)
    logger.info("v9 predict @ %s: P_long=%.4f P_short=%.4f",
                features.index[-1], p_long, p_short)

    ldc = engine.compute_ldc_latest(klines)
    if not ldc:
        logger.error("LDC compute failed")
        return 1
    logger.info("LDC @ %s: signal=%+d kernel_bear=%s kernel_bull=%s ema_dn=%s ema_up=%s",
                ldc["ts"], ldc["signal"], ldc["is_bearish_kernel"], ldc["is_bullish_kernel"],
                ldc["ema_downtrend"], ldc["ema_uptrend"])

    sig = engine.maybe_emit_signal(features, klines)
    if sig is None:
        logger.info("No hybrid signal triggered at latest bar (expected most hours).")
    else:
        logger.info("HYBRID SIGNAL TRIGGER: %s", sig)

    # Replay over last N bars to count would-be triggers
    logger.info("Replaying last 200 bars to count expected trigger frequency...")
    # We need LDC over a longer window — reuse compute_ldc_latest with full klines
    from research.ldc_python import compute_ldc_signals
    ldc_df = compute_ldc_signals(klines)
    # Align indices
    common = features.index.intersection(ldc_df.index)
    feat_aligned = features.loc[common]
    ldc_aligned = ldc_df.loc[common]
    if engine.long_model is None or engine.short_model is None:
        return 1
    feat_cols = [c for c in engine.feature_cols if c in feat_aligned.columns]
    last_200 = feat_aligned.iloc[-200:]
    if len(last_200) < 10:
        logger.warning("Not enough common bars for replay")
        return 0
    import numpy as np
    X = last_200[feat_cols].values.astype(np.float32)
    p_long_arr = engine.long_model.predict_proba(X)[:, 1]
    p_short_arr = engine.short_model.predict_proba(X)[:, 1]
    ldc_sig = ldc_aligned.loc[last_200.index, "signal"].values
    short_triggers = ((p_short_arr >= 0.65) & (ldc_sig == -1)).sum()
    long_triggers = ((p_long_arr >= 0.65) & (ldc_sig == +1)).sum()
    logger.info("Last 200 bars (~8 days): %d SHORT triggers, %d LONG triggers",
                int(short_triggers), int(long_triggers))
    logger.info("P_short distribution: min=%.3f max=%.3f mean=%.3f >=0.65: %d",
                p_short_arr.min(), p_short_arr.max(), p_short_arr.mean(),
                int((p_short_arr >= 0.65).sum()))
    logger.info("P_long  distribution: min=%.3f max=%.3f mean=%.3f >=0.65: %d",
                p_long_arr.min(), p_long_arr.max(), p_long_arr.mean(),
                int((p_long_arr >= 0.65).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
