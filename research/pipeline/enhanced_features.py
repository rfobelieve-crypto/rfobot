"""
Enhanced Feature Assembler — builds ~80 features across 8 categories.

Sources:
  1. research/ml_data/BTC_USD_15m_tabular.parquet  (existing: OHLCV, funding, OI, regime)
  2. research/ml_data/BTC_USD_15m_tick_features.parquet  (tick-level: BVC, VPIN, trade stats)

Output:
  research/ml_data/BTC_USD_15m_enhanced.parquet

Feature Categories:
  1. Basic trade & price (8)
  2. Order flow imbalance — BVC/VPIN (12)
  3. Volume dynamics (10)
  4. OI position dynamics (12)
  5. Funding rate & cost (10)
  6. Cross-variable interactions (15)
  7. Regime & statistical (10)
  8. Multi-scale lags (8-12)

Usage:
    python -m research.pipeline.enhanced_features
    python -m research.pipeline.enhanced_features --dry-run
"""
from __future__ import annotations

import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TABULAR_PATH = ROOT / "research" / "ml_data" / "BTC_USD_15m_tabular.parquet"
TICK_PATH    = ROOT / "research" / "ml_data" / "BTC_USD_15m_tick_features.parquet"
OUT_PATH     = ROOT / "research" / "ml_data" / "BTC_USD_15m_enhanced.parquet"

ZSCORE_WIN = 96   # 24h in 15m bars
SHORT_WIN  = 4    # 1h
MED_WIN    = 16   # 4h
LONG_WIN   = 96   # 24h


def _zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def _pct_rank(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    return s.rolling(win, min_periods=4).rank(pct=True)


def _slope(s: pd.Series, win: int = 8) -> pd.Series:
    """Rolling linear regression slope (OLS via np.polyfit equiv)."""
    out = s.copy() * np.nan
    vals = s.values
    for i in range(win, len(vals)):
        y = vals[i - win:i]
        if np.isnan(y).any():
            continue
        x = np.arange(win, dtype=float)
        out.iloc[i] = np.polyfit(x, y, 1)[0]
    return out


def build() -> pd.DataFrame:
    # ── Load sources ─────────────────────────────────────────────────────
    tab = pd.read_parquet(TABULAR_PATH)
    tick = pd.read_parquet(TICK_PATH)

    tab["dt"]  = pd.to_datetime(tab["ts_open"], unit="ms", utc=True)
    tick["dt"] = pd.to_datetime(tick["ts_open"], unit="ms", utc=True)

    # Merge on timestamp
    df = tab.set_index("dt").join(tick.set_index("dt").drop(columns=["ts_open"]),
                                   how="left", rsuffix="_tick")
    df = df.sort_index()
    logger.info("Merged: %d rows, %d columns", len(df), len(df.columns))

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 1: Basic Trade & Price (keep existing + add from tick)
    # ═══════════════════════════════════════════════════════════════════════
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change_pct"] = df["close"].pct_change()
    # trade_count, avg_trade_size, large_trade_ratio already from tick

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 2: Order Flow Imbalance (BVC/VPIN) — already from tick
    # Add: BVC delta ratio
    # ═══════════════════════════════════════════════════════════════════════
    total_bvc = df["bvc_buy_vol"] + df["bvc_sell_vol"]
    df["bvc_delta_ratio"] = df["bvc_delta"] / total_bvc.replace(0, np.nan)
    df["taker_delta_ratio"] = df["taker_delta"] / df["tick_total_volume"].replace(0, np.nan)

    # VPIN deviation from normal
    df["vpin_deviation"] = df["vpin"] - df["vpin_ma"]

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 3: Volume Dynamics
    # ═══════════════════════════════════════════════════════════════════════
    # vol_acceleration, price_impact, vol_entropy already from tick
    # Add: volume profile proxy (signed volume at different price levels)
    # Use rolling volume concentration (HHI of volume across bars)
    vol_s = df["tick_total_volume"]
    df["vol_zscore"] = _zscore(vol_s)
    df["vol_pct_rank"] = _pct_rank(vol_s)

    # Price impact z-score
    df["price_impact_zscore"] = _zscore(df["price_impact"])

    # Volume-weighted average price deviation
    df["vwap_deviation"] = (
        (df["close"] - (df["close"] * df["volume"]).rolling(MED_WIN).sum()
         / df["volume"].rolling(MED_WIN).sum().replace(0, np.nan))
        / df["close"]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 4: OI Position Dynamics (enhanced)
    # ═══════════════════════════════════════════════════════════════════════
    # oi, oi_delta, oi_accel, oi_divergence already exist
    oi = df["oi"]
    df["oi_pct_change"] = oi.pct_change()
    df["oi_zscore"]     = _zscore(oi)
    df["oi_delta_zscore"] = _zscore(df["oi_delta"])

    # OI-price correlation (rolling)
    df["oi_price_corr"] = df["oi"].rolling(MED_WIN, min_periods=4).corr(df["close"])

    # OI acceleration z-score
    df["oi_accel_zscore"] = _zscore(df["oi_accel"])

    # OI / volume ratio (leverage proxy)
    df["oi_volume_ratio"] = oi / df["tick_total_volume"].replace(0, np.nan)
    df["oi_volume_ratio_zscore"] = _zscore(df["oi_volume_ratio"])

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 5: Funding Rate & Cost (enhanced)
    # ═══════════════════════════════════════════════════════════════════════
    # funding_rate, funding_deviation, funding_zscore already exist
    fr = df["funding_rate"]
    df["funding_extreme"] = (fr.abs() > 0.0001).astype(int)  # |0.01%| threshold

    # Funding × signed volume (pressure weighted by crowding)
    df["funding_x_delta"] = df["funding_zscore"] * df["taker_delta"]
    df["funding_x_bvc_delta"] = df["funding_zscore"] * df["bvc_delta"]

    # Funding × ΔOI (crowding acceleration)
    df["funding_x_oi_delta"] = df["funding_zscore"] * df["oi_delta"]

    # Cumulative funding cost (rolling 32 bars = 8h, approx 1 settlement period)
    df["cum_funding_8h"] = fr.rolling(32, min_periods=1).sum()
    df["cum_funding_24h"] = fr.rolling(96, min_periods=1).sum()

    # Funding rate of change
    df["funding_roc"] = fr.diff()

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 6: Cross-Variable Interactions
    # ═══════════════════════════════════════════════════════════════════════
    # CVD × ΔOI (already have cvd_x_oi_delta, but recompute with BVC)
    df["bvc_cvd_x_oi_delta"] = df["bvc_cvd_zscore"] * df["oi_delta"]

    # Divergence scores
    # Price up but CVD/OI down = bearish divergence
    ret_sign = np.sign(df["log_return"])
    cvd_sign = np.sign(df["bvc_delta"])
    oi_sign  = np.sign(df["oi_delta"])

    df["price_cvd_divergence"]  = (ret_sign != cvd_sign).astype(int) * ret_sign
    df["price_oi_divergence"]   = (ret_sign != oi_sign).astype(int) * ret_sign

    # 4-quadrant indicator (simplified: price direction + CVD direction + OI direction)
    # Encode as: price_up=4, cvd_up=2, oi_up=1 → 0-7 state
    df["flow_state"] = (
        (ret_sign > 0).astype(int) * 4
        + (cvd_sign > 0).astype(int) * 2
        + (oi_sign > 0).astype(int)
    )

    # Squeeze probability proxy
    # High funding + high OI + CVD reversal
    df["squeeze_proxy"] = (
        df["funding_zscore"].abs()
        * df["oi_zscore"].abs()
        * (1 + df["price_cvd_divergence"].abs())
    )

    # Funding extreme + OI rising + CVD divergence = risk
    df["crowding_risk"] = (
        df["funding_extreme"]
        * (df["oi_delta"] > 0).astype(int)
        * df["price_cvd_divergence"].abs()
    )

    # Absorption ratio: volume without price movement
    df["absorption_ratio"] = (
        df["tick_total_volume"]
        / (df["close"] - df["open"]).abs().replace(0, np.nan)
    )
    df["absorption_zscore"] = _zscore(df["absorption_ratio"])

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 7: Statistical & Regime
    # ═══════════════════════════════════════════════════════════════════════
    # regime, regime_name already exist
    # Rolling kurtosis of returns (tail risk indicator)
    df["return_kurtosis"] = df["log_return"].rolling(LONG_WIN, min_periods=8).kurt()

    # Rolling skew of returns
    df["return_skew"] = df["log_return"].rolling(LONG_WIN, min_periods=8).skew()

    # Feature percentile ranks (for model robustness)
    df["vpin_pct_rank"]       = _pct_rank(df["vpin"])
    df["bvc_delta_pct_rank"]  = _pct_rank(df["bvc_delta"])
    df["oi_delta_pct_rank"]   = _pct_rank(df["oi_delta"])
    df["funding_pct_rank"]    = _pct_rank(df["funding_rate"])

    # ═══════════════════════════════════════════════════════════════════════
    # CATEGORY 8: Multi-Scale Lags & Time Series
    # ═══════════════════════════════════════════════════════════════════════
    # Key features at multiple scales: 1h (4), 4h (16), 24h (96)
    for name, col in [("bvc_delta", df["bvc_delta"]),
                       ("oi_delta", df["oi_delta"]),
                       ("vpin", df["vpin"]),
                       ("taker_delta", df["taker_delta"])]:
        for win, label in [(SHORT_WIN, "1h"), (MED_WIN, "4h"), (LONG_WIN, "24h")]:
            df[f"{name}_ma_{label}"]  = col.rolling(win, min_periods=2).mean()
            df[f"{name}_std_{label}"] = col.rolling(win, min_periods=2).std()

    # Slopes (trend direction over different horizons)
    for name, col in [("bvc_cvd", df["bvc_cvd"]),
                       ("oi", df["oi"])]:
        # Use simple diff-based slope instead of polyfit (much faster)
        for win, label in [(SHORT_WIN, "1h"), (MED_WIN, "4h")]:
            df[f"{name}_slope_{label}"] = col.diff(win) / win

    # BVC delta lags (recent order flow memory)
    for lag in [1, 2, 3, 4, 8]:
        df[f"bvc_delta_lag_{lag}"] = df["bvc_delta"].shift(lag)
        df[f"vpin_lag_{lag}"]      = df["vpin"].shift(lag)

    # ═══════════════════════════════════════════════════════════════════════
    # TARGETS (preserve from original)
    # ═══════════════════════════════════════════════════════════════════════
    # Build clean 1h and 4h return targets
    df["y_return_1h"] = df["close"].shift(-4)  / df["close"] - 1
    df["y_return_4h"] = df["close"].shift(-16) / df["close"] - 1

    # ═══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info("Final dataset: %d rows x %d columns", len(df), len(df.columns))

    # Count features (exclude metadata & targets)
    meta_cols = {"ts_open", "open", "high", "low", "close",
                 "regime", "regime_name", "y_return_1h", "y_return_4h",
                 "future_high_4h", "future_low_4h",
                 "up_move_4h", "down_move_4h", "vol_4h_proxy",
                 "up_move_vol_adj", "down_move_vol_adj",
                 "strength_raw", "strength_vol_adj",
                 "future_return_5m", "future_return_15m", "future_return_1h",
                 "label_5m", "label_15m", "label_1h",
                 "bull_bear_score"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    logger.info("Feature columns: %d", len(feat_cols))

    return df


def run(dry_run: bool = False):
    df = build()

    # Print NaN summary
    nan_rate = df.isnull().mean()
    high_nan = nan_rate[nan_rate > 0.05].sort_values(ascending=False)
    if len(high_nan) > 0:
        logger.info("\nHigh-NaN columns (>5%%):")
        for col, rate in high_nan.items():
            logger.info("  %-30s  %.1f%%", col, rate * 100)

    if dry_run:
        logger.info("DRY RUN — not saving.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=True)  # index = datetime
    logger.info("Saved to %s", OUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
