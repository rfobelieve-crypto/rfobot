"""
Feature configuration for live inference (1h bars).

Data sources: Binance REST klines (1h) + Coinglass API v4 (1h native).
All features are computable from these two sources.
"""

# ── Interval configuration ────────────────────────────────────────────────
BAR_INTERVAL = "1h"
HORIZON_BARS = 4        # 4h = 4 × 1h
ZSCORE_WINDOW = 24      # 24h lookback for z-scores
ROLLING_SHORT = 4       # 4h
ROLLING_LONG = 24       # 24h

# ── Features from Binance klines (/fapi/v1/klines) ───────────────────────
KLINE_RAW = [
    "open", "high", "low", "close", "volume",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
]

KLINE_DERIVED = [
    "log_return", "price_change_pct",
    "realized_vol_20b", "return_skew", "return_kurtosis",
    "taker_delta_ratio",
    "taker_delta_ma_4h", "taker_delta_std_4h",
    "taker_delta_ma_24h", "taker_delta_std_24h",
    "volume_ma_4h", "volume_ma_24h",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4",
    "return_lag_5", "return_lag_6", "return_lag_7", "return_lag_8",
    "return_lag_9", "return_lag_10",
]

# ── Features from Coinglass API v4 (7 endpoints, native 1h) ──────────────
COINGLASS_RAW = [
    "cg_oi_close", "cg_oi_delta", "cg_oi_accel",
    "cg_oi_agg_close", "cg_oi_agg_delta",
    "cg_oi_binance_share",
    "cg_liq_long", "cg_liq_short", "cg_liq_total",
    "cg_liq_ratio", "cg_liq_imbalance",
    "cg_ls_long_pct", "cg_ls_short_pct", "cg_ls_ratio",
    "cg_gls_long_pct", "cg_gls_short_pct", "cg_gls_ratio",  # global L/S
    "cg_ls_divergence",                                        # top vs global
    "cg_funding_close", "cg_funding_range",
    "cg_taker_buy", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
]

COINGLASS_ZSCORE = [f"{f}_zscore" for f in [
    "cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
    "cg_ls_ratio", "cg_taker_delta", "cg_funding_close",
    "cg_oi_close", "cg_liq_total",
    "cg_gls_ratio", "cg_ls_divergence",  # new
]]

COINGLASS_CROSS = [
    "cg_liq_x_oi", "cg_crowding", "cg_conviction",
]

# ── Momentum / slope / divergence features ────────────────────────────────
MOMENTUM_FEATURES = [
    "cg_oi_delta_slope_4h", "cg_oi_delta_mom_1h",
    "cg_taker_delta_slope_4h", "cg_taker_delta_mom_1h",
    "cg_funding_close_slope_4h", "cg_funding_close_mom_1h",
    "oi_price_divergence",
    "funding_taker_align",
    "vol_regime",
]

# ── Volume dynamics features (IC-validated) ──────────────────────────────
VOLUME_DYNAMICS = [
    "vol_acceleration",    # short/long volume MA ratio (IC +0.049)
    "vol_kurtosis",        # rolling volume kurtosis (IC +0.072)
    "vol_entropy",         # rolling volume entropy (IC -0.048)
    "squeeze_proxy",       # funding × OI × CVD reversal composite (IC +0.047)
]

# ── Time-of-day features (cyclical encoding) ──────────────────────────────
TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
]

# ── All feature columns (used for training and inference) ─────────────────
# v3 regime models use all features including VOLUME_DYNAMICS.
ALL_FEATURES = (KLINE_DERIVED + COINGLASS_RAW + COINGLASS_ZSCORE
                + COINGLASS_CROSS + TIME_FEATURES + MOMENTUM_FEATURES
                + VOLUME_DYNAMICS)

# ── Columns never used as features ────────────────────────────────────────
EXCLUDE = {
    "ts_open", "open", "high", "low", "close", "volume",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "y_return_4h", "actual_return_4h",
}

# ── Bull/Bear Power components (rule-based, not model features) ───────────
BBP_COMPONENTS = {
    "oi_delta": "cg_oi_delta_zscore",
    "funding": "cg_funding_close_zscore",        # inverted
    "taker_delta": "cg_taker_delta_zscore",
    "ls_ratio": "cg_ls_ratio_zscore",            # inverted
    "ls_divergence": "cg_ls_divergence_zscore",  # new: top vs global
}
