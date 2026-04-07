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
    "quote_vol_zscore",        # Binance quote volume z-score (IC +0.043)
    "quote_vol_ratio",         # quote vol vs 24h mean
]

# ── Features from Coinglass API v4 (7 endpoints, native 1h) ──────────────
COINGLASS_RAW = [
    "cg_oi_close", "cg_oi_delta", "cg_oi_accel",
    "cg_oi_agg_close", "cg_oi_agg_delta",
    "cg_oi_binance_share",
    # NEW: OI multi-window momentum (IC -0.078 ~ -0.046)
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    # NEW: OI intra-bar volatility (IC +0.047)
    "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_upper_shadow",
    # NEW: Binance share z-score (IC -0.060)
    "cg_oi_binance_share_zscore",
    "cg_liq_long", "cg_liq_short", "cg_liq_total",
    "cg_liq_ratio", "cg_liq_imbalance",
    "cg_ls_long_pct", "cg_ls_short_pct", "cg_ls_ratio",
    "cg_gls_long_pct", "cg_gls_short_pct", "cg_gls_ratio",  # global L/S
    "cg_ls_divergence",                                        # top vs global
    "cg_funding_close", "cg_funding_range",
    "cg_taker_buy", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
    # --- New Startup plan endpoints ---
    # Coinbase Premium
    "cg_cb_premium", "cg_cb_premium_rate",
    # Bitfinex Margin
    "cg_bfx_margin_long", "cg_bfx_margin_short",
    "cg_bfx_margin_ratio", "cg_bfx_margin_delta",
    # Top L/S Position Ratio
    "cg_pos_long_pct", "cg_pos_short_pct", "cg_pos_ls_ratio",
    # Futures CVD Aggregated
    "cg_fcvd_buy", "cg_fcvd_sell", "cg_fcvd_cum", "cg_fcvd_delta",
    # Spot CVD Aggregated
    "cg_scvd_buy", "cg_scvd_sell", "cg_scvd_cum", "cg_scvd_delta",
    # Liquidation Aggregated
    "cg_liq_agg_long", "cg_liq_agg_short", "cg_liq_agg_total", "cg_liq_agg_imbalance",
    # OI Coin-Margin
    "cg_oi_cm_close", "cg_oi_cm_delta",
]

COINGLASS_ZSCORE = [f"{f}_zscore" for f in [
    "cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
    "cg_ls_ratio", "cg_taker_delta", "cg_funding_close",
    "cg_oi_close", "cg_liq_total",
    "cg_gls_ratio", "cg_ls_divergence",
    # New endpoints
    "cg_cb_premium_rate", "cg_bfx_margin_ratio", "cg_bfx_margin_delta",
    "cg_pos_ls_ratio", "cg_fcvd_delta", "cg_scvd_delta",
    "cg_liq_agg_imbalance", "cg_oi_cm_delta",
]]

COINGLASS_CROSS = [
    "cg_liq_x_oi", "cg_crowding", "cg_conviction",
    # New cross-source features
    "cg_spot_futures_cvd_divergence", "cg_pos_account_divergence",
    "cg_liq_exchange_vs_agg", "cg_oi_cm_vs_usd", "cg_margin_funding_align",
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

# ── Order Flow Toxicity features (IC-validated) ────────────────────────────
TOXICITY_FEATURES = [
    "tox_bv_vpin",           # Bulk Volume VPIN (IC -0.021)
    "tox_bv_vpin_zscore",    # VPIN z-score
    "tox_accum",             # toxic flow accumulation
    "tox_accum_zscore",      # accumulation z-score
    "tox_div_taker",         # VPIN vs taker divergence
    "tox_liq_exhaust",       # toxicity × liquidity exhaustion (IC -0.024)
    "tox_funding_pressure",  # toxicity × funding × OI feedback
    "tox_pressure",          # composite toxicity pressure (signed)
    "tox_pressure_zscore",   # composite z-score (IC +0.071, p<0.00001)
]

# ── All feature columns (used for training and inference) ─────────────────
# v3 regime models use all features including VOLUME_DYNAMICS.
ALL_FEATURES = (KLINE_DERIVED + COINGLASS_RAW + COINGLASS_ZSCORE
                + COINGLASS_CROSS + TIME_FEATURES + MOMENTUM_FEATURES
                + VOLUME_DYNAMICS + TOXICITY_FEATURES)

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
