"""
Feature configuration for live inference.

Only features computable from Binance REST klines + Coinglass API.
Tick-level features (BVC, VPIN, large_cluster) are excluded — they require
raw aggTrades which are not available on Railway.
"""

# ── Features from Binance klines (/fapi/v1/klines) ─────────────────────────
# Raw: open, high, low, close, volume, taker_buy_vol, trade_count
KLINE_RAW = [
    "open", "high", "low", "close", "volume",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
]

# Derived from klines (computed in feature_builder_live)
KLINE_DERIVED = [
    "log_return", "price_change_pct",
    "realized_vol_20b", "return_skew", "return_kurtosis",
    "taker_delta_ratio",
    # Rolling stats
    "taker_delta_ma_4h", "taker_delta_std_4h",
    "taker_delta_ma_24h", "taker_delta_std_24h",
    "volume_ma_4h", "volume_ma_24h",
    # Return lags
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4",
    "return_lag_5", "return_lag_6", "return_lag_7", "return_lag_8",
    "return_lag_9", "return_lag_10",
]

# ── Features from Coinglass API v4 ─────────────────────────────────────────
COINGLASS_RAW = [
    "cg_oi_close", "cg_oi_delta", "cg_oi_accel",
    "cg_oi_agg_close", "cg_oi_agg_delta",
    "cg_oi_binance_share",
    "cg_liq_long", "cg_liq_short", "cg_liq_total",
    "cg_liq_ratio", "cg_liq_imbalance",
    "cg_ls_long_pct", "cg_ls_short_pct", "cg_ls_ratio",
    "cg_funding_close", "cg_funding_range",
    "cg_taker_buy", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
]

COINGLASS_ZSCORE = [f"{f}_zscore" for f in [
    "cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
    "cg_ls_ratio", "cg_taker_delta", "cg_funding_close",
]]

COINGLASS_CROSS = [
    "cg_liq_x_oi", "cg_crowding", "cg_conviction",
]

# ── All feature columns (used for training and inference) ──────────────────
ALL_FEATURES = KLINE_DERIVED + COINGLASS_RAW + COINGLASS_ZSCORE + COINGLASS_CROSS

# ── Columns never used as features ─────────────────────────────────────────
EXCLUDE = {
    "ts_open", "open", "high", "low", "close", "volume",
    "taker_buy_vol", "taker_buy_quote", "trade_count",
    "y_return_4h", "actual_return_4h",
}

# ── Bull/Bear Power components (rule-based, not model features) ────────────
BBP_COMPONENTS = {
    "oi_delta": "cg_oi_delta_zscore",
    "funding": "cg_funding_close_zscore",     # inverted
    "taker_delta": "cg_taker_delta_zscore",
    "ls_ratio": "cg_ls_ratio_zscore",         # inverted
}
