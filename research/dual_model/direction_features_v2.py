"""
Feature group definitions for the Direction model (4h).

Three groups for ablation experiments:
  - OLD_FEATURES: Features available before the CG endpoint upgrade
  - NEW_KEY_4:    The 4 high-priority new features (+ their z-scores)
  - NEW_CG_ALL:   All new CG features from the Startup upgrade
  - FULL:         OLD + NEW_CG_ALL (complete direction feature set)

Direction features emphasize:
  - Cross-market divergence (spot vs futures, Coinbase vs global)
  - Positioning asymmetry (top trader position vs account ratio)
  - Institutional/smart-money tilt (Bitfinex margin, Coinbase premium)
  - Sentiment extremes (Fear & Greed via snapshot, funding extremes)
"""
from __future__ import annotations

# ── Old features (pre-upgrade, available to previous direction model) ────

OLD_KLINE = [
    "realized_vol_20b", "return_kurtosis",
    "taker_delta_ratio", "taker_delta_ma_24h", "taker_delta_std_24h",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4",
    "return_lag_5", "return_lag_6", "return_lag_7", "return_lag_8",
    "return_lag_9", "return_lag_10",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "quote_vol_zscore", "quote_vol_ratio",
]

OLD_CG = [
    "cg_oi_close", "cg_oi_delta", "cg_oi_accel",
    "cg_oi_agg_close", "cg_oi_agg_delta", "cg_oi_binance_share",
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range_zscore", "cg_oi_range_pct", "cg_oi_upper_shadow",
    "cg_oi_binance_share_zscore",
    "cg_liq_long", "cg_liq_short", "cg_liq_total",
    "cg_liq_ratio", "cg_liq_imbalance",
    "cg_ls_long_pct", "cg_ls_short_pct", "cg_ls_ratio",
    "cg_gls_long_pct", "cg_gls_short_pct", "cg_gls_ratio",
    "cg_ls_divergence",
    "cg_funding_close", "cg_funding_range",
    "cg_taker_buy", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
    # Z-scores
    "cg_oi_delta_zscore", "cg_oi_agg_delta_zscore",
    "cg_liq_imbalance_zscore", "cg_ls_ratio_zscore",
    "cg_taker_delta_zscore", "cg_funding_close_zscore",
    "cg_oi_close_zscore", "cg_liq_total_zscore",
    "cg_gls_ratio_zscore", "cg_ls_divergence_zscore",
    # Cross
    "cg_liq_x_oi", "cg_crowding", "cg_conviction",
    # Momentum
    "cg_oi_delta_slope_4h", "cg_oi_delta_mom_1h",
    "cg_taker_delta_slope_4h", "cg_taker_delta_mom_1h",
    "cg_funding_close_slope_4h", "cg_funding_close_mom_1h",
    "oi_price_divergence", "funding_taker_align",
]

OLD_DERIVED = [
    "vol_regime", "vol_acceleration", "vol_kurtosis",
    "vol_entropy", "squeeze_proxy",
]

# ── Regime indicators + regime-conditioned features (2026-04-13) ──────
# Direct regime indicators let XGB learn conditional splits naturally.
# (1 - is_X) form preserves signal in 80%+ samples, masks noise in dead regime.
# See .claude/rules/mistake.md 2026-04-13 entry: never use `feat × sparse_indicator`.
REGIME_V1 = [
    "is_trending_bull",          # regime indicator (XGB learns splits)
    "is_trending_bear",          # regime indicator
    "vol_kurt_non_bear",         # vol_kurtosis × (1 - is_bear), IC +0.054 STABLE
    "oi_8h_non_bull",            # oi_close_pctchg_8h × (1 - is_bull), IC -0.0712
    "long_liq_exhaustion_4h",    # regime-normalized long liq z-score, IC +0.0667
    "cvd_persistence_12h",       # 12h CVD z-score mean-reversion, IC -0.0343
]

# ── Liquidity fragility (IC=-0.071, p=0.00002, time-stable) ──────────

LIQUIDITY_FRAGILITY = [
    "impact_asymmetry", "impact_asymmetry_zscore",
    "price_impact", "price_impact_zscore",
    "fragility", "fragility_zscore",
]

# ── Post-absorption breakout (mag IC=0.191, dir IC=0.065) ────────────

POST_ABSORPTION = [
    "post_absorb_breakout", "post_absorb_breakout_z",
    "abs_completion", "abs_completion_z",
    "flow_trend_score",
]

OLD_FEATURES = OLD_KLINE + OLD_CG + OLD_DERIVED + REGIME_V1

# ── New key 4 features (highest directional potential) ──────────────────

NEW_KEY_4 = [
    # Coinbase Premium — US institutional buying pressure
    "cg_cb_premium", "cg_cb_premium_rate", "cg_cb_premium_rate_zscore",
    # Spot vs Futures CVD divergence — structural reversal signal
    "cg_spot_futures_cvd_divergence",
    # Account vs Position ratio divergence — smart money signal
    "cg_pos_account_divergence",
    # Bitfinex Margin ratio — institutional leverage direction
    "cg_bfx_margin_ratio", "cg_bfx_margin_delta",
    "cg_bfx_margin_ratio_zscore", "cg_bfx_margin_delta_zscore",
]

# ── All new CG features (from Startup upgrade) ─────────────────────────

NEW_CG_ALL = [
    # Coinbase Premium
    "cg_cb_premium", "cg_cb_premium_rate", "cg_cb_premium_rate_zscore",
    # Bitfinex Margin
    "cg_bfx_margin_long", "cg_bfx_margin_short",
    "cg_bfx_margin_ratio", "cg_bfx_margin_delta",
    "cg_bfx_margin_ratio_zscore", "cg_bfx_margin_delta_zscore",
    # Top L/S Position Ratio
    "cg_pos_long_pct", "cg_pos_short_pct", "cg_pos_ls_ratio",
    "cg_pos_ls_ratio_zscore",
    # Futures CVD Aggregated
    "cg_fcvd_buy", "cg_fcvd_sell", "cg_fcvd_cum", "cg_fcvd_delta",
    "cg_fcvd_delta_zscore",
    # Spot CVD Aggregated
    "cg_scvd_buy", "cg_scvd_sell", "cg_scvd_cum", "cg_scvd_delta",
    "cg_scvd_delta_zscore",
    # Liquidation Aggregated
    "cg_liq_agg_long", "cg_liq_agg_short",
    "cg_liq_agg_total", "cg_liq_agg_imbalance",
    "cg_liq_agg_imbalance_zscore",
    # OI Coin-Margin
    "cg_oi_cm_close", "cg_oi_cm_delta", "cg_oi_cm_delta_zscore",
    # Cross-source
    "cg_spot_futures_cvd_divergence",
    "cg_pos_account_divergence",
    "cg_liq_exchange_vs_agg",
    "cg_oi_cm_vs_usd",
    "cg_margin_funding_align",
]

# ── Full direction feature set (old + all new) ─────────────────────────

FULL_DIRECTION = sorted(set(OLD_FEATURES + NEW_CG_ALL + LIQUIDITY_FRAGILITY + POST_ABSORPTION))

# ── Ablation groups for Experiment 2 ───────────────────────────────────

ABLATION_GROUPS = {
    "baseline_old":     OLD_FEATURES,
    "+ cb_premium":     OLD_FEATURES + ["cg_cb_premium", "cg_cb_premium_rate", "cg_cb_premium_rate_zscore"],
    "+ spot_fut_cvd":   OLD_FEATURES + ["cg_spot_futures_cvd_divergence", "cg_fcvd_delta", "cg_fcvd_delta_zscore", "cg_scvd_delta", "cg_scvd_delta_zscore"],
    "+ pos_divergence": OLD_FEATURES + ["cg_pos_account_divergence", "cg_pos_long_pct", "cg_pos_short_pct", "cg_pos_ls_ratio", "cg_pos_ls_ratio_zscore"],
    "+ bfx_margin":     OLD_FEATURES + ["cg_bfx_margin_ratio", "cg_bfx_margin_delta", "cg_bfx_margin_ratio_zscore", "cg_bfx_margin_delta_zscore"],
    "+ key_4_only":     OLD_FEATURES + NEW_KEY_4,
    "+ liq_fragility":  OLD_FEATURES + NEW_KEY_4 + LIQUIDITY_FRAGILITY,
    "+ post_absorption": OLD_FEATURES + NEW_KEY_4 + LIQUIDITY_FRAGILITY + POST_ABSORPTION,
    "full_expanded":    FULL_DIRECTION,
}


def filter_available(feature_list: list[str], df_columns: list[str]) -> list[str]:
    """Return only features that exist in the DataFrame."""
    available = set(df_columns)
    present = [f for f in feature_list if f in available]
    missing = [f for f in feature_list if f not in available]
    if missing:
        import logging
        logging.getLogger(__name__).warning(
            "Direction features missing from data: %s", missing[:10])
    return present
