"""
Feature group definitions for the Magnitude model (4h).

Magnitude features emphasize:
  - OI dynamics (expansion = volatile, contraction = calm)
  - Funding rate (extreme funding → squeeze potential)
  - Liquidation pressure (cascade potential)
  - Volatility state (realized vol, vol regime, kurtosis)
  - Crowding (overleveraged positioning)
  - Order flow pressure (taker delta, absorption)

Two groups for Experiment 3:
  - OLD_MAGNITUDE:      Features from pre-upgrade
  - EXPANDED_MAGNITUDE: + new CG features relevant to magnitude
"""
from __future__ import annotations

# ── Core magnitude features (pre-upgrade) ───────────────────────────────

VOLATILITY_FEATURES = [
    "realized_vol_20b", "return_kurtosis",
    "vol_regime", "vol_acceleration", "vol_kurtosis", "vol_entropy",
]

OI_FEATURES = [
    "cg_oi_close", "cg_oi_delta", "cg_oi_accel",
    "cg_oi_agg_close", "cg_oi_agg_delta",
    "cg_oi_close_pctchg_4h", "cg_oi_close_pctchg_8h",
    "cg_oi_close_pctchg_12h", "cg_oi_close_pctchg_24h",
    "cg_oi_range_zscore", "cg_oi_range_pct",
    "cg_oi_delta_zscore", "cg_oi_agg_delta_zscore",
    "cg_oi_close_zscore",
    "cg_oi_delta_slope_4h", "cg_oi_delta_mom_1h",
]

FUNDING_FEATURES = [
    "cg_funding_close", "cg_funding_range",
    "cg_funding_close_zscore",
    "cg_funding_close_slope_4h", "cg_funding_close_mom_1h",
]

LIQUIDATION_FEATURES = [
    "cg_liq_long", "cg_liq_short", "cg_liq_total",
    "cg_liq_ratio", "cg_liq_imbalance",
    "cg_liq_imbalance_zscore", "cg_liq_total_zscore",
    "cg_liq_x_oi",
]

CROWDING_FEATURES = [
    "cg_ls_ratio", "cg_gls_ratio",
    "cg_ls_ratio_zscore", "cg_gls_ratio_zscore",
    "cg_crowding", "cg_conviction",
    "squeeze_proxy",
]

FLOW_FEATURES = [
    "cg_taker_buy", "cg_taker_sell", "cg_taker_delta", "cg_taker_ratio",
    "cg_taker_delta_zscore",
    "cg_taker_delta_slope_4h", "cg_taker_delta_mom_1h",
    "taker_delta_ratio", "taker_delta_ma_24h", "taker_delta_std_24h",
    "funding_taker_align", "oi_price_divergence",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
]

OLD_MAGNITUDE = (
    VOLATILITY_FEATURES + OI_FEATURES + FUNDING_FEATURES +
    LIQUIDATION_FEATURES + CROWDING_FEATURES + FLOW_FEATURES +
    TIME_FEATURES
)

# ── New magnitude features (from Startup upgrade) ──────────────────────

NEW_MAGNITUDE = [
    # Aggregated liquidation (cross-exchange, better coverage)
    "cg_liq_agg_long", "cg_liq_agg_short",
    "cg_liq_agg_total", "cg_liq_agg_imbalance",
    "cg_liq_agg_imbalance_zscore",
    "cg_liq_exchange_vs_agg",
    # OI Coin-Margin (leveraged conviction signal)
    "cg_oi_cm_close", "cg_oi_cm_delta", "cg_oi_cm_delta_zscore",
    "cg_oi_cm_vs_usd",
    # Futures CVD aggregated (cross-exchange flow pressure)
    "cg_fcvd_delta", "cg_fcvd_delta_zscore",
    # Bitfinex margin (institutional leverage = squeeze potential)
    "cg_bfx_margin_delta", "cg_bfx_margin_delta_zscore",
    # Top position ratio (position-based crowding)
    "cg_pos_ls_ratio", "cg_pos_ls_ratio_zscore",
    "cg_pos_account_divergence",
    # Margin-funding alignment (crowding confirmation)
    "cg_margin_funding_align",
]

EXPANDED_MAGNITUDE = sorted(set(OLD_MAGNITUDE + NEW_MAGNITUDE))

# ── Liquidity fragility (captures "thin liquidity = bigger moves") ────

LIQUIDITY_FRAGILITY = [
    "impact_asymmetry", "impact_asymmetry_zscore",
    "price_impact", "price_impact_zscore",
    "fragility", "fragility_zscore",
]

POST_ABSORPTION = [
    "post_absorb_breakout", "post_absorb_breakout_z",
    "abs_completion", "abs_completion_z",
    "flow_trend_score",
]

EXPANDED_WITH_FRAGILITY = sorted(set(EXPANDED_MAGNITUDE + LIQUIDITY_FRAGILITY + POST_ABSORPTION))

# ── For Experiment 3 comparison ─────────────────────────────────────────

MAGNITUDE_GROUPS = {
    "baseline_old": OLD_MAGNITUDE,
    "expanded":     EXPANDED_MAGNITUDE,
    "+ liq_fragility": EXPANDED_WITH_FRAGILITY,
}


def filter_available(feature_list: list[str], df_columns: list[str]) -> list[str]:
    """Return only features that exist in the DataFrame."""
    available = set(df_columns)
    present = [f for f in feature_list if f in available]
    missing = [f for f in feature_list if f not in available]
    if missing:
        import logging
        logging.getLogger(__name__).warning(
            "Magnitude features missing from data: %s", missing[:10])
    return present
