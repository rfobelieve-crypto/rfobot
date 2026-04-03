"""
Direction-specific feature stack for BTC direction prediction.

Design principle: every feature here must have ASYMMETRIC information
content — i.e., when the feature is high, P(up) ≠ P(down). Features
that predict magnitude equally in both directions (like realized_vol)
are intentionally excluded from this module.

Feature groups:
  1. Large trade flow separation (from aggTrades or kline proxy)
  2. Short-horizon taker imbalance persistence
  3. Price response / absorption proxy
  4. Short-horizon momentum / reversal context
  5. Order book imbalance (stub for future integration)

All features use only trailing (past + current bar) data. No look-ahead.

Usage:
    from research.direction_features import build_direction_feature_set
    feat_df = build_direction_feature_set(df, config)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series, window: int = 24, min_periods: int = 6) -> pd.Series:
    """Trailing rolling z-score."""
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (s - mu) / sd


def _slope(s: pd.Series, window: int = 4) -> pd.Series:
    """Trailing OLS slope over rolling window."""
    def _polyfit_slope(arr):
        if len(arr) < 2 or np.any(np.isnan(arr)):
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]
    return s.rolling(window, min_periods=2).apply(_polyfit_slope, raw=True)


def _sign_persistence(s: pd.Series, window: int = 5) -> pd.Series:
    """Count of consecutive same-sign bars in trailing window."""
    signs = np.sign(s)
    result = np.zeros(len(s))
    for i in range(1, len(s)):
        if np.isnan(signs.iloc[i]):
            result[i] = 0
        elif signs.iloc[i] == signs.iloc[i - 1]:
            result[i] = result[i - 1] + 1
        else:
            result[i] = 1
    return pd.Series(result, index=s.index)


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Large Trade Flow Separation
# ═══════════════════════════════════════════════════════════════════════

def compute_large_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Large vs small trade flow features.

    Uses real aggTrades columns (agg_*) if available,
    otherwise falls back to kline-derived proxy.

    Returns DataFrame with directional large trade features.
    """
    out = pd.DataFrame(index=df.index)

    # ── Real aggTrades path (preferred) ──
    if "agg_large_delta" in df.columns:
        large_delta = df["agg_large_delta"].fillna(0)
        small_delta = df["agg_small_delta"].fillna(0)

        out["large_delta"] = large_delta
        out["small_delta"] = small_delta
        out["large_delta_zscore"] = _zscore(large_delta)
        out["small_delta_zscore"] = _zscore(small_delta)

        # Large/small divergence: large buyers + small sellers = strong UP signal
        out["large_small_divergence"] = large_delta - small_delta
        out["large_small_div_zscore"] = _zscore(out["large_small_divergence"])

        # Directional ratios
        if "agg_large_buy_ratio" in df.columns:
            lbr = df["agg_large_buy_ratio"].fillna(0.5)
            out["large_buy_ratio"] = lbr
            out["large_buy_ratio_zscore"] = _zscore(lbr)
            # Shift from neutral: >0.5 = large buyers dominate
            out["large_buy_bias"] = lbr - 0.5

        # Momentum of large delta
        out["large_delta_ma_4"] = large_delta.rolling(4, min_periods=1).mean()
        out["large_delta_slope_4"] = _slope(large_delta, 4)

        # Persistence: is large delta consistently positive/negative?
        out["large_delta_persistence"] = _sign_persistence(large_delta, 8)

    # ── Kline proxy path (fallback) ──
    else:
        if "taker_buy_vol" in df.columns and "volume" in df.columns:
            taker_buy = df["taker_buy_vol"].fillna(0)
            taker_sell = df["volume"].fillna(0) - taker_buy
            taker_delta = taker_buy - taker_sell

            # Use avg_trade_size as large/small proxy
            if "avg_trade_size" in df.columns:
                ats = df["avg_trade_size"]
                ats_median = ats.rolling(24, min_periods=6).median()
                # Bars with above-median avg trade size → "large" bars
                is_large_bar = (ats > ats_median).astype(float)

                out["large_bar_delta"] = taker_delta * is_large_bar
                out["small_bar_delta"] = taker_delta * (1 - is_large_bar)
                out["large_bar_delta_zscore"] = _zscore(out["large_bar_delta"])
                out["large_small_bar_div"] = out["large_bar_delta"] - out["small_bar_delta"]
            else:
                out["taker_delta_proxy"] = taker_delta
                out["taker_delta_proxy_zscore"] = _zscore(taker_delta)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Short-Horizon Taker Imbalance Persistence
# ═══════════════════════════════════════════════════════════════════════

def compute_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Taker buy/sell imbalance at multiple short horizons.

    Key insight: a single bar of imbalance is noise.
    PERSISTENT imbalance over 3-5 bars is signal.
    """
    out = pd.DataFrame(index=df.index)

    # Compute taker imbalance ratio
    if "taker_buy_vol" in df.columns and "volume" in df.columns:
        vol = df["volume"].replace(0, np.nan)
        taker_buy = df["taker_buy_vol"]
        imb = (2 * taker_buy / vol) - 1  # range: [-1, 1], 0 = balanced
    elif "taker_delta_ratio" in df.columns:
        imb = df["taker_delta_ratio"]
    elif "cg_taker_delta" in df.columns and "cg_taker_buy" in df.columns:
        total = df["cg_taker_buy"] + df["cg_taker_sell"]
        imb = df["cg_taker_delta"] / total.replace(0, np.nan)
    else:
        return out

    out["imb_1b"] = imb
    out["imb_3b"] = imb.rolling(3, min_periods=1).mean()
    out["imb_5b"] = imb.rolling(5, min_periods=2).mean()
    out["imb_8b"] = imb.rolling(8, min_periods=3).mean()

    # Std of imbalance (low std + nonzero mean = persistent directional flow)
    out["imb_std_5b"] = imb.rolling(5, min_periods=2).std()

    # Persistence: how many consecutive bars is imbalance same sign?
    out["imb_sign_persistence"] = _sign_persistence(imb, 8)

    # Imbalance acceleration: is it getting MORE one-sided?
    out["imb_slope_4b"] = _slope(imb, 4)

    # Z-score of 3b imbalance (vs 24h history)
    out["imb_3b_zscore"] = _zscore(out["imb_3b"])

    return out


# ═══════════════════════════════════════════════════════════════════════
# Group 3: Price Response / Absorption Proxy
# ═══════════════════════════════════════════════════════════════════════

def compute_absorption_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absorption = large flow with small price response.

    Directional version: separate buy-side and sell-side absorption.
    - High buy absorption → hidden sell wall → bearish
    - High sell absorption → hidden buy wall → bullish (contrarian)
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    price_move = log_ret.abs()

    # Expected move (half a vol)
    if "realized_vol_20b" in df.columns:
        expected_move = df["realized_vol_20b"] * 0.5
    else:
        expected_move = price_move.rolling(20, min_periods=5).mean() * 1.5

    expected_move = expected_move.replace(0, np.nan)

    # Under-reaction ratio: 1 = price didn't move at all, 0 = moved expected amount
    under_reaction = (1 - price_move / expected_move).clip(-2, 2)

    # Get taker delta in USD terms
    if "taker_buy_vol" in df.columns and "volume" in df.columns:
        taker_buy = df["taker_buy_vol"] * close
        taker_sell = (df["volume"] - df["taker_buy_vol"]) * close
        taker_delta = taker_buy - taker_sell
    elif "cg_taker_delta" in df.columns:
        taker_delta = df["cg_taker_delta"]
        taker_buy = df.get("cg_taker_buy", taker_delta.clip(lower=0))
        taker_sell = df.get("cg_taker_sell", (-taker_delta).clip(lower=0))
    else:
        return out

    # Buy-side absorption: strong buying + price didn't go up much
    buy_pressure = taker_buy * under_reaction
    sell_pressure = taker_sell * under_reaction

    # Directional absorption: positive = buy-side absorbed, negative = sell-side
    out["absorption_buy"] = buy_pressure
    out["absorption_sell"] = sell_pressure
    out["absorption_net"] = buy_pressure - sell_pressure  # >0: buy absorbed (bearish)
    out["absorption_net_zscore"] = _zscore(out["absorption_net"])

    # Price impact asymmetry
    # When taker buys: how much did price actually move up?
    # When taker sells: how much did price move down?
    buy_bars = taker_delta > 0
    sell_bars = taker_delta < 0

    buy_impact = pd.Series(np.nan, index=df.index)
    sell_impact = pd.Series(np.nan, index=df.index)

    buy_impact[buy_bars] = log_ret[buy_bars] / (taker_delta[buy_bars] / taker_delta[buy_bars].abs().mean()).replace(0, np.nan)
    sell_impact[sell_bars] = log_ret[sell_bars] / (taker_delta[sell_bars] / taker_delta[sell_bars].abs().mean()).replace(0, np.nan)

    out["buy_flow_price_impact"] = buy_impact.rolling(8, min_periods=2).mean()
    out["sell_flow_price_impact"] = sell_impact.rolling(8, min_periods=2).mean()

    # Impact asymmetry: positive = buys move price more than sells (bullish microstructure)
    out["impact_asymmetry"] = (
        out["buy_flow_price_impact"].fillna(0) -
        out["sell_flow_price_impact"].fillna(0)
    )

    return out


# ═══════════════════════════════════════════════════════════════════════
# Group 4: Short-Horizon Momentum / Reversal Context
# ═══════════════════════════════════════════════════════════════════════

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-horizon price momentum and reversal signals.

    Uses only OHLC data. These are the simplest directional features.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # Short-horizon returns (signed — directional by definition)
    out["ret_1b"] = close.pct_change(1)
    out["ret_2b"] = close.pct_change(2)
    out["ret_3b"] = close.pct_change(3)
    out["ret_5b"] = close.pct_change(5)

    # Signed range: how directional was recent price action?
    if "high" in df.columns and "low" in df.columns:
        high = df["high"]
        low = df["low"]

        # Wick asymmetry: upper_wick / total_range vs lower_wick / total_range
        total_range = (high - low).replace(0, np.nan)
        upper_wick = high - pd.concat([df["open"], close], axis=1).max(axis=1)
        lower_wick = pd.concat([df["open"], close], axis=1).min(axis=1) - low

        out["wick_asymmetry"] = (upper_wick - lower_wick) / total_range
        # Positive = more upper wick (rejection of highs → bearish)
        # Negative = more lower wick (rejection of lows → bullish)

        # Body ratio: how much of the range is body vs wicks
        body = (close - df["open"]).abs()
        out["body_ratio"] = body / total_range
        # High body ratio = strong conviction bar

        # Signed body as fraction of range
        out["signed_body_ratio"] = (close - df["open"]) / total_range

    # Return reversal score: recent return vs preceding return
    r1 = out["ret_1b"]
    r3 = out["ret_3b"]
    out["reversal_1v3"] = -r1 * r3.shift(1)  # positive = reversal setup

    # Cumulative return momentum with sign persistence
    out["ret_3b_persistence"] = _sign_persistence(out["ret_1b"], 8)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Group 5: Funding / Sentiment Extremes (directional contrarian)
# ═══════════════════════════════════════════════════════════════════════

def compute_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contrarian sentiment features from Coinglass data.

    Extreme funding / long-short ratio → mean-reversion signal.
    """
    out = pd.DataFrame(index=df.index)

    # Funding rate: extreme positive → too many longs → bearish contrarian
    if "cg_funding_close" in df.columns:
        f = df["cg_funding_close"]
        out["funding_zscore"] = _zscore(f, 24)
        out["funding_extreme"] = _zscore(f, 72)  # longer-term extreme

    # Long/short ratio: high ratio → crowded long → bearish
    if "cg_ls_ratio" in df.columns:
        ls = df["cg_ls_ratio"]
        out["ls_ratio_zscore"] = _zscore(ls, 24)

    # Crowding: funding * ls_ratio — double confirmation of one-sided positioning
    if "cg_crowding" in df.columns:
        out["crowding_zscore"] = _zscore(df["cg_crowding"], 24)

    # Liquidation imbalance: more long liq → bearish pressure
    if "cg_liq_imbalance" in df.columns:
        out["liq_imbalance_zscore"] = _zscore(df["cg_liq_imbalance"], 24)
        out["liq_imbalance_slope"] = _slope(df["cg_liq_imbalance"], 4)

    # OI delta direction: rising OI + rising price = confirmed trend
    if "cg_oi_delta_zscore" in df.columns and "close" in df.columns:
        ret = df["close"].pct_change(1)
        oi_z = df["cg_oi_delta_zscore"]
        # Positive = OI and price moving same direction (confirming)
        out["oi_price_confirm"] = oi_z * np.sign(ret)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Group 6: Order Book Imbalance (stub — future integration)
# ═══════════════════════════════════════════════════════════════════════

def compute_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order book imbalance features.

    Currently a stub — returns empty DataFrame.
    When real-time order book data is available, populate these columns:

    Expected columns:
        obi_l1           - Level 1 bid/ask imbalance
        obi_l5           - Top-5 level imbalance
        obi_l10          - Top-10 level imbalance
        depth_pressure   - Near-touch depth ratio (bid 0.1% / ask 0.1%)
        obi_change       - 1-bar change in OBI
        obi_persistence  - Sign persistence of OBI

    Data source: Binance depth snapshots (indicator/snapshot_collector.py)
    """
    out = pd.DataFrame(index=df.index)

    # Check if snapshot-derived columns exist
    if "depth_imbalance" in df.columns:
        out["obi_l5"] = df["depth_imbalance"]
        out["obi_l5_zscore"] = _zscore(df["depth_imbalance"])

    if "near_imbalance" in df.columns:
        out["obi_l1"] = df["near_imbalance"]
        out["obi_l1_zscore"] = _zscore(df["near_imbalance"])
        out["obi_change"] = df["near_imbalance"].diff()
        out["obi_persistence"] = _sign_persistence(df["near_imbalance"], 8)

    # Stub columns for future integration
    # These will be populated when L2 data pipeline is ready
    # out["obi_l10"] = ...
    # out["depth_pressure"] = ...

    return out


# ═══════════════════════════════════════════════════════════════════════
# Unified builder
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "large_trade": True,
    "imbalance": True,
    "absorption": True,
    "momentum": True,
    "sentiment": True,
    "orderbook": True,  # stub — only populates if columns exist
}


def build_direction_feature_set(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Build the complete direction-specific feature set.

    Parameters
    ----------
    df : source DataFrame with OHLCV + Coinglass + aggTrades columns
    config : dict controlling which feature groups to include
             (default: all enabled)

    Returns
    -------
    DataFrame with only direction-specific features (no OHLCV, no targets)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    parts = []

    if cfg["large_trade"]:
        parts.append(compute_large_trade_features(df))

    if cfg["imbalance"]:
        parts.append(compute_imbalance_features(df))

    if cfg["absorption"]:
        parts.append(compute_absorption_features(df))

    if cfg["momentum"]:
        parts.append(compute_momentum_features(df))

    if cfg["sentiment"]:
        parts.append(compute_sentiment_features(df))

    if cfg["orderbook"]:
        parts.append(compute_orderbook_features(df))

    if not parts:
        return pd.DataFrame(index=df.index)

    result = pd.concat(parts, axis=1)

    # Remove any fully-NaN columns
    result = result.dropna(axis=1, how="all")

    return result
