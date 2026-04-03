"""
Build features from live API data (Binance 1h klines + Coinglass 1h).

Replicates the formulas from ic_scan_1h.py but operates on live DataFrames.
All Coinglass data is native 1h — no forward-fill needed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

logger = logging.getLogger(__name__)


def _entropy_10bin(x):
    """Entropy of a series binned into 10 buckets."""
    counts, _ = np.histogram(x, bins=10)
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return sp_entropy(probs)

ZSCORE_WIN = 24   # 24h
SHORT_WIN = 4     # 4h
LONG_WIN = 24     # 24h


def _zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    mu = s.rolling(win, min_periods=4).mean()
    sd = s.rolling(win, min_periods=4).std().replace(0, np.nan)
    return (s - mu) / sd


def build_live_features(klines: pd.DataFrame,
                        cg_data: dict[str, pd.DataFrame],
                        depth: dict | None = None,
                        aggtrades: dict | None = None) -> pd.DataFrame:
    """
    Build feature DataFrame from live 1h API data.

    Parameters:
        klines: Binance 1h klines (OHLCV + taker_buy_vol + trade_count)
        cg_data: dict from fetch_coinglass() (native 1h data)
        depth: dict from fetch_binance_depth() (order book snapshot, optional)
        aggtrades: dict from fetch_binance_aggtrades() (large trade stats, optional)

    Returns:
        DataFrame indexed by datetime with all features.
    """
    df = klines[["open", "high", "low", "close", "volume",
                 "taker_buy_vol", "taker_buy_quote", "trade_count"]].copy()

    # ── Binance quote volume (NEW) ─────────────────────────────────────────
    if "quote_vol" in klines.columns:
        qv = pd.to_numeric(klines["quote_vol"], errors="coerce")
        df["quote_vol_zscore"] = _zscore(qv)
        df["quote_vol_ratio"] = qv / qv.rolling(LONG_WIN, min_periods=4).mean().replace(0, np.nan)

    # ── Kline-derived features ──────────────────────────────────────────────
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["price_change_pct"] = df["close"].pct_change()
    df["realized_vol_20b"] = df["log_return"].rolling(20, min_periods=4).std()
    df["return_skew"] = df["log_return"].rolling(LONG_WIN, min_periods=10).skew()
    df["return_kurtosis"] = df["log_return"].rolling(LONG_WIN, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )

    # Taker delta
    taker_sell = df["volume"] - df["taker_buy_vol"]
    taker_delta = df["taker_buy_vol"] - taker_sell
    df["taker_delta_ratio"] = taker_delta / df["volume"].replace(0, np.nan)

    df["taker_delta_ma_4h"] = taker_delta.rolling(SHORT_WIN, min_periods=2).mean()
    df["taker_delta_std_4h"] = taker_delta.rolling(SHORT_WIN, min_periods=2).std()
    df["taker_delta_ma_24h"] = taker_delta.rolling(LONG_WIN, min_periods=2).mean()
    df["taker_delta_std_24h"] = taker_delta.rolling(LONG_WIN, min_periods=2).std()

    df["volume_ma_4h"] = df["volume"].rolling(SHORT_WIN, min_periods=2).mean()
    df["volume_ma_24h"] = df["volume"].rolling(LONG_WIN, min_periods=2).mean()

    # Return lags
    for lag in range(1, 11):
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

    # ── Coinglass features (native 1h — no forward-fill needed) ────────────
    _inject_coinglass(df, cg_data)

    # ── Momentum / slope / divergence features ─────────────────────────────
    for col in ["cg_oi_delta", "cg_taker_delta", "cg_funding_close"]:
        if col in df.columns:
            df[f"{col}_slope_4h"] = df[col].rolling(4, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                raw=False,
            )
            df[f"{col}_mom_1h"] = df[col] - df[col].shift(1)

    if "cg_oi_delta_zscore" in df.columns:
        ret_cum = df["log_return"].rolling(4, min_periods=2).sum()
        ret_z = _zscore(ret_cum)
        df["oi_price_divergence"] = df["cg_oi_delta_zscore"] - ret_z

    if "cg_funding_close_zscore" in df.columns and "cg_taker_delta_zscore" in df.columns:
        df["funding_taker_align"] = (
            df["cg_funding_close_zscore"] * df["cg_taker_delta_zscore"]
        )

    df["vol_regime"] = df["realized_vol_20b"] / df[
        "realized_vol_20b"
    ].rolling(LONG_WIN, min_periods=10).mean().replace(0, np.nan)

    # ── Volume dynamics (IC-validated new features) ─────────────────────────
    vol_ma_4h = df["volume"].rolling(SHORT_WIN, min_periods=2).mean()
    vol_ma_24h = df["volume"].rolling(LONG_WIN, min_periods=4).mean()
    df["vol_acceleration"] = vol_ma_4h / vol_ma_24h.replace(0, np.nan)
    df["vol_kurtosis"] = df["volume"].rolling(LONG_WIN, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )
    df["vol_entropy"] = df["volume"].rolling(LONG_WIN, min_periods=10).apply(
        lambda x: _entropy_10bin(x), raw=False
    )

    # Squeeze proxy (extreme funding + high OI change + CVD reversal)
    if all(c in df.columns for c in [
        "cg_funding_close_zscore", "cg_oi_delta_zscore"
    ]):
        cvd_delta_z = _zscore(taker_delta.diff())
        df["squeeze_proxy"] = (
            df["cg_funding_close_zscore"].abs() *
            df["cg_oi_delta_zscore"].abs() *
            np.sign(-df["cg_funding_close_zscore"] * cvd_delta_z)
        )

    # ── Absorption / Delta-Price Divergence ──────────────────────────────
    # Large delta but small price move = limit order absorption (institutional)
    # Positive = buy-side absorption (bullish), negative = sell-side absorption (bearish)
    if "realized_vol_20b" in df.columns:
        expected_move = df["realized_vol_20b"] * 0.5  # half ATR as expected move
        price_return_abs = df["log_return"].abs()
        under_reaction = (1 - price_return_abs / expected_move.replace(0, np.nan)).clip(-2, 2)
        # taker_delta is in BTC; convert to directional USD
        taker_delta_usd = taker_delta * df["close"]
        df["absorption_score"] = taker_delta_usd * under_reaction
        df["absorption_zscore"] = _zscore(df["absorption_score"])

        # Rolling absorption momentum (sustained absorption = strong signal)
        df["absorption_ma_4h"] = df["absorption_score"].rolling(SHORT_WIN, min_periods=2).mean()
        df["absorption_cumsum_4h"] = df["absorption_score"].rolling(SHORT_WIN, min_periods=1).sum()

    # ── Liquidation enhanced features ─────────────────────────────────────
    if "cg_liq_total" in df.columns:
        # Surge detection: current bar vs rolling mean
        liq_ma = df["cg_liq_total"].rolling(LONG_WIN, min_periods=4).mean().replace(0, np.nan)
        df["cg_liq_surge"] = df["cg_liq_total"] / liq_ma

        # Cumulative liquidation pressure (4h / 8h windows)
        df["cg_liq_long_cum_4h"] = df["cg_liq_long"].rolling(SHORT_WIN, min_periods=1).sum()
        df["cg_liq_short_cum_4h"] = df["cg_liq_short"].rolling(SHORT_WIN, min_periods=1).sum()
        df["cg_liq_long_cum_8h"] = df["cg_liq_long"].rolling(8, min_periods=1).sum()
        df["cg_liq_short_cum_8h"] = df["cg_liq_short"].rolling(8, min_periods=1).sum()

        # Cascade count: consecutive bars with above-average liquidation
        above_avg = (df["cg_liq_total"] > liq_ma).astype(int)
        # Count consecutive 1s using cumsum trick
        reset = above_avg.eq(0).cumsum()
        df["cg_liq_cascade"] = above_avg.groupby(reset).cumsum()

        # Liquidation asymmetry slope (trend in long/short imbalance)
        if "cg_liq_imbalance" in df.columns:
            df["cg_liq_imbalance_slope_4h"] = df["cg_liq_imbalance"].rolling(
                SHORT_WIN, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False)

        # Liquidation vs volume ratio (market impact proxy)
        if "volume" in df.columns:
            vol_usd = df["close"] * df["volume"]
            df["cg_liq_vs_vol"] = df["cg_liq_total"] / vol_usd.replace(0, np.nan)

        # Liquidation surge z-score
        df["cg_liq_surge_zscore"] = _zscore(df["cg_liq_surge"])

    # ── Large trade flow proxy (from kline trade_count / volume) ──────────
    if "trade_count" in df.columns and "volume" in df.columns:
        tc = df["trade_count"].replace(0, np.nan)
        # Average trade size in BTC
        df["avg_trade_size"] = df["volume"] / tc
        df["avg_trade_size_zscore"] = _zscore(df["avg_trade_size"])

        # Average notional per trade (USD)
        if "quote_vol" in klines.columns:
            qv = pd.to_numeric(klines["quote_vol"], errors="coerce").reindex(df.index)
            df["avg_trade_notional"] = qv / tc
            df["avg_trade_notional_zscore"] = _zscore(df["avg_trade_notional"])

        # Trade intensity: many small trades vs few large (high = many small)
        df["trade_intensity"] = tc / df["volume"].replace(0, np.nan)
        df["trade_intensity_zscore"] = _zscore(df["trade_intensity"])

        # Trade count acceleration
        tc_ma_4h = tc.rolling(SHORT_WIN, min_periods=2).mean()
        tc_ma_24h = tc.rolling(LONG_WIN, min_periods=4).mean()
        df["trade_count_accel"] = tc_ma_4h / tc_ma_24h.replace(0, np.nan)

        # Taker buy concentration: when avg trade size is high AND taker delta is directional
        if "taker_delta_ratio" in df.columns:
            df["large_taker_signal"] = (
                df["avg_trade_size_zscore"] * df["taker_delta_ratio"]
            )

    # ── Order book depth features (snapshot, appended to last bar only) ───
    if depth and isinstance(depth, dict) and "depth_imbalance" in depth:
        # These are point-in-time snapshots — only valid for the latest bar
        # Set NaN for all historical bars, then fill last bar
        for col in ["depth_imbalance", "near_imbalance", "spread_bps"]:
            df[col] = np.nan
            if col in depth:
                df.iloc[-1, df.columns.get_loc(col)] = depth[col]
        # Depth ratio: bid/ask depth
        bid = depth.get("bid_depth_usd", 0)
        ask = depth.get("ask_depth_usd", 0)
        df["depth_bid_ask_ratio"] = np.nan
        if ask > 0:
            df.iloc[-1, df.columns.get_loc("depth_bid_ask_ratio")] = bid / ask

    # ── Large trade aggTrades features (snapshot, last bar only) ──────────
    if aggtrades and isinstance(aggtrades, dict) and "large_ratio" in aggtrades:
        for col in ["large_ratio", "large_buy_ratio", "large_delta_usd",
                     "avg_trade_usd"]:
            df[col] = np.nan
            if col in aggtrades:
                df.iloc[-1, df.columns.get_loc(col)] = aggtrades[col]
        # Large vs small delta divergence
        large_d = aggtrades.get("large_delta_usd", 0)
        small_d = aggtrades.get("small_delta_usd", 0)
        df["large_small_divergence"] = np.nan
        if abs(large_d) + abs(small_d) > 0:
            # +1 = large and small agree, -1 = diverge
            df.iloc[-1, df.columns.get_loc("large_small_divergence")] = (
                np.sign(large_d) * np.sign(small_d) if large_d != 0 and small_d != 0 else 0
            )

    # ── Time-of-day features (cyclical encoding) ───────────────────────────
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    weekday = df.index.dayofweek
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    logger.info("Live features: %d bars x %d columns", len(df), len(df.columns))
    return df


def _inject_coinglass(df: pd.DataFrame, cg_data: dict[str, pd.DataFrame]):
    """Inject Coinglass features into df (in-place). Native 1h, use merge_asof."""

    def _merge_cg(cg_df: pd.DataFrame, cols: dict[str, str]):
        if cg_df.empty:
            for target in cols.values():
                df[target] = np.nan
            return
        for src, target in cols.items():
            if src in cg_df.columns:
                left = df[["close"]].reset_index()
                right = cg_df[[src]].reset_index().rename(columns={src: target})
                # Unify datetime precision to avoid merge_asof type mismatch
                left["dt"] = pd.to_datetime(left["dt"], utc=True)
                right["dt"] = pd.to_datetime(right["dt"], utc=True)
                merged = pd.merge_asof(
                    left, right,
                    left_on="dt", right_on="dt", direction="backward",
                ).set_index("dt")
                df[target] = merged[target]
            else:
                df[target] = np.nan

    # OI (per-exchange) — with OHLC for range/shadow features
    oi = cg_data.get("oi", pd.DataFrame())
    _merge_cg(oi, {"close": "cg_oi_close"})
    # Also merge open/high/low for OI range features
    oi_ohlc = {}
    for col in ["open", "high", "low"]:
        if not oi.empty and col in oi.columns:
            oi_ohlc[col] = f"_oi_{col}"
    if oi_ohlc:
        _merge_cg(oi, oi_ohlc)

    if "cg_oi_close" in df.columns:
        df["cg_oi_delta"] = df["cg_oi_close"].diff()
        df["cg_oi_accel"] = df["cg_oi_delta"].diff()

        # NEW: OI multi-window pct change (IC -0.078 at 8h)
        for w in [4, 8, 12, 24]:
            df[f"cg_oi_close_pctchg_{w}h"] = df["cg_oi_close"].pct_change(w)

        # NEW: OI intra-bar range and shadow
        if "_oi_high" in df.columns and "_oi_low" in df.columns:
            oi_range = df["_oi_high"] - df["_oi_low"]
            df["cg_oi_range_zscore"] = _zscore(oi_range)
            df["cg_oi_range_pct"] = oi_range / df["cg_oi_close"].replace(0, np.nan)
            if "_oi_open" in df.columns:
                df["cg_oi_upper_shadow"] = (
                    df["_oi_high"] - np.maximum(df["_oi_open"], df["cg_oi_close"])
                )
        df.drop(columns=["_oi_open", "_oi_high", "_oi_low"], errors="ignore", inplace=True)

    # OI aggregated
    oi_agg = cg_data.get("oi_agg", pd.DataFrame())
    _merge_cg(oi_agg, {"close": "cg_oi_agg_close"})
    if "cg_oi_agg_close" in df.columns:
        df["cg_oi_agg_delta"] = df["cg_oi_agg_close"].diff()
    if "cg_oi_close" in df.columns and "cg_oi_agg_close" in df.columns:
        df["cg_oi_binance_share"] = (
            df["cg_oi_close"] / df["cg_oi_agg_close"].replace(0, np.nan)
        )
        # NEW: Binance share z-score (IC -0.060)
        df["cg_oi_binance_share_zscore"] = _zscore(df["cg_oi_binance_share"])

    # Liquidation
    liq = cg_data.get("liquidation", pd.DataFrame())
    liq_cols = {}
    for src, tgt in [
        ("longVolUsd", "cg_liq_long"), ("long_liquidation_usd", "cg_liq_long"),
        ("shortVolUsd", "cg_liq_short"), ("short_liquidation_usd", "cg_liq_short"),
        ("volUsd", "cg_liq_total"),
    ]:
        liq_cols[src] = tgt
    _merge_cg(liq, liq_cols)
    if "cg_liq_long" in df.columns and "cg_liq_short" in df.columns:
        total = (df["cg_liq_long"] + df["cg_liq_short"]).replace(0, np.nan)
        if "cg_liq_total" not in df.columns or df["cg_liq_total"].isna().all():
            df["cg_liq_total"] = total
        df["cg_liq_ratio"] = df["cg_liq_long"] / total
        df["cg_liq_imbalance"] = (df["cg_liq_long"] - df["cg_liq_short"]) / total

    # Long/Short ratio (top trader)
    ls = cg_data.get("long_short", pd.DataFrame())
    ls_cols = {}
    for src, tgt in [
        ("longRatio", "cg_ls_long_pct"), ("top_account_long_percent", "cg_ls_long_pct"),
        ("shortRatio", "cg_ls_short_pct"), ("top_account_short_percent", "cg_ls_short_pct"),
    ]:
        ls_cols[src] = tgt
    _merge_cg(ls, ls_cols)
    if "cg_ls_long_pct" in df.columns and "cg_ls_short_pct" in df.columns:
        df["cg_ls_ratio"] = (
            df["cg_ls_long_pct"] / df["cg_ls_short_pct"].replace(0, np.nan)
        )

    # Global Long/Short ratio (new endpoint)
    gls = cg_data.get("global_ls", pd.DataFrame())
    gls_cols = {}
    for src, tgt in [
        ("longRatio", "cg_gls_long_pct"), ("global_account_long_percent", "cg_gls_long_pct"),
        ("shortRatio", "cg_gls_short_pct"), ("global_account_short_percent", "cg_gls_short_pct"),
    ]:
        gls_cols[src] = tgt
    _merge_cg(gls, gls_cols)
    if "cg_gls_long_pct" in df.columns and "cg_gls_short_pct" in df.columns:
        df["cg_gls_ratio"] = (
            df["cg_gls_long_pct"] / df["cg_gls_short_pct"].replace(0, np.nan)
        )
    if "cg_ls_ratio" in df.columns and "cg_gls_ratio" in df.columns:
        df["cg_ls_divergence"] = df["cg_ls_ratio"] - df["cg_gls_ratio"]

    # Funding rate
    funding = cg_data.get("funding", pd.DataFrame())
    _merge_cg(funding, {"close": "cg_funding_close"})
    if not funding.empty and "high" in funding.columns and "low" in funding.columns:
        _merge_cg(funding, {"high": "_fh", "low": "_fl"})
        df["cg_funding_range"] = df.get("_fh", 0) - df.get("_fl", 0)
        df.drop(columns=["_fh", "_fl"], errors="ignore", inplace=True)
    else:
        df["cg_funding_range"] = np.nan

    # Taker buy/sell
    taker = cg_data.get("taker", pd.DataFrame())
    taker_cols = {}
    for src, tgt in [
        ("buyVolUsd", "cg_taker_buy"), ("taker_buy_volume_usd", "cg_taker_buy"),
        ("sellVolUsd", "cg_taker_sell"), ("taker_sell_volume_usd", "cg_taker_sell"),
    ]:
        taker_cols[src] = tgt
    _merge_cg(taker, taker_cols)
    if "cg_taker_buy" in df.columns and "cg_taker_sell" in df.columns:
        df["cg_taker_delta"] = df["cg_taker_buy"] - df["cg_taker_sell"]
        total = (df["cg_taker_buy"] + df["cg_taker_sell"]).replace(0, np.nan)
        df["cg_taker_ratio"] = df["cg_taker_buy"] / total

    # Z-scores
    for col in [
        "cg_oi_delta", "cg_oi_agg_delta", "cg_liq_imbalance",
        "cg_ls_ratio", "cg_taker_delta", "cg_funding_close",
        "cg_oi_close", "cg_liq_total", "cg_gls_ratio", "cg_ls_divergence",
    ]:
        if col in df.columns:
            df[f"{col}_zscore"] = _zscore(df[col])

    # Cross features
    if "cg_liq_total" in df.columns and "cg_oi_close" in df.columns:
        df["cg_liq_x_oi"] = df["cg_liq_total"] / df["cg_oi_close"].replace(0, np.nan)
    if "cg_funding_close" in df.columns and "cg_ls_ratio" in df.columns:
        df["cg_crowding"] = df["cg_funding_close"] * df["cg_ls_ratio"]
    if "cg_taker_delta_zscore" in df.columns and "cg_oi_delta_zscore" in df.columns:
        df["cg_conviction"] = df["cg_taker_delta_zscore"] * df["cg_oi_delta_zscore"]

    # ── New endpoints (Startup plan upgrade) ─────────────────────────────

    # Coinbase Premium
    cb = cg_data.get("coinbase_premium", pd.DataFrame())
    _merge_cg(cb, {"premium": "cg_cb_premium", "premium_rate": "cg_cb_premium_rate"})

    # Bitfinex Margin
    bfx = cg_data.get("bitfinex_margin", pd.DataFrame())
    _merge_cg(bfx, {"long_quantity": "cg_bfx_margin_long", "short_quantity": "cg_bfx_margin_short"})
    if "cg_bfx_margin_long" in df.columns and "cg_bfx_margin_short" in df.columns:
        total_bfx = (df["cg_bfx_margin_long"] + df["cg_bfx_margin_short"]).replace(0, np.nan)
        df["cg_bfx_margin_ratio"] = df["cg_bfx_margin_long"] / total_bfx
        df["cg_bfx_margin_delta"] = df["cg_bfx_margin_long"] - df["cg_bfx_margin_short"]

    # Top L/S Position Ratio (different from account ratio!)
    pos = cg_data.get("top_ls_position", pd.DataFrame())
    _merge_cg(pos, {
        "top_position_long_percent": "cg_pos_long_pct",
        "top_position_short_percent": "cg_pos_short_pct",
        "top_position_long_short_ratio": "cg_pos_ls_ratio",
    })

    # Futures CVD Aggregated (cross-exchange)
    fcvd = cg_data.get("futures_cvd_agg", pd.DataFrame())
    _merge_cg(fcvd, {
        "agg_taker_buy_vol": "cg_fcvd_buy",
        "agg_taker_sell_vol": "cg_fcvd_sell",
        "cum_vol_delta": "cg_fcvd_cum",
    })
    if "cg_fcvd_buy" in df.columns and "cg_fcvd_sell" in df.columns:
        df["cg_fcvd_delta"] = df["cg_fcvd_buy"] - df["cg_fcvd_sell"]

    # Spot CVD Aggregated (cross-exchange)
    scvd = cg_data.get("spot_cvd_agg", pd.DataFrame())
    _merge_cg(scvd, {
        "agg_taker_buy_vol": "cg_scvd_buy",
        "agg_taker_sell_vol": "cg_scvd_sell",
        "cum_vol_delta": "cg_scvd_cum",
    })
    if "cg_scvd_buy" in df.columns and "cg_scvd_sell" in df.columns:
        df["cg_scvd_delta"] = df["cg_scvd_buy"] - df["cg_scvd_sell"]

    # Liquidation Aggregated (cross-exchange)
    liq_agg = cg_data.get("liq_agg", pd.DataFrame())
    _merge_cg(liq_agg, {
        "aggregated_long_liquidation_usd": "cg_liq_agg_long",
        "aggregated_short_liquidation_usd": "cg_liq_agg_short",
    })
    if "cg_liq_agg_long" in df.columns and "cg_liq_agg_short" in df.columns:
        total_la = (df["cg_liq_agg_long"] + df["cg_liq_agg_short"]).replace(0, np.nan)
        df["cg_liq_agg_total"] = total_la
        df["cg_liq_agg_imbalance"] = (df["cg_liq_agg_long"] - df["cg_liq_agg_short"]) / total_la

    # OI Coin-Margin
    oicm = cg_data.get("oi_coin_margin", pd.DataFrame())
    _merge_cg(oicm, {"close": "cg_oi_cm_close"})
    if "cg_oi_cm_close" in df.columns:
        df["cg_oi_cm_delta"] = df["cg_oi_cm_close"].diff()

    # ── Z-scores for new endpoints ───────────────────────────────────────
    for col in [
        "cg_cb_premium_rate", "cg_bfx_margin_ratio", "cg_bfx_margin_delta",
        "cg_pos_ls_ratio", "cg_fcvd_delta", "cg_scvd_delta",
        "cg_liq_agg_imbalance", "cg_oi_cm_delta",
    ]:
        if col in df.columns:
            df[f"{col}_zscore"] = _zscore(df[col])

    # ── New cross-source features ────────────────────────────────────────
    # Spot vs Futures CVD divergence
    if "cg_scvd_delta_zscore" in df.columns and "cg_fcvd_delta_zscore" in df.columns:
        df["cg_spot_futures_cvd_divergence"] = (
            df["cg_scvd_delta_zscore"] - df["cg_fcvd_delta_zscore"]
        )
    # Account ratio vs Position ratio divergence
    if "cg_ls_ratio" in df.columns and "cg_pos_ls_ratio" in df.columns:
        df["cg_pos_account_divergence"] = df["cg_ls_ratio"] - df["cg_pos_ls_ratio"]
    # Binance share of total liquidations
    if "cg_liq_total" in df.columns and "cg_liq_agg_total" in df.columns:
        df["cg_liq_exchange_vs_agg"] = (
            df["cg_liq_total"] / df["cg_liq_agg_total"].replace(0, np.nan)
        )
    # Coin-margin OI share
    if "cg_oi_cm_close" in df.columns and "cg_oi_agg_close" in df.columns:
        df["cg_oi_cm_vs_usd"] = (
            df["cg_oi_cm_close"] / df["cg_oi_agg_close"].replace(0, np.nan)
        )
    # Bitfinex margin × Funding alignment
    if "cg_bfx_margin_ratio_zscore" in df.columns and "cg_funding_close_zscore" in df.columns:
        df["cg_margin_funding_align"] = (
            df["cg_bfx_margin_ratio_zscore"] * df["cg_funding_close_zscore"]
        )
