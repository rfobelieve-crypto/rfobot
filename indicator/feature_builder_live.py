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
                        aggtrades: dict | None = None,
                        options_data: dict | None = None) -> pd.DataFrame:
    """
    Build feature DataFrame from live 1h API data.

    Parameters:
        klines: Binance 1h klines (OHLCV + taker_buy_vol + trade_count)
        cg_data: dict from fetch_coinglass() (native 1h data)
        depth: dict from fetch_binance_depth() (order book snapshot, optional)
        aggtrades: dict from fetch_binance_aggtrades() (large trade stats, optional)
        options_data: dict from fetch_deribit_dvol() + fetch_deribit_options_summary() (optional)

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
    def _safe_slope(x):
        if len(x) < 2 or np.isnan(x).any() or np.std(x) == 0:
            return 0.0
        try:
            return np.polyfit(range(len(x)), x, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    for col in ["cg_oi_delta", "cg_taker_delta", "cg_funding_close"]:
        if col in df.columns:
            df[f"{col}_slope_4h"] = df[col].rolling(4, min_periods=2).apply(
                _safe_slope, raw=False,
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

    # ── Directional regime indicators (matches production regime classifier) ──
    # Reproduces vol_pct > 0.6 + ret_24h direction logic from indicator_engine.py
    # Gives XGB ability to do regime-aware tree splits on OTHER features
    _vol_24h = df["log_return"].rolling(LONG_WIN, min_periods=12).std()
    _vol_pct = _vol_24h.expanding(min_periods=72).rank(pct=True)
    _ret_24h = df["close"].pct_change(LONG_WIN)
    df["is_trending_bull"] = ((_vol_pct > 0.6) & (_ret_24h > 0.005)).astype(float)
    df["is_trending_bear"] = ((_vol_pct > 0.6) & (_ret_24h < -0.005)).astype(float)
    # During warmup (vol_pct NaN) keep them as 0 (CHOPPY default)

    # ── Volume dynamics (IC-validated new features) ─────────────────────────
    vol_ma_4h = df["volume"].rolling(SHORT_WIN, min_periods=2).mean()
    vol_ma_24h = df["volume"].rolling(LONG_WIN, min_periods=4).mean()
    df["vol_acceleration"] = vol_ma_4h / vol_ma_24h.replace(0, np.nan)
    df["vol_kurtosis"] = df["volume"].rolling(LONG_WIN, min_periods=10).apply(
        lambda x: x.kurtosis(), raw=False
    )

    # ── Regime-conditional interactions (zero out where base feature is dead) ──
    # vol_kurtosis IC dies in TRENDING_BEAR (CHOPPY +0.106, BULL +0.055, BEAR -0.012)
    # IC validated +0.0601 (p=2e-4), train +0.085 / test +0.041 STABLE
    df["vol_kurt_non_bear"] = df["vol_kurtosis"] * (1.0 - df["is_trending_bear"])

    # cg_oi_close_pctchg_8h IC dies in TRENDING_BULL (CHOPPY -0.071, BULL +0.004, BEAR -0.066)
    # IC validated -0.0676 (p=2e-5), train -0.060 / test -0.072 STABLE
    if "cg_oi_close_pctchg_8h" in df.columns:
        df["oi_8h_non_bull"] = df["cg_oi_close_pctchg_8h"] * (1.0 - df["is_trending_bull"])

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

        # Long-liq exhaustion z-score: regime-normalized 4h long liquidation
        # IC validated +0.0667 (p=3e-5) vs y_return_4h, +0.0798 vs y_dir
        # Improves on raw cg_liq_long_cum_4h (+0.0481) by removing vol-regime bias
        _ll_cum = df["cg_liq_long_cum_4h"]
        _ll_mu = _ll_cum.rolling(168, min_periods=84).mean()
        _ll_sd = _ll_cum.rolling(168, min_periods=84).std().replace(0, np.nan)
        df["long_liq_exhaustion_4h"] = (_ll_cum - _ll_mu) / _ll_sd

        # Cascade count: consecutive bars with above-average liquidation
        above_avg = (df["cg_liq_total"] > liq_ma).astype(int)
        # Count consecutive 1s using cumsum trick
        reset = above_avg.eq(0).cumsum()
        df["cg_liq_cascade"] = above_avg.groupby(reset).cumsum()

        # Liquidation asymmetry slope (trend in long/short imbalance)
        if "cg_liq_imbalance" in df.columns:
            df["cg_liq_imbalance_slope_4h"] = df["cg_liq_imbalance"].rolling(
                SHORT_WIN, min_periods=2
            ).apply(_safe_slope, raw=False)

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

    # ── Liquidity Fragility / Impact Asymmetry ─────────────────────────
    # Core insight: which side of the market is easier to push?
    # Buy impact > sell impact → buyers move price more easily
    # IC = -0.071 (p=0.00002): price tends to reverse AGAINST the fragile side
    price_move = (df["close"] - df["open"]).abs()
    vol_nz = df["volume"].replace(0, np.nan)

    # Price impact per unit volume (liquidity thinness)
    df["price_impact"] = price_move / vol_nz
    df["price_impact_zscore"] = _zscore(df["price_impact"])

    # Directional impact: separate buy-side vs sell-side
    buy_bars = taker_delta > 0
    sell_bars = taker_delta < 0

    buy_impact = pd.Series(np.nan, index=df.index)
    sell_impact = pd.Series(np.nan, index=df.index)
    taker_sell_vol = df["volume"] - df["taker_buy_vol"]

    buy_impact[buy_bars] = (
        (df["close"][buy_bars] - df["open"][buy_bars]) /
        df["taker_buy_vol"][buy_bars].replace(0, np.nan)
    )
    sell_impact[sell_bars] = (
        (df["open"][sell_bars] - df["close"][sell_bars]) /
        taker_sell_vol[sell_bars].replace(0, np.nan)
    )

    # Rolling average (8 bar) then asymmetry
    buy_impact_avg = buy_impact.rolling(8, min_periods=2).mean()
    sell_impact_avg = sell_impact.rolling(8, min_periods=2).mean()

    # Positive = buy side moves price easier, Negative = sell side easier
    df["impact_asymmetry"] = buy_impact_avg - sell_impact_avg
    df["impact_asymmetry_zscore"] = _zscore(df["impact_asymmetry"])

    # Fragility: thin liquidity + low volume
    vol_ma = df["volume"].rolling(LONG_WIN, min_periods=4).mean()
    vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)
    df["fragility"] = df["price_impact"] / vol_ratio.replace(0, np.nan)
    df["fragility_zscore"] = _zscore(df["fragility"])

    # ── Flow Regime & Post-Absorption Breakout ─────────────────────────
    # Core insight from trading experience:
    #   - High flow + price moving = trend (follow it)
    #   - Absorption was happening, now volume spikes = breakout (big move)
    #   - Low price + absorption ending = accumulation complete (expect UP)
    #   - High price + absorption ending = distribution complete (expect DOWN)
    taker_total = df["taker_buy_vol"].abs() + (df["volume"] - df["taker_buy_vol"]).abs()
    taker_total_ma = taker_total.rolling(72, min_periods=10).mean()
    flow_level = taker_total / taker_total_ma.replace(0, np.nan)

    # Flow trend: high flow × big price move = trend confirmation
    ret_4h = df["close"].pct_change(4)
    ret_4h_z = _zscore(ret_4h)
    df["flow_trend_score"] = flow_level * ret_4h_z.abs()

    # Absorption ending detection
    if "absorption_score" in df.columns:
        abs_score = df["absorption_score"].fillna(0)
        abs_ma_8h = abs_score.abs().rolling(8, min_periods=2).mean()
        abs_current = abs_score.abs()

        # Post-absorption breakout: absorption was high, now volume explodes
        vol_ma_24 = df["volume"].rolling(LONG_WIN, min_periods=4).mean()
        vol_spike = df["volume"] / vol_ma_24.replace(0, np.nan)
        df["post_absorb_breakout"] = abs_ma_8h.shift(1) * vol_spike
        df["post_absorb_breakout_z"] = _zscore(df["post_absorb_breakout"])

        # Absorption completion: was high, now dropping
        abs_ending = abs_ma_8h.shift(1) / abs_score.abs().rolling(
            LONG_WIN, min_periods=4).mean().replace(0, np.nan)
        abs_drop = (abs_ma_8h.shift(1) - abs_current > 0).astype(float)
        df["abs_completion"] = abs_ending * abs_drop
        df["abs_completion_z"] = _zscore(df["abs_completion"])

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
        # Legacy columns (backward compat)
        for col in ["large_ratio", "large_buy_ratio", "large_delta_usd",
                     "avg_trade_usd"]:
            df[col] = np.nan
            if col in aggtrades:
                df.iloc[-1, df.columns.get_loc(col)] = aggtrades[col]

        # ── agg_* columns (direction_models expected names) ──
        large_d = aggtrades.get("large_delta_usd", 0)
        small_d = aggtrades.get("small_delta_usd", 0)
        total_usd = aggtrades.get("total_usd", 0)

        agg_snap = {
            "agg_large_delta": large_d,
            "agg_small_delta": small_d,
            "agg_large_ratio": aggtrades.get("large_ratio", np.nan),
            "agg_large_buy_ratio": aggtrades.get("large_buy_ratio", np.nan),
            "agg_large_small_div": large_d - small_d,
            "agg_imbalance_div": (large_d - small_d) / max(abs(large_d) + abs(small_d), 1e-10),
            "agg_large_delta_frac": abs(large_d) / max(total_usd, 1e-10) if total_usd else np.nan,
        }
        for col, val in agg_snap.items():
            df[col] = np.nan
            df.iloc[-1, df.columns.get_loc(col)] = val

        # Rolling features (NaN for historical bars — need snapshot accumulation)
        for col in ["agg_large_delta_zscore", "agg_large_small_div_zscore",
                     "agg_large_delta_ma_4h", "agg_large_delta_ma_8h"]:
            if col not in df.columns:
                df[col] = np.nan

        # Large vs small delta divergence (legacy)
        df["large_small_divergence"] = np.nan
        if abs(large_d) + abs(small_d) > 0:
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

    # ── Deribit DVOL → bvol_* features (snapshot, last bar only) ─────────
    if options_data and isinstance(options_data, dict) and "dvol_value" in options_data:
        bvol_snap = {
            "bvol_close": options_data.get("dvol_value", np.nan),
            "bvol_open": options_data.get("dvol_open", np.nan),
            "bvol_high": options_data.get("dvol_high", np.nan),
            "bvol_low": options_data.get("dvol_low", np.nan),
        }
        for col, val in bvol_snap.items():
            df[col] = np.nan
            df.iloc[-1, df.columns.get_loc(col)] = val

        # Intra-bar range
        dvol_h = bvol_snap["bvol_high"]
        dvol_l = bvol_snap["bvol_low"]
        dvol_c = bvol_snap["bvol_close"]
        df["bvol_intra_range"] = np.nan
        df["bvol_intra_range_pct"] = np.nan
        if not np.isnan(dvol_h) and not np.isnan(dvol_l):
            df.iloc[-1, df.columns.get_loc("bvol_intra_range")] = dvol_h - dvol_l
            if dvol_c and dvol_c > 0:
                df.iloc[-1, df.columns.get_loc("bvol_intra_range_pct")] = (dvol_h - dvol_l) / dvol_c

        # Change from open
        dvol_o = bvol_snap["bvol_open"]
        df["bvol_change_1h"] = np.nan
        if not np.isnan(dvol_c) and not np.isnan(dvol_o) and dvol_o > 0:
            df.iloc[-1, df.columns.get_loc("bvol_change_1h")] = (dvol_c - dvol_o) / dvol_o

        # Rolling features need history — set NaN columns for model compatibility
        for col in ["bvol_std", "bvol_change_4h", "bvol_change_8h", "bvol_change_24h",
                     "bvol_ma_24h", "bvol_ma_72h", "bvol_ratio_ma24",
                     "bvol_zscore", "bvol_slope_4h", "bvol_accel_4h"]:
            if col not in df.columns:
                df[col] = np.nan

        logger.info("DVOL features added: close=%.1f", bvol_snap["bvol_close"])

    # ── Direction-specific features (from research pipeline) ─────────────
    try:
        from research.direction_features import build_direction_feature_set
        dir_feats = build_direction_feature_set(df)
        if not dir_feats.empty:
            # Merge without overwriting existing columns
            new_cols = [c for c in dir_feats.columns if c not in df.columns]
            if new_cols:
                df = pd.concat([df, dir_feats[new_cols]], axis=1)
                logger.info("Direction features added: %d new columns", len(new_cols))
    except ImportError:
        logger.debug("research.direction_features not available — skipping")
    except Exception as e:
        logger.warning("Direction features failed (non-critical): %s", e)

    # ── Order Flow Toxicity features (IC=+0.071, p<0.00001) ────────────
    try:
        from research.features.order_flow_toxicity import OrderFlowToxicity
        tox = OrderFlowToxicity()  # adaptive P75 threshold
        tox_df = tox.transform(df)
        tox_cols = [c for c in tox_df.columns if c.startswith("tox_") and c not in df.columns]
        if tox_cols:
            for c in tox_cols:
                df[c] = tox_df[c]
            logger.info("Toxicity features added: %d columns", len(tox_cols))
    except ImportError:
        logger.debug("order_flow_toxicity not available — skipping")
    except Exception as e:
        logger.warning("Toxicity features failed (non-critical): %s", e)

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
                # Force same datetime precision (pandas 2.x preserves original unit)
                left["dt"] = left["dt"].astype("datetime64[ns, UTC]")
                right["dt"] = right["dt"].astype("datetime64[ns, UTC]")
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

    # 12h CVD persistence z-score (mean-reversion signal)
    # IC validated -0.0343 (p=0.03) vs y_return_4h, train/test STABLE
    # NEGATIVE sign: sustained 12h aggressive buying → 4h reversal down (FOMO top)
    if "cg_fcvd_delta" in df.columns and "cg_scvd_delta" in df.columns:
        _cvd_total = df["cg_fcvd_delta"].fillna(0) + df["cg_scvd_delta"].fillna(0)
        _cvd_12h = _cvd_total.rolling(12, min_periods=6).sum()
        _cvd_mu = _cvd_12h.rolling(168, min_periods=84).mean()
        _cvd_sd = _cvd_12h.rolling(168, min_periods=84).std().replace(0, np.nan)
        df["cvd_persistence_12h"] = (_cvd_12h - _cvd_mu) / _cvd_sd
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
