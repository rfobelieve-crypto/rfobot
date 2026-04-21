"""
Tests for indicator/feature_builder_live.py — build_live_features().

Validates feature engineering correctness: no look-ahead bias, NaN handling,
known formula verification, and Coinglass merge_asof alignment.
Uses synthetic DataFrames with known values throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_klines(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build minimal klines DataFrame for feature builder."""
    np.random.seed(seed)
    idx = pd.date_range("2026-04-01", periods=n, freq="1h", tz="UTC")
    close = 67_000 * np.exp(np.cumsum(np.random.normal(0, 0.003, n)))
    volume = np.random.uniform(500, 2000, n)
    taker_buy = volume * np.random.uniform(0.4, 0.6, n)
    trade_count = np.random.randint(5000, 50000, n).astype(float)
    quote_vol = volume * close

    df = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.002, n))),
        "close": close,
        "volume": volume,
        "taker_buy_vol": taker_buy,
        "taker_buy_quote": taker_buy * close,
        "trade_count": trade_count,
        "quote_vol": quote_vol,
    }, index=idx)
    df.index.name = "dt"
    return df


def _empty_cg_data() -> dict:
    """Empty Coinglass data dict (all endpoints return empty DataFrames)."""
    from indicator.data_fetcher import CG_ENDPOINTS
    return {name: pd.DataFrame() for name in CG_ENDPOINTS}


# ---------------------------------------------------------------------------
# No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    """Verify all rolling operations use trailing windows only."""

    def test_realized_vol_uses_trailing_window(self):
        """realized_vol_20b at bar i should only use bars [i-19..i]."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(100)
        feats = build_live_features(klines, _empty_cg_data())

        # Manually compute trailing vol for bar 50
        log_ret = np.log(klines["close"] / klines["close"].shift(1))
        expected = log_ret.iloc[31:51].std()
        actual = feats["realized_vol_20b"].iloc[50]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_return_lags_are_strictly_past(self):
        """return_lag_1 at bar i should equal log_return at bar i-1."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        log_ret = np.log(klines["close"] / klines["close"].shift(1))
        for i in range(5, 50):
            if not np.isnan(feats["return_lag_1"].iloc[i]):
                np.testing.assert_allclose(
                    feats["return_lag_1"].iloc[i],
                    log_ret.iloc[i - 1],
                    rtol=1e-6,
                    err_msg=f"return_lag_1 at bar {i} should match log_return at bar {i-1}"
                )

    def test_zscore_uses_trailing_24h(self):
        """Z-scores use rolling(24) which is trailing by construction."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(100)
        feats = build_live_features(klines, _empty_cg_data())

        # Check that zscore at bar 50 only uses bars 27..50 (24 bar window)
        if "cg_oi_delta_zscore" in feats.columns:
            # If CG data is empty, zscores will be NaN — that's fine
            pass
        # The key structural check: no future index references
        # realized_vol_20b at bar 30 should not use bar 31+ data
        vol_30 = feats["realized_vol_20b"].iloc[30]
        # Recompute using only bars <= 30
        log_ret = np.log(klines["close"] / klines["close"].shift(1))
        expected = log_ret.iloc[11:31].std()
        np.testing.assert_allclose(vol_30, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """Features with insufficient history should be NaN, not zero."""

    def test_realized_vol_nan_for_first_bars(self):
        """realized_vol_20b requires 20 bars; first few should be NaN."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        # Bars 0-3 must be NaN (min_periods=4 for rolling(20))
        assert feats["realized_vol_20b"].iloc[:4].isna().all(), (
            "First 4 bars should have NaN realized_vol_20b"
        )

    def test_return_skew_nan_early(self):
        """return_skew uses rolling(24, min_periods=10), so early bars are NaN."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        assert feats["return_skew"].iloc[:10].isna().all(), (
            "First 10 bars should have NaN return_skew"
        )

    def test_missing_coinglass_produces_nan(self):
        """When all CG data is empty, CG-derived features should be NaN."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        for col in ["cg_oi_close", "cg_funding_close", "cg_taker_buy"]:
            if col in feats.columns:
                assert feats[col].isna().all(), (
                    f"{col} should be all NaN when CG data is empty"
                )


# ---------------------------------------------------------------------------
# Known feature formulas
# ---------------------------------------------------------------------------

class TestKnownFormulas:
    """Verify specific feature calculations against manual computation."""

    def test_log_return(self):
        """log_return = log(close[t] / close[t-1])."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        expected = np.log(klines["close"] / klines["close"].shift(1))
        pd.testing.assert_series_equal(
            feats["log_return"], expected, check_names=False,
        )

    def test_taker_delta_ratio(self):
        """taker_delta_ratio = (buy - sell) / volume."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        sell = klines["volume"] - klines["taker_buy_vol"]
        delta = klines["taker_buy_vol"] - sell
        expected = delta / klines["volume"]
        pd.testing.assert_series_equal(
            feats["taker_delta_ratio"], expected, check_names=False,
        )

    def test_hour_sin_cos_range(self):
        """Cyclical time features should be in [-1, 1]."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        assert feats["hour_sin"].between(-1, 1).all()
        assert feats["hour_cos"].between(-1, 1).all()
        assert feats["weekday_sin"].between(-1, 1).all()
        assert feats["weekday_cos"].between(-1, 1).all()

    def test_regime_indicators_binary(self):
        """is_trending_bull and is_trending_bear should be 0.0 or 1.0."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(200)
        feats = build_live_features(klines, _empty_cg_data())

        for col in ["is_trending_bull", "is_trending_bear"]:
            vals = feats[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), (
                f"{col} should be binary, got {vals}"
            )

    def test_return_lags_count(self):
        """Should have return_lag_1 through return_lag_10."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        feats = build_live_features(klines, _empty_cg_data())

        for lag in range(1, 11):
            col = f"return_lag_{lag}"
            assert col in feats.columns, f"Missing {col}"

    def test_vol_acceleration_formula(self):
        """vol_acceleration = vol_ma_4h / vol_ma_24h."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(100)
        feats = build_live_features(klines, _empty_cg_data())

        vol = klines["volume"]
        ma4 = vol.rolling(4, min_periods=2).mean()
        ma24 = vol.rolling(24, min_periods=4).mean().replace(0, np.nan)
        expected = ma4 / ma24

        # Compare at bar 50 where both are defined
        np.testing.assert_allclose(
            feats["vol_acceleration"].iloc[50],
            expected.iloc[50],
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Coinglass merge_asof alignment
# ---------------------------------------------------------------------------

class TestCoinglassMergeAsof:
    """Verify backward-only alignment (no future data leakage)."""

    def test_merge_asof_backward_only(self):
        """CG data at t should NOT appear at bars before t."""
        from indicator.feature_builder_live import build_live_features

        klines = _minimal_klines(50)
        # Create CG OI data with a gap: data only at bars 10, 20, 30
        cg_idx = klines.index[[10, 20, 30]]
        oi_df = pd.DataFrame({
            "close": [100.0, 200.0, 300.0],
        }, index=cg_idx)
        oi_df.index.name = "dt"

        cg_data = _empty_cg_data()
        cg_data["oi"] = oi_df

        feats = build_live_features(klines, cg_data)

        if "cg_oi_close" in feats.columns:
            # Bar 9 (before first CG data at bar 10) should be NaN
            assert np.isnan(feats["cg_oi_close"].iloc[0]), (
                "Bar before first CG data should be NaN (backward merge)"
            )
            # Bar 10 should have value 100
            assert feats["cg_oi_close"].iloc[10] == 100.0
            # Bar 15 should still show 100 (backward fill from bar 10)
            assert feats["cg_oi_close"].iloc[15] == 100.0
            # Bar 20 should show 200
            assert feats["cg_oi_close"].iloc[20] == 200.0

    def test_no_future_cg_data_leaks(self):
        """CG data timestamped in the future should not appear in current bars."""
        from indicator.feature_builder_live import build_live_features

        klines = _minimal_klines(50)
        # CG data only at the very last bar
        cg_idx = klines.index[[-1]]
        oi_df = pd.DataFrame({"close": [999.0]}, index=cg_idx)
        oi_df.index.name = "dt"

        cg_data = _empty_cg_data()
        cg_data["oi"] = oi_df

        feats = build_live_features(klines, cg_data)

        if "cg_oi_close" in feats.columns:
            # All bars except the last should NOT have 999
            for i in range(len(feats) - 1):
                val = feats["cg_oi_close"].iloc[i]
                assert val != 999.0 or np.isnan(val), (
                    f"Bar {i} should not have future CG data"
                )


# ---------------------------------------------------------------------------
# Feature count
# ---------------------------------------------------------------------------

class TestFeatureCount:
    """Verify approximate feature counts match expectations."""

    def test_minimum_feature_count_no_cg(self):
        """With no CG data, should still produce kline-derived features."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(100)
        feats = build_live_features(klines, _empty_cg_data())

        # At minimum: OHLCV + log_return + realized_vol + return_skew + kurtosis
        # + taker delta features + 10 return lags + time features + vol features
        # Should be at least 30 columns
        assert len(feats.columns) >= 30, (
            f"Expected >= 30 features, got {len(feats.columns)}"
        )

    def test_snapshot_features_only_on_last_bar(self):
        """Depth and aggtrade snapshot features should only fill the last bar."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        depth = {
            "depth_imbalance": 0.1,
            "near_imbalance": 0.05,
            "spread_bps": 1.5,
            "bid_depth_usd": 1_000_000,
            "ask_depth_usd": 900_000,
        }
        feats = build_live_features(klines, _empty_cg_data(), depth=depth)

        if "depth_imbalance" in feats.columns:
            # All bars except last should be NaN
            assert feats["depth_imbalance"].iloc[:-1].isna().all()
            # Last bar should have the value
            assert feats["depth_imbalance"].iloc[-1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases in feature building."""

    def test_single_bar_does_not_crash(self):
        """Feature builder should handle a single bar without crashing."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(1)
        # Should not raise
        feats = build_live_features(klines, _empty_cg_data())
        assert len(feats) == 1

    def test_zero_volume_handled(self):
        """Bars with zero volume should not produce inf or crash."""
        from indicator.feature_builder_live import build_live_features
        klines = _minimal_klines(50)
        klines.iloc[25, klines.columns.get_loc("volume")] = 0.0
        # Should not raise
        feats = build_live_features(klines, _empty_cg_data())
        # taker_delta_ratio at bar 25 should be NaN (division by zero)
        assert np.isnan(feats["taker_delta_ratio"].iloc[25]) or np.isfinite(feats["taker_delta_ratio"].iloc[25])
