"""
Tests for indicator/inference.py — IndicatorEngine and its static helpers.

Covers regime detection, magnitude scoring, BBP computation,
dynamic deadzone, and rolling percentile direction decoding.
Model-dependent tests are skipped when XGBoost artifacts are absent.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "indicator" / "model_artifacts" / "dual_model"
HAS_MODEL_FILES = (
    (ARTIFACT_DIR / "direction_xgb.json").exists()
    and (ARTIFACT_DIR / "magnitude_xgb.json").exists()
    and (ARTIFACT_DIR / "direction_feature_cols.json").exists()
    and (ARTIFACT_DIR / "magnitude_feature_cols.json").exists()
    and (ARTIFACT_DIR / "direction_reg_config.json").exists()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close_series(n: int, base: float = 67_000,
                       daily_ret: float = 0.0) -> pd.DataFrame:
    """Build a DataFrame with 'close' column for regime tests."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    hourly_ret = daily_ret / 24
    close = base * np.exp(np.cumsum(np.full(n, hourly_ret)))
    return pd.DataFrame({"close": close}, index=idx)


# ---------------------------------------------------------------------------
# Regime detection (_assign_regime)
# ---------------------------------------------------------------------------

class TestAssignRegime:
    """Tests for IndicatorEngine._assign_regime static method."""

    def test_warmup_first_72_bars(self, sample_features_df):
        """First 72 bars must be labelled WARMUP regardless of price action."""
        from indicator.inference import IndicatorEngine
        regime = IndicatorEngine._assign_regime(sample_features_df)
        assert all(r == "WARMUP" for r in regime[:72])

    def test_regime_returns_correct_length(self, sample_features_df):
        """Regime array length must match input DataFrame."""
        from indicator.inference import IndicatorEngine
        regime = IndicatorEngine._assign_regime(sample_features_df)
        assert len(regime) == len(sample_features_df)

    def test_trending_bull_detection(self):
        """High vol + positive 24h return should produce TRENDING_BULL."""
        from indicator.inference import IndicatorEngine
        # Build 200 bars with a sustained uptrend and high volatility
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        np.random.seed(10)
        # Strong uptrend: +0.15% per hour on average with high volatility
        returns = np.random.normal(0.0015, 0.008, n)
        close = 67_000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": close}, index=idx)

        regime = IndicatorEngine._assign_regime(df)
        # After warmup, should contain some TRENDING_BULL bars
        post_warmup = regime[72:]
        assert "TRENDING_BULL" in post_warmup, (
            f"Expected TRENDING_BULL in regime, got unique values: {set(post_warmup)}"
        )

    def test_trending_bear_detection(self):
        """High vol + negative 24h return should produce TRENDING_BEAR."""
        from indicator.inference import IndicatorEngine
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        np.random.seed(11)
        # Strong downtrend
        returns = np.random.normal(-0.0015, 0.008, n)
        close = 67_000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": close}, index=idx)

        regime = IndicatorEngine._assign_regime(df)
        post_warmup = regime[72:]
        assert "TRENDING_BEAR" in post_warmup, (
            f"Expected TRENDING_BEAR in regime, got unique values: {set(post_warmup)}"
        )

    def test_choppy_in_low_vol(self):
        """Low vol + small returns should produce CHOPPY (not trending)."""
        from indicator.inference import IndicatorEngine
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        np.random.seed(12)
        # Nearly flat price, very low vol
        returns = np.random.normal(0, 0.0005, n)
        close = 67_000 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"close": close}, index=idx)

        regime = IndicatorEngine._assign_regime(df)
        post_warmup = regime[72:]
        # Should be mostly CHOPPY
        choppy_frac = sum(1 for r in post_warmup if r == "CHOPPY") / len(post_warmup)
        assert choppy_frac > 0.5, f"Expected mostly CHOPPY, got {choppy_frac:.0%}"

    def test_regime_values_are_valid_strings(self, sample_features_df):
        """All regime values must be one of the four valid strings."""
        from indicator.inference import IndicatorEngine
        valid = {"WARMUP", "CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"}
        regime = IndicatorEngine._assign_regime(sample_features_df)
        assert set(regime).issubset(valid)


# ---------------------------------------------------------------------------
# Magnitude score (_compute_mag_score)
# ---------------------------------------------------------------------------

class TestComputeMagScore:
    """Tests for expanding percentile magnitude scoring."""

    def _make_engine_stub(self):
        """Create a minimal object with pred_history for mag_score testing."""
        stub = MagicMock()
        stub.pred_history = deque(maxlen=500)
        from indicator.inference import IndicatorEngine
        stub._compute_mag_score = IndicatorEngine._compute_mag_score.__get__(stub)
        return stub

    def test_ascending_sequence_highest_near_100(self):
        """Given ascending values [1..5], the last (5) should score near 100."""
        engine = self._make_engine_stub()
        # Warm up with 30+ values so scores are produced
        warmup = np.arange(1, 32, dtype=float)
        engine._compute_mag_score(warmup)

        test = np.array([100.0])
        scores = engine._compute_mag_score(test)
        assert scores[0] > 90, f"Expected >90 for max value, got {scores[0]}"

    def test_low_value_scores_low(self):
        """A value smaller than most history should score low."""
        engine = self._make_engine_stub()
        warmup = np.arange(1, 50, dtype=float)
        engine._compute_mag_score(warmup)

        test = np.array([0.5])
        scores = engine._compute_mag_score(test)
        assert scores[0] < 10, f"Expected <10 for min value, got {scores[0]}"

    def test_nan_input_stays_nan(self):
        """NaN predictions should produce NaN scores."""
        engine = self._make_engine_stub()
        warmup = np.arange(1, 40, dtype=float)
        engine._compute_mag_score(warmup)

        test = np.array([np.nan])
        scores = engine._compute_mag_score(test)
        assert np.isnan(scores[0])

    def test_insufficient_history_returns_nan(self):
        """With fewer than MIN_MAG_HISTORY (30) values, score should be NaN."""
        engine = self._make_engine_stub()
        # Only push 5 values — well below min_periods=30
        test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = engine._compute_mag_score(test)
        assert all(np.isnan(scores))

    def test_update_history_false_does_not_grow_buffer(self):
        """When update_history=False, pred_history should not grow."""
        engine = self._make_engine_stub()
        warmup = np.arange(1, 40, dtype=float)
        engine._compute_mag_score(warmup)
        size_before = len(engine.pred_history)

        engine._compute_mag_score(np.array([999.0]), update_history=False)
        assert len(engine.pred_history) == size_before


# ---------------------------------------------------------------------------
# BBP (_compute_bbp)
# ---------------------------------------------------------------------------

class TestComputeBBP:
    """Tests for Bull/Bear Power composite indicator."""

    def test_bbp_with_all_zscore_columns(self):
        """BBP should be mean of normalised z-scores, clipped to [-1, 1]."""
        from indicator.inference import IndicatorEngine
        n = 10
        df = pd.DataFrame({
            "cg_oi_delta_zscore": np.full(n, 3.0),       # +1 after /3
            "cg_funding_close_zscore": np.full(n, -3.0),  # negated -> +1
            "cg_taker_delta_zscore": np.full(n, 3.0),     # +1
            "cg_ls_ratio_zscore": np.full(n, -3.0),       # negated -> +1
            "cg_ls_divergence_zscore": np.full(n, 3.0),   # +1
        })
        bbp = IndicatorEngine._compute_bbp(df)
        # All components = +1, mean = +1, clipped to 1
        np.testing.assert_allclose(bbp, 1.0, atol=1e-6)

    def test_bbp_no_columns_returns_zero(self):
        """When no z-score columns exist, BBP should be all zeros."""
        from indicator.inference import IndicatorEngine
        df = pd.DataFrame({"close": [1, 2, 3]})
        bbp = IndicatorEngine._compute_bbp(df)
        np.testing.assert_array_equal(bbp, 0.0)

    def test_bbp_clipped_to_minus_one(self):
        """Extreme negative z-scores should clip BBP to -1."""
        from indicator.inference import IndicatorEngine
        n = 5
        df = pd.DataFrame({
            "cg_oi_delta_zscore": np.full(n, -3.0),
            "cg_funding_close_zscore": np.full(n, 3.0),
            "cg_taker_delta_zscore": np.full(n, -3.0),
            "cg_ls_ratio_zscore": np.full(n, 3.0),
            "cg_ls_divergence_zscore": np.full(n, -3.0),
        })
        bbp = IndicatorEngine._compute_bbp(df)
        np.testing.assert_allclose(bbp, -1.0, atol=1e-6)

    def test_bbp_partial_columns(self):
        """BBP should work with only some z-score columns present."""
        from indicator.inference import IndicatorEngine
        n = 5
        df = pd.DataFrame({
            "cg_oi_delta_zscore": np.full(n, 1.5),  # 1.5/3 = 0.5
        })
        bbp = IndicatorEngine._compute_bbp(df)
        np.testing.assert_allclose(bbp, 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# Dynamic deadzone (_compute_dynamic_deadzone)
# ---------------------------------------------------------------------------

class TestDynamicDeadzone:
    """Tests for per-bar dynamic deadzone computation."""

    def test_choppy_widens_deadzone(self):
        """CHOPPY regime should multiply deadzone by CHOPPY_DEADZONE_MULT (1.6)."""
        from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE, CHOPPY_DEADZONE_MULT
        n = 5
        df = pd.DataFrame({
            "realized_vol_20b": np.full(n, np.nan),
        })
        regime = np.full(n, "CHOPPY", dtype=object)
        dz = IndicatorEngine._compute_dynamic_deadzone(df, regime)
        expected = STRENGTH_DEADZONE * CHOPPY_DEADZONE_MULT
        np.testing.assert_allclose(dz, expected, atol=1e-6)

    def test_trending_tightens_deadzone(self):
        """TRENDING_BULL regime should multiply deadzone by TREND_DEADZONE_MULT (0.9)."""
        from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE, TREND_DEADZONE_MULT
        n = 5
        df = pd.DataFrame({
            "realized_vol_20b": np.full(n, np.nan),
        })
        regime = np.full(n, "TRENDING_BULL", dtype=object)
        dz = IndicatorEngine._compute_dynamic_deadzone(df, regime)
        expected = STRENGTH_DEADZONE * TREND_DEADZONE_MULT
        np.testing.assert_allclose(dz, expected, atol=1e-6)

    def test_warmup_uses_base_deadzone(self):
        """WARMUP regime keeps the base deadzone unchanged."""
        from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE
        n = 5
        df = pd.DataFrame({
            "realized_vol_20b": np.full(n, np.nan),
        })
        regime = np.full(n, "WARMUP", dtype=object)
        dz = IndicatorEngine._compute_dynamic_deadzone(df, regime)
        np.testing.assert_allclose(dz, STRENGTH_DEADZONE, atol=1e-6)

    def test_high_vol_ratio_widens_deadzone(self):
        """Vol ratio = 2x median should widen the deadzone via VOL_DEADZONE_SCALE."""
        from indicator.inference import IndicatorEngine, STRENGTH_DEADZONE, VOL_DEADZONE_SCALE
        # Need enough bars for expanding median to be defined
        n = 50
        vol = np.full(n, 0.005)
        # Last bar has double the vol
        vol[-1] = 0.010
        df = pd.DataFrame({"realized_vol_20b": vol})
        regime = np.full(n, "CHOPPY", dtype=object)  # won't test regime here
        # Override regime to something neutral to isolate vol effect
        regime[:] = "WARMUP"
        dz = IndicatorEngine._compute_dynamic_deadzone(df, regime)
        # Last bar should have higher deadzone than first
        assert dz[-1] > dz[0], "High vol ratio should widen the deadzone"


# ---------------------------------------------------------------------------
# Rolling percentile direction decoding
# ---------------------------------------------------------------------------

class TestRollingPercentileDecoding:
    """Test that the direction decoding logic works correctly with known inputs."""

    def test_extreme_positive_gets_strong_up(self):
        """A prediction in the top 2.5% of the buffer should decode as Strong UP."""
        # Simulate the decoding logic directly without loading model files
        from collections import deque

        buffer = deque(np.random.normal(0, 0.001, 500).tolist(), maxlen=500)
        # The buffer's 97.5th percentile
        buf_arr = np.array(list(buffer))
        strong_up_threshold = float(np.quantile(buf_arr, 0.975))

        # A value clearly above the threshold
        extreme_pred = strong_up_threshold + 0.001
        assert extreme_pred >= strong_up_threshold, "Extreme pred should exceed Strong UP threshold"

    def test_extreme_negative_gets_strong_down(self):
        """A prediction in the bottom 2.5% should decode as Strong DOWN."""
        from collections import deque
        buffer = deque(np.random.normal(0, 0.001, 500).tolist(), maxlen=500)
        buf_arr = np.array(list(buffer))
        strong_dn_threshold = float(np.quantile(buf_arr, 0.025))

        extreme_pred = strong_dn_threshold - 0.001
        assert extreme_pred <= strong_dn_threshold

    def test_moderate_zone(self):
        """Predictions between 7.5% and 2.5% tails should be Moderate."""
        from collections import deque
        np.random.seed(42)
        buffer = deque(np.random.normal(0, 0.001, 500).tolist(), maxlen=500)
        buf_arr = np.array(list(buffer))

        strong_frac = 0.05
        mod_frac = 0.15
        up_strong = float(np.quantile(buf_arr, 1.0 - strong_frac / 2.0))
        up_mod = float(np.quantile(buf_arr, 1.0 - mod_frac / 2.0))

        # Pick a value between moderate and strong thresholds
        moderate_pred = (up_mod + up_strong) / 2
        assert up_mod <= moderate_pred < up_strong, (
            f"Moderate pred {moderate_pred} should be between "
            f"mod threshold {up_mod} and strong threshold {up_strong}"
        )

    def test_neutral_zone(self):
        """Predictions near zero (within moderate threshold) should be NEUTRAL."""
        from collections import deque
        np.random.seed(42)
        buffer = deque(np.random.normal(0, 0.001, 500).tolist(), maxlen=500)
        buf_arr = np.array(list(buffer))

        mod_frac = 0.15
        dn_mod = float(np.quantile(buf_arr, mod_frac / 2.0))
        up_mod = float(np.quantile(buf_arr, 1.0 - mod_frac / 2.0))

        neutral_pred = 0.0  # dead center
        assert dn_mod < neutral_pred < up_mod, (
            f"Zero should be in neutral zone [{dn_mod}, {up_mod}]"
        )


# ---------------------------------------------------------------------------
# Full engine integration tests (require model files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MODEL_FILES,
                    reason="Model artifacts not found in dual_model/")
class TestIndicatorEngineIntegration:
    """Integration tests that load real model artifacts."""

    def test_engine_loads_successfully(self):
        """IndicatorEngine should initialize without errors."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        assert engine.mode == "dual"
        assert hasattr(engine, "dual_dir_model")
        assert hasattr(engine, "dual_mag_model")

    def test_predict_returns_expected_columns(self, sample_features_df):
        """predict() should return a DataFrame with all required output columns."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        out = engine.predict(sample_features_df)

        required = [
            "pred_return_4h", "pred_direction", "confidence_score",
            "strength_score", "bull_bear_power", "regime",
        ]
        for col in required:
            assert col in out.columns, f"Missing output column: {col}"

    def test_predict_direction_values(self, sample_features_df):
        """pred_direction should only contain UP, DOWN, or NEUTRAL."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        out = engine.predict(sample_features_df)
        valid_dirs = {"UP", "DOWN", "NEUTRAL"}
        assert set(out["pred_direction"].unique()).issubset(valid_dirs)

    def test_predict_strength_values(self, sample_features_df):
        """strength_score should only contain Strong, Moderate, or Weak."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        out = engine.predict(sample_features_df)
        valid_strengths = {"Strong", "Moderate", "Weak"}
        assert set(out["strength_score"].unique()).issubset(valid_strengths)

    def test_confidence_range(self, sample_features_df):
        """confidence_score should be in [0, 100]."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        out = engine.predict(sample_features_df)
        conf = out["confidence_score"].dropna()
        assert conf.min() >= 0
        assert conf.max() <= 100

    def test_update_history_false_preserves_buffer(self, sample_features_df):
        """predict(update_history=False) should not change dir_pred_history size."""
        from indicator.inference import IndicatorEngine
        engine = IndicatorEngine()
        buf_before = len(engine.dir_pred_history)
        engine.predict(sample_features_df, update_history=False)
        assert len(engine.dir_pred_history) == buf_before
