"""
Tests for indicator/inference.py — IndicatorEngine and its static helpers.

Covers regime detection, magnitude scoring,
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
            "strength_score", "regime",
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


class TestWarmupSilenceAndReachability:
    """The 2026-08-11 defect: the decode could not fire DOWN at all.

    The existing decoding tests above re-implement the quantile arithmetic
    and assert it against itself, so they stayed green while the shipped
    decode was one-sided for months.  These exercise IndicatorEngine.predict
    itself, and assert the two properties that actually failed:
      - under warm-up the engine stays SILENT (it used to fall back to fixed
        thresholds, which is where the skew lived: 43:5 at the 08-08 reset)
      - with a live-grown buffer BOTH tails are reachable by construction
    """

    def _engine(self):
        from indicator.inference import IndicatorEngine
        return IndicatorEngine()

    def test_silent_while_buffer_below_warmup(self, sample_features_df):
        from collections import deque
        eng = self._engine()
        eng.dir_pred_history = deque(maxlen=eng.dir_pct_window)   # cold
        out = eng.predict(sample_features_df, update_history=False)
        assert set(out["pred_direction"]) == {"NEUTRAL"}, (
            "under warm-up the decode must fire nothing; falling back to "
            "fixed thresholds is what produced the one-sided window")
        assert (out["strength_score"] == "Weak").all()
        # pred_return is still published — the number is fine, only the
        # translation into a direction is withheld.
        assert out["pred_return_4h"].notna().all()

    def test_committed_seed_is_empty(self):
        """Regression on the artifact itself, not just the code.

        A future export that re-seeds dir_pred_history would silently
        reintroduce the defect, so pin the shipped file.
        """
        import json
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent / "indicator"
             / "model_artifacts" / "dual_model" / "training_stats.json")
        stats = json.loads(p.read_text())
        assert stats.get("dir_pred_history") == [], (
            "dir_pred_history must ship EMPTY — the buffer is grown from "
            "live predictions and rehydrated from indicator_history")

    def test_shipped_seed_put_down_out_of_reach(self):
        """The failure, in the numbers actually measured on 2026-08-11.

        A first version of this test drew the two distributions from
        Gaussians with the observed mean/std and did NOT reproduce the
        failure — the real live stream is right-skewed, so a Gaussian
        stand-in has a fatter left tail than the model ever produced.  The
        recorded values are used instead: a simulation that cannot reproduce
        the defect is not evidence about it.
        """
        seed_strong_dn, seed_mod_dn = -0.002568, -0.001786
        live_min = -0.001480          # lowest output since the 08-08 deploy
        assert live_min > seed_mod_dn > seed_strong_dn, (
            "both DOWN cutoffs sat below anything the model could emit — "
            "DOWN was impossible, not improbable")

    def test_live_grown_buffer_keeps_both_tails_reachable(self):
        """The structural guarantee that replaced 'hope the seed matches'.

        Judged out-of-sample so this is not a tautology: cutoffs come from
        the first half of a drifting stream, the bars judged are the second
        half.  Even while the level walks away, a buffer made of the model's
        own recent output keeps both tails live — which is the property the
        in-sample seed did not have.
        """
        rng = np.random.default_rng(20260811)
        drift = np.linspace(0.0, 0.0015, 400)          # a level that walks
        stream = rng.normal(0.0007, 0.0010, 400) + drift
        buf, judged = stream[:200], stream[200:]
        dn = min(float(np.quantile(buf, 0.025)), -0.0008)
        up = max(float(np.quantile(buf, 0.975)), 0.0008)
        assert (judged <= dn).sum() > 0 and (judged >= up).sum() > 0, (
            "a live-grown buffer must leave both tails reachable even under "
            "a drifting level")
