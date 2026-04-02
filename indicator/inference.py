"""
Model inference — regime-conditional multi-target prediction.

v3: Uses up_move + down_move dual-model architecture.
    Direction derived from (up_pred - down_pred) asymmetry.
    Regime routing selects global or regime-specific model.
    Falls back to legacy single model if regime_models/ not found.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
REGIME_DIR = ARTIFACT_DIR / "regime_models"

# ── Parameters ────────────────────────────────────────────────────────────
STRENGTH_DEADZONE = 0.15     # |up_pred - down_pred| below this → NEUTRAL
STRONG_THRESHOLD = 80.0      # confidence percentile for Strong
MODERATE_THRESHOLD = 65.0    # confidence percentile for Moderate
HORIZON_BARS = 4             # 4h prediction horizon (4 x 1h bars)
MIN_MAG_HISTORY = 30         # minimum bars before mag_score is valid
MIN_REGIME_IC_HISTORY = 50   # minimum bars per regime before IC is meaningful
REGIME_BLEND_WEIGHT = 0.35   # weight for regime-specific model (vs global)

# Regime models to use (skip strength_vol_adj — no signal)
ACTIVE_TARGETS = ["up_move_vol_adj", "down_move_vol_adj"]


class IndicatorEngine:
    """Stateful prediction engine with regime-conditional dual-target models."""

    def __init__(self):
        if REGIME_DIR.exists():
            self._load_regime_models()
        else:
            self._load_legacy_model()

        # Regime IC tracking (persists across calls)
        self.regime_history: dict[str, list[tuple[float, float]]] = {}

    def _load_regime_models(self):
        """Load multi-target regime-conditional models with per-target features."""
        self.mode = "regime"

        # Superset feature_cols (for missing-feature warnings)
        with open(REGIME_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        with open(REGIME_DIR / "training_stats.json") as f:
            stats = json.load(f)
        self.pred_history = deque(stats.get("pred_history", []), maxlen=500)

        # Per-target feature cols (v4: target-specific feature selection)
        self.target_feature_cols: dict[str, list[str]] = {}
        for target in ACTIVE_TARGETS:
            target_fc_path = REGIME_DIR / target / "feature_cols.json"
            if target_fc_path.exists():
                with open(target_fc_path) as f:
                    self.target_feature_cols[target] = json.load(f)
            else:
                # Fallback: use shared feature_cols (backward compatible)
                self.target_feature_cols[target] = self.feature_cols

        # Load models: {target: {regime: XGBRegressor}}
        self.models: dict[str, dict[str, xgb.XGBRegressor]] = {}
        for target in ACTIVE_TARGETS:
            tdir = REGIME_DIR / target
            self.models[target] = {}
            for fname in tdir.glob("*_xgb.json"):
                if fname.name == "feature_cols.json":
                    continue
                regime_key = fname.stem.replace("_xgb", "").upper()
                if regime_key == "GLOBAL":
                    regime_key = "global"
                m = xgb.XGBRegressor()
                m.load_model(str(fname))
                self.models[target][regime_key] = m

        model_count = sum(len(v) for v in self.models.values())
        for t, fc in self.target_feature_cols.items():
            logger.info("  %s: %d features", t, len(fc))
        logger.info("IndicatorEngine v4 loaded: %d models, %d warmup bars",
                     model_count, len(self.pred_history))

    def _load_legacy_model(self):
        """Fallback: load single model (v2 compatibility)."""
        self.mode = "legacy"
        self.legacy_model = xgb.XGBRegressor()
        self.legacy_model.load_model(str(ARTIFACT_DIR / "xgb_model.json"))

        with open(ARTIFACT_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        with open(ARTIFACT_DIR / "training_stats.json") as f:
            stats = json.load(f)
        self.pred_history = deque(stats.get("pred_history", []), maxlen=500)

        logger.info("IndicatorEngine legacy loaded: %d features", len(self.feature_cols))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Run prediction on feature DataFrame.

        Returns DataFrame with: pred_return_4h, pred_direction,
        confidence_score, strength_score, bull_bear_power, regime
        """
        # Align features (check superset — individual targets may need subsets)
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            logger.warning("Missing features (zero-filled): %s", missing)
            for c in missing:
                features[c] = np.nan

        # Regime detection (trailing-only)
        regime = self._assign_regime(features)

        if self.mode == "regime":
            return self._predict_regime(features, None, regime)
        else:
            X = features[self.feature_cols].fillna(0).values
            return self._predict_legacy(features, X, regime)

    # ── Regime-conditional prediction ─────────────────────────────────────

    def _predict_regime(self, features: pd.DataFrame,
                        X: np.ndarray, regime: np.ndarray) -> pd.DataFrame:
        n = len(features)

        # Build per-target X matrices (may differ in feature count)
        target_X: dict[str, np.ndarray] = {}
        for target in ACTIVE_TARGETS:
            tc = self.target_feature_cols.get(target, self.feature_cols)
            target_X[target] = features[tc].fillna(0).values

        # Predict up_move and down_move per bar
        up_pred = np.zeros(n)
        down_pred = np.zeros(n)

        for i in range(n):
            r = regime[i]

            for target, arr in [("up_move_vol_adj", up_pred),
                                ("down_move_vol_adj", down_pred)]:
                x_i = target_X[target][i:i+1]
                models = self.models[target]
                global_model = models.get("global")
                regime_model = models.get(r)

                if global_model is None:
                    continue

                global_pred = float(global_model.predict(x_i)[0])

                if regime_model is not None and r != "WARMUP":
                    regime_pred = float(regime_model.predict(x_i)[0])
                    arr[i] = ((1 - REGIME_BLEND_WEIGHT) * global_pred +
                              REGIME_BLEND_WEIGHT * regime_pred)
                else:
                    arr[i] = global_pred

        # Ensure non-negative (up/down moves are always >= 0)
        up_pred = np.maximum(up_pred, 0)
        down_pred = np.maximum(down_pred, 0)

        # Strength = up - down (positive = bullish asymmetry)
        strength = up_pred - down_pred

        # Direction from strength asymmetry
        direction = np.where(strength > STRENGTH_DEADZONE, "UP",
                    np.where(strength < -STRENGTH_DEADZONE, "DOWN", "NEUTRAL"))

        # Synthetic pred_return_4h (for chart compatibility)
        # Scale strength to approximate return scale
        pred_return = strength * 0.003  # rough scaling factor

        # Mag score (expanding percentile of |strength|)
        mag_score = self._compute_mag_score(strength)

        # Confidence = mag_score only (regime_score removed —
        # trailing IC is unreliable: inflated during cold-start from
        # in-sample overlap, then collapses to ~0 on true OOS data)
        confidence = np.clip(mag_score, 0, 100)

        # Strength tiers
        strength_tier = np.full(n, "Weak", dtype=object)
        strength_tier[confidence >= MODERATE_THRESHOLD] = "Moderate"
        strength_tier[confidence >= STRONG_THRESHOLD] = "Strong"
        strength_tier[np.isnan(confidence)] = "Weak"

        # Bull/Bear Power
        bbp = self._compute_bbp(features)

        out = pd.DataFrame(index=features.index)
        out["pred_return_4h"] = pred_return
        out["pred_direction"] = direction
        out["confidence_score"] = confidence
        out["strength_score"] = strength_tier
        out["bull_bear_power"] = bbp
        out["regime"] = regime
        out["up_pred"] = up_pred
        out["down_pred"] = down_pred
        out["strength_raw"] = strength

        for c in ["open", "high", "low", "close"]:
            if c in features.columns:
                out[c] = features[c].values

        return out

    # ── Legacy prediction (v2 fallback) ───────────────────────────────────

    def _predict_legacy(self, features: pd.DataFrame,
                        X: np.ndarray, regime: np.ndarray) -> pd.DataFrame:
        pred_raw = self.legacy_model.predict(X)

        direction = np.where(pred_raw > 0.0006, "UP",
                    np.where(pred_raw < -0.0006, "DOWN", "NEUTRAL"))

        mag_score = self._compute_mag_score(pred_raw)
        confidence = np.clip(mag_score, 0, 100)

        strength = np.full(len(pred_raw), "Weak", dtype=object)
        strength[confidence >= MODERATE_THRESHOLD] = "Moderate"
        strength[confidence >= STRONG_THRESHOLD] = "Strong"
        strength[np.isnan(confidence)] = "Weak"

        bbp = self._compute_bbp(features)

        out = pd.DataFrame(index=features.index)
        out["pred_return_4h"] = pred_raw
        out["pred_direction"] = direction
        out["confidence_score"] = confidence
        out["strength_score"] = strength
        out["bull_bear_power"] = bbp
        out["regime"] = regime

        for c in ["open", "high", "low", "close"]:
            if c in features.columns:
                out[c] = features[c].values

        return out

    # ── Regime detection (same logic as research) ─────────────────────────

    @staticmethod
    def _assign_regime(df: pd.DataFrame) -> np.ndarray:
        """Trailing-only regime classification from close prices."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))

        ret_24h = close.pct_change(24)
        vol_24h = log_ret.rolling(24).std()
        vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

        regime = np.full(len(df), "CHOPPY", dtype=object)
        regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
        regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"
        regime[:168] = "WARMUP"

        return regime

    # ── Actual returns for regime IC ──────────────────────────────────────

    @staticmethod
    def _compute_actual_returns(df: pd.DataFrame) -> np.ndarray:
        """Compute realized 4h returns — only for bars where outcome is known.
        Last HORIZON_BARS entries are NaN (no future data used)."""
        close = df["close"].values.astype(float)
        y = np.full(len(close), np.nan)
        # Only compute for bars where the 4h outcome has already occurred
        for i in range(len(close) - HORIZON_BARS):
            y[i] = close[i + HORIZON_BARS] / close[i] - 1
        return y

    # ── Mag score (expanding percentile) ──────────────────────────────────

    def _compute_mag_score(self, pred: np.ndarray) -> np.ndarray:
        """Percentile of |pred| in expanding history. Range: 0-100."""
        abs_pred = np.abs(pred)
        mag_score = np.full(len(pred), np.nan)

        for i in range(len(pred)):
            if np.isnan(pred[i]):
                continue
            self.pred_history.append(float(abs_pred[i]))

            if len(self.pred_history) < MIN_MAG_HISTORY:
                continue

            hist_arr = np.array(list(self.pred_history)[:-1])
            mag_score[i] = (hist_arr < abs_pred[i]).sum() / len(hist_arr) * 100

        return mag_score

    # ── Regime score (trailing IC) ────────────────────────────────────────

    def _compute_regime_score(self, pred: np.ndarray, y: np.ndarray,
                              regime: np.ndarray) -> np.ndarray:
        regime_score = np.full(len(pred), np.nan)

        for i in range(len(pred)):
            if np.isnan(pred[i]):
                continue

            r = regime[i]
            if r == "WARMUP":
                regime_score[i] = 0.8
                continue

            if r in self.regime_history and len(self.regime_history[r]) >= MIN_REGIME_IC_HISTORY:
                hist = self.regime_history[r]
                preds = np.array([h[0] for h in hist])
                actuals = np.array([h[1] for h in hist])
                ic, _ = spearmanr(preds, actuals)

                if ic > 0.03:
                    regime_score[i] = 1.0
                elif ic > 0:
                    regime_score[i] = 0.6
                else:
                    regime_score[i] = 0.0
            else:
                # Cold-start: conservative until IC data accumulates
                regime_score[i] = 0.6

            if not np.isnan(y[i]):
                if r not in self.regime_history:
                    self.regime_history[r] = []
                self.regime_history[r].append((float(pred[i]), float(y[i])))

        return regime_score

    # ── Bull/Bear Power ───────────────────────────────────────────────────

    @staticmethod
    def _compute_bbp(df: pd.DataFrame) -> np.ndarray:
        """Bull/Bear Power from available Coinglass z-scores."""
        components = []

        if "cg_oi_delta_zscore" in df.columns:
            components.append(df["cg_oi_delta_zscore"].clip(-3, 3) / 3)
        if "cg_funding_close_zscore" in df.columns:
            components.append(-df["cg_funding_close_zscore"].clip(-3, 3) / 3)
        if "cg_taker_delta_zscore" in df.columns:
            components.append(df["cg_taker_delta_zscore"].clip(-3, 3) / 3)
        if "cg_ls_ratio_zscore" in df.columns:
            components.append(-df["cg_ls_ratio_zscore"].clip(-3, 3) / 3)
        if "cg_ls_divergence_zscore" in df.columns:
            components.append(df["cg_ls_divergence_zscore"].clip(-3, 3) / 3)

        if not components:
            return np.zeros(len(df))

        return pd.concat(components, axis=1).mean(axis=1).clip(-1, 1).fillna(0).values
