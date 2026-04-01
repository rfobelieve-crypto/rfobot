"""
Model inference — load exported artifacts and predict.
"""
from __future__ import annotations

import json
import logging
import pickle
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"

DEADZONE_Z = 0.1
STRONG_THRESHOLD = 70
MODERATE_THRESHOLD = 40
ROLLING_WINDOW = 48  # 2 days of 1h bars


class IndicatorEngine:
    """Stateful prediction engine with rolling z-score."""

    def __init__(self):
        # Load XGBoost model (Ridge removed — negative IC on 1h data)
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(str(ARTIFACT_DIR / "xgb_model.json"))

        with open(ARTIFACT_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        # Load warmup history for z-score
        with open(ARTIFACT_DIR / "training_stats.json") as f:
            stats = json.load(f)
        self.pred_history = deque(stats.get("pred_history", []), maxlen=500)

        logger.info("IndicatorEngine loaded: %d features, %d warmup bars",
                     len(self.feature_cols), len(self.pred_history))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Run prediction on feature DataFrame.

        Returns DataFrame with: pred_return_4h, pred_direction,
        confidence_score, strength_score, bull_bear_power, regime
        """
        # Align features
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            logger.warning("Missing features (forward-filled): %s", missing)
            for c in missing:
                features[c] = np.nan
            features = features.ffill()

        X = features[self.feature_cols].fillna(0).values

        # Predict (XGBoost only — Ridge was negative IC on 1h data)
        pred_raw = self.xgb_model.predict(X)
        agreement = np.ones(len(pred_raw))  # no ensemble → always 1.0

        # Rolling z-score using history
        pred_z = np.full(len(pred_raw), np.nan)
        confidence = np.full(len(pred_raw), np.nan)

        for i in range(len(pred_raw)):
            self.pred_history.append(float(pred_raw[i]))

            if len(self.pred_history) < 50:
                continue

            hist = list(self.pred_history)
            window = hist[-ROLLING_WINDOW:] if len(hist) >= ROLLING_WINDOW else hist
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma < 1e-10:
                pred_z[i] = 0.0
            else:
                pred_z[i] = (pred_raw[i] - mu) / sigma

            # Confidence from z-score percentile
            abs_z_hist = np.abs(np.array(window) - mu) / max(sigma, 1e-10)
            mag_pct = (abs_z_hist < abs(pred_z[i])).sum() / len(abs_z_hist) * 100
            agree_w = 0.5 + 0.5 * agreement[i]
            confidence[i] = min(mag_pct * agree_w, 100)

        # Direction & strength
        direction = np.where(pred_z > DEADZONE_Z, "UP",
                    np.where(pred_z < -DEADZONE_Z, "DOWN", "NEUTRAL"))
        direction[np.isnan(pred_z)] = "NEUTRAL"

        strength = np.full(len(pred_z), "Weak", dtype=object)
        strength[confidence >= MODERATE_THRESHOLD] = "Moderate"
        strength[confidence >= STRONG_THRESHOLD] = "Strong"
        strength[np.isnan(confidence)] = "Weak"

        # Bull/Bear Power
        bbp = self._compute_bbp(features)

        out = pd.DataFrame(index=features.index)
        out["pred_return_4h"] = pred_raw
        out["pred_direction"] = direction
        out["confidence_score"] = confidence
        out["strength_score"] = strength
        out["bull_bear_power"] = bbp
        out["regime"] = "LIVE"

        # Add OHLCV for chart
        for c in ["open", "high", "low", "close"]:
            if c in features.columns:
                out[c] = features[c].values

        return out

    def _compute_bbp(self, df: pd.DataFrame) -> np.ndarray:
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
