"""
Model inference — regime-conditional multi-target prediction.

v6: Adds dedicated direction classifier (XGB binary) as high-confidence
    override for the magnitude-derived direction signal.
    Direction model only fires when P(up) > 0.65 or P(up) < 0.35 (top ~10%).
    Otherwise falls back to magnitude-based direction (up_pred - down_pred).
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
DIRECTION_DIR = ARTIFACT_DIR / "direction_models"

# ── Parameters ────────────────────────────────────────────────────────────
STRENGTH_DEADZONE = 0.50     # base |up_pred - down_pred| below this → NEUTRAL
STRONG_THRESHOLD = 80.0      # confidence percentile for Strong
MODERATE_THRESHOLD = 65.0    # confidence percentile for Moderate
HORIZON_BARS = 4             # 4h prediction horizon (4 x 1h bars)
MIN_MAG_HISTORY = 30         # minimum bars before mag_score is valid
MIN_REGIME_IC_HISTORY = 50   # minimum bars per regime before IC is meaningful
REGIME_BLEND_WEIGHT = 0.35   # weight for regime-specific model (vs global)

# Dynamic deadzone parameters
CHOPPY_DEADZONE_MULT = 1.60  # CHOPPY regime: widen deadzone significantly
TREND_DEADZONE_MULT = 0.90   # TRENDING: slightly tighter (momentum is real)
VOL_DEADZONE_SCALE = 0.80    # how much vol ratio affects deadzone (0 = off, 1 = full)

# BBP confirmation gate
BBP_CONFIRM_THRESHOLD = 0.15 # |BBP| must exceed this to confirm direction
BBP_CONFIRM_ENABLED = True   # master switch for BBP confirmation

# Direction model (binary classifier) — high-confidence override
DIR_MODEL_ENABLED = True     # master switch for direction model
DIR_HIGH_CONF = 0.65         # P(up) > this → override to UP
DIR_LOW_CONF = 0.35          # P(up) < this → override to DOWN
DIR_CHOPPY_DISABLED = True   # disable direction override in CHOPPY regime

# Hysteresis: require stronger signal to FLIP direction vs maintain
HYSTERESIS_MULT = 1.40       # to flip UP→DOWN, need strength < -(dz * 1.4)

# Signal cooldown: minimum bars between direction flips
FLIP_COOLDOWN_BARS = 3       # after flipping direction, hold for at least 3 bars

# Dual-horizon blending
BLEND_4H = 0.65              # weight for 4h strength
BLEND_1H = 0.35              # weight for 1h strength

# Regime models to use (skip strength_vol_adj — no signal)
ACTIVE_TARGETS_4H = ["up_move_vol_adj", "down_move_vol_adj"]
ACTIVE_TARGETS_1H = ["up_move_1h_vol_adj", "down_move_1h_vol_adj"]
ACTIVE_TARGETS = ACTIVE_TARGETS_4H + ACTIVE_TARGETS_1H


class IndicatorEngine:
    """Stateful prediction engine with regime-conditional dual-target models."""

    def __init__(self):
        if REGIME_DIR.exists():
            self._load_regime_models()
        else:
            self._load_legacy_model()

        # Direction classifier (optional — enhances direction signal)
        self.dir_model = None
        self.dir_feature_cols: list[str] = []
        if DIR_MODEL_ENABLED:
            self._load_direction_model()

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
        has_1h = "up_move_1h_vol_adj" in self.models
        logger.info("IndicatorEngine v5 loaded: %d models (%s), %d warmup bars",
                     model_count,
                     "4h+1h dual-horizon" if has_1h else "4h only",
                     len(self.pred_history))

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

    def _load_direction_model(self):
        """Load dedicated direction classifier (XGBClassifier, binary)."""
        model_path = DIRECTION_DIR / "direction_xgb.json"
        fc_path = DIRECTION_DIR / "feature_cols.json"
        config_path = DIRECTION_DIR / "config.json"

        if not model_path.exists():
            logger.info("Direction model not found at %s — disabled", model_path)
            return

        try:
            self.dir_model = xgb.XGBClassifier()
            self.dir_model.load_model(str(model_path))

            with open(fc_path) as f:
                self.dir_feature_cols = json.load(f)

            with open(config_path) as f:
                dir_config = json.load(f)

            logger.info("Direction model loaded: %d features, threshold=%.2f/%.2f",
                        len(self.dir_feature_cols),
                        DIR_HIGH_CONF, DIR_LOW_CONF)
        except Exception as e:
            logger.warning("Failed to load direction model: %s", e)
            self.dir_model = None

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

        # Helper: predict one target for all bars
        def _predict_target(target_name: str) -> np.ndarray:
            arr = np.zeros(n)
            models = self.models.get(target_name)
            if models is None:
                return arr
            for i in range(n):
                r = regime[i]
                x_i = target_X[target_name][i:i+1]
                global_model = models.get("global")
                regime_model = models.get(r)
                if global_model is None:
                    continue
                global_p = float(global_model.predict(x_i)[0])
                if regime_model is not None and r != "WARMUP":
                    regime_p = float(regime_model.predict(x_i)[0])
                    arr[i] = ((1 - REGIME_BLEND_WEIGHT) * global_p +
                              REGIME_BLEND_WEIGHT * regime_p)
                else:
                    arr[i] = global_p
            return np.maximum(arr, 0)  # up/down moves >= 0

        # 4h predictions
        up_pred_4h = _predict_target("up_move_vol_adj")
        down_pred_4h = _predict_target("down_move_vol_adj")
        strength_4h = up_pred_4h - down_pred_4h

        # 1h predictions (if models available)
        up_pred_1h = _predict_target("up_move_1h_vol_adj")
        down_pred_1h = _predict_target("down_move_1h_vol_adj")
        has_1h = "up_move_1h_vol_adj" in self.models
        strength_1h = up_pred_1h - down_pred_1h

        # Blend: 4h + 1h (fall back to 4h-only if no 1h models)
        if has_1h:
            strength = BLEND_4H * strength_4h + BLEND_1H * strength_1h
            up_pred = BLEND_4H * up_pred_4h + BLEND_1H * up_pred_1h
            down_pred = BLEND_4H * down_pred_4h + BLEND_1H * down_pred_1h
            logger.debug("Dual-horizon blend: 4h=%.2f, 1h=%.2f", BLEND_4H, BLEND_1H)
        else:
            strength = strength_4h
            up_pred = up_pred_4h
            down_pred = down_pred_4h

        # Bull/Bear Power (compute early — needed for confirmation gate)
        bbp = self._compute_bbp(features)

        # ── Dynamic deadzone ──────────────────────────────────────────
        # Scale deadzone by realized vol ratio and regime
        dynamic_dz = self._compute_dynamic_deadzone(features, regime)

        # ── Direction with hysteresis + cooldown ─────────────────────
        # Initial direction from strength vs deadzone
        direction = np.full(n, "NEUTRAL", dtype=object)
        prev_dir = "NEUTRAL"
        bars_since_flip = 999  # bars since last direction change

        for i in range(n):
            dz = dynamic_dz[i]
            s = strength[i]

            # Compute candidate direction
            if prev_dir == "NEUTRAL":
                if s > dz:
                    candidate = "UP"
                elif s < -dz:
                    candidate = "DOWN"
                else:
                    candidate = "NEUTRAL"
            elif prev_dir == "UP":
                if s < -(dz * HYSTERESIS_MULT):
                    candidate = "DOWN"
                elif s > dz * 0.5:
                    candidate = "UP"
                else:
                    candidate = "NEUTRAL"
            elif prev_dir == "DOWN":
                if s > dz * HYSTERESIS_MULT:
                    candidate = "UP"
                elif s < -dz * 0.5:
                    candidate = "DOWN"
                else:
                    candidate = "NEUTRAL"
            else:
                candidate = "NEUTRAL"

            # Cooldown: if we flipped recently, hold previous direction
            if candidate != prev_dir and prev_dir != "NEUTRAL":
                if bars_since_flip < FLIP_COOLDOWN_BARS:
                    candidate = prev_dir  # hold, don't flip yet

            # Track flips
            if candidate != prev_dir:
                bars_since_flip = 0
            else:
                bars_since_flip += 1

            direction[i] = candidate
            prev_dir = candidate

        # ── BBP confirmation gate ─────────────────────────────────────
        # If BBP disagrees with strength direction, demote to NEUTRAL
        if BBP_CONFIRM_ENABLED:
            for i in range(n):
                if direction[i] == "NEUTRAL":
                    continue
                if direction[i] == "UP" and bbp[i] < -BBP_CONFIRM_THRESHOLD:
                    direction[i] = "NEUTRAL"
                elif direction[i] == "DOWN" and bbp[i] > BBP_CONFIRM_THRESHOLD:
                    direction[i] = "NEUTRAL"

        # ── Direction classifier override (high-confidence only) ────────
        dir_prob = np.full(n, 0.5)  # default: uncertain
        if self.dir_model is not None:
            try:
                dir_X = self._prepare_direction_features(features)
                dir_prob = self.dir_model.predict_proba(dir_X)[:, 1]  # P(up)
                # Only override when classifier is very confident
                # Skip override in CHOPPY regime (model unreliable in ranging markets)
                for i in range(n):
                    if DIR_CHOPPY_DISABLED and regime[i] == "CHOPPY":
                        continue
                    if dir_prob[i] > DIR_HIGH_CONF and direction[i] != "UP":
                        direction[i] = "UP"
                    elif dir_prob[i] < DIR_LOW_CONF and direction[i] != "DOWN":
                        direction[i] = "DOWN"
                n_override = int(np.sum((dir_prob > DIR_HIGH_CONF) | (dir_prob < DIR_LOW_CONF)))
                if n_override > 0:
                    logger.info("Direction model: %d/%d bars overridden", n_override, n)
            except Exception as e:
                logger.warning("Direction model inference failed: %s", e)

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
        out["dynamic_deadzone"] = dynamic_dz
        out["dir_prob_up"] = dir_prob

        for c in ["open", "high", "low", "close"]:
            if c in features.columns:
                out[c] = features[c].values

        return out

    def _prepare_direction_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for direction classifier."""
        missing = [c for c in self.dir_feature_cols if c not in features.columns]
        if missing:
            logger.debug("Direction model missing %d features (zero-filled)", len(missing))
            for c in missing:
                features[c] = np.nan
        return features[self.dir_feature_cols].fillna(0).values

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

    # ── Dynamic deadzone ────────────────────────────────────────────────

    @staticmethod
    def _compute_dynamic_deadzone(features: pd.DataFrame,
                                  regime: np.ndarray) -> np.ndarray:
        """
        Compute per-bar dynamic deadzone based on:
        1. Realized vol ratio (high vol → wider deadzone)
        2. Regime (CHOPPY → wider, TRENDING → tighter)
        """
        n = len(features)
        dz = np.full(n, STRENGTH_DEADZONE)

        # Vol-based scaling: realized_vol_24h / expanding median
        if "realized_vol_20b" in features.columns:
            vol = features["realized_vol_20b"].values.astype(float)
            # Expanding median of vol (robust reference)
            vol_series = pd.Series(vol)
            vol_median = vol_series.expanding(min_periods=24).median().values

            for i in range(n):
                if np.isnan(vol[i]) or np.isnan(vol_median[i]) or vol_median[i] <= 0:
                    continue
                vol_ratio = vol[i] / vol_median[i]
                # Scale: ratio=1 → no change, ratio=2 → deadzone × (1 + 0.8)
                vol_factor = 1.0 + (vol_ratio - 1.0) * VOL_DEADZONE_SCALE
                vol_factor = np.clip(vol_factor, 0.5, 2.5)  # guard rails
                dz[i] *= vol_factor

        # Regime-based scaling
        for i in range(n):
            r = regime[i]
            if r == "CHOPPY":
                dz[i] *= CHOPPY_DEADZONE_MULT
            elif r in ("TRENDING_BULL", "TRENDING_BEAR"):
                dz[i] *= TREND_DEADZONE_MULT
            # WARMUP: keep base deadzone

        return dz

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
