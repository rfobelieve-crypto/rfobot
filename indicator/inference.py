"""
Model inference — Dual-model direction-regression architecture.

v7.1: Direction (XGBRegressor) predicts signed 4h path return;
      Magnitude (XGBRegressor) predicts |return_4h| in σ-units.
      Direction decoded via rolling percentile (top 2.5% each tail = Strong,
      top 7.5% = Moderate).  pred_return_4h comes directly from the
      direction regression head.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
DUAL_MODEL_DIR = ARTIFACT_DIR / "dual_model"


def _cfg(key: str, default):
    """Read config value: config_overrides.json → Python default."""
    try:
        from indicator.agents.config_store import get_override
        val = get_override(key)
        return val if val is not None else default
    except Exception:
        return default


def reload_config():
    """Reload tunable parameters from config_overrides.json.

    Called at the start of each update cycle so agent-made changes
    take effect without a restart.
    """
    global STRENGTH_DEADZONE, STRONG_THRESHOLD, MODERATE_THRESHOLD
    global CHOPPY_DEADZONE_MULT, TREND_DEADZONE_MULT, BULL_CONTRA_PENALTY
    global BBP_CONFIRM_THRESHOLD, BBP_CONFIRM_ENABLED
    global HYSTERESIS_MULT, FLIP_COOLDOWN_BARS

    STRENGTH_DEADZONE   = _cfg("STRENGTH_DEADZONE", 0.50)
    STRONG_THRESHOLD    = _cfg("STRONG_THRESHOLD", 80.0)
    MODERATE_THRESHOLD  = _cfg("MODERATE_THRESHOLD", 65.0)
    CHOPPY_DEADZONE_MULT = _cfg("CHOPPY_DEADZONE_MULT", 1.60)
    TREND_DEADZONE_MULT = _cfg("TREND_DEADZONE_MULT", 0.90)
    BULL_CONTRA_PENALTY = _cfg("BULL_CONTRA_PENALTY", 1.0)
    BBP_CONFIRM_THRESHOLD = _cfg("BBP_CONFIRM_THRESHOLD", 0.15)
    BBP_CONFIRM_ENABLED = _cfg("BBP_CONFIRM_ENABLED", True)
    HYSTERESIS_MULT     = _cfg("HYSTERESIS_MULT", 1.40)
    FLIP_COOLDOWN_BARS  = _cfg("FLIP_COOLDOWN_BARS", 1)

    logger.debug("Config reloaded from overrides")


# ── Parameters ────────────────────────────────────────────────────────────
STRENGTH_DEADZONE = 0.50     # base |up_pred - down_pred| below this → NEUTRAL
STRONG_THRESHOLD = 80.0      # confidence percentile for Strong
MODERATE_THRESHOLD = 65.0    # confidence percentile for Moderate
HORIZON_BARS = 4             # 4h prediction horizon (4 x 1h bars)
MIN_MAG_HISTORY = 30         # minimum bars before mag_score is valid

# Dynamic deadzone parameters
CHOPPY_DEADZONE_MULT = 1.60  # CHOPPY regime: widen deadzone significantly
TREND_DEADZONE_MULT = 0.90   # TRENDING: slightly tighter (momentum is real)

# Regime-based contra-trend penalty.  Default 1.0 = no penalty (disabled).
#
# History: introduced as 2.5 in commit abcea69 (2026-04-09), based on an
# in-sample observation that "model IC = -0.03 in BULL regime".  That
# observation was later shown to be statistically unreliable (mistake log
# 2026-04-13).
#
# 2026-04-22 sensitivity sweep on 3696 WF OOS bars
# (research/regime_penalty_sensitivity.py) found:
#   - Overall precision is flat in [65.6%, 65.9%] across penalty ∈ [1.0, 4.0].
#     95% CIs overlap heavily — no penalty value is statistically distinguishable.
#   - Contra-trend signal precision (66–78%) is already >= overall; they are
#     not noise the penalty should suppress.
#   - Aligned signals (trend-following) precision is 64.1% — LOWER than contra.
#     The penalty's founding assumption ("contra are noise, aligned are signal")
#     is reversed by the data.
# Reverting to 1.0 (no penalty) recovers ~80 more valid signals over the
# OOS window with no measurable precision cost.  The code path is retained
# (not deleted) so a future config override can re-enable penalty instantly
# without a redeploy.
BULL_CONTRA_PENALTY = 1.0    # no-op by default; set >1 via config to re-enable
VOL_DEADZONE_SCALE = 0.80    # how much vol ratio affects deadzone (0 = off, 1 = full)

# BBP confirmation gate
BBP_CONFIRM_THRESHOLD = 0.15 # |BBP| must exceed this to confirm direction
BBP_CONFIRM_ENABLED = True   # master switch for BBP confirmation

# Hysteresis: require stronger signal to FLIP direction vs maintain
HYSTERESIS_MULT = 1.40       # to flip UP→DOWN, need strength < -(dz * 1.4)

# Signal cooldown: minimum bars between direction flips
FLIP_COOLDOWN_BARS = 1       # after flipping direction, hold for at least 1 bar


class IndicatorEngine:
    """Stateful prediction engine with dual-model direction-regression architecture."""

    def __init__(self):
        self._load_dual_model()

    def _load_dual_model(self):
        """Load dual-model artifacts (direction regressor + magnitude regressor).

        Direction head: XGBRegressor predicting signed 4h path return.
        Config in direction_reg_config.json controls rolling percentile decoding.
        Magnitude head: XGBRegressor predicting |return_4h| in σ-units.
        """
        self.mode = "dual"

        # Direction-regression config
        cfg_path = DUAL_MODEL_DIR / "direction_reg_config.json"
        with open(cfg_path) as f:
            self.dir_reg_config = json.load(f)
        self.dir_model_type = "regression"
        self.dir_decoding = self.dir_reg_config.get(
            "decoding", "rolling_percentile")
        # Rolling percentile decoding params
        self.dir_pct_window = int(
            self.dir_reg_config.get("percentile_window", 500))
        self.dir_strong_top_frac = float(
            self.dir_reg_config.get("strong_top_frac", 0.05))
        self.dir_moderate_top_frac = float(
            self.dir_reg_config.get("moderate_top_frac", 0.15))
        self.dir_warmup_bars = int(
            self.dir_reg_config.get("warmup_bars", 100))
        # Cold-start fallback (fixed ± thresholds from WF calibration)
        fb = self.dir_reg_config.get("fallback", {})
        self.dir_fallback_strong_up = float(fb.get("strong_up", 0.00175))
        self.dir_fallback_strong_dn = float(fb.get("strong_dn", -0.00175))
        self.dir_fallback_mod_up = float(fb.get("moderate_up", 0.00105))
        self.dir_fallback_mod_dn = float(fb.get("moderate_dn", -0.00105))
        # Rolling buffer of signed predictions for percentile calc
        self.dir_pred_history = deque(maxlen=self.dir_pct_window)

        self.dual_dir_model = xgb.XGBRegressor()
        self.dual_dir_model.load_model(str(DUAL_MODEL_DIR / "direction_xgb.json"))
        with open(DUAL_MODEL_DIR / "direction_feature_cols.json") as f:
            self.dual_dir_features = json.load(f)

        # Magnitude model (XGBRegressor)
        self.dual_mag_model = xgb.XGBRegressor()
        self.dual_mag_model.load_model(str(DUAL_MODEL_DIR / "magnitude_xgb.json"))
        with open(DUAL_MODEL_DIR / "magnitude_feature_cols.json") as f:
            self.dual_mag_features = json.load(f)

        # Shared feature superset (for missing-feature warnings)
        self.feature_cols = sorted(set(self.dual_dir_features + self.dual_mag_features))

        # Pred histories for mag_score + direction-regression rolling percentile
        stats_path = DUAL_MODEL_DIR / "training_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.pred_history = deque(stats.get("pred_history", []), maxlen=500)
            # Direction regression warmup buffer (signed predictions)
            dir_hist = stats.get("dir_pred_history", [])
            self.dir_pred_history = deque(
                dir_hist, maxlen=self.dir_pct_window)
        else:
            self.pred_history = deque(maxlen=500)

        dir_hist_len = len(self.dir_pred_history)
        logger.info("IndicatorEngine v7 (dual-model, dir_type=%s) loaded: "
                     "direction=%d features, magnitude=%d features, "
                     "%d mag-warmup, %d dir-warmup, decoding=%s "
                     "top=%.2f",
                     self.dir_model_type,
                     len(self.dual_dir_features), len(self.dual_mag_features),
                     len(self.pred_history),
                     dir_hist_len,
                     self.dir_decoding,
                     self.dir_strong_top_frac)

    def backfill_mag_pred(self, features: pd.DataFrame) -> pd.Series:
        """Predict magnitude only — no state changes. For backfilling historical bars."""
        if not hasattr(self, "dual_mag_model"):
            return pd.Series(np.nan, index=features.index)

        X = features.reindex(columns=self.dual_mag_features, fill_value=0).fillna(0).values
        mag_sigma = np.maximum(self.dual_mag_model.predict(X), 0)

        # Convert σ-adjusted magnitude to return scale
        if "realized_vol_20b" in features.columns:
            rvol = features["realized_vol_20b"].values.astype(float)
            rvol_median = np.nanmedian(rvol)
            if np.isnan(rvol_median) or rvol_median <= 0:
                rvol_median = 0.005
            rvol = np.where(np.isnan(rvol) | (rvol <= 0), rvol_median, rvol)
            mag = mag_sigma * rvol
        else:
            mag = mag_sigma * 0.005

        return pd.Series(mag, index=features.index)

    def predict(self, features: pd.DataFrame,
                context_features: pd.DataFrame | None = None,
                update_history: bool = True) -> pd.DataFrame:
        """
        Run prediction on feature DataFrame.

        Parameters
        ----------
        features : DataFrame to predict on.
        context_features : Optional larger DataFrame for regime calculation.
            When predicting only a few new bars, pass the full feature history
            so regime detection has enough data (needs 168+ bars).
        update_history : If True, append predictions to rolling buffers
            (dir_pred_history, pred_history). Set False for re-scoring
            historical bars to avoid contaminating the buffers.

        Returns DataFrame with: pred_return_4h, pred_direction,
        confidence_score, strength_score, bull_bear_power, regime
        """
        # Align features (check superset — individual targets may need subsets)
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            logger.warning("Missing features (zero-filled): %s", missing)
            for c in missing:
                features[c] = np.nan

        # Regime detection (trailing-only) — use context if available
        regime_source = context_features if context_features is not None else features
        full_regime = self._assign_regime(regime_source)
        if context_features is not None:
            # Map regime from full context back to the prediction rows
            regime_series = pd.Series(full_regime, index=regime_source.index)
            regime = regime_series.reindex(features.index).fillna("CHOPPY").values
        else:
            regime = full_regime

        return self._predict_dual(features, regime,
                                  update_history=update_history)

    # ── Dual-model prediction (v7) ──────────────────────────────────────

    def _predict_dual(self, features: pd.DataFrame,
                      regime: np.ndarray,
                      update_history: bool = True) -> pd.DataFrame:
        """
        Dual-model inference: independent direction + magnitude pipelines.

        Direction: XGBRegressor → signed 4h path return → rolling percentile decode
        Magnitude: XGBRegressor → |return_4h| in σ-units → × realized_vol
        Combined: pred_return_4h comes directly from direction regression head
        """
        n = len(features)

        # ── Direction model ──
        dir_missing = [c for c in self.dual_dir_features if c not in features.columns]
        if dir_missing:
            for c in dir_missing:
                features[c] = np.nan
        X_dir = features[self.dual_dir_features].fillna(0).values
        # Regression: predicts signed 4h path return directly
        dir_pred_ret = self.dual_dir_model.predict(X_dir).astype(float)
        # Synthesize a P(UP)-like value for legacy downstream code
        # (charts, BBP, log fields). sigmoid(pred_ret * 200) maps a ±1%
        # return to ≈0.88 / ≈0.12, preserving monotonicity with pred_ret.
        dir_prob = 1.0 / (1.0 + np.exp(-dir_pred_ret * 200.0))

        # ── Magnitude model ──
        mag_missing = [c for c in self.dual_mag_features if c not in features.columns]
        if mag_missing:
            for c in mag_missing:
                features[c] = np.nan
        X_mag = features[self.dual_mag_features].fillna(0).values
        mag_pred_sigma = self.dual_mag_model.predict(X_mag)
        mag_pred_sigma = np.maximum(mag_pred_sigma, 0)  # magnitude ≥ 0

        # ── Convert σ-adjusted magnitude to return scale ──
        # Model was trained on y_vol_adj_abs = |return_4h| / realized_vol,
        # so output is in σ-units (~0.5-5). Multiply by realized_vol to get
        # actual return-scale magnitude (~0.001-0.01).
        if "realized_vol_20b" in features.columns:
            rvol = features["realized_vol_20b"].values.astype(float)
            rvol_median = np.nanmedian(rvol)
            if np.isnan(rvol_median) or rvol_median <= 0:
                rvol_median = 0.005  # conservative fallback
            rvol = np.where(np.isnan(rvol) | (rvol <= 0), rvol_median, rvol)
            mag_pred = mag_pred_sigma * rvol
        else:
            logger.warning("realized_vol_20b not in features, using fallback vol=0.005")
            mag_pred = mag_pred_sigma * 0.005

        # Magnitude score is computed up-front (used in confidence display
        # — regression mode does NOT gate Strong on it).
        mag_score = self._compute_mag_score(mag_pred_sigma,
                                                update_history=update_history)

        direction = np.full(n, "NEUTRAL", dtype=object)
        strength_tier = np.full(n, "Weak", dtype=object)
        pred_return = np.zeros(n)
        confidence = np.full(n, np.nan)

        # ── Regression direction path (rolling percentile decoding) ──
        # For each bar we:
        #   1. push pred into dir_pred_history
        #   2. if buffer < warmup, use fallback fixed thresholds
        #      (calibrated at export time from full WF predictions)
        #   3. else compute upper/lower quantile cutoffs from the buffer
        #      (strong_top_frac / 2 on each side — symmetric two-tailed)
        #   4. |pred| outside the cutoffs → Strong; otherwise check
        #      moderate cutoffs; else NEUTRAL/Weak
        strong_frac = self.dir_strong_top_frac
        mod_frac = self.dir_moderate_top_frac
        warmup = self.dir_warmup_bars

        for i in range(n):
            p = float(dir_pred_ret[i])
            abs_p = abs(p)

            # Update rolling buffer BEFORE computing the cutoff so that
            # "now" is included — consistent with mag_score expanding pctile.
            # Skip when re-scoring historical bars to avoid buffer contamination.
            if update_history:
                self.dir_pred_history.append(p)
            buf_len = len(self.dir_pred_history)

            if buf_len < warmup:
                # Cold start → fixed fallback thresholds
                up_strong = self.dir_fallback_strong_up
                dn_strong = self.dir_fallback_strong_dn
                up_mod = self.dir_fallback_mod_up
                dn_mod = self.dir_fallback_mod_dn
            else:
                buf = np.fromiter(self.dir_pred_history, dtype=float)
                up_strong = float(np.quantile(buf, 1.0 - strong_frac / 2.0))
                dn_strong = float(np.quantile(buf, strong_frac / 2.0))
                up_mod = float(np.quantile(buf, 1.0 - mod_frac / 2.0))
                dn_mod = float(np.quantile(buf, mod_frac / 2.0))

            # Regime-aware contra-trend suppression: widen the threshold that
            # fights the prevailing regime, keep the aligned side untouched.
            # CLAUDE.md: "DOWN signals in BULL need 2.5x more conviction";
            # mirrored symmetrically for BEAR (UP needs 2.5x in TRENDING_BEAR).
            reg_i = regime[i] if regime is not None else "CHOPPY"
            up_mul = BULL_CONTRA_PENALTY if reg_i == "TRENDING_BEAR" else 1.0
            dn_mul = BULL_CONTRA_PENALTY if reg_i == "TRENDING_BULL" else 1.0
            up_strong_eff = up_strong * up_mul
            up_mod_eff = up_mod * up_mul
            dn_strong_eff = dn_strong * dn_mul  # negative × 2.5 → more negative
            dn_mod_eff = dn_mod * dn_mul

            # Direction decision: threshold-based using regime-adjusted cutoffs
            if p >= up_strong_eff:
                direction[i] = "UP"
                strength_tier[i] = "Strong"
            elif p <= dn_strong_eff:
                direction[i] = "DOWN"
                strength_tier[i] = "Strong"
            elif p >= up_mod_eff:
                direction[i] = "UP"
                strength_tier[i] = "Moderate"
            elif p <= dn_mod_eff:
                direction[i] = "DOWN"
                strength_tier[i] = "Moderate"
            # else direction stays NEUTRAL / tier stays Weak

            # pred_return_4h comes DIRECTLY from the regression head.
            pred_return[i] = p

            # Confidence 0-100: 80 pts from |pred_ret| scaled to the
            # current Strong cutoff magnitude, + 20 pts mag percentile.
            # Use the unadjusted ref to keep confidence comparable across regimes.
            ref = max(abs(up_strong), abs(dn_strong), 1e-6)
            ret_score = min(abs_p / ref, 1.0) ** 0.6 * 80
            ms = mag_score[i] if not np.isnan(mag_score[i]) else 50.0
            mag_bonus = (ms / 100) * 20
            confidence[i] = float(np.clip(ret_score + mag_bonus, 0, 100))

        # ── Bull/Bear Power ──
        bbp = self._compute_bbp(features)

        # ── Output ──
        out = pd.DataFrame(index=features.index)
        out["pred_return_4h"] = pred_return
        out["pred_direction"] = direction
        out["confidence_score"] = confidence
        out["strength_score"] = strength_tier
        out["bull_bear_power"] = bbp
        out["regime"] = regime
        out["up_pred"] = mag_pred       # magnitude prediction
        out["down_pred"] = mag_pred     # same (symmetric)
        out["strength_raw"] = dir_prob - 0.5  # direction strength
        out["dynamic_deadzone"] = np.full(n, 0.1)
        out["dir_prob_up"] = dir_prob
        out["mag_pred"] = mag_pred      # raw magnitude prediction
        out["dir_pred_ret"] = dir_pred_ret  # raw regression output

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
        WARMUP_BARS = 72  # 3 days hourly — enough for stable vol percentile

        close = df["close"]
        log_ret = np.log(close / close.shift(1))

        ret_24h = close.pct_change(24)
        vol_24h = log_ret.rolling(24).std()
        vol_pct = vol_24h.expanding(min_periods=WARMUP_BARS).rank(pct=True)

        regime = np.full(len(df), "CHOPPY", dtype=object)
        regime[(vol_pct > 0.6).values & (ret_24h > 0.005).values] = "TRENDING_BULL"
        regime[(vol_pct > 0.6).values & (ret_24h < -0.005).values] = "TRENDING_BEAR"
        regime[:WARMUP_BARS] = "WARMUP"

        return regime

    # ── Mag score (expanding percentile) ──────────────────────────────────

    def _compute_mag_score(self, pred: np.ndarray,
                           update_history: bool = True) -> np.ndarray:
        """Percentile of |pred| in expanding history. Range: 0-100."""
        abs_pred = np.abs(pred)
        mag_score = np.full(len(pred), np.nan)

        for i in range(len(pred)):
            if np.isnan(pred[i]):
                continue
            if update_history:
                self.pred_history.append(float(abs_pred[i]))

            if len(self.pred_history) < MIN_MAG_HISTORY:
                continue

            hist_arr = np.array(list(self.pred_history)[:-1])
            mag_score[i] = (hist_arr < abs_pred[i]).sum() / len(hist_arr) * 100

        return mag_score

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
