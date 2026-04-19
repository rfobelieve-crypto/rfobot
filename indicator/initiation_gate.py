"""
Production-time Strong-signal gating logic for the Initiation model.

Replaces the old `dir_prob > 0.60 / < 0.40` gate in the indicator pipeline.

Inputs (per bar):
    p_long_init   : P(long initiation) from XGBClassifier
    p_short_init  : P(short initiation) from XGBClassifier
    mag_pred      : magnitude model output (|ret_4h|)
    mag_percentile: rolling percentile of |mag_pred| (existing infra)
    close, high, low : current bar OHLC for live breakout check
    rolling_high_20 : trailing 20-bar high (EXCLUDING current bar)
    rolling_low_20  : trailing 20-bar low  (EXCLUDING current bar)

Outputs: InitiationSignal dataclass with direction, strength, scores.

Gate rules:
    Strong LONG:
        p_long_init  >= prob_strong_threshold  (default 0.50, tuned by training)
        AND close > rolling_high_20            (trailing breakout confirm)
        AND mag_percentile >= mag_strong_pct   (default 0.65)

    Strong SHORT: symmetric.

    Moderate: same, except lower prob threshold (default 0.40) and no breakout
              confirm required.

    Weak/neutral otherwise.

Conflict handling:
    If BOTH long and short predict Strong (rare — possible near squeeze
    reversals), pick the side with higher prob and downgrade to Moderate,
    log the conflict.

Calibration:
    The prob thresholds are set at training time from the walk-forward OOS
    distribution. `thresholds.json` next to the model artifacts carries the
    per-side cutoffs (top-5% cutoff = Strong, top-15% = Moderate).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InitiationSignal:
    direction: str      # "UP" / "DOWN" / "NEUTRAL"
    strength: str       # "Strong" / "Moderate" / "Weak"
    confidence_score: float   # 0..100
    p_long_init: float
    p_short_init: float
    mag_percentile: float
    reason: str


@dataclass
class InitiationGateConfig:
    prob_strong_long: float = 0.50
    prob_strong_short: float = 0.50
    prob_moderate_long: float = 0.40
    prob_moderate_short: float = 0.40
    mag_strong_pct: float = 0.65
    mag_moderate_pct: float = 0.50
    require_breakout_for_strong: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "InitiationGateConfig":
        if not path.exists():
            logger.warning("initiation gate config not found at %s; using defaults", path)
            return cls()
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def evaluate(
    p_long_init: float,
    p_short_init: float,
    mag_pred: float,
    mag_percentile: float,
    close: float,
    rolling_high_20: Optional[float],
    rolling_low_20: Optional[float],
    config: Optional[InitiationGateConfig] = None,
) -> InitiationSignal:
    if config is None:
        config = InitiationGateConfig()

    bo_up = rolling_high_20 is not None and close > rolling_high_20
    bo_dn = rolling_low_20 is not None and close < rolling_low_20

    # --- try Strong on each side ------------------------------------------
    long_strong = (
        p_long_init >= config.prob_strong_long
        and mag_percentile >= config.mag_strong_pct
        and (bo_up or not config.require_breakout_for_strong)
    )
    short_strong = (
        p_short_init >= config.prob_strong_short
        and mag_percentile >= config.mag_strong_pct
        and (bo_dn or not config.require_breakout_for_strong)
    )

    if long_strong and short_strong:
        logger.warning(
            "initiation conflict: both sides strong (p_long=%.3f p_short=%.3f) — downgrading",
            p_long_init, p_short_init,
        )
        # Downgrade to Moderate on higher side
        if p_long_init >= p_short_init:
            return _moderate("UP", p_long_init, p_short_init, mag_percentile,
                              "conflict downgrade")
        else:
            return _moderate("DOWN", p_long_init, p_short_init, mag_percentile,
                              "conflict downgrade")

    if long_strong:
        conf = _score(p_long_init, mag_percentile, tier="strong")
        return InitiationSignal("UP", "Strong", conf, p_long_init, p_short_init,
                                 mag_percentile, "long_strong: prob+mag+breakout")
    if short_strong:
        conf = _score(p_short_init, mag_percentile, tier="strong")
        return InitiationSignal("DOWN", "Strong", conf, p_long_init, p_short_init,
                                 mag_percentile, "short_strong: prob+mag+breakout")

    # --- Moderate ----------------------------------------------------------
    long_mod = (p_long_init >= config.prob_moderate_long
                and mag_percentile >= config.mag_moderate_pct)
    short_mod = (p_short_init >= config.prob_moderate_short
                 and mag_percentile >= config.mag_moderate_pct)

    if long_mod and not short_mod:
        return _moderate("UP", p_long_init, p_short_init, mag_percentile,
                          "long_moderate: prob+mag")
    if short_mod and not long_mod:
        return _moderate("DOWN", p_long_init, p_short_init, mag_percentile,
                          "short_moderate: prob+mag")
    if long_mod and short_mod:
        side = "UP" if p_long_init >= p_short_init else "DOWN"
        return _moderate(side, p_long_init, p_short_init, mag_percentile,
                          "both_moderate: take higher prob")

    # --- Weak --------------------------------------------------------------
    top = max(p_long_init, p_short_init)
    direction = "UP" if p_long_init >= p_short_init else "DOWN"
    if top < 0.30 and mag_percentile < 0.40:
        direction = "NEUTRAL"
    conf = _score(top, mag_percentile, tier="weak")
    return InitiationSignal(direction, "Weak", conf, p_long_init, p_short_init,
                             mag_percentile, f"weak: top_prob={top:.2f}")


def _moderate(direction: str, p_long: float, p_short: float,
              mag_pct: float, reason: str) -> InitiationSignal:
    top = p_long if direction == "UP" else p_short
    return InitiationSignal(direction, "Moderate",
                             _score(top, mag_pct, tier="moderate"),
                             p_long, p_short, mag_pct, reason)


def _score(prob: float, mag_pct: float, tier: str) -> float:
    """
    Confidence 0..100. Keeps same basic shape as existing dual-model scoring
    so chart layer doesn't need to change:
        score = 100 * mag_pct * (0.7 + 0.3 * prob_conviction)
    prob_conviction = (prob - 0.5) * 2 clipped to [0, 1].
    """
    conv = max(0.0, min(1.0, (prob - 0.5) * 2))
    raw = mag_pct * (0.7 + 0.3 * conv) * 100
    return float(round(raw, 1))
