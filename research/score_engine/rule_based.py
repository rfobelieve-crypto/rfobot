"""
Rule-based scoring model (V1).

Context: time-series bars, NOT event-relative.
No entry_price. No reclaim/break-again flags.
Scores absolute market state: exhaustion vs momentum.

Modules
───────
1. Flow momentum   (delta ratio + CVD)
2. OI analysis     (position build vs unwind)
3. Funding + liq   (crowding + forced exits)

Output
───────
reversal_score     – signals suggesting current trend will exhaust/reverse
continuation_score – signals suggesting current trend has momentum to continue
final_bias         – which is higher
confidence         – abs(rev − cont) / (rev + cont)
risk_adj_score     – confidence × normalized dominant score  (0–100)
"""
from __future__ import annotations

from research.score_engine.interface import ScoreModel, ScoreOutput

# ── Thresholds ───────────────────────────────────────────────────────────────
_DELTA_RATIO_LOW     = 0.03   # 3%  imbalance — minor
_DELTA_RATIO_HIGH    = 0.10   # 10% — elevated
_DELTA_RATIO_EXTREME = 0.20   # 20% — extreme

_OI_FLAT    = 0.5    # % — considered flat
_OI_HIGH    = 2.0    # % — significant change
_OI_EXTREME = 4.0    # %

_FR_HIGH    = 0.0005  # 0.05% / period — elevated crowding
_FR_EXTREME = 0.001   # 0.10% / period — extreme crowding

# Liquidation thresholds by timeframe (USD)
_LIQ_MED  = {"15m": 2_000_000,  "1h": 8_000_000,  "4h": 30_000_000,  "1d": 120_000_000}
_LIQ_HIGH = {"15m": 10_000_000, "1h": 40_000_000, "4h": 150_000_000, "1d": 600_000_000}

# Max raw score used for normalization (calibrated to module maxima)
_MAX_RAW = 12.0


class RuleBasedModel(ScoreModel):
    """Multi-module rule-based scorer for time-series market state bars."""

    @property
    def model_name(self) -> str:
        return "rule_based"

    def compute_score(self, features: dict, config: dict) -> ScoreOutput:
        tf = features.get("timeframe", "1h")

        flow   = self._score_flow(features)
        oi     = self._score_oi(features)
        fr_liq = self._score_funding_liq(features, tf)

        rev  = flow["rev"]  + oi["rev"]  + fr_liq["rev"]
        cont = flow["cont"] + oi["cont"] + fr_liq["cont"]

        return self._finalize(rev, cont)

    # ── Module 1: Flow momentum ───────────────────────────────────────────────

    def _score_flow(self, f: dict) -> dict:
        """
        Delta ratio + CVD momentum.

        Logic:
        - Extreme delta WITH matching CVD slope → strong continuation signal
        - Delta contradicts CVD slope → momentum divergence → reversal signal
        - CVD flip (slope sign change) → reversal signal regardless of delta
        """
        rev, cont = 0.0, 0.0

        ratio = f.get("delta_ratio")
        slope = f.get("cvd_slope")
        flip  = f.get("cvd_flip")

        # CVD flip: highest-priority reversal signal
        if flip:
            rev += 3.0

        # Delta magnitude
        if ratio is not None:
            abs_r = abs(ratio)
            if abs_r >= _DELTA_RATIO_EXTREME:
                pts = 3.0
            elif abs_r >= _DELTA_RATIO_HIGH:
                pts = 2.0
            elif abs_r >= _DELTA_RATIO_LOW:
                pts = 1.0
            else:
                pts = 0.0

            if pts > 0:
                # Divergence: delta direction contradicts CVD slope → reversal pressure
                if slope is not None:
                    delta_bullish = ratio > 0
                    slope_bullish = slope > 0
                    if delta_bullish != slope_bullish:
                        rev += pts   # divergence
                    else:
                        cont += pts  # confluence
                else:
                    cont += pts * 0.5  # no context: partial credit

        # CVD slope strength (without flip): trend momentum
        if slope is not None and not flip:
            if abs(slope) > 0:
                cont += 1.0

        return {"rev": rev, "cont": cont}

    # ── Module 2: OI analysis ─────────────────────────────────────────────────

    def _score_oi(self, f: dict) -> dict:
        """
        OI change.

        OI dropping = liquidation-driven exits = exhaustion = reversal.
        OI rising   = new positions opened = directional conviction = continuation.
        """
        rev, cont = 0.0, 0.0

        oi_pct = f.get("oi_change_pct")
        if oi_pct is None:
            return {"rev": rev, "cont": cont}

        oi_pct = float(oi_pct)

        if oi_pct <= -_OI_EXTREME:
            rev += 3.0
        elif oi_pct <= -_OI_HIGH:
            rev += 2.0
        elif oi_pct <= -_OI_FLAT:
            rev += 1.0
        elif oi_pct >= _OI_EXTREME:
            cont += 3.0
        elif oi_pct >= _OI_HIGH:
            cont += 2.0
        elif oi_pct >= _OI_FLAT:
            cont += 1.0

        return {"rev": rev, "cont": cont}

    # ── Module 3: Funding rate + liquidations ─────────────────────────────────

    def _score_funding_liq(self, f: dict, tf: str) -> dict:
        """
        Funding rate: crowding indicator.
        Extreme funding + delta confirms crowded direction → exhaustion → reversal.

        Liquidations: forced exits = fuel consumed = reversal likely.
        """
        rev, cont = 0.0, 0.0

        # Funding rate
        funding = f.get("funding_rate")
        if funding is not None:
            funding   = float(funding)
            abs_fr    = abs(funding)
            delta_dir = f.get("delta_direction", "neutral")

            if abs_fr >= _FR_EXTREME:
                # Extreme crowding; if delta confirms the crowded side → rev
                crowded_buy  = funding > 0  # longs overcrowded
                delta_is_buy = delta_dir == "buy"
                if crowded_buy == delta_is_buy:
                    rev += 2.0   # crowded + still piling in = exhaustion
                else:
                    rev += 1.0   # extreme funding alone
            elif abs_fr >= _FR_HIGH:
                rev += 1.0

        # Liquidations
        liq = f.get("liq_total_usd")
        if liq is not None:
            liq  = float(liq)
            med  = _LIQ_MED.get(tf,  8_000_000)
            high = _LIQ_HIGH.get(tf, 40_000_000)
            if liq >= high:
                rev += 2.0
            elif liq >= med:
                rev += 1.0

        return {"rev": rev, "cont": cont}

    # ── Finalize ──────────────────────────────────────────────────────────────

    def _finalize(self, rev: float, cont: float) -> ScoreOutput:
        total = rev + cont

        if total == 0:
            return ScoreOutput(
                reversal_score=0.0,
                continuation_score=0.0,
                final_bias="neutral",
                confidence=0.0,
                risk_adj_score=0.0,
                signal=0,
            )

        confidence = abs(rev - cont) / total
        bias       = "reversal" if rev >= cont else "continuation"

        # Normalize dominant score to 0–100, then weight by confidence
        normalized     = min(100.0, max(rev, cont) / _MAX_RAW * 100)
        risk_adj_score = round(normalized * (0.5 + 0.5 * confidence), 2)

        # Signal flag: conf > 0.5 AND score > 55
        signal = 1 if (confidence > 0.5 and risk_adj_score > 55) else 0

        return ScoreOutput(
            reversal_score=round(rev, 2),
            continuation_score=round(cont, 2),
            final_bias=bias,
            confidence=round(confidence, 4),
            risk_adj_score=risk_adj_score,
            signal=signal,
        )
