"""
Rule-based scoring engine v2 for liquidity sweep events.

Produces:
- reversal_score (0~6): likelihood of reversal
- continuation_score (0~6): likelihood of continuation
- confidence_score (0~1): abs(rev - cont) / (rev + cont)
- bias: "reversal" / "continuation" / "neutral"

Scoring rules (SSL example, BSL is mirrored):
  Reversal: CVD turns positive +2, CVD strong (zscore>1) +1, 4h price up +1, reclaim success +2
  Continuation: CVD doesn't turn positive +2, CVD breaks lower +1, 4h price down +1, re-break sweep low +2

Scorer version: v2
"""

import logging

logger = logging.getLogger(__name__)

SCORER_VERSION = "v2"


def score(features: dict) -> dict:
    """
    Given extracted features dict, compute scores using v2 additive rules.

    Returns dict with: reversal_score, continuation_score,
                       confidence_score, bias, scorer_version
    """
    rev = 0.0
    cont = 0.0

    side = (features.get("liquidity_side") or "").lower()
    is_bsl = side == "buy"   # buy-side liquidity = swept highs
    is_ssl = side == "sell"  # sell-side liquidity = swept lows

    # ────────────────────────────────────────────
    # Rule 1: CVD turn direction (+2)
    # SSL: CVD turns positive → reversal; doesn't → continuation
    # BSL: CVD turns negative → reversal; doesn't → continuation
    # ────────────────────────────────────────────
    cvd_turned = features.get("cvd_turned_reversal")
    if cvd_turned is not None:
        if cvd_turned:
            rev += 2
        else:
            cont += 2

    # ────────────────────────────────────────────
    # Rule 2: CVD strength via z-score (+1)
    # SSL: zscore > 1 (strong positive CVD) → reversal
    # BSL: zscore < -1 (strong negative CVD) → reversal
    # Opposite direction → continuation
    # ────────────────────────────────────────────
    zscore = features.get("cvd_zscore_2h")
    if zscore is not None:
        if is_ssl:
            if zscore > 1:
                rev += 1
            elif zscore < -1:
                cont += 1
        elif is_bsl:
            if zscore < -1:
                rev += 1
            elif zscore > 1:
                cont += 1

    # ────────────────────────────────────────────
    # Rule 3: 4h price direction (+1)
    # SSL: price up → reversal; price down → continuation
    # BSL: price down → reversal; price up → continuation
    # ────────────────────────────────────────────
    ret_4h = features.get("price_return_4h")
    if ret_4h is not None:
        if is_ssl:
            if ret_4h > 0:
                rev += 1
            elif ret_4h < 0:
                cont += 1
        elif is_bsl:
            if ret_4h < 0:
                rev += 1
            elif ret_4h > 0:
                cont += 1

    # ────────────────────────────────────────────
    # Rule 4: Reclaim / Re-break structure (+2)
    # Reclaim = price moves back past sweep level → reversal
    # Re-break = price continues past entry → continuation
    # ────────────────────────────────────────────
    reclaim = features.get("reclaim_detected")
    if reclaim is not None:
        if reclaim:
            rev += 2

    rebreak = features.get("rebreak_detected")
    if rebreak is not None:
        if rebreak:
            cont += 2

    # ────────────────────────────────────────────
    # Confidence and bias
    # ────────────────────────────────────────────
    total = rev + cont
    if total > 0:
        confidence = round(abs(rev - cont) / total, 4)
    else:
        confidence = 0.0

    diff = rev - cont
    if diff > 0:
        bias = "reversal"
    elif diff < 0:
        bias = "continuation"
    else:
        bias = "neutral"

    return {
        "reversal_score": round(rev, 4),
        "continuation_score": round(cont, 4),
        "confidence_score": confidence,
        "bias": bias,
        "scorer_version": SCORER_VERSION,
    }
