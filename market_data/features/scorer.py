"""
Rule-based scoring engine for liquidity sweep events.

Produces:
- reversal_score (0~100): likelihood of reversal
- continuation_score (0~100): likelihood of continuation
- confidence_score (0~1): data completeness
- bias: "reversal" / "continuation" / "neutral"

Scoring philosophy:
- Each rule adds/subtracts points based on evidence strength
- Rules are weighted by empirical importance
- OI/liquidation/orderbook rules are defined but skip if data is NULL
- bias = whichever score is higher (neutral if difference < threshold)

Scorer version: v1
"""

import logging

logger = logging.getLogger(__name__)

SCORER_VERSION = "v1"
NEUTRAL_THRESHOLD = 10  # minimum score difference to declare a bias


def score(features: dict) -> dict:
    """
    Given extracted features dict, compute scores.

    Returns dict with: reversal_score, continuation_score,
                       confidence_score, bias, scorer_version
    """
    rev = 50.0  # start neutral
    cont = 50.0
    rules_total = 0
    rules_fired = 0

    side = features.get("liquidity_side", "").lower()
    is_bsl = side == "buy"   # buy-side liquidity = swept highs
    is_ssl = side == "sell"  # sell-side liquidity = swept lows

    # ────────────────────────────────────────────
    # Rule 1: Pre-sweep delta direction (weight: 15)
    # BSL + pre-delta negative → smart money selling before sweep → reversal
    # SSL + pre-delta positive → smart money buying before sweep → reversal
    # ────────────────────────────────────────────
    rules_total += 1
    pre_delta = features.get("pre_delta_usd")
    if pre_delta is not None:
        rules_fired += 1
        if is_bsl and pre_delta < 0:
            rev += 15
        elif is_bsl and pre_delta > 0:
            cont += 10
        elif is_ssl and pre_delta > 0:
            rev += 15
        elif is_ssl and pre_delta < 0:
            cont += 10

    # ────────────────────────────────────────────
    # Rule 2: Post-2h delta imbalance (weight: 20)
    # BSL + negative imbalance → sellers dominate → reversal
    # SSL + positive imbalance → buyers dominate → reversal
    # ────────────────────────────────────────────
    rules_total += 1
    imb_2h = features.get("delta_imbalance_2h")
    if imb_2h is not None:
        rules_fired += 1
        if is_bsl:
            if imb_2h < -0.15:
                rev += 20
            elif imb_2h < -0.05:
                rev += 10
            elif imb_2h > 0.15:
                cont += 20
            elif imb_2h > 0.05:
                cont += 10
        elif is_ssl:
            if imb_2h > 0.15:
                rev += 20
            elif imb_2h > 0.05:
                rev += 10
            elif imb_2h < -0.15:
                cont += 20
            elif imb_2h < -0.05:
                cont += 10

    # ────────────────────────────────────────────
    # Rule 3: CVD slope direction (weight: 15)
    # BSL + negative CVD slope → selling pressure building → reversal
    # SSL + positive CVD slope → buying pressure building → reversal
    # ────────────────────────────────────────────
    rules_total += 1
    cvd_slope = features.get("cvd_slope_2h")
    if cvd_slope is not None:
        rules_fired += 1
        if is_bsl:
            if cvd_slope < 0:
                rev += 15
            else:
                cont += 10
        elif is_ssl:
            if cvd_slope > 0:
                rev += 15
            else:
                cont += 10

    # ────────────────────────────────────────────
    # Rule 4: Delta divergence (weight: 20)
    # Divergence = strong reversal signal
    # ────────────────────────────────────────────
    rules_total += 1
    div_2h = features.get("delta_divergence_2h")
    if div_2h is not None:
        rules_fired += 1
        if div_2h:
            rev += 20
        else:
            cont += 5

    # ────────────────────────────────────────────
    # Rule 5: Absorption (weight: 15)
    # Absorption = price doesn't move despite order flow → reversal setup
    # ────────────────────────────────────────────
    rules_total += 1
    absorption = features.get("absorption_detected")
    if absorption is not None:
        rules_fired += 1
        if absorption:
            rev += 15

    # ────────────────────────────────────────────
    # Rule 6: Buy/sell ratio extreme (weight: 10)
    # ────────────────────────────────────────────
    rules_total += 1
    ratio_2h = features.get("post_2h_buy_sell_ratio")
    if ratio_2h is not None:
        rules_fired += 1
        if is_bsl:
            if ratio_2h < 0.8:  # sellers dominate after BSL → reversal
                rev += 10
            elif ratio_2h > 1.2:  # buyers still dominate → continuation
                cont += 10
        elif is_ssl:
            if ratio_2h > 1.2:  # buyers dominate after SSL → reversal
                rev += 10
            elif ratio_2h < 0.8:  # sellers still dominate → continuation
                cont += 10

    # ────────────────────────────────────────────
    # Rule 7: Post-4h delta confirms 2h direction (weight: 10)
    # If 4h delta agrees with 2h signal, reinforce
    # ────────────────────────────────────────────
    rules_total += 1
    imb_4h = features.get("delta_imbalance_4h")
    if imb_4h is not None and imb_2h is not None:
        rules_fired += 1
        # Same direction = confirmation
        if (imb_2h < 0 and imb_4h < 0) or (imb_2h > 0 and imb_4h > 0):
            # Both point same way → whichever was stronger gets boost
            if is_bsl and imb_4h < -0.1:
                rev += 10
            elif is_bsl and imb_4h > 0.1:
                cont += 10
            elif is_ssl and imb_4h > 0.1:
                rev += 10
            elif is_ssl and imb_4h < -0.1:
                cont += 10

    # ────────────────────────────────────────────
    # Rule 8-10: OI / Liquidation / Orderbook (reserved)
    # These fire only when data becomes available
    # ────────────────────────────────────────────
    for reserved_field in ("oi_change_2h", "liq_buy_usd_2h", "orderbook_imbalance_2h"):
        rules_total += 1
        if features.get(reserved_field) is not None:
            rules_fired += 1
            # TODO: implement when data sources are connected

    # ────────────────────────────────────────────
    # Normalize and determine bias
    # ────────────────────────────────────────────
    # Cap scores at 0-100
    rev = max(0, min(100, rev))
    cont = max(0, min(100, cont))

    # Confidence = fraction of rules that had data to fire
    confidence = round(rules_fired / rules_total, 4) if rules_total > 0 else 0

    # Bias
    diff = rev - cont
    if diff > NEUTRAL_THRESHOLD:
        bias = "reversal"
    elif diff < -NEUTRAL_THRESHOLD:
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
