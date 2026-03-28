import logging

logger = logging.getLogger(__name__)

SCORER_VERSION = "v2_modular"


def score(features: dict) -> dict:
    """
    Modular scoring engine for post-sweep validation.

    Returns:
        {
            "reversal_score": float,
            "continuation_score": float,
            "confidence_score": float,
            "bias": str,
            "normalized_score": float,
            "final_score": float,
            "scorer_version": str,
            "modules": {
                "cvd_direction": {"rev": float, "cont": float},
                "cvd_strength": {"rev": float, "cont": float},
                "price_outcome": {"rev": float, "cont": float},
                "structure": {"rev": float, "cont": float},
            }
        }
    """
    side = (features.get("liquidity_side") or "").lower()
    is_bsl = side == "buy"   # buy-side liquidity swept highs -> reversal expected down
    is_ssl = side == "sell"  # sell-side liquidity swept lows -> reversal expected up

    cvd_direction = _score_cvd_direction(features)
    cvd_strength = _score_cvd_strength(features, is_ssl, is_bsl)
    price_outcome = _score_price_outcome(features, is_ssl, is_bsl)
    structure = _score_structure(features)

    rev = (
        cvd_direction["rev"] +
        cvd_strength["rev"] +
        price_outcome["rev"] +
        structure["rev"]
    )

    cont = (
        cvd_direction["cont"] +
        cvd_strength["cont"] +
        price_outcome["cont"] +
        structure["cont"]
    )

    return _finalize(
        rev=rev,
        cont=cont,
        modules={
            "cvd_direction": cvd_direction,
            "cvd_strength": cvd_strength,
            "price_outcome": price_outcome,
            "structure": structure,
        }
    )


def _score_cvd_direction(features: dict) -> dict:
    """
    Rule 1: CVD turn direction (+2)
    Reversal: CVD turns toward reversal direction
    Continuation: CVD does not turn
    """
    rev = 0.0
    cont = 0.0

    cvd_turned = features.get("cvd_turned_reversal")
    if cvd_turned is not None:
        if cvd_turned:
            rev += 2
        else:
            cont += 2

    return {"rev": rev, "cont": cont}


def _score_cvd_strength(features: dict, is_ssl: bool, is_bsl: bool) -> dict:
    """
    Rule 2: CVD strength via z-score (+1)
    SSL: zscore > 1 -> reversal, zscore < -1 -> continuation
    BSL: zscore < -1 -> reversal, zscore > 1 -> continuation
    """
    rev = 0.0
    cont = 0.0

    zscore = features.get("cvd_zscore_2h")
    if zscore is None:
        return {"rev": rev, "cont": cont}

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

    return {"rev": rev, "cont": cont}


def _score_price_outcome(features: dict, is_ssl: bool, is_bsl: bool) -> dict:
    """
    Rule 3: 4h price direction (+1)
    SSL: price up -> reversal, price down -> continuation
    BSL: price down -> reversal, price up -> continuation
    """
    rev = 0.0
    cont = 0.0

    ret_4h = features.get("price_return_4h")
    if ret_4h is None:
        return {"rev": rev, "cont": cont}

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

    return {"rev": rev, "cont": cont}


def _score_structure(features: dict) -> dict:
    """
    Rule 4: Structure (+2 each)
    reclaim_detected -> reversal
    rebreak_detected -> continuation
    """
    rev = 0.0
    cont = 0.0

    reclaim = features.get("reclaim_detected")
    if reclaim:
        rev += 2

    rebreak = features.get("rebreak_detected")
    if rebreak:
        cont += 2

    return {"rev": rev, "cont": cont}


def _finalize(rev: float, cont: float, modules: dict) -> dict:
    total = rev + cont

    if total == 0:
        return {
            "reversal_score": 0.0,
            "continuation_score": 0.0,
            "confidence_score": 0.0,
            "bias": "neutral",
            "normalized_score": 0.0,
            "final_score": 0.0,
            "scorer_version": SCORER_VERSION,
            "modules": modules,
        }

    confidence = abs(rev - cont) / total

    if rev > cont:
        bias = "reversal"
    elif cont > rev:
        bias = "continuation"
    else:
        bias = "neutral"

    # 理論最大分 = 2 + 1 + 1 + 2 = 6
    max_raw = 6.0
    normalized = round((max(rev, cont) / max_raw) * 100, 2)

    # 不直接乘 confidence，改成 clarity bonus
    clarity_bonus = confidence * 10
    final_score = min(100.0, round(normalized + clarity_bonus, 2))

    return {
        "reversal_score": round(rev, 4),
        "continuation_score": round(cont, 4),
        "confidence_score": round(confidence, 4),
        "bias": bias,
        "normalized_score": normalized,
        "final_score": final_score,
        "scorer_version": SCORER_VERSION,
        "modules": modules,
    }