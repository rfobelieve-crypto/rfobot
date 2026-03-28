"""
Window-specific scoring engine for event snapshots.

15m scoring (max 3 each side):
  Reversal:  cvd_sign_flip=true +1, delta>0 +1, reclaim=true +1
  Continuation: cvd_sign_flip=false +1, delta<0 +1, break_again=true +1

1h scoring (max 6 each side):
  Reversal:  cvd_sign_flip=true +2, delta>threshold +1, reclaim=true +2, price_up +1
  Continuation: cvd_sign_flip=false +2, delta<-threshold +1, break_again=true +2, reclaim=false +1

4h scoring: same as 1h (final snapshot)

confidence = abs(rev - cont) / max(rev + cont, 1)
"""

import logging

logger = logging.getLogger(__name__)

# Delta threshold for 1h/4h rules (USD)
DELTA_THRESHOLD = 0


def score_snapshot(features: dict) -> dict:
    """
    Score a snapshot based on its snapshot_type.

    Args:
        features: dict from snapshot_builder.build_snapshot()

    Returns:
        dict with reversal_score, continuation_score, confidence_score, bias
    """
    snap_type = features.get("snapshot_type", "15m")
    side = (features.get("liquidity_side") or "").lower()
    is_ssl = side == "sell"
    is_bsl = side == "buy"

    if snap_type == "15m":
        return _score_15m(features, is_ssl, is_bsl)
    else:
        # 1h and 4h use the same rules
        return _score_1h_4h(features, is_ssl, is_bsl)


def _score_15m(f: dict, is_ssl: bool, is_bsl: bool) -> dict:
    """15m scoring: max 3 reversal, 3 continuation."""
    rev = 0.0
    cont = 0.0

    # Rule 1: CVD sign flip (+1)
    cvd_flip = f.get("cvd_sign_flip")
    if cvd_flip is not None:
        if cvd_flip:
            rev += 1
        else:
            cont += 1

    # Rule 2: Delta direction (+1)
    delta = f.get("delta_value")
    if delta is not None:
        # SSL: positive delta = reversal (buyers stepping in)
        # BSL: negative delta = reversal (sellers stepping in)
        if is_ssl:
            if delta > 0:
                rev += 1
            elif delta < 0:
                cont += 1
        elif is_bsl:
            if delta < 0:
                rev += 1
            elif delta > 0:
                cont += 1

    # Rule 3: Reclaim / Break again (+1)
    reclaim = f.get("reclaim_flag")
    if reclaim is not None and reclaim:
        rev += 1

    break_again = f.get("break_again_flag")
    if break_again is not None and break_again:
        cont += 1

    return _finalize(rev, cont)


def _score_1h_4h(f: dict, is_ssl: bool, is_bsl: bool) -> dict:
    """1h / 4h scoring: max 6 reversal, 6 continuation."""
    rev = 0.0
    cont = 0.0

    # Rule 1: CVD sign flip (+2)
    cvd_flip = f.get("cvd_sign_flip")
    if cvd_flip is not None:
        if cvd_flip:
            rev += 2
        else:
            cont += 2

    # Rule 2: Delta strength (+1)
    delta = f.get("delta_value")
    if delta is not None:
        if is_ssl:
            if delta > DELTA_THRESHOLD:
                rev += 1
            elif delta < -DELTA_THRESHOLD:
                cont += 1
        elif is_bsl:
            if delta < -DELTA_THRESHOLD:
                rev += 1
            elif delta > DELTA_THRESHOLD:
                cont += 1

    # Rule 3: Reclaim / Break again (+2)
    reclaim = f.get("reclaim_flag")
    if reclaim is not None:
        if reclaim:
            rev += 2
        else:
            cont += 1  # reclaim=false → continuation signal

    break_again = f.get("break_again_flag")
    if break_again is not None and break_again:
        cont += 2

    # Rule 4: Price direction (+1)
    price_pct = f.get("price_change_pct")
    if price_pct is not None:
        if is_ssl:
            if price_pct > 0:
                rev += 1
            elif price_pct < 0:
                cont += 1
        elif is_bsl:
            if price_pct < 0:
                rev += 1
            elif price_pct > 0:
                cont += 1

    return _finalize(rev, cont)


def _finalize(rev: float, cont: float) -> dict:
    """Compute confidence and bias from raw scores."""
    total = rev + cont
    confidence = round(abs(rev - cont) / max(total, 1), 4)

    if rev > cont:
        bias = "reversal"
    elif cont > rev:
        bias = "continuation"
    else:
        bias = "neutral"

    return {
        "reversal_score": round(rev, 4),
        "continuation_score": round(cont, 4),
        "confidence_score": confidence,
        "bias": bias,
    }
