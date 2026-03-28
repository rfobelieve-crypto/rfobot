import logging

logger = logging.getLogger(__name__)

DELTA_THRESHOLD = 0


# ================================
# 主入口
# ================================
def score_snapshot(features: dict) -> dict:
    snap_type = features.get("snapshot_type", "15m")
    side = (features.get("liquidity_side") or "").lower()

    is_ssl = side == "sell"   # 掃下方 → reversal up
    is_bsl = side == "buy"    # 掃上方 → reversal down

    # 模組分數
    order_flow = _score_order_flow(features, is_ssl, is_bsl, snap_type)
    price = _score_price(features, is_ssl, is_bsl, snap_type)
    oi = _score_oi_module(features, snap_type)
    context = _score_context(features, snap_type)

    # 合併
    rev = (
        order_flow["rev"] +
        price["rev"] +
        oi["rev"] +
        context["rev"]
    )

    cont = (
        order_flow["cont"] +
        price["cont"] +
        oi["cont"] +
        context["cont"]
    )

    return _finalize(rev, cont, order_flow, price, oi, context)


# ================================
# 模組 1：Order Flow（最重要）
# ================================
def _score_order_flow(f, is_ssl, is_bsl, snap_type):
    rev, cont = 0.0, 0.0

    # 權重（HTF > LTF）
    cvd_weight = 3 if snap_type == "15m" else 5
    delta_weight = 1.5 if snap_type == "15m" else 2

    # CVD flip
    cvd_flip = f.get("cvd_sign_flip")
    if cvd_flip is not None:
        if cvd_flip:
            rev += cvd_weight
        else:
            cont += cvd_weight

    # Delta
    delta = f.get("delta_value")
    if delta is not None:
        if is_ssl:
            if delta > DELTA_THRESHOLD:
                rev += delta_weight
            elif delta < -DELTA_THRESHOLD:
                cont += delta_weight
        elif is_bsl:
            if delta < -DELTA_THRESHOLD:
                rev += delta_weight
            elif delta > DELTA_THRESHOLD:
                cont += delta_weight

    return {"rev": rev, "cont": cont}


# ================================
# 模組 2：價格結構（第二重要）
# ================================
def _score_price(f, is_ssl, is_bsl, snap_type):
    rev, cont = 0.0, 0.0

    reclaim_w = 2 if snap_type == "15m" else 4
    break_w = 2 if snap_type == "15m" else 4
    price_w = 1 if snap_type == "15m" else 2

    # reclaim
    reclaim = f.get("reclaim_flag")
    if reclaim is not None:
        if reclaim:
            rev += reclaim_w
        else:
            cont += reclaim_w * 0.5  # 沒 reclaim → 偏 continuation

    # break again
    break_again = f.get("break_again_flag")
    if break_again:
        cont += break_w

    # price direction
    price_pct = f.get("price_change_pct")
    if price_pct is not None:
        if is_ssl:
            if price_pct > 0:
                rev += price_w
            elif price_pct < 0:
                cont += price_w
        elif is_bsl:
            if price_pct < 0:
                rev += price_w
            elif price_pct > 0:
                cont += price_w

    return {"rev": rev, "cont": cont}


# ================================
# 模組 3：OI（輔助）
# ================================
def _score_oi_module(f, snap_type):
    rev, cont = 0.0, 0.0

    multiplier = 1.0 if snap_type == "15m" else 2.0

    oi_pct = f.get("oi_change_total_pct")
    if oi_pct is None:
        return {"rev": rev, "cont": cont}

    oi_pct = float(oi_pct)

    if oi_pct < 0:
        rev += 1 * multiplier
        if oi_pct <= -1.5:
            rev += 1 * multiplier
    elif oi_pct > 0:
        cont += 1 * multiplier
        if oi_pct >= 2.0:
            cont += 1 * multiplier

    return {"rev": rev, "cont": cont}


# ================================
# 模組 4：Context（目前保留）
# ================================
def _score_context(f, snap_type):
    rev, cont = 0.0, 0.0

    # 你之後可以加：
    # session, HTF trend, volatility regime

    return {"rev": rev, "cont": cont}


# ================================
# Finalize（重點）
# ================================
def _finalize(rev, cont, order_flow, price, oi, context):
    total = rev + cont

    if total == 0:
        return {
            "reversal_score": 0.0,
            "continuation_score": 0.0,
            "confidence_score": 0.0,
            "bias": "neutral",
            "final_score": 0.0,
            "modules": {
                "order_flow": order_flow,
                "price": price,
                "oi": oi,
                "context": context,
            },
        }

    bias = "reversal" if rev > cont else "continuation"
    confidence = abs(rev - cont) / total

    # normalized（粗略上限）
    max_raw = 20.0
    normalized = (max(rev, cont) / max_raw) * 100

    # 不再用乘法砍分 → 改加分
    clarity_bonus = confidence * 10
    final_score = min(100.0, round(normalized + clarity_bonus, 2))

    return {
        "reversal_score": round(rev, 2),
        "continuation_score": round(cont, 2),
        "confidence_score": round(confidence, 4),
        "bias": bias,
        "normalized_score": round(normalized, 2),
        "final_score": final_score,

        # ⭐ 這個超重要（之後做 dashboard / AI）
        "modules": {
            "order_flow": order_flow,
            "price": price,
            "oi": oi,
            "context": context,
        },
    }