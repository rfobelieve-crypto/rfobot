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
    price      = _score_price(features, is_ssl, is_bsl, snap_type)
    oi         = _score_oi_module(features, snap_type)
    funding_liq = _score_funding_liq(features, is_ssl, is_bsl, snap_type)
    context    = _score_context(features, snap_type)

    # 合併
    rev = (
        order_flow["rev"] +
        price["rev"] +
        oi["rev"] +
        funding_liq["rev"] +
        context["rev"]
    )

    cont = (
        order_flow["cont"] +
        price["cont"] +
        oi["cont"] +
        funding_liq["cont"] +
        context["cont"]
    )

    return _finalize(rev, cont, order_flow, price, oi, funding_liq, context)


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
# 模組 4：Funding Rate + Liquidation
# ================================
def _score_funding_liq(f, is_ssl, is_bsl, snap_type):
    rev, cont = 0.0, 0.0
    multiplier = 1.0 if snap_type == "15m" else 1.5

    # ── Funding Rate ──────────────────────────────────────────
    # Positive funding = longs overcrowded; negative = shorts overcrowded.
    # At SSL (sweep down): positive funding → longs still holding → reversal fuel
    # At BSL (sweep up):   negative funding → shorts still holding → reversal fuel
    # Opposite direction → continuation bias
    funding = f.get("funding_rate")
    if funding is not None:
        funding = float(funding)
        HIGH_THRESHOLD = 0.0005   # 0.05% per period — elevated
        EXTREME_THRESHOLD = 0.001  # 0.1%  per period — extreme

        if is_ssl:
            if funding > EXTREME_THRESHOLD:
                rev += 2 * multiplier   # extreme long overcrowding at SSL → reversal
            elif funding > HIGH_THRESHOLD:
                rev += 1 * multiplier
            elif funding < -HIGH_THRESHOLD:
                cont += 1 * multiplier  # shorts in control → continuation down
        elif is_bsl:
            if funding < -EXTREME_THRESHOLD:
                rev += 2 * multiplier   # extreme short overcrowding at BSL → reversal
            elif funding < -HIGH_THRESHOLD:
                rev += 1 * multiplier
            elif funding > HIGH_THRESHOLD:
                cont += 1 * multiplier  # longs still in → continuation up

    # ── Liquidations ──────────────────────────────────────────
    # Large liquidations in sweep direction = fuel consumed = reversal likely.
    # liq_sell_usd = liquidated longs (sell pressure from forced exits)
    # liq_buy_usd  = liquidated shorts (buy pressure from forced exits)
    liq_sell = f.get("liq_sell_usd")
    liq_buy  = f.get("liq_buy_usd")

    # Thresholds scale with observation window
    med_threshold = {"15m": 2_000_000, "1h": 8_000_000, "4h": 30_000_000}.get(snap_type, 5_000_000)
    high_threshold = {"15m": 10_000_000, "1h": 40_000_000, "4h": 150_000_000}.get(snap_type, 20_000_000)

    if is_ssl and liq_sell is not None:
        # Long liquidations during downward sweep → exhaustion → reversal
        if liq_sell >= high_threshold:
            rev += 2 * multiplier
        elif liq_sell >= med_threshold:
            rev += 1 * multiplier
        # If short liquidations dominate → sell-side is strange → continuation signal
        if liq_buy is not None and liq_buy is not None:
            if float(liq_buy) > float(liq_sell) * 1.5:
                cont += 1 * multiplier

    elif is_bsl and liq_buy is not None:
        # Short liquidations during upward sweep → exhaustion → reversal
        if liq_buy >= high_threshold:
            rev += 2 * multiplier
        elif liq_buy >= med_threshold:
            rev += 1 * multiplier
        # If long liquidations dominate → unusual → continuation signal
        if liq_sell is not None:
            if float(liq_sell) > float(liq_buy) * 1.5:
                cont += 1 * multiplier

    return {"rev": rev, "cont": cont}


# ================================
# 模組 5：Context（目前保留）
# ================================
def _score_context(f, snap_type):
    rev, cont = 0.0, 0.0

    # 你之後可以加：
    # session, HTF trend, volatility regime

    return {"rev": rev, "cont": cont}


# ================================
# Finalize（重點）
# ================================
def _finalize(rev, cont, order_flow, price, oi, funding_liq, context):
    total = rev + cont

    if total == 0:
        return {
            "reversal_score": 0.0,
            "continuation_score": 0.0,
            "confidence_score": 0.0,
            "bias": "neutral",
            "final_score": 0.0,
            "modules": {
                "order_flow":  order_flow,
                "price":       price,
                "oi":          oi,
                "funding_liq": funding_liq,
                "context":     context,
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
            "order_flow":  order_flow,
            "price":       price,
            "oi":          oi,
            "funding_liq": funding_liq,
            "context":     context,
        },
    }