# -*- coding: utf-8 -*-
"""Venue fee table — the single source of truth for the arb line (2026-09-03).

Why this file exists: the whole §0.75 line turned on one number nobody had
written down. The recording family was described as "both legs 0 bps", which
turned out to be Entropy's referral rebate rather than a schedule, and every
capturable figure quoted before today implicitly assumed zero fees. A fee
belongs in one place, with its source and the date it was checked, so an
estimate can never quietly diverge from it.

THE DECISION RULE (used everywhere downstream):

    a trade enters at the band and closes when the premium has come back to
    HALF the band, so it captures band/2; it crosses BOTH legs on the way in
    and BOTH on the way out, so it pays 2 x (fee_a + fee_b).

        net_bps_per_trade = band/2 - 2 * (fee_a_eff + fee_b_eff)
        required_band     = 4 * (fee_a_eff + fee_b_eff)

    fee_eff = taker_bps * (1 - rebate). Rebates are the operator's own
    account terms, not public facts — they are stated as such below.

Verified 2026-09-03 unless marked ASSUMED. Re-check before any live trade:
a promotional zero is exactly the kind of number that expires quietly.
"""
from __future__ import annotations

# venue key -> (taker bps, rebate fraction, verified?, note)
VENUES: dict[str, dict] = {
    "HL": {
        "taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": True,
        "note": "Hyperliquid docs, base tier 0.045%. Volume tiers need $5M+/14d; "
                "HYPE staking discounts reach 40% only at 500,000 HYPE.",
    },
    # Every HIP-3 builder dex reads deployerFeeScale = 1.0 with growth mode,
    # i.e. the trader pays the HL schedule and the deployer keeps its share.
    # io's effective zero comes from Entropy's referral rebate, which is a
    # promotion, not a schedule — and it is the one number a small live fill
    # still has to confirm.
    "IO": {
        "taker_bps": 4.5, "maker_bps": 1.5, "rebate": 1.0, "verified": False,
        "note": "Entropy Tier-4 referral claims 100% rebate. UNCONFIRMED on a "
                "real fill; HL docs also say deployers keep at most 50% of "
                "fees, which would cap the rebate at ~2.25 bps.",
    },
    "xyz": {"taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": True,
            "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "para": {"taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "mkts": {"taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "hyna": {"taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "lighter": {"taker_bps": 0.0, "maker_bps": 0.0, "rebate": 0.0, "verified": True,
                "note": "Lighter docs: standard account 0 maker / 0 taker. "
                        "Structural, not a promotion."},
    "lighter-rh": {"taker_bps": 0.0, "maker_bps": 0.0, "rebate": 0.0, "verified": True,
                   "note": "Same schedule; quotes in USDG, so part of any "
                           "premium is the stablecoin basis."},
    "bitget": {
        "taker_bps": 6.0, "maker_bps": 2.0, "rebate": 0.50, "verified": True,
        "note": "Bitget publishes takerFeeRate per contract (0.0006 = 6 bps). "
                "Rebate 50% stated by the operator 2026-09-03.",
    },
    "okx": {
        "taker_bps": 5.0, "maker_bps": 2.0, "rebate": 0.45, "verified": False,
        "note": "ASSUMED 0.05% taker (standard tier); OKX does not publish it "
                "on an unauthenticated endpoint. Rebate 45% stated by the "
                "operator 2026-09-03. Confirm from the account's fee page.",
    },
}

DEFAULT = {"taker_bps": 4.5, "maker_bps": 1.5, "rebate": 0.0, "verified": False,
           "note": "unknown venue - charged at the HL schedule to stay pessimistic"}


def fee_bps(venue: str, maker: bool = False, rebate: bool = True) -> float:
    """Effective cost of ONE crossing on this venue, in bps.

    rebate=False prices the SCHEDULE only. Every rebate here is account
    terms rather than a public fact (IO's 100% is explicitly UNCONFIRMED),
    so any figure a verdict rests on has to be showable both ways —
    otherwise a promotion that expires quietly takes the conclusion with
    it (2026-09-03: the §0.75 family's "both legs 0 bps" was exactly this).
    """
    v = VENUES.get(venue, DEFAULT)
    base = v["maker_bps"] if maker else v["taker_bps"]
    return base * (1.0 - (v["rebate"] if rebate else 0.0))


# Execution mode. 2026-09-03: the author of the recorder we run published his
# own account of this trade (edgeX, 60 days) and the single biggest difference
# from our model is that he rests orders instead of crossing -- "挂单便宜但不保证
# 成交，吃单一定成交但会滑点", with three documented patterns (rest A then cross
# B; rest A then try to rest B; rest both and cross whichever side is left).
# Crossing four times is the PESSIMISTIC bound, not the only way to trade, and
# on Bitget it is the difference between 6 bps and 2 bps per crossing.
MODES = {
    "taker_taker": "四次吃單（最保守；我們原本的假設）",
    "maker_taker": "一腿掛單、一腿吃單（作者的方式 A）",
    "maker_maker": "兩腿都掛單（作者的方式 C；不保證成交，未成交就切吃單）",
}


def round_trip_bps(leg_a: str, leg_b: str, mode: str = "taker_taker",
                   rebate: bool = True) -> float:
    """Both legs, in and out, under one execution mode."""
    if mode == "maker_maker":
        per_leg = (fee_bps(leg_a, True, rebate)
                   + fee_bps(leg_b, True, rebate))
    elif mode == "maker_taker":
        # rest on the cheaper-to-rest venue, cross the other
        per_leg = min(fee_bps(leg_a, True, rebate) + fee_bps(leg_b, False, rebate),
                      fee_bps(leg_a, False, rebate) + fee_bps(leg_b, True, rebate))
    else:
        per_leg = fee_bps(leg_a, False, rebate) + fee_bps(leg_b, False, rebate)
    return 2.0 * per_leg


def required_band_bps(leg_a: str, leg_b: str, mode: str = "taker_taker",
                      rebate: bool = True) -> float:
    """The band this pair needs before a trade breaks even."""
    return 2.0 * round_trip_bps(leg_a, leg_b, mode, rebate)


def net_per_trade_bps(band_bps: float, leg_a: str, leg_b: str,
                      mode: str = "taker_taker", rebate: bool = True) -> float:
    """What one round trip keeps: half the band, minus four crossings."""
    return band_bps / 2.0 - round_trip_bps(leg_a, leg_b, mode, rebate)


def unverified(leg_a: str, leg_b: str) -> list[str]:
    """Legs whose fee is assumed rather than confirmed — the board must say so."""
    return [v for v in (leg_a, leg_b)
            if not VENUES.get(v, DEFAULT).get("verified", False)]


def table() -> list[dict]:
    """Public-safe fee table for the site (rates are percentages, not dollars)."""
    return [{"venue": k, "taker_bps": v["taker_bps"],
             "rebate_pct": round(v["rebate"] * 100),
             "effective_bps": round(fee_bps(k), 2),
             "maker_bps": v["maker_bps"],
             "effective_maker_bps": round(fee_bps(k, True), 2),
             "verified": v["verified"], "note": v["note"]}
            for k, v in VENUES.items()]


if __name__ == "__main__":  # quick reference: what each combination needs
    import itertools
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"{'配對組合':<26}{'單腿費(bps)':>14}{'來回成本':>10}{'需要的帶':>10}")
    for a, b in itertools.combinations(VENUES, 2):
        print(f"{a + ' x ' + b:<26}{fee_bps(a):>6.2f}+{fee_bps(b):<7.2f}"
              f"{round_trip_bps(a, b):>10.2f}{required_band_bps(a, b):>10.2f}")
