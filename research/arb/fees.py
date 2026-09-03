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
        "taker_bps": 4.5, "rebate": 0.0, "verified": True,
        "note": "Hyperliquid docs, base tier 0.045%. Volume tiers need $5M+/14d; "
                "HYPE staking discounts reach 40% only at 500,000 HYPE.",
    },
    # Every HIP-3 builder dex reads deployerFeeScale = 1.0 with growth mode,
    # i.e. the trader pays the HL schedule and the deployer keeps its share.
    # io's effective zero comes from Entropy's referral rebate, which is a
    # promotion, not a schedule — and it is the one number a small live fill
    # still has to confirm.
    "IO": {
        "taker_bps": 4.5, "rebate": 1.0, "verified": False,
        "note": "Entropy Tier-4 referral claims 100% rebate. UNCONFIRMED on a "
                "real fill; HL docs also say deployers keep at most 50% of "
                "fees, which would cap the rebate at ~2.25 bps.",
    },
    "xyz": {"taker_bps": 4.5, "rebate": 0.0, "verified": True,
            "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "para": {"taker_bps": 4.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "mkts": {"taker_bps": 4.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "hyna": {"taker_bps": 4.5, "rebate": 0.0, "verified": True,
             "note": "deployerFeeScale 1.0 + growth mode -> HL schedule."},
    "lighter": {"taker_bps": 0.0, "rebate": 0.0, "verified": True,
                "note": "Lighter docs: standard account 0 maker / 0 taker. "
                        "Structural, not a promotion."},
    "lighter-rh": {"taker_bps": 0.0, "rebate": 0.0, "verified": True,
                   "note": "Same schedule; quotes in USDG, so part of any "
                           "premium is the stablecoin basis."},
    "bitget": {
        "taker_bps": 6.0, "rebate": 0.50, "verified": True,
        "note": "Bitget publishes takerFeeRate per contract (0.0006 = 6 bps). "
                "Rebate 50% stated by the operator 2026-09-03.",
    },
    "okx": {
        "taker_bps": 5.0, "rebate": 0.45, "verified": False,
        "note": "ASSUMED 0.05% taker (standard tier); OKX does not publish it "
                "on an unauthenticated endpoint. Rebate 45% stated by the "
                "operator 2026-09-03. Confirm from the account's fee page.",
    },
}

DEFAULT = {"taker_bps": 4.5, "rebate": 0.0, "verified": False,
           "note": "unknown venue - charged at the HL schedule to stay pessimistic"}


def fee_bps(venue: str) -> float:
    """Effective taker cost of one crossing on this venue, in bps."""
    v = VENUES.get(venue, DEFAULT)
    return v["taker_bps"] * (1.0 - v["rebate"])


def round_trip_bps(leg_a: str, leg_b: str) -> float:
    """Both legs, in and out."""
    return 2.0 * (fee_bps(leg_a) + fee_bps(leg_b))


def required_band_bps(leg_a: str, leg_b: str) -> float:
    """The band this pair needs before a trade breaks even."""
    return 2.0 * round_trip_bps(leg_a, leg_b)


def net_per_trade_bps(band_bps: float, leg_a: str, leg_b: str) -> float:
    """What one round trip actually keeps: half the band, minus four crossings."""
    return band_bps / 2.0 - round_trip_bps(leg_a, leg_b)


def unverified(leg_a: str, leg_b: str) -> list[str]:
    """Legs whose fee is assumed rather than confirmed — the board must say so."""
    return [v for v in (leg_a, leg_b)
            if not VENUES.get(v, DEFAULT).get("verified", False)]


def table() -> list[dict]:
    """Public-safe fee table for the site (rates are percentages, not dollars)."""
    return [{"venue": k, "taker_bps": v["taker_bps"],
             "rebate_pct": round(v["rebate"] * 100),
             "effective_bps": round(fee_bps(k), 2),
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
