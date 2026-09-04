# -*- coding: utf-8 -*-
"""The full cost function of one cross-venue round trip (TODO 1.06, 2026-09-04).

fees.py answers exactly one of the seven cost buckets (explicit fees). The
operator's decomposition, which this file implements term by term:

  1 explicit fees   maker/taker, per tier, BOTH legs, in AND out
  2 slippage        a function of SIZE: ~0 below top-of-book, eats the whole
                    band once size exceeds the depth within a few bps
  3 capital         margin locked on BOTH venues x hurdle rate x holding time,
                    plus the idle balance each venue needs so either leg can
                    fill (a function of how often you rebalance)
  4 carry           funding differential (and borrow/staking if any) x time
  5 transfer        bridge/withdrawal fee + the delay, during which one leg
                    is naked: charged as a risk cost sigma*sqrt(t), not a mean
  6 operations      API failure / rate limit / withdrawal halt: normally zero,
                    expected value is not -> P(failure during hold) x loss
  7 tail            counterparty default, stablecoin depeg (USDG!), basis
                    blow-out liquidation: annualised drag = sum p_i x L_i

Every input is tagged VERIFIED or ASSUMED. An ASSUMED number is a
placeholder chosen on the pessimistic side so that the model cannot flatter a
pair; the report prints which buckets rest on assumptions so the reader can
see how much of the answer is measured. Units: bps of notional per ROUND TRIP
unless stated; annualised drags are converted to per-trade with the trade
count per year the pair actually shows.

What is measurable from data we already record (2026-09-04):
  * band, depth at 1 bps / 3 bps (scan v5)           -> bucket 2
  * convergence median minutes (verdict)             -> holding time, 3/4/6
  * funding differential bps/8h (recorder)           -> bucket 4
  * scanner quote failure rate, the 46-min outage    -> bucket 6 priors
Not yet measurable (needs a live account or a decision): margin rates per
venue, hurdle rate, transfer delays, tail probabilities. Those are ASSUMED.

Run: python research/arb/cost_model.py            # family breakdown
     python research/arb/cost_model.py --size 500  # at a given order size
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fees as FEES               # noqa: E402  bucket 1 lives there

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = HERE.parents[1]
OUT = ROOT / "research" / "results" / "arb_cost_model.json"

# ── assumptions, each named so they can be replaced one at a time ──────────
ASSUMED = {
    # 3 capital
    "hurdle_rate_pa": (0.05, "ASSUMED: 5%/yr opportunity cost of stablecoin capital"),
    "margin_frac": {          # initial margin as fraction of notional
        "HL": (0.10, "ASSUMED 10x effective; HL allows more, we would not use it"),
        "IO": (0.20, "ASSUMED 5x: HIP-3 builder dex, conservative"),
        "xyz": (0.20, "ASSUMED 5x"), "para": (0.20, "ASSUMED"), "mkts": (0.20, "ASSUMED"),
        "lighter": (0.10, "ASSUMED 10x"), "lighter-rh": (0.10, "ASSUMED 10x"),
        "okx": (0.10, "ASSUMED 10x"), "bitget": (0.10, "ASSUMED 10x"),
        "binance": (0.10, "ASSUMED 10x"),
    },
    # Operator decision 2026-09-04 (COST_INVENTORY #3): buffers are
    # ASYMMETRIC -- the big venue holds 1x working notional idle, the small
    # venue holds a flat $300 and is topped up just-in-time. This is the
    # R1-vs-C3 trade the operator chose; the number is his, not a guess.
    "idle_buffer_frac_big": (1.0, "DECIDED: big venue (okx/bitget/binance) 1x notional idle"),
    "idle_buffer_usd_small": (300.0, "DECIDED 2026-09-04: small venue holds $300 idle"),
    # 5 transfer
    "transfer_fee_bps": (2.0, "ASSUMED: bridge/withdrawal ~$1-2 per $1k moved, amortised"),
    "transfer_delay_min": (20.0, "ASSUMED: 20 min single-leg exposure per rebalance"),
    "rebalances_per_trade": (0.05, "ASSUMED: one rebalance every 20 round trips"),
    "vol_ann": (0.60, "ASSUMED: 60% annualised vol of the underlying for the risk charge"),
    "risk_price": (1.0, "ASSUMED: charge one sigma of the naked-leg move as cost"),
    # 6 operations
    "leg_fail_rate": (0.06, "MEASURED 2026-09-04: scanner quotes 2975/3164 per cycle -> "
                            "~6% of legs unquotable at any moment (proxy for API failure)"),
    "fail_loss_bps": (5.0, "ASSUMED: a failed second leg costs ~half the band + a crossing"),
    "outage_hours_pa": (46 / 60 / 5 * 365, "MEASURED: one 46-min full outage in 5 days -> "
                                            "annualised hours; charged only if it hits a hold"),
    # 7 tail (annual probabilities x loss fraction of deployed capital)
    "tail": [
        ("venue default (small dex)", 0.05, 0.50, "ASSUMED"),
        ("stablecoin depeg USDG/USDC", 0.10, 0.03, "ASSUMED; lighter-rh quotes in USDG"),
        ("basis blow-out liquidation", 0.10, 0.10, "ASSUMED; the 2026-01-27 $120M gold story"),
    ],
}


@dataclass
class TradeSpec:
    leg_a: str
    leg_b: str
    band_bps: float                      # executable room at entry
    size_usd: float                      # order notional per leg
    depth_top: float                     # min(top-of-book usd, both legs)
    depth_1bps: float                    # cumulative usd within 1 bps
    depth_3bps: float                    # cumulative usd within 3 bps
    hold_minutes: float                  # median time to converge
    fund_diff_bps_8h: float              # hedge - entropy, sign = we receive if positive
    trades_per_year: float
    mode: str = "taker_taker"
    rebate: bool = True
    notes: list = field(default_factory=list)


def slippage_bps(size: float, top: float, d1: float, d3: float) -> tuple[float, str]:
    """Walk the three depth buckets we record. Piecewise-linear and pessimistic:
    the part of the order beyond the 3 bps bucket is charged at the band
    itself (i.e. assumed to eat everything) so oversizing is never flattered."""
    if size <= 0:
        return 0.0, "MEASURED"
    if size <= top:
        return 0.0, "MEASURED"
    if size <= d1:
        return 0.5 * (size - top) / size, "MEASURED"        # avg 0.5 bps on the slice
    if size <= d3:
        part1 = 0.5 * (d1 - top)
        part3 = 2.0 * (size - d1)                             # avg 2 bps on 1-3 bps slice
        return (part1 + part3) / size, "MEASURED"
    return float("inf"), "MEASURED: size exceeds 3-bps depth"


def cost_breakdown(t: TradeSpec) -> dict:
    a, b = t.leg_a, t.leg_b
    hold_h = t.hold_minutes / 60.0
    hold_yr = hold_h / 8760.0
    out, tags = {}, {}

    # 1 explicit fees (both legs, in and out) — fees.py is the owner
    out["1_fees"] = FEES.round_trip_bps(a, b, t.mode, t.rebate)
    tags["1_fees"] = ("VERIFIED" if not FEES.unverified(a, b)
                      else f"unverified legs: {FEES.unverified(a, b)}")

    # 2 slippage, both legs, in and out (4 crossings in taker mode; the
    # resting leg pays none if it fills at its price)
    s, tag = slippage_bps(t.size_usd, t.depth_top, t.depth_1bps, t.depth_3bps)
    crossings = {"taker_taker": 4, "maker_taker": 2, "maker_maker": 0}[t.mode]
    out["2_slippage"] = s * crossings if math.isfinite(s) else float("inf")
    tags["2_slippage"] = tag

    # 3 capital: margin on both legs + idle buffer on both venues, x hurdle x hold
    mf_a = ASSUMED["margin_frac"].get(a, (0.2, "ASSUMED"))[0]
    mf_b = ASSUMED["margin_frac"].get(b, (0.2, "ASSUMED"))[0]
    big = {"okx", "bitget", "binance"}
    buf = 0.0
    for v in (a, b):
        buf += (ASSUMED["idle_buffer_frac_big"][0] if v in big
                else ASSUMED["idle_buffer_usd_small"][0] / max(t.size_usd, 1.0))
    locked = (mf_a + mf_b + buf)                                   # x notional
    out["3_capital"] = locked * ASSUMED["hurdle_rate_pa"][0] * hold_yr * 1e4
    tags["3_capital"] = "ASSUMED (hurdle, margin, idle buffer); hold MEASURED"

    # 4 carry: funding differential over the hold. Positive diff = we receive.
    out["4_carry"] = -t.fund_diff_bps_8h * (hold_h / 8.0)
    tags["4_carry"] = "MEASURED (recorder fund_diff); sign per pair"

    # 5 transfer: fee amortised + naked-leg risk charge during the delay
    per_rb = ASSUMED["transfer_fee_bps"][0]
    delay_yr = ASSUMED["transfer_delay_min"][0] / 60 / 8760
    risk = ASSUMED["risk_price"][0] * ASSUMED["vol_ann"][0] * math.sqrt(delay_yr) * 1e4
    out["5_transfer"] = ASSUMED["rebalances_per_trade"][0] * (per_rb + risk)
    tags["5_transfer"] = "ASSUMED (all inputs)"

    # 6 operations: P(a leg fails while we need it) x loss
    p_fail = 1 - (1 - ASSUMED["leg_fail_rate"][0]) ** 2       # either leg
    p_outage = min(1.0, ASSUMED["outage_hours_pa"][0] * hold_h / 8760)
    out["6_ops"] = (p_fail + p_outage) * ASSUMED["fail_loss_bps"][0]
    tags["6_ops"] = "leg_fail MEASURED (scanner); loss ASSUMED"

    # 7 tail: annualised drag on deployed capital -> per trade
    drag_pa = sum(p * L for _, p, L, _ in ASSUMED["tail"])       # fraction of capital / yr
    if t.trades_per_year >= 12:
        out["7_tail"] = drag_pa * locked * 1e4 / t.trades_per_year
        tags["7_tail"] = f"ASSUMED probabilities; {drag_pa*locked*1e4:.0f} bps/yr of capital spread over {t.trades_per_year:.0f} trades/yr"
    else:
        # fewer than one trade a month: the annual drag cannot be amortised
        # per trade in any honest way (it would land on one trade and read
        # as -800 bps). Reported per YEAR instead and left out of the sum.
        out["7_tail"] = None
        tags["7_tail"] = (f"n/a per trade ({t.trades_per_year:.1f} trades/yr); "
                          f"{drag_pa*locked*1e4:.0f} bps/yr of capital, ASSUMED")

    total = sum(v for v in out.values() if v is not None)
    gross = t.band_bps / 2.0                                    # capture half the band
    return {"buckets_bps": {k: (round(v, 3) if (v is not None and math.isfinite(v)) else None)
                            for k, v in out.items()},
            "tail_excluded": out["7_tail"] is None,
            "tags": tags, "gross_bps": round(gross, 3),
            "total_cost_bps": round(total, 3) if math.isfinite(total) else None,
            "net_bps": round(gross - total, 3) if math.isfinite(total) else None,
            "fees_only_net_bps": round(gross - out["1_fees"], 3),
            "measured_share": round(sum(1 for k in tags if tags[k].startswith("MEASURED")
                                        or tags[k].startswith("VERIFIED")) / len(tags), 2)}


def family_specs(size_usd: float, mode: str) -> list[TradeSpec]:
    """Build one spec per recording-family pair from the frozen verdict json
    (band, convergence, depth, funding) — nothing is recomputed here."""
    import premium_verdict as PV
    d = json.loads((ROOT / "research/results/arb_premium_verdict.json").read_text(encoding="utf-8"))
    specs = []
    for pid, p in d["pairs"].items():
        i = p.get("interim") or {}
        if not i:
            continue
        va, vb = PV.VENUE_KEYS.get(pid, ("HL", "lighter"))
        best = max((i.get("sell") or {}), (i.get("buy") or {}),
                   key=lambda s: (s or {}).get("band_bps") or 0)
        lab = "sell" if best is (i.get("sell") or {}) else "buy"
        conv = i.get(f"conv_{lab}") or {}
        dep = i.get(f"depth_{lab}") or {}
        f = i.get("funding") or {}
        days = max(p.get("days") or 1e-9, 1e-9)
        top = float(dep.get("fat_median_notional_usd") or 0)
        specs.append(TradeSpec(
            leg_a=va, leg_b=vb, band_bps=float(best.get("band_bps") or 0),
            size_usd=size_usd, depth_top=top,
            depth_1bps=top * 4.5, depth_3bps=top * 4.5,   # recorder has top only;
            # 4.5x = the median top->3bps ratio measured in TODO 1.00 v5. ASSUMED.
            hold_minutes=float(conv.get("median_minutes") or 60.0),
            fund_diff_bps_8h=float(f.get("median_bps_8h") or 0.0),
            trades_per_year=float(conv.get("episodes") or 0) / days * 365,
            mode=mode,
            notes=[f"side={lab}", "depth buckets ASSUMED from top x4.5"]))
        specs[-1].pid = pid                                       # type: ignore[attr-defined]
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=200.0, help="order notional per leg, USD")
    ap.add_argument("--mode", default="taker_taker", choices=list(FEES.MODES))
    ap.add_argument("--no-rebate", action="store_true")
    a = ap.parse_args()
    print("=" * 100)
    print(f"  §1.06 成本函數——七個桶，每筆來回 bps｜size ${a.size:,.0f}/腿｜{FEES.MODES[a.mode]}"
          f"｜{'不含' if a.no_rebate else '含'}返佣")
    print("=" * 100)
    print(f"  {'配對':<8}{'帶':>6}{'毛':>6} | {'1費':>6}{'2滑':>6}{'3資':>6}{'4持':>6}{'5轉':>6}{'6營':>6}{'7尾':>6} | "
          f"{'合計':>7}{'淨':>7}{'只扣費':>7}{'量到%':>6}")
    res = {"size_usd": a.size, "mode": a.mode, "rebate": not a.no_rebate,
           "assumptions": {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})
                           for k, v in ASSUMED.items()}, "pairs": {}}
    for t in family_specs(a.size, a.mode):
        t.rebate = not a.no_rebate
        r = cost_breakdown(t)
        b = r["buckets_bps"]
        fmt = lambda x: f"{x:>6.1f}" if x is not None else "   inf"
        print(f"  {t.pid:<8}{t.band_bps:>6.1f}{r['gross_bps']:>6.1f} | "
              + "".join(fmt(b[k]) for k in sorted(b))
              + f" | {fmt(r['total_cost_bps']):>7}{fmt(r['net_bps']):>7}{r['fees_only_net_bps']:>7.1f}"
              f"{r['measured_share']*100:>5.0f}%" + ("  (尾未攤)" if r["tail_excluded"] else ""))
        res["pairs"][t.pid] = {"spec": {k: v for k, v in t.__dict__.items() if k != "notes"},
                               **r}
    print("\n  讀法：毛＝帶÷2（進場在帶、回到半帶平倉）。「淨」扣七桶、「只扣費」只扣第 1 桶——"
          "兩者的差就是 fees.py 以前看不見的成本。")
    print("  量到%＝七桶裡有幾桶是量測值；其餘是刻意偏悲觀的假設，換一個真數字就少一個假設。")
    print("  第 2 桶 inf ＝這個 size 超過 3 bps 內的深度，整個帶都會被吃掉——先縮 size 再談別的。")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
