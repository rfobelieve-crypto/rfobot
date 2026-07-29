# -*- coding: utf-8 -*-
"""Funding-carry screen — is delta-neutral BTC funding harvest worth a slot
in the future portfolio? Pure descriptive screen on data already in-repo.

Strategy screened: SHORT perp + LONG spot, continuously held. Positive
funding -> the short collects; negative -> it pays. This screen measures
the carry stream only:

  gross yield        sum of settlement-hour rates (00/08/16 UTC closes)
  cost drag          one-time entry+exit across both legs, amortized
  regime shape       % negative settlements, longest negative streak,
                     worst rolling 30d, monthly breakdown

NOT modeled (stated, not hidden): spot-perp basis PnL (entry/exit basis
moves), margin/liquidation mechanics on the perp leg, borrow limits.
A real deployment needs its own pre-registered gate; this only decides
whether the line earns a research slot at $1k+ scale.

Data: market_data/raw_data/cg_funding_1h.parquet (hourly OHLC of the rate,
2025-10 -> now). Units sanity-printed before any conclusion.

Run: python research/funding_carry_screen.py
Out: research/results/funding_carry_screen.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = ROOT / "market_data/raw_data/cg_funding_1h.parquet"
OUT = ROOT / "research/results/funding_carry_screen.json"

# one-time round-trip cost, both legs (bps of notional):
#   perp taker 5x2 + spot taker 10x2 = 30 conservative retail;
#   maker-ish variant 2x2 + 8x2 = 20
COST_RT_BPS = {"retail_taker": 30.0, "maker_mix": 20.0}


def main() -> int:
    df = pd.read_parquet(SRC)
    rate = df["close"].astype(float)
    rate.index = pd.to_datetime(rate.index, utc=True)

    # settlements: 00/08/16 UTC hourly closes
    st = rate[rate.index.hour.isin((0, 8, 16))]
    span_d = (st.index[-1] - st.index[0]).days
    print(f"settlements n={len(st)}  span {st.index[0].date()} -> "
          f"{st.index[-1].date()}  ({span_d}d)")
    print(f"units sanity: median {st.median():+.4f}  p05 {st.quantile(.05):+.4f}"
          f"  p95 {st.quantile(.95):+.4f}  max|.| {st.abs().max():.4f}")
    print("  -> read as PERCENT per 8h if median is ~0.01-level "
          "(0.01%/8h = 10.95%/yr baseline)")

    # treat values as % per settlement
    ann_gross = st.mean() * 3 * 365          # % per year
    neg_share = (st < 0).mean()
    # longest consecutive negative run
    neg = (st < 0).astype(int).values
    longest = cur = 0
    for v in neg:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    # worst rolling 30d sum (90 settlements)
    roll = st.rolling(90).sum().dropna()
    worst30 = roll.min()
    monthly = st.groupby(st.index.to_period("M")).sum()

    print(f"\ngross carry: {ann_gross:+.2f}%/yr annualized "
          f"(mean {st.mean():+.5f}%/settlement)")
    print(f"negative settlements: {neg_share:.0%}   longest negative streak: "
          f"{longest} settlements ({longest/3:.1f} days)")
    print(f"worst rolling 30d: {worst30:+.3f}% (carry alone)")
    print("\nmonthly carry (%):")
    for pm, v in monthly.items():
        print(f"  {pm}  {v:+.3f}")
    for tag, c in COST_RT_BPS.items():
        be_days = (c / 1e2) / (st.mean() * 3) if st.mean() > 0 else float("inf")
        net = ann_gross - 0  # one-time cost, not annual — report breakeven instead
        print(f"\ncosts[{tag}]: {c:.0f} bps round-trip both legs "
              f"-> breakeven holding ~{be_days:.0f} days; "
              f"1-yr net ≈ {ann_gross - c/1e2:+.2f}%")
    print("\nNOT modeled: spot-perp basis PnL, perp-leg margin mechanics, "
          "capital doubling (spot leg). Screen only.")

    OUT.write_text(json.dumps({
        "n_settlements": int(len(st)), "span_days": int(span_d),
        "ann_gross_pct": float(ann_gross), "neg_share": float(neg_share),
        "longest_neg_streak": int(longest), "worst_roll30_pct": float(worst30),
        "monthly_pct": {str(k): float(v) for k, v in monthly.items()},
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
