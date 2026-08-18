"""M4 prep — read-only feasibility audit for the SF execution path.

Answers, with real numbers and zero orders (2026-08-18):
  1. does OKX actually list a USDT perp for each core9 coin?
  2. at risk 0.15%/trade and the frozen 3.5-ATR stop, what notional does
     each coin's trade carry, and does it clear OKX's minSz/lotSz without
     stupid rounding error?
  3. what does 5-concurrent worst-case look like against the 2x-equity
     notional cap?

Sizing algebra: risk_usd = equity x 0.0015; stop_frac = 3.5 x ATR14/px
(per-coin, from the live 1h caches); notional = risk_usd / stop_frac;
contracts = notional / (ctVal x px), rounded DOWN to lotSz.  A coin is
infeasible if rounded contracts < minSz, and poorly sized if the lot
rounding moves realized risk by more than 20%.

Run at both the $274 baseline and the current (halted) equity so the
answer does not silently assume the CAP-2 question away.
Read-only: public instruments endpoint, no keys, no orders.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.survival_cards import CACHE, CORE9, SC  # noqa: E402

RISK_PCT = 0.15          # frozen minimum tier (limits.py sweep)
DIS = 3.5                # frozen disaster stop
MAX_CONC = 5             # frozen minimum tier concurrency
NOTIONAL_CAP_MULT = 2.0  # portfolio net-notional cap


def okx_swap_specs() -> dict[str, dict]:
    req = urllib.request.Request(
        "https://www.okx.com/api/v5/public/instruments?instType=SWAP",
        headers={"User-Agent": "sf-m4-audit/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())["data"]
    out = {}
    for d in data:
        if d.get("settleCcy") == "USDT" and d.get("state") == "live":
            out[d["instId"]] = {
                "ctVal": float(d["ctVal"]),
                "lotSz": float(d["lotSz"]),
                "minSz": float(d["minSz"]),
                "tickSz": float(d["tickSz"]),
            }
    return out


def atr_pct(sym: str) -> tuple[float, float]:
    bars = SC.load_csv(str(CACHE / f"{sym}USDT_1h.csv"))
    a = SC.atr14(bars)
    px = bars[-1][SC.C]
    # median ATR over the last 30d — sizing should not key off one quiet hour
    import statistics
    window = [x for x in a[-720:] if x]
    return px, statistics.median(window) / px


def main() -> None:
    specs = okx_swap_specs()
    for label, equity in (("baseline $274", 274.0), ("current $777", 777.0)):
        risk_usd = equity * RISK_PCT / 100
        print(f"\n════ equity {label} → risk/trade ${risk_usd:.2f} ════")
        print(f"{'sym':<6}{'inst':<16}{'px':>10}{'ATR%':>7}{'stop%':>7}"
              f"{'notional':>10}{'contracts':>11}{'feasible':>9}{'risk err':>9}")
        total_notional = 0.0
        worst = []
        for sym in CORE9:
            inst = f"{sym}-USDT-SWAP"
            spec = specs.get(inst)
            px, ap = atr_pct(sym)
            stop_frac = DIS * ap
            notional = risk_usd / stop_frac
            if spec is None:
                print(f"{sym:<6}{inst:<16}{'—':>10}{100*ap:>6.2f}%"
                      f"{100*stop_frac:>6.2f}%{notional:>10.2f}"
                      f"{'NO PERP':>11}{'✗':>9}{'—':>9}")
                continue
            raw_ct = notional / (spec["ctVal"] * px)
            lots = int(raw_ct / spec["lotSz"]) * spec["lotSz"]
            feasible = lots >= spec["minSz"] and lots > 0
            realized = lots * spec["ctVal"] * px
            err = (realized - notional) / notional if notional else 0.0
            ok = feasible and abs(err) <= 0.20
            total_notional += realized if feasible else 0.0
            worst.append((sym, realized if feasible else 0.0))
            print(f"{sym:<6}{inst:<16}{px:>10.4g}{100*ap:>6.2f}%"
                  f"{100*stop_frac:>6.2f}%{notional:>10.2f}"
                  f"{lots:>11.4g}{('✓' if ok else '✗'):>9}"
                  f"{100*err:>8.1f}%")
        worst.sort(key=lambda x: -x[1])
        top5 = sum(v for _, v in worst[:MAX_CONC])
        cap = NOTIONAL_CAP_MULT * equity
        print(f"  5-concurrent worst-case notional ${top5:.0f} vs "
              f"{NOTIONAL_CAP_MULT}x-equity cap ${cap:.0f} "
              f"({'inside' if top5 <= cap else 'OVER'} cap, "
              f"{100*top5/cap:.0f}%)")


if __name__ == "__main__":
    main()
