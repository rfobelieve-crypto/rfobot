# -*- coding: utf-8 -*-
"""FROZEN combo watchlist — registered 2026-08-02 from the systematic
screen (research/sweep_raid_combo_screen.py: 287 cells, 33 eligible under
the pre-stated rule, 8 nominees after trade-set de-dup).

These are the combos the shadow scores on FORWARD rows (`shadow_engine.py
--combos`), same day-clustered-CI arithmetic as Gate F. The in-sample
stats below are the ENTRY TICKET, not evidence — promotion of any entry
to a tradeable rule requires its own forward record plus the October
pre-registration. Definitions may not be edited; a change = a new
registration with a new date (the variant-D lesson: frozen definitions
are what make being wrong informative).

In-sample at registration (BTC+ETH ~100d, netR / t / n):
  R∧V     +0.165 t5.5 n390   R∧Q     +0.199 t5.4 n202
  R∧V∧Q   +0.267 t5.2 n112   R∧快    +0.115 t4.7 n531
  R∧快∧Q  +0.202 t4.5 n125   R       +0.085 t4.5 n926
  PA      +0.104 t4.0 n530   V∧LIQ   +0.168 t4.0 n263

Flags map 1:1 to prospectively recorded shadow-log columns:
  R    flow_reject == 1        (raid hour closed back inside)
  V    flow_vhigh == 1         (vshock >= own-symbol causal median)
  FAST flow_att_min <= 5       (blitz attack)
  Q    drv_q == 1              (BTC: raid-hour OI down + taker with break)
  PA   v7_align > 0            (BTC: V7 sides with the fade)
  LIQ  drv_liqburst >= causal median of BTC's earlier raids (>=5 priors)
BTC-scoped flags simply never fire on other symbols (blank columns).
"""
from __future__ import annotations

from statistics import median

REGISTERED = "2026-08-02"

WATCHLIST = {
    "R∧V": ("R", "V"),
    "R∧Q": ("R", "Q"),
    "R∧V∧Q": ("R", "V", "Q"),
    "R∧快": ("R", "FAST"),
    "R∧快∧Q": ("R", "FAST", "Q"),
    "R": ("R",),
    "PA": ("PA",),
    "V∧LIQ": ("V", "LIQ"),
}


def _liq_hi_keys(log: dict) -> set:
    rows = [(int(r["fill_ts"]), float(r["drv_liqburst"]),
             (r["symbol"], r.get("level_kind", "swing"), int(r["fill_ts"])))
            for r in log.values()
            if r["symbol"] == "BTC"
            and r.get("drv_liqburst") not in (None, "", "na")]
    rows.sort()
    out = set()
    for i, (fts, lb, key) in enumerate(rows):
        prior = [v for (ft2, v, _k) in rows[:i] if ft2 < fts]
        if len(prior) >= 5 and lb >= median(prior):
            out.add(key)
    return out


def flag_tests(log: dict) -> dict:
    """flag -> row-predicate, built once per log (LIQ needs global state)."""
    liq = _liq_hi_keys(log)

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "R": lambda r: str(r.get("flow_reject", "")) == "1",
        "V": lambda r: str(r.get("flow_vhigh", "")) == "1",
        "FAST": lambda r: (_num(r.get("flow_att_min")) is not None
                           and _num(r.get("flow_att_min")) <= 5),
        "Q": lambda r: str(r.get("drv_q", "")) == "1",
        "PA": lambda r: (_num(r.get("v7_align")) is not None
                         and _num(r.get("v7_align")) > 0),
        "LIQ": lambda r: (r["symbol"], r.get("level_kind", "swing"),
                          int(r["fill_ts"])) in liq,
    }


def combo_preds(log: dict) -> dict:
    """combo name -> row-predicate over shadow-log rows."""
    tests = flag_tests(log)
    return {name: (lambda r, fs=flags: all(tests[f](r) for f in fs))
            for name, flags in WATCHLIST.items()}
