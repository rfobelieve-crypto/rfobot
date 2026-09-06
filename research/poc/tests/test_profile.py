# -*- coding: utf-8 -*-
"""Stage 1 gate — synthetic tests for the volume-profile engine.

Run:  python research/poc/tests/test_profile.py     (no pytest needed)

Every test builds its own bars so the expected answer is known by construction.
The engine never touches real data here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile import Bars, build_profile, depth_between  # noqa: E402

MIN = 60_000
T0 = 1_700_000_000_000 // MIN * MIN          # a minute boundary
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def bars_from(rows, t0=T0):
    """rows: list of (open, high, low, close, volume), one per minute."""
    ts = t0 + MIN * np.arange(len(rows))
    a = np.array(rows, dtype=float)
    return Bars(ts, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4])


# ---------------------------------------------------------------- allocation
def test_single_bar_uniform():
    b = bars_from([(100, 110, 100, 105, 100)])
    p = build_profile(b, T0 + MIN, ("time", 24), "uniform", 1.0)
    # half-open range(floor(100/1), ceil(110/1)) = 100..109 -> 10 bins, 10 each
    check("uniform: 10 bins", len(p.bin_ids) == 10, f"got {len(p.bin_ids)}")
    check("uniform: 10 per bin", np.allclose(p.volumes, 10.0), f"{p.volumes}")
    check("uniform: total preserved", abs(p.total_volume - 100) < 1e-9)


def test_single_bar_close():
    b = bars_from([(100, 110, 100, 105, 100)])
    p = build_profile(b, T0 + MIN, ("time", 24), "close", 1.0)
    check("close: one bin", len(p.bin_ids) == 1, f"got {len(p.bin_ids)}")
    check("close: bin is floor(close)", p.bin_ids[0] == 105, f"{p.bin_ids}")
    check("close: all volume there", abs(p.volumes[0] - 100) < 1e-9)


def test_degenerate_bar():
    """low == high must still touch exactly one bin, not zero.

    The price has to sit EXACTLY on a bin edge for this to bite: with
    low=high=100.4 and bin=1, floor=100 and ceil=101 already give n=1, so the
    max(..., 1) guard is never reached and the test proves nothing.  The
    mutation harness caught that (2026-09-06); use 100.0.
    """
    b = bars_from([(100.0, 100.0, 100.0, 100.0, 7)])
    p = build_profile(b, T0 + MIN, ("time", 24), "uniform", 1.0)
    check("degenerate bar -> 1 bin", p is not None and len(p.bin_ids) == 1,
          f"{None if p is None else p.bin_ids}")
    check("degenerate bar volume kept", p is not None and abs(p.total_volume - 7) < 1e-9)
    # and the off-edge case must still be one bin
    b2 = bars_from([(100.4, 100.4, 100.4, 100.4, 7)])
    p2 = build_profile(b2, T0 + MIN, ("time", 24), "uniform", 1.0)
    check("degenerate bar off-edge -> 1 bin", p2 is not None and len(p2.bin_ids) == 1)


def test_narrow_bar_smaller_than_bin():
    b = bars_from([(100.1, 100.3, 100.1, 100.2, 5)])
    p = build_profile(b, T0 + MIN, ("time", 24), "uniform", 1.0)
    check("sub-bin bar -> 1 bin", len(p.bin_ids) == 1, f"{p.bin_ids}")


# ---------------------------------------------------------------------- POC
def test_poc_unique():
    b = bars_from([(100, 101, 100, 100.5, 1),
                   (100, 101, 100, 100.5, 1),
                   (200, 201, 200, 200.5, 5)])
    p = build_profile(b, T0 + 3 * MIN, ("time", 24), "uniform", 1.0)
    check("poc = the heavy bin", abs(p.poc - 200.5) < 1e-9, f"poc={p.poc}")


def test_poc_tie_takes_median_price():
    """Three tied bins at 100/200/300 -> POC must be the middle one."""
    b = bars_from([(100, 101, 100, 100.5, 9),
                   (200, 201, 200, 200.5, 9),
                   (300, 301, 300, 300.5, 9)])
    p = build_profile(b, T0 + 3 * MIN, ("time", 24), "uniform", 1.0)
    check("poc tie -> median price", abs(p.poc - 200.5) < 1e-9, f"poc={p.poc}")


def test_poc_tie_even_count():
    b = bars_from([(100, 101, 100, 100.5, 9), (200, 201, 200, 200.5, 9)])
    p = build_profile(b, T0 + 2 * MIN, ("time", 24), "uniform", 1.0)
    check("poc tie (even) -> midpoint", abs(p.poc - 150.5) < 1e-9, f"poc={p.poc}")


# ------------------------------------------------------------------- window
def test_lookahead_excludes_the_reference_bar():
    """A bar that has not closed at t_ref must not enter the profile."""
    b = bars_from([(100, 101, 100, 100.5, 1), (500, 501, 500, 500.5, 99)])
    # t_ref = close time of bar 0 -> only bar 0 is closed
    p = build_profile(b, T0 + MIN, ("time", 24), "uniform", 1.0)
    check("t_ref excludes the un-closed bar", abs(p.total_volume - 1) < 1e-9,
          f"total={p.total_volume}")
    check("t_ref excludes: last_ms is bar 0", p.last_ms == T0)


def test_time_window_left_edge():
    """('time', 1) keeps exactly the 60 bars in [t_ref-1h, t_ref)."""
    rows = [(100, 101, 100, 100.5, 1)] * 200
    b = bars_from(rows)
    t_ref = T0 + 200 * MIN
    p = build_profile(b, t_ref, ("time", 1), "uniform", 1.0)
    check("time window = 60 bars", p.n_bars == 60, f"n={p.n_bars}")
    check("time window left edge", p.first_ms == t_ref - 60 * MIN, f"{p.first_ms}")


def test_volume_window_hits_target():
    rows = [(100, 101, 100, 100.5, 2.0)] * 500
    b = bars_from(rows)
    t_ref = T0 + 500 * MIN
    adv = 100.0                     # pretend 30d average daily volume
    p = build_profile(b, t_ref, ("volume", 0.5), "uniform", 1.0,
                      avg_daily_volume=adv)
    target = 0.5 * adv              # = 50 -> 25 bars of 2.0
    check("volume window within +-5%",
          target * 0.95 <= p.total_volume <= target * 1.05,
          f"got {p.total_volume} target {target}")
    check("volume window bar count", p.n_bars == 25, f"n={p.n_bars}")


def test_volume_window_insufficient_history():
    b = bars_from([(100, 101, 100, 100.5, 1.0)] * 5)
    p = build_profile(b, T0 + 5 * MIN, ("volume", 0.5), "uniform", 1.0,
                      avg_daily_volume=1e6)
    check("volume window: too little history -> None", p is None)


# -------------------------------------------------------------- other fields
def test_vwap_is_unbinned():
    """VWAP must not move when the bin size changes; POC generally does."""
    rng = np.random.default_rng(0)
    n = 400
    px = 100 + np.cumsum(rng.normal(0, 0.3, n))
    rows = [(p, p + 0.4, p - 0.4, p + rng.normal(0, .1), abs(rng.normal(10, 3)))
            for p in px]
    b = bars_from(rows)
    t = T0 + n * MIN
    a = build_profile(b, t, ("time", 24), "uniform", 0.2)
    c = build_profile(b, t, ("time", 24), "uniform", 2.0)
    check("vwap invariant to bin size", abs(a.vwap - c.vwap) < 1e-9,
          f"{a.vwap} vs {c.vwap}")
    check("total volume invariant to bin size",
          abs(a.total_volume - c.total_volume) < 1e-6)


def test_hvn_lvn_and_next_hvn():
    b = bars_from([(100, 101, 100, 100.5, 50)] * 10
                  + [(120, 121, 120, 120.5, 50)] * 10
                  + [(110, 111, 110, 110.5, 1)] * 10)
    p = build_profile(b, T0 + 30 * MIN, ("time", 24), "uniform", 1.0)
    check("hvn non-empty", len(p.hvn) > 0)
    check("lvn holds the thin bin", any(abs(x - 110.5) < 1e-9 for x in p.lvn),
          f"lvn={p.lvn}")
    below = p.next_hvn(+1)
    above = p.next_hvn(-1)
    check("next_hvn(+1) is below the poc", below is None or below < p.poc)
    check("next_hvn(-1) is above the poc", above is None or above > p.poc)


def test_depth_between():
    b = bars_from([(100, 101, 100, 100.5, 30)] * 1
                  + [(110, 111, 110, 110.5, 40)] * 1
                  + [(120, 121, 120, 120.5, 30)] * 1)
    p = build_profile(b, T0 + 3 * MIN, ("time", 24), "uniform", 1.0)
    d = depth_between(p, 100.5, 120.5)
    check("depth strictly between endpoints", abs(d - 0.4) < 1e-9, f"got {d}")


def test_nan_bars_are_dropped_not_fatal():
    b = bars_from([(100, 101, 100, 100.5, 10),
                   (np.nan, np.nan, np.nan, np.nan, np.nan),
                   (100, 101, 100, 100.5, 10)])
    p = build_profile(b, T0 + 3 * MIN, ("time", 24), "uniform", 1.0)
    check("NaN bar dropped", p is not None and p.n_bars == 2, f"{p and p.n_bars}")


def test_empty_window_returns_none():
    b = bars_from([(100, 101, 100, 100.5, 10)])
    check("t_ref before all data -> None",
          build_profile(b, T0 - 10 * MIN, ("time", 24), "uniform", 1.0) is None)
    b2 = bars_from([(100, 101, 100, 100.5, 0.0)] * 5)
    check("zero-volume window -> None",
          build_profile(b2, T0 + 5 * MIN, ("time", 24), "uniform", 1.0) is None)


def main():
    for f in sorted([v for k, v in globals().items() if k.startswith("test_")],
                    key=lambda f: f.__code__.co_firstlineno):
        print(f.__name__)
        f()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("Stage 1 synthetic gate: ALL PASS")


if __name__ == "__main__":
    main()
