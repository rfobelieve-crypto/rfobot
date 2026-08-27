# -*- coding: utf-8 -*-
"""Guard: the fast first_hit must equal the reference, on known answers.

Written 2026-08-27 after `first_hits_batch` — a heap-based rewrite of
`liquidity_map_check.first_hit`, added purely for speed after two runs were
killed inside the O(levels x bars) original — disagreed with it on 90 of
600 sampled levels. The failure was silent in the worst way: levels whose
ESTABLISHMENT BAR already traded through them were pushed onto the heap at
that bar, popped there, failed the `j > est` guard, and vanished unmarked.
They then read as "never swept, still resting" forever, which inflates
every downstream measure that asks what liquidity is still live.

Nothing about the output looked wrong. It was caught only because the
number it produced disagreed with an earlier run of the same test.

So this pins it with SYNTHETIC bars whose answers are known by
construction, rather than by sampling real data — the discipline from
mistake.md 2026-07-29: a new instrument gets checked against a case whose
answer is already known before it is trusted anywhere.

The reference implementation is the definition. If these ever disagree,
the FAST one is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

from research.confluence_all_families import first_hits_batch   # noqa: E402
from research.liquidity_map_check import first_hit              # noqa: E402


def bar(ts, o, h, l, c, v=1.0):
    """(ts, open, high, low, close, volume) — sweep_core's O,H,L,C,V = 1..5."""
    return (ts, o, h, l, c, v)


# A tape built so every interesting case is present and hand-checkable.
#   bar 0-2  quiet around 100
#   bar 3    spikes to 110  <- an establishment bar that already crosses 105
#   bar 4    reaches 106    <- the bar that broke the fast version
#   bar 5-6  quiet
#   bar 7    dips to 90
#   bar 8    back to 100
BARS = [
    bar(0, 100, 101,  99, 100),
    bar(1, 100, 102,  98, 100),
    bar(2, 100, 101,  99, 100),
    bar(3, 100, 110,  99, 105),
    bar(4, 105, 106, 104, 105),
    bar(5, 105, 105, 103, 104),
    bar(6, 104, 104, 102, 103),
    bar(7, 103, 103,  90,  95),
    bar(8,  95, 100,  94,  99),
]


@pytest.mark.parametrize("level,why", [
    ((3, 105.0, 1), "buy level whose OWN bar already exceeds it — the case "
                    "that broke the fast version; the reference scans from "
                    "est+1, so bar 4 (high 106) is the answer, never None"),
    ((0, 101.5, 1), "crossed a couple of bars later"),
    ((0, 999.0, 1), "never crossed — must stay None"),
    ((3,  99.0, -1), "sell level whose own bar already dips below it"),
    ((0,  95.0, -1), "sell level crossed much later, at bar 7"),
    ((0,  50.0, -1), "sell level never crossed"),
    ((8, 100.0, 1), "established on the last bar — nothing after it"),
])
def test_matches_reference_on_known_cases(level, why):
    est, price, side = level
    assert first_hits_batch(BARS, [level])[0] == \
        first_hit(BARS, est, price, side), why


def test_the_establishment_bar_never_counts():
    """The specific regression: est's own bar must not mark the level.

    Level 105 is established at bar 3, whose high is 110. The reference
    scans range(est+1, n), so bar 3 cannot be the answer — bar 4 is. The
    broken version returned None here.
    """
    got = first_hits_batch(BARS, [(3, 105.0, 1)])[0]
    assert got is not None, "regression: level vanished, read as never swept"
    assert got == 4, f"expected bar 4, got {got}"


def test_batch_equals_reference_over_many_levels():
    """Whole-inventory equality, including levels that interleave.

    A per-level check can pass while the heap ordering is wrong across
    levels — the bug class this rewrite introduces is retiring the WRONG
    pending level when several are eligible on the same bar.
    """
    levels = [(e, p, s)
              for e in range(0, 8)
              for p in (90.0, 95.0, 99.5, 101.0, 103.5, 105.0, 109.0)
              for s in (1, -1)]
    fast = first_hits_batch(BARS, levels)
    ref = [first_hit(BARS, e, p, s) for e, p, s in levels]
    bad = [(levels[i], ref[i], fast[i])
           for i in range(len(levels)) if ref[i] != fast[i]]
    assert not bad, f"{len(bad)}/{len(levels)} disagree, e.g. {bad[:3]}"
