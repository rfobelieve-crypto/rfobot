"""Tests for the 1s depth event collector (F7, TODO §0.46).

The 1m collector's blind spots are the whole reason F7 exists, so the tests
pin exactly the two structures it must preserve: price banding and
cancel→re-add matching.  All through `process()` — no WS, no DB.
"""
from __future__ import annotations

import pytest

from market_data.adapters.depth_events_1s import (
    MIN_BOOK_LEVELS,
    NEAR_BPS,
    REATTACH_MS,
    DepthEvents1sCollector,
    band_of,
)

MID = 60_000.0


# Comfortably past MIN_BOOK_LEVELS so cancelling a level in a test does not
# drop the book back into warm-up (which silently un-counts everything after).
_DEPTH = MIN_BOOK_LEVELS + 10


def _warm_collector() -> DepthEvents1sCollector:
    """Collector with a populated, uncrossed book (past warm-up)."""
    c = DepthEvents1sCollector()
    bids = [[str(MID - 10 - i * 10), "1.0"] for i in range(_DEPTH)]
    asks = [[str(MID + 10 + i * 10), "1.0"] for i in range(_DEPTH)]
    c.process({"b": bids, "a": asks}, ts_ms=1_000_000)
    return c


def _bucket(c: DepthEvents1sCollector, sec_ms: int) -> dict:
    return c._buckets[sec_ms]


class TestBanding:

    def test_band_edges(self):
        assert band_of(MID * (1 + 0.0001), MID) == "near"     # 1 bp
        assert band_of(MID * (1 + 0.001), MID) == "mid"       # 10 bp
        assert band_of(MID * (1 + 0.01), MID) == "far"        # 100 bp

    def test_cancel_at_touch_lands_near_cancel_far_wall_lands_far(self):
        c = _warm_collector()
        # touch bid sits 10 ticks under mid (~1.7bp = near); deepest ~5%
        c.process({"b": [[str(MID - 10), "0.0"],
                         [str(MID - 10 - (_DEPTH - 1) * 10), "0.0"]],
                   "a": []}, ts_ms=2_000_000)
        b = _bucket(c, 2_000_000)
        assert b["bid_cancel_near"] == pytest.approx(1.0)
        assert b["bid_cancel_far"] == pytest.approx(1.0)
        assert b["bid_cancel_mid"] == 0.0

    def test_add_at_new_price_banded_against_pre_update_mid(self):
        """An add at a price not previously in the book is a pure insert
        (d = full qty) and its band comes from the book BEFORE the update —
        the state the actor acted on."""
        c = _warm_collector()
        px = MID * 1.002 + 5          # ~20.9bp above mid, off the 10-tick grid
        c.process({"b": [], "a": [[str(px), "2.0"]]}, ts_ms=3_000_000)
        assert _bucket(c, 3_000_000)["ask_add_mid"] == pytest.approx(2.0)


class TestReAddMatching:

    def test_cancel_then_readd_within_window_is_matched(self):
        c = _warm_collector()
        px = str(MID - 10)
        c.process({"b": [[px, "0.0"]], "a": []}, ts_ms=2_000_000)
        c.process({"b": [[px, "1.0"]], "a": []}, ts_ms=2_000_000 + 3_000)
        b = _bucket(c, 2_003_000)
        assert b["bid_readd"] == pytest.approx(1.0)
        # the add is STILL counted in its band — collector never nets
        assert b["bid_add_near"] == pytest.approx(1.0)

    def test_readd_after_window_is_true_withdrawal(self):
        c = _warm_collector()
        px = str(MID - 10)
        c.process({"b": [[px, "0.0"]], "a": []}, ts_ms=2_000_000)
        late = 2_000_000 + REATTACH_MS + 1_000
        c.process({"b": [[px, "1.0"]], "a": []}, ts_ms=late)
        assert _bucket(c, late)["bid_readd"] == 0.0

    def test_partial_readd_matches_only_pending_qty(self):
        c = _warm_collector()
        px = str(MID - 10)
        c.process({"b": [[px, "0.0"]], "a": []}, ts_ms=2_000_000)   # -1.0
        c.process({"b": [[px, "5.0"]], "a": []}, ts_ms=2_001_000)   # +5.0
        b = _bucket(c, 2_001_000)
        assert b["bid_readd"] == pytest.approx(1.0)   # only the cancelled 1.0
        assert b["bid_add_near"] == pytest.approx(5.0)

    def test_readd_is_side_scoped(self):
        """A bid cancel must never be matched by an ask add at the same px."""
        c = _warm_collector()
        px = MID - 10
        c.process({"b": [[str(px), "0.0"]], "a": []}, ts_ms=2_000_000)
        c.process({"b": [], "a": [[str(px), "1.0"]]}, ts_ms=2_001_000)
        b = _bucket(c, 2_001_000)
        assert b["ask_readd"] == 0.0 and b["bid_readd"] == 0.0


class TestWarmup:

    def test_thin_book_counts_nothing(self):
        c = DepthEvents1sCollector()
        c.process({"b": [[str(MID - 10), "1.0"]],
                   "a": [[str(MID + 10), "1.0"]]}, ts_ms=1_000_000)
        b = _bucket(c, 1_000_000)
        assert all(b[k] == 0.0 for k in b if k not in ("n", "mid"))
        assert c._warmup_skipped > 0

    def test_crossed_book_resets_and_rewarns(self):
        c = _warm_collector()
        # cross the book: bid above best ask
        c.process({"b": [[str(MID + 500), "1.0"]], "a": []}, ts_ms=2_000_000)
        # next message sees an empty (reset) book → warm-up, nothing counted
        c.process({"b": [[str(MID - 10), "0.5"]], "a": []}, ts_ms=2_001_000)
        assert _bucket(c, 2_001_000)["bid_add_near"] == 0.0
        assert len(c._book["bid"]) <= 2


class TestBuckets:

    def test_seconds_are_separate_buckets(self):
        c = _warm_collector()
        c.process({"b": [[str(MID - 10), "0.0"]], "a": []}, ts_ms=2_000_100)
        c.process({"b": [[str(MID - 20), "0.0"]], "a": []}, ts_ms=2_001_900)
        assert _bucket(c, 2_000_000)["bid_cancel_near"] == pytest.approx(1.0)
        assert _bucket(c, 2_001_000)["bid_cancel_near"] == pytest.approx(1.0)

    def test_bucket_records_mid(self):
        c = _warm_collector()
        c.process({"b": [[str(MID - 10), "0.5"]], "a": []}, ts_ms=2_000_000)
        assert _bucket(c, 2_000_000)["mid"] == pytest.approx(MID)
