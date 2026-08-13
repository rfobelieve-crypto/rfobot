"""Tests for the sample-window floor shared by the drift guards.

`sample_floor` decides which rows a live-performance query is allowed to see.
Every way it can be wrong widens the window and lets two definitions of a
column average together — which is the shape of both defects it was written
for (2026-08-08 level drift hiding behind rank metrics, 2026-08-13 confidence
denominator change).  So the properties worth pinning are the ones that keep
the window from silently growing.
"""
from __future__ import annotations

import pytest

from indicator.model_version import (
    CONFIDENCE_EPOCH,
    DECODE_EPOCH,
    get_current_model_deploy_date,
    sample_floor,
)


class TestSampleFloor:

    def test_never_earlier_than_model_deploy(self):
        """An epoch older than the deploy date must not drag the window back."""
        assert sample_floor("2020-01-01") >= get_current_model_deploy_date()

    def test_takes_the_latest_epoch(self):
        assert sample_floor("2020-01-01", "2030-06-05 12:00:00") \
            .startswith("2030-06-05 12:00:00")

    def test_no_epoch_still_floors_at_deploy(self):
        assert sample_floor().startswith(get_current_model_deploy_date())

    def test_bare_date_reads_as_midnight_not_end_of_day(self):
        """A same-day mid-day epoch must win over the bare date, not tie."""
        assert sample_floor("2030-06-05", "2030-06-05 09:00:00") \
            .endswith("09:00:00")

    def test_unparseable_epoch_raises_rather_than_widening(self):
        """Silently dropping a bad epoch would re-admit the rows it excludes.

        Fail-open is only acceptable when the degraded state is visible; here
        the degraded state looks exactly like a healthy wide sample.
        """
        with pytest.raises(ValueError):
            sample_floor("last tuesday")

    def test_output_is_a_datetime_mysql_will_compare(self):
        from datetime import datetime
        datetime.strptime(sample_floor(DECODE_EPOCH), "%Y-%m-%d %H:%M:%S")


class TestEpochsAreOrdered:
    """The two epochs describe real deploys; keep them honest."""

    def test_decode_epoch_precedes_confidence_epoch(self):
        assert DECODE_EPOCH < CONFIDENCE_EPOCH, (
            "the decode fix shipped before the confidence fix; if this flips, "
            "one of the constants was edited without checking the history")

    def test_epochs_are_parseable_by_the_floor_helper(self):
        for ep in (DECODE_EPOCH, CONFIDENCE_EPOCH):
            sample_floor(ep)   # raises if a constant is malformed
