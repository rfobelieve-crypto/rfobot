"""
Tests for indicator/signal_tracker.py — signal recording and outcome backfill.

All tests mock the DB connection to run without database access.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call

import pytest


# ---------------------------------------------------------------------------
# record_signal validation
# ---------------------------------------------------------------------------

class TestRecordSignal:
    """Tests for record_signal() input validation and DB writes."""

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_valid_up_strong_records(self, mock_conn_fn, mock_ensure):
        """Valid UP/Strong signal should execute an INSERT."""
        from indicator.signal_tracker import record_signal

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        record_signal(
            signal_time=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
            direction="UP",
            strength="Strong",
            p_up=0.65,
            mag_pred=0.003,
            confidence=85.0,
            entry_price=67_000.0,
            regime="TRENDING_BULL",
        )

        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_valid_down_moderate_records(self, mock_conn_fn, mock_ensure):
        """Valid DOWN/Moderate signal should execute an INSERT."""
        from indicator.signal_tracker import record_signal

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        record_signal(
            signal_time=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
            direction="DOWN",
            strength="Moderate",
            p_up=0.35,
            mag_pred=0.002,
            confidence=70.0,
            entry_price=66_000.0,
        )

        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()

    @pytest.mark.parametrize("direction", ["NEUTRAL", "up", "LEFT", "", None])
    def test_invalid_direction_does_nothing(self, direction):
        """Invalid direction should return early without DB access."""
        from indicator.signal_tracker import record_signal

        # If it tries to access DB, it will fail since we didn't mock it
        # That's the test: it should return before touching DB
        with patch("indicator.signal_tracker._get_db_conn") as mock_conn:
            record_signal(
                signal_time=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
                direction=direction,
                strength="Strong",
                p_up=0.5,
                mag_pred=0.001,
                confidence=80.0,
                entry_price=67_000.0,
            )
            mock_conn.assert_not_called()

    @pytest.mark.parametrize("strength", ["Weak", "strong", "STRONG", "", None])
    def test_invalid_strength_does_nothing(self, strength):
        """Invalid strength should return early without DB access."""
        from indicator.signal_tracker import record_signal

        with patch("indicator.signal_tracker._get_db_conn") as mock_conn:
            record_signal(
                signal_time=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
                direction="UP",
                strength=strength,
                p_up=0.5,
                mag_pred=0.001,
                confidence=80.0,
                entry_price=67_000.0,
            )
            mock_conn.assert_not_called()


# ---------------------------------------------------------------------------
# TWAP calculation and correctness
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    """Tests for backfill_outcomes() TWAP calculation and correct flag."""

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_twap_calculation(self, mock_conn_fn, mock_ensure):
        """TWAP = mean(close[t+1..t+4]) / entry - 1."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        sig_time = datetime(2026, 4, 15, 8, 0)

        # First query: unfilled signals
        unfilled_row = {
            "id": 1,
            "signal_time": sig_time,
            "direction": "UP",
            "entry_price": 67_000.0,
            "strength": "Strong",
        }

        # Second query: path closes at +1h, +2h, +3h, +4h
        path_rows = [
            {"close": 67_100.0},
            {"close": 67_200.0},
            {"close": 67_300.0},
            {"close": 67_400.0},
        ]

        cursor.fetchall.side_effect = [[unfilled_row], path_rows]

        backfill_outcomes()

        # Verify the UPDATE was called with correct TWAP
        expected_twap = (67_100 + 67_200 + 67_300 + 67_400) / 4
        expected_ret = (expected_twap - 67_000) / 67_000
        expected_exit = 67_400.0

        update_call = cursor.execute.call_args_list[-1]
        args = update_call[0][1]  # positional args tuple
        assert args[0] == pytest.approx(expected_exit)
        assert args[1] == pytest.approx(expected_ret, rel=1e-6)
        assert args[2] == 1  # correct=1 (UP + positive return)
        assert args[3] == 1  # id

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_correct_flag_up_positive(self, mock_conn_fn, mock_ensure):
        """UP direction + positive TWAP return should set correct=1."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        unfilled = {
            "id": 1,
            "signal_time": datetime(2026, 4, 15, 8, 0),
            "direction": "UP",
            "entry_price": 67_000.0,
            "strength": "Strong",
        }
        # All closes above entry -> positive TWAP -> correct for UP
        path = [{"close": 67_500.0}] * 4
        cursor.fetchall.side_effect = [[unfilled], path]

        backfill_outcomes()

        update_args = cursor.execute.call_args_list[-1][0][1]
        assert update_args[2] == 1  # correct

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_correct_flag_down_negative(self, mock_conn_fn, mock_ensure):
        """DOWN direction + negative TWAP return should set correct=1."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        unfilled = {
            "id": 2,
            "signal_time": datetime(2026, 4, 15, 8, 0),
            "direction": "DOWN",
            "entry_price": 67_000.0,
            "strength": "Moderate",
        }
        # All closes below entry -> negative TWAP -> correct for DOWN
        path = [{"close": 66_500.0}] * 4
        cursor.fetchall.side_effect = [[unfilled], path]

        backfill_outcomes()

        update_args = cursor.execute.call_args_list[-1][0][1]
        assert update_args[2] == 1  # correct

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_correct_flag_up_wrong(self, mock_conn_fn, mock_ensure):
        """UP direction + negative TWAP return should set correct=0."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        unfilled = {
            "id": 3,
            "signal_time": datetime(2026, 4, 15, 8, 0),
            "direction": "UP",
            "entry_price": 67_000.0,
            "strength": "Strong",
        }
        # All closes below entry -> negative TWAP -> wrong for UP
        path = [{"close": 66_500.0}] * 4
        cursor.fetchall.side_effect = [[unfilled], path]

        backfill_outcomes()

        update_args = cursor.execute.call_args_list[-1][0][1]
        assert update_args[2] == 0  # correct=0

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_insufficient_bars_skipped(self, mock_conn_fn, mock_ensure):
        """Signals with fewer than 4 outcome bars should be skipped."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        unfilled = {
            "id": 4,
            "signal_time": datetime(2026, 4, 15, 8, 0),
            "direction": "UP",
            "entry_price": 67_000.0,
            "strength": "Strong",
        }
        # Only 2 bars available (need 4)
        path = [{"close": 67_100.0}, {"close": 67_200.0}]
        cursor.fetchall.side_effect = [[unfilled], path]

        backfill_outcomes()

        # Should only have 2 execute calls (SELECT unfilled + SELECT path)
        # No UPDATE should happen
        update_calls = [
            c for c in cursor.execute.call_args_list
            if "UPDATE" in str(c)
        ]
        assert len(update_calls) == 0

    @patch("indicator.signal_tracker._ensure_table")
    @patch("indicator.signal_tracker._get_db_conn")
    def test_no_unfilled_signals_noop(self, mock_conn_fn, mock_ensure):
        """When there are no unfilled signals, no updates should happen."""
        from indicator.signal_tracker import backfill_outcomes

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor
        mock_conn_fn.return_value = conn

        cursor.fetchall.return_value = []  # no unfilled signals

        backfill_outcomes()

        # Only the initial SELECT should have been called
        assert cursor.execute.call_count == 1


# ---------------------------------------------------------------------------
# Backward compat alias
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Test the record_strong_signal backward-compat alias."""

    @patch("indicator.signal_tracker.record_signal")
    def test_alias_passes_strong(self, mock_record):
        """record_strong_signal should call record_signal with strength='Strong'."""
        from indicator.signal_tracker import record_strong_signal

        record_strong_signal(
            signal_time=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
            direction="UP",
            p_up=0.65,
            mag_pred=0.003,
            confidence=85.0,
            entry_price=67_000.0,
        )

        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        # It's called with positional args in the source
        args = mock_record.call_args[0]
        assert args[2] == "Strong"  # strength parameter
