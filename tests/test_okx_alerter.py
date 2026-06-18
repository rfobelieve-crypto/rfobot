"""Unit tests for indicator/okx/alerter.py.

Confirms the Telegram critical channel sender behaves correctly: respects
config gates, swallows network failures, and never raises.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from indicator.okx.alerter import (
    format_exit_alert, format_kill_alert, send_critical,
)


# ── send_critical ────────────────────────────────────────────────────


class TestSendCritical:
    def test_empty_chat_id_returns_false(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            assert send_critical("", "msg") is False

    def test_missing_token_returns_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            assert send_critical("chat", "msg") is False

    def test_200_response_returns_true(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            with patch("indicator.okx.alerter.requests.post") as post:
                post.return_value = MagicMock(status_code=200, text="")
                assert send_critical("chat", "msg") is True
        post.assert_called_once()
        call_kwargs = post.call_args.kwargs
        assert call_kwargs["json"]["chat_id"] == "chat"
        assert call_kwargs["json"]["text"] == "msg"
        assert call_kwargs["json"]["parse_mode"] == "Markdown"

    def test_non_200_response_returns_false(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            with patch("indicator.okx.alerter.requests.post") as post:
                post.return_value = MagicMock(status_code=429,
                                              text="rate limit")
                assert send_critical("chat", "msg") is False

    def test_exception_swallowed_returns_false(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            with patch("indicator.okx.alerter.requests.post",
                       side_effect=RuntimeError("network")):
                # Must not raise
                assert send_critical("chat", "msg") is False

    def test_markdown_400_falls_back_to_plain_text(self):
        # 2026-06-19 bug: an exit reason like "opp_signal" has an unbalanced
        # '_' that makes Telegram reject the Markdown message (400), silently
        # dropping EVERY exit notification.  send_critical must retry once as
        # plain text so the alert still gets through.
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            with patch("indicator.okx.alerter.requests.post") as post:
                post.side_effect = [
                    MagicMock(status_code=400, text="can't parse entities"),
                    MagicMock(status_code=200, text=""),
                ]
                assert send_critical("chat", "*EXIT* (opp_signal)") is True
        assert post.call_count == 2
        first, second = post.call_args_list
        assert first.kwargs["json"]["parse_mode"] == "Markdown"
        # retry must NOT carry parse_mode (that's what makes it succeed)
        assert "parse_mode" not in second.kwargs["json"]

    def test_429_does_not_double_send(self):
        # Only a 400 (parse error) is retryable as plain text; rate-limit /
        # server errors aren't fixed by dropping parse_mode.
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok"}):
            with patch("indicator.okx.alerter.requests.post") as post:
                post.return_value = MagicMock(status_code=429, text="rate")
                assert send_critical("chat", "msg") is False
        post.assert_called_once()


# ── format_exit_alert (Markdown-safety regression) ───────────────────


class TestFormatExitAlert:
    @pytest.mark.parametrize("reason",
                             ["opp_signal", "trail_stop", "time_cap",
                              "manual_close_trail_bug"])
    def test_reason_is_backtick_wrapped(self, reason):
        # The reason MUST sit inside a code span so its '_' renders literally
        # and doesn't break legacy Markdown (the bug that ate exit alerts).
        msg = format_exit_alert(
            stage_label="live", direction="SHORT", reason=reason,
            entry_price=63761.99, exit_price=62337.8,
            gross_pct=0.0223, net_pct=0.0215, equity_after=104.08,
        )
        assert f"`{reason}`" in msg
        # underscores only ever appear inside the backtick code span → the
        # message outside code spans has no stray italic markers
        assert reason in msg


# ── format_kill_alert ────────────────────────────────────────────────


class TestFormatKillAlert:
    def test_basic_format(self):
        msg = format_kill_alert(
            trigger_id="A4", severity="HALT",
            reason="reconciliation mismatch",
            stage_label="testnet",
        )
        assert "A4" in msg
        assert "HALT" in msg
        assert "reconciliation mismatch" in msg
        assert "TESTNET" in msg

    def test_context_appended(self):
        msg = format_kill_alert(
            trigger_id="CAP-3", severity="HALT",
            reason="daily loss",
            stage_label="live",
            context={"day_change_pct": -52.0, "day_start": 100.0},
        )
        assert "day_change_pct" in msg
        assert "day_start" in msg

    def test_context_capped_at_5_items(self):
        # Avoid wall-of-text alerts.  Use a key shape that can't collide
        # with format-string vocabulary.
        big_ctx = {f"ctx{i}": i for i in range(20)}
        msg = format_kill_alert(
            trigger_id="X", severity="HALT", reason="r",
            stage_label="testnet", context=big_ctx,
        )
        # Only the first 5 ctx entries appear
        assert msg.count("ctx") == 5
