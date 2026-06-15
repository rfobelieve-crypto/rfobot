"""Tests for indicator/okx/runner.py singleton + env gating.

Mocks the heavy executor.start() chain via patching the OkxClient and
OkxStateStore that runner instantiates internally.  Verifies:
  - OFF state when env flag absent
  - One-time init on flag enabled
  - Failed init disables until reset
  - get_approval_gate exposes the same gate as get_executor
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from indicator.okx import runner


@pytest.fixture(autouse=True)
def _reset_singleton():
    runner.reset_for_tests()
    yield
    runner.reset_for_tests()


@pytest.fixture
def _live_env():
    """Set the minimum required env vars for load_okx_config_from_env."""
    env_set = {
        "OKX_EXECUTOR_ENABLED": "1",
        "STAGE": "live",
        "OKX_API_KEY_LIVE": "k",
        "OKX_API_SECRET_LIVE": "s",
        "OKX_PASSPHRASE_LIVE": "p",
        "TG_CRITICAL_CHAT_ID": "critical",
    }
    saved = {k: os.environ.get(k) for k in env_set}
    os.environ.update(env_set)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestIsEnabled:
    def test_unset_means_off(self, monkeypatch):
        monkeypatch.delenv("OKX_EXECUTOR_ENABLED", raising=False)
        assert not runner.is_enabled()

    def test_explicit_zero_means_off(self, monkeypatch):
        monkeypatch.setenv("OKX_EXECUTOR_ENABLED", "0")
        assert not runner.is_enabled()

    def test_one_means_on(self, monkeypatch):
        monkeypatch.setenv("OKX_EXECUTOR_ENABLED", "1")
        assert runner.is_enabled()

    def test_true_lowercase_means_on(self, monkeypatch):
        monkeypatch.setenv("OKX_EXECUTOR_ENABLED", "true")
        assert runner.is_enabled()


class TestGetExecutor:
    def test_flag_off_returns_none(self, monkeypatch):
        monkeypatch.delenv("OKX_EXECUTOR_ENABLED", raising=False)
        assert runner.get_executor() is None

    def test_flag_on_init_returns_singleton(self, _live_env):
        # Patch the live-cap check too — $200 cap for live; our default
        # cfg has $100 so passes
        with patch("indicator.okx.runner.V7OkxExecutor") as MockExe, \
                patch("indicator.okx.runner.OkxClient") as _MockClient, \
                patch("indicator.okx.runner.OkxStateStore") as _MockStore, \
                patch("indicator.okx.runner.PositionReconciler") as _MockRec:
            exe = MagicMock()
            exe.get_status.return_value = MagicMock(value="ACTIVE")
            MockExe.return_value = exe
            result = runner.get_executor()
        assert result is exe
        # start() called exactly once
        exe.start.assert_called_once()

    def test_second_call_returns_same_instance(self, _live_env):
        with patch("indicator.okx.runner.V7OkxExecutor") as MockExe, \
                patch("indicator.okx.runner.OkxClient"), \
                patch("indicator.okx.runner.OkxStateStore"), \
                patch("indicator.okx.runner.PositionReconciler"):
            exe = MagicMock()
            exe.get_status.return_value = MagicMock(value="ACTIVE")
            MockExe.return_value = exe
            first = runner.get_executor()
            second = runner.get_executor()
        assert first is second
        # Only constructed once
        MockExe.assert_called_once()

    def test_init_failure_disables(self, _live_env):
        with patch("indicator.okx.runner.V7OkxExecutor",
                   side_effect=RuntimeError("boom")), \
                patch("indicator.okx.runner.OkxClient"), \
                patch("indicator.okx.runner.OkxStateStore"), \
                patch("indicator.okx.runner.PositionReconciler"):
            assert runner.get_executor() is None
            # Subsequent calls also return None without retry
            assert runner.get_executor() is None

    def test_validate_failure_disables(self, monkeypatch):
        monkeypatch.setenv("OKX_EXECUTOR_ENABLED", "1")
        # No live keys set → validate_okx_config raises
        assert runner.get_executor() is None
        # And stays disabled
        assert runner.get_executor() is None


class TestGetApprovalGate:
    def test_off_returns_none(self, monkeypatch):
        monkeypatch.delenv("OKX_EXECUTOR_ENABLED", raising=False)
        assert runner.get_approval_gate() is None

    def test_on_returns_gate(self, _live_env):
        with patch("indicator.okx.runner.V7OkxExecutor") as MockExe, \
                patch("indicator.okx.runner.OkxClient"), \
                patch("indicator.okx.runner.OkxStateStore"), \
                patch("indicator.okx.runner.PositionReconciler"):
            exe = MagicMock()
            exe.get_status.return_value = MagicMock(value="ACTIVE")
            MockExe.return_value = exe
            gate = runner.get_approval_gate()
        assert gate is not None
