"""Unit tests for indicator/okx/accounts.py + multi-account runner.

No MySQL / no OKX network — DB-touching paths are exercised only up to
their guard clauses; crypto + prefix logic tested directly.
"""
from __future__ import annotations

import os

import pytest

from cryptography.fernet import Fernet


@pytest.fixture(autouse=True)
def _master_key(monkeypatch):
    monkeypatch.setenv("OKX_CRED_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("TG_ADMIN_CHAT_ID", "111")


class TestEncryption:
    def test_roundtrip(self):
        from indicator.okx.accounts import decrypt, encrypt
        for secret in ("plain", "with@at", "0aB!#$%^&*()", "長度測試" * 8):
            assert decrypt(encrypt(secret)) == secret

    def test_missing_master_key_raises(self, monkeypatch):
        monkeypatch.delenv("OKX_CRED_MASTER_KEY")
        from indicator.okx.accounts import encrypt
        with pytest.raises(RuntimeError, match="OKX_CRED_MASTER_KEY"):
            encrypt("x")

    def test_wrong_key_fails_decrypt(self, monkeypatch):
        from indicator.okx.accounts import decrypt, encrypt
        ct = encrypt("secret")
        monkeypatch.setenv("OKX_CRED_MASTER_KEY",
                           Fernet.generate_key().decode())
        with pytest.raises(Exception):
            decrypt(ct)


class TestGuards:
    def test_admin_gate(self):
        from indicator.okx.accounts import _is_admin
        assert _is_admin("111")
        assert not _is_admin("222")
        assert not _is_admin("")

    def test_add_account_label_guards(self):
        from indicator.okx.accounts import add_account
        assert "保留" in add_account("main", "k", "s", "p", 100)
        assert "label" in add_account("no-dash", "k", "s", "p", 100)
        assert "label" in add_account("x" * 33, "k", "s", "p", 100)

    def test_add_account_capital_ceiling(self):
        # Must match validate_okx_config Stage-3 live ceiling ($200)
        from indicator.okx.accounts import MAX_CAPITAL_USD, add_account
        assert MAX_CAPITAL_USD == 200.0
        assert "capital" in add_account("friend_a", "k", "s", "p", 201)
        assert "capital" in add_account("friend_a", "k", "s", "p", 0)


class TestTablePrefix:
    def test_prefix_format(self):
        from indicator.okx.accounts import table_prefix_for
        assert table_prefix_for(3) == "v7_okx_a3"
        assert table_prefix_for(42) == "v7_okx_a42"

    def test_prefix_is_int_coerced(self):
        from indicator.okx.accounts import table_prefix_for
        assert table_prefix_for("7") == "v7_okx_a7"

    def test_suffix_list_matches_state_store_tables(self):
        # Every table the OkxStateStore touches must be cloned per account.
        from indicator.okx.accounts import OKX_TABLE_SUFFIXES
        assert set(OKX_TABLE_SUFFIXES) == {
            "positions", "kill_log", "reconciliation_log",
            "executor_status", "balance_snapshots", "approvals",
        }


class TestRunnerMultiAccount:
    def test_disabled_returns_empty(self, monkeypatch):
        monkeypatch.delenv("OKX_EXECUTOR_ENABLED", raising=False)
        from indicator.okx import runner
        runner.reset_for_tests()
        assert runner.get_account_executors() == []

    def test_load_failure_returns_empty_not_raise(self, monkeypatch):
        # DB unreachable in CI → load_active_accounts logs + returns [];
        # the fan-out must degrade to "no friend accounts", never crash.
        monkeypatch.setenv("OKX_EXECUTOR_ENABLED", "1")
        from indicator.okx import runner
        runner.reset_for_tests()
        assert runner.get_account_executors() == []

    def test_reset_clears_account_executors(self):
        from indicator.okx import runner
        runner._ACCT_EXECUTORS[99] = object()
        runner.reset_for_tests()
        assert runner._ACCT_EXECUTORS == {}
