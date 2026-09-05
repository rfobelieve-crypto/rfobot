# -*- coding: utf-8 -*-
"""宣告式範圍守衛的反向證明（2026-09-05）。

守衛加進來的當天必須看到它綠一次、也紅一次，否則不知道它有沒有在測量
（mistake.md 2026-09-03）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.declared_scope import Scope, ScopeShrunk  # noqa: E402


def test_full_scope_passes():
    Scope("t", expect_n=88, expect_days=365).check(actual_n=88, actual_days=365)


def test_missing_symbols_raises():
    """B2 的病：88 個標的被截成 25 個。"""
    with pytest.raises(ScopeShrunk, match="標的數"):
        Scope("B2 長尾", expect_n=88, expect_days=365).check(actual_n=25, actual_days=365)


def test_short_history_raises():
    """B 的病：宣告 365 天，實得 90 天（Bitget funding 上限）。"""
    with pytest.raises(ScopeShrunk, match="時間跨度"):
        Scope("B", expect_n=19, expect_days=365).check(actual_n=19, actual_days=90)


def test_named_reason_allows_and_is_recorded():
    s = Scope("X", expect_n=10)
    s.check(actual_n=8, allow_shrink="2 個標的桶數不足 200，事前規則排除")
    assert s.log and "事前規則" in s.log[0]
    assert s.as_dict()["shrink_notes"]


def test_silent_shrink_is_impossible():
    """沒有理由就不能過——這條是整個模組存在的理由。"""
    with pytest.raises(ScopeShrunk):
        Scope("X", expect_n=10).check(actual_n=9, allow_shrink="")
