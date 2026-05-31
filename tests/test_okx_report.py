"""Unit tests for indicator/okx/report.py — Sharpe + summary aggregation.

DB-touching functions are tested via the format helpers (no MySQL
needed); pure math helpers verified directly.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from indicator.okx.report import (
    annualised_sharpe,
    format_okx_report,
    per_trade_sharpe,
)


# ── per_trade_sharpe ──────────────────────────────────────────────────


class TestPerTradeSharpe:
    def test_empty_returns_none(self):
        assert per_trade_sharpe([]) is None

    def test_single_trade_returns_none(self):
        assert per_trade_sharpe([0.01]) is None

    def test_constant_returns_none_div_zero(self):
        # std == 0 → undefined
        assert per_trade_sharpe([0.01, 0.01, 0.01]) is None

    def test_typical_positive_mix(self):
        # +1%, -0.5%, +2% → mean ~0.83%, std ~1.26%
        result = per_trade_sharpe([0.01, -0.005, 0.02])
        assert result is not None
        assert 0.4 < result < 1.0   # rough sanity range

    def test_all_losses_negative_sharpe(self):
        result = per_trade_sharpe([-0.01, -0.02, -0.005])
        assert result is not None
        assert result < 0


# ── annualised_sharpe ─────────────────────────────────────────────────


class TestAnnualisedSharpe:
    def test_zero_trades_per_year(self):
        assert annualised_sharpe([0.01, 0.02], trades_per_year=0) is None

    def test_typical_scaling(self):
        # base sharpe ~1, 365 trades/year → annualised ~sqrt(365) ≈ 19.1
        pt = per_trade_sharpe([0.01, -0.005, 0.015, -0.002, 0.008])
        ann = annualised_sharpe([0.01, -0.005, 0.015, -0.002, 0.008],
                                  trades_per_year=365)
        assert pt is not None
        assert ann == pytest.approx(pt * math.sqrt(365))

    def test_low_frequency_dampens(self):
        # 24 trades/year (V7 pace) means annualised ≈ base × ~4.9
        pt = per_trade_sharpe([0.01, -0.005, 0.02])
        ann = annualised_sharpe([0.01, -0.005, 0.02], trades_per_year=24)
        assert ann is not None
        assert ann == pytest.approx(pt * math.sqrt(24))


# ── format_okx_report ─────────────────────────────────────────────────


class TestFormatReport:
    def _base_summary(self, **overrides) -> dict:
        d = {
            "current_equity_usd": 155.0,
            "available_usd": 155.0,
            "balance_age_sec": 5,
            "initial_capital_usd": 155.0,
            "eq_pct_from_initial": 0.0,
            "executor_status": "ACTIVE",
            "executor_reason": "ready_to_trade",
            "executor_changed_at": datetime(2026, 6, 1, 12, 0),
            "n_closed": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "avg_net_bps": 0.0,
            "cum_net_pct": 0.0,
            "cum_equity_pct": 0.0,
            "sharpe_per_trade": None,
            "sharpe_annualised": None,
            "open_position": None,
            "recent_trades": [],
            "kill_log_7d": [],
        }
        d.update(overrides)
        return d

    def test_empty_state_renders(self):
        out = format_okx_report(self._base_summary())
        assert "OKX LIVE Stage 3" in out
        assert "Equity: $155.00" in out
        assert "ACTIVE" in out
        assert "No closed trades yet" in out
        assert "No open position" in out

    def test_with_trades(self):
        summary = self._base_summary(
            n_closed=3, wins=2, win_rate_pct=66.67,
            avg_net_bps=120.0, cum_net_pct=3.6,
            cum_equity_pct=36.0, sharpe_per_trade=0.85,
            sharpe_annualised=4.2,
        )
        out = format_okx_report(summary)
        assert "Trades: 3" in out
        assert "2/3" in out
        assert "+120.0 bps" in out
        assert "+3.60%" in out
        assert "Sharpe/trade: 0.85" in out
        assert "annualised: 4.20" in out

    def test_with_open_position(self):
        summary = self._base_summary(
            open_position={
                "id": 1, "direction": "LONG", "entry_tier": "Strong",
                "entry_price": 73500.0, "current_stop": 72800.0,
                "size_contracts": 2,
                "entry_time": datetime.utcnow() - timedelta(hours=5),
            }
        )
        out = format_okx_report(summary)
        assert "Open #1" in out
        assert "LONG" in out
        assert "$73500" in out
        assert "stop $72800" in out

    def test_halted_status_shows_yellow(self):
        out = format_okx_report(self._base_summary(executor_status="HALTED"))
        assert "🟡" in out
        assert "HALTED" in out

    def test_demoted_status_shows_red(self):
        out = format_okx_report(self._base_summary(executor_status="DEMOTED"))
        assert "🔴" in out

    def test_kill_log_present(self):
        summary = self._base_summary(kill_log_7d=[
            {"ts": datetime(2026, 5, 31, 14, 0), "trigger_id": "A2",
             "severity": "DEMOTE", "context": "{}"},
        ])
        out = format_okx_report(summary)
        assert "Kill log" in out
        assert "A2" in out
        assert "DEMOTE" in out

    def test_balance_missing_handled(self):
        summary = self._base_summary(current_equity_usd=None,
                                       available_usd=None,
                                       balance_age_sec=None)
        out = format_okx_report(summary)
        assert "no balance snapshot yet" in out
        # Should not crash
