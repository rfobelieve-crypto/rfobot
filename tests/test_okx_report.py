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
    bootstrap_mean_ci_bps,
    compute_gate_b_status,
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
            "wins_net": 0,
            "win_rate_pct_net": 0.0,
            "wins_equity": 0,
            "win_rate_pct_equity": 0.0,
            "avg_net_bps": 0.0,
            "cum_net_pct": 0.0,
            "cum_equity_pct": 0.0,
            "sharpe_per_trade": None,
            "sharpe_annualised": None,
            "gate_b": {
                "status": "accumulating",
                "n_closed": 0,
                "sample_min": 30,
                "sample_target": 50,
                "progress_pct": 0.0,
                "avg_net_bps": 0.0,
                "ci_lo_bps": None,
                "ci_hi_bps": None,
                "summary": "累積中 0/30 (目標 50)",
            },
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
            wins_net=2, win_rate_pct_net=66.67,
            wins_equity=1, win_rate_pct_equity=33.33,
            avg_net_bps=120.0, cum_net_pct=3.6,
            cum_equity_pct=36.0, sharpe_per_trade=0.85,
            sharpe_annualised=4.2,
        )
        out = format_okx_report(summary)
        assert "Trades: 3" in out
        # Three WR lines now rendered side-by-side
        assert "WR gross:" in out
        assert "WR net:" in out
        assert "WR equity:" in out
        assert "2/3" in out                  # gross + net both 2/3 (line shared text)
        assert "1/3" in out                  # equity 1/3
        assert "(wallet truth)" in out
        assert "+120.0 bps" in out
        assert "+3.60%" in out
        assert "Sharpe/trade: 0.85" in out
        assert "annualised: 4.20" in out

    def test_wr_definitions_diverge(self):
        """Realistic case: more gross wins than equity wins (cost + funding eat
        thin-margin gross winners)."""
        summary = self._base_summary(
            n_closed=10,
            wins=6, win_rate_pct=60.0,           # gross 60%
            wins_net=5, win_rate_pct_net=50.0,   # net 50%
            wins_equity=4, win_rate_pct_equity=40.0,  # equity 40%
            avg_net_bps=-5.0, cum_net_pct=-0.5,
            cum_equity_pct=-0.8,
        )
        out = format_okx_report(summary)
        assert "60%" in out  # gross
        assert "50%" in out  # net
        assert "40%" in out  # equity

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

    def test_gate_b_accumulating_renders(self):
        out = format_okx_report(self._base_summary())
        assert "Gate B" in out
        assert "累積中" in out
        assert "🟡" in out

    def test_gate_b_passed_renders_green(self):
        summary = self._base_summary(
            n_closed=40, gate_b={
                "status": "passed", "n_closed": 40,
                "sample_min": 30, "sample_target": 50,
                "progress_pct": 80.0, "avg_net_bps": 15.2,
                "ci_lo_bps": 3.4, "ci_hi_bps": 27.1,
                "summary": "avg net +15.2 bps, 95% CI [+3.4, +27.1] — Gate B 通過",
            }
        )
        out = format_okx_report(summary)
        assert "Gate B" in out
        assert "通過" in out
        assert "🟢" in out


# ── Gate B status logic ───────────────────────────────────────────────


class TestGateBStatus:
    def test_accumulating_below_min_sample(self):
        result = compute_gate_b_status(
            n_closed=10, net_pcts=[0.005] * 10, avg_net_bps=50.0)
        assert result["status"] == "accumulating"
        assert "累積中" in result["summary"]
        assert result["n_closed"] == 10

    def test_failed_when_avg_negative(self):
        # Negative avg even with 30+ trades → fail
        result = compute_gate_b_status(
            n_closed=30,
            net_pcts=[-0.001] * 30,
            avg_net_bps=-10.0,
        )
        assert result["status"] == "failed"
        assert "edge 未驗到" in result["summary"]

    def test_passed_when_avg_and_ci_lower_positive(self):
        # Consistently positive trades — CI lower should be > 0
        net_pcts = [0.005, 0.008, 0.003, 0.010, 0.006] * 8  # n=40, all +
        avg_net_bps = sum(net_pcts) / len(net_pcts) * 10000
        result = compute_gate_b_status(
            n_closed=40, net_pcts=net_pcts, avg_net_bps=avg_net_bps)
        assert result["status"] == "passed"
        assert result["ci_lo_bps"] > 0

    def test_marginal_when_ci_straddles_zero(self):
        # Wide spread — avg slightly positive but CI lower < 0
        net_pcts = [0.05, -0.04, 0.06, -0.05, 0.03, -0.02] * 6  # n=36, noisy
        avg_net_bps = sum(net_pcts) / len(net_pcts) * 10000
        result = compute_gate_b_status(
            n_closed=36, net_pcts=net_pcts, avg_net_bps=avg_net_bps)
        # Avg may be slightly positive but CI lower should straddle 0
        if avg_net_bps > 0:
            assert result["status"] == "marginal"
            assert result["ci_lo_bps"] <= 0

    def test_progress_pct_caps_at_100(self):
        result = compute_gate_b_status(
            n_closed=100, net_pcts=[0.0] * 100, avg_net_bps=0.0)
        assert result["progress_pct"] == 100.0


class TestBootstrapCI:
    def test_too_few_samples_returns_none(self):
        assert bootstrap_mean_ci_bps([0.01, 0.02]) is None

    def test_positive_constant_ci_stays_positive(self):
        # All trades exactly +50 bps — CI must be tight around +50
        ci = bootstrap_mean_ci_bps([0.005] * 30)
        assert ci is not None
        lo, hi = ci
        assert lo == pytest.approx(50.0, abs=0.1)
        assert hi == pytest.approx(50.0, abs=0.1)

    def test_deterministic_with_seed(self):
        net_pcts = [0.005, -0.003, 0.007, -0.001, 0.002] * 6  # n=30
        ci1 = bootstrap_mean_ci_bps(net_pcts, seed=42)
        ci2 = bootstrap_mean_ci_bps(net_pcts, seed=42)
        assert ci1 == ci2
