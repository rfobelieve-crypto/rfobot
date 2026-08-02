# -*- coding: utf-8 -*-
"""組合層風控引擎的規則測試（docs/PORTFOLIO_RISK_FRAMEWORK.md）。

這份測試的用意不是覆蓋率，是**把設計稿的每一條規則釘成可執行的斷言**。
設計稿會被改、會被忘記；測試不會。特別是「引擎只會拒絕或縮小、永遠
不會放大」這條——它是整個框架的安全底線，用隨機輸入的性質測試守住。
"""
from __future__ import annotations

import random

import pytest

from indicator.portfolio import (Decision, Intent, OpenPosition,
                                 PortfolioLimits, PortfolioState,
                                 StrategyLimits, decide, default_limits)


def mk_state(**kw) -> PortfolioState:
    kw.setdefault("equity_usd", 1000.0)
    return PortfolioState(**kw)


def v7_intent(**kw) -> Intent:
    base = dict(strategy="v7", symbol="BTC-USD", side="LONG", risk_pct=0.25)
    base.update(kw)
    return Intent(**base)          # type: ignore[arg-type]


class TestAccountLayerSupremacy:
    """帳戶層一觸發，任何策略、任何情況都不得開倉（§6 不變量）。"""

    def test_account_halt_blocks_everything(self):
        d = decide(v7_intent(), mk_state(account_halted=True))
        assert d.rejected and d.reason == "account_halted"

    def test_account_halt_wins_over_a_perfectly_fine_intent(self):
        ok = decide(v7_intent(), mk_state())
        assert ok.approved
        blocked = decide(v7_intent(), mk_state(account_halted=True))
        assert blocked.rejected


class TestStrategyLayerCaps:
    def test_demoted_strategy_cannot_open(self):
        d = decide(v7_intent(), mk_state(demoted={"v7"}))
        assert d.rejected and d.reason == "strategy_demoted"

    def test_demote_outranks_daily_halt(self):
        """終態要蓋過當日狀態——理由要指向最根本的那條。"""
        d = decide(v7_intent(),
                   mk_state(demoted={"v7"}, strategy_day_r={"v7": -99}))
        assert d.reason == "strategy_demoted"

    def test_drawdown_cap_triggers_demote_reason(self):
        d = decide(v7_intent(), mk_state(strategy_dd_pct={"v7": -6.0}))
        assert d.rejected and d.reason == "strategy_dd_cap"

    def test_daily_r_cap_halts(self):
        d = decide(v7_intent(), mk_state(strategy_day_r={"v7": -5.0}))
        assert d.rejected and d.reason == "strategy_daily_r"

    def test_daily_pct_cap_halts(self):
        d = decide(v7_intent(), mk_state(strategy_day_pct={"v7": -2.0}))
        assert d.rejected and d.reason == "strategy_daily_pct"

    def test_just_inside_the_cap_still_trades(self):
        """邊界不能寬鬆一格也不能嚴格一格——cap 是 ≤ 觸發。"""
        d = decide(v7_intent(), mk_state(strategy_day_r={"v7": -4.99}))
        assert d.approved

    def test_one_strategy_halt_does_not_touch_another(self):
        """薄策略組合的本意：一條線壞了降級它自己，不拖全家（§3.2）。"""
        st = mk_state(strategy_day_r={"v7": -99})
        assert decide(v7_intent(), st).rejected
        sweep = Intent("sweep", "ETH-USD", "LONG", 0.15)
        assert decide(sweep, st).approved


class TestConcurrency:
    def test_v7_single_seat(self):
        st = mk_state(open_positions=[
            OpenPosition("v7", "BTC-USD", "LONG", 0.25)])
        d = decide(v7_intent(symbol="ETH-USD"), st)
        assert d.rejected and d.reason == "concurrency_cap"

    def test_sweep_gets_five_seats(self):
        pos = [OpenPosition("sweep", f"C{i}-USD", "LONG", 0.15)
               for i in range(4)]
        d = decide(Intent("sweep", "X-USD", "LONG", 0.15),
                   mk_state(open_positions=pos))
        assert d.approved
        pos.append(OpenPosition("sweep", "X-USD", "LONG", 0.15))
        d2 = decide(Intent("sweep", "Y-USD", "LONG", 0.15),
                    mk_state(open_positions=pos))
        assert d2.rejected and d2.reason == "concurrency_cap"

    def test_other_strategies_positions_do_not_consume_my_seats(self):
        pos = [OpenPosition("sweep", f"C{i}-USD", "LONG", 0.15)
               for i in range(5)]
        assert decide(v7_intent(), mk_state(open_positions=pos)).approved


class TestFakeDiversification:
    """9 幣教訓的制度化：同一份曝險不得領兩份預算（§3.4）。"""

    def test_same_symbol_same_side_shrinks_to_smallest_book(self):
        st = mk_state(open_positions=[
            OpenPosition("sweep", "BTC-USD", "LONG", 0.15)])
        d = decide(v7_intent(), st)          # v7 想要 0.25
        assert d.approved and d.risk_pct == pytest.approx(0.15)

    def test_opposite_side_is_not_a_collision(self):
        st = mk_state(open_positions=[
            OpenPosition("sweep", "BTC-USD", "SHORT", 0.15)])
        d = decide(v7_intent(), st)
        assert d.approved and d.risk_pct == pytest.approx(0.25)

    def test_different_symbol_is_not_a_collision(self):
        st = mk_state(open_positions=[
            OpenPosition("sweep", "ETH-USD", "LONG", 0.15)])
        assert decide(v7_intent(), st).risk_pct == pytest.approx(0.25)

    def test_correlated_pair_shares_one_book(self):
        st = mk_state(
            open_positions=[OpenPosition("sweep", "ETH-USD", "LONG", 0.15)],
            correlations={("sweep", "v7"): (0.71, 30)})
        d = decide(v7_intent(), st)
        assert d.approved and d.risk_pct == pytest.approx(0.15)

    def test_correlation_needs_enough_days(self):
        """樣本不足不是證據——8 天的 ρ=0.9 不該動預算。"""
        st = mk_state(
            open_positions=[OpenPosition("sweep", "ETH-USD", "LONG", 0.15)],
            correlations={("sweep", "v7"): (0.9, 8)})
        assert decide(v7_intent(), st).risk_pct == pytest.approx(0.25)

    def test_negative_correlation_also_squeezes(self):
        """|ρ| > 門檻——反向連動一樣是「同一條線」的證據。"""
        st = mk_state(
            open_positions=[OpenPosition("sweep", "ETH-USD", "LONG", 0.15)],
            correlations={("sweep", "v7"): (-0.8, 40)})
        assert decide(v7_intent(), st).risk_pct == pytest.approx(0.15)


class TestNetNotional:
    def test_same_side_same_symbol_counts_once(self):
        st = mk_state(open_positions=[
            OpenPosition("v7", "BTC-USD", "LONG", 0.25, notional_mult=1.2),
            OpenPosition("sweep", "BTC-USD", "LONG", 0.15, notional_mult=1.1),
        ])
        assert st.net_notional_mult() == pytest.approx(2.3)

    def test_opposite_sides_net_off(self):
        st = mk_state(open_positions=[
            OpenPosition("v7", "BTC-USD", "LONG", 0.25, notional_mult=1.2),
            OpenPosition("sweep", "BTC-USD", "SHORT", 0.15, notional_mult=1.1),
        ])
        assert st.net_notional_mult() == pytest.approx(0.1)

    def test_total_cap_rejects(self):
        st = mk_state(open_positions=[
            OpenPosition("sweep", "ETH-USD", "LONG", 0.15, notional_mult=2.0)])
        d = decide(v7_intent(), st)
        assert d.rejected and d.reason == "total_notional_cap"


class TestFilterOnlySeat:
    def test_filter_line_cannot_open(self):
        d = decide(Intent("cancel", "BTC-USD", "LONG", 0.1), mk_state())
        assert d.rejected and d.reason == "filter_only"


class TestUnknownStrategy:
    def test_unregistered_gets_the_conservative_default_not_a_free_pass(self):
        d = decide(Intent("brand_new", "BTC-USD", "LONG", 5.0), mk_state())
        assert d.approved
        assert d.risk_pct == pytest.approx(StrategyLimits(name="x").risk_pct_per_trade)


class TestSafetyFloor:
    """整份設計的安全底線：引擎只能拒絕或縮小，永遠不能放大。"""

    def test_never_approves_more_than_requested_property(self):
        rng = random.Random(7)
        strategies = ["v7", "sweep", "unknown"]
        for _ in range(400):
            it = Intent(rng.choice(strategies), rng.choice(["BTC-USD", "ETH-USD"]),
                        rng.choice(["LONG", "SHORT"]),
                        round(rng.uniform(0.01, 3.0), 3))
            st = mk_state(
                open_positions=[
                    OpenPosition(rng.choice(strategies),
                                 rng.choice(["BTC-USD", "ETH-USD"]),
                                 rng.choice(["LONG", "SHORT"]),
                                 round(rng.uniform(0.05, 0.3), 3),
                                 notional_mult=round(rng.uniform(0, 0.6), 2))
                    for _ in range(rng.randint(0, 3))],
                strategy_day_r={s: rng.uniform(-6, 3) for s in strategies},
                strategy_dd_pct={s: rng.uniform(-8, 0) for s in strategies},
            )
            d = decide(it, st)
            assert d.risk_pct <= it.risk_pct + 1e-9, (it, d)
            if d.approved:
                assert d.risk_pct > 0

    def test_rejection_always_carries_a_machine_readable_reason(self):
        st = mk_state(account_halted=True)
        d = decide(v7_intent(), st)
        assert d.reason and " " not in d.reason      # 短碼，不是句子
        assert d.detail                              # 給人看的說明也要在


class TestLimitsAreExplicit:
    def test_defaults_match_the_design_doc(self):
        lim = default_limits()
        assert lim.max_total_notional_mult == 2.0
        assert lim.get("v7").max_concurrent == 1
        assert lim.get("sweep").max_concurrent == 5
        assert lim.get("cancel").filter_only is True
        # 提案值尚未拍板，這個旗標要跟著數字一起活著
        from indicator.portfolio.limits import PENDING_SIGNOFF
        assert PENDING_SIGNOFF is True


class TestNoLiveWiring:
    """P0 的紀律：沒有任何 live 模組 import 這個套件（設計稿 §4）。"""

    def test_package_is_not_imported_by_the_trading_path(self):
        import ast
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        watched = [root / "indicator" / "app.py",
                   root / "indicator" / "okx" / "executor.py",
                   root / "indicator" / "okx" / "runner.py",
                   root / "BTC_perp_data.py"]
        for f in watched:
            if not f.exists():
                continue
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mod = ""
                if isinstance(node, ast.Import):
                    mod = " ".join(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                assert "portfolio" not in mod, (
                    f"{f.name} 已 import 組合框架——P1 之前不該接線")
