"""Tests for the TV-alert event-card poller (pure parts — no DB)."""
import numpy as np
import pandas as pd

from market_data.tasks.tv_alert_poller import format_card, state_distribution


def feat_row(**kw) -> pd.Series:
    base = dict(shock=1.2, skew15=0.05, net15=-0.02, vshock=0.8,
                taker_ratio=0.10, ret_1m=0.0, mid=118000.0)
    base.update(kw)
    return pd.Series(base)


CALM = {"state": "calm", "zh": "平靜", "emoji": "⚫", "direction": "NONE"}
VUP = {"state": "vacuum_up", "zh": "向上真空", "emoji": "🟢", "direction": "UP"}


class TestStateDistribution:
    def test_orders_and_counts(self):
        states = [CALM] * 3 + [VUP] * 2
        assert state_distribution(states) == "平靜3 · 向上真空2"

    def test_empty(self):
        assert state_distribution([]) == ""


class TestFormatCard:
    ROW = {"received_ms": 1789000000000, "event": "H4_resistance",
           "liquidity_side": "sell", "price": 118250.0}

    def test_contains_level_state_and_window(self):
        card = format_card(self.ROW, VUP, feat_row(), "平靜88 · 向上真空2", 90)
        assert "H4_resistance" in card and "(sell)" in card
        assert "118,250" in card
        assert "🟢 向上真空→UP" in card
        assert "回看90m: 平靜88 · 向上真空2" in card
        assert "非信號" in card and "勿作交易依據" in card

    def test_no_markdown_specials_break_plain_text(self):
        # plain-text send (no parse_mode) — mistake.md 2026-06-19; underscore
        # in the level name must simply pass through.
        card = format_card(self.ROW, CALM, feat_row(), "平靜90", 90)
        assert "H4_resistance" in card

    def test_nan_features_render_as_question_mark(self):
        card = format_card(self.ROW, CALM,
                           feat_row(shock=np.nan, taker_ratio=np.nan),
                           "平靜90", 90)
        assert "shock ?x" in card and "taker ?" in card

    def test_missing_price_and_side(self):
        row = {"received_ms": 1789000000000, "event": "", "liquidity_side": "",
               "price": None}
        card = format_card(row, CALM, feat_row(), "平靜90", 90)
        assert "@ ?" in card and "(" not in card.splitlines()[1]
