"""Tests for the TV-alert event-card poller (pure parts — no DB)."""
import numpy as np
import pandas as pd

from market_data.tasks.cancel_playbook_watcher import verdict_keyboard
from market_data.tasks.tv_alert_poller import (
    format_card, format_stage1_card, format_stage2_card, hr_flags,
    state_distribution)


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


def seg_frame(ba, bc, aa, ac, n=5) -> pd.DataFrame:
    return pd.DataFrame({"ba": [ba] * n, "bc": [bc] * n,
                         "aa": [aa] * n, "ac": [ac] * n})


class TestHRFlags:
    """H-R 凍結雙旗標 — side 語義: buy=BSL 向上掃(被掃側=ask)。"""

    def test_up_sweep_reversal(self):
        # ask 回補(add>cancel) + bid 淨撤 → 反轉 DOWN
        f = hr_flags(seg_frame(ba=1, bc=5, aa=9, ac=2), "buy")
        assert f["refill"] and f["opp_pull"] and f["reversal"]
        assert f["rev_dir"] == "DOWN"

    def test_up_sweep_continuation(self):
        # ask 持續撤、bid 重建 → 兩旗皆滅
        f = hr_flags(seg_frame(ba=9, bc=1, aa=1, ac=8), "buy")
        assert not f["refill"] and not f["opp_pull"] and not f["reversal"]

    def test_down_sweep_mirror(self):
        # sell=SSL 向下掃: bid 回補 + ask 淨撤 → 反轉 UP
        f = hr_flags(seg_frame(ba=9, bc=2, aa=1, ac=5), "sell")
        assert f["reversal"] and f["rev_dir"] == "UP"


class TestVerdictKeyboard:
    def test_four_buttons_and_callback_format(self):
        kb = verdict_keyboard("tv", 123456)
        flat = [b for row_ in kb["inline_keyboard"] for b in row_]
        assert len(flat) == 4
        datas = [b["callback_data"] for b in flat]
        assert datas[0] == "ceb|tv|123456|up"
        assert {d.split("|")[3] for d in datas} == {
            "up", "down", "unsure", "skip"}
        # Telegram callback_data hard limit
        assert all(len(d.encode()) <= 64 for d in datas)


class TestStageCards:
    ROW = {"received_ms": 1789000000000, "event": "PDH_sweep",
           "liquidity_side": "buy", "price": 118250.0, "window_mins": 90}

    def test_stage1_reports_facts_only(self):
        c = format_stage1_card(self.ROW)
        assert "第1段" in c and "塵埃落定" in c
        assert "結論" not in c            # 掃穿瞬間不判讀

    def test_stage2_reversal_conclusion(self):
        flags = {"refill": True, "opp_pull": True, "reversal": True,
                 "rev_dir": "DOWN", "ask_net": -5.0, "bid_net": 3.0}
        c = format_stage2_card(self.ROW, flags, CALM, 8, 7.0)
        assert "反轉條件成立" in c and "DOWN" in c and "✓" in c

    def test_stage2_continuation_keeps_base_rate_warning(self):
        flags = {"refill": False, "opp_pull": True, "reversal": False,
                 "rev_dir": "DOWN", "ask_net": 5.0, "bid_net": 3.0}
        c = format_stage2_card(self.ROW, flags, CALM, 8, 15.0)
        assert "未現" in c and "2/3" in c
