"""Tests for the TV-alert event-card poller (pure parts — no DB)."""
import numpy as np
import pandas as pd

from market_data.tasks.cancel_playbook_watcher import (
    action_keyboard, compute_features, first_hit_verdict, format_outcome_reply,
    verdict_mark)
from market_data.tasks.tv_alert_poller import (
    format_card, format_stage1_card, format_stage2_card,
    format_tv_outcome_reply, hr_call_direction, hr_flags, state_distribution)


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
        assert "H4_resistance" in card and "(sell·賣方流動性·低點)" in card
        assert "118,250" in card
        assert "🟢 向上真空" in card and "偏漲" in card
        assert "發生了什麼:" in card and "掛單面:" in card   # 白話翻譯層
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


class TestHumanize:
    """白話翻譯層 — display-only, 凍結定義不碰（2026-07-20）。"""

    def test_story_absorption_up_mentions_reversal(self):
        from market_data.tasks.cancel_playbook_watcher import humanize_story
        s = humanize_story("absorption", "UP", vshock=28.9, taker_ratio=-0.50)
        assert "吸收" in s and "反轉向上" in s and "29 倍" in s and "50%" in s

    def test_story_vacuum_down_and_state_alias(self):
        from market_data.tasks.cancel_playbook_watcher import humanize_story
        assert "向下真空" in humanize_story("vacuum", "DOWN")
        assert "向下真空" in humanize_story("vacuum_down", "DOWN")
        assert "真破" in humanize_story("cascade", "UP", 5.0, 0.4)

    def test_book_direction_language(self):
        from market_data.tasks.cancel_playbook_watcher import humanize_book
        up = humanize_book(shock=3.6, skew15=0.35, net15=0.32)
        assert "3.6 倍" in up and "阻力在變薄" in up and "上方掛單淨撤離" in up
        flat = humanize_book(shock=1.0, skew15=0.01, net15=-0.02)
        assert "均衡" in flat and "無明顯淨撤離" in flat
        assert humanize_book() == "掛單資料不足"

    def test_book_dominant_action_appended(self):
        from market_data.tasks.cancel_playbook_watcher import humanize_book
        t = humanize_book(shock=3.6, skew15=0.1, net15=0.05, dominant="賣方加單")
        assert "具體動作: 賣方加單佔主導" in t
        assert "具體動作" not in humanize_book(shock=3.6)   # dominant=None 不強加


class TestDominantFlowAction:
    """FROZEN v1 (2026-07-21)：撤單/加單機制拆解——net15 只給淨值方向，看
    不出是「這側撤單」還是「對側加單」造成的同一個淨值；使用者指出這個
    含糊處後加上。ba/bc/aa/ac 四個 shock 裡最大的那個就是主導動作。"""

    def test_ask_cancel_dominant(self):
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        r = pd.Series(dict(ba_shock=1.0, bc_shock=1.2, aa_shock=0.9, ac_shock=4.5))
        assert dominant_flow_action(r) == "賣方撤單"

    def test_ask_add_dominant(self):
        # 這正是使用者今天指出的情境：不是撤單造成不平衡，是賣方在加單
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        r = pd.Series(dict(ba_shock=1.1, bc_shock=1.0, aa_shock=5.2, ac_shock=1.3))
        assert dominant_flow_action(r) == "賣方加單"

    def test_bid_side_mirrors(self):
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        r = pd.Series(dict(ba_shock=1.0, bc_shock=3.8, aa_shock=1.0, ac_shock=1.0))
        assert dominant_flow_action(r) == "買方撤單"
        r2 = pd.Series(dict(ba_shock=6.0, bc_shock=1.0, aa_shock=1.0, ac_shock=1.0))
        assert dominant_flow_action(r2) == "買方加單"

    def test_below_threshold_is_none(self):
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        r = pd.Series(dict(ba_shock=1.1, bc_shock=1.3, aa_shock=1.0, ac_shock=1.4))
        assert dominant_flow_action(r) is None

    def test_missing_or_nan_values(self):
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        assert dominant_flow_action(pd.Series(dtype=float)) is None
        r = pd.Series(dict(ba_shock=np.nan, bc_shock=np.nan,
                           aa_shock=np.nan, ac_shock=np.nan))
        assert dominant_flow_action(r) is None

    def test_compute_features_produces_per_side_shock_columns(self):
        # 整合測試：compute_features() 真的算出 *_shock，且最後一根(賣方
        # 加單狂噴)能被 dominant_flow_action 正確抓到——不是只測孤立函數。
        from market_data.tasks.cancel_playbook_watcher import dominant_flow_action
        n = 90
        df = pd.DataFrame({
            "ba": [10.0] * n, "bc": [10.0] * n,
            "aa": [10.0] * (n - 1) + [200.0],   # 最後一根賣方狂加單
            "ac": [10.0] * n,
            "vol": [100.0] * n, "dlt": [0.0] * n,
            "mid": [64000.0] * n,
        })
        feat = compute_features(df)
        for col in ("ba_shock", "bc_shock", "aa_shock", "ac_shock"):
            assert col in feat.columns
        last = feat.iloc[-1]
        assert last["aa_shock"] > 10          # 遠高於基線
        assert dominant_flow_action(last) == "賣方加單"


class TestActionKeyboard:
    """行動鍵 (2026-07-20) — 判讀鍵退役, 按鈕改成給工具。"""

    def test_four_buttons_and_callback_format(self):
        kb = action_keyboard("tv", 123456)
        flat = [b for row_ in kb["inline_keyboard"] for b in row_]
        assert len(flat) == 4
        datas = [b["callback_data"] for b in flat]
        assert datas[0] == "cfa|tv|123456|zoom"
        assert {d.split("|")[3] for d in datas} == {
            "zoom", "deep", "star", "dismiss"}
        # Telegram callback_data hard limit
        assert all(len(d.encode()) <= 64 for d in datas)


class TestOutcomeReply:
    """對答案 reply (2026-07-20) — 純格式函數。"""

    def test_playbook_hit_with_stats(self):
        e = dict(playbook="absorption", direction="UP", px=64400.05,
                 minute_start_ms=1789000000000, fwd_ret_60m=0.0042,
                 fwd_ret_120m=0.0031, hit_60m=1)
        t = format_outcome_reply(e, (20, 0.45))
        assert "對答案" in t and "吸收" in t and "偏漲" in t
        assert "（反轉）" in t                          # 2026-07-20: 判讀類型標籤
        assert "判讀結果: ✅ 對了（以 60m 為準）" in t   # 2026-07-21: 醒目標頭行
        assert "60m +0.42% ✅ 對了" in t and "120m +0.31% ✅ 對了" in t
        assert "近 20 筆命中率 45%" in t and "勿作交易依據" in t

    def test_playbook_call_tag_true_break_is_continuation(self):
        e = dict(playbook="true_break", direction="DOWN", px=64333.0,
                 minute_start_ms=1789000000000, fwd_ret_60m=0.0062,
                 fwd_ret_120m=None, hit_60m=0)
        t = format_outcome_reply(e, None)
        assert "真破（順勢延續）" in t

    def test_playbook_miss_no_stats(self):
        e = dict(playbook="true_break", direction="DOWN", px=64333.0,
                 minute_start_ms=1789000000000, fwd_ret_60m=0.0062,
                 fwd_ret_120m=None, hit_60m=0)
        t = format_outcome_reply(e, None)
        assert "判讀結果: ❌ 錯了（以 60m 為準）" in t
        assert "❌ 錯了" in t and "120m" not in t and "命中率" not in t

    def test_verdict_mark_none_direction_or_missing_ret(self):
        assert verdict_mark("NONE", 0.01) == ""
        assert verdict_mark("UP", None) == ""
        assert verdict_mark("UP", 0.01) == "✅ 對了"
        assert verdict_mark("UP", -0.01) == "❌ 錯了"
        assert verdict_mark("DOWN", -0.01) == "✅ 對了"

    def test_playbook_reply_shows_first_hit_line(self):
        e = dict(playbook="true_break", direction="UP", px=64000.0,
                 minute_start_ms=1789000000000, fwd_ret_60m=-0.001,
                 fwd_ret_120m=0.012, hit_60m=0, first_hit_result="hit")
        t = format_outcome_reply(e, None)
        assert "先觸價判斷(±0.5%): ✅ 對了" in t

    def test_playbook_reply_omits_first_hit_line_when_absent(self):
        e = dict(playbook="true_break", direction="UP", px=64000.0,
                 minute_start_ms=1789000000000, fwd_ret_60m=0.001,
                 fwd_ret_120m=None, hit_60m=1, first_hit_result=None)
        t = format_outcome_reply(e, None)
        assert "先觸價判斷" not in t


class TestFirstHitVerdict:
    """FROZEN v1 (2026-07-21): ±0.5% 門檻沿用 outcome_tracker.py 既有值，
    120 分鐘窗沿用既有 fwd_ret_120m 窗口——並行診斷，不取代固定時間點指標。"""

    def test_up_hits_target_first(self):
        mids = [64100, 64200, 64350]   # entry 64000, target 64320(+0.5%)
        assert first_hit_verdict(64000.0, "UP", mids) == "hit"

    def test_up_hits_opposite_first(self):
        mids = [63900, 63680, 64500]   # opposite 63680(-0.5%) 先於 target
        assert first_hit_verdict(64000.0, "UP", mids) == "miss"

    def test_down_mirrors(self):
        assert first_hit_verdict(64000.0, "DOWN", [63900, 63670]) == "hit"
        assert first_hit_verdict(64000.0, "DOWN", [64100, 64330]) == "miss"

    def test_neither_touched_is_unresolved(self):
        assert first_hit_verdict(64000.0, "UP", [64050, 64100, 63980]) \
            == "unresolved"

    def test_none_direction_or_empty_mids(self):
        assert first_hit_verdict(64000.0, "NONE", [64500]) is None
        assert first_hit_verdict(64000.0, "UP", []) is None

    def test_tv_reply_returns_only(self):
        row = dict(received_ms=1789000000000, event="H4_resistance",
                   fwd_ret_30m=-0.001, fwd_ret_60m=0.002, fwd_ret_120m=0.005)
        t = format_tv_outcome_reply(row)
        assert "H4_resistance" in t
        assert "30m -0.10%" in t and "60m +0.20%" in t and "120m +0.50%" in t

    def test_tv_reply_missing_data(self):
        row = dict(received_ms=1789000000000, event="",
                   fwd_ret_30m=None, fwd_ret_60m=None, fwd_ret_120m=None)
        assert "資料不足" in format_tv_outcome_reply(row)

    def test_tv_reply_no_verdict_omits_call_line(self):
        # 純關卡快訊(無 side)或尚未進 stage2 者 hr_verdict 為 None
        row = dict(received_ms=1789000000000, event="H4_resistance",
                   fwd_ret_30m=0.001, fwd_ret_60m=0.002, fwd_ret_120m=0.003)
        t = format_tv_outcome_reply(row)
        assert "原始判讀" not in t

    def test_tv_reply_reversal_verdict_shows_call(self):
        row = dict(received_ms=1789000000000, event="BSL_65058",
                   liquidity_side="buy", hr_verdict="reversal",
                   fwd_ret_30m=-0.002, fwd_ret_60m=-0.003, fwd_ret_120m=None)
        t = format_tv_outcome_reply(row)
        assert "原始判讀: 反轉（預期 偏跌 🔴）" in t

    def test_tv_reply_continuation_verdict_shows_call(self):
        row = dict(received_ms=1789000000000, event="SSL_63800",
                   liquidity_side="sell", hr_verdict="continuation",
                   fwd_ret_30m=-0.002, fwd_ret_60m=None, fwd_ret_120m=None)
        t = format_tv_outcome_reply(row)
        assert "原始判讀: 延續（預期 偏跌 🔴）" in t
        assert "判讀結果" not in t          # 60m/120m 都缺資料，不硬判對錯

    def test_tv_reply_reversal_with_60m_shows_verdict_line(self):
        # side=buy(BSL) + reversal → 預期 DOWN；fwd_ret_60m=-0.003(跌了) → 對了
        row = dict(received_ms=1789000000000, event="BSL_65058",
                   liquidity_side="buy", hr_verdict="reversal",
                   fwd_ret_30m=-0.001, fwd_ret_60m=-0.003, fwd_ret_120m=-0.004)
        t = format_tv_outcome_reply(row)
        assert "判讀結果: ✅ 對了（以 60m 為準）" in t
        assert "60m -0.30% ✅ 對了" in t and "120m -0.40% ✅ 對了" in t

    def test_tv_reply_continuation_wrong_direction(self):
        # side=sell(SSL) + continuation → 預期 DOWN；fwd_ret_60m=+0.002(漲了) → 錯了
        row = dict(received_ms=1789000000000, event="SSL_63800",
                   liquidity_side="sell", hr_verdict="continuation",
                   fwd_ret_30m=None, fwd_ret_60m=0.002, fwd_ret_120m=None)
        t = format_tv_outcome_reply(row)
        assert "判讀結果: ❌ 錯了（以 60m 為準）" in t

    def test_tv_reply_shows_first_hit_line(self):
        row = dict(received_ms=1789000000000, event="BSL_65058",
                   liquidity_side="buy", hr_verdict="continuation",
                   fwd_ret_30m=0.001, fwd_ret_60m=-0.001, fwd_ret_120m=0.02,
                   first_hit_result="hit")
        t = format_tv_outcome_reply(row)
        assert "先觸價判斷(±0.5%): ✅ 對了" in t

    def test_tv_reply_omits_first_hit_line_when_absent(self):
        row = dict(received_ms=1789000000000, event="H4_resistance",
                   fwd_ret_30m=0.001, fwd_ret_60m=None, fwd_ret_120m=None,
                   first_hit_result=None)
        t = format_tv_outcome_reply(row)
        assert "先觸價判斷" not in t


class TestHrCallDirection:
    def test_buy_side_mirrors(self):
        assert hr_call_direction("buy", "reversal") == "DOWN"
        assert hr_call_direction("buy", "continuation") == "UP"

    def test_sell_side_mirrors(self):
        assert hr_call_direction("sell", "reversal") == "UP"
        assert hr_call_direction("sell", "continuation") == "DOWN"

    def test_inferred_side_suffix_stripped(self):
        assert hr_call_direction("buy?", "reversal") == "DOWN"


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
