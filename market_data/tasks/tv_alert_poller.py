"""TV alert poller — cancel-flow state cards for user-drawn levels.

Event-source revival (TODO 2026-07-17 §4.5): the main bot's /tv webhook
(Service 1) logs every valid BTC TradingView alert into `tv_alert_events`
— DB as bus, share data not code. This poller (Service 2 daemon):

  1. picks up unprocessed rows once their trigger minute has closed,
     computes the cancel-flow display state at trigger (`classify_state`
     — the same single state function behind /cancelstate) plus the
     lookback-window state distribution ({"window": N} in the TV alert
     message JSON, default 90),
  2. pushes a simplified event card to Telegram (research aid, NOT a
     signal; the full two-stage card + inline buttons is A3 proper),
  3. backfills fwd mid returns at 30/60/120m — the same 判定窗 treatment
     as machine-detected `cancel_playbook_events`, so "human-drawn level"
     vs "machine-detected event" hit rates stay comparable (人的位置感
     alpha 檢定的資料基礎).

The legacy liquidity-event pipeline in /tv is untouched; this line only
reads its own table and never touches the executor / v7 pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.db import get_db_conn
from market_data.tasks.cancel_playbook_watcher import (
    DEF_VERSION, DIR_ZH, FIRST_HIT_ZH, GATE_SHOCK, STATE_META, _tg_creds,
    action_keyboard, classify_state, compute_features, first_hit_verdict,
    humanize_book, humanize_story, load_frame, state_color, verdict_mark)

logger = logging.getLogger(__name__)

POLL_SEC = 60
HORIZONS_MIN = (30, 60, 120)          # 判定窗 same as cancel_playbook_events
NO_DATA_GRACE_MIN = 10                # give collector this long before giving up
MAX_AGE_MIN = 24 * 60                 # older unprocessed rows → expired
STAGE2_MIN_AGE = 3                    # sweep 第2段最早觸發 (TODO spec 3-15min)
STAGE2_FORCE_AGE = 15                 # 15min 未回落也強制出第2段
STAGE2_MAX_AGE = 30                   # 超過=錯過窗口(停機等), 靜默關閉
_DIST_ORDER = ["calm", "rotation", "surge", "cascade", "absorption",
               "vacuum_up", "vacuum_down"]


def ensure_schema() -> None:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tv_alert_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                received_ms BIGINT NOT NULL,
                symbol VARCHAR(32) NOT NULL DEFAULT '',
                event VARCHAR(64) NOT NULL DEFAULT '',
                liquidity_side VARCHAR(8) NOT NULL DEFAULT '',
                price DOUBLE NULL,
                window_mins INT NOT NULL DEFAULT 90,
                raw_json TEXT NULL,
                processed TINYINT NOT NULL DEFAULT 0,
                state VARCHAR(16) NULL,
                trigger_mid DOUBLE NULL,
                fwd_ret_30m DOUBLE NULL,
                fwd_ret_60m DOUBLE NULL,
                fwd_ret_120m DOUBLE NULL,
                outcome_done TINYINT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_recv (received_ms),
                INDEX idx_todo (processed, outcome_done)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
            # sweep two-stage tracking (A3) + 對答案 reply (2026-07-20)
            for ddl in (
                "ALTER TABLE tv_alert_events "
                "ADD COLUMN stage2_done TINYINT NOT NULL DEFAULT 0",
                "ALTER TABLE tv_alert_events "
                "ADD COLUMN tg_message_id BIGINT NULL",
                "ALTER TABLE tv_alert_events "
                "ADD COLUMN outcome_replied TINYINT NOT NULL DEFAULT 0",
                # 2026-07-20: 存 sweep 第2段的 H-R 判讀（反轉/延續），供
                # 對答案 reply 引用「這張卡當初判的是什麼」
                "ALTER TABLE tv_alert_events "
                "ADD COLUMN hr_verdict VARCHAR(12) NULL",
                # 2026-07-21: first-hit-wins 平行診斷（見 cancel_playbook_
                # watcher.first_hit_verdict 凍結定義）
                "ALTER TABLE tv_alert_events "
                "ADD COLUMN first_hit_result VARCHAR(12) NULL",
            ):
                try:
                    cur.execute(ddl)
                except Exception:
                    pass      # 1060 duplicate column = already there
            # A3 半自動 eyeball log — 按鈕即日誌。Service 1 只 INSERT 判讀,
            # 本表由 Service 2 建 + 回填。UNIQUE(source, source_id) + INSERT
            # IGNORE = 首判鎖定不可事後改(凍結規則的機器強制版); skip 不落表。
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cancel_eyeball_log (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                source VARCHAR(12) NOT NULL,
                source_id BIGINT NOT NULL,
                event_ms BIGINT NOT NULL,
                card_state VARCHAR(24) NULL,
                card_direction VARCHAR(8) NULL,
                verdict VARCHAR(12) NOT NULL,
                verdict_ms BIGINT NOT NULL,
                fwd_ret_60m DOUBLE NULL,
                fwd_ret_120m DOUBLE NULL,
                outcome_done TINYINT NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uq_src (source, source_id),
                INDEX idx_out (outcome_done, event_ms)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        conn.commit()
        logger.info("tv_alert_events + cancel_eyeball_log tables ready")
    finally:
        conn.close()




def _q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return pd.DataFrame(cur.fetchall() or [])


def state_distribution(states: list[dict]) -> str:
    counts: dict[str, int] = {}
    for s in states:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    return " · ".join(f"{STATE_META[k][0]}{counts[k]}"
                      for k in _DIST_ORDER if counts.get(k))


def _row_head(row: dict) -> tuple[str, str, str, str]:
    """(t_tpe, label, side, px_s) — shared card header fields."""
    t = (pd.Timestamp(int(row["received_ms"]), unit="ms")
         + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
    label = str(row.get("event") or "level")
    side = str(row.get("liquidity_side") or "").strip()
    px = row.get("price")
    try:
        px_s = f"{float(px):,.0f}" if px else "?"
    except (TypeError, ValueError):
        px_s = "?"
    return t, label, side, px_s


def hr_flags(seg: pd.DataFrame, side: str) -> dict:
    """H-R 凍結雙旗標（TODO 2026-07-17 登記；顯示/日誌用，非信號）。

    (a) 被掃側淨回填: 被掃側 Σ(cancel−add) < 0（回補>撤離=供給重建）
    (b) 對側淨撤離:   對側   Σ(cancel−add) > 0（支撐/壓力抽走）
    side='buy' = BSL（上方流動性）被掃 = 向上掃 → 被掃側=ask、反轉=DOWN；
    side='sell' 鏡像。兩個 categorical 旗標、不調參。"""
    ask_net = float((seg["ac"] - seg["aa"]).sum())
    bid_net = float((seg["bc"] - seg["ba"]).sum())
    if side == "buy":
        refill, opp_pull, rev = ask_net < 0, bid_net > 0, "DOWN"
    else:
        refill, opp_pull, rev = bid_net < 0, ask_net > 0, "UP"
    return {"refill": refill, "opp_pull": opp_pull,
            "reversal": refill and opp_pull, "rev_dir": rev,
            "ask_net": ask_net, "bid_net": bid_net}


def _side_zh(side: str) -> str:
    """liquidity_side → 白話（2026-07-20 使用者定名：buy=BSL=買方流動性·高點
    / sell=SSL=賣方流動性·低點；? = 推斷）。"""
    s = side.rstrip("?")
    guess = "·推斷" if side.endswith("?") else ""
    if s == "buy":
        return f"買方流動性·高點{guess}"
    if s == "sell":
        return f"賣方流動性·低點{guess}"
    return ""


def format_stage1_card(row: dict) -> str:
    """sweep 二段式第 1 段：掃穿瞬間=資訊黑洞，只報事實、不判讀、無按鈕。"""
    t, label, side, px_s = _row_head(row)
    return "\n".join([
        "⚡ 掃穿事件·第1段：只報事實（研究·非信號）",
        f"關卡: {label} ({side}·{_side_zh(side)}) @ {px_s} | {t} TPE",
        "價格剛掃穿這個關卡，現在是瀑布瞬間——此刻的撤單都是止損保護的"
        "雜訊，看不出方向，先不判讀。",
        f"塵埃落定後（約 {STAGE2_MIN_AGE}-{STAGE2_FORCE_AGE} 分鐘）會自動"
        "送第2段判讀卡（含按鈕）",
    ])


def format_stage2_card(row: dict, flags: dict, cur: dict,
                       n_seg: int, age_min: float) -> str:
    """sweep 第 2 段：塵埃落定，H-R 旗標結論 + 四鍵——決策時刻在這裡。"""
    t, label, side, px_s = _row_head(row)
    a = "✓ 有（供給重建）" if flags["refill"] else "✗ 無"
    b = "✓ 有（另一側在讓路）" if flags["opp_pull"] else "✗ 無"
    concl = (f"反轉條件成立 → 預期價格收回關卡（方向 {flags['rev_dir']}）"
             if flags["reversal"]
             else "反轉條件未現 → 傾向延續掃穿方向（歷史基率：掃穿後約 2/3 延續）")
    return "\n".join([
        f"🧲 掃穿第2段·塵埃落定（觸發後 {age_min:.0f} 分鐘）",
        f"關卡: {label} ({side}·{_side_zh(side)}) @ {px_s} | 觸發 {t} TPE",
        "檢查兩個反轉條件（H-R 凍結旗標）:",
        f"① 被掃側掛單回補了嗎: {a}",
        f"② 對側掛單在撤退嗎: {b}（統計掃穿後 {n_seg} 分鐘）",
        f"結論: {concl}",
        f"當下狀態: {cur['emoji']} {cur['zh']} | 120 分鐘後自動對答案",
        f"def {DEF_VERSION} · 旗標=凍結定義 · 盲接反轉=-EV · 勿作交易依據",
    ])


def format_card(row: dict, cur: dict, feat_last: pd.Series,
                dist: str, n_lookback: int) -> str:
    """Simplified TV event card (pure text — plain, no parse_mode)."""
    t, label, side, px_s = _row_head(row)

    def g(k, spec):
        v = feat_last.get(k)
        try:
            return format(float(v), spec) if pd.notna(v) else "?"
        except (TypeError, ValueError):
            return "?"

    def num(k):
        v = feat_last.get(k)
        try:
            return float(v) if pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    state_line = f"{cur['emoji']} {cur['zh']}" + (
        f" → {DIR_ZH[cur['direction']]}" if cur["direction"] != "NONE" else "")
    lines = [
        "📍 你畫的關卡有動靜（研究·非信號）",
        f"關卡: {label}" + (f" ({side}·{_side_zh(side)})" if side else "")
        + f" @ {px_s} | {t} TPE",
        f"當下狀態: {state_line}",
        "發生了什麼: " + humanize_story(cur["state"], cur["direction"],
                                   num("vshock"), num("taker_ratio")),
        "掛單面: " + humanize_book(num("shock"), num("skew15"), num("net15")),
        f"回看{n_lookback}m: {dist or '無資料'}",
        "接下來: 120 分鐘自動對答案；你的判讀請按下面按鈕",
        f"原始值: shock {g('shock', '.1f')}x 毛 {g('skew15', '+.2f')}"
        f" 淨 {g('net15', '+.2f')} 量 {g('vshock', '.1f')}x"
        f" taker {g('taker_ratio', '+.2f')}",
        f"def {DEF_VERSION} · 勿作交易依據",
    ]
    return "\n".join(lines)


def _msg_id(resp) -> int:
    """message_id from a Telegram send response; 0 = sent but id unknown."""
    try:
        return int(resp.json()["result"]["message_id"])
    except Exception:
        return 0


def _send_tg(text: str, reply_markup: dict | None = None) -> int | None:
    """Returns message_id (int, 0=id unknown) on success, None on failure."""
    token, chat = _tg_creds()
    if not token or not chat:
        logger.warning("tv card skipped: TG creds missing")
        return None
    try:
        data = {"chat_id": chat, "text": text}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, timeout=15)
        return _msg_id(resp) if resp.status_code == 200 else None
    except Exception:
        logger.exception("tv card send failed")
        return None


def _send_tg_photo(caption: str, png: bytes,
                   reply_markup: dict | None = None) -> int | None:
    """Returns message_id (int, 0=id unknown) on success, None on failure."""
    token, chat = _tg_creds()
    if not token or not chat:
        return None
    try:
        data = {"chat_id": chat, "caption": caption[:1024]}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data=data,
            files={"photo": ("tv_zoom.png", png, "image/png")}, timeout=30)
        return _msg_id(resp) if resp.status_code == 200 else None
    except Exception:
        logger.exception("tv photo send failed")
        return None


def render_zoom_png(recv_ms: int, window: int, level_px: float | None,
                    win_feat: pd.DataFrame) -> bytes | None:
    """A2-3 sweep 特寫圖: 1m K 線 + 狀態色格 + 被掃價位線 + 觸發時刻線.

    plotly+kaleido (both in the marketdata image); any failure → None and
    the caller falls back to the text card. Right side extends only as far
    as data exists at render time (觸發後 ~1min) — the A3 second-stage card
    will own the post-event view."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        logger.info("zoom card: plotly unavailable, text fallback")
        return None
    try:
        t0 = recv_ms - window * 60_000
        t1 = min(int(time.time() * 1000), recv_ms + 90 * 60_000)
        resp = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": t0, "endTime": t1, "limit": 1000}, timeout=20)
        kl = resp.json()
        if not isinstance(kl, list) or len(kl) < 10:
            return None
        tpe = pd.Timedelta(hours=8)
        times = [pd.Timestamp(int(k[0]), unit="ms") + tpe for k in kl]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.88, 0.12], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=times,
            open=[float(k[1]) for k in kl], high=[float(k[2]) for k in kl],
            low=[float(k[3]) for k in kl], close=[float(k[4]) for k in kl],
            increasing_line_color="#26a269", decreasing_line_color="#e01b24",
            showlegend=False), row=1, col=1)
        xs, cs = [], []
        for m, r in win_feat.iterrows():
            col = state_color(classify_state(r))
            xs.append(pd.Timestamp(int(m) * 60_000, unit="ms") + tpe)
            cs.append(col or "#1c212b")     # calm=暗底格, 帶子連續可見
        fig.add_trace(go.Bar(x=xs, y=[1] * len(xs), marker_color=cs,
                             showlegend=False), row=2, col=1)
        if level_px:
            fig.add_hline(y=float(level_px), line_dash="dash",
                          line_color="#f2b544", row=1, col=1)
        fig.add_vline(x=pd.Timestamp(recv_ms, unit="ms") + tpe,
                      line_dash="dot", line_color="#f2b544")
        fig.update_layout(
            template="plotly_dark", width=900, height=520,
            margin=dict(l=45, r=25, t=40, b=25), showlegend=False,
            xaxis_rangeslider_visible=False,
            title=f"TV 快訊特寫 回看{window}m (TPE) · 色格=撤單狀態 · 研究非信號")
        fig.update_yaxes(visible=False, row=2, col=1)
        return fig.to_image(format="png", scale=2)
    except Exception:
        logger.exception("zoom card render failed")
        return None


def process_new(send: bool = True) -> int:
    """Handle unprocessed alerts whose trigger minute has closed.

    2026-07-20 修復重複發送 bug：寫入順序改成「先標記 processed=1 完成，
    DB 寫入成功才送 Telegram」。舊版先送訊息、最後才 UPDATE processed=1，
    若那次 DB 寫入失敗（連線問題/逾時），下一輪 poll（每 60s）的 SELECT
    WHERE processed=0 會再次撈到同一筆、重新送一次——理論上會一路重送到
    MAX_AGE_MIN。已用 process_stage2 的同款 bug 實測證實過（tv_alert_events
    某筆 stage2_done 從未寫成功，卡片卻在 16、21 分鐘各送一次）。寧可偶爾
    因 send 失敗漏發一次，也不要無限重複轟炸。每列包 try/except，一列
    出錯不影響同批其他列。
    """
    now_ms = int(time.time() * 1000)
    closed_floor = (now_ms // 60_000) * 60_000     # current minute start
    conn = get_db_conn()
    try:
        rows = _q(conn, "SELECT id, received_ms, symbol, event, "
                        "liquidity_side, price, window_mins "
                        "FROM tv_alert_events WHERE processed=0 "
                        "AND received_ms < %s ORDER BY id LIMIT 10",
                  (closed_floor,))
    finally:
        conn.close()
    if rows.empty:
        return 0

    done = 0
    for _, r in rows.iterrows():
        try:
            row = r.to_dict()
            rid = int(row["id"])
            recv_ms = int(row["received_ms"])
            age_min = (now_ms - recv_ms) / 60_000
            window = int(row["window_mins"] or 90)
            result_state, card, win_feat = "no_data", None, None

            if age_min > MAX_AGE_MIN:
                result_state = "expired"
            else:
                df = load_frame(lookback_min=int(window + 90 + age_min))
                if not df.empty:
                    feat = compute_features(df)
                    trig_min = recv_ms // 60_000
                    win = feat.loc[(feat.index > trig_min - window)
                                   & (feat.index <= trig_min)]
                    if not win.empty:
                        win_feat = win
                        states = [classify_state(x) for _, x in win.iterrows()]
                        cur_state = states[-1]
                        result_state = cur_state["state"]
                        card = format_card(row, cur_state, win.iloc[-1],
                                           state_distribution(states), len(win))
                if result_state == "no_data" and age_min < NO_DATA_GRACE_MIN:
                    continue      # collector may just be lagging — retry

            side = str(row.get("liquidity_side") or "").strip()
            stage1, png, is_sweep = None, None, False
            if card and send:
                # 2026-07-18 使用者定版: 快訊價位即其標記的獵取位 → 無 side 但
                # 有價格一律視為 sweep, 方向由「觸發價 vs 觸發前 mid」推斷,
                # 存 buy?/sell?(與明示聲明可分層), 同走二段式+H-R 旗標
                if (side not in ("buy", "sell") and row.get("price")
                        and win_feat is not None
                        and win_feat["mid"].notna().any()):
                    try:
                        mids = win_feat["mid"].dropna()
                        ref = float(mids.iloc[-6] if len(mids) >= 6
                                    else mids.iloc[0])
                        side = ("buy" if float(row["price"]) >= ref
                                else "sell") + "?"
                        row["liquidity_side"] = side
                        conn2 = get_db_conn()
                        try:
                            with conn2.cursor() as cur2:
                                cur2.execute("UPDATE tv_alert_events SET "
                                             "liquidity_side=%s WHERE id=%s",
                                             (side, rid))
                            conn2.commit()
                        finally:
                            conn2.close()
                    except (TypeError, ValueError):
                        side = ""
                if side.rstrip("?") in ("buy", "sell"):
                    # sweep 二段式第 1 段: 只報事實+特寫圖, 無按鈕 —
                    # 行動鍵在第 2 段 (process_stage2, 塵埃落定後)
                    is_sweep = True
                    stage1 = format_stage1_card(row)
                    if win_feat is not None:
                        try:
                            px = (float(row["price"])
                                  if row.get("price") else None)
                        except (TypeError, ValueError):
                            px = None
                        png = render_zoom_png(recv_ms, window, px, win_feat)

            # 先寫 DB 標記完成，寫入成功才送訊息（見上方 docstring）
            conn3 = get_db_conn()
            try:
                with conn3.cursor() as cur3:
                    cur3.execute(
                        "UPDATE tv_alert_events SET processed=1, state=%s "
                        "WHERE id=%s", (result_state, rid))
                conn3.commit()
            finally:
                conn3.close()

            if card and send:
                if is_sweep:
                    sent_mid = _send_tg_photo(stage1, png) if png else None
                    if sent_mid is None:
                        sent_mid = _send_tg(stage1)
                else:
                    # 純關卡快訊: 單段卡 + 行動鍵
                    sent_mid = _send_tg(card, action_keyboard("tv", rid))
                if sent_mid:
                    conn4 = get_db_conn()
                    try:
                        with conn4.cursor() as cur4:
                            cur4.execute(
                                "UPDATE tv_alert_events SET "
                                "tg_message_id=COALESCE(%s, tg_message_id) "
                                "WHERE id=%s", (sent_mid, rid))
                        conn4.commit()
                    finally:
                        conn4.close()
            done += 1
        except Exception:
            logger.exception("process_new failed for one row (id=%s), "
                             "will retry next cycle", row.get("id")
                             if isinstance(row, dict) else "?")
    return done


def process_stage2(send: bool = True) -> int:
    """sweep 第 2 段：強度回落（或 15min 強制）後推 H-R 旗標結論卡 + 四鍵。"""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        rows = _q(conn, "SELECT id, received_ms, event, liquidity_side, "
                        "price, window_mins FROM tv_alert_events "
                        "WHERE processed=1 AND stage2_done=0 "
                        "AND liquidity_side IN ('buy','sell','buy?','sell?') "
                        "AND received_ms <= %s ORDER BY id LIMIT 5",
                  (now_ms - STAGE2_MIN_AGE * 60_000,))
    finally:
        conn.close()
    if rows.empty:
        return 0

    done = 0
    for _, r in rows.iterrows():
        try:
            row = r.to_dict()
            rid = int(row["id"])
            recv_ms = int(row["received_ms"])
            age_min = (now_ms - recv_ms) / 60_000
            mark_done, card, kb, png, flags = False, None, None, None, None

            if age_min > STAGE2_MAX_AGE:
                mark_done = True                  # missed window (downtime)
            else:
                df = load_frame(lookback_min=int(age_min + 150))
                if not df.empty:
                    feat = compute_features(df)
                    trig_min = recv_ms // 60_000
                    seg = feat.loc[feat.index > trig_min]
                    if not seg.empty:
                        last = seg.iloc[-1]
                        calmed = (np.isfinite(last["shock"])
                                  and last["shock"] < GATE_SHOCK
                                  and (not np.isfinite(last["vshock"])
                                       or last["vshock"] < 3.0))
                        if calmed or age_min >= STAGE2_FORCE_AGE:
                            flags = hr_flags(
                                seg, str(row["liquidity_side"]).rstrip("?"))
                            card = format_stage2_card(
                                row, flags, classify_state(last), len(seg),
                                age_min)
                            kb = action_keyboard("tv", rid)
                            if send:
                                try:
                                    px = (float(row["price"])
                                          if row.get("price") else None)
                                except (TypeError, ValueError):
                                    px = None
                                png = render_zoom_png(
                                    recv_ms, int(row["window_mins"] or 90),
                                    px, seg)
                            mark_done = True
                if not mark_done and age_min >= STAGE2_FORCE_AGE + 5:
                    mark_done = True              # depth data missing — give up

            if not mark_done:
                continue      # 還在等塵埃落定，下一輪再檢查，不寫 DB 不送訊息

            # 2026-07-20 修復重複發送 bug：這裡原本的 UPDATE 是 3 個 %s
            # 佔位符但只傳 2 個參數（漏了 hr_verdict）——pymysql 每次執行
            # 必定拋例外，導致 stage2_done 從來沒寫成功過，下一輪 poll 又
            # 撈到同一筆重送一次（實測：同一事件 16 分鐘、21 分鐘各送一次）。
            # 順便把寫入順序也反過來：先寫 DB 標記完成，成功才送訊息——
            # DB 若再出問題，寧可漏發一次也不要無限重複轟炸。
            hr_verdict = ("reversal" if flags is not None and flags["reversal"]
                          else ("continuation" if flags is not None else None))
            conn2 = get_db_conn()
            try:
                with conn2.cursor() as cur2:
                    cur2.execute(
                        "UPDATE tv_alert_events SET stage2_done=1, "
                        "hr_verdict=COALESCE(%s, hr_verdict) WHERE id=%s",
                        (hr_verdict, rid))
                conn2.commit()
            finally:
                conn2.close()

            if card and send:
                sent_mid = _send_tg_photo(card, png, kb) if png else None
                if sent_mid is None:
                    sent_mid = _send_tg(card, kb)
                if sent_mid:
                    # 第2段卡的 message_id 覆蓋第1段 → 對答案 reply 錨最新卡
                    conn3 = get_db_conn()
                    try:
                        with conn3.cursor() as cur3:
                            cur3.execute(
                                "UPDATE tv_alert_events SET "
                                "tg_message_id=COALESCE(%s, tg_message_id) "
                                "WHERE id=%s", (sent_mid, rid))
                        conn3.commit()
                    finally:
                        conn3.close()
            done += 1
        except Exception:
            logger.exception("process_stage2 failed for one row (id=%s), "
                             "will retry next cycle", row.get("id")
                             if isinstance(row, dict) else "?")
    return done


def backfill_eyeball() -> None:
    """判定窗到期 → 回填 cancel_eyeball_log 的 fwd mid returns（60/120m）。
    與機器事件同一條回填哲學：人的判讀與機器判讀用同一把尺量結果。"""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, event_ms FROM cancel_eyeball_log "
                       "WHERE outcome_done=0 AND event_ms <= %s",
                 (now_ms - 121 * 60_000,))
        if due.empty:
            return
        due["event_ms"] = pd.to_numeric(due["event_ms"])
        lo = int(due["event_ms"].min()) - 2 * 60_000
        hi = int(due["event_ms"].max()) + 122 * 60_000
        ob = _q(conn, "SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' "
                      "AND ts_ms BETWEEN %s AND %s ORDER BY ts_ms", (lo, hi))
        if ob.empty:
            return
        ob["ts_ms"] = pd.to_numeric(ob["ts_ms"])
        ob["mid"] = pd.to_numeric(ob["mid"])
        mids = ob.groupby(ob["ts_ms"] // 60_000)["mid"].last()
        with conn.cursor() as cur:
            for _, r in due.iterrows():
                m0 = int(r["event_ms"]) // 60_000
                if m0 not in mids.index:
                    cur.execute("UPDATE cancel_eyeball_log SET outcome_done=1 "
                                "WHERE id=%s", (int(r["id"]),))
                    continue
                base = float(mids.loc[m0])
                sets, vals = [], []
                for h in (60, 120):
                    if m0 + h in mids.index:
                        sets.append(f"fwd_ret_{h}m=%s")
                        vals.append(float(mids.loc[m0 + h] / base - 1))
                sets.append("outcome_done=1")
                vals.append(int(r["id"]))
                cur.execute("UPDATE cancel_eyeball_log SET " + ", ".join(sets)
                            + " WHERE id=%s", vals)
        conn.commit()
    except Exception:
        logger.exception("eyeball outcome backfill failed")
    finally:
        conn.close()


def backfill_outcomes_tv() -> None:
    """Fill fwd mid returns for processed alerts past the 120m 判定窗.

    2026-07-21: SELECT 放寬成「outcome_done=0 或 first_hit_result 缺」，
    讓已完成判定窗的舊事件（含今天早一點的 sweep 事件）也能被撈回來補算
    first-hit-wins 平行診斷（見 cancel_playbook_watcher.first_hit_verdict
    凍結定義），不用另寫一次性回填腳本。只有帶 hr_verdict 的 sweep 事件才
    有方向可判；純關卡快訊維持 first_hit_result 空白。"""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, received_ms ms, price, liquidity_side, "
                       "hr_verdict FROM tv_alert_events WHERE processed=1 "
                       "AND (outcome_done=0 OR first_hit_result IS NULL) "
                       "AND state NOT IN ('expired','no_data') "
                       "AND received_ms <= %s",
                 (now_ms - (HORIZONS_MIN[-1] + 1) * 60_000,))
        if due.empty:
            return
        due["ms"] = pd.to_numeric(due["ms"])
        lo = int(due["ms"].min()) - 2 * 60_000
        hi = int(due["ms"].max()) + (HORIZONS_MIN[-1] + 2) * 60_000
        ob = _q(conn, "SELECT ts_ms, mid_price mid FROM orderbook_snapshots_1m "
                      "WHERE canonical_symbol='BTC-USD' "
                      "AND ts_ms BETWEEN %s AND %s ORDER BY ts_ms", (lo, hi))
        if ob.empty:
            return
        ob["ts_ms"] = pd.to_numeric(ob["ts_ms"])
        ob["mid"] = pd.to_numeric(ob["mid"])
        mids = ob.groupby(ob["ts_ms"] // 60_000)["mid"].last()
        with conn.cursor() as cur:
            for _, r in due.iterrows():
                m0 = int(r["ms"]) // 60_000
                base = mids.loc[m0] if m0 in mids.index else (
                    float(r["price"]) if r["price"] else None)
                if not base:
                    cur.execute("UPDATE tv_alert_events SET outcome_done=1 "
                                "WHERE id=%s", (int(r["id"]),))
                    continue
                sets, vals = ["trigger_mid=%s"], [float(base)]
                for h in HORIZONS_MIN:
                    if m0 + h in mids.index:
                        sets.append(f"fwd_ret_{h}m=%s")
                        vals.append(float(mids.loc[m0 + h] / base - 1))
                sets.append("outcome_done=1")

                verdict = r.get("hr_verdict")
                if verdict in ("reversal", "continuation"):
                    d = hr_call_direction(str(r.get("liquidity_side") or ""),
                                          verdict)
                    fwd_mids = mids.loc[(mids.index > m0)
                                        & (mids.index <= m0 + 120)].tolist()
                    fh = first_hit_verdict(float(base), d, fwd_mids)
                    if fh:
                        sets.append("first_hit_result=%s")
                        vals.append(fh)

                vals.append(int(r["id"]))
                cur.execute("UPDATE tv_alert_events SET " + ", ".join(sets)
                            + " WHERE id=%s", vals)
        conn.commit()
    except Exception:
        logger.exception("tv alert outcome backfill failed")
    finally:
        conn.close()


def hr_call_direction(side: str, verdict: str) -> str:
    """side='buy'(BSL 向上掃)+verdict='reversal'→DOWN, 'continuation'→UP；
    side='sell' 鏡像。與 hr_flags 的方向語意保持一致（單一來源）。"""
    up_side = str(side).rstrip("?") == "buy"
    if verdict == "reversal":
        return "DOWN" if up_side else "UP"
    return "UP" if up_side else "DOWN"


def format_tv_outcome_reply(row: dict) -> str:
    """TV 事件「對答案」reply 文字。若當初 sweep 第2段有判讀（反轉/延續），
    先講「這張卡原本判什麼」，再報實際走勢（2026-07-20 使用者反饋加上）。
    2026-07-21 補上醒目的「判讀結果」標頭行(對/錯一眼可見，以 60m 為主、
    無 60m 才退 120m)——之前只丟走勢數字，讀者要自己心算比對方向。無
    hr_verdict(純關卡快訊，沒有方向宣告)則不判對錯，維持只報走勢。"""
    t = (pd.Timestamp(int(row["received_ms"]), unit="ms")
         + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
    label = str(row.get("event") or "level")

    verdict = row.get("hr_verdict")
    d = None
    if verdict in ("reversal", "continuation"):
        d = hr_call_direction(row.get("liquidity_side") or "", verdict)
    m60 = verdict_mark(d, row.get("fwd_ret_60m")) if d else ""
    m120 = verdict_mark(d, row.get("fwd_ret_120m")) if d else ""

    parts = []
    for h, m in ((30, ""), (60, m60), (120, m120)):
        v = row.get(f"fwd_ret_{h}m")
        if v is not None:
            parts.append(f"{h}m {float(v):+.2%}" + (f" {m}" if m else ""))

    lines = [f"↩️ 對答案: 關卡 {label}（{t} 觸發）"]
    if verdict in ("reversal", "continuation"):
        call_zh = "反轉" if verdict == "reversal" else "延續"
        lines.append(f"原始判讀: {call_zh}（預期 {DIR_ZH.get(d, d)}）")
        if m60 or m120:
            lines.append(f"判讀結果: {m60 or m120}"
                         f"{'（以 60m 為準）' if m60 else '（以 120m 為準）'}")
    lines.append("之後走勢: " + (" · ".join(parts) if parts else "資料不足"))

    fh = row.get("first_hit_result")
    if fh in FIRST_HIT_ZH:
        lines.append(f"先觸價判斷(±0.5%): {FIRST_HIT_ZH[fh]}（另一把尺，見說明）")

    lines.append("(研究對照 · 勿作交易依據)")
    return "\n".join(lines)


def reply_outcomes_tv() -> None:
    """判定窗回填完成 → reply 原事件卡（同 watcher.reply_outcomes 哲學）。"""
    token, chat = _tg_creds()
    if not token or not chat:
        return
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, received_ms, event, liquidity_side, "
                       "hr_verdict, first_hit_result, fwd_ret_30m, "
                       "fwd_ret_60m, fwd_ret_120m, tg_message_id "
                       "FROM tv_alert_events "
                       "WHERE outcome_done=1 AND outcome_replied=0 "
                       "AND tg_message_id IS NOT NULL AND tg_message_id > 0 "
                       "LIMIT 10")
        if due.empty:
            return
        for _, r in due.iterrows():
            row = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            permanent = False
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={"chat_id": chat,
                          "text": format_tv_outcome_reply(row),
                          "reply_to_message_id": int(row["tg_message_id"]),
                          "allow_sending_without_reply": "true"},
                    timeout=15)
                permanent = (resp.status_code == 200
                             or 400 <= resp.status_code < 500)
            except Exception:
                logger.exception("tv outcome reply failed")
            if permanent:
                with conn.cursor() as cur:
                    cur.execute("UPDATE tv_alert_events SET outcome_replied=1 "
                                "WHERE id=%s", (int(row["id"]),))
                conn.commit()
    finally:
        conn.close()


def poll_loop() -> None:
    ensure_schema()
    logger.info("tv-alert poller started (def=%s)", DEF_VERSION)
    while True:
        try:
            process_new()
            process_stage2()
            backfill_outcomes_tv()
            reply_outcomes_tv()
            backfill_eyeball()
        except Exception:
            logger.exception("tv alert poll cycle failed")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ensure_schema()
    n = process_new(send=False)
    n2 = process_stage2(send=False)
    backfill_outcomes_tv()
    backfill_eyeball()
    print(f"one-shot: processed {n} alerts, {n2} stage2 (send=False)")
