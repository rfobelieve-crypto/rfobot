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
    DEF_VERSION, GATE_SHOCK, STATE_META, _tg_creds, classify_state,
    compute_features, load_frame, state_color, verdict_keyboard)

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
            # sweep two-stage tracking (A3, 2026-07-19) — additive column
            try:
                cur.execute("ALTER TABLE tv_alert_events "
                            "ADD COLUMN stage2_done TINYINT NOT NULL DEFAULT 0")
            except Exception:
                pass          # 1060 duplicate column = already there
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


def format_stage1_card(row: dict) -> str:
    """sweep 二段式第 1 段：掃穿瞬間=資訊黑洞，只報事實、不判讀、無按鈕。"""
    t, label, side, px_s = _row_head(row)
    return "\n".join([
        "⚡ 掃穿事件·第1段（研究·非信號）",
        f"關卡: {label} ({side}) @ {px_s} | {t} TPE",
        "瀑布中＝資訊黑洞（撤單全是保護性雜訊）——判讀待塵埃落定",
        f"第2段卡（H-R 旗標＋判讀鍵）將於強度回落後 {STAGE2_MIN_AGE}-"
        f"{STAGE2_FORCE_AGE}min 自動送達",
    ])


def format_stage2_card(row: dict, flags: dict, cur: dict,
                       n_seg: int, age_min: float) -> str:
    """sweep 第 2 段：塵埃落定，H-R 旗標結論 + 四鍵——決策時刻在這裡。"""
    t, label, side, px_s = _row_head(row)
    a = "✓" if flags["refill"] else "✗"
    b = "✓" if flags["opp_pull"] else "✗"
    concl = (f"反轉條件成立 → 預期回收關卡（{flags['rev_dir']}）"
             if flags["reversal"]
             else "反轉條件未現 → 傾向延續（掃穿延續基率 ~2/3）")
    return "\n".join([
        f"🧲 掃穿第2段·塵埃落定（+{age_min:.0f}m）",
        f"關卡: {label} ({side}) @ {px_s} | 觸發 {t} TPE",
        f"H-R 旗標: 被掃側淨回填 {a} · 對側淨撤離 {b}（段 n={n_seg}m）",
        f"結論: {concl}",
        f"當前狀態: {cur['emoji']} {cur['zh']} | 判定窗 120m 自動回填",
        f"def {DEF_VERSION} · 旗標=凍結定義 · 盲接反轉=-EV · 勿作交易依據",
    ])


def format_card(row: dict, cur: dict, feat_last: pd.Series,
                dist: str, n_lookback: int) -> str:
    """Simplified TV event card (pure text — plain, no parse_mode)."""
    t = (pd.Timestamp(int(row["received_ms"]), unit="ms")
         + pd.Timedelta(hours=8)).strftime("%m-%d %H:%M")
    label = str(row.get("event") or "level")
    side = str(row.get("liquidity_side") or "").strip()
    px = row.get("price")
    px_s = f"{float(px):,.0f}" if px else "?"

    def g(k, spec):
        v = feat_last.get(k)
        try:
            return format(float(v), spec) if pd.notna(v) else "?"
        except (TypeError, ValueError):
            return "?"

    cur_label = cur["zh"] + (f"→{cur['direction']}"
                             if cur["direction"] != "NONE" else "")
    lines = [
        "📍 TV 快訊事件卡（研究·非信號）",
        f"關卡: {label}" + (f" ({side})" if side else "") + f" @ {px_s} | {t} TPE",
        f"觸發時狀態: {cur['emoji']} {cur_label} | shock {g('shock', '.1f')}x"
        f" 毛 {g('skew15', '+.2f')} 淨 {g('net15', '+.2f')}"
        f" 量 {g('vshock', '.1f')}x taker {g('taker_ratio', '+.2f')}",
        f"回看{n_lookback}m: {dist or '無資料'}",
        f"判定窗 120m 自動回填 · def {DEF_VERSION} · 勿作交易依據",
    ]
    return "\n".join(lines)


def _send_tg(text: str, reply_markup: dict | None = None) -> bool:
    token, chat = _tg_creds()
    if not token or not chat:
        logger.warning("tv card skipped: TG creds missing")
        return False
    try:
        data = {"chat_id": chat, "text": text}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, timeout=15)
        return resp.status_code == 200
    except Exception:
        logger.exception("tv card send failed")
        return False


def _send_tg_photo(caption: str, png: bytes,
                   reply_markup: dict | None = None) -> bool:
    token, chat = _tg_creds()
    if not token or not chat:
        return False
    try:
        data = {"chat_id": chat, "caption": caption[:1024]}
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data=data,
            files={"photo": ("tv_zoom.png", png, "image/png")}, timeout=30)
        return resp.status_code == 200
    except Exception:
        logger.exception("tv photo send failed")
        return False


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
            cs.append(col or "rgba(0,0,0,0)")
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
    """Handle unprocessed alerts whose trigger minute has closed."""
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
                continue          # collector may just be lagging — retry

        if card and send:
            side = str(row.get("liquidity_side") or "").strip()
            if side in ("buy", "sell"):
                # sweep 二段式第 1 段: 只報事實+特寫圖, 無按鈕 —
                # 判讀鍵在第 2 段 (process_stage2, 塵埃落定後)
                stage1 = format_stage1_card(row)
                png = None
                if win_feat is not None:
                    try:
                        px = float(row["price"]) if row.get("price") else None
                    except (TypeError, ValueError):
                        px = None
                    png = render_zoom_png(recv_ms, window, px, win_feat)
                if not (png and _send_tg_photo(stage1, png)):
                    _send_tg(stage1)
            else:
                # 純關卡快訊: 單段卡 + 四鍵按鈕即日誌
                _send_tg(card, verdict_keyboard("tv", rid))

        conn = get_db_conn()
        try:
            with conn.cursor() as cur2:
                cur2.execute(
                    "UPDATE tv_alert_events SET processed=1, state=%s "
                    "WHERE id=%s", (result_state, rid))
            conn.commit()
        finally:
            conn.close()
        done += 1
    return done


def process_stage2(send: bool = True) -> int:
    """sweep 第 2 段：強度回落（或 15min 強制）後推 H-R 旗標結論卡 + 四鍵。"""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        rows = _q(conn, "SELECT id, received_ms, event, liquidity_side, "
                        "price, window_mins FROM tv_alert_events "
                        "WHERE processed=1 AND stage2_done=0 "
                        "AND liquidity_side IN ('buy','sell') "
                        "AND received_ms <= %s ORDER BY id LIMIT 5",
                  (now_ms - STAGE2_MIN_AGE * 60_000,))
    finally:
        conn.close()
    if rows.empty:
        return 0

    done = 0
    for _, r in rows.iterrows():
        row = r.to_dict()
        rid = int(row["id"])
        recv_ms = int(row["received_ms"])
        age_min = (now_ms - recv_ms) / 60_000
        mark_done, card, kb, png = False, None, None, None

        if age_min > STAGE2_MAX_AGE:
            mark_done = True                      # missed window (downtime)
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
                        flags = hr_flags(seg, str(row["liquidity_side"]))
                        card = format_stage2_card(
                            row, flags, classify_state(last), len(seg), age_min)
                        kb = verdict_keyboard("tv", rid)
                        if send:
                            try:
                                px = (float(row["price"])
                                      if row.get("price") else None)
                            except (TypeError, ValueError):
                                px = None
                            png = render_zoom_png(
                                recv_ms, int(row["window_mins"] or 90), px, seg)
                        mark_done = True
            if not mark_done and age_min >= STAGE2_FORCE_AGE + 5:
                mark_done = True                  # depth data missing — give up

        if card and send:
            if not (png and _send_tg_photo(card, png, kb)):
                _send_tg(card, kb)
        if mark_done:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur2:
                    cur2.execute("UPDATE tv_alert_events SET stage2_done=1 "
                                 "WHERE id=%s", (rid,))
                conn.commit()
            finally:
                conn.close()
            done += 1
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
    """Fill fwd mid returns for processed alerts past the 120m 判定窗."""
    now_ms = int(time.time() * 1000)
    conn = get_db_conn()
    try:
        due = _q(conn, "SELECT id, received_ms ms, price "
                       "FROM tv_alert_events WHERE processed=1 "
                       "AND outcome_done=0 AND state NOT IN ('expired','no_data') "
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
                vals.append(int(r["id"]))
                cur.execute("UPDATE tv_alert_events SET " + ", ".join(sets)
                            + " WHERE id=%s", vals)
        conn.commit()
    except Exception:
        logger.exception("tv alert outcome backfill failed")
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
