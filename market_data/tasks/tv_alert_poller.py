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

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.db import get_db_conn
from market_data.tasks.cancel_playbook_watcher import (
    DEF_VERSION, STATE_META, _tg_creds, classify_state, compute_features,
    load_frame)

logger = logging.getLogger(__name__)

POLL_SEC = 60
HORIZONS_MIN = (30, 60, 120)          # 判定窗 same as cancel_playbook_events
NO_DATA_GRACE_MIN = 10                # give collector this long before giving up
MAX_AGE_MIN = 24 * 60                 # older unprocessed rows → expired
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
        conn.commit()
        logger.info("tv_alert_events table ready")
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


def _send_tg(text: str) -> bool:
    token, chat = _tg_creds()
    if not token or not chat:
        logger.warning("tv card skipped: TG creds missing")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text}, timeout=15)
        return resp.status_code == 200
    except Exception:
        logger.exception("tv card send failed")
        return False


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
        result_state, card = "no_data", None

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
                    states = [classify_state(x) for _, x in win.iterrows()]
                    cur_state = states[-1]
                    result_state = cur_state["state"]
                    card = format_card(row, cur_state, win.iloc[-1],
                                       state_distribution(states), len(win))
            if result_state == "no_data" and age_min < NO_DATA_GRACE_MIN:
                continue          # collector may just be lagging — retry

        if card and send:
            _send_tg(card)

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
            backfill_outcomes_tv()
        except Exception:
            logger.exception("tv alert poll cycle failed")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ensure_schema()
    n = process_new(send=False)
    backfill_outcomes_tv()
    print(f"one-shot: processed {n} alerts (send=False)")
