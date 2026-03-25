"""
純事件追蹤模組 — 流動性掃蕩 outcome tracker（不含 delta）

設計原則：
- 與現有 current_event 流程完全獨立
- 支援多事件同時追蹤
- 每個事件有 15m 和 1h 兩個觀察窗口
- first hit wins（±0.5%）
- 不追蹤 order flow / delta
"""

import uuid
import time
import threading
import logging
import pymysql

from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

tz_taipei = timezone(timedelta(hours=8))

# =========================================================
# 設定
# =========================================================
OUTCOME_THRESHOLD = 0.005  # ±0.5%

OBSERVATION_WINDOWS = {
    "15m": 15 * 60,   # 900 秒
    "1h":  60 * 60,   # 3600 秒
}

# =========================================================
# 狀態
# =========================================================
tracker_lock = threading.Lock()
active_trackers: list[dict] = []

# 已完成但尚未被主程式取走的 tracker 摘要
_finished_summaries_lock = threading.Lock()
_finished_summaries: list[str] = []


# =========================================================
# 資料表初始化
# =========================================================
def init_sweep_outcomes_table(get_db_conn_func):
    """建立 sweep_outcomes 資料表。傳入 get_db_conn 函式避免循環 import。"""
    conn = get_db_conn_func()
    try:
        sql = """
        CREATE TABLE IF NOT EXISTS sweep_outcomes (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            event_uuid VARCHAR(64) NOT NULL,
            symbol VARCHAR(50) NOT NULL,
            event_type VARCHAR(10) NOT NULL COMMENT 'BSL or SSL',
            event_price DECIMAL(18,8) NOT NULL,
            upper_target DECIMAL(18,8) NOT NULL,
            lower_target DECIMAL(18,8) NOT NULL,
            trigger_ts INT NOT NULL,
            trigger_time_text VARCHAR(32),
            tv_time VARCHAR(64),

            -- 15 分鐘觀察窗口
            outcome_15m VARCHAR(20) DEFAULT 'pending'
                COMMENT 'reversal / continuation / unresolved',
            outcome_15m_hit_price DECIMAL(18,8) DEFAULT NULL,
            outcome_15m_hit_ts INT DEFAULT NULL,
            outcome_15m_latency_s INT DEFAULT NULL,

            -- 1 小時觀察窗口
            outcome_1h VARCHAR(20) DEFAULT 'pending'
                COMMENT 'reversal / continuation / unresolved',
            outcome_1h_hit_price DECIMAL(18,8) DEFAULT NULL,
            outcome_1h_hit_ts INT DEFAULT NULL,
            outcome_1h_latency_s INT DEFAULT NULL,

            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            UNIQUE KEY uk_event_uuid (event_uuid),
            INDEX idx_symbol (symbol),
            INDEX idx_event_type (event_type),
            INDEX idx_trigger_ts (trigger_ts)
        );
        """
        with conn.cursor() as cursor:
            cursor.execute(sql)
        logger.info("sweep_outcomes table ready")
    finally:
        conn.close()


# =========================================================
# Tracker 建立
# =========================================================
def create_tracker(event_type_raw: str, liquidity_side: str,
                   price: float, symbol: str, tv_time: str) -> dict:
    """
    建立一個新的 outcome tracker。

    Parameters:
        event_type_raw: TradingView 原始 event 名稱
        liquidity_side: "buy" (BSL) 或 "sell" (SSL)
        price: 掃蕩觸發價格
        symbol: 幣種
        tv_time: TradingView 時間字串
    """
    # 映射 liquidity_side → BSL / SSL
    if liquidity_side == "buy":
        event_type = "BSL"
    elif liquidity_side == "sell":
        event_type = "SSL"
    else:
        event_type = liquidity_side.upper()

    upper_target = price * (1 + OUTCOME_THRESHOLD)
    lower_target = price * (1 - OUTCOME_THRESHOLD)
    now_ts = int(time.time())

    tracker = {
        "event_uuid": str(uuid.uuid4()),
        "symbol": symbol,
        "event_type": event_type,
        "event_price": price,
        "upper_target": upper_target,
        "lower_target": lower_target,
        "trigger_ts": now_ts,
        "trigger_time_text": datetime.now(tz_taipei).strftime("%Y-%m-%d %H:%M:%S"),
        "tv_time": tv_time,

        # 每個觀察窗口的狀態
        "windows": {},
    }

    for window_name, window_seconds in OBSERVATION_WINDOWS.items():
        tracker["windows"][window_name] = {
            "seconds": window_seconds,
            "outcome": "pending",      # pending → reversal / continuation / unresolved
            "hit_price": None,
            "hit_ts": None,
            "latency_s": None,
        }

    return tracker


# =========================================================
# 價格更新（由 on_message 呼叫）
# =========================================================
def feed_price(price: float, trade_ts: int):
    """
    餵入最新價格，檢查所有 active tracker 的 hit 狀態。
    應在 WebSocket on_message 中呼叫（BTC 成交時）。
    """
    with tracker_lock:
        for tracker in active_trackers:
            _check_hits(tracker, price, trade_ts)


def _check_hits(tracker: dict, price: float, trade_ts: int):
    """檢查單一 tracker 的所有 pending 窗口是否被 hit。"""
    # 忽略 tracker 建立之前的成交
    if trade_ts < tracker["trigger_ts"]:
        return

    now_ts = int(time.time())
    age = now_ts - tracker["trigger_ts"]

    for window_name, window in tracker["windows"].items():
        if window["outcome"] != "pending":
            continue

        # 超過觀察時間 → unresolved
        if age > window["seconds"]:
            window["outcome"] = "unresolved"
            continue

        # first hit wins
        hit_side = None
        if price >= tracker["upper_target"]:
            hit_side = "upper"
        elif price <= tracker["lower_target"]:
            hit_side = "lower"

        if not hit_side:
            continue

        # 判定 outcome
        window["outcome"] = _classify_outcome(tracker["event_type"], hit_side)
        window["hit_price"] = price
        window["hit_ts"] = trade_ts
        window["latency_s"] = max(0, trade_ts - tracker["trigger_ts"])


def _classify_outcome(event_type: str, hit_side: str) -> str:
    """
    BSL（掃高點）：先跌 = reversal，先漲 = continuation
    SSL（掃低點）：先漲 = reversal，先跌 = continuation
    """
    if event_type == "BSL":
        return "reversal" if hit_side == "lower" else "continuation"
    elif event_type == "SSL":
        return "reversal" if hit_side == "upper" else "continuation"
    return "unknown"


# =========================================================
# 供主程式取走合併通知
# =========================================================
def get_latest_finished_summary() -> str | None:
    """
    取走最新一筆完成的 tracker 摘要。
    被取走後就不會重複發送（fallback thread 會發現已被移除）。
    """
    with _finished_summaries_lock:
        if _finished_summaries:
            return _finished_summaries.pop(0)
    return None


# =========================================================
# 註冊新事件
# =========================================================
def register_event(event_type_raw: str, liquidity_side: str,
                   price: float, symbol: str, tv_time: str) -> dict:
    """建立 tracker 並加入 active 列表。回傳 tracker 供通知用。"""
    tracker = create_tracker(event_type_raw, liquidity_side, price, symbol, tv_time)

    with tracker_lock:
        active_trackers.append(tracker)

    logger.info("Tracker registered: %s %s @ %.2f [%s]",
                tracker["event_type"], symbol, price, tracker["event_uuid"][:8])
    return tracker


# =========================================================
# Watchdog（定期檢查到期 & 存入 DB）
# =========================================================
def outcome_watchdog(get_db_conn_func, send_message_func, chat_id: str):
    """
    背景執行緒：每 2 秒掃一次 active_trackers。
    所有窗口都結束後（非 pending），存入 DB 並發送通知。
    """
    while True:
        try:
            finished = []

            with tracker_lock:
                now_ts = int(time.time())

                for tracker in active_trackers:
                    # 先處理超時的 pending 窗口
                    age = now_ts - tracker["trigger_ts"]
                    for window in tracker["windows"].values():
                        if window["outcome"] == "pending" and age > window["seconds"]:
                            window["outcome"] = "unresolved"

                    # 檢查是否全部窗口都結束
                    all_done = all(
                        w["outcome"] != "pending"
                        for w in tracker["windows"].values()
                    )
                    if all_done:
                        finished.append(tracker)

                for t in finished:
                    active_trackers.remove(t)

            # lock 外處理 DB 和通知
            for tracker in finished:
                try:
                    _save_to_db(tracker, get_db_conn_func)
                    logger.info("Tracker saved: %s", tracker["event_uuid"][:8])
                except Exception as e:
                    logger.exception("Failed to save tracker: %s", e)

                # 存入 buffer，供主程式 event_watchdog 合併通知
                summary = format_tracker_summary(tracker)
                with _finished_summaries_lock:
                    _finished_summaries.append(summary)

                # 若主程式的 delta 追蹤沒跑（被 skip 的情況），自己發
                # 主程式會透過 get_latest_finished_summary() 取走，
                # 5 秒後還沒被取走就自己發
                def _fallback_send(msg=summary):
                    time.sleep(5)
                    with _finished_summaries_lock:
                        if msg in _finished_summaries:
                            _finished_summaries.remove(msg)
                            send_message_func(chat_id, msg)

                threading.Thread(target=_fallback_send, daemon=True).start()

            time.sleep(2)

        except Exception as e:
            logger.exception("outcome_watchdog error: %s", e)
            time.sleep(5)


# =========================================================
# 中途通知（15m 窗口結束時）
# =========================================================
def check_interim_notifications(send_message_func, chat_id: str):
    """
    背景執行緒：每 2 秒檢查一次。
    當 15m 窗口剛結束（outcome 不再是 pending），且 1h 還在進行中，
    發送一次中途通知。
    """
    notified_15m: set[str] = set()

    while True:
        try:
            to_notify = []

            with tracker_lock:
                for tracker in active_trackers:
                    uid = tracker["event_uuid"]
                    w15 = tracker["windows"].get("15m", {})
                    w1h = tracker["windows"].get("1h", {})

                    if (uid not in notified_15m
                            and w15.get("outcome") != "pending"
                            and w1h.get("outcome") == "pending"):
                        to_notify.append(dict(tracker))
                        notified_15m.add(uid)

            for tracker in to_notify:
                try:
                    msg = format_15m_interim(tracker)
                    send_message_func(chat_id, msg)
                except Exception as e:
                    logger.exception("Failed to send 15m interim: %s", e)

            time.sleep(2)

        except Exception as e:
            logger.exception("check_interim_notifications error: %s", e)
            time.sleep(5)


# =========================================================
# 格式化
# =========================================================
def format_tracker_notification(tracker: dict) -> str:
    """事件開始通知。"""
    return (
        f"📍 流動性掃蕩事件追蹤啟動\n"
        f"類型: {tracker['event_type']}\n"
        f"幣種: {tracker['symbol']}\n"
        f"觸發價: {tracker['event_price']:.2f}\n"
        f"上目標 (+0.5%): {tracker['upper_target']:.2f}\n"
        f"下目標 (-0.5%): {tracker['lower_target']:.2f}\n"
        f"觀察窗口: 15m / 1h\n"
        f"UUID: {tracker['event_uuid'][:8]}"
    )


def format_15m_interim(tracker: dict) -> str:
    """15 分鐘窗口結束的中途通知。"""
    w = tracker["windows"]["15m"]
    hit_info = ""
    if w["hit_price"] is not None:
        hit_info = (
            f"命中價: {w['hit_price']:.2f}\n"
            f"延遲: {w['latency_s']}s"
        )
    else:
        hit_info = "未命中目標"

    return (
        f"⏱ 15 分鐘結果 [{tracker['event_type']} {tracker['symbol']}]\n"
        f"結果: {w['outcome']}\n"
        f"{hit_info}\n"
        f"1h 窗口仍在追蹤中...\n"
        f"UUID: {tracker['event_uuid'][:8]}"
    )


def format_tracker_summary(tracker: dict) -> str:
    """事件完成的完整摘要。"""
    lines = [
        f"✅ 流動性掃蕩追蹤完成",
        f"類型: {tracker['event_type']}",
        f"幣種: {tracker['symbol']}",
        f"觸發價: {tracker['event_price']:.2f}",
        f"時間: {tracker['trigger_time_text']}",
        "─" * 30,
    ]

    for window_name in ["15m", "1h"]:
        w = tracker["windows"][window_name]
        lines.append(f"[{window_name}] {w['outcome']}")
        if w["hit_price"] is not None:
            lines.append(f"  命中價: {w['hit_price']:.2f}")
            lines.append(f"  延遲: {w['latency_s']}s")

    lines.append("─" * 30)
    lines.append(f"UUID: {tracker['event_uuid'][:8]}")
    return "\n".join(lines)


def format_active_trackers_report() -> str:
    """回報目前所有 active tracker 狀態。"""
    with tracker_lock:
        if not active_trackers:
            return "目前沒有進行中的掃蕩追蹤事件"

        lines = [f"📊 進行中的掃蕩追蹤 ({len(active_trackers)} 個)"]

        for t in active_trackers:
            age = int(time.time()) - t["trigger_ts"]
            lines.append("─" * 30)
            lines.append(f"{t['event_type']} {t['symbol']} @ {t['event_price']:.2f}")
            lines.append(f"已經過: {age}s")

            for wn, w in t["windows"].items():
                status = w["outcome"]
                if status == "pending":
                    remaining = w["seconds"] - age
                    status = f"pending ({max(0, remaining)}s 剩餘)"
                lines.append(f"  [{wn}] {status}")

            lines.append(f"  UUID: {t['event_uuid'][:8]}")

        return "\n".join(lines)


# =========================================================
# DB 儲存
# =========================================================
def _save_to_db(tracker: dict, get_db_conn_func):
    w15 = tracker["windows"]["15m"]
    w1h = tracker["windows"]["1h"]

    sql = """
    INSERT INTO sweep_outcomes (
        event_uuid, symbol, event_type, event_price,
        upper_target, lower_target,
        trigger_ts, trigger_time_text, tv_time,
        outcome_15m, outcome_15m_hit_price, outcome_15m_hit_ts, outcome_15m_latency_s,
        outcome_1h, outcome_1h_hit_price, outcome_1h_hit_ts, outcome_1h_latency_s
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s
    )
    """

    conn = get_db_conn_func()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                tracker["event_uuid"],
                tracker["symbol"],
                tracker["event_type"],
                tracker["event_price"],
                tracker["upper_target"],
                tracker["lower_target"],
                tracker["trigger_ts"],
                tracker["trigger_time_text"],
                tracker["tv_time"],
                w15["outcome"], w15["hit_price"], w15["hit_ts"], w15["latency_s"],
                w1h["outcome"], w1h["hit_price"], w1h["hit_ts"], w1h["latency_s"],
            ))
    finally:
        conn.close()
