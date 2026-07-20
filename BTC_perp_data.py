import os
import json
import time
import threading
import logging
import uuid

import requests
import websocket
import pymysql

from flask import Flask, request
from datetime import datetime, timedelta, timezone

import outcome_tracker
from market_data.query.flow_context import get_event_flow_context, format_flow_context
from market_data.query.snapshot_query import (
    get_latest_snapshots, get_snapshots_by_uuid,
    get_latest_scores, get_event_history, get_pending_snapshot_count,
)

# =========================================================
# 基本設定
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def load_config():
    """
    優先使用 Railway / 雲端環境變數
    若沒有，再退回本機 config.json
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    debug_raw = os.getenv("DEBUG")
    port_raw = os.getenv("PORT")
    allowed_users_raw = os.getenv("ALLOWED_USERS")

    if token:
        debug_mode = str(debug_raw or "false").lower() == "true"
        port = int(port_raw or 5000)

        allowed_users = []
        if allowed_users_raw:
            allowed_users = [x.strip() for x in allowed_users_raw.split(",") if x.strip()]
        elif chat_id:
            allowed_users = [str(chat_id).strip()]

        return {
            "telegram_bot_token": token,
            "telegram_chat_id": str(chat_id or ""),
            "debug": debug_mode,
            "port": port,
            "allowed_users": allowed_users,
            "source": "environment"
        }

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            "找不到 config.json，且環境變數 TELEGRAM_BOT_TOKEN 也未設定。"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    allowed_users = config.get("allowed_users", [])
    if not allowed_users and config.get("telegram_chat_id"):
        allowed_users = [str(config.get("telegram_chat_id")).strip()]

    return {
        "telegram_bot_token": config["telegram_bot_token"],
        "telegram_chat_id": str(config.get("telegram_chat_id", "")),
        "debug": bool(config.get("debug", True)),
        "port": int(config.get("port", 5000)),
        "allowed_users": [str(x).strip() for x in allowed_users if str(x).strip()],
        "source": "config.json"
    }


config = load_config()

TOKEN = config["telegram_bot_token"]
CHAT_ID = config["telegram_chat_id"]
DEBUG_MODE = config["debug"]
PORT = config["port"]
ALLOWED_USERS = config["allowed_users"]

TV_WEBHOOK_SECRET = os.getenv("TV_WEBHOOK_SECRET", "")

# MySQL: use shared DB helper (supports env / .env / config.json)
from shared.db import get_db_conn as _shared_get_db_conn, get_db_info as _shared_get_db_info

MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "")

API_URL = f"https://api.telegram.org/bot{TOKEN}"
HOST = "0.0.0.0"

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN 未設定，無法啟動。")

# =========================================================
# Log 設定
# =========================================================
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# 時間設定 / outcome 設定
# =========================================================
tz_taipei = timezone(timedelta(hours=8))

# 固定的報告視窗
TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60
}

# 事件規則
EVENT_OBSERVATION_SECONDS = 14400  # 4 小時

# first hit 偵測門檻（以 entry_price 為基準）
FIRST_HIT_LEVELS = [0.005, 0.01]  # ±0.5%, ±1.0%

# 各觀察窗口秒數
FLOW_WINDOWS = {
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}

# =========================================================
# 幣種設定（目前只做 BTC）
# =========================================================
TRACK_SYMBOLS = {
    "BTC": "BTC-USDT-SWAP"
}

# =========================================================
# 全域狀態
# =========================================================
data_lock = threading.Lock()
event_lock = threading.Lock()

# =========================================================
# 合約面值（contract size）
# BTC-USDT-SWAP: 0.01 BTC / contract
# =========================================================
CONTRACT_SIZES = {
    "BTC": 0.01,
}

taker_data = {
    symbol: {
        tf: [] for tf in TIMEFRAMES
    } for symbol in TRACK_SYMBOLS
}

bot_status = {
    "ws_connected": False,
    "last_ws_message_ts": 0,
    "last_trade_ts": 0,
    "reconnect_count": 0,
    "total_trades": 0,
    "last_error": ""
}

current_event = None

app = Flask(__name__)

# =========================================================
# 工具函式
# =========================================================
def current_ts() -> int:
    return int(time.time())


def now_taipei_str() -> str:
    return datetime.now(tz_taipei).strftime("%Y-%m-%d %H:%M:%S")


def format_number(x: float) -> str:
    abs_x = abs(x)
    if abs_x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if abs_x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs_x >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:,.0f}"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_get_trade_timestamp(trade: dict) -> int:
    try:
        if "ts" in trade:
            return int(int(trade["ts"]) / 1000)
    except Exception:
        pass
    return current_ts()


def get_symbol_from_instid(inst_id: str) -> str:
    try:
        return inst_id.split("-")[0]
    except Exception:
        return ""


def format_duration_minutes(seconds: int) -> str:
    return f"{seconds // 60} 分鐘"


def send_photo(chat_id: str, image_bytes: bytes, caption: str = "") -> None:
    if not chat_id:
        return
    try:
        resp = requests.post(
            f"{API_URL}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30
        )
        if resp.status_code != 200:
            logger.error("Telegram sendPhoto failed: %s - %s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("send_photo error: %s", e)


def send_message(chat_id: str, text: str) -> None:
    if not chat_id:
        logger.warning("send_message skipped: chat_id is empty")
        return

    try:
        resp = requests.post(
            f"{API_URL}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if resp.status_code != 200:
            logger.error("Telegram sendMessage failed: %s - %s",
                         resp.status_code, resp.text[:300])
            # Fallback: strip HTML tags + send without parse_mode so the
            # user sees clean text instead of literal <b>...</b>.
            if "parse" in resp.text.lower() or resp.status_code == 400:
                try:
                    import re as _re
                    # Strip simple HTML tags (no nested attrs, no scripts —
                    # our reports only use <b>/<i>/<code>/<pre>).
                    plain = _re.sub(r"<[^>]+>", "", text)
                    resp2 = requests.post(
                        f"{API_URL}/sendMessage",
                        data={"chat_id": chat_id, "text": plain},
                        timeout=10,
                    )
                    if resp2.status_code != 200:
                        logger.error("Telegram fallback also failed: %s",
                                     resp2.text[:300])
                except Exception:
                    logger.exception("fallback send failed")
    except Exception as e:
        logger.exception("send_message error: %s", e)


def _chunk_text_safe(text: str, max_len: int = 3900) -> list[str]:
    """Split a long HTML message into chunks at \\n\\n boundaries.

    All <b>/<i>/<code> tags in our reports are within a single line, so
    splitting at blank-line boundaries never breaks a tag.  Lines that
    individually exceed max_len fall back to \\n split, then hard slice.
    """
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current = ""
    # Use \n\n as primary boundary
    for block in text.split("\n\n"):
        candidate = current + ("\n\n" if current else "") + block
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        # block itself may be > max_len — split on \n
        if len(block) > max_len:
            buf = ""
            for line in block.split("\n"):
                cand = buf + ("\n" if buf else "") + line
                if len(cand) <= max_len:
                    buf = cand
                    continue
                if buf:
                    chunks.append(buf)
                    buf = ""
                if len(line) > max_len:
                    # last resort hard-slice (rare)
                    for i in range(0, len(line), max_len):
                        chunks.append(line[i:i + max_len])
                else:
                    buf = line
            if buf:
                current = buf
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks


def send_long_message(chat_id: str, text: str, max_chunk: int = 3900) -> None:
    """Send a Telegram message, chunking at safe HTML boundaries."""
    chunks = _chunk_text_safe(text, max_len=max_chunk)
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        if total > 1:
            chunk = f"<i>({i}/{total})</i>\n{chunk}"
        send_message(chat_id, chunk)


# =========================================================
# Session 判斷（依 UTC+8 台北時間）
# =========================================================
def determine_session(event_ts: int) -> str:
    """
    根據事件發生的台北時間 (UTC+8) 判斷交易時段。
    Asia:      08:00–16:00 (UTC+8)
    London:    16:00–21:00 (UTC+8)
    NY:        21:00–05:00 (UTC+8，跨日)
    Off-hours: 05:00–08:00 (UTC+8)
    可依需求調整時段邊界。
    """
    taipei_hour = datetime.fromtimestamp(event_ts, tz=tz_taipei).hour

    if 8 <= taipei_hour < 16:
        return "Asia"
    elif 16 <= taipei_hour < 21:
        return "London"
    elif taipei_hour >= 21 or taipei_hour < 5:
        return "NY"
    else:
        return "Off-hours"


# =========================================================
# Result 分類（依 return 值判定 reversal / continuation / neutral）
# =========================================================
def classify_result(liquidity_side: str, return_val: float, window: str) -> str:
    """
    根據 return 值與 liquidity_side 判定結果。

    1h 門檻：±0.5% = reversal/continuation，< 0.3% = neutral
    4h 門檻：±1.0% = reversal/continuation，< 0.5% = neutral
    中間區域回傳 None（尚未達到判定門檻）。

    BSL (buy)：跌 = reversal，漲 = continuation
    SSL (sell)：漲 = reversal，跌 = continuation
    """
    if return_val is None:
        return None

    if window == "1h":
        threshold = 0.5
        neutral_zone = 0.3
    elif window == "4h":
        threshold = 1.0
        neutral_zone = 0.5
    else:
        return None

    ls = str(liquidity_side or "").lower()

    if ls == "buy":   # BSL：掃高點後
        if return_val <= -threshold:
            return "reversal"
        elif return_val >= threshold:
            return "continuation"
        elif abs(return_val) < neutral_zone:
            return "neutral"
    elif ls == "sell":  # SSL：掃低點後
        if return_val >= threshold:
            return "reversal"
        elif return_val <= -threshold:
            return "continuation"
        elif abs(return_val) < neutral_zone:
            return "neutral"

    return None  # 介於 neutral_zone 與 threshold 之間


def outcome_label_from_hit(liquidity_side: str, first_hit_side: str) -> str:
    liquidity_side = str(liquidity_side or "").lower()
    first_hit_side = str(first_hit_side or "").lower()

    if liquidity_side == "buy":
        if first_hit_side == "upper":
            return "buy_continuation"
        if first_hit_side == "lower":
            return "buy_reversal"
        return "buy_neutral"

    if liquidity_side == "sell":
        if first_hit_side == "lower":
            return "sell_continuation"
        if first_hit_side == "upper":
            return "sell_reversal"
        return "sell_neutral"

    return "unknown"


# =========================================================
# MySQL
# =========================================================
def get_db_conn():
    return _shared_get_db_conn()


def column_exists(conn, table_name: str, column_name: str) -> bool:
    sql = """
    SELECT COUNT(*) AS cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = %s
      AND TABLE_NAME = %s
      AND COLUMN_NAME = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (MYSQL_DB, table_name, column_name))
        row = cursor.fetchone()
        return bool(row and row["cnt"] > 0)


def ensure_column(conn, table_name: str, column_name: str, column_ddl: str):
    if not column_exists(conn, table_name, column_name):
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}"
        with conn.cursor() as cursor:
            cursor.execute(sql)
        logger.info("✅ Added column: %s.%s", table_name, column_name)


def table_exists(conn, table_name: str) -> bool:
    sql = """
    SELECT COUNT(*) AS cnt
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (MYSQL_DB, table_name))
        row = cursor.fetchone()
        return bool(row and row["cnt"] > 0)


def init_db():
    """
    初始化 v2 資料表。
    若舊 liquidity_events 存在，自動改名為 liquidity_events_v1 保留舊資料。
    新表 liquidity_events 使用全新 schema。
    """
    logger.info("🚀 開始初始化 MySQL 資料表...")

    conn = get_db_conn()
    try:
        # ── Migration：舊表改名 ──
        if table_exists(conn, "liquidity_events"):
            # 檢查是否為舊版 schema（有 outcome_label 欄位代表舊版）
            if column_exists(conn, "liquidity_events", "outcome_label") \
               and not column_exists(conn, "liquidity_events", "result_1h"):
                if not table_exists(conn, "liquidity_events_v1"):
                    with conn.cursor() as cursor:
                        cursor.execute("RENAME TABLE liquidity_events TO liquidity_events_v1")
                    logger.info("✅ 舊表已改名為 liquidity_events_v1")
                else:
                    # v1 已存在，刪掉舊的 liquidity_events 讓新表建立
                    # （理論上不會走到這裡，除非重複部署）
                    with conn.cursor() as cursor:
                        cursor.execute("DROP TABLE IF EXISTS liquidity_events")
                    logger.info("⚠️ 舊表已刪除（v1 備份已存在）")

        # ── 建立新表 ──
        sql = """
        CREATE TABLE IF NOT EXISTS liquidity_events (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,

            -- 事件基本資訊
            event_time VARCHAR(32) NOT NULL          COMMENT '事件時間 = 進場時間 (UTC+8)',
            symbol VARCHAR(50) NOT NULL,
            liquidity_side VARCHAR(20) NOT NULL       COMMENT 'buy=BSL / sell=SSL',
            entry_price DECIMAL(18,8) NOT NULL        COMMENT '進場價格 = trigger 當下價格',

            -- 掃蕩參考
            sweep_ref_price DECIMAL(18,8) DEFAULT NULL COMMENT '被掃的前高/前低價格',
            sweep_size_pct DECIMAL(10,4) DEFAULT NULL  COMMENT '掃蕩幅度百分比',

            -- 累積 delta（各觀察窗口）
            delta_15m DECIMAL(20,8) DEFAULT NULL,
            delta_1h DECIMAL(20,8) DEFAULT NULL,
            delta_4h DECIMAL(20,8) DEFAULT NULL,

            -- 累積 flow（15m）
            flow_buy_15m DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_sell_15m DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_trades_15m INT NOT NULL DEFAULT 0,

            -- 累積 flow（1h）
            flow_buy_1h DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_sell_1h DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_trades_1h INT NOT NULL DEFAULT 0,

            -- 累積 flow（4h）
            flow_buy_4h DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_sell_4h DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_trades_4h INT NOT NULL DEFAULT 0,

            -- forward return
            return_15m DECIMAL(10,4) DEFAULT NULL     COMMENT '進場後15分鐘報酬率%',
            return_1h DECIMAL(10,4) DEFAULT NULL      COMMENT '進場後1小時報酬率%',
            return_4h DECIMAL(10,4) DEFAULT NULL      COMMENT '進場後4小時報酬率%',

            -- first hit
            first_hit_side VARCHAR(20) DEFAULT 'none' COMMENT 'upper/lower/none',
            first_hit_price DECIMAL(18,8) DEFAULT NULL,
            first_hit_time VARCHAR(32) DEFAULT NULL,
            first_hit_delta DECIMAL(20,8) DEFAULT NULL COMMENT '命中時的累積 delta',

            -- 市場環境
            session VARCHAR(20) DEFAULT NULL          COMMENT 'Asia/London/NY/Off-hours',

            -- 結果分類
            result_1h VARCHAR(20) DEFAULT NULL        COMMENT 'reversal/continuation/neutral',
            result_4h VARCHAR(20) DEFAULT NULL        COMMENT 'reversal/continuation/neutral',

            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            -- 內部追蹤用（不用於統計）
            event_uuid VARCHAR(64) NOT NULL,
            trigger_ts INT NOT NULL,
            tv_time VARCHAR(64) DEFAULT NULL,
            event_type VARCHAR(50) DEFAULT NULL,

            UNIQUE KEY uk_event_uuid (event_uuid),
            INDEX idx_trigger_ts (trigger_ts),
            INDEX idx_liquidity_side (liquidity_side),
            INDEX idx_symbol (symbol),
            INDEX idx_session (session),
            INDEX idx_result_1h (result_1h),
            INDEX idx_result_4h (result_4h)
        );
        """
        with conn.cursor() as cursor:
            cursor.execute(sql)

        logger.info("✅ MySQL liquidity_events (v2) 資料表就緒")
    finally:
        conn.close()


def update_event_registry(event: dict):
    """Update event_registry with 4h results (unified lifecycle)."""
    sql = """
    UPDATE event_registry SET
        status = %s,
        result_1h = %s,
        result_4h = %s,
        return_1h = %s,
        return_4h = %s,
        finished_at = NOW()
    WHERE event_uuid = %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                "finished",
                event.get("result_1h"),
                event.get("result_4h"),
                event.get("return_1h"),
                event.get("return_4h"),
                event["event_uuid"],
            ))
        logger.info("Updated event_registry for %s: status=finished", event["event_uuid"][:8])
    except Exception:
        logger.exception("Failed to update event_registry for %s", event["event_uuid"][:8])
    finally:
        conn.close()


def save_event_to_db(event: dict):
    sql = """
    INSERT INTO liquidity_events (
        event_uuid, event_type, event_time, symbol, liquidity_side, entry_price,
        trigger_ts, tv_time,
        sweep_ref_price, sweep_size_pct,
        delta_15m, delta_1h, delta_4h,
        flow_buy_15m, flow_sell_15m, flow_trades_15m,
        flow_buy_1h, flow_sell_1h, flow_trades_1h,
        flow_buy_4h, flow_sell_4h, flow_trades_4h,
        return_15m, return_1h, return_4h,
        first_hit_side, first_hit_price, first_hit_time, first_hit_delta,
        session, result_1h, result_4h
    ) VALUES (
        %s, %s, %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s
    )
    """

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                event["event_uuid"],
                event.get("event_type"),
                event["event_time"],
                event["symbol"],
                event["liquidity_side"],
                event["entry_price"],
                event["trigger_ts"],
                event.get("tv_time"),
                event.get("sweep_ref_price"),
                event.get("sweep_size_pct"),
                # delta = buy - sell
                event["flow_buy_15m"] - event["flow_sell_15m"],
                event["flow_buy_1h"] - event["flow_sell_1h"],
                event["flow_buy_4h"] - event["flow_sell_4h"],
                event["flow_buy_15m"],
                event["flow_sell_15m"],
                event["flow_trades_15m"],
                event["flow_buy_1h"],
                event["flow_sell_1h"],
                event["flow_trades_1h"],
                event["flow_buy_4h"],
                event["flow_sell_4h"],
                event["flow_trades_4h"],
                event.get("return_15m"),
                event.get("return_1h"),
                event.get("return_4h"),
                event.get("first_hit_side", "none"),
                event.get("first_hit_price"),
                event.get("first_hit_time"),
                event.get("first_hit_delta"),
                event.get("session"),
                event.get("result_1h"),
                event.get("result_4h"),
            ))
    finally:
        conn.close()


# =========================================================
# Event
# =========================================================
def create_event(event_type: str, liquidity_side: str, price: float,
                 symbol: str, tv_time: str, sweep_ref_price: float = None):
    """
    建立新的 liquidity event。

    entry_price = trigger 當下價格
    event_time  = trigger 當下時間
    觀察窗口：15m / 1h / 4h
    """
    trigger_ts = current_ts()

    # sweep_size_pct：掃蕩幅度百分比
    sweep_size_pct = None
    if sweep_ref_price and sweep_ref_price > 0:
        if liquidity_side == "buy":
            sweep_size_pct = (price - sweep_ref_price) / sweep_ref_price * 100
        elif liquidity_side == "sell":
            sweep_size_pct = (sweep_ref_price - price) / sweep_ref_price * 100

    # first hit 目標價位（±0.5%, ±1.0%）
    hit_targets = []
    for pct in FIRST_HIT_LEVELS:
        hit_targets.append(("upper", price * (1 + pct)))
        hit_targets.append(("lower", price * (1 - pct)))

    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": event_type,
        "liquidity_side": liquidity_side,
        "entry_price": price,                          # 進場價 = trigger 當下價格
        "event_time": now_taipei_str(),                # 進場時間 = trigger 當下時間
        "symbol": symbol,
        "tv_time": tv_time,
        "trigger_ts": trigger_ts,
        "observation_seconds": EVENT_OBSERVATION_SECONDS,

        # 狀態
        "status": "active",
        "finished": False,

        # 掃蕩參考
        "sweep_ref_price": sweep_ref_price,
        "sweep_size_pct": sweep_size_pct,

        # ── 各窗口 flow 統計 ──
        "flow_buy_15m": 0.0,
        "flow_sell_15m": 0.0,
        "flow_trades_15m": 0,
        "_flow_15m_locked": False,

        "flow_buy_1h": 0.0,
        "flow_sell_1h": 0.0,
        "flow_trades_1h": 0,
        "_flow_1h_locked": False,

        "flow_buy_4h": 0.0,
        "flow_sell_4h": 0.0,
        "flow_trades_4h": 0,
        "_flow_4h_locked": False,

        # ── forward return ──
        "return_15m": None,
        "return_1h": None,
        "return_4h": None,
        "_return_15m_ts": trigger_ts + 900,
        "_return_1h_ts":  trigger_ts + 3600,
        "_return_4h_ts":  trigger_ts + 14400,

        # ── first hit ──
        "first_hit_side": "none",          # upper / lower / none
        "first_hit_price": None,
        "first_hit_time": None,
        "first_hit_delta": None,           # 命中時的累積 delta
        "_hit_targets": hit_targets,       # 內部：[(side, target_price), ...]
        "_first_hit_done": False,

        # ── result 分類 ──
        "result_1h": None,
        "result_4h": None,

        # ── 市場環境 ──
        "session": determine_session(trigger_ts),
    }


def detect_first_hit(event: dict, price: float, trade_ts: int):
    """
    多層 first hit 偵測（±0.5%, ±1.0%）。
    只記錄第一個被碰到的目標。
    命中後記錄 first_hit_delta = 當下累積 delta。
    """
    if event["_first_hit_done"]:
        return

    for side, target in event["_hit_targets"]:
        hit = False
        if side == "upper" and price >= target:
            hit = True
        elif side == "lower" and price <= target:
            hit = True

        if hit:
            # 計算命中當下的累積 delta（用尚未鎖定的最大窗口）
            if not event["_flow_4h_locked"]:
                cum_delta = event["flow_buy_4h"] - event["flow_sell_4h"]
            elif not event["_flow_1h_locked"]:
                cum_delta = event["flow_buy_1h"] - event["flow_sell_1h"]
            else:
                cum_delta = event["flow_buy_15m"] - event["flow_sell_15m"]

            event["first_hit_side"] = side
            event["first_hit_price"] = float(price)
            event["first_hit_time"] = datetime.fromtimestamp(trade_ts, tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
            event["first_hit_delta"] = cum_delta
            event["_first_hit_done"] = True
            return


def generate_event_summary(event: dict) -> str:
    """事件完成的 Telegram 通知摘要（v2）。"""

    def fmt_return(val):
        if val is None:
            return "N/A"
        emoji = "🟢" if val > 0 else "🔴" if val < 0 else "🟡"
        return f"{val:+.4f}% {emoji}"

    def fmt_delta(buy, sell):
        d = buy - sell
        emoji = "🟢" if d > 0 else "🔴" if d < 0 else "🟡"
        return f"{format_number(d)} {emoji}"

    def fmt_result(val):
        if val is None:
            return "N/A"
        labels = {"reversal": "REVERSAL", "continuation": "CONTINUATION", "neutral": "NEUTRAL"}
        return labels.get(val, val)

    sweep_ref_text = f"{event['sweep_ref_price']:.2f}" if event.get("sweep_ref_price") else "N/A"
    sweep_size_text = f"{event['sweep_size_pct']:.4f}%" if event.get("sweep_size_pct") is not None else "N/A"
    fh_price = f"{event['first_hit_price']:.2f}" if event.get("first_hit_price") else "N/A"
    fh_time = event.get("first_hit_time") or "N/A"
    fh_delta = format_number(event["first_hit_delta"]) if event.get("first_hit_delta") is not None else "N/A"

    lines = [
        "✅ 流動性事件完成 (v2)",
        f"liquidity_side: {event['liquidity_side']}",
        f"symbol: {event['symbol']}",
        f"entry_price: {event['entry_price']:.2f}",
        f"event_time: {event['event_time']}",
        f"session: {event.get('session', 'N/A')}",
        "─" * 30,
        f"sweep_ref_price: {sweep_ref_text}",
        f"sweep_size_pct: {sweep_size_text}",
        "─" * 30,
        f"first_hit_side: {event['first_hit_side']}",
        f"first_hit_price: {fh_price}",
        f"first_hit_time: {fh_time}",
        f"first_hit_delta: {fh_delta}",
        "─" * 30,
        "[15m]",
        f"  return: {fmt_return(event.get('return_15m'))}",
        f"  delta: {fmt_delta(event['flow_buy_15m'], event['flow_sell_15m'])}",
        f"  buy/sell: {format_number(event['flow_buy_15m'])} / {format_number(event['flow_sell_15m'])}",
        f"  trades: {event['flow_trades_15m']}",
        "[1h]",
        f"  return: {fmt_return(event.get('return_1h'))}",
        f"  delta: {fmt_delta(event['flow_buy_1h'], event['flow_sell_1h'])}",
        f"  buy/sell: {format_number(event['flow_buy_1h'])} / {format_number(event['flow_sell_1h'])}",
        f"  trades: {event['flow_trades_1h']}",
        f"  result: {fmt_result(event.get('result_1h'))}",
        "[4h]",
        f"  return: {fmt_return(event.get('return_4h'))}",
        f"  delta: {fmt_delta(event['flow_buy_4h'], event['flow_sell_4h'])}",
        f"  buy/sell: {format_number(event['flow_buy_4h'])} / {format_number(event['flow_sell_4h'])}",
        f"  trades: {event['flow_trades_4h']}",
        f"  result: {fmt_result(event.get('result_4h'))}",
        "─" * 30,
        f"UUID: {event['event_uuid'][:8]}",
    ]
    return "\n".join(lines)


def generate_current_event_report() -> str:
    """回報目前進行中的事件狀態（v2）。"""
    with event_lock:
        if not current_event:
            return "目前沒有進行中的事件"

        e = current_event
        age = current_ts() - e["trigger_ts"]
        remaining = max(0, e["observation_seconds"] - age)

        def fmt_return(val):
            if val is None:
                return "pending"
            emoji = "🟢" if val > 0 else "🔴" if val < 0 else "🟡"
            return f"{val:+.4f}% {emoji}"

        delta_now = e["flow_buy_4h"] - e["flow_sell_4h"]
        delta_emoji = "🟢" if delta_now > 0 else "🔴" if delta_now < 0 else "🟡"

        return (
            f"🚧 事件進行中\n"
            f"liquidity_side: {e['liquidity_side']}\n"
            f"symbol: {e['symbol']}\n"
            f"entry_price: {e['entry_price']:.2f}\n"
            f"session: {e.get('session', 'N/A')}\n"
            f"elapsed: {age}s / {e['observation_seconds']}s (剩餘 {remaining}s)\n"
            f"first_hit: {e['first_hit_side']}\n"
            f"─" * 30 + "\n"
            f"return_15m: {fmt_return(e.get('return_15m'))}\n"
            f"return_1h: {fmt_return(e.get('return_1h'))}\n"
            f"return_4h: {fmt_return(e.get('return_4h'))}\n"
            f"delta_now: {format_number(delta_now)} {delta_emoji}\n"
            f"UUID: {e['event_uuid'][:8]}"
        )


# =========================================================
# 清理舊資料
# =========================================================
def clean_old_data():
    while True:
        try:
            now_ts = current_ts()
            cutoff = {
                tf: now_ts - minutes * 60
                for tf, minutes in TIMEFRAMES.items()
            }

            with data_lock:
                for symbol in taker_data:
                    for tf in TIMEFRAMES:
                        taker_data[symbol][tf] = [
                            d for d in taker_data[symbol][tf]
                            if d["timestamp"] >= cutoff[tf]
                        ]

            time.sleep(30)

        except Exception as e:
            logger.exception("clean_old_data error: %s", e)
            time.sleep(5)


# =========================================================
# 報告生成
# =========================================================
def generate_report(symbol: str) -> str:
    if symbol not in taker_data:
        return f"❌ 不支援的幣種：{symbol}"

    now = datetime.now(tz_taipei).strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"📊 [{symbol}-USDT] 合約 Taker 動能分析",
        f"查詢時間：{now} 台北時間",
        "─" * 42,
        "時間區間   買入金額        賣出金額        淨值"
    ]

    with data_lock:
        for tf in TIMEFRAMES:
            buys = sum(d["amount"] for d in taker_data[symbol][tf] if d["type"] == "buy")
            sells = sum(d["amount"] for d in taker_data[symbol][tf] if d["type"] == "sell")
            net = buys - sells

            emoji = "🟢" if net > 0 else "🔴" if net < 0 else "🟡"

            lines.append(
                f"{tf:<8}  {format_number(buys):<12}  {format_number(sells):<12}  {format_number(net):>10} {emoji}"
            )

    return "\n".join(lines)


def generate_all_report() -> str:
    parts = []
    for symbol in TRACK_SYMBOLS:
        parts.append(generate_report(symbol))
    return "\n\n".join(parts)


def generate_status_report() -> str:
    now = datetime.now(tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
    last_ws_msg = (
        datetime.fromtimestamp(bot_status["last_ws_message_ts"], tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
        if bot_status["last_ws_message_ts"] else "N/A"
    )
    last_trade = (
        datetime.fromtimestamp(bot_status["last_trade_ts"], tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
        if bot_status["last_trade_ts"] else "N/A"
    )

    whitelist_text = ", ".join(ALLOWED_USERS) if ALLOWED_USERS else "未設定"

    with event_lock:
        event_status = "有進行中事件" if current_event and not current_event["finished"] else "無進行中事件"

    db_info = _shared_get_db_info()
    mysql_status = f"已設定 (via {db_info['source']})" if db_info["host"] != "NOT SET" else "未設定完整"

    return (
        f"🤖 Bot 狀態報告\n"
        f"設定來源：{config['source']}\n"
        f"查詢時間：{now}\n"
        f"WS 連線：{'✅ 已連線' if bot_status['ws_connected'] else '❌ 未連線'}\n"
        f"最後 WS 訊息：{last_ws_msg}\n"
        f"最後成交資料：{last_trade}\n"
        f"總成交筆數：{bot_status['total_trades']}\n"
        f"重連次數：{bot_status['reconnect_count']}\n"
        f"白名單：{whitelist_text}\n"
        f"TV Secret：{'已設定' if TV_WEBHOOK_SECRET else '未設定'}\n"
        f"MySQL：{mysql_status}\n"
        f"事件狀態：{event_status}\n"
        f"觀察時間：{EVENT_OBSERVATION_SECONDS} 秒 ({EVENT_OBSERVATION_SECONDS // 3600}h)\n"
        f"First hit 門檻：±{', ±'.join(str(l*100) + '%' for l in FIRST_HIT_LEVELS)}\n"
        f"最後錯誤：{bot_status['last_error'] or '無'}"
    )


# =========================================================
# WebSocket 邏輯
# =========================================================
def on_message(ws, message):
    bot_status["last_ws_message_ts"] = current_ts()

    try:
        data = json.loads(message)

        if "event" in data:
            event = data.get("event")
            if event == "subscribe":
                logger.info("Subscribed: %s", data)
            elif event == "error":
                logger.error("OKX event error: %s", data)
                bot_status["last_error"] = str(data)
            else:
                logger.debug("WS event: %s", data)
            return

        if "data" not in data or "arg" not in data:
            logger.debug("Ignored message: %s", data)
            return

        channel = data["arg"].get("channel")
        if channel != "trades":
            return

        trades = data.get("data", [])
        if not trades:
            return

        new_entries = []

        for trade in trades:
            inst_id = trade.get("instId", "")
            symbol = get_symbol_from_instid(inst_id)

            if symbol not in taker_data:
                continue

            trade_side = trade.get("side", "").lower()
            if trade_side not in ("buy", "sell"):
                continue

            try:
                contracts = float(trade["sz"])
                price = float(trade["px"])
            except (KeyError, ValueError, TypeError):
                continue

            trade_ts = safe_get_trade_timestamp(trade)

            contract_size = CONTRACT_SIZES.get(symbol, 1.0)
            base_qty = contracts * contract_size
            amount = base_qty * price

            entry = {
                "timestamp": trade_ts,
                "amount": amount,
                "type": trade_side,
                "contracts": contracts,
                "contract_size": contract_size,
                "base_qty": base_qty,
                "price": price,
                "inst_id": inst_id
            }
            new_entries.append((symbol, entry))

            bot_status["last_trade_ts"] = trade_ts
            bot_status["total_trades"] += 1

            # outcome tracker：餵價格（不需要 lock，內部自帶）
            if symbol == "BTC":
                outcome_tracker.feed_price(price, trade_ts)

            # 事件統計與 hit 檢查：只做 BTC
            with event_lock:
                if current_event and not current_event["finished"] and symbol == "BTC":
                    e = current_event
                    age = current_ts() - e["trigger_ts"]

                    if age <= e["observation_seconds"]:
                        # ── 各窗口 flow 累加 ──
                        for wname, wsec in FLOW_WINDOWS.items():
                            locked_key = f"_flow_{wname}_locked"
                            if not e[locked_key]:
                                if age <= wsec:
                                    if trade_side == "buy":
                                        e[f"flow_buy_{wname}"] += amount
                                    else:
                                        e[f"flow_sell_{wname}"] += amount
                                    e[f"flow_trades_{wname}"] += 1
                                else:
                                    e[locked_key] = True

                        # ── first hit 偵測 ──
                        detect_first_hit(e, price, trade_ts)

                        # ── forward return 快照 ──
                        ep = e["entry_price"]
                        # return = (price_t - entry_price) / entry_price * 100
                        if e["return_15m"] is None and trade_ts >= e["_return_15m_ts"]:
                            e["return_15m"] = round((price - ep) / ep * 100, 4)
                        if e["return_1h"] is None and trade_ts >= e["_return_1h_ts"]:
                            e["return_1h"] = round((price - ep) / ep * 100, 4)
                            # 1h return 到手 → 計算 result_1h
                            e["result_1h"] = classify_result(e["liquidity_side"], e["return_1h"], "1h")
                        if e["return_4h"] is None and trade_ts >= e["_return_4h_ts"]:
                            e["return_4h"] = round((price - ep) / ep * 100, 4)
                            # 4h return 到手 → 計算 result_4h
                            e["result_4h"] = classify_result(e["liquidity_side"], e["return_4h"], "4h")

        if not new_entries:
            return

        with data_lock:
            for symbol, entry in new_entries:
                for tf in TIMEFRAMES:
                    taker_data[symbol][tf].append(entry)

        logger.debug("Received %d trades", len(new_entries))

    except Exception as e:
        bot_status["last_error"] = str(e)
        logger.exception("on_message error: %s", e)


def on_error(ws, error):
    bot_status["ws_connected"] = False
    bot_status["last_error"] = str(error)
    logger.error("WebSocket error: %s", error)


def on_close(ws, close_status_code, close_msg):
    bot_status["ws_connected"] = False
    logger.warning("WebSocket closed: code=%s msg=%s", close_status_code, close_msg)


def on_open(ws):
    bot_status["ws_connected"] = True
    logger.info("WebSocket connected.")

    args = [
        {
            "channel": "trades",
            "instId": TRACK_SYMBOLS[symbol]
        }
        for symbol in TRACK_SYMBOLS
    ]

    subscribe_msg = {"op": "subscribe", "args": args}
    ws.send(json.dumps(subscribe_msg))
    logger.info("Subscribe sent: %s", subscribe_msg)


def ws_watchdog():
    while True:
        try:
            now_ts = current_ts()
            last_msg = bot_status["last_ws_message_ts"]

            if bot_status["ws_connected"] and last_msg and (now_ts - last_msg > 90):
                logger.warning("No WS message for over 90s. Possible stale connection.")

            time.sleep(30)
        except Exception as e:
            logger.exception("ws_watchdog error: %s", e)
            time.sleep(5)


def event_watchdog():
    global current_event

    while True:
        try:
            finished_event = None

            with event_lock:
                if current_event and not current_event["finished"]:
                    age = current_ts() - current_event["trigger_ts"]

                    if age >= current_event["observation_seconds"]:
                        # 鎖定所有未鎖定的 flow 窗口
                        for wname in FLOW_WINDOWS:
                            current_event[f"_flow_{wname}_locked"] = True

                        # 補算 result（若 return 已有值但 result 還沒算）
                        if current_event.get("result_1h") is None and current_event.get("return_1h") is not None:
                            current_event["result_1h"] = classify_result(
                                current_event["liquidity_side"], current_event["return_1h"], "1h"
                            )
                        if current_event.get("result_4h") is None and current_event.get("return_4h") is not None:
                            current_event["result_4h"] = classify_result(
                                current_event["liquidity_side"], current_event["return_4h"], "4h"
                            )

                        current_event["finished"] = True
                        current_event["status"] = "finished"
                        finished_event = dict(current_event)
                        current_event = None

            if finished_event:
                try:
                    save_event_to_db(finished_event)
                    logger.info("✅ Event saved to DB: %s", finished_event["event_uuid"])
                except Exception as db_error:
                    logger.exception("save_event_to_db error: %s", db_error)

                # Update event_registry with 4h results (unified lifecycle)
                try:
                    update_event_registry(finished_event)
                except Exception as reg_err:
                    logger.exception("update_event_registry error: %s", reg_err)

                # 合併 delta 結果 + sweep tracker 結果
                summary = generate_event_summary(finished_event)
                sweep_report = outcome_tracker.get_latest_finished_summary()
                if sweep_report:
                    summary += "\n" + "─" * 30 + "\n" + sweep_report

                # Attach event-period market flow context (OKX+Binance)
                try:
                    obs_minutes = finished_event["observation_seconds"] // 60
                    event_ctx = get_event_flow_context(
                        "BTC-USD", finished_event["trigger_ts"], obs_minutes
                    )
                    summary += "\n" + "─" * 30 + "\n"
                    summary += format_flow_context(event_ctx, title=f"事件期間 {obs_minutes}m 市場流")
                except Exception as flow_err:
                    logger.warning("Failed to get event flow context: %s", flow_err)

                send_message(CHAT_ID, summary)
                logger.info("Event finished: %s", finished_event["event_uuid"])

            time.sleep(2)

        except Exception as e:
            logger.exception("event_watchdog error: %s", e)
            time.sleep(5)


def start_ws_forever():
    reconnect_delay = 5

    while True:
        try:
            logger.info("Starting WebSocket connection...")
            ws = websocket.WebSocketApp(
                OKX_WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10,
                reconnect=0
            )

        except Exception as e:
            bot_status["last_error"] = str(e)
            logger.exception("start_ws_forever error: %s", e)

        bot_status["ws_connected"] = False
        bot_status["reconnect_count"] += 1
        logger.warning("Reconnecting in %s seconds...", reconnect_delay)
        time.sleep(reconnect_delay)


# =========================================================
# Snapshot / Score / History 報告
# =========================================================
def _bias_emoji(bias):
    if bias == "reversal":
        return "🔄"
    elif bias == "continuation":
        return "➡️"
    return "⚖️"


def _ts_to_taipei(ts):
    from datetime import datetime, timezone, timedelta
    dt = datetime.fromtimestamp(int(ts), tz=timezone(timedelta(hours=8)))
    return dt.strftime("%m/%d %H:%M")


def generate_snapshot_report(uuid_prefix: str = None) -> str:
    """Generate snapshot report. If uuid_prefix given, show that event; else latest."""
    try:
        if uuid_prefix:
            rows = get_snapshots_by_uuid(uuid_prefix)
        else:
            rows = get_latest_snapshots(limit=1)

        # If no snapshots yet, try showing latest event from registry
        if not rows:
            return _snapshot_from_registry(uuid_prefix)

        # Group by event
        event_uuid = rows[0]["event_uuid"]
        side = rows[0]["liquidity_side"]
        price = rows[0]["trigger_price"]
        ts = rows[0]["trigger_ts"]

        # Build set of existing snapshot types
        existing = {r["snapshot_type"] for r in rows}

        lines = [
            f"📸 事件快照 [{side.upper()}] @ {float(price):,.2f}",
            f"時間: {_ts_to_taipei(ts)}",
            f"UUID: {event_uuid[:8]}",
            "─" * 36,
        ]

        for snap_type in ("15m", "1h", "4h"):
            matched = [r for r in rows if r["snapshot_type"] == snap_type]
            if not matched:
                lines.append(f"\n[{snap_type}] ⏳ pending")
                continue

            r = matched[0]
            bias = r["bias"]
            emoji = _bias_emoji(bias)
            rev = float(r["reversal_score"])
            cont = float(r["continuation_score"])
            conf = float(r["confidence_score"])
            final = r.get("final_score")

            score_str = f"  rev={rev:.0f} cont={cont:.0f} conf={conf:.2f}"
            if final is not None:
                score_str += f"  score={float(final):.1f}"
            lines.append(f"\n[{snap_type}] {emoji} {bias}")
            lines.append(score_str)

            if r.get("delta_value") is not None:
                delta = float(r["delta_value"])
                is_ssl_snap = r.get("liquidity_side", "").lower() == "sell"
                delta_aligned = (is_ssl_snap and delta > 0) or (not is_ssl_snap and delta < 0)
                lines.append(f"  delta: {format_number(delta)} {'🟢' if delta_aligned else '🔴'}")
            if r.get("cvd_sign_flip") is not None:
                lines.append(f"  cvd_flip: {'Yes ✅' if r['cvd_sign_flip'] else 'No ❌'}")
            if r.get("price_change_pct") is not None:
                pct = float(r["price_change_pct"])
                lines.append(f"  price: {pct:+.4f}% {'🟢' if pct > 0 else '🔴'}")
            if r.get("reclaim_flag") is not None:
                lines.append(f"  reclaim: {'Yes ✅' if r['reclaim_flag'] else 'No ❌'}")
            if r.get("break_again_flag") is not None:
                lines.append(f"  break_again: {'Yes ⚠️' if r['break_again_flag'] else 'No'}")
            if r.get("oi_change_total_pct") is not None:
                oi_pct = float(r["oi_change_total_pct"])
                lines.append(f"  OI: {oi_pct:+.2f}% {'📉' if oi_pct < 0 else '📈'}")
            if r.get("funding_rate") is not None:
                fr = float(r["funding_rate"]) * 100
                lines.append(f"  funding: {fr:+.4f}%")
            if r.get("liq_total_usd") is not None:
                liq = float(r["liq_total_usd"])
                if liq >= 100_000:
                    liq_buy = float(r.get("liq_buy_usd") or 0)
                    liq_sell = float(r.get("liq_sell_usd") or 0)
                    lines.append(f"  liq: ${liq/1e6:.1f}M (long-liq=${liq_sell/1e6:.1f}M short-liq=${liq_buy/1e6:.1f}M)")
            if r.get("label"):
                lines.append(f"  label: {r['label']} 🏷️")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("generate_snapshot_report error")
        return f"快照查詢失敗: {e}"


def _snapshot_from_registry(uuid_prefix: str = None) -> str:
    """Fallback: show event from registry when no snapshots exist yet."""
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                if uuid_prefix:
                    cur.execute(
                        "SELECT * FROM event_registry WHERE event_uuid LIKE %s LIMIT 1",
                        (uuid_prefix + "%",))
                else:
                    cur.execute(
                        "SELECT * FROM event_registry ORDER BY trigger_ts DESC LIMIT 1")
                ev = cur.fetchone()
        finally:
            conn.close()

        if not ev:
            if uuid_prefix:
                return f"找不到 UUID 開頭為 {uuid_prefix} 的事件"
            return "目前沒有任何事件資料"

        lines = [
            f"📸 事件快照 [{ev['liquidity_side'].upper()}] @ {float(ev['entry_price']):,.2f}",
            f"時間: {_ts_to_taipei(ev['trigger_ts'])}",
            f"UUID: {ev['event_uuid'][:8]}",
            "─" * 36,
            "\n[15m] ⏳ pending",
            "\n[1h] ⏳ pending",
            "\n[4h] ⏳ pending",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"查詢失敗: {e}"


def generate_score_report() -> str:
    """Show latest scoring results from event_feature_snapshots."""
    try:
        rows = get_latest_scores(limit=5)
        if not rows:
            return "目前沒有評分資料（等待第一個 15m 快照完成）"

        lines = ["📊 最近事件評分", "─" * 36]

        for r in rows:
            side = r["liquidity_side"]
            price = float(r["entry_price"])
            bias = r["bias"]
            emoji = _bias_emoji(bias)
            rev = float(r["reversal_score"])
            cont = float(r["continuation_score"])
            conf = float(r["confidence_score"])
            final = r.get("final_score")
            snap_type = r.get("snapshot_type", "?")
            label = r.get("label") or "-"

            score_str = f"  rev={rev:.0f} cont={cont:.0f} conf={conf:.2f}"
            if final is not None:
                score_str += f"  score={float(final):.1f}"

            lines.append(
                f"\n{emoji} [{side.upper()}] @ {price:,.2f}"
                f"  {_ts_to_taipei(r['trigger_ts'])} [{snap_type}]"
            )
            lines.append(score_str)

            extras = []
            if r.get("oi_change_total_pct") is not None:
                oi_pct = float(r["oi_change_total_pct"])
                extras.append(f"OI:{oi_pct:+.2f}%")
            if r.get("funding_rate") is not None:
                fr = float(r["funding_rate"]) * 100
                extras.append(f"FR:{fr:+.4f}%")
            if r.get("liq_total_usd") is not None and float(r["liq_total_usd"]) > 0:
                liq = float(r["liq_total_usd"])
                extras.append(f"liq:${liq/1e6:.1f}M")
            if extras:
                lines.append(f"  {' | '.join(extras)}")

            lines.append(f"  label={label}  UUID:{r['event_uuid'][:8]}")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("generate_score_report error")
        return f"評分查詢失敗: {e}"


def generate_history_report() -> str:
    """Show recent events with bias evolution across 15m → 1h → 4h."""
    try:
        rows = get_event_history(limit=5)
        if not rows:
            return "目前沒有事件歷史資料"

        lines = ["📋 事件歷史（bias 演化）", "─" * 36]

        for r in rows:
            side = r["liquidity_side"]
            price = float(r["entry_price"])
            label = r.get("label") or "pending"

            lines.append(
                f"\n{'BSL' if side == 'buy' else 'SSL'} @ {price:,.2f}"
                f"  {_ts_to_taipei(r['trigger_ts'])}"
            )

            # Show bias evolution: 15m → 1h → 4h
            evolution = []
            for window in ("15m", "1h", "4h"):
                b = r.get(f"bias_{window}")
                c = r.get(f"conf_{window}")
                if b:
                    emoji = _bias_emoji(b)
                    conf_str = f"{float(c):.2f}" if c else "?"
                    evolution.append(f"{window}:{emoji}{b[:4]}({conf_str})")
                else:
                    evolution.append(f"{window}:⏳")

            lines.append(f"  {' → '.join(evolution)}")
            lines.append(f"  final: {label} 🏷️  UUID: {r['event_uuid'][:8]}")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("generate_history_report error")
        return f"歷史查詢失敗: {e}"


def generate_snapshot_status_report() -> str:
    """Show snapshot runner status: pending counts."""
    try:
        counts = get_pending_snapshot_count()
        total = sum(counts.values())

        lines = [
            "⚙️ Snapshot Runner 狀態",
            "─" * 36,
            f"待處理 15m: {counts.get('15m', 0)}",
            f"待處理  1h: {counts.get('1h', 0)}",
            f"待處理  4h: {counts.get('4h', 0)}",
            f"總計: {total}",
        ]

        if total == 0:
            lines.append("\n✅ 所有快照已完成")
        else:
            lines.append(f"\n⏳ {total} 筆快照等待計算")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("generate_snapshot_status_report error")
        return f"狀態查詢失敗: {e}"


# =========================================================
# Flask Routes
# =========================================================
@app.route("/", methods=["GET"])
def index():
    return "OKX BTC Liquidity Outcome Bot is running."


@app.route("/tv", methods=["POST"])
def tradingview_webhook():
    global current_event

    try:
        data = request.get_json(silent=True)

        if not data:
            logger.warning("TV webhook received empty body")
            return {"status": "ignored", "reason": "empty body"}, 200

        secret = str(data.get("secret", "")).strip()
        if TV_WEBHOOK_SECRET and secret != TV_WEBHOOK_SECRET:
            logger.warning("Invalid TV webhook secret")
            return {"status": "forbidden"}, 403

        logger.info("TV webhook received: %s",
                    {k: v for k, v in data.items() if k != "secret"})

        event = str(data.get("event", "unknown")).strip()
        liquidity_side = str(data.get("liquidity_side", "unknown")).strip().lower()
        price = safe_float(data.get("price", 0), 0.0)
        tv_time = str(data.get("time", "")).strip()
        symbol = str(data.get("symbol", "")).strip()

        # 可選：被掃的前高/前低參考價（由 TradingView alert 提供）
        sweep_ref_price = safe_float(data.get("sweep_ref_price", 0), 0.0) or None

        # 只接 BTC 事件
        if "BTC" not in symbol.upper():
            logger.warning("Ignored non-BTC TV event: %s", symbol)
            return {"status": "ignored", "reason": "only BTC supported"}, 200

        # ── tv_alert_events bus（2026-07-17，additive）──
        # 每個通過 secret+BTC 驗證的 TV 快訊也落一筆 tv_alert_events，
        # 供 Service 2 撤單事件卡輪詢（DB 當匯流排，share data not code）。
        # 放在 liquidity_side gate 之前：純關卡快訊（無 side）也要收。
        # 可選欄位 {"window": N} 自訂回看分鐘數。絕不影響下方既有管線。
        try:
            _w = data.get("window", 90)
            tv_window = int(_w) if str(_w).strip().isdigit() else 90
            tv_window = max(15, min(360, tv_window))
            _payload = json.dumps({k: v for k, v in data.items()
                                   if k != "secret"}, ensure_ascii=False)[:2000]
            _tv_conn = get_db_conn()
            try:
                with _tv_conn.cursor() as _tv_cur:
                    _tv_cur.execute(
                        "INSERT INTO tv_alert_events (received_ms, symbol, "
                        "event, liquidity_side, price, window_mins, raw_json) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (int(time.time() * 1000), symbol[:32], event[:64],
                         liquidity_side[:8] if liquidity_side != "unknown" else "",
                         price if price > 0 else None, tv_window, _payload))
                _tv_conn.commit()
            finally:
                _tv_conn.close()
        except Exception as tv_bus_err:
            logger.warning("tv_alert_events insert failed (non-fatal): %s",
                           tv_bus_err)

        if liquidity_side not in ("buy", "sell"):
            logger.warning("Ignored invalid liquidity_side: %s", liquidity_side)
            return {"status": "ignored", "reason": "invalid liquidity_side"}, 200

        if price <= 0:
            logger.warning("Ignored invalid trigger price: %s", price)
            return {"status": "ignored", "reason": "invalid price"}, 200

        new_event = create_event(event, liquidity_side, price, symbol, tv_time,
                                 sweep_ref_price=sweep_ref_price)

        # 純事件追蹤（獨立於 current_event，支援多事件同時追蹤）
        tracker = outcome_tracker.register_event(event, liquidity_side, price, "BTC", tv_time)

        # 立刻寫入 event_registry，讓 snapshot runner 可以開始追蹤
        try:
            from market_data.features.snapshot_repository import register_event as _reg_event
            _reg_event(
                event_uuid=new_event["event_uuid"],
                event_type=event,
                symbol=symbol,
                liquidity_side=liquidity_side,
                entry_price=price,
                trigger_ts=new_event["trigger_ts"],
                sweep_ref_price=sweep_ref_price,
            )
        except Exception as reg_err:
            logger.warning("Failed to register event to registry: %s", reg_err)

        # 立刻觸發 OI 採集，確保 baseline 精確（不等 60s 輪詢）
        def _collect_oi_baseline():
            try:
                from market_data.adapters.oi_collector import collect_once as _oi_collect
                n = _oi_collect()
                logger.info("OI baseline collected on event arrival: %d sources", n)
            except Exception as oi_err:
                logger.warning("OI baseline collection failed: %s", oi_err)

        threading.Thread(target=_collect_oi_baseline, daemon=True,
                         name="oi-baseline").start()

        with event_lock:
            event_skipped = False
            if current_event and not current_event["finished"]:
                logger.warning(
                    "New TV event ignored because another event is still running: %s",
                    current_event["event_uuid"]
                )
                event_skipped = True
            else:
                current_event = new_event

        # 2026-07-20: 舊「收到 TradingView 快訊」Telegram 通知已移除（使用者
        # 反饋它跟撤單流 bot 的事件卡重複、且誤導成「V7 bot 也在講這件事」）。
        # 底層追蹤不受影響——create_event / outcome_tracker.register_event /
        # snapshot_repository 都已在上面跑完，liquidity_events、
        # sweep_outcomes 照常寫入（H-R 反轉檢定 + 互動圖表 ⚡ 標記的資料源）。
        # 只是不再送這則會被誤認成「訊號」的重複訊息。log 保留可追蹤性。
        logger.info(
            "TV sweep tracker started uuid=%s side=%s px=%s skipped=%s",
            tracker["event_uuid"][:8], liquidity_side, price, event_skipped)

        if event_skipped:
            return {"status": "partial", "reason": "sweep tracker started, delta event skipped"}, 200

        return {"status": "ok"}, 200

    except Exception as e:
        logger.exception("TradingView webhook error: %s", e)
        return {"status": "error", "message": str(e)}, 200


INDICATOR_SERVICE_URL = os.getenv("INDICATOR_SERVICE_URL", "")


def _indicator_admin_headers() -> dict:
    """Auth header for the indicator service's admin-guarded routes.

    The indicator service fail-closes those routes when ADMIN_HEAL_TOKEN is
    unset there, so an empty dict here just means "send nothing" — the guard
    on the other side decides.
    """
    tok = os.getenv("ADMIN_HEAL_TOKEN", "")
    return {"X-Admin-Token": tok} if tok else {}


_INDICATOR_BUTTONS = json.dumps({"inline_keyboard": [
    [
        {"text": "\U0001f4ca Chart", "callback_data": "chart"},
        {"text": "\U0001f4cb Status", "callback_data": "status"},
    ],
    [
        {"text": "\U0001f4c8 Perf", "callback_data": "perf"},
        {"text": "\U0001f4e6 DB", "callback_data": "db"},
    ],
    [
        {"text": "\U0001f4b0 LIVE Perf", "callback_data": "okx_perf"},
    ],
]})


def _send_photo_with_buttons(chat_id: str, png: bytes, caption: str):
    """Send photo with inline keyboard buttons."""
    try:
        resp = requests.post(
            f"{API_URL}/sendPhoto",
            data={
                "chat_id": chat_id,
                "caption": caption,
                "reply_markup": _INDICATOR_BUTTONS,
            },
            files={"photo": ("indicator.png", png, "image/png")},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error("sendPhoto with buttons failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("send_photo_with_buttons error: %s", e)


def _handle_indicator_chart(chat_id: str):
    """Fetch indicator chart from Indicator service and send to user."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        import base64
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/indicator-chart", timeout=30)
        if resp.status_code != 200:
            send_message(chat_id, f"❌ Indicator 服務未就緒 ({resp.status_code})")
            return
        data = resp.json()
        png = base64.b64decode(data["png_base64"])
        _send_photo_with_buttons(chat_id, png, data.get("caption", "BTC 4h Indicator"))
    except Exception as e:
        logger.exception("indicator chart fetch error: %s", e)
        send_message(chat_id, f"❌ 取得指標圖表失敗: {e}")


def _handle_indicator_status(chat_id: str):
    """Fetch indicator status from Indicator service and send to user."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/indicator-status", timeout=15,
                            headers=_indicator_admin_headers())
        if resp.status_code != 200:
            send_message(chat_id, f"❌ Indicator 服務未就緒 ({resp.status_code})")
            return
        data = resp.json()
        send_message(chat_id, f"📊 <b>Indicator Status</b>\n\n{data.get('text', 'N/A')}")
    except Exception as e:
        logger.exception("indicator status fetch error: %s", e)
        send_message(chat_id, f"❌ 取得指標狀態失敗: {e}")


def _handle_indicator_db(chat_id: str):
    """Fetch DB stats from Indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/indicator-db-stats", timeout=15,
                            headers=_indicator_admin_headers())
        if resp.status_code != 200:
            send_message(chat_id, f"❌ Indicator 服務未就緒 ({resp.status_code})")
            return
        data = resp.json()
        send_message(chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("indicator db stats error: %s", e)
        send_message(chat_id, f"❌ 取得資料庫狀態失敗: {e}")


def _handle_indicator_perf(chat_id: str):
    """Fetch model performance from Indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/indicator-perf", timeout=30,
                            headers=_indicator_admin_headers())
        if resp.status_code != 200:
            send_message(chat_id, f"❌ Indicator 服務未就緒 ({resp.status_code})")
            return
        data = resp.json()
        send_message(chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("indicator perf error: %s", e)
        send_message(chat_id, f"❌ 取得模型表現失敗: {e}")


def _handle_alpha_decay(chat_id: str):
    """Fetch alpha decay monitor from Indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/alpha-decay", timeout=30)
        if resp.status_code != 200:
            send_message(chat_id, f"❌ Indicator 服務未就緒 ({resp.status_code})")
            return
        data = resp.json()
        send_message(chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("alpha decay error: %s", e)
        send_message(chat_id, f"❌ Alpha decay 檢查失敗: {e}")


def _say_via(api, chat_id: str, text: str) -> None:
    """Reply via the given bot API (cancel bot) or the main bot."""
    if api:
        try:
            requests.post(f"{api}/sendMessage",
                          data={"chat_id": chat_id, "text": text,
                                "parse_mode": "HTML"}, timeout=15)
        except Exception:
            logger.exception("say_via failed")
    else:
        send_message(chat_id, text)


def _handle_cancel_flow(chat_id: str, hours: int = 48, api: str = None):
    """Send link to the interactive cancel-flow review chart (research tool).

    /research/* is admin-guarded; embed ?token= in the URL so the link opens
    from a phone browser — same pattern as the Dashboard button (operator-only
    chat, so the URL stays private). The page re-renders on every load.
    """
    if not INDICATOR_SERVICE_URL:
        _say_via(api, chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    url = f"{INDICATOR_SERVICE_URL}/research/cancel-flow-i?hours={hours}"
    tok = os.getenv("ADMIN_HEAL_TOKEN", "")
    if tok:
        url += f"&token={tok}"
    window = "全 depth 時代" if hours == 0 else f"最近 {hours}h"
    _say_via(api, chat_id,
        f"<b>撤單流覆盤（互動圖·研究非信號）</b>\n\n"
        f"<a href=\"{url}\">點擊開啟互動覆盤圖</a>\n\n"
        f"視窗: {window}（/cancelflow 168 改週視角, 0=全期間）\n"
        f"1m K棒 + 成交量 + 毛撤單偏斜 + 淨偏斜(撤−加) + 強度, 五面板同步\n"
        f"▲▼=v7 Strong · 開啟後等數秒 render · edge 待 8/10 判決")


def _handle_cancel_analyze(chat_id: str, mins: int = 90, api: str = None):
    """Cancel-flow deterministic window summary (research aid, not a signal)."""
    if not INDICATOR_SERVICE_URL:
        _say_via(api, chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/research/cancel-analyze",
                            params={"mins": mins}, timeout=120,
                            headers=_indicator_admin_headers())
        try:
            data = resp.json()
        except Exception:
            data = {}
        if resp.status_code != 200:
            _say_via(api, chat_id,
                     f"❌ 撤單分析失敗 ({resp.status_code}) {data.get('error', '')}")
            return
        _say_via(api, chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("cancel analyze error: %s", e)
        _say_via(api, chat_id, f"❌ 撤單分析失敗: {e}")


def _handle_cancel_state(chat_id: str, mins: int = 90, api: str = None):
    """Current cancel-flow display state, one-liner (research aid, not a signal)."""
    if not INDICATOR_SERVICE_URL:
        _say_via(api, chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/research/cancel-analyze",
                            params={"mode": "state", "mins": mins}, timeout=120,
                            headers=_indicator_admin_headers())
        try:
            data = resp.json()
        except Exception:
            data = {}
        if resp.status_code != 200:
            _say_via(api, chat_id,
                     f"❌ 撤單狀態查詢失敗 ({resp.status_code}) {data.get('error', '')}")
            return
        _say_via(api, chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("cancel state error: %s", e)
        _say_via(api, chat_id, f"❌ 撤單狀態查詢失敗: {e}")


def _remove_inline_buttons(chat_id: str, message_id, api: str = None) -> None:
    """Clear a card's inline keyboard once the verdict locks in."""
    if not message_id:
        return
    try:
        requests.post(f"{api or API_URL}/editMessageReplyMarkup",
                      data={"chat_id": chat_id, "message_id": message_id,
                            "reply_markup": json.dumps({"inline_keyboard": []})},
                      timeout=10)
    except Exception:
        pass


def _handle_eyeball_verdict(chat_id: str, cb_data: str, message_id=None,
                            api: str = None):
    """A3 按鈕即日誌: ceb|{src}|{id}|{verdict} → cancel_eyeball_log。

    首判 INSERT IGNORE 鎖定（前瞻紀錄不可事後改）；skip 不落表。
    表由 Service 2 poller 建置與回填——這裡只寫人的判讀（DB as bus，
    share data not code）。api 指定時（撤單獨立 bot）回覆走該 bot。"""
    def say(t):
        if api:
            try:
                requests.post(f"{api}/sendMessage",
                              data={"chat_id": chat_id, "text": t}, timeout=10)
            except Exception:
                pass
        else:
            send_message(chat_id, t)
    try:
        parts = cb_data.split("|")
        if len(parts) != 4:
            return
        _, src, sid_s, verdict = parts
        if (src not in ("tv", "pb")
                or verdict not in ("up", "down", "agree", "opposite",
                                   "unsure", "skip")):
            return
        sid = int(sid_s)
        if verdict == "skip":
            _remove_inline_buttons(chat_id, message_id, api)
            say(f"✗ 略過不記 ({src}#{sid})")
            return

        event_ms = state = direction = None
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                if src == "tv":
                    cur.execute("SELECT received_ms, state, liquidity_side "
                                "FROM tv_alert_events WHERE id=%s", (sid,))
                    r = cur.fetchone()
                    if r:
                        event_ms = int(r["received_ms"])
                        state = r.get("state")
                        direction = r.get("liquidity_side") or None
                else:
                    cur.execute("SELECT minute_start_ms, playbook, direction "
                                "FROM cancel_playbook_events WHERE id=%s", (sid,))
                    r = cur.fetchone()
                    if r:
                        event_ms = int(r["minute_start_ms"])
                        state = r.get("playbook")
                        direction = r.get("direction")
                if event_ms is None:
                    say(f"❌ 找不到事件 ({src}#{sid})")
                    return
                cur.execute(
                    "INSERT IGNORE INTO cancel_eyeball_log "
                    "(source, source_id, event_ms, card_state, card_direction, "
                    " verdict, verdict_ms) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (src, sid, event_ms, state, direction, verdict,
                     int(time.time() * 1000)))
                inserted = cur.rowcount > 0
                prev = None
                if not inserted:
                    cur.execute("SELECT verdict FROM cancel_eyeball_log "
                                "WHERE source=%s AND source_id=%s", (src, sid))
                    pr = cur.fetchone()
                    prev = pr["verdict"] if pr else "?"
            conn.commit()
        finally:
            conn.close()

        zh = {"up": "🔼 漲", "down": "🔽 跌", "agree": "🟢 同意",
              "opposite": "🔴 相反", "unsure": "⏸ 不確定"}
        if inserted:
            _remove_inline_buttons(chat_id, message_id, api)
            say(f"✅ 已落表 {src}#{sid} → {zh.get(verdict, verdict)}"
                f"（前瞻紀錄，判定窗到期自動回填）")
        else:
            say(f"🔒 {src}#{sid} 首判已鎖定（{zh.get(prev, prev)}）"
                f"——前瞻紀錄不可事後改")
    except Exception as e:
        logger.exception("eyeball verdict error: %s", e)
        say(f"❌ 判讀落表失敗: {e}")


def _handle_cancel_action(chat_id: str, cb_data: str, message_id=None,
                          api: str = None):
    """行動鍵 (2026-07-20，取代四鍵判讀): cfa|{src}|{id}|{action}。

    zoom=事件窗互動覆盤圖連結 deep=90m 五步摘要
    star=收藏（cancel_eyeball_log verdict='star'，回填照舊）
    dismiss=收鍵盤。舊 ceb| 判讀處理器保留給歷史卡片。"""
    def say(t):
        try:
            requests.post(f"{api or API_URL}/sendMessage",
                          data={"chat_id": chat_id, "text": t}, timeout=10)
        except Exception:
            pass
    try:
        parts = cb_data.split("|")
        if len(parts) != 4:
            return
        _, src, sid_s, action = parts
        if (src not in ("tv", "pb")
                or action not in ("zoom", "deep", "star", "dismiss")):
            return
        sid = int(sid_s)
        if action == "dismiss":
            _remove_inline_buttons(chat_id, message_id, api)
            return
        if action == "deep":
            _handle_cancel_analyze(chat_id, 90, api)
            return
        event_ms = state = direction = None
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                if src == "tv":
                    cur.execute("SELECT received_ms ms, state, "
                                "liquidity_side d FROM tv_alert_events "
                                "WHERE id=%s", (sid,))
                else:
                    cur.execute("SELECT minute_start_ms ms, playbook state, "
                                "direction d FROM cancel_playbook_events "
                                "WHERE id=%s", (sid,))
                r = cur.fetchone()
                if not r:
                    say(f"❌ 找不到事件 ({src}#{sid})")
                    return
                event_ms = int(r["ms"])
                state, direction = r.get("state"), r.get("d")
                if action == "star":
                    cur.execute(
                        "INSERT IGNORE INTO cancel_eyeball_log "
                        "(source, source_id, event_ms, card_state, "
                        " card_direction, verdict, verdict_ms) "
                        "VALUES (%s,%s,%s,%s,%s,'star',%s)",
                        (src, sid, event_ms, state, direction or None,
                         int(time.time() * 1000)))
                    conn.commit()
        finally:
            conn.close()
        if action == "star":
            say(f"⭐ 已收藏 {src}#{sid}（判定窗到期自動回填結果）")
            return
        # zoom: 互動覆盤圖，視窗涵蓋事件時刻（事件越久遠窗開越大）
        age_h = max(0.0, (time.time() * 1000 - event_ms) / 3_600_000)
        _handle_cancel_flow(chat_id, max(2, int(age_h) + 2), api)
    except Exception as e:
        logger.exception("cancel action error: %s", e)
        say(f"❌ 行動鍵失敗: {e}")


# ── 撤單流獨立 bot webhook（2026-07-19，設 CANCEL_TG_BOT_TOKEN 才註冊）──
CANCEL_TG_TOKEN = os.getenv("CANCEL_TG_BOT_TOKEN", "").strip()
if CANCEL_TG_TOKEN:
    _CANCEL_API = f"https://api.telegram.org/bot{CANCEL_TG_TOKEN}"

    def _cancel_menu(chat_id: str) -> None:
        kb = {"inline_keyboard": [
            [{"text": "📈 覆盤圖 48h", "callback_data": "cf_chart"},
             {"text": "🎛 當前狀態", "callback_data": "cf_state"}],
            [{"text": "📋 五步摘要 90m", "callback_data": "cf_analyze"}],
        ]}
        try:
            requests.post(f"{_CANCEL_API}/sendMessage", data={
                "chat_id": chat_id, "parse_mode": "HTML",
                "text": ("<b>撤單流研究 bot</b>（研究·非信號）\n\n"
                         "/cancelflow [h] - 互動覆盤圖（168=週, 0=全期）\n"
                         "/cancelstate [m] - 六態狀態一行\n"
                         "/cancelanalyze [m] - 五步摘要\n\n"
                         "事件卡自動推送：watcher 劇本 + TV 快訊"
                         "（sweep 二段式+H-R 旗標）\n"
                         "卡片按鈕：特寫圖/90m深入/收藏/忽略；"
                         "判定窗到期自動回覆「對答案」"),
                "reply_markup": json.dumps(kb)}, timeout=15)
        except Exception:
            logger.exception("cancel menu send failed")

    @app.route(f"/cancelbot/{CANCEL_TG_TOKEN}", methods=["POST"])
    def cancel_bot_webhook():
        """撤單流 bot：指令/選單 + ceb| 判讀 callback，回覆全走 _CANCEL_API。"""
        try:
            data = request.get_json(silent=True) or {}
            cb = data.get("callback_query")
            if cb:
                try:
                    requests.post(f"{_CANCEL_API}/answerCallbackQuery",
                                  data={"callback_query_id": cb.get("id", "")},
                                  timeout=5)
                except Exception:
                    pass
                cb_chat = str(cb.get("message", {}).get("chat", {})
                              .get("id", ""))
                if ALLOWED_USERS and cb_chat not in ALLOWED_USERS:
                    return "ok"
                cb_data = cb.get("data", "")
                mid = cb.get("message", {}).get("message_id")
                if cb_data.startswith("ceb|"):
                    threading.Thread(
                        target=_handle_eyeball_verdict,
                        args=(cb_chat, cb_data, mid, _CANCEL_API),
                        daemon=True).start()
                elif cb_data.startswith("cfa|"):
                    threading.Thread(
                        target=_handle_cancel_action,
                        args=(cb_chat, cb_data, mid, _CANCEL_API),
                        daemon=True).start()
                elif cb_data == "cf_chart":
                    threading.Thread(target=_handle_cancel_flow,
                                     args=(cb_chat, 48, _CANCEL_API),
                                     daemon=True).start()
                elif cb_data == "cf_state":
                    threading.Thread(target=_handle_cancel_state,
                                     args=(cb_chat, 90, _CANCEL_API),
                                     daemon=True).start()
                elif cb_data == "cf_analyze":
                    threading.Thread(target=_handle_cancel_analyze,
                                     args=(cb_chat, 90, _CANCEL_API),
                                     daemon=True).start()
                return "ok"

            msg = data.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = str(msg.get("text", "")).strip()
            if not chat_id or not text:
                return "ok"
            if ALLOWED_USERS and chat_id not in ALLOWED_USERS:
                return "ok"
            parts = text.split()
            cmd = parts[0].split("@")[0].lower()
            arg = (int(parts[1]) if len(parts) > 1 and parts[1].isdigit()
                   else None)
            if cmd in ("/start", "/help", "/menu"):
                _cancel_menu(chat_id)
            elif cmd == "/cancelflow":
                threading.Thread(
                    target=_handle_cancel_flow,
                    args=(chat_id, 48 if arg is None else arg, _CANCEL_API),
                    daemon=True).start()
            elif cmd == "/cancelstate":
                threading.Thread(
                    target=_handle_cancel_state,
                    args=(chat_id, arg or 90, _CANCEL_API),
                    daemon=True).start()
            elif cmd == "/cancelanalyze":
                threading.Thread(
                    target=_handle_cancel_analyze,
                    args=(chat_id, arg or 90, _CANCEL_API),
                    daemon=True).start()
            return "ok"
        except Exception:
            logger.exception("cancel bot webhook error")
            return "ok"


def _handle_signal_perf(chat_id: str):
    """Fetch Strong signal performance report from Indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "INDICATOR_SERVICE_URL not set")
        return
    try:
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/signal-perf", timeout=30)
        data = resp.json()
        send_message(chat_id, data.get("text", "N/A"))
    except Exception as e:
        logger.exception("signal perf error: %s", e)
        send_message(chat_id, f"❌ 信號績效查詢失敗: {e}")


def _handle_meeting(chat_id: str):
    """Trigger AI agent meeting via Indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "\u274c INDICATOR_SERVICE_URL \u672a\u8a2d\u5b9a")
        return
    try:
        resp = requests.get(
            f"{INDICATOR_SERVICE_URL}/meeting",
            params={"sync": "1", "chat_id": chat_id},
            timeout=300,  # meetings can take a few minutes
            headers=_indicator_admin_headers(),
        )
        data = resp.json()
        if data.get("status") != "ok":
            send_message(chat_id, f"\u274c Meeting \u5931\u6557: {data.get('error', 'unknown')}")
    except requests.Timeout:
        send_message(chat_id, "\u26a0\ufe0f Meeting \u903e\u6642\uff08>5\u5206\u9418\uff09\uff0c\u53ef\u80fd\u4ecd\u5728\u904b\u884c\u4e2d")
    except Exception as e:
        logger.exception("meeting error: %s", e)
        send_message(chat_id, f"\u274c Meeting \u5931\u6557: {e}")


def _handle_force_update(chat_id: str):
    """Trigger manual indicator update cycle (sync mode — waits for result)."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "INDICATOR_SERVICE_URL not set")
        return
    try:
        send_message(chat_id, "⏳ Updating... (may take 30-60s)")
        resp = requests.get(f"{INDICATOR_SERVICE_URL}/force-update?sync=1", timeout=120,
                            headers=_indicator_admin_headers())
        data = resp.json()
        if resp.status_code == 200 and data.get("status") == "ok":
            detail = data.get("detail", {})
            tg = detail.get("telegram_send", "?")
            bars = detail.get("bars_predicted", "?")
            chart_kb = detail.get("chart_bytes", 0) // 1024
            direction = detail.get("direction", "?")
            mode = detail.get("engine_mode", "?")
            send_message(chat_id,
                f"✅ Update complete\n"
                f"Model: {mode} | Direction: {direction}\n"
                f"Bars: {bars} | Chart: {chart_kb}KB\n"
                f"TG send: {tg}")
        else:
            err = data.get("error", f"HTTP {resp.status_code}")
            send_message(chat_id, f"❌ Update failed: {err}")
    except requests.exceptions.Timeout:
        send_message(chat_id, "⚠️ Update timed out (>120s). Check /health for status.")
    except Exception as e:
        logger.exception("force update error: %s", e)
        send_message(chat_id, f"❌ Update error: {e}")


def _send_help(chat_id: str):
    """Send help message with inline keyboard."""
    import json as json_mod
    help_msg = (
        "<b>BTC Market Intelligence</b>\n\n"
        "<b>--- 核心指標 ---</b>\n"
        "/chart - 4h 多空預測指標圖\n"
        "/ichart - 互動圖表 (可放大/十字線)\n"
        "/perf - 模型表現 + Strong 信號績效\n"
        "/db - 資料庫累積狀態\n"
        "/ind_status - 指標系統狀態\n"
        "\n<b>--- 流動性監控 ---</b>\n"
        "/flow_futures_btc - BTC taker flow\n"
        "/flow_futures_all - 全幣種 flow\n"
        "/status - Bot 狀態\n"
        "\n<b>--- 事件追蹤 ---</b>\n"
        "/event_status - 進行中事件\n"
        "/decay - Alpha Decay 監控\n"
        "/snap - 最新事件快照\n"
        "/score - 最近事件評分\n"
        "/history - 事件歷史\n"
        "\n<b>--- 其他 ---</b>\n"
        "/flow_chart - 訂單流圖表\n"
        "(撤單流研究功能已移至獨立 bot)\n\n"
        "<i>也可直接點擊下方按鈕操作</i>"
    )
    keyboard = json_mod.dumps({"inline_keyboard": [
        [
            {"text": "\U0001f4ca Chart", "callback_data": "chart"},
            {"text": "\U0001f4c8 Perf", "callback_data": "perf"},
            {"text": "\U0001f4e6 DB", "callback_data": "db"},
        ],
        [
            {"text": "\U0001f30a Flow BTC", "callback_data": "flow_btc"},
            {"text": "\U0001f30d Flow All", "callback_data": "flow_all"},
            {"text": "\u2699\ufe0f Status", "callback_data": "status"},
        ],
        [
            {"text": "\U0001f4c8 iChart", "callback_data": "ichart"},
            {"text": "\U0001f4c9 Decay", "callback_data": "decay"},
            {"text": "\u2753 Help", "callback_data": "help"},
        ],
        [
            {"text": "\U0001f4b0 LIVE Perf", "callback_data": "okx_perf"},
        ],
    ]})
    url = f"{API_URL}/sendMessage"
    try:
        requests.post(url, data={
            "chat_id": chat_id,
            "text": help_msg,
            "parse_mode": "HTML",
            "reply_markup": keyboard,
        }, timeout=10)
    except Exception as e:
        logger.warning("Help send failed: %s", e)


def _handle_okx_perf(chat_id: str) -> None:
    """Fetch OKX live cohort report via the indicator service."""
    if not INDICATOR_SERVICE_URL:
        send_message(chat_id, "❌ INDICATOR_SERVICE_URL 未設定")
        return
    try:
        resp = requests.get(
            f"{INDICATOR_SERVICE_URL}/okx-perf", timeout=30,
            headers=_indicator_admin_headers(),
        )
        data = resp.json() if resp.status_code == 200 else {}
        text = data.get("text", "❌ OKX perf 查詢失敗")
        send_message(chat_id, text)
    except Exception as e:
        logger.exception("okx_perf error: %s", e)
        send_message(chat_id, f"❌ OKX perf 查詢失敗: {e}")


def _handle_okx_approval_response(chat_id: str, raw_cmd: str) -> None:
    """Worker thread: route /yes_<id> /no_<id> to the approval gate.

    Each step's failure is reported back to the chat so the operator
    knows whether to retry, re-issue manually, or escalate.
    """
    try:
        raw = raw_cmd.strip()
        verb_part, _, id_part = raw.partition("_")
        verb = verb_part.lower().lstrip("/")
        try:
            approval_id = int(id_part)
        except ValueError:
            send_message(chat_id,
                f"⚠️ 無效的 approval id: `{id_part}`\n格式: /yes_42 或 /no_42")
            return

        from indicator.okx import runner as okx_runner
        gate = okx_runner.get_approval_gate()
        if gate is None:
            send_message(chat_id, "⚠️ OKX executor 未啟用 (OKX_EXECUTOR_ENABLED!=1)")
            return

        if verb == "no":
            decision = gate.deny(approval_id, decided_by=chat_id)
            if decision.ok:
                send_message(chat_id, f"❌ Approval #{approval_id} DENIED")
            else:
                send_message(chat_id,
                    f"⚠️ Approval #{approval_id} 拒絕失敗: {decision.status}")
            return

        # verb == "yes"
        decision = gate.approve(approval_id, decided_by=chat_id)
        if not decision.ok or decision.intent is None:
            send_message(chat_id,
                f"⚠️ Approval #{approval_id} 核准失敗: {decision.status} {decision.reason}")
            return

        executor = okx_runner.get_executor()
        if executor is None:
            send_message(chat_id,
                "⚠️ Executor 已停用 — 核准記錄保留但無法執行")
            return

        # Optional drift check: pull a fresh price via REST balance ping
        # is not the right signal; for now we trust the intent (operator
        # is the human gate).  Drift detection is wired but not used
        # until we have a cheap latest-price source.
        result = executor.execute_approved_intent(
            decision.intent, approval_id=approval_id,
        )
        if result.action == "open":
            d = result.detail
            send_message(chat_id,
                f"✅ Approval #{approval_id} EXECUTED\n"
                f"pos #{d.get('position_id')} {d.get('side')} "
                f"@ {d.get('entry_price'):.2f}\n"
                f"size: {d.get('size_contracts')} contracts  "
                f"stop: {d.get('current_stop'):.2f}")
        else:
            send_message(chat_id,
                f"⚠️ Approval #{approval_id} 核准但執行失敗: "
                f"{result.action} {result.detail}")
    except Exception:
        logger.exception("okx_approval_response_failed")
        try:
            send_message(chat_id, f"⚠️ Approval 處理失敗 (見 log): `{raw_cmd}`")
        except Exception:
            pass


@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data:
            return "ok"

        # Handle inline button callback (from Indicator chart buttons)
        callback = data.get("callback_query")
        if callback:
            cb_data = callback.get("data", "")
            cb_chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
            cb_id = callback.get("id", "")
            # Answer callback to remove loading spinner
            try:
                requests.post(
                    f"{API_URL}/answerCallbackQuery",
                    data={"callback_query_id": cb_id}, timeout=5,
                )
            except Exception:
                pass
            if ALLOWED_USERS and cb_chat_id not in ALLOWED_USERS:
                return "ok"
            if cb_data.startswith("ceb|"):
                # A3 撤單事件卡四鍵判讀 → cancel_eyeball_log
                cb_msg_id = callback.get("message", {}).get("message_id")
                threading.Thread(target=_handle_eyeball_verdict,
                                 args=(cb_chat_id, cb_data, cb_msg_id),
                                 daemon=True).start()
                return "ok"
            if cb_data == "chart":
                threading.Thread(target=_handle_indicator_chart, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "status":
                threading.Thread(target=_handle_indicator_status, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "perf":
                threading.Thread(target=_handle_indicator_perf, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "db":
                threading.Thread(target=_handle_indicator_db, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "flow_btc":
                send_message(cb_chat_id, generate_report("BTC"))
            elif cb_data == "flow_all":
                send_message(cb_chat_id, generate_all_report())
            elif cb_data == "ichart":
                if INDICATOR_SERVICE_URL:
                    url_link = f"{INDICATOR_SERVICE_URL}/live-chart"
                    send_message(cb_chat_id,
                        f"<b>Interactive Chart</b>\n\n"
                        f"<a href=\"{url_link}\">點擊開啟互動圖表</a>\n\n"
                        f"功能: 放大縮小 / 拖曳平移 / 十字線游標")
                else:
                    send_message(cb_chat_id, "INDICATOR_SERVICE_URL 未設定")
            elif cb_data == "decay":
                threading.Thread(target=_handle_alpha_decay, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "okx_perf":
                threading.Thread(target=_handle_okx_perf, args=(cb_chat_id,), daemon=True).start()
            elif cb_data == "help":
                _send_help(cb_chat_id)
            return "ok"

        message = data.get("message", {})
        chat = message.get("chat", {})
        text = message.get("text", "")

        if not chat or not text:
            return "ok"

        chat_id = str(chat.get("id", "")).strip()
        chat_type = str(chat.get("type", "")).strip().lower()
        raw_text = text.strip().split("@")[0]  # preserve case for UUID args
        cmd = raw_text.lower()

        if chat_type != "private":
            logger.warning("Rejected non-private chat: %s (%s)", chat_id, chat_type)
            return "ok"

        if ALLOWED_USERS and chat_id not in ALLOWED_USERS:
            logger.warning("Unauthorized access: %s", chat_id)
            return "ok"

        logger.info("Telegram command received: %s from %s", cmd, chat_id)

        if cmd == "/flow_futures_btc":
            send_message(chat_id, generate_report("BTC"))

        elif cmd == "/flow_futures_all":
            send_message(chat_id, generate_all_report())

        elif cmd == "/status":
            send_message(chat_id, generate_status_report())

        elif cmd == "/event_status":
            send_message(chat_id, generate_current_event_report())

        elif cmd == "/sweep_status":
            send_message(chat_id, outcome_tracker.format_active_trackers_report())

        elif cmd == "/snap" or cmd.startswith("/snap "):
            # /snap → latest event; /snap abc123 → specific event
            parts = raw_text.split(None, 1)
            uuid_prefix = parts[1] if len(parts) > 1 else None
            send_message(chat_id, generate_snapshot_report(uuid_prefix))

        elif cmd == "/score":
            send_message(chat_id, generate_score_report())

        elif cmd == "/history":
            send_message(chat_id, generate_history_report())

        elif cmd == "/snap_status":
            send_message(chat_id, generate_snapshot_status_report())

        elif cmd == "/chart":
            # /chart → 4h indicator chart (core)
            threading.Thread(target=_handle_indicator_chart, args=(chat_id,), daemon=True).start()

        elif cmd == "/ichart":
            # /ichart → interactive chart link
            if INDICATOR_SERVICE_URL:
                url = f"{INDICATOR_SERVICE_URL}/live-chart"
                send_message(chat_id,
                    f"<b>Interactive Chart</b>\n\n"
                    f"<a href=\"{url}\">點擊開啟互動圖表</a>\n\n"
                    f"功能: 放大縮小 / 拖曳平移 / 十字線游標\n"
                    f"時間快選: 24h / 3d / 7d / All")
            else:
                send_message(chat_id, "INDICATOR_SERVICE_URL 未設定")

        elif cmd == "/ind_status":
            threading.Thread(target=_handle_indicator_status, args=(chat_id,), daemon=True).start()

        elif cmd == "/db":
            threading.Thread(target=_handle_indicator_db, args=(chat_id,), daemon=True).start()

        elif cmd == "/perf":
            threading.Thread(target=_handle_indicator_perf, args=(chat_id,), daemon=True).start()

        elif cmd == "/decay":
            threading.Thread(target=_handle_alpha_decay, args=(chat_id,), daemon=True).start()

        # 撤單流指令已全數移至獨立 bot（2026-07-19，cancel_bot_webhook）

        elif cmd == "/update":
            threading.Thread(target=_handle_force_update, args=(chat_id,), daemon=True).start()

        elif cmd == "/flow_chart" or cmd.startswith("/flow_chart "):
            # /flow_chart → BTC-USD 1h 7d (legacy flow chart)
            parts = raw_text.split()
            tf  = parts[1] if len(parts) > 1 else "1h"
            days = int(parts[2]) if len(parts) > 2 else 7
            send_message(chat_id, f"⏳ 產生圖表中 ({tf} {days}d)...")

            def _send_chart(cid, timeframe, lookback):
                try:
                    from research.storage.schema import ensure_schema
                    from research.bar_generator.runner import run_once
                    from research.viz.chart_builder import load_and_build
                    from research.config.settings import ChartConfig
                    ensure_schema()
                    run_once("BTC-USD", timeframe, lookback_days=lookback)
                    config = ChartConfig(symbol="BTC-USD", timeframe=timeframe, lookback_days=lookback)
                    fig = load_and_build("BTC-USD", timeframe, lookback_days=lookback, config=config)
                    img = fig.to_image(format="png", width=1600, height=900, scale=1.5)
                    send_photo(cid, img, caption=f"BTC-USD {timeframe} {lookback}d")
                except Exception as e:
                    logger.exception("chart command error: %s", e)
                    send_message(cid, f"❌ 圖表產生失敗: {e}")

            threading.Thread(target=_send_chart, args=(chat_id, tf, days), daemon=True).start()

        elif cmd in ["/start", "/help"]:
            _send_help(chat_id)

        elif cmd in ("/okx_perf", "/okxperf"):
            threading.Thread(
                target=_handle_okx_perf, args=(chat_id,), daemon=True,
            ).start()

        elif cmd.startswith("/yes_") or cmd.startswith("/no_"):
            # OKX Stage 3 manual-approval response.  /yes_<id> approves
            # the pending trade and submits it; /no_<id> denies.
            threading.Thread(
                target=_handle_okx_approval_response,
                args=(chat_id, raw_text),
                daemon=True,
            ).start()

        elif cmd.startswith("/okx_addacct"):
            # Follow-trading account registration (admin only). The raw
            # message carries API credentials — handler deletes it first.
            # NOTE: use the un-split original text — raw_text truncates at
            # "@" (bot-mention stripping) but a passphrase may contain "@".
            from indicator.okx.accounts import handle_addacct
            threading.Thread(
                target=handle_addacct,
                args=(chat_id, text.strip(), message.get("message_id")),
                daemon=True,
            ).start()

        elif cmd == "/okx_accounts":
            from indicator.okx.accounts import handle_accounts_list
            threading.Thread(
                target=handle_accounts_list, args=(chat_id,), daemon=True,
            ).start()

        elif cmd.startswith("/okx_pauseacct"):
            from indicator.okx.accounts import handle_acct_status
            threading.Thread(
                target=handle_acct_status,
                args=(chat_id, raw_text, "PAUSED"), daemon=True,
            ).start()

        elif cmd.startswith("/okx_resumeacct"):
            from indicator.okx.accounts import handle_acct_status
            threading.Thread(
                target=handle_acct_status,
                args=(chat_id, raw_text, "ACTIVE"), daemon=True,
            ).start()

        elif cmd.startswith("/okx_delacct"):
            from indicator.okx.accounts import handle_acct_status
            threading.Thread(
                target=handle_acct_status,
                args=(chat_id, raw_text, "DELETE"), daemon=True,
            ).start()

        else:
            send_message(chat_id, "❓未知指令，輸入 /help 查看支援功能")

        return "ok"

    except Exception as e:
        logger.exception("Webhook error: %s", e)
        return "ok"


# =========================================================
# 主程式
# =========================================================
def _snapshot_loop():
    """Background: compute pending snapshots every 60s."""
    import time as _time
    from market_data.features.snapshot_runner import process_once as _snap_once
    _time.sleep(10)  # brief delay so DB is ready after startup
    while True:
        try:
            _snap_once()
        except Exception:
            logger.exception("Snapshot runner error")
        _time.sleep(60)


def _oi_loop():
    """Background: collect OI every 60s."""
    import time as _time
    from market_data.adapters.oi_collector import collect_once as _oi_once
    while True:
        try:
            _oi_once()
        except Exception:
            logger.exception("OI collector error")
        _time.sleep(60)


def _funding_loop():
    """Background: collect funding rates every 60s."""
    import time as _time
    from market_data.adapters.funding_collector import collect_once as _fr_once
    while True:
        try:
            _fr_once()
        except Exception:
            logger.exception("Funding collector error")
        _time.sleep(60)


def _orderbook_l20_loop():
    """Background: collect Binance perp L20 orderbook snapshots every 60s.

    Pure historical accumulation for Phase 3 R&D (orderbook imbalance
    features).  Production v7 indicator does NOT read this table.
    """
    import time as _time
    from market_data.adapters.orderbook_l20_collector import (
        collect_once as _ob_once,
        _ensure_schema as _ob_ensure,
    )
    try:
        _ob_ensure()
    except Exception:
        logger.exception("Orderbook L20 schema setup failed")
    while True:
        try:
            _ob_once()
        except Exception:
            logger.exception("Orderbook L20 collector error")
        _time.sleep(60)


def start_background_threads():
    threading.Thread(target=start_ws_forever, daemon=True).start()
    threading.Thread(target=clean_old_data, daemon=True).start()
    threading.Thread(target=ws_watchdog, daemon=True).start()
    threading.Thread(target=event_watchdog, daemon=True).start()

    # outcome tracker 背景執行緒
    threading.Thread(
        target=outcome_tracker.outcome_watchdog,
        args=(get_db_conn, send_message, CHAT_ID),
        daemon=True
    ).start()
    threading.Thread(
        target=outcome_tracker.check_interim_notifications,
        args=(send_message, CHAT_ID),
        daemon=True
    ).start()

    # snapshot runner (每 60s 計算待處理快照)
    threading.Thread(target=_snapshot_loop, daemon=True, name="snapshot-runner").start()
    logger.info("Snapshot runner started.")

    # OI collector (每 60s REST 抓取)
    threading.Thread(target=_oi_loop, daemon=True, name="oi-collector").start()
    logger.info("OI collector started.")

    # Funding rate collector (每 60s REST 抓取)
    threading.Thread(target=_funding_loop, daemon=True, name="funding-collector").start()
    logger.info("Funding rate collector started.")

    # Orderbook L20 collector (每 60s REST 抓取, Phase 3 R&D accumulation only)
    threading.Thread(target=_orderbook_l20_loop, daemon=True, name="orderbook-l20-collector").start()
    logger.info("Orderbook L20 collector started.")

    # Liquidation collector (OKX + Binance WebSocket)
    try:
        from market_data.adapters.liquidation_collector import start_all as _liq_start
        _liq_start()
        logger.info("Liquidation collector started.")
    except Exception:
        logger.exception("Liquidation collector failed to start")


if __name__ == "__main__":
    logger.info("✅ BTC 流動性結果監控機器人啟動中... (v2)")
    logger.info("Observation seconds: %s (%dh)", EVENT_OBSERVATION_SECONDS, EVENT_OBSERVATION_SECONDS // 3600)
    logger.info("First hit levels: %s", FIRST_HIT_LEVELS)
    logger.info("Config source: %s", config["source"])
    logger.info("Port: %s", PORT)
    logger.info("Allowed users: %s", ALLOWED_USERS if ALLOWED_USERS else "ALL")
    logger.info("TV webhook secret: %s", "SET" if TV_WEBHOOK_SECRET else "NOT SET")
    logger.info("MySQL host: %s", MYSQL_HOST if MYSQL_HOST else "NOT SET")
    logger.info("MySQL db: %s", MYSQL_DB if MYSQL_DB else "NOT SET")

    try:
        init_db()
    except Exception as e:
        logger.exception("❌ init_db 失敗: %s", e)

    try:
        outcome_tracker.init_sweep_outcomes_table(get_db_conn)
    except Exception as e:
        logger.exception("❌ sweep_outcomes table init 失敗: %s", e)

    try:
        from market_data.adapters.oi_schema import ensure_oi_schema
        ensure_oi_schema()
    except Exception as e:
        logger.exception("OI schema init failed: %s", e)

    try:
        from market_data.adapters.extra_schema import ensure_extra_schema
        ensure_extra_schema()
    except Exception as e:
        logger.exception("Extra schema init failed: %s", e)

    # Snapshot / registry tables — create directly to avoid migration parser issues
    try:
        _conn = get_db_conn()
        try:
            with _conn.cursor() as _cur:
                _cur.execute("""
                CREATE TABLE IF NOT EXISTS event_registry (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    event_uuid VARCHAR(64) NOT NULL,
                    event_type VARCHAR(50) DEFAULT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    liquidity_side VARCHAR(20) NOT NULL,
                    entry_price DECIMAL(18,8) NOT NULL,
                    trigger_ts INT NOT NULL,
                    sweep_ref_price DECIMAL(18,8) DEFAULT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_event_uuid (event_uuid),
                    INDEX idx_trigger_ts (trigger_ts)
                )""")
                _cur.execute("""
                CREATE TABLE IF NOT EXISTS event_feature_snapshots (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    event_uuid VARCHAR(64) NOT NULL,
                    event_type VARCHAR(50) DEFAULT NULL,
                    canonical_symbol VARCHAR(20) NOT NULL,
                    liquidity_side VARCHAR(20) NOT NULL,
                    trigger_price DECIMAL(18,8) NOT NULL,
                    trigger_ts INT NOT NULL,
                    snapshot_type VARCHAR(10) NOT NULL,
                    snapshot_ts INT NOT NULL,
                    delta_value DECIMAL(30,10) DEFAULT NULL,
                    cvd_change DECIMAL(30,10) DEFAULT NULL,
                    cvd_sign_flip BOOLEAN DEFAULT NULL,
                    price_change_pct DECIMAL(10,4) DEFAULT NULL,
                    reclaim_flag BOOLEAN DEFAULT NULL,
                    break_again_flag BOOLEAN DEFAULT NULL,
                    reversal_score DECIMAL(10,4) NOT NULL DEFAULT 0,
                    continuation_score DECIMAL(10,4) NOT NULL DEFAULT 0,
                    confidence_score DECIMAL(10,4) NOT NULL DEFAULT 0,
                    bias VARCHAR(20) NOT NULL DEFAULT 'neutral',
                    label VARCHAR(20) DEFAULT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_event_snapshot (event_uuid, snapshot_type),
                    INDEX idx_trigger_ts (trigger_ts),
                    INDEX idx_snapshot_type (snapshot_type),
                    INDEX idx_bias (bias)
                )""")

                # Add lifecycle columns to event_registry (idempotent)
                for col_name, col_def in [
                    ("status",      "VARCHAR(20) DEFAULT 'active'"),
                    ("result_1h",   "VARCHAR(50) DEFAULT NULL"),
                    ("result_4h",   "VARCHAR(50) DEFAULT NULL"),
                    ("return_1h",   "DECIMAL(10,4) DEFAULT NULL"),
                    ("return_4h",   "DECIMAL(10,4) DEFAULT NULL"),
                    ("finished_at", "DATETIME DEFAULT NULL"),
                ]:
                    try:
                        _cur.execute(
                            f"ALTER TABLE event_registry ADD COLUMN {col_name} {col_def}"
                        )
                    except Exception:
                        pass  # column already exists

                # Add all extended columns to event_feature_snapshots (idempotent)
                for col_name, col_def in [
                    # OI columns
                    ("oi_baseline_okx",       "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_baseline_binance",   "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_snapshot_okx",       "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_snapshot_binance",   "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_change_okx",         "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_change_binance",     "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_change_okx_pct",     "DECIMAL(10,4) DEFAULT NULL"),
                    ("oi_change_binance_pct", "DECIMAL(10,4) DEFAULT NULL"),
                    ("oi_change_total",       "DECIMAL(30,4) DEFAULT NULL"),
                    ("oi_change_total_pct",   "DECIMAL(10,4) DEFAULT NULL"),
                    # Score columns
                    ("final_score",           "DECIMAL(10,4) DEFAULT NULL"),
                    ("normalized_score",      "DECIMAL(10,4) DEFAULT NULL"),
                    # Funding + liquidation columns
                    ("funding_rate",          "DECIMAL(20,8) DEFAULT NULL"),
                    ("liq_buy_usd",           "DECIMAL(30,4) DEFAULT NULL"),
                    ("liq_sell_usd",          "DECIMAL(30,4) DEFAULT NULL"),
                    ("liq_total_usd",         "DECIMAL(30,4) DEFAULT NULL"),
                    ("liq_count",             "INT DEFAULT NULL"),
                ]:
                    try:
                        _cur.execute(
                            f"ALTER TABLE event_feature_snapshots ADD COLUMN {col_name} {col_def}"
                        )
                    except Exception:
                        pass  # column already exists

            logger.info("✅ event_registry + event_feature_snapshots tables ready")
        finally:
            _conn.close()
    except Exception as e:
        logger.exception("snapshot tables init failed (may already exist): %s", e)

    start_background_threads()
    app.run(host=HOST, port=PORT)