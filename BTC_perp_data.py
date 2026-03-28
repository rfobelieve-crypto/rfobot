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
from market_data.query.flow_context import get_pre_sweep_context, get_event_flow_context, format_flow_context

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
        debug_mode = str(debug_raw or "true").lower() == "true"
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


def send_message(chat_id: str, text: str) -> None:
    if not chat_id:
        logger.warning("send_message skipped: chat_id is empty")
        return

    try:
        resp = requests.post(
            f"{API_URL}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=10
        )
        if resp.status_code != 200:
            logger.error("Telegram sendMessage failed: %s - %s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("send_message error: %s", e)


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

        logger.info("TV webhook received: %s", data)

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

        # 合併成一則通知
        msg_lines = [
            "📩 收到 TradingView 快訊",
            f"event: {event}",
            f"liquidity_side: {liquidity_side} ({tracker['event_type']})",
            f"price: {price}",
            f"time: {tv_time}",
            f"symbol: {symbol}",
            "─" * 30,
            f"[掃蕩追蹤] 已啟動 (±0.5%, 15m/1h)",
            f"  上目標: {tracker['upper_target']:.2f}",
            f"  下目標: {tracker['lower_target']:.2f}",
            f"  UUID: {tracker['event_uuid'][:8]}",
        ]

        if event_skipped:
            msg_lines.append("─" * 30)
            msg_lines.append("[flow 追蹤] 略過（已有事件進行中）")
        else:
            msg_lines.append("─" * 30)
            msg_lines.append(f"[flow 追蹤] 已啟動 (15m/1h/4h, first hit ±0.5%/±1.0%)")
            msg_lines.append(f"  entry_price: {new_event['entry_price']:.2f}")
            msg_lines.append(f"  session: {new_event['session']}")
            msg_lines.append(f"  UUID: {new_event['event_uuid'][:8]}")

        # Attach pre-sweep market flow context (OKX+Binance combined)
        try:
            pre_ctx = get_pre_sweep_context("BTC-USD", lookback_minutes=5)
            msg_lines.append("─" * 30)
            msg_lines.append(format_flow_context(pre_ctx, title="掃蕩前 5m 市場流"))
        except Exception as flow_err:
            logger.warning("Failed to get pre-sweep flow context: %s", flow_err)

        send_message(CHAT_ID, "\n".join(msg_lines))

        if event_skipped:
            return {"status": "partial", "reason": "sweep tracker started, delta event skipped"}, 200

        return {"status": "ok"}, 200

    except Exception as e:
        logger.exception("TradingView webhook error: %s", e)
        return {"status": "error", "message": str(e)}, 200


@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data:
            return "ok"

        message = data.get("message", {})
        chat = message.get("chat", {})
        text = message.get("text", "")

        if not chat or not text:
            return "ok"

        chat_id = str(chat.get("id", "")).strip()
        chat_type = str(chat.get("type", "")).strip().lower()
        text = text.strip().lower()

        if chat_type != "private":
            logger.warning("Rejected non-private chat: %s (%s)", chat_id, chat_type)
            return "ok"

        if ALLOWED_USERS and chat_id not in ALLOWED_USERS:
            logger.warning("Unauthorized access: %s", chat_id)
            return "ok"

        logger.info("Telegram command received: %s from %s", text, chat_id)

        if text == "/flow_futures_btc":
            send_message(chat_id, generate_report("BTC"))

        elif text == "/flow_futures_all":
            send_message(chat_id, generate_all_report())

        elif text == "/status":
            send_message(chat_id, generate_status_report())

        elif text == "/event_status":
            send_message(chat_id, generate_current_event_report())

        elif text == "/sweep_status":
            send_message(chat_id, outcome_tracker.format_active_trackers_report())

        elif text in ["/start", "/help"]:
            help_msg = (
                "📊 BTC 流動性結果監控機器人\n\n"
                "支援指令：\n"
                "/flow_futures_btc\n"
                "/flow_futures_all\n"
                "/status\n"
                "/event_status\n"
                "/sweep_status"
            )
            send_message(chat_id, help_msg)

        else:
            send_message(chat_id, "❓未知指令，輸入 /help 查看支援功能")

        return "ok"

    except Exception as e:
        logger.exception("Webhook error: %s", e)
        return "ok"


# =========================================================
# 主程式
# =========================================================
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

    start_background_threads()
    app.run(host=HOST, port=PORT)