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
EVENT_OBSERVATION_SECONDS = 3600  # 1 小時
OUTCOME_PCT = 0.01                # ±1%

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
# Session 判斷（依 UTC 時間）
# =========================================================
def determine_session(event_ts: int) -> str:
    """
    根據事件發生的 UTC 時間判斷交易時段。
    Asia:   00:00–08:00 UTC (台北 08:00–16:00)
    London: 08:00–13:00 UTC (台北 16:00–21:00)
    NY:     13:00–21:00 UTC (台北 21:00–05:00)
    其餘歸類為 Off-hours。
    可依需求調整時段邊界。
    """
    from datetime import timezone as _tz
    utc_hour = datetime.fromtimestamp(event_ts, tz=_tz.utc).hour

    if 0 <= utc_hour < 8:
        return "Asia"
    elif 8 <= utc_hour < 13:
        return "London"
    elif 13 <= utc_hour < 21:
        return "NY"
    else:
        return "Off-hours"


# =========================================================
# Reaction Type 判斷
# =========================================================
def determine_reaction_type(liquidity_side: str, first_hit_side: str) -> str:
    """
    判斷 first_hit_side 與 sweep 後預期延續方向的關係。

    BSL (buy / 掃高點)：
      預期延續方向 = upper（繼續往上）
      first_hit_side == upper → acceptance（市場接受突破）
      first_hit_side == lower → rejection（市場拒絕突破）

    SSL (sell / 掃低點)：
      預期延續方向 = lower（繼續往下）
      first_hit_side == lower → acceptance
      first_hit_side == upper → rejection

    未命中 → neutral
    """
    liquidity_side = str(liquidity_side or "").lower()
    first_hit_side = str(first_hit_side or "").lower()

    if first_hit_side == "none" or not first_hit_side:
        return "neutral"

    if liquidity_side == "buy":
        return "acceptance" if first_hit_side == "upper" else "rejection"
    elif liquidity_side == "sell":
        return "acceptance" if first_hit_side == "lower" else "rejection"

    return "unknown"


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
    if not MYSQL_HOST or not MYSQL_USER or not MYSQL_DB:
        raise ValueError("MySQL 環境變數未完整設定")

    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        ssl={"ssl": {}}
    )


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


def init_db():
    logger.info("🚀 開始初始化 MySQL 資料表...")

    conn = get_db_conn()
    try:
        sql = """
        CREATE TABLE IF NOT EXISTS liquidity_events (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            event_uuid VARCHAR(64) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            liquidity_side VARCHAR(20) NOT NULL,
            symbol VARCHAR(50) NOT NULL,
            trigger_price DECIMAL(18,8) NOT NULL,
            tv_time VARCHAR(64),
            trigger_ts INT NOT NULL,
            trigger_time_text VARCHAR(32),
            observation_seconds INT NOT NULL,

            upper_target DECIMAL(18,8) NOT NULL,
            lower_target DECIMAL(18,8) NOT NULL,

            first_hit_side VARCHAR(20) DEFAULT 'none',
            outcome_label VARCHAR(50) DEFAULT 'unknown',
            hit_price DECIMAL(18,8) DEFAULT NULL,
            hit_ts INT DEFAULT NULL,
            hit_latency_seconds INT DEFAULT NULL,

            flow_until_hit_buy DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_until_hit_sell DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_until_hit_delta DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_until_hit_trades INT NOT NULL DEFAULT 0,

            flow_1h_buy DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_1h_sell DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_1h_delta DECIMAL(20,8) NOT NULL DEFAULT 0,
            flow_1h_trades INT NOT NULL DEFAULT 0,

            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_event_uuid (event_uuid),
            INDEX idx_trigger_ts (trigger_ts),
            INDEX idx_liquidity_side (liquidity_side),
            INDEX idx_symbol (symbol),
            INDEX idx_outcome_label (outcome_label)
        );
        """
        with conn.cursor() as cursor:
            cursor.execute(sql)

        # 若舊表已存在，補欄位
        ensure_column(conn, "liquidity_events", "observation_seconds", "INT NOT NULL DEFAULT 3600")
        ensure_column(conn, "liquidity_events", "upper_target", "DECIMAL(18,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "lower_target", "DECIMAL(18,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "first_hit_side", "VARCHAR(20) DEFAULT 'none'")
        ensure_column(conn, "liquidity_events", "outcome_label", "VARCHAR(50) DEFAULT 'unknown'")
        ensure_column(conn, "liquidity_events", "hit_price", "DECIMAL(18,8) DEFAULT NULL")
        ensure_column(conn, "liquidity_events", "hit_ts", "INT DEFAULT NULL")
        ensure_column(conn, "liquidity_events", "hit_latency_seconds", "INT DEFAULT NULL")
        ensure_column(conn, "liquidity_events", "flow_until_hit_buy", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_until_hit_sell", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_until_hit_delta", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_until_hit_trades", "INT NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_1h_buy", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_1h_sell", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_1h_delta", "DECIMAL(20,8) NOT NULL DEFAULT 0")
        ensure_column(conn, "liquidity_events", "flow_1h_trades", "INT NOT NULL DEFAULT 0")

        # ── 新增欄位：edge 統計用 ──
        ensure_column(conn, "liquidity_events", "sweep_ref_price", "DECIMAL(18,8) DEFAULT NULL COMMENT '被掃的前高/前低價格'")
        ensure_column(conn, "liquidity_events", "sweep_size_pct", "DECIMAL(10,4) DEFAULT NULL COMMENT '掃蕩幅度百分比'")
        ensure_column(conn, "liquidity_events", "entry_time", "VARCHAR(32) DEFAULT NULL COMMENT '進場時間（K棒收盤）'")
        ensure_column(conn, "liquidity_events", "entry_price", "DECIMAL(18,8) DEFAULT NULL COMMENT '進場價格（K棒收盤價）'")
        ensure_column(conn, "liquidity_events", "return_5m", "DECIMAL(10,4) DEFAULT NULL COMMENT '進場後5分鐘報酬率%'")
        ensure_column(conn, "liquidity_events", "return_15m", "DECIMAL(10,4) DEFAULT NULL COMMENT '進場後15分鐘報酬率%'")
        ensure_column(conn, "liquidity_events", "return_1h", "DECIMAL(10,4) DEFAULT NULL COMMENT '進場後1小時報酬率%'")
        ensure_column(conn, "liquidity_events", "session", "VARCHAR(20) DEFAULT NULL COMMENT 'Asia/London/NY/Off-hours'")
        ensure_column(conn, "liquidity_events", "reaction_type", "VARCHAR(20) DEFAULT NULL COMMENT 'acceptance/rejection/neutral'")

        logger.info("✅ MySQL table 檢查 / 建立完成")
    finally:
        conn.close()


def save_event_to_db(event: dict):
    sql = """
    INSERT INTO liquidity_events (
        event_uuid, event_type, liquidity_side, symbol, trigger_price,
        tv_time, trigger_ts, trigger_time_text, observation_seconds,
        upper_target, lower_target,
        first_hit_side, outcome_label, hit_price, hit_ts, hit_latency_seconds,
        flow_until_hit_buy, flow_until_hit_sell, flow_until_hit_delta, flow_until_hit_trades,
        flow_1h_buy, flow_1h_sell, flow_1h_delta, flow_1h_trades,
        sweep_ref_price, sweep_size_pct,
        entry_time, entry_price,
        return_5m, return_15m, return_1h,
        session, reaction_type
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s
    )
    """

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                event["event_uuid"],
                event["event_type"],
                event["liquidity_side"],
                event["symbol"],
                event["price"],
                event["tv_time"],
                event["trigger_ts"],
                event["trigger_time_text"],
                event["observation_seconds"],
                event["upper_target"],
                event["lower_target"],
                event["first_hit_side"],
                event["outcome_label"],
                event["hit_price"],
                event["hit_ts"],
                event["hit_latency_seconds"],
                event["flow_until_hit_buy"],
                event["flow_until_hit_sell"],
                event["flow_until_hit_buy"] - event["flow_until_hit_sell"],
                event["flow_until_hit_trades"],
                event["flow_1h_buy"],
                event["flow_1h_sell"],
                event["flow_1h_buy"] - event["flow_1h_sell"],
                event["flow_1h_trades"],
                event.get("sweep_ref_price"),
                event.get("sweep_size_pct"),
                event.get("entry_time"),
                event.get("entry_price"),
                event.get("return_5m"),
                event.get("return_15m"),
                event.get("return_1h"),
                event.get("session"),
                event.get("reaction_type"),
            ))
    finally:
        conn.close()


# =========================================================
# Event
# =========================================================
def create_event(event_type: str, liquidity_side: str, price: float,
                  symbol: str, tv_time: str, sweep_ref_price: float = None):
    upper_target = price * (1 + OUTCOME_PCT)
    lower_target = price * (1 - OUTCOME_PCT)

    trigger_ts = current_ts()

    # ── entry_time：事件觸發當下那根 1 分鐘 K 棒的收盤時間（秒級 UTC timestamp）
    # 取天花板到下一分鐘邊界，例如 13:42:37 → 13:43:00
    entry_candle_close_ts = (trigger_ts // 60 + 1) * 60

    # ── sweep_size_pct：掃蕩幅度百分比
    sweep_size_pct = None
    if sweep_ref_price and sweep_ref_price > 0:
        if liquidity_side == "buy":
            # BSL: trigger_price 高於 ref (前高)
            sweep_size_pct = (price - sweep_ref_price) / sweep_ref_price * 100
        elif liquidity_side == "sell":
            # SSL: trigger_price 低於 ref (前低)
            sweep_size_pct = (sweep_ref_price - price) / sweep_ref_price * 100

    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": event_type,
        "liquidity_side": liquidity_side,
        "price": price,
        "symbol": symbol,
        "tv_time": tv_time,
        "trigger_ts": trigger_ts,
        "trigger_time_text": now_taipei_str(),
        "observation_seconds": EVENT_OBSERVATION_SECONDS,

        "upper_target": upper_target,
        "lower_target": lower_target,

        # 狀態
        "status": "active",         # active / hit_locked / finished
        "finished": False,

        # hit 結果
        "first_hit_side": "none",   # upper / lower / none
        "outcome_label": "unknown",
        "hit_price": None,
        "hit_ts": None,
        "hit_latency_seconds": None,

        # until hit 統計
        "flow_until_hit_buy": 0.0,
        "flow_until_hit_sell": 0.0,
        "flow_until_hit_trades": 0,

        # 完整 1h 統計
        "flow_1h_buy": 0.0,
        "flow_1h_sell": 0.0,
        "flow_1h_trades": 0,

        # ── 新增欄位：掃蕩參考價 & 幅度 ──
        "sweep_ref_price": sweep_ref_price,       # 被掃的前高 / 前低價格
        "sweep_size_pct": sweep_size_pct,          # 掃蕩幅度 %

        # ── 新增欄位：固定進場 ──
        "entry_candle_close_ts": entry_candle_close_ts,  # 內部用，K 棒收盤 ts
        "entry_time": None,                        # K 棒收盤時間（格式化字串）
        "entry_price": None,                       # K 棒收盤價（最後一筆成交價）
        "_entry_price_locked": False,               # 內部狀態：entry_price 是否已鎖定

        # ── 新增欄位：forward return ──
        "return_5m": None,                         # 進場後 5 分鐘報酬率 %
        "return_15m": None,                        # 進場後 15 分鐘報酬率 %
        "return_1h": None,                         # 進場後 1 小時報酬率 %
        "_return_5m_ts": entry_candle_close_ts + 300,    # 內部：5m 快照時間
        "_return_15m_ts": entry_candle_close_ts + 900,   # 內部：15m 快照時間
        "_return_1h_ts": entry_candle_close_ts + 3600,   # 內部：1h 快照時間

        # ── 新增欄位：市場環境 ──
        "session": determine_session(trigger_ts),  # Asia / London / NY / Off-hours

        # ── 新增欄位：reaction type ──
        "reaction_type": None,                     # acceptance / rejection / neutral
    }


def detect_first_hit(event: dict, price: float, trade_ts: int):
    """
    只要碰到就算 hit。
    hit 後立刻鎖定 label，但事件仍持續到 observation_seconds 結束，
    用來收完整 1h flow，同時這段期間不接受新事件。
    同時設定 reaction_type。
    """
    if event["status"] != "active":
        return

    hit_side = None

    if price >= event["upper_target"]:
        hit_side = "upper"
    elif price <= event["lower_target"]:
        hit_side = "lower"

    if not hit_side:
        return

    event["first_hit_side"] = hit_side
    event["outcome_label"] = outcome_label_from_hit(event["liquidity_side"], hit_side)
    event["hit_price"] = float(price)
    event["hit_ts"] = int(trade_ts)
    event["hit_latency_seconds"] = max(0, int(trade_ts) - int(event["trigger_ts"]))
    event["status"] = "hit_locked"

    # 設定 reaction_type
    event["reaction_type"] = determine_reaction_type(event["liquidity_side"], hit_side)


def finalize_neutral_if_needed(event: dict):
    """
    若 observation_seconds 到時仍未 hit，標記 neutral。
    同時設定 reaction_type 為 neutral。
    """
    if event["first_hit_side"] == "none":
        event["outcome_label"] = outcome_label_from_hit(event["liquidity_side"], "none")
        event["reaction_type"] = "neutral"
        event["status"] = "hit_locked"


def generate_event_summary(event: dict) -> str:
    flow_until_hit_delta = event["flow_until_hit_buy"] - event["flow_until_hit_sell"]
    flow_1h_delta = event["flow_1h_buy"] - event["flow_1h_sell"]

    until_hit_emoji = "🟢" if flow_until_hit_delta > 0 else "🔴" if flow_until_hit_delta < 0 else "🟡"
    flow_1h_emoji = "🟢" if flow_1h_delta > 0 else "🔴" if flow_1h_delta < 0 else "🟡"

    hit_price_text = f"{event['hit_price']}" if event["hit_price"] is not None else "N/A"
    hit_ts_text = (
        datetime.fromtimestamp(event["hit_ts"], tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
        if event["hit_ts"] else "N/A"
    )

    # 新增欄位的顯示值
    entry_price_text = f"{event['entry_price']:.2f}" if event.get("entry_price") else "N/A"
    entry_time_text = event.get("entry_time") or "N/A"
    sweep_ref_text = f"{event['sweep_ref_price']:.2f}" if event.get("sweep_ref_price") else "N/A"
    sweep_size_text = f"{event['sweep_size_pct']:.4f}%" if event.get("sweep_size_pct") is not None else "N/A"

    def fmt_return(val):
        if val is None:
            return "N/A"
        emoji = "🟢" if val > 0 else "🔴" if val < 0 else "🟡"
        return f"{val:+.4f}% {emoji}"

    lines = [
        "✅ 流動性事件完成",
        f"event: {event['event_type']}",
        f"liquidity_side: {event['liquidity_side']}",
        f"symbol: {event['symbol']}",
        f"trigger_price: {event['price']}",
        f"upper_target: {event['upper_target']:.2f}",
        f"lower_target: {event['lower_target']:.2f}",
        f"tv_time: {event['tv_time'] or 'N/A'}",
        f"trigger_time: {event['trigger_time_text']}",
        f"observation: {format_duration_minutes(event['observation_seconds'])}",
        f"session: {event.get('session', 'N/A')}",
        f"event_uuid: {event['event_uuid']}",
        "─" * 30,
        f"sweep_ref_price: {sweep_ref_text}",
        f"sweep_size_pct: {sweep_size_text}",
        f"entry_time: {entry_time_text}",
        f"entry_price: {entry_price_text}",
        "─" * 30,
        f"first_hit_side: {event['first_hit_side']}",
        f"outcome_label: {event['outcome_label']}",
        f"reaction_type: {event.get('reaction_type') or 'N/A'}",
        f"hit_price: {hit_price_text}",
        f"hit_time: {hit_ts_text}",
        f"hit_latency_seconds: {event['hit_latency_seconds'] if event['hit_latency_seconds'] is not None else 'N/A'}",
        "─" * 30,
        f"return_5m: {fmt_return(event.get('return_5m'))}",
        f"return_15m: {fmt_return(event.get('return_15m'))}",
        f"return_1h: {fmt_return(event.get('return_1h'))}",
        "─" * 30,
        f"flow_until_hit_buy: {format_number(event['flow_until_hit_buy'])}",
        f"flow_until_hit_sell: {format_number(event['flow_until_hit_sell'])}",
        f"flow_until_hit_delta: {format_number(flow_until_hit_delta)} {until_hit_emoji}",
        f"flow_until_hit_trades: {event['flow_until_hit_trades']}",
        "─" * 30,
        f"flow_1h_buy: {format_number(event['flow_1h_buy'])}",
        f"flow_1h_sell: {format_number(event['flow_1h_sell'])}",
        f"flow_1h_delta: {format_number(flow_1h_delta)} {flow_1h_emoji}",
        f"flow_1h_trades: {event['flow_1h_trades']}",
    ]
    return "\n".join(lines)


def generate_current_event_report() -> str:
    with event_lock:
        if not current_event:
            return "目前沒有進行中的事件"

        age = current_ts() - current_event["trigger_ts"]
        flow_until_hit_delta = current_event["flow_until_hit_buy"] - current_event["flow_until_hit_sell"]
        flow_1h_delta = current_event["flow_1h_buy"] - current_event["flow_1h_sell"]

        return (
            f"🚧 事件進行中\n"
            f"event: {current_event['event_type']}\n"
            f"liquidity_side: {current_event['liquidity_side']}\n"
            f"symbol: {current_event['symbol']}\n"
            f"trigger_price: {current_event['price']}\n"
            f"upper_target: {current_event['upper_target']:.2f}\n"
            f"lower_target: {current_event['lower_target']:.2f}\n"
            f"status: {current_event['status']}\n"
            f"first_hit_side: {current_event['first_hit_side']}\n"
            f"outcome_label: {current_event['outcome_label']}\n"
            f"elapsed: {age}s / {current_event['observation_seconds']}s\n"
            f"hit_latency_seconds: {current_event['hit_latency_seconds'] if current_event['hit_latency_seconds'] is not None else 'N/A'}\n"
            f"flow_until_hit_delta: {format_number(flow_until_hit_delta)}\n"
            f"flow_1h_delta: {format_number(flow_1h_delta)}\n"
            f"event_uuid: {current_event['event_uuid']}"
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

    mysql_status = "已設定" if MYSQL_HOST and MYSQL_USER and MYSQL_DB else "未設定完整"

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
        f"觀察時間：{EVENT_OBSERVATION_SECONDS} 秒 ({format_duration_minutes(EVENT_OBSERVATION_SECONDS)})\n"
        f"Outcome 門檻：±{OUTCOME_PCT * 100:.2f}%\n"
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
                    age = current_ts() - current_event["trigger_ts"]

                    if age <= current_event["observation_seconds"]:
                        # 完整 1h flow：永遠累加到 1 小時結束
                        if trade_side == "buy":
                            current_event["flow_1h_buy"] += amount
                        else:
                            current_event["flow_1h_sell"] += amount
                        current_event["flow_1h_trades"] += 1

                        # until hit：只有還沒 hit 時才累加
                        if current_event["first_hit_side"] == "none":
                            if trade_side == "buy":
                                current_event["flow_until_hit_buy"] += amount
                            else:
                                current_event["flow_until_hit_sell"] += amount
                            current_event["flow_until_hit_trades"] += 1

                            detect_first_hit(current_event, price, trade_ts)

                    # ── 新增：entry_price 追蹤 ──
                    # 在觸發 K 棒結束前，持續更新 entry_price（最後一筆 = 收盤價）
                    if not current_event.get("_entry_price_locked"):
                        if trade_ts < current_event["entry_candle_close_ts"]:
                            # 還在同一根 K 棒內，更新最新價
                            current_event["entry_price"] = price
                        else:
                            # K 棒已收盤，鎖定 entry_price
                            if current_event["entry_price"] is None:
                                current_event["entry_price"] = price
                            current_event["entry_time"] = datetime.fromtimestamp(
                                current_event["entry_candle_close_ts"], tz_taipei
                            ).strftime("%Y-%m-%d %H:%M:%S")
                            current_event["_entry_price_locked"] = True

                    # ── 新增：forward return 快照 ──
                    # 當交易時間超過各快照時間點，記錄報酬率
                    if current_event.get("_entry_price_locked") and current_event["entry_price"]:
                        ep = current_event["entry_price"]
                        ls = current_event["liquidity_side"]

                        # 報酬率方向：BSL 做空 → 價格下跌為正；SSL 做多 → 價格上漲為正
                        # 這裡統一用「若 sweep 後反轉的方向」為正報酬
                        if ls == "buy":
                            # BSL 預期反轉向下 → short → return = (entry - current) / entry * 100
                            sign = -1.0
                        else:
                            # SSL 預期反轉向上 → long → return = (current - entry) / entry * 100
                            sign = 1.0

                        if current_event["return_5m"] is None and trade_ts >= current_event["_return_5m_ts"]:
                            current_event["return_5m"] = round(sign * (price - ep) / ep * 100, 4)
                        if current_event["return_15m"] is None and trade_ts >= current_event["_return_15m_ts"]:
                            current_event["return_15m"] = round(sign * (price - ep) / ep * 100, 4)
                        if current_event["return_1h"] is None and trade_ts >= current_event["_return_1h_ts"]:
                            current_event["return_1h"] = round(sign * (price - ep) / ep * 100, 4)

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
                        finalize_neutral_if_needed(current_event)
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
            msg_lines.append("[delta 追蹤] 略過（已有事件進行中）")
        else:
            msg_lines.append("─" * 30)
            msg_lines.append(f"[delta 追蹤] 已啟動 (±1%, {format_duration_minutes(new_event['observation_seconds'])})")
            msg_lines.append(f"  上目標: {new_event['upper_target']:.2f}")
            msg_lines.append(f"  下目標: {new_event['lower_target']:.2f}")
            msg_lines.append(f"  UUID: {new_event['event_uuid'][:8]}")

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
    logger.info("✅ BTC 流動性結果監控機器人啟動中...")
    logger.info("Observation seconds: %s", EVENT_OBSERVATION_SECONDS)
    logger.info("Outcome pct: %.4f", OUTCOME_PCT)
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