import os
import json
import time
import threading
import logging
import requests
import websocket

from flask import Flask, request
from datetime import datetime, timedelta, timezone

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
# 時間設定
# =========================================================
tz_taipei = timezone(timedelta(hours=8))

TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440
}

# =========================================================
# 幣種設定
# =========================================================
TRACK_SYMBOLS = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP"
}

# =========================================================
# 全域狀態
# =========================================================
data_lock = threading.Lock()
event_lock = threading.Lock()

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


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def create_event(event_type: str, side: str, price: float, symbol: str, tv_time: str):
    return {
        "event_type": event_type,
        "side": side,
        "price": price,
        "symbol": symbol,
        "tv_time": tv_time,
        "trigger_ts": current_ts(),
        "trigger_time_text": now_taipei_str(),
        "window_seconds": 60,
        "buy_amount": 0.0,
        "sell_amount": 0.0,
        "trade_count": 0,
        "finished": False,
        "symbol_stats": {
            "BTC": {
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "trade_count": 0
            },
            "ETH": {
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "trade_count": 0
            }
        }
    }


def generate_event_summary(event: dict) -> str:
    delta = event["buy_amount"] - event["sell_amount"]
    emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "🟡"

    btc_delta = (
        event["symbol_stats"]["BTC"]["buy_amount"] -
        event["symbol_stats"]["BTC"]["sell_amount"]
    )
    eth_delta = (
        event["symbol_stats"]["ETH"]["buy_amount"] -
        event["symbol_stats"]["ETH"]["sell_amount"]
    )

    btc_emoji = "🟢" if btc_delta > 0 else "🔴" if btc_delta < 0 else "🟡"
    eth_emoji = "🟢" if eth_delta > 0 else "🔴" if eth_delta < 0 else "🟡"

    lines = [
        "✅ 流動性事件完成",
        f"event: {event['event_type']}",
        f"side: {event['side']}",
        f"price: {event['price']}",
        f"symbol: {event['symbol']}",
        f"tv_time: {event['tv_time'] or 'N/A'}",
        f"trigger_time: {event['trigger_time_text']}",
        f"window: {event['window_seconds']}s",
        "─" * 30,
        f"total_buy: {format_number(event['buy_amount'])}",
        f"total_sell: {format_number(event['sell_amount'])}",
        f"total_delta: {format_number(delta)} {emoji}",
        f"total_trades: {event['trade_count']}",
        "─" * 30,
        f"BTC buy: {format_number(event['symbol_stats']['BTC']['buy_amount'])}",
        f"BTC sell: {format_number(event['symbol_stats']['BTC']['sell_amount'])}",
        f"BTC delta: {format_number(btc_delta)} {btc_emoji}",
        f"BTC trades: {event['symbol_stats']['BTC']['trade_count']}",
        "─" * 30,
        f"ETH buy: {format_number(event['symbol_stats']['ETH']['buy_amount'])}",
        f"ETH sell: {format_number(event['symbol_stats']['ETH']['sell_amount'])}",
        f"ETH delta: {format_number(eth_delta)} {eth_emoji}",
        f"ETH trades: {event['symbol_stats']['ETH']['trade_count']}",
    ]
    return "\n".join(lines)


def generate_current_event_report() -> str:
    with event_lock:
        if not current_event:
            return "目前沒有進行中的事件"

        age = current_ts() - current_event["trigger_ts"]
        delta = current_event["buy_amount"] - current_event["sell_amount"]

        return (
            f"🚧 事件進行中\n"
            f"event: {current_event['event_type']}\n"
            f"side: {current_event['side']}\n"
            f"price: {current_event['price']}\n"
            f"symbol: {current_event['symbol']}\n"
            f"tv_time: {current_event['tv_time'] or 'N/A'}\n"
            f"elapsed: {age}s / {current_event['window_seconds']}s\n"
            f"buy_amount: {format_number(current_event['buy_amount'])}\n"
            f"sell_amount: {format_number(current_event['sell_amount'])}\n"
            f"delta: {format_number(delta)}\n"
            f"trade_count: {current_event['trade_count']}"
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
        f"事件狀態：{event_status}\n"
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

            side = trade.get("side", "").lower()
            if side not in ("buy", "sell"):
                continue

            try:
                size = float(trade["sz"])
                price = float(trade["px"])
            except (KeyError, ValueError, TypeError):
                continue

            trade_ts = safe_get_trade_timestamp(trade)
            amount = size * price

            entry = {
                "timestamp": trade_ts,
                "amount": amount,
                "type": side
            }
            new_entries.append((symbol, entry))
            bot_status["last_trade_ts"] = trade_ts
            bot_status["total_trades"] += 1

            with event_lock:
                if current_event and not current_event["finished"]:
                    age = current_ts() - current_event["trigger_ts"]

                    if age <= current_event["window_seconds"]:
                        if side == "buy":
                            current_event["buy_amount"] += amount
                            current_event["symbol_stats"][symbol]["buy_amount"] += amount
                        elif side == "sell":
                            current_event["sell_amount"] += amount
                            current_event["symbol_stats"][symbol]["sell_amount"] += amount

                        current_event["trade_count"] += 1
                        current_event["symbol_stats"][symbol]["trade_count"] += 1

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

                    if age >= current_event["window_seconds"]:
                        current_event["finished"] = True
                        finished_event = dict(current_event)
                        current_event = None

            if finished_event:
                send_message(CHAT_ID, generate_event_summary(finished_event))
                logger.info("Event finished: %s", finished_event)

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
    return "OKX Taker Flow Bot is running."


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
        side = str(data.get("side", "unknown")).strip()
        price = safe_float(data.get("price", 0), 0.0)
        tv_time = str(data.get("time", "")).strip()
        symbol = str(data.get("symbol", "")).strip()

        new_event = create_event(event, side, price, symbol, tv_time)

        with event_lock:
            current_event = new_event

        msg = (
            "📩 收到 TradingView 快訊\n"
            f"event: {event}\n"
            f"side: {side}\n"
            f"price: {price}\n"
            f"time: {tv_time}\n"
            f"symbol: {symbol}\n"
            f"window: {new_event['window_seconds']}s\n"
            "事件監控已啟動"
        )

        send_message(CHAT_ID, msg)

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

        elif text == "/flow_futures_eth":
            send_message(chat_id, generate_report("ETH"))

        elif text == "/flow_futures_all":
            send_message(chat_id, generate_all_report())

        elif text == "/status":
            send_message(chat_id, generate_status_report())

        elif text == "/event_status":
            send_message(chat_id, generate_current_event_report())

        elif text in ["/start", "/help"]:
            help_msg = (
                "📊 合約資金流監控機器人\n\n"
                "支援指令：\n"
                "/flow_futures_btc\n"
                "/flow_futures_eth\n"
                "/flow_futures_all\n"
                "/status\n"
                "/event_status"
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


if __name__ == "__main__":
    logger.info("✅ Taker 資金動能監控機器人啟動中...")
    logger.info("Config source: %s", config["source"])
    logger.info("Port: %s", PORT)
    logger.info("Allowed users: %s", ALLOWED_USERS if ALLOWED_USERS else "ALL")
    logger.info("TV webhook secret: %s", "SET" if TV_WEBHOOK_SECRET else "NOT SET")

    start_background_threads()
    app.run(host=HOST, port=PORT)
