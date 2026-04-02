"""
Flask app — serves indicator chart on Railway.

Routes:
    GET /        → latest chart PNG
    GET /health  → JSON health check
    GET /json    → latest prediction JSON

Architecture:
    - Loads pre-computed indicator history (from local walk-forward pipeline)
    - Every 1h: fetches new data, predicts ONLY new bars, appends to history
    - Chart shows historical (identical to local) + live tail
"""
from __future__ import annotations

import os
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from flask import Flask, Response, jsonify
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
HISTORY_PATH = ARTIFACT_DIR / "indicator_history.parquet"

# ── State ────────────────────────────────────────────────────────────────────
_state = {
    "chart_png": b"",
    "last_update": None,
    "last_prediction": None,
    "status": "initializing",
    "error": None,
    "indicator_df": None,  # full indicator DataFrame (history + live)
}
_lock = threading.Lock()
_engine = None

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _send_telegram_photo(png: bytes, caption: str):
    """Send chart PNG to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not set, skipping photo send")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        resp = requests.post(url, data={
            "chat_id": CHAT_ID,
            "caption": caption,
        }, files={
            "photo": ("indicator.png", png, "image/png"),
        }, timeout=30)
        if resp.ok:
            logger.info("Telegram photo sent OK")
        else:
            logger.error("Telegram photo failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)


def _send_telegram_text(message: str):
    """Send HTML text alert to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=15)
    except Exception as e:
        logger.error("Telegram text send failed: %s", e)


def _load_history() -> pd.DataFrame:
    """Load pre-computed indicator history from local pipeline."""
    if HISTORY_PATH.exists():
        df = pd.read_parquet(HISTORY_PATH)
        logger.info("Loaded indicator history: %d bars, %s ~ %s",
                     len(df), df.index[0], df.index[-1])
        return df
    logger.warning("No indicator history found at %s", HISTORY_PATH)
    return pd.DataFrame()


def update_cycle():
    """Fetch new data, predict only new bars, append to history."""
    from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
    from indicator.feature_builder_live import build_live_features
    from indicator.inference import IndicatorEngine
    from indicator.chart_renderer import render_chart

    global _engine

    try:
        logger.info("Update cycle starting...")

        # Initialize engine on first run
        if _engine is None:
            _engine = IndicatorEngine()

        # Load or get existing indicator data
        with _lock:
            indicator_df = _state["indicator_df"]
        if indicator_df is None:
            indicator_df = _load_history()

        # 1. Fetch live data
        klines = fetch_binance_klines(limit=500)
        cg_data = fetch_coinglass(interval="1h", limit=500)

        # 2. Build features for ALL fetched bars
        features = build_live_features(klines, cg_data)

        # 3. Backfill OHLC into history from klines (history only has 'close')
        for col in ["open", "high", "low"]:
            if col in klines.columns and (col not in indicator_df.columns or indicator_df[col].isna().all()):
                indicator_df[col] = klines[col].reindex(indicator_df.index)
        # Update close from klines for better accuracy
        if "close" in klines.columns:
            kline_close = klines["close"].reindex(indicator_df.index)
            mask = kline_close.notna()
            indicator_df.loc[mask, "close"] = kline_close[mask]
            for col in ["open", "high", "low"]:
                if col in klines.columns:
                    kline_col = klines[col].reindex(indicator_df.index)
                    m = kline_col.notna()
                    indicator_df.loc[m, col] = kline_col[m]

        # 4. Find new bars (after history ends)
        if not indicator_df.empty:
            last_hist_time = indicator_df.index[-1]
            new_features = features[features.index > last_hist_time]
        else:
            new_features = features

        # 5. Predict only new bars
        if len(new_features) > 0:
            new_predictions = _engine.predict(new_features)
            indicator_df = pd.concat([indicator_df, new_predictions])
            indicator_df = indicator_df[~indicator_df.index.duplicated(keep="last")]
            indicator_df = indicator_df.sort_index()
            logger.info("Appended %d new bars. Total: %d", len(new_predictions), len(indicator_df))
        else:
            logger.info("No new bars to predict")

        # 6. Render chart (last 200 bars that have OHLC data)
        chart_df = indicator_df.dropna(subset=["open", "high", "low", "close"])
        png = render_chart(chart_df, last_n=200)

        # 6. Update state
        last_row = indicator_df.iloc[-1]
        with _lock:
            _state["indicator_df"] = indicator_df
            _state["chart_png"] = png
            _state["last_update"] = datetime.now(timezone.utc).isoformat()
            _state["last_prediction"] = {
                "time": str(indicator_df.index[-1]),
                "direction": str(last_row.get("pred_direction", "NEUTRAL")),
                "confidence": float(last_row["confidence_score"]) if not _is_nan(last_row.get("confidence_score")) else 0,
                "strength": str(last_row.get("strength_score", "Weak")),
                "pred_return_4h": float(last_row.get("pred_return_4h", 0)),
                "bull_bear_power": float(last_row.get("bull_bear_power", 0)),
                "close": float(last_row["close"]) if "close" in last_row.index else 0,
            }
            _state["status"] = "healthy"
            _state["error"] = None

        # 7. Send to Telegram
        direction = str(last_row.get("pred_direction", "?"))
        conf = float(last_row["confidence_score"]) if not _is_nan(last_row.get("confidence_score")) else 0
        strength = str(last_row.get("strength_score", "Weak"))
        pred_ret = float(last_row.get("pred_return_4h", 0)) * 100
        bbp = float(last_row.get("bull_bear_power", 0))
        price = float(last_row["close"]) if "close" in last_row.index else 0
        now_str = datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC")

        arrow = "\U0001f53c" if direction == "UP" else "\U0001f53d" if direction == "DOWN" else "\u2796"
        caption = (
            f"{arrow} BTC 4h Indicator | {now_str}\n"
            f"Price: ${price:,.0f}\n"
            f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
            f"Pred: {pred_ret:+.2f}% | BBP: {bbp:+.2f}"
        )
        _send_telegram_photo(png, caption)

        # Strong signal alert
        if strength == "Strong" and direction in ("UP", "DOWN"):
            regime = str(last_row.get("regime", "?"))
            icon = "\U0001f7e2" if direction == "UP" else "\U0001f534"
            alert = (
                f"{icon} <b>STRONG {direction} SIGNAL</b>\n\n"
                f"BTC ${price:,.0f}\n"
                f"Pred: {pred_ret:+.2f}% (4h)\n"
                f"Confidence: {conf:.0f}\n"
                f"Regime: {regime}\n\n"
                f"{now_str}"
            )
            _send_telegram_text(alert)
            logger.info("Strong signal alert sent: %s", direction)

        logger.info("Update complete: %s conf=%.0f %s",
                     direction, conf, strength)

    except Exception as e:
        logger.exception("Update cycle failed")
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)


def _is_nan(v):
    try:
        import math
        return v is None or math.isnan(v)
    except (TypeError, ValueError):
        return False


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def chart():
    with _lock:
        png = _state["chart_png"]
    if not png:
        return Response("Chart not ready yet. Please wait for first update cycle.",
                        status=503)
    return Response(png, mimetype="image/png")


@app.route("/health")
def health():
    with _lock:
        return jsonify({
            "status": _state["status"],
            "last_update": _state["last_update"],
            "last_prediction": _state["last_prediction"],
            "error": _state["error"],
        })


@app.route("/test-telegram")
def test_telegram():
    """Send a test message to verify Telegram integration."""
    has_token = bool(BOT_TOKEN)
    has_chat = bool(CHAT_ID)
    if not has_token or not has_chat:
        return jsonify({"error": "Missing credentials",
                        "has_token": has_token, "has_chat": has_chat}), 500
    _send_telegram_text("Test from Railway indicator service.")
    with _lock:
        png = _state["chart_png"]
    if png:
        _send_telegram_photo(png, "Test chart from Railway")
    return jsonify({"status": "sent", "chat_id": CHAT_ID[:4] + "****"})


@app.route("/json")
def prediction_json():
    with _lock:
        pred = _state["last_prediction"]
    if not pred:
        return jsonify({"error": "No prediction yet"}), 503
    return jsonify(pred)


# ── Scheduler ────────────────────────────────────────────────────────────────

def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(update_cycle, "cron", minute="0", misfire_grace_time=300)
    scheduler.start()
    logger.info("Scheduler started: updates at :00 every hour")

    # Run first update immediately
    threading.Thread(target=update_cycle, daemon=True).start()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
