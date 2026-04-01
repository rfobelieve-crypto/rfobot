"""
Flask app — serves indicator chart on Railway.

Routes:
    GET /        → latest chart PNG
    GET /health  → JSON health check
    GET /json    → latest prediction JSON

Auto-updates every 15 minutes via APScheduler.
"""
from __future__ import annotations

import os
import logging
import threading
from datetime import datetime, timezone

from flask import Flask, Response, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── State ────────────────────────────────────────────────────────────────────
_state = {
    "chart_png": b"",
    "last_update": None,
    "last_prediction": None,
    "status": "initializing",
    "error": None,
}
_lock = threading.Lock()


def update_cycle():
    """Fetch data, build features, predict, render chart."""
    from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
    from indicator.feature_builder_live import build_live_features
    from indicator.inference import IndicatorEngine
    from indicator.chart_renderer import render_chart

    global _engine
    if "_engine" not in globals() or _engine is None:
        _engine = IndicatorEngine()

    try:
        logger.info("Update cycle starting...")

        # 1. Fetch data
        klines = fetch_binance_klines(limit=500)
        cg_data = fetch_coinglass(interval="30m", limit=500)

        # 2. Build features
        features = build_live_features(klines, cg_data)

        # 3. Predict
        predictions = _engine.predict(features)

        # 4. Render chart
        png = render_chart(predictions, last_n=100)

        # 5. Update state
        last_row = predictions.iloc[-1]
        with _lock:
            _state["chart_png"] = png
            _state["last_update"] = datetime.now(timezone.utc).isoformat()
            _state["last_prediction"] = {
                "time": str(predictions.index[-1]),
                "direction": str(last_row["pred_direction"]),
                "confidence": float(last_row["confidence_score"]) if not _is_nan(last_row["confidence_score"]) else 0,
                "strength": str(last_row["strength_score"]),
                "pred_return_4h": float(last_row["pred_return_4h"]),
                "bull_bear_power": float(last_row["bull_bear_power"]),
                "close": float(last_row["close"]),
            }
            _state["status"] = "healthy"
            _state["error"] = None

        logger.info("Update complete: %s conf=%.0f %s",
                     last_row["pred_direction"],
                     last_row["confidence_score"] if not _is_nan(last_row["confidence_score"]) else 0,
                     last_row["strength_score"])

    except Exception as e:
        logger.exception("Update cycle failed")
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)


def _is_nan(v):
    try:
        import math
        return math.isnan(v)
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


@app.route("/json")
def prediction_json():
    with _lock:
        pred = _state["last_prediction"]
    if not pred:
        return jsonify({"error": "No prediction yet"}), 503
    return jsonify(pred)


# ── Scheduler ────────────────────────────────────────────────────────────────

_engine = None


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    # Run at :01, :16, :31, :46 (1 min after 15m bar close)
    scheduler.add_job(update_cycle, "cron", minute="1,16,31,46", misfire_grace_time=300)
    scheduler.start()
    logger.info("Scheduler started: updates at :01, :16, :31, :46")

    # Run first update immediately
    threading.Thread(target=update_cycle, daemon=True).start()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
