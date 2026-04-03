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
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from flask import Flask, Response, jsonify, request
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


def _make_reply_markup():
    """Inline keyboard with quick action buttons."""
    import json
    return json.dumps({"inline_keyboard": [
        [
            {"text": "\U0001f4ca Chart", "callback_data": "chart"},
            {"text": "\U0001f4cb Status", "callback_data": "status"},
        ],
    ]})


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
            "reply_markup": _make_reply_markup(),
        }, files={
            "photo": ("indicator.png", png, "image/png"),
        }, timeout=30)
        if resp.ok:
            logger.info("Telegram photo sent OK")
        else:
            logger.error("Telegram photo failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)


def _send_telegram_text(message: str, chat_id: str = ""):
    """Send HTML text alert to Telegram."""
    cid = chat_id or CHAT_ID
    if not BOT_TOKEN or not cid:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={
            "chat_id": cid,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=15)
    except Exception as e:
        logger.error("Telegram text send failed: %s", e)


def _send_telegram_photo_to(chat_id: str, png: bytes, caption: str):
    """Send chart PNG to a specific chat with inline buttons."""
    if not BOT_TOKEN or not chat_id:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        requests.post(url, data={
            "chat_id": chat_id,
            "reply_markup": _make_reply_markup(),
            "caption": caption,
        }, files={
            "photo": ("indicator.png", png, "image/png"),
        }, timeout=30)
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)


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
    from indicator.data_fetcher import (
        fetch_binance_klines, fetch_coinglass,
        fetch_binance_depth, fetch_binance_aggtrades,
        fetch_cg_options, fetch_cg_etf_flow,
        fetch_deribit_dvol, fetch_deribit_options_summary,
    )
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
        depth = fetch_binance_depth()
        aggtrades = fetch_binance_aggtrades()

        # Fetch options + ETF + DVOL data (direction signals — accumulating for future model)
        options_data = {}
        try:
            options_data.update(fetch_cg_options())
        except Exception as e:
            logger.warning("CG options fetch failed: %s", e)
        try:
            options_data.update(fetch_cg_etf_flow())
        except Exception as e:
            logger.warning("CG ETF flow fetch failed: %s", e)
        try:
            options_data.update(fetch_deribit_dvol())
        except Exception as e:
            logger.warning("Deribit DVOL fetch failed: %s", e)
        try:
            options_data.update(fetch_deribit_options_summary())
        except Exception as e:
            logger.warning("Deribit options fetch failed: %s", e)
        if options_data:
            logger.info("Options/ETF data: %d fields", len(options_data))

        # Diagnose Coinglass data quality
        cg_status = {}
        for name, df in cg_data.items():
            cg_status[name] = {"rows": len(df), "empty": df.empty}
        cg_ok = sum(1 for v in cg_status.values() if not v["empty"])
        cg_fail = sum(1 for v in cg_status.values() if v["empty"])
        logger.info("Coinglass status: %d/%d endpoints OK, failed: %s",
                     cg_ok, len(cg_status),
                     [k for k, v in cg_status.items() if v["empty"]] or "none")
        with _lock:
            _state["cg_status"] = cg_status

        if cg_fail == len(cg_status):
            logger.error("ALL Coinglass endpoints failed — BBP will be zero")

        # 2. Build features for ALL fetched bars
        features = build_live_features(klines, cg_data, depth=depth, aggtrades=aggtrades)

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
        TZ_TPE = timezone(timedelta(hours=8))
        now_str = datetime.now(TZ_TPE).strftime("%m/%d %H:%M UTC+8")

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
                f"⏰ {now_str}"
            )
            _send_telegram_text(alert)
            logger.info("Strong signal alert sent: %s", direction)

        logger.info("Update complete: %s conf=%.0f %s",
                     direction, conf, strength)

        # Save depth/aggtrades/options snapshots to MySQL + parquet
        try:
            from indicator.snapshot_collector import (
                save_depth_snapshot, save_aggtrades_snapshot,
                save_options_snapshot, save_indicator_history,
            )
            save_depth_snapshot(depth)
            save_aggtrades_snapshot(aggtrades)
            save_options_snapshot(options_data)
            save_indicator_history(indicator_df)
        except Exception as snap_err:
            logger.warning("Snapshot save failed (non-critical): %s", snap_err)

        # Run signal quality monitor (lightweight, won't block)
        try:
            from indicator.monitor_icir import run_monitor
            run_monitor()
        except Exception as mon_err:
            logger.warning("Monitor check failed (non-critical): %s", mon_err)

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
            "cg_status": _state.get("cg_status"),
        })


@app.route("/test-telegram")
def test_telegram():
    """Send a test message to verify Telegram integration."""
    # Debug: show all TELEGRAM* env vars (masked)
    tg_vars = {k: v[:6] + "****" for k, v in os.environ.items()
                if "TELEGRAM" in k.upper() or "TG" in k.upper() or "BOT" in k.upper()}
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return jsonify({"error": "Missing credentials",
                        "has_token": bool(token), "has_chat": bool(chat),
                        "env_keys_found": tg_vars}), 500
    _send_telegram_text("\u2705 Test from Railway indicator service.")
    with _lock:
        png = _state["chart_png"]
    if png:
        _send_telegram_photo(png, "Test chart from Railway")
    return jsonify({"status": "sent", "env_keys_found": tg_vars})


@app.route("/indicator-chart", methods=["GET"])
def indicator_chart_api():
    """API for main bot to fetch indicator chart + caption."""
    with _lock:
        png = _state["chart_png"]
        pred = _state["last_prediction"]
        last_update = _state["last_update"]
    if not png or not pred:
        return jsonify({"error": "Chart not ready"}), 503

    direction = pred.get("direction", "?")
    conf = pred.get("confidence", 0)
    strength = pred.get("strength", "?")
    pred_ret = pred.get("pred_return_4h", 0) * 100
    bbp = pred.get("bull_bear_power", 0)
    price = pred.get("close", 0)
    arrow = "\U0001f53c" if direction == "UP" else "\U0001f53d" if direction == "DOWN" else "\u2796"
    caption = (
        f"{arrow} BTC 4h Indicator\n"
        f"Price: ${price:,.0f}\n"
        f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
        f"Pred: {pred_ret:+.2f}% | BBP: {bbp:+.2f}\n"
        f"Updated: {last_update or '?'}"
    )
    import base64
    return jsonify({
        "caption": caption,
        "png_base64": base64.b64encode(png).decode(),
        "prediction": pred,
    })


@app.route("/json")
def prediction_json():
    with _lock:
        pred = _state["last_prediction"]
    if not pred:
        return jsonify({"error": "No prediction yet"}), 503
    return jsonify(pred)


@app.route("/indicator-status", methods=["GET"])
def indicator_status_api():
    """API for main bot to fetch indicator status text."""
    with _lock:
        status = _state["status"]
        last_update = _state["last_update"]
        pred = _state["last_prediction"]
        error = _state["error"]
    lines = [
        f"Status: {status}",
        f"Last update: {last_update or 'N/A'}",
    ]
    if pred:
        lines.append(f"Direction: {pred['direction']} ({pred['strength']})")
        lines.append(f"Confidence: {pred['confidence']:.0f}")
        lines.append(f"Price: ${pred['close']:,.0f}")
    if error:
        lines.append(f"Error: {error}")
    return jsonify({"text": "\n".join(lines)})


@app.route("/diag")
def diagnostics():
    """Live diagnostics — check Coinglass API and BBP pipeline."""
    import math

    # List ALL env var names in this container (no values, just names)
    all_env_names = sorted(os.environ.keys())
    cg_key_raw = os.environ.get("COINGLASS_API_KEY", "")
    diag = {
        "coinglass_api_key_set": bool(cg_key_raw),
        "coinglass_api_key_len": len(cg_key_raw),
        "all_env_var_names": all_env_names,
        "cg_status": _state.get("cg_status", "no update yet"),
    }

    with _lock:
        indicator_df = _state.get("indicator_df")
    if indicator_df is not None and not indicator_df.empty:
        last10 = indicator_df.tail(10)
        bbp_vals = last10["bull_bear_power"].tolist()
        diag["last_10_bbp"] = [
            round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None
            for v in bbp_vals
        ]
        diag["last_10_times"] = [str(t) for t in last10.index]
        diag["bbp_all_zero"] = all(v == 0 for v in bbp_vals if not (isinstance(v, float) and math.isnan(v)))
        diag["total_bars"] = len(indicator_df)
        diag["history_range"] = f"{indicator_df.index[0]} ~ {indicator_df.index[-1]}"
    else:
        diag["indicator_df"] = "not loaded yet"

    return jsonify(diag)


# ── Scheduler ────────────────────────────────────────────────────────────────

def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    scheduler.add_job(update_cycle, "cron", minute="2", misfire_grace_time=300)
    scheduler.start()
    logger.info("Scheduler started: updates at :02 every hour")

    # Run first update immediately
    threading.Thread(target=update_cycle, daemon=True).start()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
