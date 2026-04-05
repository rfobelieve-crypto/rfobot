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

import numpy as np
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
            {"text": "\U0001f9f9 Sweep", "callback_data": "sweep"},
            {"text": "\u2753 Help", "callback_data": "help"},
        ],
    ]})


def _send_telegram_photo(png: bytes, caption: str) -> str:
    """Send chart PNG to Telegram. Returns status string for diagnostics."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not set, skipping photo send")
        return f"skipped: token={'set' if BOT_TOKEN else 'MISSING'}, chat={'set' if CHAT_ID else 'MISSING'}"
    if not png or len(png) < 100:
        logger.error("Chart PNG is empty or too small (%d bytes)", len(png) if png else 0)
        return f"skipped: png_empty ({len(png) if png else 0} bytes)"
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
            return "ok"
        else:
            logger.error("Telegram photo failed: %s %s", resp.status_code, resp.text[:200])
            return f"failed: {resp.status_code} {resp.text[:100]}"
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)
        return f"error: {e}"


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
    """Load indicator history: parquet (git) + MySQL (runtime rows after deploy)."""
    # Step 1: load parquet base (shipped with git)
    if HISTORY_PATH.exists():
        df = pd.read_parquet(HISTORY_PATH)
        logger.info("Loaded parquet history: %d bars, %s ~ %s",
                     len(df), df.index[0], df.index[-1])
    else:
        df = pd.DataFrame()
        logger.warning("No parquet history at %s", HISTORY_PATH)

    # Step 2: backfill from MySQL (rows added after last deploy)
    try:
        df = _backfill_from_mysql(df)
    except Exception as e:
        logger.warning("MySQL backfill failed (non-critical): %s", e)

    return df


def _backfill_from_mysql(df: pd.DataFrame) -> pd.DataFrame:
    """Merge MySQL indicator_history rows that are missing from parquet."""
    from shared.db import get_db_conn

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            if df.empty:
                cur.execute("SELECT * FROM indicator_history ORDER BY dt")
            else:
                last_dt = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                cur.execute("SELECT * FROM indicator_history WHERE dt > %s ORDER BY dt", (last_dt,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("MySQL backfill: no new rows")
        return df

    # Direction/strength/regime code → string mapping
    dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
    str_map = {3: "Strong", 2: "Moderate", 1: "Weak"}
    regime_map = {2: "TRENDING_BULL", -2: "TRENDING_BEAR", 0: "CHOPPY", -99: "WARMUP"}

    records = []
    for r in rows:
        records.append({
            "dt": pd.Timestamp(r["dt"], tz="UTC"),
            "open": r.get("open", 0),
            "high": r.get("high", 0),
            "low": r.get("low", 0),
            "close": r.get("close", 0),
            "pred_return_4h": r.get("pred_return_4h", 0),
            "pred_direction": dir_map.get(int(r.get("pred_direction_code", 0)), "NEUTRAL"),
            "confidence_score": r.get("confidence_score", 0),
            "strength_score": str_map.get(int(r.get("strength_code", 1)), "Weak"),
            "bull_bear_power": r.get("bull_bear_power", 0),
            "regime": regime_map.get(int(r.get("regime_code", 0)), "CHOPPY"),
            "up_pred": r.get("up_pred", 0),
            "down_pred": r.get("down_pred", 0),
            "strength_raw": r.get("strength_raw", 0),
            "dynamic_deadzone": r.get("dynamic_deadzone", 0),
            "dir_prob_up": r.get("dir_prob_up", 0.5),
            "mag_pred": r.get("mag_pred", 0),
        })

    mysql_df = pd.DataFrame(records).set_index("dt").sort_index()

    if df.empty:
        logger.info("MySQL backfill: loaded %d rows (no parquet base)", len(mysql_df))
        return mysql_df

    # Normalize datetime precision before concat
    df.index = df.index.astype("datetime64[ns, UTC]")
    mysql_df.index = mysql_df.index.astype("datetime64[ns, UTC]")
    combined = pd.concat([df, mysql_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.info("MySQL backfill: added %d rows (total: %d)", len(mysql_df), len(combined))
    return combined


def update_cycle() -> dict:
    """Fetch new data, predict only new bars, append to history. Returns diagnostic info."""
    from indicator.data_fetcher import (
        fetch_binance_klines, fetch_coinglass,
        fetch_binance_depth, fetch_binance_aggtrades,
        fetch_cg_options, fetch_cg_etf_flow,
        fetch_deribit_dvol, fetch_deribit_options_summary,
        fetch_cg_fear_greed, fetch_cg_etf_aum,
        fetch_cg_futures_netflow, fetch_cg_spot_netflow,
        fetch_cg_hl_whale_positions,
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

        # Normalize index to ns precision (avoid ms vs s merge errors)
        if indicator_df is not None and not indicator_df.empty:
            indicator_df.index = indicator_df.index.as_unit("ns")

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

        # Fetch sentiment/whale/netflow snapshot data
        sentiment_data = {}
        for fetch_fn, label in [
            (fetch_cg_fear_greed, "Fear/Greed"),
            (fetch_cg_etf_aum, "ETF AUM"),
            (fetch_cg_futures_netflow, "Futures netflow"),
            (fetch_cg_spot_netflow, "Spot netflow"),
            (fetch_cg_hl_whale_positions, "HL whale positions"),
        ]:
            try:
                sentiment_data.update(fetch_fn())
            except Exception as e:
                logger.warning("%s fetch failed: %s", label, e)
        if sentiment_data:
            logger.info("Sentiment data: %d fields", len(sentiment_data))

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
        features = build_live_features(klines, cg_data, depth=depth, aggtrades=aggtrades,
                                       options_data=options_data)

        # Normalize all datetime indices to same precision
        if hasattr(klines.index, 'as_unit'):
            klines.index = klines.index.as_unit("ns")
        if hasattr(features.index, 'as_unit'):
            features.index = features.index.as_unit("ns")

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

        # 4. Find bars that need prediction:
        #    - new bars after history ends
        #    - gap bars (in features range but missing from history)
        if not indicator_df.empty:
            existing_times = set(indicator_df.index)
            missing_mask = ~features.index.isin(existing_times)
            new_features = features[missing_mask]
            if len(new_features) < len(features[features.index > indicator_df.index[-1]]):
                # At minimum, predict all bars after history end
                after_end = features[features.index > indicator_df.index[-1]]
                new_features = features[missing_mask | features.index.isin(after_end.index)]
                new_features = new_features[~new_features.index.duplicated(keep="last")]
        else:
            new_features = features

        # 5. Predict new + gap bars
        if len(new_features) > 0:
            new_predictions = _engine.predict(new_features)
            if hasattr(new_predictions.index, 'as_unit'):
                new_predictions.index = new_predictions.index.as_unit("ns")
            indicator_df = pd.concat([indicator_df, new_predictions])
            indicator_df = indicator_df[~indicator_df.index.duplicated(keep="last")]
            indicator_df = indicator_df.sort_index()
            logger.info("Predicted %d bars (new + gap fill). Total: %d",
                        len(new_predictions), len(indicator_df))
        else:
            logger.info("No new bars to predict")

        # 5b. Backfill mag_pred for historical bars that don't have it
        if "mag_pred" not in indicator_df.columns:
            indicator_df["mag_pred"] = np.nan
        missing_mag = indicator_df["mag_pred"].isna() | (indicator_df["mag_pred"] == 0)
        backfill_idx = indicator_df.index[missing_mag].intersection(features.index)
        if len(backfill_idx) > 0:
            indicator_df.loc[backfill_idx, "mag_pred"] = _engine.backfill_mag_pred(
                features.loc[backfill_idx]
            )
            logger.info("Backfilled mag_pred for %d historical bars", len(backfill_idx))

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
                "dir_prob_up": float(last_row.get("dir_prob_up", 0.5)),
                "mag_pred": float(last_row.get("mag_pred", 0)),
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

        dir_prob = float(last_row.get("dir_prob_up", 0.5))
        mag = float(last_row.get("mag_pred", 0))

        arrow = "\U0001f53c" if direction == "UP" else "\U0001f53d" if direction == "DOWN" else "\u2796"
        caption = (
            f"{arrow} BTC 4h Indicator | {now_str}\n"
            f"Price: ${price:,.0f}\n"
            f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
            f"P(UP): {dir_prob:.0%} | Mag: {mag:.2%} | BBP: {bbp:+.2f}"
        )
        tg_result = _send_telegram_photo(png, caption)

        # Strong signal alert
        if strength == "Strong" and direction in ("UP", "DOWN"):
            regime = str(last_row.get("regime", "?"))
            if direction == "UP":
                icon = "\U0001f7e2\u25b2"  # 🟢▲
                label = "STRONG BULLISH"
            else:
                icon = "\U0001f534\u25bc"  # 🔴▼
                label = "STRONG BEARISH"
            alert = (
                f"{icon} <b>{label} SIGNAL</b>\n\n"
                f"BTC ${price:,.0f}\n"
                f"P(UP): {dir_prob:.0%} | Mag: {mag:.2%}\n"
                f"Confidence: {conf:.0f}\n"
                f"Regime: {regime}\n\n"
                f"⏰ {now_str}"
            )
            _send_telegram_text(alert)
            logger.info("Strong signal alert sent: %s", direction)

            # Record Strong signal for performance tracking
            try:
                from indicator.signal_tracker import record_strong_signal
                sig_time = indicator_df.index[-1]
                if hasattr(sig_time, 'to_pydatetime'):
                    sig_time = sig_time.to_pydatetime()
                record_strong_signal(
                    signal_time=sig_time, direction=direction,
                    p_up=dir_prob, mag_pred=mag, confidence=conf,
                    entry_price=price, regime=regime,
                )
            except Exception as e:
                logger.warning("Signal tracker record failed: %s", e)

        # Backfill 4h outcomes for past Strong signals
        try:
            from indicator.signal_tracker import backfill_outcomes
            backfill_outcomes()
        except Exception as e:
            logger.warning("Signal tracker backfill failed: %s", e)

        logger.info("Update complete: %s conf=%.0f %s",
                     direction, conf, strength)

        # Save depth/aggtrades/options/sentiment snapshots to MySQL + parquet
        try:
            from indicator.snapshot_collector import (
                save_depth_snapshot, save_aggtrades_snapshot,
                save_options_snapshot, save_sentiment_snapshot,
                save_indicator_history,
            )
            save_depth_snapshot(depth)
            save_aggtrades_snapshot(aggtrades)
            save_options_snapshot(options_data)
            save_sentiment_snapshot(sentiment_data)
            save_indicator_history(indicator_df)
        except Exception as snap_err:
            logger.warning("Snapshot save failed (non-critical): %s", snap_err)

        # Run signal quality monitor (lightweight, won't block)
        try:
            from indicator.monitor_icir import run_monitor
            run_monitor()
        except Exception as mon_err:
            logger.warning("Monitor check failed (non-critical): %s", mon_err)

        return {
            "engine_mode": _engine.mode if _engine else "?",
            "bars_predicted": len(new_features) if 'new_features' in dir() else 0,
            "total_bars": len(indicator_df),
            "chart_bytes": len(png),
            "direction": direction,
            "telegram_send": tg_result,
        }

    except Exception as e:
        logger.exception("Update cycle failed")
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)
        raise


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
    engine_mode = _engine.mode if _engine else "not_loaded"
    with _lock:
        return jsonify({
            "status": _state["status"],
            "engine_mode": engine_mode,
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


@app.route("/live-chart", methods=["GET"])
def live_chart():
    """Interactive HTML chart with zoom, pan, crosshair."""
    try:
        from indicator.chart_interactive import render_interactive_chart
        with _lock:
            indicator_df = _state["indicator_df"]
        if indicator_df is None or indicator_df.empty:
            return "<h3>Chart not ready</h3>", 503
        chart_df = indicator_df.dropna(subset=["open", "high", "low", "close"])
        last_n = request.args.get("n", 200, type=int)
        html = render_interactive_chart(chart_df, last_n=last_n)
        return Response(html, mimetype="text/html")
    except Exception as e:
        logger.exception("live-chart error: %s", e)
        return f"<pre>Error: {e}</pre>", 500


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
    dir_prob = pred.get("dir_prob_up", 0.5)
    mag = pred.get("mag_pred", 0)
    bbp = pred.get("bull_bear_power", 0)
    price = pred.get("close", 0)
    arrow = "\U0001f53c" if direction == "UP" else "\U0001f53d" if direction == "DOWN" else "\u2796"
    caption = (
        f"{arrow} BTC 4h Indicator\n"
        f"Price: ${price:,.0f}\n"
        f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
        f"P(UP): {dir_prob:.0%} | Mag: {mag:.2%} | BBP: {bbp:+.2f}\n"
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


@app.route("/db-diag")
def db_diagnostics():
    """Show MySQL env var names (no values) for debugging."""
    mysql_vars = {k: f"{v[:3]}***" if v else "EMPTY"
                  for k, v in os.environ.items()
                  if "MYSQL" in k.upper() or "DB" in k.upper()}
    all_env = sorted(os.environ.keys())
    try:
        from shared.db import get_db_info
        db_info = get_db_info()
    except Exception as e:
        db_info = {"error": str(e)}
    return jsonify({
        "mysql_vars_found": mysql_vars,
        "db_info": db_info,
        "all_env_names": all_env,
    })


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


@app.route("/indicator-db-stats", methods=["GET"])
def indicator_db_stats():
    """API: database accumulation stats for Telegram /db command."""
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        tables = [
            ("indicator_history", "預測歷史"),
            ("indicator_depth_snapshots", "Depth"),
            ("indicator_aggtrades_snapshots", "AggTrades"),
            ("indicator_options_snapshots", "Options/ETF/DVOL"),
            ("indicator_sentiment_snapshots", "Sentiment"),
        ]
        lines = ["<b>📦 資料庫狀態</b>\n"]
        with conn.cursor() as cur:
            for table, label in tables:
                try:
                    cur.execute(
                        f"SELECT COUNT(*) as cnt, MIN(dt) as first_dt, MAX(dt) as last_dt "
                        f"FROM `{table}`"
                    )
                    row = cur.fetchone()
                    cnt = row["cnt"]
                    first = str(row["first_dt"])[:16] if row["first_dt"] else "-"
                    last = str(row["last_dt"])[:16] if row["last_dt"] else "-"
                    lines.append(f"<b>{label}</b>: {cnt} 筆")
                    if cnt > 0:
                        lines.append(f"  {first} ~ {last}")
                except Exception:
                    lines.append(f"<b>{label}</b>: ❌ 表不存在")
        conn.close()
        return jsonify({"text": "\n".join(lines)})
    except Exception as e:
        return jsonify({"text": f"❌ 資料庫連線失敗: {e}"}), 500


@app.route("/indicator-perf", methods=["GET"])
def indicator_performance():
    """API: direction model live performance for Telegram /perf command."""
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Get predictions that are old enough to have 4h outcomes (> 4h ago)
            cur.execute("""
                SELECT dt, `close`, pred_return_4h, pred_direction_code,
                       confidence_score, dir_prob_up,
                       COALESCE(mag_pred, 0) as mag_pred,
                       COALESCE(regime_code, 0) as regime_code
                FROM indicator_history
                ORDER BY dt DESC
                LIMIT 200
            """)
            rows = cur.fetchall()
        conn.close()

        if len(rows) < 5:
            return jsonify({"text": "⏳ 數據不足（需至少 5 筆歷史紀錄）"})

        import pandas as pd
        df = pd.DataFrame(rows)
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values("dt").reset_index(drop=True)

        # Compute actual 4h returns (shift -4)
        df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
        # Only evaluate rows with known outcomes
        eval_df = df.dropna(subset=["actual_4h"]).copy()

        if len(eval_df) < 3:
            return jsonify({"text": "⏳ 尚無足夠的已實現結果（需等待 4h+）"})

        # Direction accuracy
        eval_df["pred_dir"] = eval_df["pred_direction_code"].map({1: "UP", -1: "DOWN", 0: "NEUTRAL"})
        eval_df["actual_dir"] = eval_df["actual_4h"].apply(lambda x: "UP" if x > 0.001 else ("DOWN" if x < -0.001 else "NEUTRAL"))

        # Filter non-neutral predictions
        active = eval_df[eval_df["pred_dir"] != "NEUTRAL"]
        if len(active) > 0:
            dir_acc = (active["pred_dir"] == active["actual_dir"]).mean() * 100
        else:
            dir_acc = 0

        # Direction model (dir_prob_up) accuracy
        dir_model_active = eval_df[(eval_df["dir_prob_up"] > 0.65) | (eval_df["dir_prob_up"] < 0.35)]
        if len(dir_model_active) > 0:
            dm_pred = dir_model_active["dir_prob_up"].apply(lambda p: "UP" if p > 0.5 else "DOWN")
            dm_acc = (dm_pred == dir_model_active["actual_dir"]).mean() * 100
            dm_count = len(dir_model_active)
        else:
            dm_acc = 0
            dm_count = 0

        # Spearman IC
        from scipy.stats import spearmanr
        ic, _ = spearmanr(eval_df["pred_return_4h"], eval_df["actual_4h"])

        # Strong signal performance
        strong = eval_df[eval_df["confidence_score"] >= 80]
        if len(strong) > 0:
            strong_active = strong[strong["pred_dir"] != "NEUTRAL"]
            strong_acc = (strong_active["pred_dir"] == strong_active["actual_dir"]).mean() * 100 if len(strong_active) > 0 else 0
        else:
            strong_acc = 0

        # Magnitude model IC (if available)
        mag_active = eval_df[eval_df["mag_pred"] > 0]
        if len(mag_active) >= 5:
            mag_ic, _ = spearmanr(mag_active["mag_pred"], mag_active["actual_4h"].abs())
            mag_ic_str = f"{mag_ic:.3f}"
        else:
            mag_ic_str = "N/A"

        # Regime breakdown
        regime_map = {2: "TRENDING_BULL", -2: "TRENDING_BEAR", 0: "CHOPPY", -99: "WARMUP"}
        eval_df["regime"] = eval_df["regime_code"].map(regime_map).fillna("UNKNOWN")
        current_regime = regime_map.get(int(df.iloc[-1].get("regime_code", 0)), "UNKNOWN")

        regime_lines = []
        for regime_name in ["TRENDING_BULL", "TRENDING_BEAR", "CHOPPY"]:
            r_df = eval_df[eval_df["regime"] == regime_name]
            r_active = r_df[r_df["pred_dir"] != "NEUTRAL"]
            if len(r_active) >= 2:
                r_acc = (r_active["pred_dir"] == r_active["actual_dir"]).mean() * 100
                r_label = {"TRENDING_BULL": "趨勢多", "TRENDING_BEAR": "趨勢空", "CHOPPY": "盤整"}[regime_name]
                regime_lines.append(f"  {r_label}: {r_acc:.1f}% ({len(r_active)} 筆)")

        lines = [
            "<b>📊 模型表現 (Live)</b>\n",
            f"評估期間: {len(eval_df)} 筆 ({str(eval_df['dt'].iloc[0])[:10]} ~ {str(eval_df['dt'].iloc[-1])[:10]})",
            f"\n<b>方向準確率</b>",
            f"  全部信號: {dir_acc:.1f}% ({len(active)} 筆)",
            f"  Strong 信號: {strong_acc:.1f}% ({len(strong)} 筆)",
            f"\n<b>Direction Model (P(UP))</b>",
            f"  高信心準確率: {dm_acc:.1f}% ({dm_count} 筆觸發)",
            f"\n<b>Regime 拆解</b> (當前: {current_regime})",
        ]
        if regime_lines:
            lines.extend(regime_lines)
        else:
            lines.append("  數據不足")
        lines.extend([
            f"\n<b>IC</b>",
            f"  Direction IC: {ic:.3f}",
            f"  Magnitude IC: {mag_ic_str}",
            f"\n<b>最新預測</b>",
            f"  Price: ${df.iloc[-1]['close']:,.0f}",
            f"  P(UP): {df.iloc[-1]['dir_prob_up']:.2f}",
            f"  Mag: {df.iloc[-1]['mag_pred']:.4f}" if df.iloc[-1]['mag_pred'] > 0 else "",
        ])
        lines = [l for l in lines if l]  # remove empty
        return jsonify({"text": "\n".join(lines)})
    except Exception as e:
        logger.exception("Performance calc failed")
        return jsonify({"text": f"❌ 表現計算失敗: {e}"}), 500


@app.route("/signal-perf", methods=["GET"])
def signal_perf_api():
    """API: Strong signal performance report for Telegram."""
    try:
        from indicator.signal_tracker import get_performance_report
        report = get_performance_report()
        return jsonify({"text": report})
    except Exception as e:
        logger.exception("Signal perf failed")
        return jsonify({"text": f"❌ 信號績效查詢失敗: {e}"}), 500


@app.route("/force-update", methods=["POST", "GET"])
def force_update():
    """Manually trigger an update cycle (for testing).

    ?sync=1 runs synchronously and returns result/error (timeout-sensitive).
    Default: async (returns immediately, check /health for result).
    """
    sync = request.args.get("sync", "0") == "1"
    if sync:
        try:
            result = update_cycle()
            return jsonify({"status": "ok", "error": None, "detail": result})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    else:
        threading.Thread(target=update_cycle, daemon=True).start()
        return jsonify({"status": "update_triggered"})


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
