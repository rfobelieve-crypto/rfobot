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
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


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
            {"text": "\U0001f4cb Dashboard", "url": "https://enchanting-emotion-production-4b4d.up.railway.app/dashboard"},
            {"text": "\U0001f30d Flow All", "callback_data": "flow_all"},
            {"text": "\u2699\ufe0f Status", "callback_data": "status"},
        ],
        [
            {"text": "\U0001f4c8 iChart", "callback_data": "ichart"},
            {"text": "\U0001f4c9 Decay", "callback_data": "decay"},
            {"text": "\U0001f4cb Meeting", "callback_data": "meeting"},
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


def _send_discord_photo(png: bytes, caption: str) -> str:
    """Send chart PNG to Discord via webhook. Returns status string."""
    if not DISCORD_WEBHOOK_URL:
        return "skipped: DISCORD_WEBHOOK_URL not set"
    if not png or len(png) < 100:
        return f"skipped: png_empty ({len(png) if png else 0} bytes)"
    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            data={"content": caption},
            files={"file": ("indicator.png", png, "image/png")},
            timeout=30,
        )
        if resp.ok:
            logger.info("Discord photo sent OK")
            return "ok"
        else:
            logger.error("Discord photo failed: %s %s", resp.status_code, resp.text[:200])
            return f"failed: {resp.status_code}"
    except Exception as e:
        logger.error("Discord photo send failed: %s", e)
        return f"error: {e}"


def _send_discord_text(message: str) -> str:
    """Send text message to Discord via webhook."""
    if not DISCORD_WEBHOOK_URL:
        return "skipped"
    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=15,
        )
        return "ok" if resp.ok else f"failed: {resp.status_code}"
    except Exception as e:
        logger.error("Discord text send failed: %s", e)
        return f"error: {e}"


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
    from indicator.inference import IndicatorEngine, reload_config
    from indicator.chart_renderer import render_chart

    global _engine

    try:
        reload_config()  # pick up any agent-made config changes
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

        # 2b. NaN guard — check feature quality before prediction
        try:
            from indicator.health_monitor import HealthMonitor
            safe, nan_msg = HealthMonitor.nan_guard(features, threshold=0.5)
            if not safe:
                logger.warning("NaN guard: %s — prediction may be unreliable", nan_msg)
            else:
                logger.debug("NaN guard: %s", nan_msg)
        except Exception as e:
            logger.warning("NaN guard check failed: %s", e)

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
        #    Pass full features as context so regime detection has 500 bars of history
        if len(new_features) > 0:
            new_predictions = _engine.predict(new_features, context_features=features)
            if hasattr(new_predictions.index, 'as_unit'):
                new_predictions.index = new_predictions.index.as_unit("ns")
            indicator_df = pd.concat([indicator_df, new_predictions])
            indicator_df = indicator_df[~indicator_df.index.duplicated(keep="last")]
            indicator_df = indicator_df.sort_index()
            logger.info("Predicted %d bars (new + gap fill). Total: %d",
                        len(new_predictions), len(indicator_df))
        else:
            logger.info("No new bars to predict")

        # 5b. Re-predict historical bars with the current model — full rescore
        # (direction + strength + confidence + mag + bbp). Rationale: after any
        # model swap (e.g. binary → regression) stale historical triangles
        # otherwise linger in the 200-bar window for ~8 days.
        overlap_idx = indicator_df.index.intersection(features.index)
        if len(overlap_idx) > 0:
            repredicted = _engine.predict(features.loc[overlap_idx], context_features=features,
                                             update_history=False)
            if hasattr(repredicted.index, 'as_unit'):
                repredicted.index = repredicted.index.as_unit("ns")
            pred_cols = [c for c in repredicted.columns
                         if c not in ("open", "high", "low", "close")]
            for col in pred_cols:
                indicator_df.loc[overlap_idx, col] = repredicted[col]
            logger.info("Re-scored %d historical bars with current model", len(overlap_idx))

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

        risk_text = ""

        arrow = "\U0001f53c" if direction == "UP" else "\U0001f53d" if direction == "DOWN" else "\u2796"
        caption = (
            f"{arrow} BTC 4h Indicator | {now_str}\n"
            f"Price: ${price:,.0f}\n"
            f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
            f"Mag: {mag*100:.3f}% | BBP: {bbp:+.2f}"
            f"{risk_text}"
        )
        tg_result = _send_telegram_photo(png, caption)
        dc_result = _send_discord_photo(png, caption)

        # Signal alert (Telegram + Discord push for Strong + Moderate)
        if strength in ("Strong", "Moderate") and direction in ("UP", "DOWN"):
            regime = str(last_row.get("regime", "?"))
            is_strong = (strength == "Strong")
            if direction == "UP":
                icon = "\U0001f7e2\u25b2" if is_strong else "\U0001f7e2\u25b3"  # 🟢▲ / 🟢△
                label = "STRONG BULLISH" if is_strong else "MODERATE BULLISH"
            else:
                icon = "\U0001f534\u25bc" if is_strong else "\U0001f534\u25bd"  # 🔴▼ / 🔴▽
                label = "STRONG BEARISH" if is_strong else "MODERATE BEARISH"
            # SHAP explanation for Strong signals only
            shap_text = ""
            shap_json_str = ""
            if is_strong:
                try:
                    from indicator.signal_explainer import explain_strong_signal, format_shap_for_telegram
                    import json as _json
                    last_features = features.iloc[-1].to_dict() if len(features) > 0 else {}
                    shap_result = explain_strong_signal(last_features, direction)
                    if shap_result:
                        shap_text = format_shap_for_telegram(shap_result, direction)
                        shap_json_str = _json.dumps(shap_result, ensure_ascii=False)
                except Exception as e:
                    logger.warning("SHAP explanation failed (non-critical): %s", e)

            risk_line = ""
            alert = (
                f"{icon} <b>{label} SIGNAL</b>\n\n"
                f"BTC ${price:,.0f}\n"
                f"P(UP): {dir_prob:.0%} | Mag: {mag:.2f}x\n"
                f"Confidence: {conf:.0f}{risk_line}\n"
                f"Regime: {regime}"
                f"{shap_text}\n\n"
                f"\u23f0 {now_str}"
            )
            _send_telegram_text(alert)
            # Discord uses plain text (no HTML tags)
            dc_alert = alert.replace("<b>", "**").replace("</b>", "**")
            _send_discord_text(dc_alert)
            logger.info("%s signal alert sent: %s", strength, direction)

        # Record Strong + Moderate signals for performance tracking
        if strength in ("Strong", "Moderate") and direction in ("UP", "DOWN"):
            try:
                from indicator.signal_tracker import record_signal
                sig_time = indicator_df.index[-1]
                if hasattr(sig_time, 'to_pydatetime'):
                    sig_time = sig_time.to_pydatetime()
                shap_json_str = shap_json_str if strength == "Strong" else ""
                record_signal(
                    signal_time=sig_time, direction=direction,
                    strength=strength,
                    p_up=dir_prob, mag_pred=mag, confidence=conf,
                    entry_price=price,
                    regime=str(last_row.get("regime", "")),
                    shap_json=shap_json_str,
                )
            except Exception as e:
                logger.warning("Signal tracker record failed: %s", e)

        # Backfill 4h outcomes for past signals (Strong + Moderate)
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

        # Run system health monitor
        try:
            from indicator.health_monitor import HealthMonitor
            _hm = HealthMonitor()
            with _lock:
                _health = _hm.check_all(
                    dict(_state), features_df=features,
                    cg_status=cg_status, engine=_engine,
                )
                _state["health"] = _health
            if _health["overall_status"] == "critical":
                logger.error("HEALTH CRITICAL: %s",
                             [a["detail"] for a in _health["alerts"]])
            elif _health["overall_status"] == "degraded":
                logger.warning("HEALTH DEGRADED: %s",
                               [a["detail"] for a in _health["alerts"]])
        except Exception as he:
            logger.warning("Health monitor failed: %s", he)

        # IC decay check — delegated to monitor_icir (has proper thresholds + cooldown)
        try:
            from indicator.monitor_icir import run_monitor
            run_monitor()
        except Exception as ic_err:
            logger.debug("IC monitor skipped: %s", ic_err)

        # Persist direction prediction buffer so restarts don't lose warmup
        try:
            _persist_pred_buffer(_engine)
        except Exception as buf_err:
            logger.debug("Buffer persist skipped: %s", buf_err)

        return {
            "engine_mode": _engine.mode if _engine else "?",
            "bars_predicted": len(new_features) if 'new_features' in dir() else 0,
            "total_bars": len(indicator_df),
            "chart_bytes": len(png),
            "direction": direction,
            "telegram_send": tg_result,
            "discord_send": dc_result,
        }

    except Exception as e:
        logger.exception("Update cycle failed")
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)
        raise


def _persist_pred_buffer(engine):
    """Save engine's dir_pred_history to training_stats.json for restart resilience."""
    if engine is None:
        return
    buf = getattr(engine, 'dir_pred_history', None)
    if not buf or len(buf) < 10:
        return
    import json as _json
    stats_path = Path("indicator/model_artifacts/dual_model/training_stats.json")
    if not stats_path.exists():
        return
    with open(stats_path) as f:
        stats = _json.load(f)
    stats["dir_pred_history"] = [float(x) for x in buf]
    with open(stats_path, "w") as f:
        _json.dump(stats, f, indent=2)
    logger.debug("Persisted dir_pred_history: %d values", len(buf))


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
        f"P(UP): {dir_prob:.0%} | Mag: {mag:.2f}x | BBP: {bbp:+.2f}\n"
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
    """API: comprehensive model performance for Telegram /perf command.

    Design principle: only show numbers that reflect actual OOS performance.
    - Walk-forward OOS baseline = authoritative reference (updated on retrain)
    - tracked_signals filtered to current model deploy = real live performance
    - indicator_history IC is in-sample (re-predicted every cycle) → reference only
    """
    # Walk-forward OOS baseline — updated each time model is retrained.
    # Source: research/dual_model/train_direction_reg_4h.py (77-fold expanding window)
    OOS_BASELINE = {
        "ic": 0.183,
        "auc": 0.597,
        "strong_wr": 69.2,
        "strong_wr_ci": "[63-75%]",
        "n_folds": 77,
        "train_end": "2026-04-16",
    }
    # Signals before this date came from different model versions or
    # contaminated rolling buffers — do not count toward current model perf.
    CURRENT_MODEL_DEPLOY = "2026-04-17"

    try:
        from shared.db import get_db_conn
        import pandas as pd

        conn = get_db_conn()

        # ── Section 1: Tracked signals (current model only) ──
        strong_lines = []
        strong_total = strong_filled = strong_wins = 0
        alerts = []
        try:
            with conn.cursor() as cur:
                # Current model signals
                cur.execute("""
                    SELECT direction, strength,
                           COUNT(*) as total,
                           SUM(filled) as filled_cnt,
                           SUM(correct) as wins,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM tracked_signals
                    WHERE signal_time >= %s AND strength = 'Strong'
                    GROUP BY direction, strength
                """, (CURRENT_MODEL_DEPLOY,))
                cur_rows = cur.fetchall()

                for r in cur_rows:
                    d = r["direction"]
                    total = int(r["total"] or 0)
                    filled = int(r["filled_cnt"] or 0)
                    wins = int(r["wins"] or 0)
                    strong_total += total
                    strong_filled += filled
                    strong_wins += wins
                    if filled > 0:
                        wr = wins / filled * 100
                        avg_r = float(r["avg_ret"] or 0) * 100
                        icon = "🟢" if d == "UP" else "🔴"
                        strong_lines.append(
                            f"  {icon} {d}: {wr:.0f}% ({wins}W/{filled-wins}L, avg={avg_r:+.2f}%)")

                # Pending (unfilled)
                cur.execute("""
                    SELECT COUNT(*) as cnt
                    FROM tracked_signals
                    WHERE signal_time >= %s AND strength = 'Strong' AND filled = 0
                """, (CURRENT_MODEL_DEPLOY,))
                pending = int(cur.fetchone()["cnt"] or 0)

                # All-time Strong for alert threshold (need 20+ for significance)
                cur.execute("""
                    SELECT COUNT(*) as total, SUM(correct) as wins
                    FROM tracked_signals WHERE filled = 1 AND strength = 'Strong'
                      AND signal_time >= %s
                """, (CURRENT_MODEL_DEPLOY,))
                sr = cur.fetchone()
                s_total_all = int(sr["total"] or 0)
                s_wins_all = int(sr["wins"] or 0)
                if s_total_all >= 20:
                    s_wr_all = s_wins_all / s_total_all * 100
                    if s_wr_all < 55:
                        alerts.append(f"🔴 Strong 累積勝率 {s_wr_all:.0f}% (< 55%) — 建議重訓")

        except Exception as e:
            logger.warning("Tracked signals query failed: %s", e)

        # ── Section 2: Recent signals list ──
        recent_lines = []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT signal_time, direction, strength, confidence, entry_price,
                           exit_price, actual_return_4h, correct, filled
                    FROM tracked_signals
                    WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 3 DAY)
                      AND strength = 'Strong'
                    ORDER BY signal_time DESC LIMIT 10
                """)
                recent = cur.fetchall()
                from datetime import timezone as tz, timedelta as td
                TZ8 = tz(td(hours=8))
                for r in recent:
                    sig_utc = r["signal_time"]
                    if hasattr(sig_utc, 'replace'):
                        sig_utc = sig_utc.replace(tzinfo=tz.utc)
                    sig_local = sig_utc.astimezone(TZ8)
                    t = sig_local.strftime("%m-%d %H:%M")
                    d = r["direction"]
                    icon = "🟢▲" if d == "UP" else "🔴▼"
                    entry = float(r["entry_price"])
                    conf = float(r["confidence"])
                    if r["filled"]:
                        ret = float(r["actual_return_4h"])
                        mark = "✅" if r["correct"] else "❌"
                        recent_lines.append(f"  {icon} {t} ${entry:,.0f} c={conf:.0f} → {ret*100:+.2f}% {mark}")
                    else:
                        recent_lines.append(f"  {icon} {t} ${entry:,.0f} c={conf:.0f} → ⏳")
        except Exception as e:
            logger.warning("Recent signals query failed: %s", e)

        # ── Section 3: Current regime ──
        current_regime = "UNKNOWN"
        last_price = 0
        last_dir_prob = 0.5
        last_mag = 0
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT `close`, dir_prob_up, COALESCE(mag_pred, 0) as mag_pred,
                           COALESCE(regime_code, 0) as regime_code
                    FROM indicator_history ORDER BY dt DESC LIMIT 1
                """)
                last_row = cur.fetchone()
                if last_row:
                    regime_map = {2: "TRENDING_BULL", -2: "TRENDING_BEAR", 0: "CHOPPY", -99: "WARMUP"}
                    current_regime = regime_map.get(int(last_row["regime_code"]), "UNKNOWN")
                    last_price = float(last_row["close"])
                    last_dir_prob = float(last_row["dir_prob_up"])
                    last_mag = float(last_row["mag_pred"])
        except Exception as e:
            logger.warning("Latest prediction query failed: %s", e)

        conn.close()

        # ── Build report ──
        lines = ["<b>📊 模型表現 (Dual Model)</b>\n"]

        # OOS baseline
        lines.append("<b>Walk-Forward OOS 基準</b> (權威參考)")
        lines.append(f"  IC: {OOS_BASELINE['ic']:.3f} | AUC: {OOS_BASELINE['auc']:.3f}")
        lines.append(f"  Strong 勝率: {OOS_BASELINE['strong_wr']:.1f}% {OOS_BASELINE['strong_wr_ci']}")
        lines.append(f"  ({OOS_BASELINE['n_folds']} folds, 訓練截止 {OOS_BASELINE['train_end']})")

        # Live tracked performance (current model)
        lines.append(f"\n<b>部署後實戰</b> (Strong, {CURRENT_MODEL_DEPLOY}~)")
        if strong_filled > 0:
            total_wr = strong_wins / strong_filled * 100
            lines.append(f"  總計: {total_wr:.0f}% ({strong_wins}W/{strong_filled-strong_wins}L)")
            lines.extend(strong_lines)
        elif strong_total > 0:
            lines.append(f"  {strong_total} 筆信號，尚無結算")
        else:
            lines.append("  尚無信號 (模型剛部署)")
        if pending > 0:
            lines.append(f"  等待結算: {pending} 筆")

        # Recent signals
        if recent_lines:
            lines.append(f"\n<b>最近信號</b>")
            lines.extend(recent_lines)

        # SHAP stats
        try:
            from indicator.signal_explainer import get_shap_stats_report
            shap_stats = get_shap_stats_report()
            if shap_stats:
                lines.append(shap_stats)
        except Exception as e:
            logger.warning("SHAP stats failed: %s", e)

        # Alerts
        if alerts:
            lines.append(f"\n<b>⚠️ 警報</b>")
            lines.extend(f"  {a}" for a in alerts)

        # Latest prediction
        lines.append(f"\n<b>最新預測</b> (Regime: {current_regime})")
        lines.append(f"  Price: ${last_price:,.0f}")
        lines.append(f"  P(UP): {last_dir_prob:.2f}")
        if last_mag > 0:
            lines.append(f"  Mag: {last_mag:.2%}")

        lines = [l for l in lines if l]
        return jsonify({"text": "\n".join(lines)})
    except Exception as e:
        logger.exception("Performance calc failed")
        return jsonify({"text": f"❌ 表現計算失敗: {e}"}), 500


@app.route("/alpha-decay", methods=["GET"])
def alpha_decay_api():
    """API: Alpha decay monitor — 5 early-warning signals."""
    try:
        from indicator.alpha_decay_monitor import run_full_check, format_telegram_report
        results = run_full_check()
        report = format_telegram_report(results)
        return jsonify({"text": report, "data": results})
    except Exception as e:
        logger.exception("Alpha decay check failed")
        return jsonify({"text": f"❌ Alpha decay 檢查失敗: {e}"}), 500


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


@app.route("/meeting", methods=["GET", "POST"])
def meeting_route():
    """Trigger full agent sweep: all 5 agents investigate + self-heal."""
    chat_id = request.args.get("chat_id", CHAT_ID)
    sync = request.args.get("sync", "0") == "1"

    def _run_sweep():
        try:
            from indicator.agents.watchdog import run_on_demand
            result = run_on_demand()
            return {"status": "ok", **result}
        except Exception as e:
            logger.exception("Agent sweep failed")
            _send_telegram_text(f"\u274c Agent sweep failed: {e}", chat_id=chat_id)
            return {"status": "error", "error": str(e)}

    if sync:
        return jsonify(_run_sweep())
    else:
        threading.Thread(target=_run_sweep, daemon=True).start()
        return jsonify({"status": "sweep_triggered",
                        "note": "Running all 5 agents. Results will be sent to Telegram."})


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


@app.route("/dashboard")
def dashboard_route():
    """System diagnostic dashboard — tabbed shell."""
    from indicator.dashboard import render_dashboard_shell
    return Response(render_dashboard_shell(), mimetype="text/html")


@app.route("/dashboard/tab/<tab_name>")
def dashboard_tab(tab_name):
    """Render a single dashboard tab as HTML fragment (AJAX)."""
    from indicator.dashboard import render_tab
    with _lock:
        state = dict(_state)
    html = render_tab(tab_name, state, _engine)
    return Response(html, mimetype="text/html")


def _dashboard_old():
    """OLD dashboard — replaced by dashboard.py module."""
    from datetime import timedelta
    import pandas as pd

    TZ8 = timezone(timedelta(hours=8))
    now = datetime.now(TZ8).strftime("%Y-%m-%d %H:%M UTC+8")

    # ── Collect all data ──
    sections = {}

    # 1. Engine & prediction state
    with _lock:
        pred = _state.get("last_prediction", {})
        status = _state.get("status", "unknown")
        last_update = _state.get("last_update", "N/A")
        error = _state.get("error")
        cg_status = _state.get("cg_status")

    engine_info = "N/A"
    if _engine:
        dir_n = len(getattr(_engine, 'dual_dir_features', getattr(_engine, 'dir_feature_cols', [])))
        mag_n = len(getattr(_engine, 'dual_mag_features', []))
        engine_info = f"{_engine.mode} | Dir={dir_n} feat | Mag={mag_n} feat"

    # 2. Signal tracker stats (dual model period only)
    sig_stats = {}
    try:
        from indicator.signal_tracker import _ensure_table, _get_db_conn, TABLE
        from indicator.monitor_icir import DUAL_MODEL_START as _DM_START
        _ensure_table()
        sconn = _get_db_conn()
        with sconn.cursor() as cur:
            for tier in ["Strong", "Moderate"]:
                cur.execute(f"""
                    SELECT COUNT(*) as t, SUM(filled) as f, SUM(correct) as w,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret
                    FROM `{TABLE}` WHERE strength = %s
                      AND signal_time >= %s
                """, (tier, _DM_START))
                r = cur.fetchone()
                tt, ff, ww = int(r["t"] or 0), int(r["f"] or 0), int(r["w"] or 0)
                sig_stats[tier] = {
                    "total": tt, "filled": ff, "wins": ww,
                    "wr": f"{ww/ff*100:.0f}%" if ff > 0 else "N/A",
                    "avg_ret": f"{float(r['avg_ret'] or 0)*100:+.2f}%" if ff > 0 else "N/A",
                }
            # Recent 5
            cur.execute(f"""
                SELECT signal_time, direction, strength, confidence, entry_price,
                       actual_return_4h, correct, filled
                FROM `{TABLE}` ORDER BY signal_time DESC LIMIT 5
            """)
            sig_stats["recent"] = cur.fetchall()
        sconn.close()
    except Exception as e:
        sig_stats["error"] = str(e)

    risk_info = {}

    # 4. DB health
    db_health = {}
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM indicator_history")
            db_health["indicator_history"] = cur.fetchone()["n"]
            try:
                cur.execute(f"SELECT COUNT(*) as n FROM `{TABLE}`")
                db_health["tracked_signals"] = cur.fetchone()["n"]
            except Exception:
                db_health["tracked_signals"] = "N/A"
        conn.close()
        db_health["status"] = "OK"
    except Exception as e:
        db_health["status"] = f"ERROR: {e}"

    # 5. Coinglass status — simplify dict to summary
    cg_text = "N/A"
    if cg_status and isinstance(cg_status, dict):
        total = len(cg_status)
        ok_count = sum(1 for v in cg_status.values()
                       if isinstance(v, dict) and not v.get("empty", True))
        cg_text = f"{ok_count}/{total} endpoints OK"
    elif cg_status and isinstance(cg_status, str):
        cg_text = cg_status

    # ── Build HTML ──
    def card(title, value, subtitle="", color="#4fc3f7"):
        return f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value" style="color:{color}">{value}</div>
          <div class="card-sub">{subtitle}</div>
        </div>"""

    def status_dot(ok):
        c = "#4caf50" if ok else "#f44336"
        return f'<span style="color:{c}">●</span>'

    # Risk color
    risk_color = {"HIGH": "#f44336", "MEDIUM": "#ff9800", "LOW": "#4caf50"}.get(
        risk_info.get("risk_level", ""), "#999")

    # Recent signals HTML
    recent_html = ""
    for r in sig_stats.get("recent", []):
        t = r["signal_time"]
        if hasattr(t, "replace"):
            t = t.replace(tzinfo=timezone.utc).astimezone(TZ8).strftime("%m/%d %H:%M")
        d = r["direction"]
        s = r["strength"][0]
        icon = "▲" if d == "UP" else "▼"
        dc = "#4caf50" if d == "UP" else "#f44336"
        if r["filled"]:
            ret = float(r["actual_return_4h"]) * 100
            ok = "✓" if r["correct"] else "✗"
            oc = "#4caf50" if r["correct"] else "#f44336"
            recent_html += f'<tr><td>{t}</td><td>[{s}]</td><td style="color:{dc}">{icon} {d}</td><td>{r["confidence"]:.0f}</td><td>${r["entry_price"]:,.0f}</td><td style="color:{oc}">{ret:+.2f}% {ok}</td></tr>'
        else:
            recent_html += f'<tr><td>{t}</td><td>[{s}]</td><td style="color:{dc}">{icon} {d}</td><td>{r["confidence"]:.0f}</td><td>${r["entry_price"]:,.0f}</td><td style="color:#999">⏳</td></tr>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Dashboard</title>
<meta http-equiv="refresh" content="300">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,sans-serif; padding:16px; }}
  h1 {{ color:#58a6ff; font-size:20px; margin-bottom:4px; }}
  .subtitle {{ color:#8b949e; font-size:12px; margin-bottom:16px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:20px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }}
  .card-title {{ color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }}
  .card-value {{ font-size:24px; font-weight:700; margin:4px 0; }}
  .card-sub {{ color:#8b949e; font-size:11px; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  .section-title {{ color:#58a6ff; font-size:14px; font-weight:600; margin-bottom:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:#8b949e; padding:6px 8px; border-bottom:1px solid #30363d; font-size:11px; text-transform:uppercase; }}
  td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
  .tag-ok {{ background:#1b4332; color:#4caf50; }}
  .tag-warn {{ background:#3e2723; color:#ff9800; }}
  .tag-err {{ background:#3c1111; color:#f44336; }}
</style></head><body>
<h1>Quant Diagnostic Dashboard</h1>
<div class="subtitle">{now} | Auto-refresh 5min</div>

<div class="grid">
  {card("Status", status_dot(status=="healthy") + " " + status,
        f"Last: {last_update[:19] if last_update != 'N/A' else 'N/A'}")}
  {card("Direction", pred.get("direction","?"),
        f'P(UP)={pred.get("dir_prob_up",0):.0%} | {pred.get("strength","?")}',
        "#4caf50" if pred.get("direction")=="UP" else "#f44336" if pred.get("direction")=="DOWN" else "#999")}
  {card("Confidence", f'{pred.get("confidence",0):.0f}',
        f'Price: ${pred.get("close",0):,.0f}')}
  {card("Risk", f'{risk_info.get("risk_score","?")}/100',
        risk_info.get("risk_level","?"), risk_color)}
  {card("Market Entropy", f'{risk_info.get("market_entropy","?")}'
        if isinstance(risk_info.get("market_entropy"), float)
        else "N/A",
        f'z={risk_info.get("market_entropy_zscore","?")}')}
  {card("Model", "Dual v7",
        f'Dir=99f Mag=91f')}
</div>

<div class="section">
  <div class="section-title">Signal Performance</div>
  <div class="grid" style="grid-template-columns:1fr 1fr;">
    <div>
      <table>
        <tr><th>Tier</th><th>Total</th><th>Win Rate</th><th>Avg Ret</th></tr>
        <tr><td>🔥 Strong</td><td>{sig_stats.get("Strong",{}).get("total",0)}</td>
            <td>{sig_stats.get("Strong",{}).get("wr","N/A")}</td>
            <td>{sig_stats.get("Strong",{}).get("avg_ret","N/A")}</td></tr>
        <tr><td>📈 Moderate</td><td>{sig_stats.get("Moderate",{}).get("total",0)}</td>
            <td>{sig_stats.get("Moderate",{}).get("wr","N/A")}</td>
            <td>{sig_stats.get("Moderate",{}).get("avg_ret","N/A")}</td></tr>
      </table>
    </div>
    <div>
      <table>
        <tr><th>Time</th><th>Tier</th><th>Dir</th><th>Conf</th><th>Entry</th><th>Result</th></tr>
        {recent_html}
      </table>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-title">System Health</div>
  <table>
    <tr><th>Component</th><th>Status</th><th>Detail</th></tr>
    <tr><td>MySQL</td><td>{status_dot(db_health.get("status")=="OK")} {db_health.get("status","?")}</td>
        <td>indicator_history: {db_health.get("indicator_history","?")} rows | tracked_signals: {db_health.get("tracked_signals","?")} rows</td></tr>
    <tr><td>Coinglass</td><td>{status_dot("OK" in cg_text or "14/" in cg_text)} {cg_text}</td><td></td></tr>
    <tr><td>Model Engine</td><td>{status_dot(_engine is not None)} Loaded</td><td>{engine_info}</td></tr>
    <tr><td>Error</td><td>{status_dot(error is None)} {"None" if error is None else error}</td><td></td></tr>
  </table>
</div>

</body></html>"""
    return Response(html, mimetype="text/html")


# ── Scheduler ────────────────────────────────────────────────────────────────

def _run_watchdog_quick():
    """Gatekeeper check: free if healthy, invokes Claude only on anomalies."""
    try:
        from indicator.agents.watchdog import run_quick_sweep
        run_quick_sweep()
    except Exception as e:
        logger.error("Watchdog quick sweep failed: %s", e)


def _run_watchdog_full():
    """Gatekeeper + periodic model/signal deep check."""
    try:
        from indicator.agents.watchdog import run_full_sweep
        result = run_full_sweep()
        # Task 3: if watchdog found issues, trigger repair_actions
        _maybe_trigger_repair(result)
    except Exception as e:
        logger.error("Watchdog full sweep failed: %s", e)


def _maybe_trigger_repair(watchdog_result: dict):
    """Trigger repair_actions agent if watchdog found degraded/critical domains."""
    if not watchdog_result:
        return
    agent_results = watchdog_result.get("agent_results", {})
    has_issues = any(
        r.get("status") in ("degraded", "critical")
        for r in agent_results.values()
    )
    if not has_issues:
        return
    try:
        logger.info("Watchdog found issues — triggering repair_actions agent")
        from indicator.agents.repair_actions import execute_repair_tool
        # repair_actions doesn't have a BaseAgent subclass; it's invoked by
        # other agents via their tool-use loop. The Meeting agent is the one
        # with cross-domain authority. Trigger a meeting instead.
        _run_agent_scheduled("Meeting")
    except Exception as e:
        logger.error("Repair trigger failed: %s", e)


# ── Scheduled agent runners ───────────────────────────────────────────

def _run_agent_scheduled(agent_name: str):
    """Run a single agent with DB logging. Used by APScheduler jobs."""
    try:
        agent = _get_agent_instance(agent_name)
        if agent is None:
            logger.error("Unknown agent for scheduling: %s", agent_name)
            return
        agent.run_with_logging()
    except Exception as e:
        logger.error("Scheduled agent %s failed: %s", agent_name, e)


def _get_agent_instance(agent_name: str):
    """Lazy-import and instantiate an agent by name."""
    if agent_name == "DataCollector":
        from indicator.agents.data_collector import DataCollectorAgent
        return DataCollectorAgent()
    elif agent_name == "Infra":
        from indicator.agents.infra import InfraAgent
        return InfraAgent()
    elif agent_name == "FeatureGuard":
        from indicator.agents.feature_guard import FeatureGuardAgent
        return FeatureGuardAgent()
    elif agent_name == "ModelEval":
        from indicator.agents.model_eval import ModelEvalAgent
        return ModelEvalAgent()
    elif agent_name == "SignalTracker":
        from indicator.agents.signal_tracker import SignalTrackerAgent
        return SignalTrackerAgent()
    elif agent_name == "Meeting":
        from indicator.agents.meeting import MeetingAgent
        return MeetingAgent()  # no domain_results = it'll call get_domain_reports tool
    return None


def _run_meeting_scheduled():
    """Meeting agent: run all domain agents first, then cross-correlate."""
    try:
        from indicator.agents.coordinator import AgentCoordinator
        from indicator.agents.meeting import MeetingAgent
        coordinator = AgentCoordinator()
        domain_results = coordinator.run_all()
        meeting = MeetingAgent(domain_results=domain_results)
        meeting.run_with_logging()
    except Exception as e:
        logger.error("Scheduled Meeting agent failed: %s", e)


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()

    # ── Core jobs (existing) ──
    scheduler.add_job(update_cycle, "cron", minute="2",
                      misfire_grace_time=300, max_instances=1,
                      id="update_cycle")

    # Agent watchdog: quick sweep every hour at :15, full sweep every 4h at :20
    scheduler.add_job(_run_watchdog_quick, "cron", minute="15",
                      misfire_grace_time=300, max_instances=1,
                      id="watchdog_quick")
    scheduler.add_job(_run_watchdog_full, "cron", hour="*/4", minute="20",
                      misfire_grace_time=600, max_instances=1,
                      id="watchdog_full")

    # ── Scheduled agents ──
    # data_collector: every 2h at :30
    scheduler.add_job(_run_agent_scheduled, "cron", args=["DataCollector"],
                      hour="*/2", minute="30",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_data_collector")

    # infra: every 6h at :45
    scheduler.add_job(_run_agent_scheduled, "cron", args=["Infra"],
                      hour="*/6", minute="45",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_infra")

    # feature_guard: every 6h at :15 (offset from infra)
    scheduler.add_job(_run_agent_scheduled, "cron", args=["FeatureGuard"],
                      hour="1,7,13,19", minute="15",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_feature_guard")

    # model_eval: daily 03:00 UTC
    scheduler.add_job(_run_agent_scheduled, "cron", args=["ModelEval"],
                      hour="3", minute="0",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_model_eval")

    # signal_tracker: every hour at :30
    scheduler.add_job(_run_agent_scheduled, "cron", args=["SignalTracker"],
                      minute="30",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_signal_tracker")

    # meeting: daily 09:00 UTC (runs all domain agents + cross-correlation)
    scheduler.add_job(_run_meeting_scheduled, "cron",
                      hour="9", minute="0",
                      misfire_grace_time=3600, max_instances=1,
                      id="agent_meeting")

    # ── Weekly summary ──
    from indicator.agents._summary import weekly_agent_summary
    scheduler.add_job(weekly_agent_summary, "cron",
                      day_of_week="sun", hour="2", minute="0",
                      misfire_grace_time=3600, max_instances=1,
                      id="weekly_agent_summary")

    scheduler.start()
    logger.info(
        "Scheduler started: %d jobs registered — "
        "update :02, watchdog :15/4h:20, "
        "agents (DC 2h:30, Infra 6h:45, FG 6h:15, ME 03:00, ST :30, Meeting 09:00), "
        "weekly summary Sun 02:00",
        len(scheduler.get_jobs()),
    )

    # Run first update immediately
    threading.Thread(target=update_cycle, daemon=True).start()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_scheduler()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
