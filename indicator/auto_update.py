"""
Auto-update indicator chart and send to Telegram.

Runs as a standalone script (Windows Task Scheduler or manual):
  1. Fetch latest Binance klines + Coinglass data
  2. Run prediction on new bars
  3. Generate chart PNG
  4. Send to Telegram chat

Usage:
    python -m indicator.auto_update
    python -m indicator.auto_update --once   (single run, no loop)
"""
from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
HISTORY_PATH = ARTIFACT_DIR / "indicator_history.parquet"
CHART_PATH = Path(__file__).parent.parent / "research" / "eda_charts" / "indicator_live.png"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CG_API_KEY = os.environ.get("COINGLASS_API_KEY", "")


def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        df = pd.read_parquet(HISTORY_PATH)
        logger.info("History: %d bars, ends %s", len(df), df.index[-1])
    else:
        df = pd.DataFrame()

    # Backfill from MySQL (rows added after last deploy)
    try:
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

        if rows:
            dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
            str_map = {3: "Strong", 2: "Moderate", 1: "Weak"}
            regime_map = {2: "TRENDING_BULL", -2: "TRENDING_BEAR", 0: "CHOPPY", -99: "WARMUP"}
            records = []
            for r in rows:
                records.append({
                    "dt": pd.Timestamp(r["dt"], tz="UTC"),
                    "open": r.get("open", 0), "high": r.get("high", 0),
                    "low": r.get("low", 0), "close": r.get("close", 0),
                    "pred_return_4h": r.get("pred_return_4h", 0),
                    "pred_direction": dir_map.get(int(r.get("pred_direction_code", 0)), "NEUTRAL"),
                    "confidence_score": r.get("confidence_score", 0),
                    "strength_score": str_map.get(int(r.get("strength_code", 1)), "Weak"),
                    "bull_bear_power": r.get("bull_bear_power", 0),
                    "regime": regime_map.get(int(r.get("regime_code", 0)), "CHOPPY"),
                    "up_pred": r.get("up_pred", 0), "down_pred": r.get("down_pred", 0),
                    "strength_raw": r.get("strength_raw", 0),
                    "dynamic_deadzone": r.get("dynamic_deadzone", 0),
                    "dir_prob_up": r.get("dir_prob_up", 0.5),
                    "mag_pred": r.get("mag_pred", 0),
                })
            mysql_df = pd.DataFrame(records).set_index("dt").sort_index()
            if df.empty:
                df = mysql_df
            else:
                df.index = df.index.astype("datetime64[ns, UTC]")
                mysql_df.index = mysql_df.index.astype("datetime64[ns, UTC]")
                df = pd.concat([df, mysql_df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            logger.info("MySQL backfill: +%d rows, total %d", len(rows), len(df))
    except Exception as e:
        logger.warning("MySQL backfill failed: %s", e)

    return df


def save_history(df: pd.DataFrame):
    df.to_parquet(HISTORY_PATH, index=True)


def check_data_freshness(klines: pd.DataFrame, cg_data: dict) -> list[str]:
    """Check if data sources are fresh. Returns list of warning messages."""
    warnings = []
    now = datetime.now(timezone.utc)
    max_age_hours = 3  # alert if data is older than 3 hours

    # Check Binance klines
    if klines.empty:
        warnings.append("Binance klines: NO DATA")
    else:
        age = (now - klines.index[-1].to_pydatetime()).total_seconds() / 3600
        if age > max_age_hours:
            warnings.append(f"Binance klines: stale ({age:.1f}h old)")

    # Check each Coinglass endpoint
    for name, df in cg_data.items():
        if df.empty:
            warnings.append(f"CG {name}: NO DATA")
        elif hasattr(df.index, 'max'):
            try:
                last_ts = df.index.max().to_pydatetime()
                age = (now - last_ts).total_seconds() / 3600
                if age > max_age_hours:
                    warnings.append(f"CG {name}: stale ({age:.1f}h old)")
            except Exception:
                pass

    return warnings


def send_telegram_alert(message: str):
    """Send plain text alert to Telegram."""
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
        logger.error("Alert send failed: %s", e)


def run_update(indicator_df: pd.DataFrame) -> tuple[pd.DataFrame, bytes]:
    """Fetch new data, predict, render chart. Returns (updated_df, png_bytes)."""
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

    # Fetch live data
    klines = fetch_binance_klines(limit=500)
    cg_data = fetch_coinglass(interval="1h", limit=500)
    depth = fetch_binance_depth()
    aggtrades = fetch_binance_aggtrades()

    # Fetch options + ETF + DVOL data (direction signals)
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
        logger.info("Options/ETF data: %d fields fetched", len(options_data))

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
        logger.info("Sentiment data: %d fields fetched", len(sentiment_data))

    # Data freshness check
    freshness_warnings = check_data_freshness(klines, cg_data)
    if freshness_warnings:
        alert = "<b>Data Freshness Alert</b>\n" + "\n".join(f"- {w}" for w in freshness_warnings)
        logger.warning("Data freshness issues: %s", freshness_warnings)
        send_telegram_alert(alert)

    features = build_live_features(klines, cg_data, depth=depth, aggtrades=aggtrades)

    # Normalize datetime index precision (avoid ms vs s merge errors)
    if not indicator_df.empty and hasattr(indicator_df.index, 'as_unit'):
        indicator_df.index = indicator_df.index.as_unit("ns")
    if hasattr(klines.index, 'as_unit'):
        klines.index = klines.index.as_unit("ns")
    if hasattr(features.index, 'as_unit'):
        features.index = features.index.as_unit("ns")

    # Backfill OHLC from klines into history
    for col in ["open", "high", "low", "close"]:
        if col in klines.columns:
            kline_col = klines[col].reindex(indicator_df.index)
            mask = kline_col.notna()
            if mask.any():
                indicator_df.loc[mask, col] = kline_col[mask]

    # Find and predict new bars
    if not indicator_df.empty:
        new_features = features[features.index > indicator_df.index[-1]]
    else:
        new_features = features

    if len(new_features) > 0:
        engine = IndicatorEngine()
        new_predictions = engine.predict(new_features)
        if hasattr(new_predictions.index, 'as_unit'):
            new_predictions.index = new_predictions.index.as_unit("ns")
        indicator_df = pd.concat([indicator_df, new_predictions])
        indicator_df = indicator_df[~indicator_df.index.duplicated(keep="last")]
        indicator_df = indicator_df.sort_index()
        logger.info("Added %d new bars. Total: %d", len(new_predictions), len(indicator_df))

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

    # Render chart
    chart_df = indicator_df.dropna(subset=["open", "high", "low", "close"])
    png = render_chart(chart_df, last_n=200)

    # Save chart to disk
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHART_PATH, "wb") as f:
        f.write(png)

    return indicator_df, png


def send_telegram(png: bytes, caption: str):
    """Send chart PNG to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not set, skipping send")
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
            logger.info("Telegram: sent OK")
        else:
            logger.error("Telegram: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("Telegram send failed: %s", e)


def run_once(indicator_df: pd.DataFrame) -> pd.DataFrame:
    """Single update cycle."""
    indicator_df, png = run_update(indicator_df)

    # Build caption
    last = indicator_df.iloc[-1]
    direction = last.get("pred_direction", "?")
    conf = last.get("confidence_score", 0)
    strength = last.get("strength_score", "?")
    dir_prob = last.get("dir_prob_up", 0.5)
    mag = last.get("mag_pred", 0)
    bbp = last.get("bull_bear_power", 0)
    price = last.get("close", 0)
    TZ_TPE = timezone(timedelta(hours=8))
    now = datetime.now(TZ_TPE).strftime("%m/%d %H:%M UTC+8")

    arrow = "🔼" if direction == "UP" else "🔽" if direction == "DOWN" else "➖"
    caption = (
        f"{arrow} BTC 4h Indicator | {now}\n"
        f"Price: ${price:,.0f}\n"
        f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
        f"P(UP): {dir_prob:.0%} | Mag: {mag:.2%} | BBP: {bbp:+.2f}"
    )

    send_telegram(png, caption)

    # Strong signal alert
    if strength == "Strong" and direction in ("UP", "DOWN"):
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
            f"Regime: {last.get('regime', '?')}\n\n"
            f"⏰ {now}"
        )
        send_telegram_alert(alert)
        logger.info("Strong signal alert sent: %s", direction)

    save_history(indicator_df)
    return indicator_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Single run, no loop")
    ap.add_argument("--interval", type=int, default=3600, help="Loop interval in seconds (default 3600=1h)")
    args = ap.parse_args()

    indicator_df = load_history()

    if args.once:
        run_once(indicator_df)
        return

    # Loop mode
    logger.info("Starting loop, interval=%ds", args.interval)
    while True:
        try:
            indicator_df = run_once(indicator_df)
        except Exception as e:
            logger.exception("Update failed: %s", e)
        logger.info("Next update in %ds...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
