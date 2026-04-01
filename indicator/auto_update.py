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
from datetime import datetime, timezone

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
        return df
    return pd.DataFrame()


def save_history(df: pd.DataFrame):
    df.to_parquet(HISTORY_PATH, index=True)


def run_update(indicator_df: pd.DataFrame) -> tuple[pd.DataFrame, bytes]:
    """Fetch new data, predict, render chart. Returns (updated_df, png_bytes)."""
    from indicator.data_fetcher import fetch_binance_klines, fetch_coinglass
    from indicator.feature_builder_live import build_live_features
    from indicator.inference import IndicatorEngine
    from indicator.chart_renderer import render_chart

    # Fetch live data
    klines = fetch_binance_klines(limit=500)
    cg_data = fetch_coinglass(interval="1h", limit=500)
    features = build_live_features(klines, cg_data)

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
        indicator_df = pd.concat([indicator_df, new_predictions])
        indicator_df = indicator_df[~indicator_df.index.duplicated(keep="last")]
        indicator_df = indicator_df.sort_index()
        logger.info("Added %d new bars. Total: %d", len(new_predictions), len(indicator_df))

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
    pred_ret = last.get("pred_return_4h", 0) * 100
    bbp = last.get("bull_bear_power", 0)
    price = last.get("close", 0)
    now = datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC")

    arrow = "🔼" if direction == "UP" else "🔽" if direction == "DOWN" else "➖"
    caption = (
        f"{arrow} BTC 4h Indicator | {now}\n"
        f"Price: ${price:,.0f}\n"
        f"Direction: {direction} | Confidence: {conf:.0f} ({strength})\n"
        f"Pred: {pred_ret:+.2f}% | BBP: {bbp:+.2f}"
    )

    send_telegram(png, caption)
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
