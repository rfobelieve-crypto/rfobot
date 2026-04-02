"""
ICIR & Signal Quality Monitor — runs after each update cycle.
Computes rolling metrics from OOS tracking data and sends Telegram
alerts when quality degrades.

Usage (standalone):
    python -m indicator.monitor_icir

Called from app.py after each hourly update (lightweight, <1s).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACKING_FILE = Path("research/oos_tracking.csv")
ALERT_LOG = Path("research/monitor_alerts.csv")

BOT_TOKEN = os.environ.get("INDICATOR_BOT_TOKEN", "")
CHAT_ID = os.environ.get("INDICATOR_CHAT_ID", "")

TZ_TPE = timezone(timedelta(hours=8))

# ── Thresholds ────────────────────────────────────────────────────────────
MIN_BARS_FOR_EVAL = 30       # minimum OOS bars before any evaluation
ACCURACY_WARN = 0.42         # overall accuracy below this → warning
ACCURACY_CRITICAL = 0.35     # below this → critical
STRONG_ACCURACY_WARN = 0.45  # Strong tier accuracy below this → warning
FLIP_RATE_WARN = 0.40        # direction changes / total bars > this → warning
NEUTRAL_RATE_WARN = 0.70     # NEUTRAL > 70% → model may be too conservative
ALERT_COOLDOWN_H = 12        # don't repeat same alert within 12h


def load_oos_data() -> pd.DataFrame | None:
    """Load OOS tracking CSV. Returns None if insufficient data."""
    if not TRACKING_FILE.exists():
        return None
    df = pd.read_csv(TRACKING_FILE)
    if len(df) < MIN_BARS_FOR_EVAL:
        return None
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute signal quality metrics from OOS data."""
    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total if total > 0 else 0

    # By tier
    tier_metrics = {}
    for tier in ["Strong", "Moderate", "Weak"]:
        sub = df[df["strength_score"] == tier]
        if len(sub) >= 5:
            tier_metrics[tier] = {
                "count": len(sub),
                "accuracy": sub["correct"].sum() / len(sub),
            }

    # Flip rate (direction changes between consecutive signal bars)
    directions = df["pred_direction"].values
    flips = sum(1 for i in range(1, len(directions))
                if directions[i] != directions[i-1])
    flip_rate = flips / total if total > 1 else 0

    # Recent trend (last 20 bars vs overall)
    recent = df.tail(20)
    recent_acc = recent["correct"].sum() / len(recent) if len(recent) >= 10 else None

    # Directional return (positive if signal was correct)
    dir_returns = []
    for _, row in df.iterrows():
        if row["pred_direction"] == "UP":
            dir_returns.append(row["actual_4h_ret"])
        else:
            dir_returns.append(-row["actual_4h_ret"])
    avg_dir_return = np.mean(dir_returns) * 100 if dir_returns else 0

    return {
        "total_bars": total,
        "accuracy": accuracy,
        "recent_accuracy": recent_acc,
        "tier_metrics": tier_metrics,
        "flip_rate": flip_rate,
        "avg_dir_return": avg_dir_return,
    }


def check_alerts(metrics: dict) -> list[str]:
    """Check metrics against thresholds. Returns list of alert messages."""
    alerts = []

    acc = metrics["accuracy"]
    total = metrics["total_bars"]

    if acc < ACCURACY_CRITICAL:
        alerts.append(
            f"🔴 <b>CRITICAL: OOS accuracy {acc:.1%}</b> ({total} bars)\n"
            f"Below {ACCURACY_CRITICAL:.0%} threshold — model may be broken"
        )
    elif acc < ACCURACY_WARN:
        alerts.append(
            f"🟡 <b>WARNING: OOS accuracy {acc:.1%}</b> ({total} bars)\n"
            f"Below {ACCURACY_WARN:.0%} — monitor closely"
        )

    # Strong tier degradation
    strong = metrics["tier_metrics"].get("Strong", {})
    if strong and strong["accuracy"] < STRONG_ACCURACY_WARN:
        alerts.append(
            f"🟡 <b>Strong signal accuracy {strong['accuracy']:.1%}</b> "
            f"({strong['count']} signals)\n"
            f"Below {STRONG_ACCURACY_WARN:.0%} — Strong signals unreliable"
        )

    # Recent accuracy dropping
    recent = metrics["recent_accuracy"]
    if recent is not None and recent < acc - 0.10:
        alerts.append(
            f"🟡 <b>Recent accuracy dropping</b>\n"
            f"Last 20 bars: {recent:.1%} vs overall {acc:.1%} (Δ{recent-acc:+.1%})"
        )

    # Flip rate
    if metrics["flip_rate"] > FLIP_RATE_WARN:
        alerts.append(
            f"🟡 <b>High flip rate: {metrics['flip_rate']:.1%}</b>\n"
            f"Direction changing too frequently"
        )

    return alerts


def should_send_alert(alert_type: str) -> bool:
    """Check cooldown — don't spam the same alert."""
    if not ALERT_LOG.exists():
        return True
    try:
        log = pd.read_csv(ALERT_LOG)
        same_type = log[log["alert_type"] == alert_type]
        if same_type.empty:
            return True
        last_time = pd.to_datetime(same_type["timestamp"].iloc[-1])
        if (datetime.now(timezone.utc) - last_time).total_seconds() > ALERT_COOLDOWN_H * 3600:
            return True
        return False
    except Exception:
        return True


def record_alert(alert_type: str):
    """Record alert timestamp for cooldown."""
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert_type": alert_type,
    }])
    if ALERT_LOG.exists():
        existing = pd.read_csv(ALERT_LOG)
        combined = pd.concat([existing, row], ignore_index=True)
        # Keep only last 100 alerts
        combined.tail(100).to_csv(ALERT_LOG, index=False)
    else:
        row.to_csv(ALERT_LOG, index=False)


def send_telegram_alert(message: str):
    """Send alert to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials not set, logging only")
        logger.warning("ALERT: %s", message)
        return
    import requests
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=15)
    except Exception as e:
        logger.error("Alert send failed: %s", e)


def run_monitor() -> dict:
    """
    Main monitor function. Call after each update cycle.
    Returns metrics dict (for logging/testing).
    """
    df = load_oos_data()
    if df is None:
        logger.info("ICIR monitor: insufficient OOS data, skipping")
        return {}

    metrics = compute_metrics(df)
    alerts = check_alerts(metrics)

    for alert in alerts:
        # Use first line as alert type for cooldown
        alert_type = alert.split("\n")[0][:50]
        if should_send_alert(alert_type):
            now_str = datetime.now(TZ_TPE).strftime("%m/%d %H:%M UTC+8")
            full_msg = f"📊 <b>Signal Monitor</b> | {now_str}\n\n{alert}"
            send_telegram_alert(full_msg)
            record_alert(alert_type)
            logger.warning("Alert sent: %s", alert_type)

    # Log summary
    logger.info(
        "Monitor: %d OOS bars, acc=%.1f%%, strong=%s, flip_rate=%.1f%%",
        metrics["total_bars"],
        metrics["accuracy"] * 100,
        f"{metrics['tier_metrics'].get('Strong', {}).get('accuracy', 0):.1%}"
        if "Strong" in metrics["tier_metrics"] else "N/A",
        metrics["flip_rate"] * 100,
    )

    return metrics


def print_report():
    """Print full report to stdout (for manual runs)."""
    df = load_oos_data()
    if df is None:
        print(f"No OOS data yet (need {MIN_BARS_FOR_EVAL}+ bars in {TRACKING_FILE})")
        return

    metrics = compute_metrics(df)
    now_str = datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M UTC+8")

    print(f"=== SIGNAL QUALITY MONITOR | {now_str} ===")
    print(f"OOS bars: {metrics['total_bars']}")
    print(f"Overall accuracy: {metrics['accuracy']:.1%}")
    if metrics["recent_accuracy"] is not None:
        print(f"Recent (last 20): {metrics['recent_accuracy']:.1%}")
    print(f"Flip rate: {metrics['flip_rate']:.1%}")
    print(f"Avg directional return: {metrics['avg_dir_return']:+.4f}%")
    print()

    print("=== BY TIER ===")
    for tier in ["Strong", "Moderate", "Weak"]:
        t = metrics["tier_metrics"].get(tier)
        if t:
            print(f"  {tier:10s}: {t['count']:3d} signals, accuracy={t['accuracy']:.1%}")
        else:
            print(f"  {tier:10s}: insufficient data")
    print()

    alerts = check_alerts(metrics)
    if alerts:
        print("=== ALERTS ===")
        for a in alerts:
            # Strip HTML for console
            clean = a.replace("<b>", "").replace("</b>", "")
            print(f"  {clean}")
    else:
        print("=== STATUS: ALL OK ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_report()
