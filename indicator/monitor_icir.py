"""
ICIR & Signal Quality Monitor — runs after each update cycle.

Computes rolling metrics DIRECTLY from indicator_history (MySQL),
no longer depends on manually-generated oos_tracking.csv.

Sends Telegram alerts when quality degrades.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALERT_LOG = Path("research/monitor_alerts.csv")

BOT_TOKEN = os.environ.get("INDICATOR_BOT_TOKEN", "")
CHAT_ID = os.environ.get("INDICATOR_CHAT_ID", "")

TZ_TPE = timezone(timedelta(hours=8))

# ── Thresholds ────────────────────────────────────────────────────────────
MIN_BARS_FOR_EVAL = 30       # minimum bars before any evaluation
ACCURACY_WARN = 0.42         # overall accuracy below this → warning
ACCURACY_CRITICAL = 0.35     # below this → critical
STRONG_ACCURACY_WARN = 0.50  # Strong tier accuracy below this → warning
FLIP_RATE_WARN = 0.40        # direction flip rate > this → warning
NEUTRAL_RATE_WARN = 0.70     # NEUTRAL > 70% → too conservative
ALERT_COOLDOWN_H = 12        # cooldown per alert type


def _get_db_conn():
    from shared.db import get_db_conn
    return get_db_conn()


def load_live_oos(lookback_bars: int = 720) -> pd.DataFrame | None:
    """
    Load recent indicator_history from MySQL and compute actual 4h outcomes.

    Returns DataFrame with columns:
      dt, pred_direction, strength, confidence, pred_return_4h,
      actual_4h_ret, correct
    """
    try:
        conn = _get_db_conn()
    except Exception as e:
        logger.warning("DB connection failed: %s", e)
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_direction_code, strength_code,
                       confidence_score, pred_return_4h, dir_prob_up
                FROM indicator_history
                ORDER BY dt DESC
                LIMIT %s
            """, (lookback_bars,))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows or len(rows) < MIN_BARS_FOR_EVAL:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("dt").reset_index(drop=True)

    # Map codes to strings
    dir_map = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
    str_map = {3: "Strong", 2: "Moderate", 1: "Weak"}
    df["pred_direction"] = df["pred_direction_code"].map(dir_map).fillna("NEUTRAL")
    df["strength_score"] = df["strength_code"].map(str_map).fillna("Weak")

    # Compute actual 4h return (shift -4 on close)
    df["close_4h_later"] = df["close"].shift(-4)
    df["actual_4h_ret"] = df["close_4h_later"] / df["close"] - 1

    # Correctness (only for directional predictions)
    df["correct"] = np.nan
    up_mask = df["pred_direction"] == "UP"
    dn_mask = df["pred_direction"] == "DOWN"
    df.loc[up_mask, "correct"] = (df.loc[up_mask, "actual_4h_ret"] > 0).astype(float)
    df.loc[dn_mask, "correct"] = (df.loc[dn_mask, "actual_4h_ret"] < 0).astype(float)

    # Drop rows without 4h outcome (last 4 bars)
    df = df.dropna(subset=["actual_4h_ret"])

    if len(df) < MIN_BARS_FOR_EVAL:
        return None

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute signal quality metrics from live OOS data."""
    # Only directional signals
    directional = df[df["pred_direction"].isin(["UP", "DOWN"])]
    total = len(directional)

    if total == 0:
        return {"total_bars": len(df), "directional_bars": 0, "accuracy": 0,
                "tier_metrics": {}, "flip_rate": 0, "neutral_rate": 1.0}

    accuracy = directional["correct"].mean()

    # By tier
    tier_metrics = {}
    for tier in ["Strong", "Moderate", "Weak"]:
        sub = directional[directional["strength_score"] == tier]
        if len(sub) >= 5:
            tier_metrics[tier] = {
                "count": len(sub),
                "accuracy": sub["correct"].mean(),
                "avg_ret": sub["actual_4h_ret"].mean(),
            }

    # Flip rate
    directions = df["pred_direction"].values
    flips = sum(1 for i in range(1, len(directions))
                if directions[i] != directions[i - 1])
    flip_rate = flips / len(df) if len(df) > 1 else 0

    # Neutral rate
    neutral_rate = (df["pred_direction"] == "NEUTRAL").mean()

    # Recent trend (last 48 bars = 2 days)
    recent = directional.tail(48)
    recent_acc = recent["correct"].mean() if len(recent) >= 10 else None

    # Directional return (positive if correct direction)
    dir_returns = []
    for _, row in directional.iterrows():
        r = row["actual_4h_ret"]
        if row["pred_direction"] == "DOWN":
            r = -r
        dir_returns.append(r)
    avg_dir_return = np.mean(dir_returns) * 100 if dir_returns else 0

    # Spearman IC (pred_return_4h vs actual)
    ic = 0.0
    if "pred_return_4h" in df.columns and len(df) >= 30:
        from scipy.stats import spearmanr
        valid = df.dropna(subset=["pred_return_4h", "actual_4h_ret"])
        if len(valid) >= 30:
            ic, _ = spearmanr(valid["pred_return_4h"], valid["actual_4h_ret"])

    return {
        "total_bars": len(df),
        "directional_bars": total,
        "accuracy": accuracy,
        "recent_accuracy": recent_acc,
        "tier_metrics": tier_metrics,
        "flip_rate": flip_rate,
        "neutral_rate": neutral_rate,
        "avg_dir_return": avg_dir_return,
        "ic": ic,
    }


def check_alerts(metrics: dict) -> list[str]:
    """Check metrics against thresholds."""
    alerts = []
    acc = metrics["accuracy"]
    total = metrics["directional_bars"]

    if total < MIN_BARS_FOR_EVAL:
        return alerts

    if acc < ACCURACY_CRITICAL:
        alerts.append(
            f"🔴 <b>CRITICAL: Direction accuracy {acc:.1%}</b> ({total} bars)\n"
            f"Below {ACCURACY_CRITICAL:.0%} — model may be broken"
        )
    elif acc < ACCURACY_WARN:
        alerts.append(
            f"🟡 <b>WARNING: Direction accuracy {acc:.1%}</b> ({total} bars)\n"
            f"Below {ACCURACY_WARN:.0%} — monitor closely"
        )

    # Strong tier
    strong = metrics["tier_metrics"].get("Strong", {})
    if strong and strong["accuracy"] < STRONG_ACCURACY_WARN and strong["count"] >= 10:
        alerts.append(
            f"🟡 <b>Strong accuracy {strong['accuracy']:.1%}</b> "
            f"({strong['count']} signals)\n"
            f"Below {STRONG_ACCURACY_WARN:.0%} — Strong signals unreliable"
        )

    # Moderate tier
    moderate = metrics["tier_metrics"].get("Moderate", {})
    if moderate and moderate["accuracy"] < ACCURACY_WARN and moderate["count"] >= 10:
        alerts.append(
            f"🟡 <b>Moderate accuracy {moderate['accuracy']:.1%}</b> "
            f"({moderate['count']} signals)"
        )

    # Recent accuracy dropping
    recent = metrics.get("recent_accuracy")
    if recent is not None and recent < acc - 0.10:
        alerts.append(
            f"🟡 <b>Recent accuracy dropping</b>\n"
            f"Last 48h: {recent:.1%} vs overall {acc:.1%}"
        )

    # Flip rate
    if metrics["flip_rate"] > FLIP_RATE_WARN:
        alerts.append(
            f"🟡 <b>High flip rate: {metrics['flip_rate']:.1%}</b>\n"
            f"Direction changing too frequently"
        )

    # Neutral rate
    if metrics["neutral_rate"] > NEUTRAL_RATE_WARN:
        alerts.append(
            f"🟡 <b>Neutral rate: {metrics['neutral_rate']:.1%}</b>\n"
            f"Model too conservative (>{NEUTRAL_RATE_WARN:.0%} NEUTRAL)"
        )

    # IC check
    ic = metrics.get("ic", 0)
    if total >= 100 and ic < 0:
        alerts.append(
            f"🔴 <b>Negative IC: {ic:+.3f}</b>\n"
            f"Predictions inversely correlated with outcomes — consider retraining"
        )

    return alerts


def should_send_alert(alert_type: str) -> bool:
    if not ALERT_LOG.exists():
        return True
    try:
        log = pd.read_csv(ALERT_LOG)
        same = log[log["alert_type"] == alert_type]
        if same.empty:
            return True
        last = pd.to_datetime(same["timestamp"].iloc[-1])
        return (datetime.now(timezone.utc) - last).total_seconds() > ALERT_COOLDOWN_H * 3600
    except Exception:
        return True


def record_alert(alert_type: str):
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert_type": alert_type,
    }])
    if ALERT_LOG.exists():
        existing = pd.read_csv(ALERT_LOG)
        pd.concat([existing, row], ignore_index=True).tail(100).to_csv(ALERT_LOG, index=False)
    else:
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(ALERT_LOG, index=False)


def send_telegram_alert(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured, alert logged only: %s",
                        message.replace("<b>", "").replace("</b>", "")[:100])
        return
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("Alert send failed: %s", e)


def run_monitor() -> dict:
    """
    Main monitor function. Call after each update cycle.
    Reads directly from indicator_history (MySQL), no CSV dependency.
    """
    df = load_live_oos(lookback_bars=720)
    if df is None:
        logger.info("ICIR monitor: insufficient OOS data, skipping")
        return {}

    metrics = compute_metrics(df)

    # Send alerts
    alerts = check_alerts(metrics)
    for alert in alerts:
        alert_type = alert.split("\n")[0][:50]
        if should_send_alert(alert_type):
            now_str = datetime.now(TZ_TPE).strftime("%m/%d %H:%M UTC+8")
            send_telegram_alert(f"📊 <b>Signal Monitor</b> | {now_str}\n\n{alert}")
            record_alert(alert_type)

    # Log summary
    strong_str = "N/A"
    if "Strong" in metrics["tier_metrics"]:
        s = metrics["tier_metrics"]["Strong"]
        strong_str = f"{s['accuracy']:.1%} ({s['count']})"
    mod_str = "N/A"
    if "Moderate" in metrics["tier_metrics"]:
        m = metrics["tier_metrics"]["Moderate"]
        mod_str = f"{m['accuracy']:.1%} ({m['count']})"

    logger.info(
        "Monitor: %d bars, dir_acc=%.1f%%, Strong=%s, Moderate=%s, IC=%.3f, flip=%.1f%%",
        metrics["total_bars"], metrics["accuracy"] * 100,
        strong_str, mod_str,
        metrics.get("ic", 0), metrics["flip_rate"] * 100,
    )

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_live_oos()
    if df is None:
        print("No data available")
    else:
        metrics = compute_metrics(df)
        print(f"Bars: {metrics['total_bars']}, Directional: {metrics['directional_bars']}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"IC: {metrics.get('ic', 0):+.3f}")
        for tier, m in metrics["tier_metrics"].items():
            print(f"  {tier}: {m['count']} sigs, acc={m['accuracy']:.1%}")
