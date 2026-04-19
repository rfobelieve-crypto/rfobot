"""
Alpha Decay Monitor — 5 early-warning signals for model degradation.

Signals:
  1. Rolling IC trend (7d / 30d) — from indicator_history
  2. Feature importance drift — from importance CSV snapshots
  3. Signal churn rate — direction flip frequency over time
  4. Confidence-WR decoupling — high confidence but low win rate
  5. Strong signal yield decay — Strong signal frequency declining

Each signal returns a status (healthy / warning / critical) + detail.
Combined into monthly report or on-demand via /decay command.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

CURRENT_MODEL_DEPLOY = "2026-04-17"


def _get_db_conn():
    from shared.db import get_db_conn
    return get_db_conn()


# ── Signal 1: Rolling IC Trend ──────────────────────────────────────────

def check_ic_trend() -> dict:
    """Compare 7d IC vs 30d IC. Declining trend = early decay."""
    try:
        conn = _get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dt, close, pred_return_4h
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    ORDER BY dt ASC
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 50:
            return {"status": "insufficient_data", "detail": f"只有 {len(rows)} 筆，需要 50+"}

        df = pd.DataFrame(rows)
        df["close_4h"] = df["close"].shift(-4)
        df["actual_ret"] = df["close_4h"] / df["close"] - 1
        df = df.dropna(subset=["actual_ret", "pred_return_4h"])

        from scipy.stats import spearmanr

        # 30d IC
        ic_30d, _ = spearmanr(df["pred_return_4h"], df["actual_ret"])

        # 7d IC
        df["dt_parsed"] = pd.to_datetime(df["dt"])
        cutoff_7d = pd.Timestamp.now("UTC").tz_localize(None) - timedelta(days=7)
        recent = df[df["dt_parsed"] >= cutoff_7d]
        ic_7d = None
        if len(recent) >= 20:
            ic_7d, _ = spearmanr(recent["pred_return_4h"], recent["actual_ret"])

        # Status
        if ic_7d is not None and ic_7d < -0.05:
            status = "critical"
        elif ic_7d is not None and ic_7d < 0.02:
            status = "warning"
        elif ic_30d < 0:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "ic_30d": round(ic_30d, 4) if not np.isnan(ic_30d) else None,
            "ic_7d": round(ic_7d, 4) if ic_7d is not None and not np.isnan(ic_7d) else None,
            "n_bars_30d": len(df),
            "n_bars_7d": len(recent) if ic_7d is not None else 0,
            "detail": f"IC 30d={ic_30d:+.3f}, 7d={ic_7d:+.3f}" if ic_7d is not None
                      else f"IC 30d={ic_30d:+.3f}, 7d=N/A (不足 20 筆)",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Signal 2: Feature Importance Drift ──────────────────────────────────

def check_importance_drift() -> dict:
    """Check if top-10 features shifted since last snapshot."""
    try:
        from pathlib import Path

        history_dir = Path("indicator/model_artifacts/dual_model/history")
        current_csv = Path("indicator/model_artifacts/dual_model/direction_importance.csv")

        if not current_csv.exists():
            return {"status": "error", "detail": "direction_importance.csv 不存在"}

        current = pd.read_csv(current_csv)
        current.columns = ["feature", "importance"] + list(current.columns[2:])
        top10_current = set(current.nlargest(10, "importance")["feature"].tolist())

        # Find latest snapshot
        if not history_dir.exists():
            return {"status": "no_history", "detail": "無歷史快照，需先跑 snapshot",
                    "top10": list(top10_current)}

        snapshots = sorted(history_dir.glob("direction_importance_*.csv"))
        if not snapshots:
            return {"status": "no_history", "detail": "無歷史快照",
                    "top10": list(top10_current)}

        prev = pd.read_csv(snapshots[-1])
        prev.columns = ["feature", "importance"] + list(prev.columns[2:])
        top10_prev = set(prev.nlargest(10, "importance")["feature"].tolist())

        overlap = len(top10_current & top10_prev)
        added = top10_current - top10_prev
        removed = top10_prev - top10_current

        if overlap < 5:
            status = "critical"
        elif overlap < 7:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "top10_overlap": overlap,
            "snapshot_file": snapshots[-1].name,
            "added_to_top10": list(added),
            "dropped_from_top10": list(removed),
            "detail": f"Top-10 overlap: {overlap}/10"
                      + (f" (新增: {', '.join(added)})" if added else ""),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Signal 3: Signal Churn Rate ─────────────────────────────────────────

def check_churn_rate() -> dict:
    """Measure direction flip rate over different windows.

    High churn = model is guessing, not predicting.
    Increasing churn over time = alpha decaying.
    """
    try:
        conn = _get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dt, pred_direction_code
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                    ORDER BY dt ASC
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 48:
            return {"status": "insufficient_data", "detail": f"只有 {len(rows)} 筆"}

        dirs = [int(r["pred_direction_code"] or 0) for r in rows]
        dts = [r["dt"] for r in rows]

        def flip_rate(seq):
            if len(seq) < 2:
                return 0
            flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
            return flips / (len(seq) - 1)

        # Overall 14d
        fr_14d = flip_rate(dirs)

        # First 7d vs last 7d
        mid = len(dirs) // 2
        fr_first_half = flip_rate(dirs[:mid])
        fr_second_half = flip_rate(dirs[mid:])

        # Last 48h
        fr_48h = flip_rate(dirs[-48:])

        # Trend: increasing churn is a warning
        churn_increasing = fr_second_half > fr_first_half + 0.05

        if fr_48h > 0.50:
            status = "critical"
        elif fr_48h > 0.40 or churn_increasing:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "flip_rate_14d": round(fr_14d, 3),
            "flip_rate_first_7d": round(fr_first_half, 3),
            "flip_rate_last_7d": round(fr_second_half, 3),
            "flip_rate_48h": round(fr_48h, 3),
            "churn_increasing": churn_increasing,
            "detail": f"Churn: 48h={fr_48h:.1%}, 趨勢={'↑ 惡化' if churn_increasing else '穩定'}",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Signal 4: Confidence-WR Decoupling ──────────────────────────────────

def check_confidence_wr_decoupling() -> dict:
    """High confidence + low win rate = model overconfident = alpha decayed.

    Compare: top-30% confidence signals' WR vs bottom-30% confidence signals' WR.
    If they're the same, confidence score has lost discriminative power.
    """
    try:
        conn = _get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT confidence, correct, actual_return_4h, direction
                    FROM tracked_signals
                    WHERE filled = 1 AND strength = 'Strong'
                      AND signal_time >= %s
                    ORDER BY signal_time ASC
                """, (CURRENT_MODEL_DEPLOY,))
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 20:
            return {"status": "insufficient_data",
                    "detail": f"只有 {len(rows)} 筆 filled Strong signals，需要 20+"}

        df = pd.DataFrame(rows)
        df["confidence"] = df["confidence"].astype(float)
        df["correct"] = df["correct"].astype(int)

        # Split into confidence terciles
        q_hi = df["confidence"].quantile(0.70)
        q_lo = df["confidence"].quantile(0.30)

        hi_conf = df[df["confidence"] >= q_hi]
        lo_conf = df[df["confidence"] <= q_lo]

        wr_hi = hi_conf["correct"].mean() if len(hi_conf) >= 5 else None
        wr_lo = lo_conf["correct"].mean() if len(lo_conf) >= 5 else None
        wr_all = df["correct"].mean()

        # Decoupling: high-conf WR should be > low-conf WR
        # If not, confidence is meaningless
        if wr_hi is not None and wr_lo is not None:
            gap = wr_hi - wr_lo
            if gap < -0.05:
                status = "critical"  # high-conf is WORSE than low-conf
            elif gap < 0.05:
                status = "warning"   # no difference = confidence is noise
            else:
                status = "healthy"
        else:
            gap = None
            status = "insufficient_data"

        return {
            "status": status,
            "n_signals": len(df),
            "wr_overall": round(wr_all, 3),
            "wr_high_conf": round(wr_hi, 3) if wr_hi is not None else None,
            "wr_low_conf": round(wr_lo, 3) if wr_lo is not None else None,
            "conf_threshold_hi": round(q_hi, 1) if q_hi is not None else None,
            "conf_threshold_lo": round(q_lo, 1) if q_lo is not None else None,
            "wr_gap": round(gap, 3) if gap is not None else None,
            "detail": (
                f"高信心 WR={wr_hi:.0%} vs 低信心 WR={wr_lo:.0%} (差距={gap:+.1%})"
                if wr_hi is not None and wr_lo is not None
                else f"整體 WR={wr_all:.0%}，分群樣本不足"
            ),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Signal 5: Strong Signal Yield ───────────────────────────────────────

def check_signal_yield() -> dict:
    """Track Strong signal frequency over time.

    Declining yield = model becoming more conservative or market regime shifted.
    This isn't necessarily bad, but sustained decline suggests rolling percentile
    thresholds are tightening as volatility drops.
    """
    try:
        conn = _get_db_conn()
        try:
            with conn.cursor() as cur:
                # Prediction counts by week
                cur.execute("""
                    SELECT
                        YEARWEEK(dt, 1) AS yw,
                        COUNT(*) AS total_bars,
                        SUM(strength_code = 3) AS strong_bars,
                        SUM(pred_direction_code != 0) AS directional_bars
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    GROUP BY YEARWEEK(dt, 1)
                    ORDER BY yw ASC
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if len(rows) < 2:
            return {"status": "insufficient_data", "detail": "不足 2 週數據"}

        weeks = []
        for r in rows:
            total = int(r["total_bars"] or 0)
            strong = int(r["strong_bars"] or 0)
            directional = int(r["directional_bars"] or 0)
            strong_pct = strong / total * 100 if total > 0 else 0
            dir_pct = directional / total * 100 if total > 0 else 0
            weeks.append({
                "week": str(r["yw"]),
                "total": total,
                "strong": strong,
                "strong_pct": round(strong_pct, 1),
                "directional_pct": round(dir_pct, 1),
            })

        # Trend: compare last week vs average of previous weeks
        if len(weeks) >= 2:
            prev_avg = np.mean([w["strong_pct"] for w in weeks[:-1]])
            last = weeks[-1]["strong_pct"]
            yield_change = last - prev_avg
        else:
            prev_avg = weeks[0]["strong_pct"]
            last = prev_avg
            yield_change = 0

        if last < 1.0:  # < 1% Strong signals
            status = "warning"
        elif yield_change < -5:  # dropped 5pp+
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "weeks": weeks,
            "latest_strong_pct": last,
            "prev_avg_strong_pct": round(prev_avg, 1),
            "yield_change_pp": round(yield_change, 1),
            "detail": f"Strong 佔比: 最近={last:.1f}%, 之前均值={prev_avg:.1f}% ({yield_change:+.1f}pp)",
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Combined Report ─────────────────────────────────────────────────────

STATUS_ICON = {
    "healthy": "🟢",
    "warning": "🟡",
    "critical": "🔴",
    "error": "⚫",
    "insufficient_data": "⏳",
    "no_history": "⏳",
}


def run_full_check() -> dict:
    """Run all 5 decay signals. Returns structured results."""
    results = {
        "ic_trend": check_ic_trend(),
        "importance_drift": check_importance_drift(),
        "churn_rate": check_churn_rate(),
        "confidence_wr": check_confidence_wr_decoupling(),
        "signal_yield": check_signal_yield(),
    }

    # Overall status: worst of all signals
    statuses = [r["status"] for r in results.values()]
    if "critical" in statuses:
        results["overall"] = "critical"
    elif "warning" in statuses:
        results["overall"] = "warning"
    elif all(s in ("healthy", "insufficient_data", "no_history") for s in statuses):
        results["overall"] = "healthy"
    else:
        results["overall"] = "unknown"

    results["timestamp"] = datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M UTC+8")
    return results


def format_telegram_report(results: dict) -> str:
    """Format decay check results as Telegram HTML."""
    overall = results.get("overall", "unknown")
    overall_icon = STATUS_ICON.get(overall, "❓")
    ts = results.get("timestamp", "")

    lines = [
        f"{overall_icon} <b>Alpha Decay Monitor</b> | {ts}\n",
    ]

    signal_names = {
        "ic_trend": "IC 趨勢",
        "importance_drift": "特徵重要性漂移",
        "churn_rate": "信號翻轉率",
        "confidence_wr": "信心-勝率脫鉤",
        "signal_yield": "Strong 信號產量",
    }

    for key, label in signal_names.items():
        r = results.get(key, {})
        icon = STATUS_ICON.get(r.get("status", "error"), "❓")
        detail = r.get("detail", "N/A")
        lines.append(f"{icon} <b>{label}</b>: {detail}")

    # Add actionable advice for warnings/criticals
    warnings = []
    if results.get("ic_trend", {}).get("status") == "critical":
        warnings.append("→ IC 深度負值，考慮重訓模型")
    if results.get("churn_rate", {}).get("churn_increasing"):
        warnings.append("→ 翻轉率上升，模型可能在猜測")
    if results.get("confidence_wr", {}).get("status") in ("warning", "critical"):
        warnings.append("→ 信心分數失去鑑別力，confidence 公式需校準")
    if results.get("importance_drift", {}).get("status") == "critical":
        warnings.append("→ Top 特徵大幅變動，數據源可能異常")

    if warnings:
        lines.append("\n<b>建議動作</b>:")
        lines.extend(warnings)

    return "\n".join(lines)
