"""
Weekly Agent Run Summary — queries agent_runs table and sends report.

Scheduled: every Sunday 02:00 UTC via APScheduler.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))


def weekly_agent_summary():
    """Query past 7 days of agent_runs, build report, send to Telegram."""
    try:
        report = _build_report()
        _send_report(report)
    except Exception as e:
        logger.error("Weekly agent summary failed: %s", e)


def _build_report() -> str:
    from shared.db import get_db_conn

    now = datetime.now(TZ_TPE)
    week_ago = now - timedelta(days=7)
    date_range = f"{week_ago.strftime('%m/%d')} ~ {now.strftime('%m/%d')}"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Per-agent stats
            cur.execute("""
                SELECT agent_name,
                       COUNT(*) AS total_runs,
                       SUM(status = 'success') AS successes,
                       SUM(status = 'failed') AS failures,
                       AVG(TIMESTAMPDIFF(SECOND, started_at, finished_at)) AS avg_dur_s,
                       MAX(CASE WHEN status = 'failed' THEN started_at END) AS last_fail,
                       MAX(CASE WHEN status = 'failed' THEN error_message END) AS last_err
                FROM agent_runs
                WHERE started_at >= DATE_SUB(UTC_TIMESTAMP(), INTERVAL 7 DAY)
                GROUP BY agent_name
                ORDER BY agent_name
            """)
            rows = cur.fetchall()

            if not rows:
                return (
                    f"📊 <b>Weekly Agent Runs Summary</b>\n"
                    f"({date_range})\n\n"
                    f"本週無 agent 執行紀錄。"
                )

            lines = [
                f"📊 <b>Weekly Agent Runs Summary</b>",
                f"({date_range})\n",
            ]

            warnings = []
            for r in rows:
                name = r["agent_name"]
                total = int(r["total_runs"] or 0)
                ok = int(r["successes"] or 0)
                fail = int(r["failures"] or 0)
                rate = ok / total * 100 if total > 0 else 0
                avg_s = float(r["avg_dur_s"] or 0)

                icon = "✅" if fail == 0 else "⚠️"
                dur_str = f"{avg_s:.0f}s" if avg_s < 60 else f"{avg_s/60:.1f}m"
                lines.append(
                    f"{icon} <b>{name}</b>: {total} runs, "
                    f"{rate:.0f}% OK, avg {dur_str}"
                )

                if fail > 0:
                    last_fail = r["last_fail"]
                    last_err = (r["last_err"] or "?")[:80]
                    fail_time = ""
                    if last_fail:
                        if hasattr(last_fail, "strftime"):
                            fail_time = last_fail.strftime("%m-%d %H:%M")
                        else:
                            fail_time = str(last_fail)[:16]
                    warnings.append(
                        f"  → {name}: {fail}x 失敗, 最近 {fail_time}\n"
                        f"    <code>{last_err}</code>"
                    )

            if warnings:
                lines.append("\n🚨 <b>需要注意</b>:")
                lines.extend(warnings)
            else:
                lines.append("\n✅ 所有 agent 運行正常")

            # Alpha Decay section
            try:
                from indicator.alpha_decay_monitor import run_full_check, STATUS_ICON
                decay = run_full_check()
                lines.append("\n📉 <b>Alpha Decay Monitor</b>")
                signal_names = {
                    "ic_trend": "IC 趨勢",
                    "importance_drift": "特徵漂移",
                    "churn_rate": "信號翻轉",
                    "confidence_wr": "信心-勝率",
                    "signal_yield": "Strong 產量",
                }
                for key, label in signal_names.items():
                    r = decay.get(key, {})
                    icon = STATUS_ICON.get(r.get("status", "error"), "❓")
                    detail = r.get("detail", "N/A")
                    lines.append(f"  {icon} {label}: {detail}")
            except Exception as e:
                lines.append(f"\n📉 Alpha Decay: 檢查失敗 ({e})")

            return "\n".join(lines)
    finally:
        conn.close()


def _send_report(text: str):
    """Send to AGENT_CHAT_ID (not INDICATOR_CHAT_ID)."""
    import requests

    bot_token = (
        os.environ.get("AGENT_BOT_TOKEN")
        or os.environ.get("INDICATOR_BOT_TOKEN", "")
    )
    chat_id = (
        os.environ.get("AGENT_CHAT_ID")
        or os.environ.get("INDICATOR_CHAT_ID", "")
    )
    if not bot_token or not chat_id:
        logger.warning("Weekly summary: Telegram not configured")
        return

    # Split if > 4096
    if len(text) <= 4096:
        _tg_send(bot_token, chat_id, text)
    else:
        # Split by double newline
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > 4000:
                chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            chunks.append(current)
        for chunk in chunks:
            _tg_send(bot_token, chat_id, chunk)


def _tg_send(bot_token: str, chat_id: str, text: str):
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("Weekly summary TG send failed: %s", e)
