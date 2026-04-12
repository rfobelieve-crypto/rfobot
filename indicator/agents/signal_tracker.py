"""
SignalTracker Agent — AI-powered signal quality analyst.

Focuses specifically on Strong + Moderate signals: are they profitable?
Are certain patterns (direction, regime, time-of-day) consistently wrong?
Is there a systematic bias Claude can detect in the win/loss patterns?
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))
DUAL_MODEL_START = "2026-03-20"


class SignalTrackerAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "SignalTracker"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior trading signal analyst reviewing the quality of an AI \
prediction system's Strong and Moderate signals. Your focus is whether \
the signals are making money and whether there are exploitable patterns \
in the wins and losses.

## Signal System
- **Strong** (confidence ≥ 80): highest conviction, should have best win rate
- **Moderate** (confidence ≥ 65): medium conviction, acceptable if profitable
- Each signal has: direction (UP/DOWN), entry_price, 4h exit_price, actual_return
- Signals are tracked in `tracked_signals` MySQL table
- Outcome filled automatically 4h after signal via `backfill_outcomes()`

## What You Analyze

### 1. Win Rate Quality
- Strong should be >55%, ideally >60%. Below 50% = coin flip = useless.
- Moderate should be >48%. Below 45% = worse than random.
- Compare UP vs DOWN: some models are better at predicting one direction.
- Win rate × average win vs loss rate × average loss = expectancy.

### 2. Pattern Detection
- **Direction bias**: does the model predict UP more than DOWN? Is one direction more accurate?
- **Regime bias**: signals in CHOPPY vs TRENDING may have very different win rates.
- **Time-of-day**: some hours (e.g., US open, Asia close) may be systematically harder.
- **Streak analysis**: is the model in a losing streak? How long? Is this within normal variance?
- **Confidence calibration**: do higher confidence signals actually win more?

### 3. Return Quality
- Average win size vs average loss size (reward/risk ratio)
- Are wins small but losses large? That's a hidden problem even with >50% win rate.
- Maximum drawdown from following the signals.

### 4. Backfill Health
- Are outcomes being filled? Pending unfilled signals >8h old = backfill broken.
- Missing outcomes corrupt all statistics.

## Judgment
- 20+ filled signals = enough to start judging. <20 = too early, note variance.
- Don't panic at 3 consecutive losses — that's normal at 55% win rate.
- DO panic at 8+ consecutive losses or win rate dropping below 40% over 30+ signals.
- Look for actionable patterns: "DOWN signals in BULL regime are 0/5" → suggest disabling.

Respond in Traditional Chinese for summary/report."""

    def collect_context(self) -> dict:
        """Load all signal data for Claude to analyze."""
        context = {}

        try:
            from shared.db import get_db_conn
            conn = get_db_conn()

            context["signals_by_tier"] = self._load_tier_stats(conn)
            context["signals_by_direction"] = self._load_direction_stats(conn)
            context["signals_by_regime"] = self._load_regime_stats(conn)
            context["recent_signals"] = self._load_recent(conn)
            context["streak_analysis"] = self._load_streaks(conn)
            context["backfill_health"] = self._load_backfill_status(conn)
            context["hourly_performance"] = self._load_hourly(conn)

            conn.close()
        except Exception as e:
            context["error"] = str(e)

        return context

    def get_available_actions(self) -> list[dict]:
        return [
            {"name": "ALERT", "description": "Send Telegram alert about signal quality"},
            {"name": "LOG", "description": "Log finding"},
            {"name": "NONE", "description": "No action"},
        ]

    def _load_tier_stats(self, conn) -> dict:
        result = {}
        with conn.cursor() as cur:
            for tier in ["Strong", "Moderate"]:
                cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(filled) as filled,
                           SUM(correct) as wins,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret,
                           AVG(CASE WHEN filled=1 AND correct=1 THEN actual_return_4h END) as avg_win_ret,
                           AVG(CASE WHEN filled=1 AND correct=0 THEN actual_return_4h END) as avg_loss_ret,
                           MAX(CASE WHEN filled=1 THEN actual_return_4h END) as best,
                           MIN(CASE WHEN filled=1 THEN actual_return_4h END) as worst
                    FROM tracked_signals
                    WHERE strength = %s AND signal_time >= %s
                """, (tier, DUAL_MODEL_START))
                r = cur.fetchone()
                total = int(r["total"] or 0)
                filled = int(r["filled"] or 0)
                wins = int(r["wins"] or 0)

                tier_data = {
                    "total": total,
                    "filled": filled,
                    "pending": total - filled,
                    "wins": wins,
                    "losses": filled - wins,
                    "win_rate_pct": round(wins / filled * 100, 1) if filled > 0 else None,
                    "avg_return_pct": round(float(r["avg_ret"] or 0) * 100, 3),
                    "avg_win_pct": round(float(r["avg_win_ret"] or 0) * 100, 3) if r["avg_win_ret"] else None,
                    "avg_loss_pct": round(float(r["avg_loss_ret"] or 0) * 100, 3) if r["avg_loss_ret"] else None,
                    "best_pct": round(float(r["best"] or 0) * 100, 3),
                    "worst_pct": round(float(r["worst"] or 0) * 100, 3),
                }

                # Expectancy
                if filled > 0 and tier_data["avg_win_pct"] and tier_data["avg_loss_pct"]:
                    wr = wins / filled
                    tier_data["expectancy_pct"] = round(
                        wr * tier_data["avg_win_pct"] + (1 - wr) * tier_data["avg_loss_pct"], 3
                    )

                result[tier] = tier_data
        return result

    def _load_direction_stats(self, conn) -> dict:
        result = {}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT direction, strength,
                       COUNT(*) as cnt,
                       SUM(correct) as wins,
                       AVG(actual_return_4h) as avg_ret
                FROM tracked_signals
                WHERE filled = 1 AND signal_time >= %s
                GROUP BY direction, strength
            """, (DUAL_MODEL_START,))
            for r in cur.fetchall():
                key = f"{r['direction']}_{r['strength']}"
                cnt = int(r["cnt"])
                wins = int(r["wins"] or 0)
                result[key] = {
                    "count": cnt,
                    "wins": wins,
                    "win_rate_pct": round(wins / cnt * 100, 1) if cnt > 0 else None,
                    "avg_return_pct": round(float(r["avg_ret"] or 0) * 100, 3),
                }
        return result

    def _load_regime_stats(self, conn) -> dict:
        result = {}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT regime, direction,
                       COUNT(*) as cnt,
                       SUM(correct) as wins,
                       AVG(actual_return_4h) as avg_ret
                FROM tracked_signals
                WHERE filled = 1 AND signal_time >= %s AND regime != ''
                GROUP BY regime, direction
            """, (DUAL_MODEL_START,))
            for r in cur.fetchall():
                key = f"{r['regime']}_{r['direction']}"
                cnt = int(r["cnt"])
                wins = int(r["wins"] or 0)
                result[key] = {
                    "count": cnt,
                    "wins": wins,
                    "win_rate_pct": round(wins / cnt * 100, 1) if cnt > 0 else None,
                    "avg_return_pct": round(float(r["avg_ret"] or 0) * 100, 3),
                }
        return result

    def _load_recent(self, conn) -> list:
        """Last 30 filled signals with full detail."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, strength, confidence,
                       entry_price, exit_price, actual_return_4h, correct, regime
                FROM tracked_signals
                WHERE filled = 1 AND signal_time >= %s
                ORDER BY signal_time DESC
                LIMIT 30
            """, (DUAL_MODEL_START,))
            return [
                {
                    "time": str(r["signal_time"]),
                    "dir": r["direction"],
                    "tier": r["strength"],
                    "conf": round(float(r["confidence"]), 1) if r["confidence"] else None,
                    "entry": float(r["entry_price"]) if r["entry_price"] else None,
                    "exit": float(r["exit_price"]) if r["exit_price"] else None,
                    "ret_pct": round(float(r["actual_return_4h"]) * 100, 3) if r["actual_return_4h"] else None,
                    "correct": bool(r["correct"]) if r["correct"] is not None else None,
                    "regime": r["regime"] or "unknown",
                }
                for r in cur.fetchall()
            ]

    def _load_streaks(self, conn) -> dict:
        """Analyze win/loss streaks."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT correct FROM tracked_signals
                WHERE filled = 1 AND signal_time >= %s
                ORDER BY signal_time ASC
            """, (DUAL_MODEL_START,))
            results = [bool(r["correct"]) for r in cur.fetchall()]

        if not results:
            return {"total_signals": 0}

        # Current streak
        current_streak = 1
        is_winning = results[-1]
        for i in range(len(results) - 2, -1, -1):
            if results[i] == is_winning:
                current_streak += 1
            else:
                break

        # Longest streaks
        max_win, max_loss = 0, 0
        cur_win, cur_loss = 0, 0
        for r in results:
            if r:
                cur_win += 1
                cur_loss = 0
                max_win = max(max_win, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss = max(max_loss, cur_loss)

        return {
            "total_signals": len(results),
            "current_streak": current_streak,
            "current_streak_type": "WIN" if is_winning else "LOSS",
            "longest_win_streak": max_win,
            "longest_loss_streak": max_loss,
            "last_10": ["W" if r else "L" for r in results[-10:]],
        }

    def _load_backfill_status(self, conn) -> dict:
        """Check if outcome backfill is working."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as pending,
                       MIN(signal_time) as oldest_pending,
                       MAX(signal_time) as newest_pending
                FROM tracked_signals
                WHERE filled = 0 AND signal_time >= %s
            """, (DUAL_MODEL_START,))
            r = cur.fetchone()

            pending = int(r["pending"] or 0)
            oldest = r["oldest_pending"]
            age_hours = None
            if oldest:
                oldest_utc = oldest.replace(tzinfo=timezone.utc) if hasattr(oldest, 'replace') else oldest
                age_hours = round((datetime.now(timezone.utc) - oldest_utc).total_seconds() / 3600, 1)

            return {
                "pending_count": pending,
                "oldest_pending": str(oldest) if oldest else None,
                "oldest_age_hours": age_hours,
                "newest_pending": str(r["newest_pending"]) if r["newest_pending"] else None,
                "healthy": pending == 0 or (age_hours is not None and age_hours < 8),
            }

    def _load_hourly(self, conn) -> dict:
        """Performance by hour-of-day (UTC)."""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT HOUR(signal_time) as hour,
                       COUNT(*) as cnt,
                       SUM(correct) as wins
                FROM tracked_signals
                WHERE filled = 1 AND signal_time >= %s
                GROUP BY HOUR(signal_time)
                ORDER BY hour
            """, (DUAL_MODEL_START,))
            result = {}
            for r in cur.fetchall():
                h = int(r["hour"])
                cnt = int(r["cnt"])
                wins = int(r["wins"] or 0)
                if cnt >= 3:
                    result[f"{h:02d}:00"] = {
                        "count": cnt,
                        "win_rate_pct": round(wins / cnt * 100, 1),
                    }
        return result


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = SignalTrackerAgent()
    if "--context-only" in sys.argv:
        ctx = agent.collect_context()
        print(_json.dumps(ctx, indent=2, default=str, ensure_ascii=False))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"\nDiagnosis:\n{_json.dumps(result.diagnosis, indent=2, ensure_ascii=False)}")
