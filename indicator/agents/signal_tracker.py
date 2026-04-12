"""
SignalTracker Agent — Tool-Use Architecture.

Claude investigates signal quality: win rates, streaks, biases,
and hourly/regime patterns. It decides how deep to drill based
on what it finds.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

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
You are a senior trading signal analyst. Your job is to determine whether \
the prediction system's Strong and Moderate signals are actually making money.

## Investigation Protocol
1. Check overall win rates by tier (the most important metric)
2. Look at recent signals (last 72h) — are we in a losing streak?
3. Check direction bias (UP vs DOWN performance)
4. Check regime-dependent performance
5. Check if backfill is working (unfilled signals = broken tracking)
6. Optionally: check hourly performance for time-of-day patterns
7. Conclude

## Thresholds (use judgment)
- Strong win rate: >55% = good, 50-55% = acceptable, <50% = broken
- Moderate: >48% = acceptable, <45% = worse than random
- 3 consecutive losses at 55% win rate = normal (happens 9% of the time)
- 8+ consecutive losses = extremely unlikely, investigate
- Expectancy = win_rate × avg_win + (1-win_rate) × avg_loss — must be positive
- Pending signals >8h old = backfill is broken

## Actionable Patterns
- "DOWN signals in BULL regime: 0/5" → suggest disabling contra-trend signals
- "Strong signals at hour 04:00 UTC: 1/8" → possible Asian session weakness
- "UP signals 70% of all signals" → model has directional bias

## Actions
AUTO (safe):
- Backfill broken (pending signals >8h) → `repair_trigger_update`

SUGGEST ONLY (★★★★★ — never auto-modify signal parameters):
- Strong win rate < 45% over 50+ signals → `suggest_pause_signals` with evidence
- Strong win rate 45-50% over 100+ → `suggest_threshold_change` STRONG_THRESHOLD
- Contra-trend losing (DOWN in BULL 0/5) → `suggest_threshold_change` BULL_CONTRA_PENALTY
- Always include: sample size, time range, regime context in evidence.

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "get_tier_stats",
                "description": "Get win rate, avg return, best/worst, expectancy for each tier (Strong, Moderate).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_recent_signals",
                "description": "Get recent filled signals with full detail (time, direction, tier, return, correct). Use count param to control how many.",
                "input_schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer", "description": "Number of signals (default 20)"}},
                    "required": [],
                },
            },
            {
                "name": "get_direction_breakdown",
                "description": "Win rate broken down by direction (UP/DOWN) × tier.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_regime_breakdown",
                "description": "Win rate broken down by regime × direction. Reveals regime-dependent biases.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_streak_analysis",
                "description": "Current win/loss streak, longest streaks, and last 10 results.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_backfill_status",
                "description": "Check if outcome backfill is working: pending signals, oldest unfilled age.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_hourly_performance",
                "description": "Win rate by hour-of-day (UTC). Reveals time-dependent patterns.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            handlers = {
                "get_tier_stats": lambda: self._tier_stats(conn),
                "get_recent_signals": lambda: self._recent(conn, tool_input.get("count", 20)),
                "get_direction_breakdown": lambda: self._direction(conn),
                "get_regime_breakdown": lambda: self._regime(conn),
                "get_streak_analysis": lambda: self._streaks(conn),
                "get_backfill_status": lambda: self._backfill(conn),
                "get_hourly_performance": lambda: self._hourly(conn),
            }
            handler = handlers.get(tool_name)
            if not handler:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return json.dumps(handler(), default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            conn.close()

    def _tier_stats(self, conn) -> dict:
        result = {}
        with conn.cursor() as cur:
            for tier in ["Strong", "Moderate"]:
                cur.execute("""
                    SELECT COUNT(*) as t, SUM(filled) as f, SUM(correct) as w,
                           AVG(CASE WHEN filled=1 THEN actual_return_4h END) as avg_ret,
                           AVG(CASE WHEN filled=1 AND correct=1 THEN actual_return_4h END) as avg_win,
                           AVG(CASE WHEN filled=1 AND correct=0 THEN actual_return_4h END) as avg_loss,
                           MAX(CASE WHEN filled=1 THEN actual_return_4h END) as best,
                           MIN(CASE WHEN filled=1 THEN actual_return_4h END) as worst
                    FROM tracked_signals WHERE strength=%s AND signal_time>=%s
                """, (tier, DUAL_MODEL_START))
                r = cur.fetchone()
                t, f, w = int(r["t"] or 0), int(r["f"] or 0), int(r["w"] or 0)
                d = {"total": t, "filled": f, "pending": t-f, "wins": w, "losses": f-w,
                     "win_rate_pct": round(w/f*100, 1) if f > 0 else None,
                     "avg_return_pct": round(float(r["avg_ret"] or 0)*100, 3),
                     "avg_win_pct": round(float(r["avg_win"] or 0)*100, 3) if r["avg_win"] else None,
                     "avg_loss_pct": round(float(r["avg_loss"] or 0)*100, 3) if r["avg_loss"] else None,
                     "best_pct": round(float(r["best"] or 0)*100, 3),
                     "worst_pct": round(float(r["worst"] or 0)*100, 3)}
                if f > 0 and d["avg_win_pct"] and d["avg_loss_pct"]:
                    wr = w / f
                    d["expectancy_pct"] = round(wr * d["avg_win_pct"] + (1-wr) * d["avg_loss_pct"], 3)
                result[tier] = d
        return result

    def _recent(self, conn, count) -> list:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, strength, confidence, entry_price,
                       exit_price, actual_return_4h, correct, filled, regime
                FROM tracked_signals WHERE signal_time >= %s
                ORDER BY signal_time DESC LIMIT %s
            """, (DUAL_MODEL_START, count))
            return [
                {"time": str(r["signal_time"]), "dir": r["direction"], "tier": r["strength"],
                 "conf": round(float(r["confidence"]), 1) if r["confidence"] else None,
                 "entry": float(r["entry_price"]) if r["entry_price"] else None,
                 "ret_pct": round(float(r["actual_return_4h"])*100, 3) if r["actual_return_4h"] else None,
                 "correct": bool(r["correct"]) if r["correct"] is not None else None,
                 "filled": bool(r["filled"]), "regime": r["regime"] or "?"}
                for r in cur.fetchall()
            ]

    def _direction(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT direction, strength, COUNT(*) as n, SUM(correct) as w,
                       AVG(actual_return_4h) as avg_ret
                FROM tracked_signals WHERE filled=1 AND signal_time>=%s
                GROUP BY direction, strength
            """, (DUAL_MODEL_START,))
            result = {}
            for r in cur.fetchall():
                k = f"{r['direction']}_{r['strength']}"
                n = int(r["n"])
                w = int(r["w"] or 0)
                result[k] = {"count": n, "wins": w,
                             "win_rate_pct": round(w/n*100, 1) if n > 0 else None,
                             "avg_return_pct": round(float(r["avg_ret"] or 0)*100, 3)}
            return result

    def _regime(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT regime, direction, COUNT(*) as n, SUM(correct) as w,
                       AVG(actual_return_4h) as avg_ret
                FROM tracked_signals WHERE filled=1 AND signal_time>=%s AND regime!=''
                GROUP BY regime, direction
            """, (DUAL_MODEL_START,))
            result = {}
            for r in cur.fetchall():
                k = f"{r['regime']}_{r['direction']}"
                n = int(r["n"])
                w = int(r["w"] or 0)
                result[k] = {"count": n, "wins": w,
                             "win_rate_pct": round(w/n*100, 1) if n > 0 else None,
                             "avg_return_pct": round(float(r["avg_ret"] or 0)*100, 3)}
            return result

    def _streaks(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT correct FROM tracked_signals
                WHERE filled=1 AND signal_time>=%s ORDER BY signal_time ASC
            """, (DUAL_MODEL_START,))
            results = [bool(r["correct"]) for r in cur.fetchall()]
        if not results:
            return {"total": 0}
        # Current streak
        streak = 1
        is_win = results[-1]
        for i in range(len(results)-2, -1, -1):
            if results[i] == is_win:
                streak += 1
            else:
                break
        # Max streaks
        mw, ml, cw, cl = 0, 0, 0, 0
        for r in results:
            if r:
                cw += 1; cl = 0; mw = max(mw, cw)
            else:
                cl += 1; cw = 0; ml = max(ml, cl)
        return {
            "total": len(results),
            "current_streak": streak, "current_type": "WIN" if is_win else "LOSS",
            "longest_win": mw, "longest_loss": ml,
            "last_10": ["W" if r else "L" for r in results[-10:]],
        }

    def _backfill(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as n, MIN(signal_time) as oldest, MAX(signal_time) as newest
                FROM tracked_signals WHERE filled=0 AND signal_time>=%s
            """, (DUAL_MODEL_START,))
            r = cur.fetchone()
            pending = int(r["n"] or 0)
            age_h = None
            if r["oldest"]:
                o = r["oldest"].replace(tzinfo=timezone.utc) if hasattr(r["oldest"], 'replace') else r["oldest"]
                age_h = round((datetime.now(timezone.utc) - o).total_seconds() / 3600, 1)
            return {"pending": pending, "oldest": str(r["oldest"]) if r["oldest"] else None,
                    "oldest_age_hours": age_h, "healthy": pending == 0 or (age_h is not None and age_h < 8)}

    def _hourly(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT HOUR(signal_time) as h, COUNT(*) as n, SUM(correct) as w
                FROM tracked_signals WHERE filled=1 AND signal_time>=%s
                GROUP BY HOUR(signal_time) ORDER BY h
            """, (DUAL_MODEL_START,))
            return {f"{r['h']:02d}:00": {"count": int(r["n"]),
                    "win_rate_pct": round(int(r["w"] or 0)/int(r["n"])*100, 1)}
                    for r in cur.fetchall() if int(r["n"]) >= 3}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = SignalTrackerAgent()
    if "--context-only" in sys.argv:
        for tool in agent.get_tools():
            print(f"\n--- {tool['name']} ---")
            print(agent.execute_tool(tool["name"], {}))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Tools: {result.tool_calls_made} | Turns: {result.turns_used}")
