"""
Watchdog — Autonomous agent scheduler.

Runs agents on a schedule. Each agent investigates, diagnoses, and
self-heals if it finds problems. The watchdog just decides WHEN to run,
not what to do — that's up to each agent's Claude brain.

Schedule:
  - Every 4 hours: Full sweep (all 5 agents)
  - Every 1 hour: Quick sweep (DataCollector + Infra only — cheapest checks)
  - On-demand: /meeting triggers all agents

Integration:
  Called from indicator/app.py via APScheduler, or standalone via CLI.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
TZ_TPE = timezone(timedelta(hours=8))

# Cooldown: don't run the same agent again within this many seconds
AGENT_COOLDOWNS = {
    "DataCollector": 1800,   # 30 min
    "FeatureGuard": 3600,    # 1h
    "ModelEval": 7200,       # 2h
    "SignalTracker": 7200,   # 2h
    "Infra": 1800,           # 30 min
}

_last_run: dict[str, float] = {}


def _should_run(agent_name: str) -> bool:
    """Check if enough time has passed since last run."""
    last = _last_run.get(agent_name, 0)
    cooldown = AGENT_COOLDOWNS.get(agent_name, 3600)
    return (time.time() - last) > cooldown


def run_quick_sweep() -> dict:
    """Quick sweep: DataCollector + Infra (cheapest, most critical).

    Run every hour alongside the prediction update cycle.
    """
    from indicator.agents.data_collector import DataCollectorAgent
    from indicator.agents.infra import InfraAgent

    results = {}
    agents = [("DataCollector", DataCollectorAgent), ("Infra", InfraAgent)]

    for name, cls in agents:
        if not _should_run(name):
            results[name] = {"skipped": True, "reason": "cooldown"}
            continue
        try:
            agent = cls()
            result = agent.run()
            _last_run[name] = time.time()
            results[name] = {
                "status": result.status,
                "summary": result.summary,
                "actions": result.actions_taken,
                "duration_s": result.duration_s,
            }
            logger.info("[Watchdog] %s: %s (%.1fs)", name, result.status, result.duration_s)
        except Exception as e:
            logger.exception("[Watchdog] %s crashed: %s", name, e)
            results[name] = {"status": "error", "error": str(e)}

    return results


def run_full_sweep() -> dict:
    """Full sweep: all 5 agents. Run every 4 hours."""
    from indicator.agents.coordinator import AgentCoordinator

    logger.info("[Watchdog] Starting full sweep...")
    t0 = time.time()

    coordinator = AgentCoordinator()
    agent_results = coordinator.run_all()

    results = {}
    for r in agent_results:
        _last_run[r.agent_name] = time.time()
        results[r.agent_name] = {
            "status": r.status,
            "summary": r.summary,
            "actions": r.actions_taken,
            "alerts": len(r.alerts_sent),
            "tools": r.tool_calls_made,
            "duration_s": r.duration_s,
        }

    total = time.time() - t0
    statuses = [r["status"] for r in results.values()]
    overall = "critical" if "critical" in statuses else \
              "degraded" if "degraded" in statuses else \
              "error" if "error" in statuses else "healthy"

    logger.info("[Watchdog] Full sweep done in %.1fs — overall=%s", total, overall)

    # Send summary to Telegram
    _send_sweep_summary(results, total, overall)

    return {"overall": overall, "agents": results, "total_s": round(total, 1)}


def _send_sweep_summary(results: dict, total_s: float, overall: str):
    """Send sweep summary to Telegram (only if something is wrong or actions were taken)."""
    # Skip notification if everything is healthy and no actions taken
    all_healthy = all(r["status"] == "healthy" for r in results.values())
    any_actions = any(r.get("actions") for r in results.values())

    if all_healthy and not any_actions:
        return  # silent when healthy

    import os
    import requests

    bot_token = os.environ.get("AGENT_BOT_TOKEN") or os.environ.get("INDICATOR_BOT_TOKEN", "")
    chat_id = os.environ.get("AGENT_CHAT_ID") or os.environ.get("INDICATOR_CHAT_ID", "")
    if not bot_token or not chat_id:
        return

    icons = {"healthy": "\U0001f7e2", "degraded": "\U0001f7e1", "critical": "\U0001f534", "error": "\u26ab"}
    overall_icon = icons.get(overall, "\u26aa")

    lines = [f"{overall_icon} <b>Watchdog Sweep</b>\n"]
    for name, r in results.items():
        icon = icons.get(r["status"], "\u26aa")
        line = f"{icon} <b>{name}</b>: {r.get('summary', r['status'])}"
        if r.get("actions"):
            line += f"\n  \U0001f527 " + ", ".join(r["actions"])
        lines.append(line)

    lines.append(f"\n<i>Total: {total_s:.0f}s</i>")

    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("[Watchdog] Telegram send failed: %s", e)


# ── CLI ────────────────────────────────────────────��────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "quick":
        result = run_quick_sweep()
    else:
        result = run_full_sweep()

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
