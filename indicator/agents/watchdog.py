"""
Watchdog — Gatekeeper + autonomous agent scheduler.

Cost-optimized: runs FREE rule-based checks first, only invokes
Claude (API cost) when something looks abnormal.

Flow:
  1. Gatekeeper: run tools directly (no Claude) → collect raw metrics
  2. Rule check: compare against thresholds
  3. If ALL healthy → stop ($0 cost)
  4. If anomaly found → invoke the relevant agent(s) with Claude brain
     Claude investigates deeper, diagnoses, and triggers repair/suggest

Schedule:
  - Every hour at :15  → gatekeeper check (free unless anomaly)
  - Every 4h at :20    → full sweep (gatekeeper first, then Claude if needed)
  - On-demand /meeting → full sweep (always invokes Claude for all agents)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
TZ_TPE = timezone(timedelta(hours=8))

# ── Gatekeeper thresholds (rule-based, no Claude needed) ────────────

RULES = {
    "binance_age_max_min": 120,       # Binance data older than 2h → alarm
    "db_ping_max_ms": 500,            # DB ping > 500ms → alarm
    "db_connections_max": 20,         # too many connections → alarm
    "scheduler_gap_max_min": 130,     # last update > 2h10m → alarm
    "nan_rate_max_pct": 30.0,         # feature NaN > 30% → alarm
    "ic_7d_alarm_below": -0.05,       # 7d IC deeply negative → alarm
    "signal_winrate_alarm_below": 40, # Strong win rate < 40% → alarm
}


# ── Gatekeeper: free rule checks ────────────────────────────────────

def _check_data_health() -> dict:
    """Check Binance data freshness (no Claude)."""
    try:
        from indicator.agents.data_collector import DataCollectorAgent
        agent = DataCollectorAgent()
        raw = json.loads(agent.execute_tool("check_binance_klines", {}))
        if not raw.get("available"):
            return {"ok": False, "reason": "Binance klines unavailable"}
        age = raw.get("age_minutes", 999)
        if age > RULES["binance_age_max_min"]:
            return {"ok": False, "reason": f"Binance data {age:.0f}min old (>{RULES['binance_age_max_min']})"}
        return {"ok": True, "age_min": round(age, 1)}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


def _check_db_health() -> dict:
    """Check DB connectivity and connections (no Claude)."""
    try:
        from indicator.agents.infra import InfraAgent
        agent = InfraAgent()
        raw = json.loads(agent.execute_tool("check_database", {}))
        if not raw.get("connected"):
            return {"ok": False, "reason": f"DB down: {raw.get('error', '?')}"}
        ping = raw.get("ping_ms", 999)
        conns = raw.get("active_connections", 0)
        issues = []
        if ping > RULES["db_ping_max_ms"]:
            issues.append(f"ping={ping:.0f}ms")
        if conns > RULES["db_connections_max"]:
            issues.append(f"connections={conns}")
        if issues:
            return {"ok": False, "reason": f"DB issues: {', '.join(issues)}"}
        return {"ok": True, "ping_ms": round(ping, 1), "connections": conns}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


def _check_scheduler_health() -> dict:
    """Check if scheduler is firing on time (no Claude)."""
    try:
        from indicator.agents.infra import InfraAgent
        agent = InfraAgent()
        raw = json.loads(agent.execute_tool("check_scheduler", {}))
        age = raw.get("age_minutes")
        if age is None:
            return {"ok": False, "reason": "no data in indicator_history"}
        if age > RULES["scheduler_gap_max_min"]:
            return {"ok": False, "reason": f"last update {age:.0f}min ago (>{RULES['scheduler_gap_max_min']})"}
        return {"ok": True, "age_min": round(age, 1)}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


def _check_feature_health() -> dict:
    """Check feature NaN rate (no Claude)."""
    try:
        from indicator.agents.feature_guard import FeatureGuardAgent
        agent = FeatureGuardAgent()
        raw = json.loads(agent.execute_tool("build_features", {}))
        nan_pct = raw.get("last_row_nan_pct", 0)
        if nan_pct > RULES["nan_rate_max_pct"]:
            return {"ok": False, "reason": f"NaN rate {nan_pct}% (>{RULES['nan_rate_max_pct']}%)"}
        return {"ok": True, "nan_pct": nan_pct}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


def _check_model_health() -> dict:
    """Check IC trend (no Claude)."""
    try:
        from indicator.agents.model_eval import ModelEvalAgent
        agent = ModelEvalAgent()
        raw = json.loads(agent.execute_tool("get_ic_overview", {}))
        ic_7d = raw.get("last_7d")
        if ic_7d is None:
            return {"ok": True, "note": "insufficient data"}
        ic_val = ic_7d.get("ic")
        if ic_val is not None and ic_val < RULES["ic_7d_alarm_below"]:
            return {"ok": False, "reason": f"7d IC = {ic_val} (<{RULES['ic_7d_alarm_below']})"}
        return {"ok": True, "ic_7d": ic_val}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


def _check_signal_health() -> dict:
    """Check signal win rates (no Claude)."""
    try:
        from indicator.agents.signal_tracker import SignalTrackerAgent
        agent = SignalTrackerAgent()
        raw = json.loads(agent.execute_tool("get_tier_stats", {}))
        strong = raw.get("Strong", {})
        wr = strong.get("win_rate_pct")
        filled = strong.get("filled", 0)
        if wr is not None and filled >= 20 and wr < RULES["signal_winrate_alarm_below"]:
            return {"ok": False, "reason": f"Strong win rate {wr}% ({filled} signals)"}
        return {"ok": True, "strong_wr": wr, "filled": filled}
    except Exception as e:
        return {"ok": False, "reason": f"check failed: {e}"}


# Gatekeeper check → agent mapping
GATE_TO_AGENT = {
    "data": "DataCollector",
    "db": "Infra",
    "scheduler": "Infra",
    "features": "FeatureGuard",
    "model": "ModelEval",
    "signals": "SignalTracker",
}


def run_gatekeeper() -> dict:
    """Run all gatekeeper checks. Returns which domains have issues."""
    checks = {
        "data": _check_data_health,
        "db": _check_db_health,
        "scheduler": _check_scheduler_health,
        "features": _check_feature_health,
        "model": _check_model_health,
        "signals": _check_signal_health,
    }

    results = {}
    alarms = []

    for name, fn in checks.items():
        t0 = time.time()
        result = fn()
        result["check_time_s"] = round(time.time() - t0, 2)
        results[name] = result
        if not result.get("ok"):
            alarms.append(name)

    return {"checks": results, "alarms": alarms, "all_ok": len(alarms) == 0}


# ── Agent invocation (only when gatekeeper triggers) ────────────────

def _invoke_agent(agent_name: str) -> dict:
    """Invoke a single agent with Claude brain. This costs API tokens."""
    from indicator.agents.data_collector import DataCollectorAgent
    from indicator.agents.feature_guard import FeatureGuardAgent
    from indicator.agents.model_eval import ModelEvalAgent
    from indicator.agents.signal_tracker import SignalTrackerAgent
    from indicator.agents.infra import InfraAgent

    agent_map = {
        "DataCollector": DataCollectorAgent,
        "FeatureGuard": FeatureGuardAgent,
        "ModelEval": ModelEvalAgent,
        "SignalTracker": SignalTrackerAgent,
        "Infra": InfraAgent,
    }

    cls = agent_map.get(agent_name)
    if not cls:
        return {"status": "error", "error": f"Unknown agent: {agent_name}"}

    try:
        agent = cls()
        result = agent.run()
        return {
            "status": result.status,
            "summary": result.summary,
            "actions": result.actions_taken,
            "alerts": len(result.alerts_sent),
            "tools": result.tool_calls_made,
            "duration_s": result.duration_s,
        }
    except Exception as e:
        logger.exception("[Watchdog] Agent %s failed", agent_name)
        return {"status": "error", "error": str(e)}


# ── Public API ──────────────────────────────────────────────────────

def run_quick_sweep() -> dict:
    """Hourly check: gatekeeper first, Claude only if needed.

    Cost: $0 when healthy. ~$0.03 per agent invoked.
    """
    logger.info("[Watchdog] Quick sweep — running gatekeeper...")
    gate = run_gatekeeper()

    if gate["all_ok"]:
        logger.info("[Watchdog] All clear — no Claude needed ($0)")
        return {"mode": "quick", "gatekeeper": gate, "agents_invoked": [], "cost": "$0"}

    # Determine which agents to invoke
    alarms = gate["alarms"]
    agents_needed = list(set(GATE_TO_AGENT[a] for a in alarms if a in GATE_TO_AGENT))
    logger.info("[Watchdog] Alarms: %s → invoking agents: %s", alarms, agents_needed)

    agent_results = {}
    for name in agents_needed:
        agent_results[name] = _invoke_agent(name)

    # Send summary
    _send_alarm_summary(gate, agent_results)

    return {
        "mode": "quick",
        "gatekeeper": gate,
        "agents_invoked": agents_needed,
        "agent_results": agent_results,
    }


def run_full_sweep() -> dict:
    """4-hourly deep scan: gatekeeper first, then Claude for troubled domains.

    If gatekeeper is all-clear, only invokes ModelEval + SignalTracker
    (slower-changing metrics that benefit from periodic Claude analysis).
    """
    logger.info("[Watchdog] Full sweep — running gatekeeper...")
    gate = run_gatekeeper()

    agents_needed = []

    if gate["all_ok"]:
        # Even when healthy, do periodic deep checks on model/signals (every 4h)
        agents_needed = ["ModelEval", "SignalTracker"]
        logger.info("[Watchdog] Gatekeeper clear — periodic model+signal check only")
    else:
        # Invoke agents for troubled domains + always include model/signals
        alarm_agents = set(GATE_TO_AGENT[a] for a in gate["alarms"] if a in GATE_TO_AGENT)
        alarm_agents.update(["ModelEval", "SignalTracker"])
        agents_needed = list(alarm_agents)
        logger.info("[Watchdog] Alarms: %s → invoking: %s", gate["alarms"], agents_needed)

    t0 = time.time()
    agent_results = {}
    for name in agents_needed:
        agent_results[name] = _invoke_agent(name)

    total = time.time() - t0

    # Send summary (only if issues or actions)
    has_issues = any(r.get("status") in ("degraded", "critical") for r in agent_results.values())
    has_actions = any(r.get("actions") for r in agent_results.values())
    if has_issues or has_actions or not gate["all_ok"]:
        _send_alarm_summary(gate, agent_results)

    return {
        "mode": "full",
        "gatekeeper": gate,
        "agents_invoked": agents_needed,
        "agent_results": agent_results,
        "total_s": round(total, 1),
    }


def run_on_demand() -> dict:
    """On-demand full scan: invoke ALL agents regardless of gatekeeper.

    Triggered by /meeting button. Always costs API tokens.
    """
    logger.info("[Watchdog] On-demand sweep — all agents...")
    gate = run_gatekeeper()

    t0 = time.time()
    all_agents = ["DataCollector", "FeatureGuard", "ModelEval", "SignalTracker", "Infra"]
    agent_results = {}
    for name in all_agents:
        agent_results[name] = _invoke_agent(name)

    total = time.time() - t0
    _send_alarm_summary(gate, agent_results)

    return {
        "mode": "on_demand",
        "gatekeeper": gate,
        "agents_invoked": all_agents,
        "agent_results": agent_results,
        "total_s": round(total, 1),
    }


# ── Telegram notification ───────────────────────────────────────────

def _send_alarm_summary(gate: dict, agent_results: dict):
    """Send summary to Telegram."""
    import requests

    bot_token = os.environ.get("AGENT_BOT_TOKEN") or os.environ.get("INDICATOR_BOT_TOKEN", "")
    chat_id = os.environ.get("AGENT_CHAT_ID") or os.environ.get("INDICATOR_CHAT_ID", "")
    if not bot_token or not chat_id:
        return

    icons = {"healthy": "\U0001f7e2", "degraded": "\U0001f7e1",
             "critical": "\U0001f534", "error": "\u26ab"}

    # Gatekeeper summary
    lines = ["\U0001f916 <b>Watchdog Report</b>\n"]

    # Show gatekeeper alarms
    for check_name, check_result in gate.get("checks", {}).items():
        if check_result.get("ok"):
            continue
        lines.append(f"\u26a0\ufe0f <b>{check_name}</b>: {check_result.get('reason', '?')}")

    if gate.get("all_ok"):
        lines.append("\U0001f7e2 Gatekeeper: all clear")

    # Agent results
    if agent_results:
        lines.append("")
        for name, r in agent_results.items():
            icon = icons.get(r.get("status", "error"), "\u26ab")
            summary = r.get("summary", r.get("status", "?"))
            # Truncate long summaries
            if len(summary) > 100:
                summary = summary[:100] + "..."
            line = f"{icon} <b>{name}</b>: {summary}"
            if r.get("actions"):
                line += "\n  \U0001f527 " + ", ".join(str(a)[:60] for a in r["actions"])
            lines.append(line)

    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("[Watchdog] Telegram send failed: %s", e)


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    if mode == "quick":
        result = run_quick_sweep()
    elif mode == "full":
        result = run_full_sweep()
    elif mode == "gate":
        result = run_gatekeeper()
    elif mode == "demand":
        result = run_on_demand()
    else:
        print(f"Usage: python -m indicator.agents.watchdog [quick|full|gate|demand]")
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
