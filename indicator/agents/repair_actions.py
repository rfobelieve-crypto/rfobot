"""
Repair Actions — Autonomous repairs + suggestions with strict risk control.

Risk framework (from operator's risk matrix):
  ★☆☆☆☆  DB cleanup, scheduler restart      → AUTO (safe)
  ★★☆☆☆  Data update, cache fallback         → AUTO (safe)
  ★★★☆☆  NaN/clipping handling               → AUTO with log
  ★★★★☆  Regime logic, signal generation      → SUGGEST ONLY
  ★★★★★  Hyperparameters, feature cols        → SUGGEST ONLY (most lethal)

AUTO actions execute immediately and notify operator.
SUGGEST actions only send a Telegram recommendation — operator decides.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
TZ_TPE = timezone(timedelta(hours=8))

CACHE_DIR = Path("indicator/model_artifacts/.data_cache")
AUDIT_PATH = Path("indicator/model_artifacts/repair_audit.jsonl")


def _audit_log(agent: str, action: str, detail: str, result: str):
    """Append to audit log (every action, auto or suggest)."""
    entry = {
        "time": datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent,
        "action": action,
        "detail": detail,
        "result": result,
    }
    try:
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error("Audit log write failed: %s", e)


def _send_tg(message: str):
    """Send Telegram notification."""
    import requests
    bot_token = os.environ.get("AGENT_BOT_TOKEN") or os.environ.get("INDICATOR_BOT_TOKEN", "")
    chat_id = os.environ.get("AGENT_CHAT_ID") or os.environ.get("INDICATOR_CHAT_ID", "")
    if not bot_token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
    except Exception as e:
        logger.error("Telegram send failed: %s", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTO ACTIONS — safe to execute without human approval (★~★★)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clear_stale_cache(agent: str = "unknown") -> dict:
    """[AUTO ★★] Delete cache files older than 12h to force fresh fetches.

    Safe because: only affects data freshness, not model behavior.
    """
    if not CACHE_DIR.exists():
        return {"cleared": 0, "note": "cache dir does not exist"}

    cleared = []
    now_ts = datetime.now().timestamp()
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            age_h = (now_ts - f.stat().st_mtime) / 3600
            if age_h > 12:
                f.unlink()
                cleared.append({"file": f.name, "age_hours": round(age_h, 1)})
        except Exception as e:
            logger.error("Failed to clear cache %s: %s", f.name, e)

    _audit_log(agent, "AUTO:clear_stale_cache", f"cleared {len(cleared)} files",
               json.dumps(cleared, ensure_ascii=False))
    return {"cleared": len(cleared), "files": cleared}


def trigger_force_update(agent: str = "unknown") -> dict:
    """[AUTO ★★] Trigger an immediate prediction update cycle.

    Safe because: just re-runs the normal update pipeline with fresh data.
    """
    import requests as req

    urls = []
    svc = os.environ.get("INDICATOR_SERVICE_URL", "")
    if svc:
        urls.append(f"{svc}/force-update?sync=0")
    urls.append("http://localhost:8080/force-update?sync=0")

    for url in urls:
        try:
            resp = req.get(url, timeout=10)
            if resp.status_code == 200:
                _audit_log(agent, "AUTO:trigger_force_update", url, "triggered")
                return {"triggered": True, "url": url}
        except Exception:
            continue

    _audit_log(agent, "AUTO:trigger_force_update", "all URLs failed", "failed")
    return {"triggered": False, "error": "Could not reach indicator service"}


def kill_idle_db_connections(max_idle_s: int = 300, agent: str = "unknown") -> dict:
    """[AUTO ★] Kill MySQL connections idle for > max_idle_s seconds.

    Safe because: only kills Sleep-state connections, never active queries.
    """
    from shared.db import get_db_conn
    conn = get_db_conn()
    killed = []
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW PROCESSLIST")
            for p in cur.fetchall():
                state = p.get("Command", "")
                idle_time = int(p.get("Time", 0))
                pid = p.get("Id")
                if state == "Sleep" and idle_time > max_idle_s and pid:
                    try:
                        cur.execute(f"KILL {pid}")
                        killed.append({"id": pid, "idle_s": idle_time})
                    except Exception:
                        pass
    finally:
        conn.close()

    _audit_log(agent, "AUTO:kill_idle_connections",
               f"threshold={max_idle_s}s", f"killed {len(killed)}")
    return {"killed": len(killed), "connections": killed}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUGGEST ACTIONS — send recommendation to operator, never auto-execute
# ★★★★★ Hyperparameters, thresholds, signal logic
# ★★★★☆ Regime logic, signal generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def suggest_threshold_change(
    key: str, current_value: float, suggested_value: float,
    evidence: str, agent: str = "unknown",
) -> dict:
    """[SUGGEST ★★★★★] Recommend a threshold change to the operator.

    NEVER auto-executes. Sends a Telegram message with evidence and
    the operator decides whether to apply it manually.
    """
    msg = (
        f"💡 <b>[{agent}] 參數建議</b>\n\n"
        f"<b>參數:</b> <code>{key}</code>\n"
        f"<b>目前值:</b> {current_value}\n"
        f"<b>建議值:</b> {suggested_value}\n\n"
        f"<b>依據:</b>\n{evidence}\n\n"
        f"<i>⚠️ 此為建議，不會自動執行。\n"
        f"如需套用，請手動修改 config_overrides.json\n"
        f"或使用指令: /adjust {key} {suggested_value}</i>"
    )
    _send_tg(msg)
    _audit_log(agent, "SUGGEST:threshold_change",
               f"{key}: {current_value} → {suggested_value}", evidence[:200])
    return {
        "action": "suggest_only",
        "key": key,
        "current": current_value,
        "suggested": suggested_value,
        "telegram_sent": True,
    }


def suggest_pause_signals(
    tier: str, win_rate: float, sample_size: int,
    evidence: str, agent: str = "unknown",
) -> dict:
    """[SUGGEST ★★★★★] Recommend pausing a signal tier.

    Signal generation is ★★★★☆ risk — never auto-modify.
    """
    msg = (
        f"🚨 <b>[{agent}] 信號暫停建議</b>\n\n"
        f"<b>層級:</b> {tier}\n"
        f"<b>目前勝率:</b> {win_rate:.1f}% ({sample_size} 筆)\n\n"
        f"<b>依據:</b>\n{evidence}\n\n"
        f"<i>⚠️ 此為建議，不會自動執行。\n"
        f"如需暫停，請手動設定 {tier.upper()}_THRESHOLD = 100</i>"
    )
    _send_tg(msg)
    _audit_log(agent, "SUGGEST:pause_signals",
               f"{tier}: {win_rate:.1f}% over {sample_size}", evidence[:200])
    return {
        "action": "suggest_only",
        "tier": tier,
        "win_rate": win_rate,
        "sample_size": sample_size,
        "telegram_sent": True,
    }


def suggest_widen_deadzone(
    current_dz: float, suggested_mult: float,
    evidence: str, agent: str = "unknown",
) -> dict:
    """[SUGGEST ★★★★★] Recommend widening the neutral deadzone.

    Deadzone is a hyperparameter (★★★★★) — never auto-adjust.
    """
    suggested_dz = round(current_dz * suggested_mult, 3)
    msg = (
        f"💡 <b>[{agent}] Deadzone 建議</b>\n\n"
        f"<b>目前:</b> STRENGTH_DEADZONE = {current_dz}\n"
        f"<b>建議:</b> {suggested_dz} ({suggested_mult}x)\n\n"
        f"<b>依據:</b>\n{evidence}\n\n"
        f"<i>⚠️ 此為建議，不會自動執行。</i>"
    )
    _send_tg(msg)
    _audit_log(agent, "SUGGEST:widen_deadzone",
               f"{current_dz} → {suggested_dz} ({suggested_mult}x)", evidence[:200])
    return {
        "action": "suggest_only",
        "current": current_dz,
        "suggested": suggested_dz,
        "multiplier": suggested_mult,
        "telegram_sent": True,
    }


def suggest_regime_change(
    what: str, evidence: str, agent: str = "unknown",
) -> dict:
    """[SUGGEST ★★★★☆] Recommend a regime detection logic change.

    Regime logic is ★★★★☆ risk — never auto-modify.
    """
    msg = (
        f"💡 <b>[{agent}] Regime 建議</b>\n\n"
        f"<b>建議:</b> {what}\n\n"
        f"<b>依據:</b>\n{evidence}\n\n"
        f"<i>⚠️ 此為建議，需要人工評估後手動修改。</i>"
    )
    _send_tg(msg)
    _audit_log(agent, "SUGGEST:regime_change", what, evidence[:200])
    return {"action": "suggest_only", "suggestion": what, "telegram_sent": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Definitions (Anthropic API format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPAIR_TOOLS = {
    # ── AUTO tools (safe) ──────────────────────────────────────────
    "clear_stale_cache": {
        "name": "repair_clear_stale_cache",
        "description": (
            "[AUTO-SAFE] Delete local cache files older than 12h to force fresh API fetches. "
            "Use when: data source is alive but system serves stale cache. "
            "Risk: ★★ — only affects data freshness."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "trigger_force_update": {
        "name": "repair_trigger_update",
        "description": (
            "[AUTO-SAFE] Trigger an immediate prediction update cycle. "
            "Use when: scheduler stuck (last update >2h) or after fixing data issues. "
            "Risk: ★★ — just re-runs normal pipeline."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "kill_idle_connections": {
        "name": "repair_kill_idle_connections",
        "description": (
            "[AUTO-SAFE] Kill MySQL connections idle >5min. "
            "Use when: active_connections > 20 or connection errors. "
            "Risk: ★ — only kills Sleep-state, never active queries."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },

    # ── SUGGEST tools (human approval required) ────────────────────
    "suggest_threshold": {
        "name": "suggest_threshold_change",
        "description": (
            "[SUGGEST-ONLY] Send a threshold change RECOMMENDATION to the operator via Telegram. "
            "This does NOT modify anything — the operator decides. "
            "Use when: data shows a threshold may need adjustment. "
            "Risk: ★★★★★ — hyperparameters affect model output distribution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Parameter name (e.g. STRONG_THRESHOLD)"},
                "current_value": {"type": "number", "description": "Current value of the parameter"},
                "suggested_value": {"type": "number", "description": "Recommended new value"},
                "evidence": {"type": "string", "description": "Data evidence supporting this change (IC, win rate, sample size)"},
            },
            "required": ["key", "current_value", "suggested_value", "evidence"],
        },
    },
    "suggest_pause": {
        "name": "suggest_pause_signals",
        "description": (
            "[SUGGEST-ONLY] Recommend pausing a signal tier to the operator. "
            "Does NOT actually pause — operator decides. "
            "Use when: tier win rate < 45% over 50+ filled signals. "
            "Risk: ★★★★★ — affects signal generation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["Strong", "Moderate"]},
                "win_rate": {"type": "number", "description": "Current win rate %"},
                "sample_size": {"type": "integer", "description": "Number of filled signals"},
                "evidence": {"type": "string", "description": "Detailed evidence"},
            },
            "required": ["tier", "win_rate", "sample_size", "evidence"],
        },
    },
    "suggest_deadzone": {
        "name": "suggest_widen_deadzone",
        "description": (
            "[SUGGEST-ONLY] Recommend widening the neutral deadzone to the operator. "
            "Does NOT actually change — operator decides. "
            "Use when: IC near zero, signals are noise. "
            "Risk: ★★★★★ — deadzone is a core hyperparameter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_dz": {"type": "number", "description": "Current STRENGTH_DEADZONE value"},
                "suggested_mult": {"type": "number", "description": "Suggested multiplier (1.0~3.0)"},
                "evidence": {"type": "string", "description": "IC data and duration supporting this"},
            },
            "required": ["current_dz", "suggested_mult", "evidence"],
        },
    },
    "suggest_regime": {
        "name": "suggest_regime_change",
        "description": (
            "[SUGGEST-ONLY] Recommend a regime detection logic change. "
            "Does NOT modify anything — operator decides. "
            "Risk: ★★★★☆ — regime affects blend weights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "what": {"type": "string", "description": "What to change and why"},
                "evidence": {"type": "string", "description": "Data supporting the suggestion"},
            },
            "required": ["what", "evidence"],
        },
    },
}

# Which agents get which tools
AGENT_REPAIR_MAP = {
    "DataCollector": ["clear_stale_cache", "trigger_force_update"],
    "FeatureGuard": ["clear_stale_cache", "trigger_force_update"],
    "ModelEval": ["suggest_threshold", "suggest_deadzone", "suggest_regime"],
    "SignalTracker": ["suggest_threshold", "suggest_pause", "suggest_deadzone"],
    "Infra": ["trigger_force_update", "kill_idle_connections"],
}


def get_repair_tools_for(agent_name: str) -> list[dict]:
    """Get repair/suggest tool definitions for a specific agent."""
    tool_keys = AGENT_REPAIR_MAP.get(agent_name, [])
    return [REPAIR_TOOLS[k] for k in tool_keys if k in REPAIR_TOOLS]


def execute_repair_tool(agent_name: str, tool_name: str, tool_input: dict) -> str:
    """Execute a repair or suggest tool call."""
    dispatch = {
        # AUTO actions
        "repair_clear_stale_cache": lambda: clear_stale_cache(agent=agent_name),
        "repair_trigger_update": lambda: trigger_force_update(agent=agent_name),
        "repair_kill_idle_connections": lambda: kill_idle_db_connections(agent=agent_name),
        # SUGGEST actions (Telegram only, no system changes)
        "suggest_threshold_change": lambda: suggest_threshold_change(
            key=tool_input["key"],
            current_value=tool_input["current_value"],
            suggested_value=tool_input["suggested_value"],
            evidence=tool_input["evidence"],
            agent=agent_name,
        ),
        "suggest_pause_signals": lambda: suggest_pause_signals(
            tier=tool_input["tier"],
            win_rate=tool_input["win_rate"],
            sample_size=tool_input["sample_size"],
            evidence=tool_input["evidence"],
            agent=agent_name,
        ),
        "suggest_widen_deadzone": lambda: suggest_widen_deadzone(
            current_dz=tool_input["current_dz"],
            suggested_mult=tool_input["suggested_mult"],
            evidence=tool_input["evidence"],
            agent=agent_name,
        ),
        "suggest_regime_change": lambda: suggest_regime_change(
            what=tool_input["what"],
            evidence=tool_input["evidence"],
            agent=agent_name,
        ),
    }

    handler = dispatch.get(tool_name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return json.dumps(handler(), default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})
