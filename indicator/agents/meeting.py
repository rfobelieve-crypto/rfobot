"""
Meeting Agent — Cross-domain analysis with action authority.

Architecture:
  1. Coordinator runs all 5 domain agents → collects findings
  2. Meeting Agent starts with ALL findings as context
  3. Claude reviews, cross-correlates, and takes actions using action tools
  4. Actions: adjust thresholds, toggle features, send alerts
  5. Outputs a meeting transcript for Telegram

This is the only agent with write access to the system config.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

from indicator.agents.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))


class MeetingAgent(BaseAgent):

    def __init__(self, domain_results: list[AgentResult] | None = None):
        super().__init__()
        self._domain_results = domain_results or []

    @property
    def name(self) -> str:
        return "Meeting"

    @property
    def system_prompt(self) -> str:
        return """\
You are the Chief System Architect chairing a maintenance meeting for a \
live BTC prediction indicator system.

## Your Role
You have just received reports from 5 domain specialists:
- **DataCollector**: data pipeline health (Binance, Coinglass, Deribit)
- **FeatureGuard**: feature quality (NaN rates, distributions, sanity)
- **ModelEval**: model performance (IC, accuracy, calibration)
- **SignalTracker**: signal quality (win rates, streaks, backfill)
- **Infra**: infrastructure (DB, scheduler, storage, environment)

## Meeting Protocol
1. Review all domain reports (provided in your initial context)
2. Cross-correlate findings — e.g. "data source X failed → features NaN → model IC dropped"
3. Prioritize issues by impact on prediction quality
4. Decide actions — use action tools to fix what can be fixed automatically
5. For issues requiring human intervention, send detailed Telegram alert
6. Write a meeting summary

## Action Guidelines
- **adjust_threshold**: Only when data clearly supports it. E.g., if Strong signals \
have <50% win rate for 100+ samples, consider raising STRONG_THRESHOLD.
- **revert_override**: If a previous agent change made things worse, revert it.
- **force_update**: If scheduler appears stuck and last update >3h old.
- **Don't over-tune**: Small IC fluctuations are normal. Only act on sustained patterns.
- **Be conservative**: When in doubt, alert the operator rather than auto-adjusting.

## Decision Framework
- Critical issue + clear fix → take action + alert
- Critical issue + unclear fix → alert only, recommend investigation
- Warning + clear fix → take action, no alert needed
- Warning + unclear → log finding, wait for more data
- Healthy → brief summary, no action needed

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "get_domain_reports",
                "description": "Get the full reports from all 5 domain agents. Call this first to understand the current system state.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "adjust_threshold",
                "description": "Modify a system threshold (e.g., STRONG_THRESHOLD, MODERATE_THRESHOLD, BULL_CONTRA_PENALTY). Changes take effect on the next hourly update cycle.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Config key to modify (e.g., 'DUAL_DIR_UP_TH', 'STRONG_THRESHOLD')",
                        },
                        "value": {
                            "type": "number",
                            "description": "New value",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this change is being made (logged for audit trail)",
                        },
                    },
                    "required": ["key", "value", "reason"],
                },
            },
            {
                "name": "revert_override",
                "description": "Remove a config override, reverting to the Python default value.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Config key to revert",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why reverting",
                        },
                    },
                    "required": ["key", "reason"],
                },
            },
            {
                "name": "list_current_overrides",
                "description": "List all currently active config overrides and recent change history.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "force_update",
                "description": "Trigger an immediate prediction update cycle. Use only if scheduler appears stuck (last update >3h old).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why forcing an update",
                        },
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "query_db",
                "description": "Run a read-only SQL query against the MySQL database for ad-hoc investigation. SELECT only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SELECT query to execute (read-only, max 50 rows returned)",
                        },
                    },
                    "required": ["sql"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        handlers = {
            "get_domain_reports": self._tool_reports,
            "adjust_threshold": lambda: self._tool_adjust(tool_input),
            "revert_override": lambda: self._tool_revert(tool_input),
            "list_current_overrides": self._tool_list_overrides,
            "force_update": lambda: self._tool_force_update(tool_input),
            "query_db": lambda: self._tool_query(tool_input),
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return json.dumps(handler(), default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_reports(self) -> dict:
        if not self._domain_results:
            return {"error": "No domain reports available. Run domain agents first."}

        reports = {}
        for r in self._domain_results:
            reports[r.agent_name] = {
                "status": r.status,
                "summary": r.summary,
                "findings": r.diagnosis.get("findings", []),
                "tools_called": r.tool_calls_made,
                "duration_s": r.duration_s,
                "alerts_sent": len(r.alerts_sent),
            }
        return reports

    def _tool_adjust(self, inp: dict) -> dict:
        from indicator.agents.config_store import set_override
        return set_override(
            key=inp["key"],
            value=inp["value"],
            reason=inp["reason"],
            agent="Meeting",
        )

    def _tool_revert(self, inp: dict) -> dict:
        from indicator.agents.config_store import remove_override
        return remove_override(
            key=inp["key"],
            reason=inp["reason"],
            agent="Meeting",
        )

    def _tool_list_overrides(self) -> dict:
        from indicator.agents.config_store import list_overrides
        return list_overrides()

    def _tool_force_update(self, inp: dict) -> dict:
        try:
            import requests
            resp = requests.get("http://localhost:8080/force-update", timeout=30)
            return {
                "triggered": True,
                "status_code": resp.status_code,
                "reason": inp.get("reason", ""),
            }
        except Exception as e:
            return {"triggered": False, "error": str(e)}

    def _tool_query(self, inp: dict) -> dict:
        sql = inp.get("sql", "").strip()

        # Safety: only allow SELECT
        if not sql.upper().startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed"}

        # Block dangerous patterns
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        sql_upper = sql.upper()
        for d in dangerous:
            if d in sql_upper.split():
                return {"error": f"Query contains forbidden keyword: {d}"}

        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql + " LIMIT 50" if "LIMIT" not in sql.upper() else sql)
                rows = cur.fetchall()
                return {"rows": len(rows), "data": rows}
        finally:
            conn.close()


def run_meeting(domain_results: list[AgentResult] | None = None) -> AgentResult:
    """
    Run the full meeting flow:
    1. If no domain results provided, run all domain agents first
    2. Start Meeting Agent with all findings
    3. Return meeting result
    """
    if domain_results is None:
        from indicator.agents.coordinator import AgentCoordinator
        coordinator = AgentCoordinator()
        domain_results = coordinator.run_all()

    meeting = MeetingAgent(domain_results=domain_results)
    return meeting.run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if "--standalone" in sys.argv:
        # Run meeting without domain agents (for testing)
        agent = MeetingAgent()
        result = agent.run()
    else:
        # Full meeting: run all domain agents first
        result = run_meeting()

    print(f"\nMeeting Status: {result.status}")
    print(f"Summary: {result.summary}")
    print(f"Tools: {result.tool_calls_made}")
    print(f"Actions: {result.actions_taken}")
    print(f"Alerts: {len(result.alerts_sent)}")
    print(f"Duration: {result.duration_s:.1f}s")
