"""
Base AI Agent — Tool-Use Agentic Loop.

Architecture: Claude gets a set of tools (check_binance, query_db, etc.)
and drives the investigation itself. It decides what to check, in what order,
and can drill down when it finds something suspicious.

Flow:
  1. Agent starts with a system prompt + initial task message
  2. Claude calls tools to investigate
  3. Agent executes tools, returns results to Claude
  4. Claude continues investigating or concludes with diagnosis
  5. Loop until Claude produces final diagnosis (stop_reason = "end_turn")

This is fundamentally different from collect→analyze:
  - Claude chooses WHAT to investigate (not everything)
  - Claude can drill down (call tool A, then based on result, call tool B)
  - Claude can correlate across tools in real time
  - Much more token-efficient for healthy systems (Claude skips what looks fine)
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

# ── Config ───────────────────────────────────────────────────────────────
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "15"))
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")

AGENT_BOT_TOKEN = os.environ.get("AGENT_BOT_TOKEN", "")
AGENT_CHAT_ID = os.environ.get("AGENT_CHAT_ID", "")


@dataclass
class AgentResult:
    """Structured output from an agent run."""
    agent_name: str
    status: str                       # healthy | degraded | critical | error
    summary: str                      # one-line summary in Traditional Chinese
    diagnosis: dict = field(default_factory=dict)
    raw_response: str = ""
    tool_calls_made: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    alerts_sent: list[str] = field(default_factory=list)
    turns_used: int = 0
    timestamp: str = ""
    duration_s: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M:%S")


class BaseAgent(ABC):
    """
    Abstract base for tool-use AI agents.

    Subclasses define:
      - name: agent identifier
      - system_prompt: Claude's role and expertise
      - get_tools(): list of tool definitions (Anthropic tool schema)
      - execute_tool(name, input): run a tool and return result
    """

    def __init__(self):
        self._client = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt with deep domain expertise."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """
        Return tool definitions in Anthropic API format.

        Each tool: {
            "name": "check_binance_klines",
            "description": "Fetch recent Binance klines and return freshness/latency",
            "input_schema": {
                "type": "object",
                "properties": { ... },
                "required": [ ... ]
            }
        }
        """

    @abstractmethod
    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool call and return the result as a string.
        This is where the actual system interaction happens.
        """

    # ── Claude API ──────────────────────────────────────────────────────

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = AGENT_API_KEY
            if not api_key:
                raise RuntimeError(
                    "AGENT_API_KEY not set. "
                    "Set it in env or .env to enable AI agents."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    # ── Telegram ────────────────────────────────────────────────────────

    def send_alert(self, message: str):
        bot_token = AGENT_BOT_TOKEN or os.environ.get("INDICATOR_BOT_TOKEN", "")
        chat_id = AGENT_CHAT_ID or os.environ.get("INDICATOR_CHAT_ID", "")
        if not bot_token or not chat_id:
            logger.warning("[%s] Telegram not configured: %s", self.name, message[:100])
            return
        import requests
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=15,
            )
        except Exception as e:
            logger.error("[%s] Alert send failed: %s", self.name, e)

    # ── Shared tools available to all agents ────────────────────────────

    def _repair_tools(self) -> list[dict]:
        """Repair tools specific to this agent's domain."""
        try:
            from indicator.agents.repair_actions import get_repair_tools_for
            return get_repair_tools_for(self.name)
        except Exception:
            return []

    def _shared_tools(self) -> list[dict]:
        """Tools every agent gets: alert, conclude, + domain repair tools."""
        return [
            {
                "name": "send_telegram_alert",
                "description": (
                    "Send an alert to the operator via Telegram. "
                    "Use for critical or warning findings that need human attention. "
                    "Message should be HTML-formatted, concise, and in Traditional Chinese."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "HTML-formatted alert message",
                        },
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "conclude",
                "description": (
                    "Submit your final diagnosis. Call this when you've gathered "
                    "enough information to make a judgment. This ends the investigation."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "critical"],
                            "description": "Overall system status for your domain",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-line summary in Traditional Chinese",
                        },
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                                    "title": {"type": "string"},
                                    "detail": {"type": "string"},
                                },
                            },
                            "description": "List of findings from your investigation",
                        },
                        "telegram_report": {
                            "type": "string",
                            "description": "2-5 line HTML report for Telegram (Traditional Chinese)",
                        },
                    },
                    "required": ["status", "summary", "findings"],
                },
            },
        ]

    def _execute_shared_tool(self, name: str, inp: dict) -> str | None:
        """Handle shared tools. Returns result string, or None if not a shared tool."""
        if name == "send_telegram_alert":
            msg = inp.get("message", "")
            self.send_alert(f"🤖 <b>[{self.name}]</b>\n{msg}")
            return "Alert sent successfully."
        if name == "conclude":
            # This is handled in the run loop
            return json.dumps(inp)
        return None

    # ── Agentic loop ────────────────────────────────────────────────────

    def run(self) -> AgentResult:
        """
        Run the agentic tool-use loop.

        Claude investigates using tools, then concludes with diagnosis.
        """
        t0 = time.time()
        logger.info("[%s] Starting investigation...", self.name)

        client = self._get_client()
        all_tools = self.get_tools() + self._shared_tools() + self._repair_tools()
        tool_calls_made = []
        actions_taken = []
        alerts_sent = []
        diagnosis = {}

        now = datetime.now(TZ_TPE)
        initial_message = (
            f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+8 "
            f"({now.strftime('%A')})\n\n"
            f"Begin your investigation. Use the tools available to check "
            f"system health in your domain. Start with the most critical "
            f"checks first. When you have enough information, call `conclude` "
            f"with your diagnosis.\n\n"
            f"Respond in Traditional Chinese for all user-facing text."
        )

        messages = [{"role": "user", "content": initial_message}]

        for turn in range(MAX_TURNS):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=self.system_prompt,
                    tools=all_tools,
                    messages=messages,
                )
            except Exception as e:
                logger.error("[%s] Claude API failed on turn %d: %s", self.name, turn, e)
                return AgentResult(
                    agent_name=self.name,
                    status="error",
                    summary=f"Claude API 呼叫失敗: {e}",
                    turns_used=turn,
                    duration_s=time.time() - t0,
                )

            # Process response blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check if Claude is done (no tool use)
            if response.stop_reason == "end_turn":
                # Claude finished without calling conclude — extract text
                text_parts = [b.text for b in assistant_content if hasattr(b, "text")]
                diagnosis = {
                    "status": "healthy",
                    "summary": " ".join(text_parts)[:200] if text_parts else "調查完成",
                    "findings": [],
                }
                break

            # Process tool calls
            tool_results = []
            concluded = False

            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                logger.info("[%s] Turn %d: calling %s", self.name, turn, tool_name)
                tool_calls_made.append(tool_name)

                # Try shared tools first
                shared_result = self._execute_shared_tool(tool_name, tool_input)

                if tool_name == "conclude":
                    diagnosis = tool_input
                    concluded = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Investigation concluded.",
                    })
                    break
                elif tool_name == "send_telegram_alert":
                    alerts_sent.append(tool_input.get("message", ""))
                    actions_taken.append(f"Sent alert: {tool_input.get('message', '')[:50]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": shared_result,
                    })
                elif shared_result is not None:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": shared_result,
                    })
                elif tool_name.startswith("repair_") or tool_name.startswith("suggest_"):
                    # Repair (auto) or Suggest (human-approval) tool
                    try:
                        from indicator.agents.repair_actions import execute_repair_tool
                        result = execute_repair_tool(self.name, tool_name, tool_input)
                        actions_taken.append(f"{tool_name}: {json.dumps(tool_input, ensure_ascii=False)[:80]}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        })
                    except Exception as e:
                        logger.exception("[%s] Tool %s failed", self.name, tool_name)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"Tool error: {e}",
                            "is_error": True,
                        })
                else:
                    # Domain-specific tool
                    try:
                        result = self.execute_tool(tool_name, tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        })
                    except Exception as e:
                        logger.exception("[%s] Tool %s failed", self.name, tool_name)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"Tool error: {e}",
                            "is_error": True,
                        })

            if concluded:
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # Send Telegram report for degraded/critical OR when repairs were made
        status = diagnosis.get("status", "healthy")
        tg_report = diagnosis.get("telegram_report", "")
        if tg_report and status in ("degraded", "critical"):
            icon = "🔴" if status == "critical" else "🟡"
            full_msg = f"{icon} <b>[{self.name}]</b>\n{tg_report}"
            self.send_alert(full_msg)
            alerts_sent.append(full_msg)

        # Notify when AUTO repair actions were taken (suggest tools handle their own TG)
        auto_actions = [a for a in actions_taken if a.startswith("repair_")]
        if auto_actions:
            repair_msg = (
                f"🔧 <b>[{self.name}] 自動修復</b>\n"
                + "\n".join(f"  → {a}" for a in auto_actions)
            )
            self.send_alert(repair_msg)
            alerts_sent.append(repair_msg)

        duration = time.time() - t0
        summary = diagnosis.get("summary", "調查完成")

        logger.info("[%s] Done in %.1fs — status=%s, %d tool calls, %d turns",
                     self.name, duration, status, len(tool_calls_made), turn + 1)

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            diagnosis=diagnosis,
            tool_calls_made=tool_calls_made,
            actions_taken=actions_taken,
            alerts_sent=alerts_sent,
            turns_used=turn + 1,
            duration_s=duration,
        )

    # ── Scheduled execution wrapper ────────────────────────────────────

    def run_with_logging(self) -> AgentResult:
        """Execute agent with automatic DB logging and failure alerting."""
        run_id = _log_agent_start(self.name)
        try:
            result = self.run()
            _log_agent_finish(run_id, "success", result.summary[:2000])
            return result
        except Exception as e:
            _log_agent_finish(run_id, "failed", error_message=str(e)[:2000])
            self.send_alert(f"❌ <b>{self.name}</b> 排程執行失敗:\n<code>{e}</code>")
            raise


# ── agent_runs DB helpers ──────────────────────────────────────────────

_agent_runs_ensured = False


def _ensure_agent_runs_table():
    global _agent_runs_ensured
    if _agent_runs_ensured:
        return
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS `agent_runs` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `agent_name` VARCHAR(50) NOT NULL,
                        `started_at` DATETIME NOT NULL,
                        `finished_at` DATETIME DEFAULT NULL,
                        `status` ENUM('running', 'success', 'failed') NOT NULL,
                        `output_summary` TEXT DEFAULT NULL,
                        `error_message` TEXT DEFAULT NULL,
                        INDEX `idx_agent_name` (`agent_name`),
                        INDEX `idx_started_at` (`started_at`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            _agent_runs_ensured = True
        finally:
            conn.close()
    except Exception as e:
        logger.error("Failed to ensure agent_runs table: %s", e)


def _log_agent_start(agent_name: str) -> Optional[int]:
    """Insert a 'running' row and return its id."""
    _ensure_agent_runs_table()
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO `agent_runs` (agent_name, started_at, status) "
                    "VALUES (%s, UTC_TIMESTAMP(), 'running')",
                    (agent_name,),
                )
                cur.execute("SELECT LAST_INSERT_ID() AS id")
                return cur.fetchone()["id"]
        finally:
            conn.close()
    except Exception as e:
        logger.error("agent_runs log_start failed: %s", e)
        return None


def _log_agent_finish(run_id: Optional[int], status: str,
                      output_summary: str = "", error_message: str = ""):
    """Update the row with finish time and status."""
    if run_id is None:
        return
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE `agent_runs` SET finished_at=UTC_TIMESTAMP(), "
                    "status=%s, output_summary=%s, error_message=%s WHERE id=%s",
                    (status, output_summary or None, error_message or None, run_id),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.error("agent_runs log_finish failed: %s", e)
