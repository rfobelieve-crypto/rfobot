"""
Base AI Agent — the brain framework for all domain agents.

Philosophy: Claude IS the maintenance engineer. The code just gives it
eyes (context collection) and hands (action execution). Claude decides
what's wrong, why, and what to do about it.

Each agent:
  1. collect_context()  → raw system state (numbers, not booleans)
  2. analyze()          → Claude reads the raw data, thinks like a SRE
  3. act()              → Claude's recommended actions get executed
  4. run()              → full cycle: collect → analyze → act → alert
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

# ── Config ───────────────────────────────────────────────────────────────
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")

# Telegram for agent alerts (reuse indicator bot or separate)
AGENT_BOT_TOKEN = os.environ.get("AGENT_BOT_TOKEN", "")
AGENT_CHAT_ID = os.environ.get("AGENT_CHAT_ID", "")

# Response schema that every agent must follow
RESPONSE_SCHEMA = """\
Respond with ONLY this JSON (no markdown fences, no extra text):
{
  "status": "healthy | degraded | critical",
  "summary": "one-line human-readable summary in Traditional Chinese",
  "findings": [
    {
      "severity": "critical | warning | info",
      "title": "short title",
      "detail": "what you observed, why it matters, what could happen if ignored",
      "action": "exactly what should be done — be specific (retry X, check Y, restart Z)",
      "auto_action": "ALERT | RETRY_ENDPOINT | LOG | NONE"
    }
  ],
  "telegram_report": "2-5 line HTML summary suitable for Telegram push (use <b> for bold, keep it concise)",
  "next_check_minutes": 60
}"""


@dataclass
class AgentResult:
    """Structured output from an agent run."""
    agent_name: str
    status: str
    summary: str
    diagnosis: dict = field(default_factory=dict)
    raw_response: str = ""
    actions_taken: list[str] = field(default_factory=list)
    alerts_sent: list[str] = field(default_factory=list)
    context_snapshot: dict = field(default_factory=dict)
    timestamp: str = ""
    duration_s: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(TZ_TPE).strftime("%Y-%m-%d %H:%M:%S")


class BaseAgent(ABC):
    """
    Abstract base for all AI maintenance agents.

    Claude is the brain — subclasses just define:
      - What data to show Claude (collect_context)
      - What Claude's role is (system_prompt)
      - What actions are available (available_actions + execute_action)
    """

    def __init__(self):
        self._client = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier, e.g. 'DataCollector'."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt defining this agent's expertise and domain knowledge.
        This is the agent's "brain" — it should contain deep operational
        knowledge, not just rules.
        """

    @abstractmethod
    def collect_context(self) -> dict:
        """
        Gather RAW system state for Claude to analyze.

        Feed raw numbers, not pre-digested booleans. Let Claude
        interpret what's normal vs abnormal. Include enough context
        for Claude to reason about root causes and correlations.
        """

    def get_available_actions(self) -> list[dict]:
        """
        Define what actions this agent can take.
        Override in subclass to add domain-specific actions.
        Returns list of {"name": ..., "description": ...}
        """
        return [
            {"name": "ALERT", "description": "Send Telegram alert to operator"},
            {"name": "LOG", "description": "Log finding for record-keeping"},
            {"name": "NONE", "description": "No action needed, just noting"},
        ]

    def execute_action(self, action_name: str, finding: dict, context: dict) -> str:
        """
        Execute an action recommended by Claude.
        Override in subclass to add domain-specific action handlers.
        Returns description of what was done.
        """
        if action_name == "ALERT":
            msg = finding.get("detail", finding.get("title", "Unknown issue"))
            self.send_alert(
                f"🤖 <b>[{self.name}]</b>\n"
                f"<b>{finding.get('title', '')}</b>\n"
                f"{msg}"
            )
            return f"Alert sent: {finding.get('title', '')}"
        elif action_name == "LOG":
            logger.info("[%s] %s: %s", self.name, finding.get("title"), finding.get("detail"))
            return f"Logged: {finding.get('title', '')}"
        return f"No handler for action: {action_name}"

    # ── Claude integration ──────────────────────────────────────────────

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = AGENT_API_KEY
            if not api_key:
                raise RuntimeError(
                    "AGENT_API_KEY not set. "
                    "Set it in Railway env vars or local .env to enable AI agents."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def analyze(self, context: dict) -> tuple[dict, str]:
        """
        Send collected context to Claude for analysis.

        Returns (parsed_diagnosis, raw_response_text).
        """
        client = self._get_client()

        now = datetime.now(TZ_TPE)
        actions_desc = "\n".join(
            f"  - {a['name']}: {a['description']}"
            for a in self.get_available_actions()
        )

        user_message = (
            f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+8 "
            f"(weekday={now.strftime('%A')})\n\n"
            f"## Raw System Context\n\n"
            f"{json.dumps(context, indent=2, default=str, ensure_ascii=False)}\n\n"
            f"## Available Actions\n{actions_desc}\n\n"
            f"Analyze the above data. Think step by step:\n"
            f"1. What is the current state of each component?\n"
            f"2. Are there any anomalies, correlations, or emerging patterns?\n"
            f"3. What is the root cause if something is wrong?\n"
            f"4. What specific action should be taken?\n\n"
            f"{RESPONSE_SCHEMA}"
        )

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text
            parsed = self._parse_response(raw)
            return parsed, raw
        except Exception as e:
            logger.error("[%s] Claude API call failed: %s", self.name, e)
            fallback = {
                "status": "error",
                "summary": f"Claude API 呼叫失敗: {e}",
                "findings": [],
                "telegram_report": f"⚫ [{self.name}] Claude API 無法連線",
                "next_check_minutes": 15,
            }
            return fallback, str(e)

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from Claude's response, handling markdown fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("[%s] Could not parse response JSON: %s", self.name, e)
            return {
                "status": "error",
                "summary": f"Response parse failed: {raw[:200]}",
                "findings": [],
                "telegram_report": "",
                "next_check_minutes": 60,
            }

    # ── Telegram ────────────────────────────────────────────────────────

    def send_alert(self, message: str):
        """Send alert via Telegram."""
        bot_token = AGENT_BOT_TOKEN or os.environ.get("INDICATOR_BOT_TOKEN", "")
        chat_id = AGENT_CHAT_ID or os.environ.get("INDICATOR_CHAT_ID", "")

        if not bot_token or not chat_id:
            logger.warning("[%s] Telegram not configured: %s",
                           self.name, message[:100])
            return

        import requests
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=15,
            )
        except Exception as e:
            logger.error("[%s] Alert send failed: %s", self.name, e)

    # ── Main run cycle ──────────────────────────────────────────────────

    def run(self) -> AgentResult:
        """Full agent cycle: collect → analyze → act → alert."""
        t0 = time.time()
        logger.info("[%s] Starting run...", self.name)

        # 1. Collect raw context
        try:
            context = self.collect_context()
        except Exception as e:
            logger.exception("[%s] collect_context failed", self.name)
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary=f"Context collection failed: {e}",
                duration_s=time.time() - t0,
            )

        # 2. Claude analyzes
        diagnosis, raw = self.analyze(context)
        status = diagnosis.get("status", "error")
        summary = diagnosis.get("summary", "No summary")

        # 3. Execute Claude's recommended actions
        actions_taken = []
        for finding in diagnosis.get("findings", []):
            auto_action = finding.get("auto_action", "NONE")
            if auto_action and auto_action != "NONE":
                try:
                    result = self.execute_action(auto_action, finding, context)
                    actions_taken.append(result)
                except Exception as e:
                    logger.exception("[%s] Action %s failed", self.name, auto_action)
                    actions_taken.append(f"FAILED {auto_action}: {e}")

        # 4. Send Telegram report if degraded/critical
        alerts_sent = []
        tg_report = diagnosis.get("telegram_report", "")
        if tg_report and status in ("degraded", "critical"):
            icon = "🔴" if status == "critical" else "🟡"
            full_msg = f"{icon} <b>[{self.name}]</b>\n{tg_report}"
            self.send_alert(full_msg)
            alerts_sent.append(full_msg)

        duration = time.time() - t0
        logger.info("[%s] Done in %.1fs — status=%s: %s",
                     self.name, duration, status, summary)

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            diagnosis=diagnosis,
            raw_response=raw,
            actions_taken=actions_taken,
            alerts_sent=alerts_sent,
            context_snapshot=context,
            duration_s=duration,
        )
