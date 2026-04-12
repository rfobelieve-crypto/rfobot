"""
Agent Coordinator — orchestrates all AI maintenance agents.

Usage:
  python -m indicator.agents.coordinator                        # run all agents
  python -m indicator.agents.coordinator --agent DataCollector  # run one agent
  python -m indicator.agents.coordinator --context-only         # collect only, no Claude
  python -m indicator.agents.coordinator --report               # Telegram-formatted output
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta

from indicator.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))

# ── Agent registry ──────────────────────────────────────────────────────

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def _load_registry():
    """Lazy-load all agent classes."""
    if AGENT_REGISTRY:
        return

    from indicator.agents.data_collector import DataCollectorAgent
    from indicator.agents.feature_guard import FeatureGuardAgent
    from indicator.agents.model_eval import ModelEvalAgent
    from indicator.agents.signal_tracker import SignalTrackerAgent
    from indicator.agents.infra import InfraAgent

    AGENT_REGISTRY["DataCollector"] = DataCollectorAgent
    AGENT_REGISTRY["FeatureGuard"] = FeatureGuardAgent
    AGENT_REGISTRY["ModelEval"] = ModelEvalAgent
    AGENT_REGISTRY["SignalTracker"] = SignalTrackerAgent
    AGENT_REGISTRY["Infra"] = InfraAgent


# ── Coordinator ─────────────────────────────────────────────────────────

class AgentCoordinator:
    """Runs agents and lets Claude see the aggregate picture."""

    def __init__(self, agent_names: list[str] | None = None):
        _load_registry()
        if agent_names:
            self.agents = {
                name: AGENT_REGISTRY[name]()
                for name in agent_names
                if name in AGENT_REGISTRY
            }
        else:
            self.agents = {name: cls() for name, cls in AGENT_REGISTRY.items()}

    def run_all(self) -> list[AgentResult]:
        """Run all registered agents sequentially."""
        results = []
        t0 = time.time()

        logger.info("Coordinator: running %d agents: %s",
                     len(self.agents), list(self.agents.keys()))

        for name, agent in self.agents.items():
            try:
                result = agent.run()
                results.append(result)
                logger.info("  %s: %s (%.1fs)", name, result.status, result.duration_s)
            except Exception as e:
                logger.exception("Agent %s crashed", name)
                results.append(AgentResult(
                    agent_name=name,
                    status="error",
                    summary=f"Agent crashed: {e}",
                ))

        total_time = time.time() - t0
        statuses = [r.status for r in results]
        overall = "critical" if "critical" in statuses else \
                  "degraded" if "degraded" in statuses else \
                  "error" if "error" in statuses else "healthy"

        logger.info("Coordinator: done in %.1fs — overall=%s", total_time, overall)
        return results

    def collect_all_context(self) -> dict:
        """Collect context from all agents without calling Claude."""
        contexts = {}
        for name, agent in self.agents.items():
            try:
                t0 = time.time()
                contexts[name] = agent.collect_context()
                contexts[name]["_collect_time_s"] = round(time.time() - t0, 2)
            except Exception as e:
                contexts[name] = {"error": str(e)}
        return contexts

    def format_report(self, results: list[AgentResult]) -> str:
        """Format results as Telegram-ready HTML report."""
        now = datetime.now(TZ_TPE).strftime("%m/%d %H:%M")

        status_icons = {
            "healthy": "\U0001f7e2",
            "degraded": "\U0001f7e1",
            "critical": "\U0001f534",
            "error": "\u26ab",
        }

        lines = [f"<b>\U0001f916 System Health Report</b> | {now}\n"]

        for r in results:
            icon = status_icons.get(r.status, "\u26aa")
            lines.append(f"{icon} <b>{r.agent_name}</b>: {r.summary}")

            # Include Claude's telegram_report if available
            tg = r.diagnosis.get("telegram_report", "")
            if tg and r.status in ("degraded", "critical"):
                lines.append(f"  {tg}")

        # Overall status
        statuses = [r.status for r in results]
        if "critical" in statuses:
            lines.append(f"\n\u26a0\ufe0f <b>Action required</b>")
        elif all(s == "healthy" for s in statuses):
            lines.append(f"\n\u2705 All systems healthy")

        # Timing
        total = sum(r.duration_s for r in results)
        lines.append(f"\n<i>Total scan: {total:.0f}s</i>")

        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run AI maintenance agents")
    parser.add_argument("--agent", type=str, help="Run specific agent by name")
    parser.add_argument("--context-only", action="store_true",
                        help="Collect context only, no Claude call")
    parser.add_argument("--report", action="store_true",
                        help="Print Telegram-formatted report")
    args = parser.parse_args()

    _load_registry()

    if args.agent and args.agent not in AGENT_REGISTRY:
        print(f"Unknown agent: {args.agent}")
        print(f"Available: {', '.join(AGENT_REGISTRY.keys())}")
        return

    agent_names = [args.agent] if args.agent else None
    coordinator = AgentCoordinator(agent_names=agent_names)

    if args.context_only:
        contexts = coordinator.collect_all_context()
        print(json.dumps(contexts, indent=2, default=str, ensure_ascii=False))
        return

    results = coordinator.run_all()

    if args.report:
        print(coordinator.format_report(results))
    else:
        for r in results:
            print(f"\n{'='*60}")
            print(f"Agent: {r.agent_name} | Status: {r.status} | {r.duration_s:.1f}s")
            print(f"Summary: {r.summary}")
            if r.actions_taken:
                for a in r.actions_taken:
                    print(f"  -> {a}")
            # Print Claude's findings
            findings = r.diagnosis.get("findings", [])
            for f in findings:
                sev = f.get("severity", "info")
                icon = {"critical": "\U0001f534", "warning": "\U0001f7e1", "info": "\U0001f535"}.get(sev, "")
                print(f"  {icon} [{sev}] {f.get('title', '')}: {f.get('detail', '')}")
            print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
