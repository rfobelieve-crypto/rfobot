# rfobot-orderflow — MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes a **live BTC quantitative trading system's** signals to any
MCP-capable AI assistant (Claude Desktop, Cursor, ...).

The bet: market analysis is shifting from *staring at charts* to *asking
an AI*. This puts proprietary order-flow alpha where that new interface
can reach it — a private data layer for agent-native trading research.

## What it exposes (4 read-only tools)

| Tool | Returns | Why an AI can't get this elsewhere |
|---|---|---|
| `get_current_signal` | V7 dual-XGBoost direction + tier + confidence + SHAP drivers | proprietary model output |
| `get_orderflow_snapshot` | live L20 depth / imbalance / spread | self-built direct-exchange pipeline |
| `get_track_record` | signal-layer hit rate + trade-layer win rates + caveat | private, verifiable performance |
| `get_risk_frame` | 3xATR stop, Kelly fraction, 2.0x cap + vol-drag rationale | risk maths, not just a prediction |

Every response carries a `not financial advice` disclaimer.

## The hard boundary (the point)

This server is a **read-only downstream consumer**. It never touches the
trading system — no order submission, no position changes, no kill
switches, no imports of the executor / reconciler / inference hot path.

That isolation is **machine-enforced**, not a promise:
`tests/test_agent_boundary.py` AST-scans every agent file for banned
imports, greps for any SQL write outside the agent's own namespace, and
asserts importing the agent never drags a trading module into
`sys.modules`. If the boundary is ever crossed, CI fails and it doesn't
ship. See [`.claude/rules/agent-boundary.md`](../../.claude/rules/agent-boundary.md).

## Try it in 60 seconds (no database needed)

Seed mode returns canned demo data, so a reviewer can run it with zero
infrastructure:

```bash
pip install mcp
OKX_AGENT_SEED=1 python -m indicator.agent.server
```

Add to Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rfobot-orderflow": {
      "command": "python",
      "args": ["-m", "indicator.agent.server"],
      "cwd": "/absolute/path/to/rfobot",
      "env": { "OKX_AGENT_SEED": "1" }
    }
  }
}
```

Then ask Claude: *"Use rfobot-orderflow to get the current BTC signal and
frame the risk for a long at spot."* Drop `OKX_AGENT_SEED` (and point
`shared/db` at the live database) to serve real signals.

## Architecture

```
quant system  ──writes──▶  MySQL  ──SELECT only──▶  agent/queries.py
(untouched)                                          agent/server.py (MCP)
                                                          │ stdio
                                                          ▼
                                              Claude Desktop / Cursor / any MCP client
```

Third component, peer to the main bot and market-data services; shares
only MySQL, and only for reads — same contract as the rest of the repo
(`.claude/rules/architecture.md`).
