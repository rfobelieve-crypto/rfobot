# Agent Boundary Rules

The MCP agent (`indicator/agent/`) exposes the quant system's signals to
external AI assistants (Claude Desktop, Cursor, ...). It is a READ-ONLY
downstream consumer. It is NOT part of the trading system and must never
influence it.

## The One Invariant
Data flows ONE direction: quant system → MySQL → agent → caller.
The agent reads. It never writes back, never triggers, never mutates
trading state.

## Hard Prohibitions (enforced by test, not discipline)
Files under `indicator/agent/` MUST NOT import, at any depth:
- `indicator.okx.executor` / `reconciler` / `kill_checks` / `runner`
  / `approval` / `client` / `rest` / `ws_private`
- `indicator.inference` / `IndicatorEngine` / `update_cycle` (the hot path)
- `indicator.okx.accounts` (holds decrypted friend credentials)
- Any function that submits orders, amends positions, sets leverage,
  or fires kill switches

## Allowed
- Read-only SELECT via `shared/db.get_db_conn()` on existing tables:
  `tracked_signals`, `v7_okx_positions`, `orderbook_snapshots_1m`,
  `flow_bars_1m`, `indicator_history`
- The agent's own new tables, prefixed `agent_*` (e.g. `agent_request_log`).
  Writes go ONLY to its own namespace, never to quant tables.
- Pure computation (Kelly, ATR maths) on values already read from DB.

## Physical Isolation
The agent is a THIRD component, peer to main-bot and market-data. It shares
only MySQL, and only for reads — mirroring `.claude/rules/architecture.md`
("share DB, don't import each other's modules"). It runs as a separate
process (stdio MCP server); it is never wired into `update_cycle` or any
scheduler.

## Credential / IP Protection
- The agent exposes model OUTPUT (direction, tier, confidence, drivers),
  never model internals (feature definitions, cutoffs, weights).
- The agent never reads `okx_accounts` (encrypted credentials live there).
- Every tool response that could be read as advice carries a
  `not financial advice` disclaimer field.

## Enforcement — `tests/test_agent_boundary.py` (CI-gated)
1. AST-scan every file under `indicator/agent/` for banned imports →
   fail if any executor/reconciler/kill/inference/accounts reference exists.
2. Grep agent SQL strings for INSERT/UPDATE/DELETE against non-`agent_*`
   tables → fail on any write to a quant table.
3. Assert no trading-executor module is pulled into `sys.modules` as a
   side effect of importing the agent package.

Mirrors the AST signature-parity test from mistake.md 2026-06-16 —
boundaries are enforced by machine, not by memory.

## Change Rule
Want a tool that needs data not already in the DB? STOP. Do NOT import a
hot-path module to compute it live. Either (a) the quant system already
persists it — read that table, or (b) it does not belong in the agent.
Adding a compute path that reaches into the trading system is the one
change this file exists to forbid.
