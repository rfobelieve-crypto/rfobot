"""MCP server exposing the rfobot quant system's live signals to any
MCP-capable AI assistant (Claude Desktop, Cursor, ...).

Read-only. Every tool delegates to indicator/agent/queries.py, which
contains only SELECTs against the existing quant tables — this server
imports NOTHING from the trading executor / reconciler / inference hot
path. See .claude/rules/agent-boundary.md; enforced by
tests/test_agent_boundary.py.

Run (stdio transport):
    OKX_AGENT_SEED=1 python -m indicator.agent.server      # demo, no DB
    python -m indicator.agent.server                       # live DB

Claude Desktop config (~/.../claude_desktop_config.json):
    {
      "mcpServers": {
        "rfobot-orderflow": {
          "command": "python",
          "args": ["-m", "indicator.agent.server"],
          "cwd": "/path/to/rfobot",
          "env": {"OKX_AGENT_SEED": "1"}
        }
      }
    }
"""
from __future__ import annotations

import time
from typing import Optional

import anyio
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from indicator.agent import queries

mcp = FastMCP("rfobot-orderflow")


@mcp.tool()
def get_current_signal() -> dict:
    """Latest BTC directional signal from the live V7 dual-XGBoost model.

    Returns direction (UP/DOWN/NEUTRAL), tier (Strong/Moderate/Weak),
    confidence 0-100, current market regime, and the top SHAP driver
    features behind the call. This is proprietary model output not
    available from any public API.
    """
    return queries.latest_signal()


@mcp.tool()
def get_orderflow_snapshot() -> dict:
    """Current BTC order-flow microstructure snapshot.

    Returns mid price, spread, L20 bid/ask depth in USD, and the L20
    order-book imbalance, sourced from a self-built direct-exchange data
    pipeline (not a public aggregator).
    """
    return queries.orderflow_snapshot()


@mcp.tool()
def get_track_record(window_days: Optional[int] = None) -> dict:
    """Verifiable performance of the signal engine and live trades.

    Signal layer: Strong-tier hit rate over tracked signals. Trade layer:
    closed-trade win rates under multiple definitions (gross / net).
    Includes an explicit caveat that signal accuracy is not trading
    profit. Pass window_days to scope the signal-layer stats.
    """
    return queries.track_record(window_days)


@mcp.tool()
def get_risk_frame(entry_price: float, direction: str,
                   atr: Optional[float] = None) -> dict:
    """Risk framing for a hypothetical entry — analysis, never an order.

    Computes a 3xATR stop anchor, the Kelly-optimal fraction, the 2.0x
    hard leverage cap and its volatility-drag rationale. Pure maths on
    the inputs plus the latest recorded ATR; touches no live position and
    submits nothing.
    """
    return queries.risk_frame(entry_price, direction, atr)


@mcp.tool()
def analyze_cancel_flow(minutes: int = 90, t_from: Optional[str] = None,
                        t_to: Optional[str] = None,
                        include_perp: bool = True) -> dict:
    """Forensic analysis of BTC order-book cancellation flow for a window.

    Covers the live right edge (default: last `minutes` minutes) or any
    historical window (t_from/t_to as TPE "YYYY-MM-DD HH:MM", UTC+8).
    Returns the frozen-definition features per minute (cancel shock,
    gross skew, net skew), taker-volume bursts, gate minutes, recorded
    playbook events with outcomes, and a deterministic lean verdict —
    structured so the calling assistant can narrate the五步 read
    (鬧鐘→毛/淨→量→價→持續). Research/eyeball aid over a self-built
    exchange data pipeline; NOT a trading signal (edge unverified until
    the 2026-08-10 pre-registered verdict).
    """
    return queries.cancel_flow_analysis(minutes, t_from, t_to, include_perp)


@mcp.tool()
def log_verdict(direction: str, basis: str = "",
                event_source: Optional[str] = None,
                event_id: Optional[int] = None) -> dict:
    """Log THIS assistant's own prospective market judgement (三方判讀對照).

    Third cohort of the judgment experiment — machine (frozen playbooks)
    vs human (four-button eyeball log) vs LLM (this tool). Call AFTER
    forming a view from the analysis tools: direction UP/DOWN/UNSURE plus
    a short basis (what evidence drove the call). The verdict is stamped
    to the current minute (prospective by construction), scored later on
    60/120m forward mid returns, and can never be edited. Optionally link
    the event card being judged (event_source 'tv' or 'pb' + event_id).
    Research log only — never an order, not financial advice.
    """
    return queries.log_agent_verdict(direction, basis, event_source, event_id)


@mcp.tool()
def get_verdict_stats() -> dict:
    """Prospective hit-rate of this assistant's logged verdicts.

    Lazily backfills matured verdicts (>=121 min old) with 60/120m forward
    mid returns, then returns per-direction counts, hit rates and recent
    entries. The formal three-cohort comparison runs as a repo research
    script once samples accumulate.
    """
    return queries.agent_verdict_stats()


# ── Public, unauthenticated feed for the product website ────────────────
#
# custom_route bypasses the AGENT_MCP_TOKEN path gate by design (see
# FastMCP.custom_route docstring: "will not require authorization ...
# intended to be public"). That's deliberate here — this is the one route
# meant for an anonymous browser/Vercel fetch, not an MCP client.
#
# Scope is intentionally narrower than get_current_signal(): direction /
# tier / confidence / regime only. top_drivers (SHAP feature names) and
# model_version are dropped — those describe HOW the model thinks, one
# step past the "direction + confidence" the site is scoped to show.
_feed_cache: dict = {"data": None, "ts": 0.0}
_FEED_CACHE_TTL_S = 30.0


@mcp.custom_route("/public/signal-feed", methods=["GET"])
async def public_signal_feed(request: Request) -> JSONResponse:
    now = time.monotonic()
    if _feed_cache["data"] is None or now - _feed_cache["ts"] > _FEED_CACHE_TTL_S:
        full = await anyio.to_thread.run_sync(queries.latest_signal)
        _feed_cache["data"] = {
            "signal_time": full.get("signal_time"),
            "direction": full.get("direction"),
            "tier": full.get("tier"),
            "confidence": full.get("confidence"),
            "regime": full.get("regime"),
            "entry_price": full.get("entry_price"),
            "disclaimer": full.get("disclaimer"),
        }
        _feed_cache["ts"] = now
    resp = JSONResponse(_feed_cache["data"])
    resp.headers["Cache-Control"] = "public, max-age=30"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# Same public/unauthenticated deal as above, heavier query (full-table
# aggregates + a balance-snapshot scan for MDD) so it gets a longer cache.
_track_record_cache: dict = {"data": None, "ts": 0.0}
_TRACK_RECORD_CACHE_TTL_S = 120.0


@mcp.custom_route("/public/track-record", methods=["GET"])
async def public_track_record_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if (_track_record_cache["data"] is None
            or now - _track_record_cache["ts"] > _TRACK_RECORD_CACHE_TTL_S):
        data = await anyio.to_thread.run_sync(queries.public_track_record)
        _track_record_cache["data"] = data
        _track_record_cache["ts"] = now
    resp = JSONResponse(_track_record_cache["data"])
    resp.headers["Cache-Control"] = "public, max-age=120"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def main() -> None:
    mcp.run()   # stdio transport by default


if __name__ == "__main__":
    main()
