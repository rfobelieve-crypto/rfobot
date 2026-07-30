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

import os

import anyio
import requests
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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


# Waitlist signup — the one WRITE this service does, and only into its
# own agent_* namespace (see queries.submit_waitlist / agent-boundary.md).
# Called server-to-server from the product site's Next.js API route, not
# directly from a browser, so CORS is moot here — but the header's added
# anyway for parity with the GET routes above.
@mcp.custom_route("/public/waitlist", methods=["POST"])
async def public_waitlist_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid body"}, status_code=400)
    email = body.get("email") if isinstance(body, dict) else None
    note = body.get("note") if isinstance(body, dict) else None
    result = await anyio.to_thread.run_sync(
        queries.submit_waitlist, email, note or "", "product-site")
    status = 200 if result.get("ok") else 400
    resp = JSONResponse(result, status_code=status)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# Signal history — the page that consumes this sits behind the product
# site's own login (see /public/register + /public/login below), but
# this endpoint itself is still a plain public GET (same reasoning as the
# other public_* routes: the login is a UX gate in Next.js, not a
# security boundary, and the payload carries the same no-model-internals
# discipline as everything else here).
_history_cache: dict = {"data": None, "ts": 0.0}
_HISTORY_CACHE_TTL_S = 60.0


@mcp.custom_route("/public/signal-history", methods=["GET"])
async def public_signal_history_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if _history_cache["data"] is None or now - _history_cache["ts"] > _HISTORY_CACHE_TTL_S:
        data = await anyio.to_thread.run_sync(queries.public_signal_history, 50)
        _history_cache["data"] = data
        _history_cache["ts"] = now
    resp = JSONResponse(_history_cache["data"])
    resp.headers["Cache-Control"] = "public, max-age=60"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# Self-hosted accounts — register + login. Same WRITE discipline as the
# waitlist route: only agent_user_accounts (agent_* namespace), password
# hashed server-side (see queries.py), never returned or logged.
@mcp.custom_route("/public/register", methods=["POST"])
async def public_register_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid body"}, status_code=400)
    email = body.get("email") if isinstance(body, dict) else None
    password = body.get("password") if isinstance(body, dict) else None
    result = await anyio.to_thread.run_sync(queries.register_user, email, password)
    status = 200 if result.get("ok") else 400
    resp = JSONResponse(result, status_code=status)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@mcp.custom_route("/public/login", methods=["POST"])
async def public_login_route(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid body"}, status_code=400)
    email = body.get("email") if isinstance(body, dict) else None
    password = body.get("password") if isinstance(body, dict) else None
    result = await anyio.to_thread.run_sync(queries.verify_user, email, password)
    status = 200 if result.get("ok") else 401
    resp = JSONResponse(result, status_code=status)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ── Public chart images — pure HTTP relay, never a render ───────────────
#
# This service does not import chart_renderer.py or plot_cancel_flow.py.
# Both PNGs are already computed on the indicator service's own hourly/
# on-demand cycle; these routes just fetch and re-serve those bytes. That
# keeps the change entirely outside agent-boundary.md's import allow-list
# (an HTTP GET is not a Python import) and off tests/test_agent_boundary.py's
# AST scan — see .claude/rules/agent-boundary.md "share DB, don't import
# code" and its Change Rule.
INDICATOR_BASE_URL = os.environ.get(
    "INDICATOR_BASE_URL", "https://enchanting-emotion-production-4b4d.up.railway.app")
INDICATOR_ADMIN_TOKEN = os.environ.get("INDICATOR_ADMIN_TOKEN", "")


def _fetch_origin_png(url: str, token: str = "") -> bytes | None:
    """Sync HTTP GET, run off the event loop via anyio.to_thread. `token` is
    positional-or-keyword, not keyword-only — anyio.to_thread.run_sync only
    forwards *args, not **kwargs, so the call site below passes it
    positionally. Never raises — a failed/timed-out origin fetch degrades to
    None, which the caller turns into "serve stale cache, else empty" (see
    _proxy_png), mirroring product-site's own "null on failure, never
    throw" rule for every other /public/* consumer (lib/signalFeed.ts and
    siblings)."""
    try:
        headers = {"X-Admin-Token": token} if token else {}
        r = requests.get(url, headers=headers, timeout=95)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


async def _proxy_png(cache: dict, ttl_s: float, url: str, *, token: str = "") -> Response:
    now = time.monotonic()
    if cache["bytes"] is None or now - cache["ts"] > ttl_s:
        fetched = await anyio.to_thread.run_sync(_fetch_origin_png, url, token)
        if fetched is not None:
            cache["bytes"] = fetched
            cache["ts"] = now
    if cache["bytes"] is None:
        resp = Response(content=b"", status_code=503)
    else:
        resp = Response(content=cache["bytes"], media_type="image/png")
        resp.headers["Cache-Control"] = f"public, max-age={int(ttl_s)}"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# Same shape as _proxy_png, `text/html` instead of `image/png` — reuses
# _fetch_origin_png as-is (it's content-type-agnostic, just returns raw
# `.content` bytes) for the two interactive Lightweight-Charts pages
# (2026-07-24, "make every chart interactive" — see /public/live-chart and
# /public/cancel-flow-chart-i below). Framed with X-Frame-Options: none so
# product-site can iframe it — the origin service itself sets no
# frame-blocking header, but being explicit here means this route stays
# iframe-able even if the origin's own headers ever change.
async def _proxy_html(cache: dict, ttl_s: float, url: str, *, token: str = "") -> Response:
    now = time.monotonic()
    if cache["bytes"] is None or now - cache["ts"] > ttl_s:
        fetched = await anyio.to_thread.run_sync(_fetch_origin_png, url, token)
        if fetched is not None:
            cache["bytes"] = fetched
            cache["ts"] = now
    if cache["bytes"] is None:
        resp = Response(
            content=b"<h3>Chart temporarily unavailable</h3>",
            status_code=503, media_type="text/html")
    else:
        resp = Response(content=cache["bytes"], media_type="text/html")
        resp.headers["Cache-Control"] = f"public, max-age={int(ttl_s)}"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# V7 chart re-serves the indicator service's in-memory PNG (dark variant,
# see indicator/chart_renderer.py `dark=` param) — cheap on the origin, so
# the cache here is just normal freshness control, not rate-limiting.
_v7_chart_cache: dict = {"bytes": None, "ts": 0.0}
_V7_CHART_CACHE_TTL_S = 300.0  # matches the ~hourly bar cadence with margin


@mcp.custom_route("/public/chart", methods=["GET"])
async def public_chart_route(request: Request) -> Response:
    return await _proxy_png(
        _v7_chart_cache, _V7_CHART_CACHE_TTL_S, f"{INDICATOR_BASE_URL}/chart-dark")


# Cancel-flow chart's origin (/research/cancel-flow) RE-RENDERS from scratch
# on every hit — a subprocess call, up to 90s (indicator/app.py). The cache
# here is load-bearing, not cosmetic: without it, a public route would let
# anyone repeatedly trigger 90s subprocess renders on the indicator service.
# Do not lower this without also changing the origin route to cache itself.
_cancel_chart_cache: dict = {"bytes": None, "ts": 0.0}
_CANCEL_CHART_CACHE_TTL_S = 120.0


@mcp.custom_route("/public/cancel-flow-chart", methods=["GET"])
async def public_cancel_flow_chart_route(request: Request) -> Response:
    return await _proxy_png(
        _cancel_chart_cache, _CANCEL_CHART_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/research/cancel-flow", token=INDICATOR_ADMIN_TOKEN)


# Per-coin cancel-flow KPI stats (2026-07-24) — direct SELECT over the
# whitelisted depth_deltas_1m table (queries.public_cancel_flow_stats),
# same TTL-cache-then-serve pattern as the other /public/* JSON routes
# above (signal-feed / track-record / signal-history), not the PNG-proxy
# pattern used by the chart routes.
_cancel_stats_cache: dict = {"data": None, "ts": 0.0}
_CANCEL_STATS_CACHE_TTL_S = 60.0


@mcp.custom_route("/public/cancel-flow-stats", methods=["GET"])
async def public_cancel_flow_stats_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if (_cancel_stats_cache["data"] is None
            or now - _cancel_stats_cache["ts"] > _CANCEL_STATS_CACHE_TTL_S):
        data = await anyio.to_thread.run_sync(queries.public_cancel_flow_stats, 120)
        _cancel_stats_cache["data"] = data
        _cancel_stats_cache["ts"] = now
    resp = JSONResponse(_cancel_stats_cache["data"])
    resp.headers["Cache-Control"] = "public, max-age=60"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ── Interactive (Lightweight-Charts) chart pages, 2026-07-24 ────────────
# "Make every chart on the site interactive" — proxies indicator/app.py's
# already-built interactive HTML pages (same lightweight-charts library,
# same zoom/pan/crosshair sync, same live OKX markers) instead of the
# static PNGs above. product-site iframes these directly.

# /live-chart is an open route (not admin-guarded) that renders synchronously
# from in-memory state — cheap, no subprocess — so a short TTL is fine.
_v7_live_chart_cache: dict = {"bytes": None, "ts": 0.0}
_V7_LIVE_CHART_CACHE_TTL_S = 60.0


@mcp.custom_route("/public/live-chart", methods=["GET"])
async def public_live_chart_route(request: Request) -> Response:
    return await _proxy_html(
        _v7_live_chart_cache, _V7_LIVE_CHART_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/live-chart")


# /research/cancel-flow-i is the interactive twin of /research/cancel-flow:
# same subprocess re-render cost (up to 100s), same admin-token guard, same
# reasoning as _cancel_chart_cache above for why the cache here is
# load-bearing, not cosmetic.
_cancel_chart_i_cache: dict = {"bytes": None, "ts": 0.0}
_CANCEL_CHART_I_CACHE_TTL_S = 120.0


@mcp.custom_route("/public/cancel-flow-chart-i", methods=["GET"])
async def public_cancel_flow_chart_i_route(request: Request) -> Response:
    return await _proxy_html(
        _cancel_chart_i_cache, _CANCEL_CHART_I_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/research/cancel-flow-i?hours=48",
        token=INDICATOR_ADMIN_TOKEN)


# ── Strategy #2 (sweep-failure) shadow surfaces, 2026-07-30 ─────────────
# product-site's multi-strategy upgrade. Same patterns as above:
#   liquidity-map  — HTML proxy of the indicator's shadow-review page. The
#                    origin re-renders via subprocess (up to 110s), so the
#                    cache here is load-bearing exactly like the cancel-flow
#                    routes. Symbol is pinned to BTC on purpose: a symbol
#                    passthrough would give the public 29 distinct cache
#                    keys = 29 ways to trigger origin subprocess renders.
#   sweep-status   — JSON gate progress read from the shadow CSV shipped in
#                    this image (research/results/sweep_shadow_log.csv,
#                    fresh as of the last deploy — the hourly recorder runs
#                    on the operator machine, so `asof` is surfaced for
#                    honesty). Lazy import of the research module: pure
#                    computation + CSV read, no banned trading-path imports
#                    (tests/test_agent_boundary.py stays the referee).
_liq_map_cache: dict = {"bytes": None, "ts": 0.0}
_LIQ_MAP_CACHE_TTL_S = 300.0


@mcp.custom_route("/public/liquidity-map", methods=["GET"])
async def public_liquidity_map_route(request: Request) -> Response:
    return await _proxy_html(
        _liq_map_cache, _LIQ_MAP_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/research/shadow-review?symbol=BTC&hours=72",
        token=INDICATOR_ADMIN_TOKEN)


_sweep_status_cache: dict = {"data": None, "ts": 0.0}
_SWEEP_STATUS_CACHE_TTL_S = 300.0


def _sweep_status_payload() -> dict:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sf = root / "research" / "sweep_failure"
    if not sf.exists():
        return {"error": "research/ not present in this image"}
    for p in (str(root), str(root / "research"), str(sf)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import shadow_engine as SE  # noqa: PLC0415  (lazy: research is optional)
    log = SE.read_log()
    gate = SE.gate_stats(log)
    closed = sorted(
        (r for r in log.values() if r["status"] == "CLOSED" and r["net_r"] != ""),
        key=lambda r: int(r["fill_ts"]), reverse=True)
    recent = [{
        "symbol": r["symbol"], "kind": r.get("level_kind", "swing"),
        "fill_utc": r["fill_utc"], "variant_b": r.get("variant_b") == "1",
        "net_r": round(float(r["net_r"]), 3),
    } for r in closed[:6]]
    asof = max((r.get("first_seen_utc") or "" for r in log.values()), default="")
    return {"gate": gate, "recent": recent, "asof_utc": asof,
            "mode": "shadow", "disclaimer":
            "Forward shadow validation in progress — not a live strategy, "
            "not financial advice."}


@mcp.custom_route("/public/sweep-status", methods=["GET"])
async def public_sweep_status_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if (_sweep_status_cache["data"] is None
            or now - _sweep_status_cache["ts"] > _SWEEP_STATUS_CACHE_TTL_S):
        try:
            data = await anyio.to_thread.run_sync(_sweep_status_payload)
        except Exception as e:  # noqa: BLE001 — degrade to 503, never crash
            data = {"error": f"sweep status unavailable: {type(e).__name__}"}
        _sweep_status_cache["data"] = data
        _sweep_status_cache["ts"] = now
    payload = _sweep_status_cache["data"]
    resp = JSONResponse(payload, status_code=503 if "error" in payload else 200)
    resp.headers["Cache-Control"] = "public, max-age=300"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def main() -> None:
    mcp.run()   # stdio transport by default


if __name__ == "__main__":
    main()
