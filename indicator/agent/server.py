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
            # Liveness passthrough. Without these a consumer cannot tell a
            # quiet market from a dead upstream — both look like "same
            # signal as last poll, HTTP 200". These say nothing about how
            # the model thinks, so they stay inside the public scope.
            "published_utc": full.get("published_utc"),
            "signal_age_hours": full.get("signal_age_hours"),
            "upstream_last_bar": full.get("upstream_last_bar"),
            "upstream_age_minutes": full.get("upstream_age_minutes"),
            "upstream_live": full.get("upstream_live"),
            # 1-bit sign of the latest bar's raw prediction — powers the
            # product side's conviction-decay exit (§0.51 attribution's
            # main positive contributor). Sign only; magnitude withheld.
            "pred_sign": full.get("pred_sign"),
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


# V7 cumulative performance PNG (2026-08-02). Rendered by a subprocess on
# the indicator service (matplotlib, ~10-20s with the terrain panel), so
# the TTL here is doing real work: without it every page view would fan
# out a fresh render. 30 min matches how fast the inputs actually move —
# tracked_signals backfills 4h after a signal fires.
_v7_accum_cache: dict = {"bytes": None, "ts": 0.0}
_V7_ACCUM_CACHE_TTL_S = 1800.0


# Interactive twin of /public/v7-accum — an already-rendered HTML page
# (Lightweight Charts), relayed with the same token the PNG route uses.
# This is what product-site iframes; the PNG stays for image-only
# surfaces. Shorter TTL than the PNG: the HTML is what people actually
# look at, and the render is the same subprocess cost either way.
_v7_accum_i_cache: dict = {"bytes": None, "ts": 0.0}
_V7_ACCUM_I_CACHE_TTL_S = 900.0


@mcp.custom_route("/public/v7-accum-i", methods=["GET"])
async def public_v7_accum_i_route(request: Request) -> Response:
    return await _proxy_html(
        _v7_accum_i_cache, _V7_ACCUM_I_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/research/v7-accum-i", token=INDICATOR_ADMIN_TOKEN)


@mcp.custom_route("/public/v7-accum", methods=["GET"])
async def public_v7_accum_route(request: Request) -> Response:
    return await _proxy_png(
        _v7_accum_cache, _V7_ACCUM_CACHE_TTL_S,
        f"{INDICATOR_BASE_URL}/research/v7-accum", token=INDICATOR_ADMIN_TOKEN)


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
        f"{INDICATOR_BASE_URL}/research/shadow-review?symbol=BTC&hours=2160",
        token=INDICATOR_ADMIN_TOKEN)


_sweep_status_cache: dict = {"data": None, "ts": 0.0}
_SWEEP_STATUS_CACHE_TTL_S = 300.0


_v7_clock_cache: dict = {"data": None, "ts": 0.0}
_V7_CLOCK_CACHE_TTL_S = 600.0


def _v7_clock_cached() -> dict | None:
    """Adoption clock, cached 10 min.  DB row first, origin second.

    2026-08-20: the origin route (/research/v7-clock) shells out to a
    script that needs the LOCAL kline cache — absent from the Railway
    image — so in production it failed every request and silently served
    the JSON committed at build time (the card sat at asof 08-10 /
    trigger 4/60 while the truth was 34/60).  The local hourly train now
    publishes the clock into `v7_veto_clock` (research/v7_veto_publish.py,
    same off-cloud-recorder family as raid_signals_live); reading that row
    is the boundary-compliant path.  The origin fetch remains only as a
    fallback for environments without the table.
    """
    import json as _j
    now = time.monotonic()
    if (_v7_clock_cache["data"] is not None
            and now - _v7_clock_cache["ts"] < _V7_CLOCK_CACHE_TTL_S):
        return _v7_clock_cache["data"]
    try:
        from shared.db import get_db_conn
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM v7_veto_clock WHERE id=1")
                row = cur.fetchone()
        finally:
            conn.close()
        if row:
            data = _j.loads(row["payload"])
            _v7_clock_cache["data"] = data
            _v7_clock_cache["ts"] = now
            return data
    except Exception:  # noqa: BLE001
        pass
    raw = _fetch_origin_png(f"{INDICATOR_BASE_URL}/research/v7-clock",
                            INDICATOR_ADMIN_TOKEN)
    if not raw:
        return None
    try:
        data = _j.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    _v7_clock_cache["data"] = data
    _v7_clock_cache["ts"] = now
    return data


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
    import combo_watchlist as CW  # noqa: PLC0415
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
    # full ledger board (2026-08-02, operator: "很多都只記錄在資料庫沒有
    # 顯示出來...不好管理") — every cohort and every frozen watchlist combo,
    # same clustered-CI arithmetic, plain-language labels included so the
    # site renders 白話 without duplicating definitions.
    cohorts = []
    # D/E 是看著已累積的列註冊的（D 2026-08-01、E 2026-08-02），註冊前那段
    # 是挑選期資料 —— 拿它計分是自我證成。實測（2026-08-07 拆分）：E 看板
    # +0.589 全靠挑選期（真前瞻 n=2 為 −0.206）；D 反而是健康形狀（挑選期
    # +0.013 平、真前瞻 +0.195 CI-low +0.077）。A/B/C 的規則在 log 起點
    # （07-28/29）就凍結，不受影響。
    _dp, _ep = SE.is_variant_d, SE.variant_e_pred(log)
    for key, zh, pred in (
            ("A", "A 全樣本（無濾網·對照組）", lambda r: True),
            ("B", "B 淺穿越（主 gate n/1400）",
             lambda r: str(r.get("variant_b", "")) == "1"),
            ("C", "C ＋收回（1m 縮回價位內）", SE.is_variant_c),
            ("D", "D ＋量能（訂單流組合）",
             lambda r: _dp(r) and (r.get("first_seen_utc") or "") >= "2026-08-01"),
            ("E", "E 三面板盤感（OI↓∧CVD順破∧清算高）",
             lambda r: _ep(r) and (r.get("first_seen_utc") or "") >= "2026-08-02")):
        st = SE.gate_stats(log, pred)
        # core9 並列（2026-08-20）：跟單機器人**只交易 core9**，但這張表原本
        # 不分 universe 全算，把 added20 也算進去 —— 產品端照這些數字選變體，
        # 選到的卻是另一個母體。實測差距足以改變排序：D 在全樣本顯著、
        # 在 core9 不顯著；R∧V 全樣本 CI-low +0.070、core9 −0.113。
        # 不動 gate 的算術（那是凍結的時鐘），只並列出來讓人自己比。
        st9 = SE.gate_stats(log, lambda r, _p=pred: _p(r)
                            and str(r.get("universe", "")) == "core9")
        cohorts.append({"key": key, "label_zh": zh, **st, "core9": st9})
    combo_zh = {
        "R∧V": "放量刺、縮回來", "R∧Q": "縮回＋確認掃止損",
        "R∧V∧Q": "三重確認（歷史最肥）", "R∧快": "五分鐘搶完就跑",
        "R∧快∧Q": "快閃＋止損確認", "R": "有縮回就算（最寬）",
        "PA": "V7 也站這邊", "V∧LIQ": "放量＋清算噴"}
    combos = []
    for name, pred in CW.combo_preds(log).items():
        # forward_only: 組合是看著 07-28→08-02 的列挑出來的，那段是挑選期
        # 資料 —— 拿它給被挑出來的組合打分是自我證成（R∧Q 看板 +0.82，
        # 真前瞻 n=4 CI −0.206）。網站只呈現註冊後的列。
        fwd = CW.forward_only(pred)
        st = SE.gate_stats(log, fwd)
        st9 = SE.gate_stats(log, lambda r, _f=fwd: _f(r)
                            and str(r.get("universe", "")) == "core9")
        combos.append({"key": name, "label_zh": combo_zh.get(name, ""),
                       **st, "core9": st9})
    try:
        clocks = queries.public_research_clocks()
    except Exception:  # noqa: BLE001
        clocks = None
    v7f = None
    try:
        import json as _json
        # Live from the indicator service. Reading the committed file
        # meant the board froze at whatever was in the image at build
        # time; the file is now only the fallback when the origin is
        # unreachable, and it carries its own asof stamp so a stale
        # answer is visibly stale rather than quietly wrong.
        v7f = _v7_clock_cached()
        vp = root / "research" / "results" / "v7_veto_clock.json"
        if v7f is None and vp.exists():
            v7f = _json.loads(vp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        v7f = None
    return {"gate": gate, "recent": recent, "asof_utc": asof,
            "v7_filters": v7f,
            "cohorts": cohorts, "combos": combos, "clocks": clocks,
            "universes": {"default": "all（core9 + added20）",
                          "core9": "跟單機器人實際交易的凍結籃子 —— "
                                   "產品端要看的是這一組"},
            "watchlist_registered": CW.REGISTERED,
            "mode": "shadow", "disclaimer":
            "Forward shadow validation in progress — not a live strategy, "
            "not financial advice."}


_raid_signals_cache: dict = {"data": None, "ts": 0.0}
_RAID_SIGNALS_CACHE_TTL_S = 120.0
RAID_SIGNAL_MAX_AGE_H = 8          # HOLD window; older rows can't be acted on
# Frozen rule constants, mirrored from research/sweep_failure/sweep_core.py
# (DIS / HOLD).  Published in the payload so followers size and exit from
# ONE source of truth instead of hardcoding them client-side.
RAID_STOP_ATR = 3.5
RAID_HOLD_H = 8


def _raid_signals_payload() -> dict:
    """OPEN variant-B raid signals for the JARVIS follow bot (2026-08-19).

    The bridge existed on the JARVIS side (RaidBot -> FLOW_RAID_URL) but
    had no upstream to point at — this is that upstream.  Field names are
    the shadow log's own (symbol / side / entry_px / atr / fill_ts /
    level_kind), so the recorder stays the single source of truth and the
    consumer needs no translation layer.  `side` only became available in
    M2 (2026-08-18); rows without it are dropped rather than guessed —
    a follow bot must never infer direction.

    Reads `raid_signals_live`, written hourly by
    research/raid_signals_publish.py.  It used to read the shadow CSV
    inside this image, which was 8 days stale because the recorder runs on
    the operator machine — the endpoint returned 0 rows forever while
    looking healthy (2026-08-20).  Same quant-persists/agent-selects shape
    as weather_station; the agent computes nothing.

    Recorder-only surface: it reports what the frozen recorder recorded.
    No orders, no gate arithmetic, no account data.
    """
    import time as _t
    from shared.db import get_db_conn
    now = int(_t.time())
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, side, level_kind, fill_ts, fill_utc,"
                " entry_px, atr, stop_px, risk_frac, pierce_atr, universe,"
                " variants, updated_at FROM raid_signals_live"
                " ORDER BY fill_ts DESC")
            rows = cur.fetchall()
    finally:
        conn.close()
    out = []
    published = None
    for r in rows:
        fill_ts = int(r["fill_ts"])
        if now - fill_ts > RAID_SIGNAL_MAX_AGE_H * 3600:
            continue          # publisher may lag; never serve a stale signal
        published = published or r["updated_at"]
        out.append({
            "symbol": r["symbol"], "side": r["side"],
            "level_kind": r["level_kind"],
            "fill_ts": fill_ts, "fill_utc": r["fill_utc"],
            "entry_px": float(r["entry_px"]), "atr": float(r["atr"]),
            "universe": r["universe"],
            "pierce_atr": float(r["pierce_atr"] or 0),
            # Echoed even though the publisher already guarantees them:
            # RaidBot re-checks variant_b/status itself, and a consumer that
            # defensively re-validates should be able to.  Dropping them made
            # every row fail its check while the payload looked correct
            # (2026-08-19, found by diffing the consumer's rules against the
            # payload rather than reading either side's prose).
            "variant_b": "1", "status": "OPEN",
            "variants": [v for v in str(r["variants"]).split(",") if v],
            # Sizing inputs: followers must NOT hardcode the frozen
            # constants.  size_base = (equity x risk_pct) / |entry - stop|
            # keeps per-trade risk fixed, which is the unit the research
            # mean-R is denominated in; a fixed-notional client would carry
            # 2.7x more risk on ADA than on BTC and could never be compared
            # to the shadow ledger.
            "stop_px": float(r["stop_px"]),
            "risk_frac": float(r["risk_frac"]),
            "hold_hours": RAID_HOLD_H,
        })
    return {"list": out, "count": len(out),
            "published_utc": str(published) if published else None,
            "asof_utc": _t.strftime("%Y-%m-%d %H:%M:%S", _t.gmtime(now)),
            "max_age_h": RAID_SIGNAL_MAX_AGE_H,
            "rules": {"stop_atr_mult": RAID_STOP_ATR,
                      "hold_hours": RAID_HOLD_H,
                      "gate_variant": "B",
                      "gate_universe": "core9"},
            "mode": "shadow",
            "disclaimer": "Forward shadow validation in progress — the "
                          "strategy has not passed its gate. Not financial "
                          "advice."}


@mcp.custom_route("/public/raid-signals", methods=["GET"])
async def public_raid_signals_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if (_raid_signals_cache["data"] is None
            or now - _raid_signals_cache["ts"] > _RAID_SIGNALS_CACHE_TTL_S):
        try:
            data = await anyio.to_thread.run_sync(_raid_signals_payload)
        except Exception as e:  # noqa: BLE001 — degrade to 503, never crash
            data = {"error": f"raid signals unavailable: {type(e).__name__}"}
        _raid_signals_cache["data"] = data
        _raid_signals_cache["ts"] = now
    payload = _raid_signals_cache["data"]
    resp = JSONResponse(payload, status_code=503 if "error" in payload else 200)
    resp.headers["Cache-Control"] = "public, max-age=120"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


_raid_pending_cache: dict = {"data": None, "ts": 0.0}
_RAID_PENDING_CACHE_TTL_S = 30.0     # armed levels are time-critical


def _raid_pending_payload() -> dict:
    """Armed-but-unfilled raid levels — the §0.57 fix surface.

    Why this exists next to /public/raid-signals: that feed reports fills
    that ALREADY happened, so a batch consumer acts 65-342 min late and
    enters after the edge has been spent (measured: 0.1328 R/trade, 158%
    of variant B's frozen edge — research/sweep_realizable.py). This feed
    reports levels that are ARMED: the sweep has occurred, the retest has
    not. A consumer watching its own price feed fills AT trigger_px, which
    is the price the frozen backtest assumes.

    Consumption contract: enter when price touches trigger_px in the
    direction implied by `side`, size off `risk_frac`, stop at `stop_px`,
    abandon after `expires_ts`. Variants stop at A/B by construction (C/D
    need 1m flow measured at the fill, which does not exist while the level
    is only armed). Rows vanish once filled or expired — an empty list is a
    normal market state, not an outage; check `asof_utc` for liveness.
    """
    from shared.db import get_db_conn
    import time as _t
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, side, level_kind, sweep_ts, sweep_utc, "
                "trigger_px, stop_px, atr, risk_frac, pierce_atr, variants, "
                "expires_ts, universe, updated_at "
                "FROM raid_pending_levels ORDER BY sweep_ts DESC")
            rows = cur.fetchall()
    finally:
        conn.close()
    now = int(_t.time())
    out, newest = [], None
    for r in rows:
        if int(r["expires_ts"]) <= now:
            continue                      # never serve an expired invitation
        newest = max(newest or r["updated_at"], r["updated_at"])
        out.append({
            "symbol": r["symbol"], "side": r["side"],
            "level_kind": r["level_kind"],
            "sweep_ts": int(r["sweep_ts"]), "sweep_utc": r["sweep_utc"],
            "trigger_px": float(r["trigger_px"]),
            "stop_px": float(r["stop_px"]), "atr": float(r["atr"]),
            "risk_frac": float(r["risk_frac"]),
            "pierce_atr": float(r["pierce_atr"]),
            "variants": [v for v in str(r["variants"]).split(",") if v],
            "expires_ts": int(r["expires_ts"]),
            "expires_in_min": round((int(r["expires_ts"]) - now) / 60),
            "universe": r["universe"],
        })
    return {
        "list": out, "count": len(out),
        "asof_utc": str(newest) if newest else None,
        "rules": {"stop_atr": RAID_STOP_ATR, "hold_h": RAID_HOLD_H,
                  "entry": "touch trigger_px", "variants_available": ["A", "B"]},
        "mode": "shadow",
        "disclaimer": "Forward shadow validation in progress — not a live "
                      "strategy, not financial advice.",
    }


@mcp.custom_route("/public/raid-pending", methods=["GET"])
async def public_raid_pending_route(request: Request) -> JSONResponse:
    now = time.monotonic()
    if (_raid_pending_cache["data"] is None
            or now - _raid_pending_cache["ts"] > _RAID_PENDING_CACHE_TTL_S):
        try:
            data = await anyio.to_thread.run_sync(_raid_pending_payload)
        except Exception as e:  # noqa: BLE001 — degrade, never crash
            data = {"error": f"raid pending unavailable: {type(e).__name__}"}
        _raid_pending_cache["data"] = data
        _raid_pending_cache["ts"] = now
    payload = _raid_pending_cache["data"]
    resp = JSONResponse(payload, status_code=503 if "error" in payload else 200)
    resp.headers["Cache-Control"] = "public, max-age=30"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


_weather_cache: dict = {"data": None, "ts": 0.0}
_WEATHER_CACHE_TTL_S = 300.0


def _weather_payload() -> dict:
    """Read the weather-station snapshot the quant system persists hourly.

    Boundary note (2026-08-17): the agent only SELECTs — the battery is
    computed and written by research/weather_station_publish.py on the
    quant side.  Computing it here would need kline history the DB does
    not hold, which is exactly the reach-into-the-trading-system pattern
    agent-boundary.md forbids."""
    import json as _json
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT payload, updated_at FROM weather_station "
                        "WHERE id = 1")
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return {"error": "weather station snapshot not yet published"}
    data = _json.loads(row["payload"])
    data["asof_utc"] = str(row["updated_at"])
    return data


@mcp.custom_route("/public/weather-station", methods=["GET"])
async def public_weather_station_route(request: Request) -> JSONResponse:
    """Crowd-strategy weather station for the site dashboard — which
    popular-strategy crowds the market is feeding/starving, with each
    gauge's evidence tier.  States and ratios only; no sizes, no dollars,
    no model internals."""
    now = time.monotonic()
    if (_weather_cache["data"] is None
            or now - _weather_cache["ts"] > _WEATHER_CACHE_TTL_S):
        try:
            data = await anyio.to_thread.run_sync(_weather_payload)
        except Exception as e:  # noqa: BLE001 — degrade to 503, never crash
            data = {"error": f"weather station unavailable: {type(e).__name__}"}
        _weather_cache["data"] = data
        _weather_cache["ts"] = now
    payload = _weather_cache["data"]
    resp = JSONResponse(payload, status_code=503 if "error" in payload else 200)
    resp.headers["Cache-Control"] = "public, max-age=300"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


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


_live_status_cache: dict = {"data": None, "ts": 0.0}
_LIVE_STATUS_CACHE_TTL_S = 60.0


@mcp.custom_route("/public/live-status", methods=["GET"])
async def public_live_status_route(request: Request) -> JSONResponse:
    """V7/OKX execution surface for the site dashboard — percentages,
    directions and timing only (queries.public_live_status keeps sizes and
    dollar equity out of the public layer)."""
    now = time.monotonic()
    if (_live_status_cache["data"] is None
            or now - _live_status_cache["ts"] > _LIVE_STATUS_CACHE_TTL_S):
        try:
            data = await anyio.to_thread.run_sync(queries.public_live_status)
        except Exception as e:  # noqa: BLE001
            data = {"error": f"live status unavailable: {type(e).__name__}"}
        _live_status_cache["data"] = data
        _live_status_cache["ts"] = now
    payload = _live_status_cache["data"]
    resp = JSONResponse(payload, status_code=503 if "error" in payload else 200)
    resp.headers["Cache-Control"] = "public, max-age=60"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def main() -> None:
    mcp.run()   # stdio transport by default


if __name__ == "__main__":
    main()
