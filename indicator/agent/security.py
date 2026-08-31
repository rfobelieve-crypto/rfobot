# -*- coding: utf-8 -*-
"""Public-surface security for the agent service (TODO §0.83, 2026-08-31).

Threat model (user-supplied, severity order): (1) signal tampering,
(2) bulk signal scraping, (4) account takeover / brute force. This module
implements the flow_system half:

  * rate_gate(request)  — per-IP sliding-window rate limit on every
    /public/* route. Default 120 req/min per IP; auth-shaped POST routes
    (login/register/waitlist) get a much tighter 10/min because they are
    the credential-stuffing surface. 429 with Retry-After on breach.
    In-memory by design: this service is a single Railway process, and a
    limiter that needs the DB would let a scraper load the DB — the thing
    it exists to prevent.

  * sign(payload) / attach_signature(resp, payload) — HMAC-SHA256 over
    canonical JSON with SIGNAL_SIGNING_KEY, emitted as an
    `X-Signal-Signature: v1=<hex>` response header on the ACTIONABLE
    feeds (signal-feed / raid-signals / raid-pending / raid-outcomes).
    A consumer that pins the shared key can detect any in-path tampering
    of the signal body. Key unset → header simply absent (fail-open on
    integrity *metadata*: an unsigned feed must not break existing
    consumers; the product side alerts when the header disappears once
    it has seen it).

Boundary notes: stdlib only, no DB writes, no trading imports — stays
inside agent-boundary.md. tests/test_agent_security.py holds the unit
tests plus the structural guard ("every /public route calls rate_gate").
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time

try:                                    # starlette ships with the deployed
    from starlette.responses import JSONResponse   # image (via mcp); local
except ImportError:                     # research envs may lack it — the
    JSONResponse = None                 # limiter logic stays testable.

# ── rate limiting ─────────────────────────────────────────────────────────

WINDOW_S = 60.0
DEFAULT_LIMIT = int(os.environ.get("AGENT_PUBLIC_RATE_LIMIT", "120"))
STRICT_LIMIT = int(os.environ.get("AGENT_AUTH_RATE_LIMIT", "10"))
STRICT_PATHS = ("/public/login", "/public/register", "/public/waitlist")
_MAX_IPS = 10000          # memory cap; beyond this the table is cleared

_lock = threading.Lock()
_hits: dict = {}          # (ip, bucket) -> [timestamps]
_blocked_count = 0


def client_ip(request) -> str:
    """First hop of X-Forwarded-For (Railway edge sets it), else peer."""
    try:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
        return (request.client.host or "?")[:64]
    except Exception:
        return "?"


def _limit_for(path: str) -> int:
    return STRICT_LIMIT if path in STRICT_PATHS else DEFAULT_LIMIT


def check(ip: str, path: str, now: float | None = None) -> bool:
    """True = allowed. Sliding 60s window per (ip, strict|default bucket)."""
    global _blocked_count
    now = time.monotonic() if now is None else now
    strict = path in STRICT_PATHS
    key = (ip, "strict" if strict else "default")
    limit = _limit_for(path)
    with _lock:
        if len(_hits) > _MAX_IPS:
            _hits.clear()
        q = _hits.setdefault(key, [])
        cutoff = now - WINDOW_S
        while q and q[0] < cutoff:
            q.pop(0)
        if len(q) >= limit:
            _blocked_count += 1
            if _blocked_count % 50 == 1:
                print(f"[security] rate-limited {ip} on {path} "
                      f"(blocked so far: {_blocked_count})", flush=True)
            return False
        q.append(now)
        return True


def rate_gate(request):
    """None = pass. JSONResponse(429) = blocked (return it as-is)."""
    if JSONResponse is None:
        return None
    try:
        path = request.url.path
        if not check(client_ip(request), path):
            resp = JSONResponse(
                {"error": "rate limited", "retry_after_s": int(WINDOW_S)},
                status_code=429)
            resp.headers["Retry-After"] = str(int(WINDOW_S))
            return resp
    except Exception:
        # the guard must never take the service down
        return None
    return None


# ── signal signing ────────────────────────────────────────────────────────

def _key() -> bytes | None:
    k = os.environ.get("SIGNAL_SIGNING_KEY", "").strip()
    return k.encode("utf-8") if len(k) >= 16 else None


def sign(payload) -> str | None:
    """HMAC-SHA256 over canonical JSON. None when no key is configured."""
    k = _key()
    if k is None:
        return None
    try:
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, default=str)
        return hmac.new(k, canon.encode("utf-8"), hashlib.sha256).hexdigest()
    except Exception:
        return None


def attach_signature(resp, payload) -> None:
    """Add X-Signal-Signature to an already-built response (no-op keyless)."""
    s = sign(payload)
    if s:
        resp.headers["X-Signal-Signature"] = f"v1={s}"
        # the header must survive CORS for a browser-side verifier
        resp.headers["Access-Control-Expose-Headers"] = "X-Signal-Signature"
