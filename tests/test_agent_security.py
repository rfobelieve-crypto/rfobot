# -*- coding: utf-8 -*-
"""Security layer tests (TODO §0.83).

Three groups:
  1. unit — the sliding-window limiter and the HMAC signer behave as
     specified (strict paths tighter, window slides, keyless = no sig).
  2. STRUCTURAL — every /public/* route in server.py calls
     security.rate_gate as its first statement. This is the facade-guard
     pattern (mistake.md 2026-06-17): the rule "new public routes must be
     rate-limited" is enforced by machine, not memory. Reverse-proven:
     drop the gate from one handler and test 2 goes red naming it.
  3. signing surface — the four actionable feeds attach X-Signal-Signature.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from indicator.agent import security

SERVER = pathlib.Path(__file__).parent.parent / "indicator" / "agent" / "server.py"


# ── 1. limiter units ─────────────────────────────────────────────────────

def test_default_limit_allows_then_blocks(monkeypatch):
    monkeypatch.setattr(security, "_hits", {})
    now = 1000.0
    ip = "203.0.113.9"
    for i in range(security.DEFAULT_LIMIT):
        assert security.check(ip, "/public/signal-feed", now + i * 0.01)
    assert not security.check(ip, "/public/signal-feed", now + 5)


def test_window_slides(monkeypatch):
    monkeypatch.setattr(security, "_hits", {})
    now = 2000.0
    ip = "203.0.113.10"
    for i in range(security.DEFAULT_LIMIT):
        assert security.check(ip, "/public/chart", now + i * 0.01)
    assert not security.check(ip, "/public/chart", now + 10)
    # 61s later the window has slid — allowed again
    assert security.check(ip, "/public/chart", now + security.WINDOW_S + 11)


def test_strict_paths_much_tighter(monkeypatch):
    monkeypatch.setattr(security, "_hits", {})
    now = 3000.0
    ip = "203.0.113.11"
    for i in range(security.STRICT_LIMIT):
        assert security.check(ip, "/public/login", now + i * 0.01)
    assert not security.check(ip, "/public/login", now + 1)
    assert security.STRICT_LIMIT < security.DEFAULT_LIMIT
    # the strict bucket is separate: default-path traffic still allowed
    assert security.check(ip, "/public/signal-feed", now + 2)


def test_ips_are_isolated(monkeypatch):
    monkeypatch.setattr(security, "_hits", {})
    now = 4000.0
    for i in range(security.STRICT_LIMIT):
        assert security.check("1.1.1.1", "/public/login", now + i * 0.01)
    assert not security.check("1.1.1.1", "/public/login", now + 1)
    assert security.check("2.2.2.2", "/public/login", now + 1)


# ── 2. structural: every public route is gated ───────────────────────────

ROUTE_RE = re.compile(
    r'@mcp\.custom_route\("(/public/[^"]+)", methods=\[[^\]]*\]\)\n'
    r'async def (\w+)\(request: Request\)[^\n]*:\n(.*?)(?=\n@|\nasync def|\ndef |\Z)',
    re.S)


def test_every_public_route_calls_rate_gate():
    src = SERVER.read_text(encoding="utf-8")
    routes = ROUTE_RE.findall(src)
    assert len(routes) >= 20, f"route scan broken? found {len(routes)}"
    missing = [f"{path} ({fn})" for path, fn, body in routes
               if "security.rate_gate(request)" not in body.split("\n\n")[0]]
    assert not missing, f"public routes without rate_gate: {missing}"


def test_route_scan_sees_known_routes():
    """Guard the guard: if the regex ever stops matching, fail loudly
    rather than silently passing an empty list."""
    src = SERVER.read_text(encoding="utf-8")
    paths = [p for p, _f, _b in ROUTE_RE.findall(src)]
    for must in ("/public/signal-feed", "/public/login", "/public/chart"):
        assert must in paths, f"{must} not seen by scanner"


# ── 3. signing ───────────────────────────────────────────────────────────

def test_sign_keyless_is_none(monkeypatch):
    monkeypatch.delenv("SIGNAL_SIGNING_KEY", raising=False)
    assert security.sign({"a": 1}) is None


def test_sign_deterministic_and_key_order_free(monkeypatch):
    monkeypatch.setenv("SIGNAL_SIGNING_KEY", "test-key-0123456789abcdef")
    s1 = security.sign({"a": 1, "b": "x"})
    s2 = security.sign({"b": "x", "a": 1})
    assert s1 == s2 and isinstance(s1, str) and len(s1) == 64
    s3 = security.sign({"a": 2, "b": "x"})
    assert s3 != s1


def test_short_key_refused(monkeypatch):
    monkeypatch.setenv("SIGNAL_SIGNING_KEY", "short")
    assert security.sign({"a": 1}) is None


def test_attach_signature_signs_body_bytes(monkeypatch):
    """The signature target is the raw body — the exact bytes a verifier
    receives — never a canonical re-serialization (Python/JS float
    divergence: 100.0 vs 100 would break day one)."""
    import hashlib as _hl
    import hmac as _hm
    monkeypatch.setenv("SIGNAL_SIGNING_KEY", "test-key-0123456789abcdef")

    class FakeResp:
        body = b'{"confidence":100.0,"direction":"UP"}'
        headers = {}
    r = FakeResp()
    security.attach_signature(r)
    want = _hm.new(b"test-key-0123456789abcdef", FakeResp.body,
                   _hl.sha256).hexdigest()
    assert r.headers["X-Signal-Signature"] == f"v1={want}"
    assert "X-Signal-Signature" in r.headers["Access-Control-Expose-Headers"]


def test_attach_signature_keyless_noop(monkeypatch):
    monkeypatch.delenv("SIGNAL_SIGNING_KEY", raising=False)

    class FakeResp:
        body = b"{}"
        headers = {}
    r = FakeResp()
    security.attach_signature(r)
    assert "X-Signal-Signature" not in r.headers


def test_actionable_feeds_attach_signature():
    src = SERVER.read_text(encoding="utf-8")
    assert src.count("security.attach_signature(") >= 4, (
        "signal-feed + raid-signals + raid-pending + raid-outcomes "
        "must attach X-Signal-Signature")


def test_rate_gate_never_raises():
    class Broken:
        @property
        def url(self):
            raise RuntimeError("boom")
    assert security.rate_gate(Broken()) is None
