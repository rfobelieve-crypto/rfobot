"""Regression tests for the indicator service admin-route guard.

The guard (indicator/app.py `_admin_guard`) must:
  1. fail-closed (503) on protected routes when ADMIN_HEAL_TOKEN is unset
  2. reject (403) wrong/missing tokens when the env var is set
  3. accept the token via X-Admin-Token header or ?token= query param
  4. leave the public product surface (/, /health, /json, charts) untouched
  5. keep every @app.route decorator bound to its original view function
     (mistake.md 2026-05-31: an Edit once detached a decorator silently)
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# indicator.app calls load_dotenv() at import, injecting local .env values
# (incl. OKX live keys) into os.environ — that leaks into later test files
# (e.g. test_okx_runner assumes no live keys). Snapshot & restore around the
# import so this file has zero env side effects on the rest of the suite.
_ENV_SNAPSHOT = os.environ.copy()
from indicator.app import app, _ADMIN_EXACT_PATHS, _ADMIN_PATH_PREFIXES  # noqa: E402
os.environ.clear()
os.environ.update(_ENV_SNAPSHOT)

TOKEN = "test-secret-token"

# One representative per guard category; cheap GETs that never reach heavy
# handler logic when the guard blocks them.
PROTECTED_SAMPLES = [
    "/force-update",
    "/okx-status",
    "/okx-admin/heal",
    "/admin/db-health-all",
    "/dashboard",
    "/db-diag",
    "/test-telegram",
    "/meeting",
]

PUBLIC_SAMPLES = ["/", "/health", "/json", "/live-chart", "/indicator-chart"]


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def with_token(monkeypatch):
    monkeypatch.setenv("ADMIN_HEAL_TOKEN", TOKEN)


@pytest.fixture()
def without_token(monkeypatch):
    monkeypatch.delenv("ADMIN_HEAL_TOKEN", raising=False)


@pytest.mark.parametrize("path", PROTECTED_SAMPLES)
def test_fail_closed_when_env_unset(client, without_token, path):
    resp = client.get(path)
    assert resp.status_code == 503, (
        f"{path} must fail closed without ADMIN_HEAL_TOKEN, got {resp.status_code}"
    )
    assert "not configured" in resp.get_json()["error"]


@pytest.mark.parametrize("path", PROTECTED_SAMPLES)
def test_reject_missing_token(client, with_token, path):
    resp = client.get(path)
    assert resp.status_code == 403


@pytest.mark.parametrize("path", PROTECTED_SAMPLES)
def test_reject_wrong_token(client, with_token, path):
    resp = client.get(path, headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 403
    resp = client.get(f"{path}?token=wrong")
    assert resp.status_code == 403


def test_accept_header_token(client, with_token):
    # /db-diag touches the DB and may 500 locally — anything except 403/503
    # proves the guard let the request through.
    resp = client.get("/db-diag", headers={"X-Admin-Token": TOKEN})
    assert resp.status_code not in (403, 503)


def test_accept_query_token(client, with_token):
    resp = client.get(f"/db-diag?token={TOKEN}")
    assert resp.status_code not in (403, 503)


@pytest.mark.parametrize("path", PUBLIC_SAMPLES)
def test_public_routes_not_guarded(client, without_token, path):
    resp = client.get(path)
    # Public routes may legitimately 503 with "not ready" state, but must
    # never return the guard's lock message.
    if resp.status_code == 503 and resp.is_json:
        assert "not configured" not in (resp.get_json() or {}).get("error", "")
    assert resp.status_code != 403


def test_every_admin_route_in_url_map_is_covered():
    """Any route under a sensitive family must match the guard lists, so a
    future admin route added to app.py can't silently ship unguarded."""
    sensitive_markers = ("/admin/", "/okx-admin/", "/dashboard")
    for rule in app.url_map.iter_rules():
        path = str(rule.rule)
        if any(m in path for m in sensitive_markers):
            covered = (path in _ADMIN_EXACT_PATHS
                       or path.startswith(_ADMIN_PATH_PREFIXES))
            assert covered, f"unguarded sensitive route: {path}"


def test_route_bindings_intact():
    """Decorator-detachment regression (mistake.md 2026-05-31): each known
    endpoint must still map to its original view function name."""
    bindings = {str(r.rule): r.endpoint for r in app.url_map.iter_rules()}
    expected = {
        "/": "chart",
        "/health": "health",
        "/json": "prediction_json",
        "/force-update": "force_update",
        "/okx-admin/heal": "okx_admin_heal_api",
        "/dashboard": "dashboard_route",
    }
    for path, endpoint in expected.items():
        assert bindings.get(path) == endpoint, (
            f"{path} bound to {bindings.get(path)!r}, expected {endpoint!r}"
        )
