"""Enforce the agent isolation boundary (.claude/rules/agent-boundary.md).

The MCP agent must be a read-only downstream consumer. These tests make
that a machine-checked invariant, not a matter of discipline — mirroring
the AST signature-parity test from mistake.md 2026-06-16.

If any of these fail, the agent has grown a path into the trading system
and MUST NOT ship.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

AGENT_DIR = pathlib.Path(__file__).resolve().parent.parent / "indicator" / "agent"

# Modules the agent may never pull in, at any import depth in its own files.
BANNED_IMPORT_SUBSTR = (
    "indicator.okx.executor",
    "indicator.okx.reconciler",
    "indicator.okx.kill_checks",
    "indicator.okx.runner",
    "indicator.okx.approval",
    "indicator.okx.client",
    "indicator.okx.rest",
    "indicator.okx.ws_private",
    "indicator.okx.accounts",   # holds decrypted credentials
    "indicator.inference",
)

# Only these tables may be written by the agent (its own namespace).
WRITE_RE = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE)\s+INTO\s+`?(\w+)`?",
                      re.IGNORECASE)


def _agent_py_files() -> list[pathlib.Path]:
    return sorted(AGENT_DIR.glob("*.py"))


def test_agent_dir_exists():
    assert AGENT_DIR.is_dir(), "indicator/agent/ missing"
    assert _agent_py_files(), "no python files under indicator/agent/"


def test_no_banned_imports():
    """AST-scan every agent file for imports of trading/hot-path modules."""
    offenders: list[str] = []
    for path in _agent_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            for name in names:
                for banned in BANNED_IMPORT_SUBSTR:
                    if banned in name:
                        offenders.append(f"{path.name}: imports {name}")
    assert not offenders, "Agent reaches into the trading system:\n" + \
        "\n".join(offenders)


def test_no_writes_to_quant_tables():
    """Any SQL write in agent code must target an agent_* table only."""
    offenders: list[str] = []
    for path in _agent_py_files():
        src = path.read_text(encoding="utf-8")
        for m in WRITE_RE.finditer(src):
            verb, table = m.group(1), m.group(2)
            if not table.lower().startswith("agent_"):
                offenders.append(f"{path.name}: {verb} INTO {table}")
    assert not offenders, "Agent writes to non-agent tables:\n" + \
        "\n".join(offenders)


def test_importing_agent_does_not_load_executor():
    """Importing the agent package must not drag any trading module into
    sys.modules as a side effect.

    Run in a SUBPROCESS with a pristine interpreter so the check measures
    exactly what the agent import pulls in — and so it never mutates the
    parent test session's sys.modules (which would corrupt other tests).

    Skips when the `mcp` package is not importable in this interpreter
    (e.g. local Python 3.9 — mcp needs >=3.10): server.py cannot even be
    imported there, so the probe would fail for an unrelated reason. CI
    runs 3.11 with requirements.indicator installed, so the boundary
    stays machine-enforced where it matters.
    """
    pytest.importorskip("mcp")
    import subprocess
    import sys

    probe = (
        "import sys; "
        "import indicator.agent.queries, indicator.agent.server; "
        "banned=('indicator.okx.executor','indicator.okx.reconciler',"
        "'indicator.okx.kill_checks','indicator.okx.runner',"
        "'indicator.okx.approval','indicator.okx.client',"
        "'indicator.okx.rest','indicator.okx.ws_private',"
        "'indicator.okx.accounts','indicator.inference'); "
        "leaked=[m for m in sys.modules "
        "if any(m==b or m.startswith(b+'.') for b in banned)]; "
        "print('LEAKED:'+','.join(leaked)); "
        "sys.exit(1 if leaked else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(AGENT_DIR.parent.parent),
        env={**__import__("os").environ, "OKX_AGENT_SEED": "1"},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, \
        f"agent import leaked trading modules:\n{proc.stdout}\n{proc.stderr}"


def test_seed_mode_needs_no_db(monkeypatch):
    """In seed mode every tool returns data without touching MySQL."""
    monkeypatch.setenv("OKX_AGENT_SEED", "1")
    import importlib
    from indicator.agent import queries
    importlib.reload(queries)

    sig = queries.latest_signal()
    assert sig["direction"] in ("UP", "DOWN", "NEUTRAL")
    assert sig["_source"] == "seed"
    assert "disclaimer" in sig

    of = queries.orderflow_snapshot()
    assert of["_source"] == "seed"
    assert "imbalance_l20" in of

    tr = queries.track_record()
    assert tr["_source"] == "seed"
    assert tr["gate_a_signal_layer"]["n"] == 739

    rf = queries.risk_frame(63000.0, "UP")
    assert rf["direction"] == "LONG"
    assert rf["leverage_hard_cap"] == 2.0
    assert rf["atr_stop_price"] < 63000.0     # long stop below entry
    assert "disclaimer" in rf


def test_every_tool_carries_disclaimer(monkeypatch):
    """No tool may return advice-shaped data without the disclaimer."""
    monkeypatch.setenv("OKX_AGENT_SEED", "1")
    import importlib
    from indicator.agent import queries
    importlib.reload(queries)
    for fn in (lambda: queries.latest_signal(),
               lambda: queries.orderflow_snapshot(),
               lambda: queries.track_record(),
               lambda: queries.risk_frame(63000.0, "DOWN")):
        out = fn()
        assert "disclaimer" in out, f"{fn} missing disclaimer"
