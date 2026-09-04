# -*- coding: utf-8 -*-
"""Structural guard: no registered clock may carry an undecidable criterion.

Why this is a test and not a checklist (2026-09-04): the terrain adoption
trigger ran for a month on a threshold of 8pp against a quantity whose
standard error, at the registered sample size, was 11.6pp. Nothing failed;
it simply accumulated a sample that could never answer the question. A
checklist would not have caught it, because the checklist was followed --
`.claude/rules/mistake.md` 2026-08-26 already says "代入現有數字算一次",
and that was done. What was missing is the SECOND question: how precise is
that number.

Same shape as tests/test_agent_boundary.py and tests/test_okx_client.py:
the rule is enforced by machine, because three separate incidents have shown
it is not enforced by memory.

An entry may be exempt, but only with a NAMED reason -- an unexplained
exemption is how a guard quietly stops guarding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.prereg_power import CLOCKS, Z_POWER  # noqa: E402


def test_registry_is_not_empty():
    assert CLOCKS, "registry emptied — that is not how a clock is retired"


@pytest.mark.parametrize("c", CLOCKS, ids=lambda c: c.name[:34])
def test_clock_can_actually_decide(c):
    ok, why = c.verdict()
    # UNDERPOWERED is a fact to record, not a build break: a CI-based rule
    # cannot pass by noise, it can only fail to conclude. But it must be
    # ACKNOWLEDGED by name, with the sample size that would decide it --
    # otherwise this test goes permanently red and stops being read
    # (mistake.md 2026-09-03: a light that is always red is a broken light).
    if c.status() == "UNDERPOWERED":
        assert c.underpowered_ack, (
            f"{c.name}: {why}\n"
            f"Acknowledge it in the registry (underpowered_ack=...) with what "
            f"happens at the deadline, or redesign it. Do NOT loosen the "
            f"criterion.")
        assert "n≈" in why, "the ack must carry the deciding sample size"
        return
    if c.exempt:
        assert len(c.exempt) > 40, (
            f"{c.name}: exemption must state WHY, in a sentence someone can "
            f"argue with — got {c.exempt!r}")
        return
    # The terrain trigger is kept in the registry on purpose: it is the
    # worked example this file exists for, and deleting it would delete the
    # evidence. It is allowed to stay red.
    if c.name.startswith("地形扳機"):
        assert not ok, (
            "the terrain trigger must stay BLOCKED — it is the reference "
            "case; if it ever passes, the power maths has broken")
        assert c.ratio() < 1.0
        return
    assert ok, f"{c.name}: {why}"


def test_point_estimate_criteria_need_threshold_above_noise():
    """A point-estimate threshold below its own standard error decides nothing."""
    for c in CLOCKS:
        if c.ci_based or c.exempt or c.name.startswith("地形扳機"):
            continue
        assert c.ratio() >= Z_POWER, (
            f"{c.name}: threshold {c.threshold}{c.unit} is only "
            f"{c.ratio():.2f}x its SE ({c.se():.4g}{c.unit}); needs "
            f"{Z_POWER}x for 80% power. Fix the DESIGN (more sample, a "
            f"different metric, or a CI-based criterion) — never the "
            f"threshold, which only makes a powerless design more powerless.")


def test_se_units_match_threshold_units():
    """Reverse proof for the unit bug found on the day this was written.

    The proportion SEs are returned in percentage points because the
    thresholds are in percentage points. When they were returned as
    fractions the gate silently passed everything -- a guard measuring in
    the wrong unit is a guard with no measuring ability at all.
    """
    prop = [c for c in CLOCKS if c.kind in ("prop2", "prop1")]
    assert prop, "no proportion clocks left to check the units against"
    for c in prop:
        # any real proportion SE in pp is >0.1; a fraction would be <0.1
        assert c.se() > 0.1, (
            f"{c.name}: SE {c.se():.6g} looks like a fraction, not "
            f"percentage points — the unit bug is back")
