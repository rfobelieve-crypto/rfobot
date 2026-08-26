# -*- coding: utf-8 -*-
"""Guard: a column the payload SELECTs must actually reach the consumer.

Written 2026-08-26 after `/public/raid-pending` shipped with `regime_cell`
in its SELECT list, in the CREATE TABLE, in the publisher, and in the spec
handed to the product side -- and nowhere in the dict it returns. The
product side built their regime filter against that promise; they would
have received `undefined`, and `undefined !== 'RANGING'` blocks EVERY
signal while logging skip-reason "regime", i.e. a total outage wearing the
costume of a working filter.

Nothing caught it because every layer that was easy to check was correct.
The consumer-visible shape is only observable when the list is non-empty,
and the list is empty whenever the market is quiet -- so the one moment the
bug is visible is the one moment nobody is looking.

Same family as the facade-conformance guard (tests/test_okx_client.py,
mistake.md 2026-06-17): a value that must survive several layers, where
each layer looks fine alone. Enforced by AST rather than by memory.

The check is deliberately narrow: for each public payload builder listed
below, every column named in its SQL SELECT must appear as a dict-literal
key somewhere in that same function. Columns used purely as control flow
(filtering, ordering, freshness) are exempted by name, and every exemption
has to say why -- an unexplained exemption is how this bug comes back.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[1] / "indicator" / "agent" / "server.py"

# builder -> columns that legitimately never reach the consumer, with reason
EXEMPT = {
    "_raid_pending_payload": {
        "expires_ts": "re-emitted as expires_in_min AND as itself",
        "updated_at": "aggregated into asof_utc, not per row",
    },
    "_raid_signals_payload": {
        "updated_at": "aggregated into asof_utc, not per row",
    },
    "_raid_outcomes_payload": {
        "updated_at": "aggregated into asof_utc, not per row",
    },
    "_weather_payload": {
        "payload": "JSON blob; json.loads'd and its contents are the payload",
        "updated_at": "aggregated into asof_utc, not per row",
    },
}
BUILDERS = list(EXEMPT)


def _select_columns(fn: ast.FunctionDef) -> set[str]:
    """Columns named in the function's SQL SELECT ... FROM.

    Each string constant is examined ON ITS OWN. Python merges adjacent
    string literals into a single Constant, so a multi-line query is one
    node and needs no stitching — whereas joining every constant in the
    function (the first version here) let prose from the return dict land
    between fragments, and the regex happily spanned it. That produced a
    "column list" made of sentences, which is how this guard announced
    itself when the raid-outcomes endpoint was added (2026-08-26). Loud,
    but wrong; a fragile instrument is still an instrument to fix.
    """
    m = None
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, str)):
            continue
        hit = re.search(r"SELECT\s+(.*?)\s+FROM\s", n.value, re.I | re.S)
        if hit:
            m = hit
            break
    if not m:
        return set()
    cols = set()
    for part in m.group(1).split(","):
        part = part.strip()
        if not part or "*" in part:
            continue
        # strip "x AS y" -> y, and any qualifier
        alias = re.split(r"\s+AS\s+", part, flags=re.I)
        cols.add(alias[-1].strip().split(".")[-1])
    return cols


def _dict_keys(fn: ast.FunctionDef) -> set[str]:
    """Every string key of every dict literal built inside the function."""
    return {
        k.value for d in ast.walk(fn) if isinstance(d, ast.Dict)
        for k in d.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


def _find(name: str) -> ast.FunctionDef:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in server.py — renamed? "
                         "Update this guard rather than deleting it.")


@pytest.mark.parametrize("builder", BUILDERS)
def test_selected_columns_reach_the_payload(builder):
    fn = _find(builder)
    selected = _select_columns(fn)
    assert selected, f"{builder}: no SELECT parsed — guard is not actually checking"
    emitted = _dict_keys(fn)
    missing = selected - emitted - set(EXEMPT[builder])
    assert not missing, (
        f"{builder} SELECTs {sorted(missing)} but never puts them in any "
        f"returned dict. Either emit them, or add an entry to EXEMPT with a "
        f"reason. Dropping a column silently is the 2026-08-26 regime_cell bug."
    )


def test_home_regime_is_a_list_not_a_string():
    """TODO §0.59b: the shape change is the loud-failure mechanism.

    A consumer still doing `=== "RANGING"` must break visibly. If this ever
    reverts to a bare string, that safety property is gone.
    """
    fn = _find("_raid_pending_payload")
    for d in ast.walk(fn):
        if not isinstance(d, ast.Dict):
            continue
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and k.value == "home_regime":
                assert isinstance(v, (ast.List, ast.Tuple)), (
                    "home_regime must stay a list (§0.59b: RANGING ∪ "
                    "TREND_DOWN); a bare string silently restores the old "
                    "single-cell rule in every consumer.")
                cells = {e.value for e in v.elts if isinstance(e, ast.Constant)}
                assert cells == {"RANGING", "TREND_DOWN"}, (
                    f"home_regime is {sorted(cells)}; §0.59b froze "
                    "{'RANGING','TREND_DOWN'}. Changing it needs a TODO entry, "
                    "not an edit here.")
                return
    raise AssertionError("home_regime not emitted at all")
