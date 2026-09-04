# -*- coding: utf-8 -*-
"""The one place flow_system knows where the arbitrage line lives.

2026-09-04: the arb line moved out of this repo into its own
(Desktop/flowbot/arb, its own GitHub repo, its own account, its own
credentials). CLAUDE.md always described it as a separate line that shares
nothing with the trading path except two read-only display layers; now the
filesystem says so too.

Why a module instead of a few relative paths: the move had ELEVEN holders of
the old path, and one of them (the Windows scheduled task for the watchdog)
does not appear in any grep of the repo. mistake.md 2026-08-29 is the same
shape -- a layout change validated on the writer and one reader, while a
second reader silently counted from zero. Next time this moves it should be
one line here, not another enumeration.

Set ARB_HOME to override (e.g. when the engine runs on the Tokyo VPS and
this repo only publishes). Everything else derives from it.

Nothing here imports the engine or the arb library at module scope; callers
that need `arblib` call `add_to_path()` first. The boundary in
.claude/rules/agent-boundary.md is unchanged: flow_system reads arb
artifacts, never the other way round.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent          # flow_system/

#: Root of the arbitrage repo. Default is the sibling directory.
HOME = Path(os.environ.get("ARB_HOME") or (_REPO.parent / "arb")).resolve()

#: Where the recorder writes minute CSVs and the scanner writes scan CSVs.
LOGS = HOME / "engine" / "logs"
SCAN = LOGS / "scan"

#: Where the scorers write their verdict JSON (read by arb_publish.py).
RESULTS = HOME / "results"

#: The judgement library (fees, cost model, verdict scorers).
LIB = HOME / "arblib"


def add_to_path() -> None:
    """Make `import arblib` work from inside flow_system.

    Only the display bridge needs this. It imports the arb line's cost
    model deliberately -- copying those numbers into this repo would be a
    second implementation, and a second implementation disagrees silently
    (mistake.md 2026-08-26).
    """
    p = str(HOME)
    if p not in sys.path:
        sys.path.insert(0, p)


def missing() -> str | None:
    """Human-readable reason the arb line is not reachable, else None.

    Callers should surface this rather than treating an absent directory as
    'no data yet' -- those two look identical from a file count, and the
    freshness board exists because that difference matters.
    """
    if not HOME.is_dir():
        return f"ARB_HOME not found: {HOME}"
    if not LIB.is_dir():
        return f"arblib missing under {HOME}"
    if not LOGS.is_dir():
        return f"engine logs missing under {HOME}"
    return None
