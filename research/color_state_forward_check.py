"""Ad hoc check (2026-07-20): does the display-layer colour state the user
watches on the review-chart ribbon (`classify_state` in
cancel_playbook_watcher.py) actually lead short-horizon price, net of cost?

Not a new pre-registered family — reuses classify_state/compute_features
VERBATIM (zero new thresholds, zero tuning). Triggered by a real-time
observation ("green tile then it rises, red tile then it falls, feels
profitable") that needs the same discipline as every other anecdote in this
project: check n, check both directions of the claim, check magnitude net
of cost, don't let "feels precise right now" stand untested (mistake.md
2026-06-02: a 5-minute check here is cheap insurance against weeks of
downstream cost).

Method: classify every minute in the collected depth_deltas_1m history,
group forward mid returns at h in {1,3,5,15,30,60} min by the resulting
colour bucket (vacuum_up / vacuum_down / absorption's own UP/DOWN colouring
/ cascade / calm baseline), report win rate (sign match), mean bps, and a
rough cost-adjusted verdict against the same ~8bps (2x maker) frontier
that killed the general subhourly line in Phase 0 (2026-07-18 G2).

Labelled honestly: SMOKE unless n comfortably supports the claim at the
SHORTEST horizon the user is actually watching (not just the official
60m hit_60m tracked in cancel_playbook_events, which only covers the
alertable gated playbooks and has never once seen a vacuum event fire).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from market_data.tasks.cancel_playbook_watcher import (
    classify_state, compute_features, load_frame)

HORIZONS_MIN = (1, 3, 5, 15, 30, 60)
COST_BPS = 8.0        # same 2x-maker frontier that killed subhourly G2


def colour_bucket(row: pd.Series) -> str | None:
    """Map a feature row to the exact colour the ribbon would draw, or
    None for calm (draws nothing — not a claim, excluded from the test)."""
    s = classify_state(row)
    if s["state"] == "calm":
        return None
    if s["state"] == "absorption":
        return "green" if s["direction"] == "UP" else "red"
    if s["state"] == "vacuum_up":
        return "green"
    if s["state"] == "vacuum_down":
        return "red"
    return "grey"          # cascade/rotation/surge — no directional colour


def main() -> int:
    df = load_frame(lookback_min=999_999)     # everything collected so far
    if df.empty:
        print("no depth_deltas_1m data")
        return 0
    feat = compute_features(df)
    feat = feat.dropna(subset=["shock"])
    buckets = feat.apply(colour_bucket, axis=1)
    mid = feat["mid"]

    print(f"total classified minutes: {len(feat)} "
          f"({feat.index.min()}..{feat.index.max()} in raw minute units)\n")
    print("bucket distribution:")
    print(buckets.value_counts(dropna=False).to_string())
    print()

    for bucket, want_sign in (("green", +1), ("red", -1)):
        sub_idx = buckets[buckets == bucket].index
        n = len(sub_idx)
        print(f"── {bucket} tiles (n={n}) — claim: price moves "
              f"{'UP' if want_sign > 0 else 'DOWN'} after ──")
        if n < 5:
            print("  too few to say anything\n")
            continue
        for h in HORIZONS_MIN:
            fwd = (mid.shift(-h) / mid - 1.0)
            y = fwd.loc[sub_idx].dropna()
            if len(y) < 5:
                print(f"  h={h:>2}m | insufficient ({len(y)})")
                continue
            hit = float(((y > 0) == (want_sign > 0)).mean())
            mean_bps = float(y.mean()) * want_sign * 1e4   # signed w/ claim direction
            se = float(y.std()) / np.sqrt(len(y)) * 1e4
            verdict = ("clears cost" if mean_bps > COST_BPS
                       else ("wrong sign" if mean_bps < 0 else "below cost"))
            print(f"  h={h:>2}m | n={len(y):>5} | win-rate {hit:.1%} | "
                  f"mean {mean_bps:+.2f}bps (se {se:.2f}) | {verdict}")
        print()

    print(f"Cost frontier used: {COST_BPS}bps (2x maker, same bar that "
          f"killed subhourly Phase 0 G2). This is exploratory, not a "
          f"pre-registered family — treat as a first honest look, not a verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
