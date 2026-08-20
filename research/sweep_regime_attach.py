# -*- coding: utf-8 -*-
"""Gate F verdict attachment — variant B netR decomposed by concurrent
per-coin ADX regime.  Registered 2026-08-20, BEFORE the n=1400 verdict.

Role and non-role, stated up front:
  - This is an ATTACHMENT, not a criterion.  Gate F's frozen conditions
    (n>=1400, clustered CI-low > 0, >=6/9 coins) are untouched and this
    file cannot pass or fail anything.
  - Its job is to make the verdict READABLE: if Gate F fails, distinguish
    "the edge is dead" (losing in RANGING too — the mechanism's home turf
    per B-P9, RANGING +0.075 vs TRENDING +0.016, CI clear of zero, 8/9
    coins) from "the accumulation window was regime-unlucky" (losses
    concentrated in TRENDING, RANGING still positive).  Those two verdicts
    have opposite follow-ups, and without this attachment they print the
    same FAIL line.
  - Pre-committed reading, so verdict day cannot bend it:
      RANGING meanR > 0 and TRENDING meanR < 0  -> "regime headwind" note
      RANGING meanR <= 0                        -> "edge dead in its home
                                                    regime" note
    Neither note reopens the gate; a regime-headwind FAIL still fails.
    What it changes is the FOLLOW-UP question (re-accumulate under a
    fresh window vs kill the line), which is decided by the user then.

Also prints the regime MIX of the accumulation window vs the 29-month
base rate — a window with 2x the base-rate TRENDING share is itself part
of the story.

Uses only frozen instruments: ADX(14) 25/20 (§0.49d winner) on the same
per-coin kline caches the shadow engine maintains.  Read-only.

    python research/sweep_regime_attach.py        # human-readable
    (also writes research/results/sweep_regime_attach.json for the
     verdict-day record)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))

import sweep_core as SC                                    # noqa: E402
from research.crowd_battery2 import adx_state              # noqa: E402

LOG = ROOT / "research" / "results" / "sweep_shadow_log.csv"
CACHE = ROOT / "research" / "sweep_failure" / ".cache"
OUT = ROOT / "research" / "results" / "sweep_regime_attach.json"
STATES = ("TRENDING", "NEUTRAL", "RANGING")


def max_drawdown(seq):
    peak = cum = mdd = 0.0
    for x in seq:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def main() -> int:
    rows = []
    with open(LOG, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "CLOSED" or r.get("variant_b") != "1":
                continue
            try:
                rows.append({"sym": r["symbol"],
                             "ts": int(float(r["fill_ts"])),
                             "r": float(r["net_r"])})
            except (ValueError, KeyError, TypeError):
                continue
    rows.sort(key=lambda x: x["ts"])
    if not rows:
        print("no variant-B rows")
        return 1

    adx: dict[str, dict[int, str]] = {}
    base_mix: dict[str, int] = defaultdict(int)
    for s in sorted({x["sym"] for x in rows}):
        fp = CACHE / f"{s}USDT_1h.csv"
        if fp.exists():
            adx[s] = adx_state(SC.load_csv(str(fp)))
            for v in adx[s].values():
                base_mix[v] += 1

    by_state: dict[str, list[float]] = defaultdict(list)
    unlabelled = 0
    for x in rows:
        lab = adx.get(x["sym"], {}).get(x["ts"] // 3600 * 3600)
        if lab is None:
            unlabelled += 1
            continue
        by_state[lab].append(x["r"])

    span = (datetime.fromtimestamp(rows[0]["ts"], timezone.utc).date(),
            datetime.fromtimestamp(rows[-1]["ts"], timezone.utc).date())
    n_lab = sum(len(v) for v in by_state.values())
    base_n = sum(base_mix.values())
    print(f"variant B closed n={len(rows)}  {span[0]} -> {span[1]}  "
          f"(labelled {n_lab}, unlabelled {unlabelled})\n")
    print(f"{'state':9} {'n':>5} {'share':>7} {'base':>7} "
          f"{'meanR':>9} {'sumR':>9} {'MDD':>8}")
    out_states = {}
    for s in STATES:
        v = by_state.get(s, [])
        share = len(v) / n_lab if n_lab else 0.0
        base = base_mix.get(s, 0) / base_n if base_n else 0.0
        m = sum(v) / len(v) if v else None
        print(f"{s:9} {len(v):5d} {share*100:6.1f}% {base*100:6.1f}% "
              f"{(m if m is not None else float('nan')):+9.4f} "
              f"{sum(v):+9.2f} {max_drawdown(v):+8.2f}")
        out_states[s] = {"n": len(v), "share": round(share, 4),
                         "base_share": round(base, 4),
                         "meanR": round(m, 4) if m is not None else None,
                         "sumR": round(sum(v), 3),
                         "mdd": round(max_drawdown(v), 3)}

    # ── sufficiency guard (added on the first run, 2026-08-20, and it
    # TIGHTENS the instrument): fills cluster in time, so 850 fills over
    # 23 days are ~a handful of independent regime episodes, not 850
    # observations.  The very first run printed "edge-dead" off a window
    # whose RANGING bucket contradicts the 2.5-year CI-grade B-P9 — that
    # is exactly the over-eager reading this guard exists to silence.
    # Readings render only when (a) the log spans >=60 days and (b) the
    # day-clustered bootstrap CI of the RANGING-bucket mean excludes zero.
    import random
    random.seed(7)

    def day_cluster_ci(state):
        by_day = defaultdict(list)
        for x in rows:
            lab = adx.get(x["sym"], {}).get(x["ts"] // 3600 * 3600)
            if lab == state:
                by_day[x["ts"] // 86400].append(x["r"])
        days = list(by_day.values())
        if len(days) < 8:
            return None
        ms = []
        for _ in range(2000):
            s = [v for _ in range(len(days))
                 for v in days[random.randrange(len(days))]]
            ms.append(sum(s) / len(s))
        ms.sort()
        return ms[50], ms[1950], len(days)

    span_days = (rows[-1]["ts"] - rows[0]["ts"]) / 86400.0
    ci = day_cluster_ci("RANGING")
    if ci:
        print(f"\nRANGING day-clustered CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"({ci[2]} trading days)")
    rg = out_states["RANGING"]["meanR"]
    tr = out_states["TRENDING"]["meanR"]
    if span_days < 60 or ci is None or (ci[0] < 0 < ci[1]):
        note = (f"insufficient for a reading (span {span_days:.0f}d < 60d "
                f"or RANGING CI spans zero) -- pattern so far: RANGING "
                f"{rg:+.4f} / TRENDING {tr:+.4f}, "
                f"{'OPPOSITE of B-P9' if (rg or 0) < (tr or 0) else 'consistent with B-P9'}"
                " -- reading deferred to verdict day")
    elif rg is not None and rg > 0 and tr is not None and tr < 0:
        note = ("regime-headwind pattern: home regime (RANGING) positive, "
                "TRENDING negative -- a FAIL would still fail, but the "
                "follow-up question is re-accumulation, not burial")
    elif rg is not None and rg <= 0:
        note = ("edge-dead pattern: losing in RANGING (CI clear of zero), "
                "the mechanism's home regime per B-P9 -- regime cannot "
                "excuse this")
    else:
        note = "mixed pattern"
    print(f"pre-committed reading: {note}")

    OUT.write_text(json.dumps({
        "asof_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "n_total": len(rows), "states": out_states, "reading": note,
        "role": "Gate F attachment -- explanatory only, never a criterion",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
