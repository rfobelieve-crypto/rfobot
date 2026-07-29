# -*- coding: utf-8 -*-
"""Cancel playbooks — PATH analysis instead of fixed-time endpoints.

The user's methodological point (2026-07-29): hit_60m samples a single
instant, so a signal whose price runs the right way and then comes back
scores as a failure even though the move was there. For an execution
overlay that is the wrong question — an overlay does not need the endpoint,
it needs to know whether exploitable movement appears.

So this measures the PATH after each event:
  MFE  max favorable excursion within the window (signed by direction)
  MAE  max adverse excursion
  barrier race  for a sweep of barrier sizes, does the favorable level get
        touched before the adverse one (generalises the frozen ±0.5%
        first_hit to every scale)

The control is what makes it mean anything: every event is paired with a
random control minute drawn from the SAME DAY and scored with the SAME
direction. BTC moves 30bps in two hours regardless of cancel flow; only the
event-minus-control difference is evidence.

Status: EXPLORATORY, not a pre-registered gate. The barrier sweep is a
multiple comparison by construction (that is exactly the shape mistake.md
2026-06-20 warns about), so a level that looks good here earns a
pre-registered forward test — never a deployment.

Run: python research/cancel_path_analysis.py
Out: research/results/cancel_path_analysis.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn  # noqa: E402

OUT = ROOT / "research/results/cancel_path_analysis.json"
WINDOW_MIN = 120
BARRIERS = [0.001, 0.002, 0.003, 0.005, 0.008]      # 10 / 20 / 30 / 50 / 80 bps
SYMBOL = "BTC-USD"
RNG = random.Random(11)


def fetch_all():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT minute_start_ms ms, playbook, direction, px "
                "FROM cancel_playbook_events "
                "WHERE canonical_symbol=%s AND direction IN ('UP','DOWN') "
                "AND px IS NOT NULL ORDER BY minute_start_ms", (SYMBOL,))
            ev = cur.fetchall()
            lo = min(e["ms"] for e in ev) - 86_400_000
            hi = max(e["ms"] for e in ev) + WINDOW_MIN * 60_000
            cur.execute(
                "SELECT ts_ms, mid_price FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol=%s AND ts_ms BETWEEN %s AND %s "
                "ORDER BY ts_ms", (SYMBOL, lo, hi))
            book = cur.fetchall()
    finally:
        conn.close()
    mids = {int(r["ts_ms"]) // 60_000 * 60_000: float(r["mid_price"])
            for r in book if r["mid_price"] is not None}
    return ev, mids


def path_stats(mids: dict, t0_ms: int, px0: float, d: int) -> dict | None:
    """MFE/MAE in bps + first-touch verdict per barrier. d=+1 UP, -1 DOWN."""
    seq = []
    for k in range(1, WINDOW_MIN + 1):
        m = mids.get(t0_ms + k * 60_000)
        if m is not None:
            seq.append(m)
    if len(seq) < WINDOW_MIN // 2:          # need at least half the window
        return None
    rel = [d * (m - px0) / px0 for m in seq]
    out = {"mfe": max(rel) * 1e4, "mae": min(rel) * 1e4, "n_min": len(seq)}
    for b in BARRIERS:
        verdict = "unresolved"
        for r in rel:
            if r >= b:
                verdict = "hit"
                break
            if r <= -b:
                verdict = "miss"
                break
        out[f"b{int(b*1e4)}"] = verdict
    return out


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {}
    mfe = [r["mfe"] for r in rows]
    mae = [r["mae"] for r in rows]
    res = {"n": len(rows),
           "mfe_mean": sum(mfe) / len(mfe), "mae_mean": sum(mae) / len(mae)}
    for b in BARRIERS:
        k = f"b{int(b*1e4)}"
        h = sum(1 for r in rows if r[k] == "hit")
        m = sum(1 for r in rows if r[k] == "miss")
        res[k] = {"hit": h, "miss": m,
                  "rate": (h / (h + m)) if (h + m) else None,
                  "resolved_pct": 100 * (h + m) / len(rows)}
    return res


def main() -> int:
    ev, mids = fetch_all()
    print(f"events={len(ev)}  book minutes={len(mids)}")

    by_day = defaultdict(list)
    for ms in mids:
        by_day[ms // 86_400_000].append(ms)

    ev_rows, ct_rows = [], []
    per_pb = defaultdict(list)
    for e in ev:
        d = 1 if e["direction"] == "UP" else -1
        t0 = int(e["ms"]) // 60_000 * 60_000
        s = path_stats(mids, t0, float(e["px"]), d)
        if s is None:
            continue
        ev_rows.append(s)
        per_pb[e["playbook"]].append(s)
        # paired control: random minute, same day, same direction
        pool = by_day.get(t0 // 86_400_000, [])
        for _ in range(3):                    # a few tries for coverage
            if not pool:
                break
            tc = pool[RNG.randrange(len(pool))]
            c = path_stats(mids, tc, mids[tc], d)
            if c is not None:
                ct_rows.append(c)
                break

    E = summarize(ev_rows, "events")
    C = summarize(ct_rows, "control")
    print(f"\nusable events={E.get('n')}  controls={C.get('n')}")
    print(f"\n{'':<10}{'MFE bps':>10}{'MAE bps':>10}{'MFE/|MAE|':>11}")
    for tag, S in (("events", E), ("control", C)):
        print(f"{tag:<10}{S['mfe_mean']:>10.1f}{S['mae_mean']:>10.1f}"
              f"{S['mfe_mean']/abs(S['mae_mean']):>11.2f}")
    print(f"{'diff':<10}{E['mfe_mean']-C['mfe_mean']:>+10.1f}"
          f"{E['mae_mean']-C['mae_mean']:>+10.1f}")

    print(f"\nbarrier race (favorable touched before adverse):")
    print(f"{'barrier':<10}{'events':>16}{'control':>16}{'edge':>8}")
    for b in BARRIERS:
        k = f"b{int(b*1e4)}"
        e_, c_ = E[k], C[k]
        er = e_["rate"]
        cr = c_["rate"]
        if er is None or cr is None:
            continue
        print(f"{int(b*1e4):>4} bps  {er*100:>7.0f}% ({e_['hit']}/{e_['hit']+e_['miss']:<4})"
              f"{cr*100:>8.0f}% ({c_['hit']}/{c_['hit']+c_['miss']:<4})"
              f"{(er-cr)*100:>+7.1f}pp")

    print(f"\nper playbook (MFE / MAE bps):")
    for pb, rows in sorted(per_pb.items()):
        S = summarize(rows, pb)
        print(f"  {pb:<12} n={S['n']:<4} MFE {S['mfe_mean']:>6.1f}  "
              f"MAE {S['mae_mean']:>6.1f}  ratio {S['mfe_mean']/abs(S['mae_mean']):.2f}")

    OUT.write_text(json.dumps({"events": E, "control": C,
                               "per_playbook": {k: summarize(v, k)
                                                for k, v in per_pb.items()}},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("READ: only the event-minus-control difference is evidence; the "
          "barrier sweep is exploratory (multiple comparisons by design).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
