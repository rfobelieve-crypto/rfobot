"""V7 decay decomposition — is the 59.5% -> 53.7% WR decay real rot, or the
broken decoder stuffing the sample with its weak side?

PRE-REGISTERED 2026-08-17 (TODO §0.49b) before any number was produced.

Suspicion: the recent-90d cohort was mostly fired by the arithmetically
one-sided decode (UP-only for months, see mistake.md 2026-08-11) while LONG
is the -26bps side of this system.  If the decay is a COMPOSITION artifact
(direction/regime mix shifted under a broken instrument), the edge did not
rot and the standing CLAUDE.md claim needs rewriting; if within-cell WR
dropped — especially inside CALM, the mean-reverter's home turf — the rot
is real and the retrain cadence stays non-negotiable.

Frozen method: Strong signals split ERA1 (older than 90d) / ERA2 (last
90d, same boundary as the standing claim); cells = direction x trend
bucket (trend_z exactly as frozen in survival_cards.py).  Oaxaca-style:

    WR2 - WR1 = sum_c (s2_c - s1_c) * w1_c      [composition]
              + sum_c s2_c * (w2_c - w1_c)      [within-cell]

where s = cell share, w = cell WR.  Cells empty in one era contribute via
whichever term is computable; both terms and the residual-free identity are
printed.  ERA2 n is small — bootstrap CIs reported honestly, no forced
verdict.  Read-only research code.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.survival_cards import (  # noqa: E402
    CACHE, SC, bucket_of, trend_z_series)

BOOT_N = 2000
SEED = 7
ERA_DAYS = 90


def load_signals():
    from shared.db import get_db_conn
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT signal_time, direction, correct FROM tracked_signals "
                "WHERE strength='Strong' AND actual_return_4h IS NOT NULL "
                "AND direction IN ('UP','DOWN')")
            return cur.fetchall()
    finally:
        conn.close()


def cellify(rows, zmap, cutoff):
    eras = {"ERA1": [], "ERA2": []}
    for r in rows:
        t = r["signal_time"].replace(tzinfo=timezone.utc)
        z = zmap.get(int(t.timestamp()) // 3600 * 3600)
        if z is None:
            continue
        cell = (r["direction"], bucket_of(z))
        eras["ERA2" if t >= cutoff else "ERA1"].append(
            (cell, int(r["correct"] or 0)))
    return eras


def stats(era):
    n = len(era)
    share = defaultdict(float)
    wr = {}
    by = defaultdict(list)
    for cell, c in era:
        by[cell].append(c)
    for cell, v in by.items():
        share[cell] = len(v) / n
        wr[cell] = sum(v) / len(v)
    return n, share, wr


def decompose(era1, era2):
    n1, s1, w1 = stats(era1)
    n2, s2, w2 = stats(era2)
    cells = sorted(set(s1) | set(s2))
    comp = sum((s2.get(c, 0) - s1.get(c, 0)) * w1.get(c, w2.get(c, 0))
               for c in cells)
    within = sum(s2.get(c, 0) * (w2.get(c, w1.get(c, 0)) - w1.get(c, w2.get(c, 0)))
                 for c in cells)
    total = (sum(c for _, c in era2) / n2) - (sum(c for _, c in era1) / n1)
    return total, comp, within


def main():
    bars = SC.load_csv(str(CACHE / "BTCUSDT_1h.csv"))
    zmap = trend_z_series(bars)
    rows = load_signals()
    cutoff = datetime.now(timezone.utc) - timedelta(days=ERA_DAYS)
    eras = cellify(rows, zmap, cutoff)
    n1, s1, w1 = stats(eras["ERA1"])
    n2, s2, w2 = stats(eras["ERA2"])

    print(f"ERA1 (>{ERA_DAYS}d): n={n1}  WR={100*sum(c for _,c in eras['ERA1'])/n1:.1f}%")
    print(f"ERA2 (last {ERA_DAYS}d): n={n2}  WR={100*sum(c for _,c in eras['ERA2'])/n2:.1f}%")
    print(f"\n{'cell':<16}{'ERA1 share':>11}{'ERA1 WR':>9}{'ERA2 share':>11}{'ERA2 WR':>9}")
    for c in sorted(set(s1) | set(s2)):
        print(f"{c[0]+'×'+c[1]:<16}"
              f"{100*s1.get(c,0):>10.1f}%{(100*w1[c] if c in w1 else float('nan')):>8.0f}%"
              f"{100*s2.get(c,0):>10.1f}%{(100*w2[c] if c in w2 else float('nan')):>8.0f}%")

    total, comp, within = decompose(eras["ERA1"], eras["ERA2"])
    print(f"\n分解  總差 {100*total:+.1f}pp = 組成效應 {100*comp:+.1f}pp "
          f"+ 格內效應 {100*within:+.1f}pp")

    # bootstrap the split (resample signals within each era)
    rng = random.Random(SEED)
    comps, withins = [], []
    for _ in range(BOOT_N):
        b1 = [eras["ERA1"][rng.randrange(n1)] for _ in range(n1)]
        b2 = [eras["ERA2"][rng.randrange(n2)] for _ in range(n2)]
        try:
            _, c_, w_ = decompose(b1, b2)
            comps.append(c_)
            withins.append(w_)
        except ZeroDivisionError:
            continue
    comps.sort()
    withins.sort()
    ci = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v))])
    cl, ch = ci(comps)
    wl, wh = ci(withins)
    print(f"  組成效應 CI95 [{100*cl:+.1f},{100*ch:+.1f}]pp")
    print(f"  格內效應 CI95 [{100*wl:+.1f},{100*wh:+.1f}]pp")

    # the specific frozen sub-question: did CALM cells rot?
    print("\nCALM 格內對照（均值回歸的主場）:")
    for d in ("UP", "DOWN"):
        c = (d, "CALM")
        a = f"{100*w1[c]:.0f}% (n={int(s1.get(c,0)*n1)})" if c in w1 else "—"
        b = f"{100*w2[c]:.0f}% (n={int(s2.get(c,0)*n2)})" if c in w2 else "—"
        print(f"  {d:<5} ERA1 {a:<16} ERA2 {b}")


if __name__ == "__main__":
    main()
