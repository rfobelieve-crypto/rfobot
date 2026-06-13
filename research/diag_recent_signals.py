"""
One-off diagnostic: are recent signals actually less accurate?

Pulls tracked_signals, splits last 14d vs the prior baseline window,
breaks down by tier + direction, computes sign-acc + avg return + mag IC,
and checks for model-version mixing (2026-04-13 calibration lesson).

NOT a deploy gate. Just answers "is the 'feels inaccurate' real or noise?"
"""
import sys
from datetime import datetime, timezone, timedelta
from scipy.stats import spearmanr
import numpy as np

sys.path.insert(0, ".")
from shared.db import get_db_conn

NOW = datetime.now(timezone.utc)
CUT = NOW - timedelta(days=14)

conn = get_db_conn()
try:
    with conn.cursor() as cur:
        # latest model deploy reference + version mix
        cur.execute("""
            SELECT model_version, COUNT(*) n,
                   MIN(signal_time) first_t, MAX(signal_time) last_t
            FROM tracked_signals
            WHERE filled = 1 AND signal_time >= %s
            GROUP BY model_version ORDER BY last_t DESC
        """, (CUT,))
        print("=== model_version mix in last 14d (filled signals) ===")
        for r in cur.fetchall():
            print(f"  {r['model_version'] or '(null)'}: n={r['n']}  {r['first_t']} -> {r['last_t']}")

        def window_stats(label, where, params):
            cur.execute(f"""
                SELECT direction, strength, confidence, mag_pred,
                       actual_return_4h, correct
                FROM tracked_signals
                WHERE filled = 1 AND {where}
            """, params)
            rows = cur.fetchall()
            print(f"\n=== {label} (n={len(rows)}) ===")
            if not rows:
                print("  (no filled signals)")
                return
            for tier in ("Strong", "Moderate"):
                for d in ("UP", "DOWN"):
                    sub = [r for r in rows if r["strength"] == tier and r["direction"] == d]
                    if not sub:
                        continue
                    n = len(sub)
                    wins = sum(int(r["correct"]) for r in sub)
                    avg = np.mean([float(r["actual_return_4h"]) for r in sub]) * 100
                    print(f"  {tier:8s} {d:4s}: {wins}/{n} = {wins/n*100:4.0f}% sign-acc  "
                          f"avg_ret={avg:+.2f}%")
            # overall + mag IC
            allwin = sum(int(r["correct"]) for r in rows)
            print(f"  {'ALL':8s}     : {allwin}/{len(rows)} = {allwin/len(rows)*100:4.0f}% sign-acc")
            mp = [abs(float(r["mag_pred"])) for r in rows]
            ar = [abs(float(r["actual_return_4h"])) for r in rows]
            if len(set(mp)) > 2:
                ic, p = spearmanr(mp, ar)
                print(f"  mag IC (|pred| vs |actual|): {ic:+.3f} (p={p:.2f}, n={len(rows)})")

        window_stats("LAST 14d", "signal_time >= %s", (CUT,))
        window_stats("PRIOR baseline (14-60d ago)",
                     "signal_time < %s AND signal_time >= %s",
                     (CUT, NOW - timedelta(days=60)))
finally:
    conn.close()
