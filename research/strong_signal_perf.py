"""
Strong Signal Performance Tracker — end-to-end production signal outcome.

Unlike calibration_check.py (which looks at all dir_prob_up samples), this
tool only looks at signals that actually triggered a production alert
(confidence >= 65 for Moderate, >= 80 for Strong). These are the samples
that matter for user-facing performance.

Model-version guard: by default, only include signals recorded AFTER the
current direction_xgb.json mtime. This prevents mixing win rates across
different model versions (see mistake log 2026-04-13).

Usage:
    python research/strong_signal_perf.py                 # auto-guard active
    python research/strong_signal_perf.py --since 2026-04-01
    python research/strong_signal_perf.py --no-version-guard
    python research/strong_signal_perf.py --wilson        # show Wilson CI

Output:
    research/results/strong_signal_perf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_FILE = Path("indicator/model_artifacts/dual_model/direction_xgb.json")
OUT_JSON = Path("research/results/strong_signal_perf.json")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def get_model_deploy_time() -> datetime | None:
    if not MODEL_FILE.exists():
        return None
    return datetime.fromtimestamp(MODEL_FILE.stat().st_mtime, tz=timezone.utc)


def _connect_railway():
    """Connect to Railway MySQL (same pattern as review_signals.py)."""
    import pymysql
    return pymysql.connect(
        host='caboose.proxy.rlwy.net', port=18766, user='root',
        password='lKukMzubRgBHgUaFAJnqwQKVDWQOJLkA', database='railway',
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_signals(since: datetime | None):
    conn = _connect_railway()
    try:
        with conn.cursor() as cur:
            if since is not None:
                cur.execute("""
                    SELECT signal_time, direction, strength, p_up, mag_pred,
                           confidence, entry_price, regime, exit_price,
                           actual_return_4h, correct, filled
                    FROM tracked_signals
                    WHERE signal_time >= %s
                    ORDER BY signal_time ASC
                """, (since.strftime("%Y-%m-%d %H:%M:%S"),))
            else:
                cur.execute("""
                    SELECT signal_time, direction, strength, p_up, mag_pred,
                           confidence, entry_price, regime, exit_price,
                           actual_return_4h, correct, filled
                    FROM tracked_signals
                    ORDER BY signal_time ASC
                """)
            return list(cur.fetchall())
    finally:
        conn.close()


def summarize_group(rows: list[dict], label: str) -> dict:
    filled = [r for r in rows if r.get("filled") and r.get("correct") is not None]
    total = len(rows)
    n_filled = len(filled)
    n_correct = sum(1 for r in filled if r["correct"] == 1)

    if n_filled == 0:
        return {
            "label": label,
            "total": total,
            "filled": 0,
            "win_rate": None,
            "ci_lo": None, "ci_hi": None,
            "avg_confidence": float(np.mean([r["confidence"] for r in rows])) if rows else None,
            "avg_return": None,
        }

    win_rate = n_correct / n_filled
    ci_lo, ci_hi = wilson_ci(n_correct, n_filled)
    avg_ret = float(np.mean([r["actual_return_4h"] for r in filled]))
    # Return in direction of signal (positive = correct direction)
    dir_rets = [
        r["actual_return_4h"] if r["direction"] == "UP" else -r["actual_return_4h"]
        for r in filled
    ]
    avg_dir_ret = float(np.mean(dir_rets))

    return {
        "label": label,
        "total": total,
        "filled": n_filled,
        "pending": total - n_filled,
        "correct": n_correct,
        "win_rate": win_rate,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "avg_confidence": float(np.mean([r["confidence"] for r in rows])),
        "avg_return_raw": avg_ret,
        "avg_return_directional": avg_dir_ret,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None,
                    help="Only include signals on/after this UTC date (YYYY-MM-DD)")
    ap.add_argument("--no-version-guard", action="store_true",
                    help="Disable model-version guard")
    ap.add_argument("--recent-days", type=int, default=None,
                    help="Only look at last N days (overrides version guard)")
    args = ap.parse_args()

    # Determine cutoff
    cutoff: datetime | None = None
    guard_source = "none"

    if args.recent_days is not None:
        from datetime import timedelta
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=args.recent_days)
        guard_source = f"--recent-days={args.recent_days}"
    elif args.since:
        cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        guard_source = f"--since={args.since}"
    elif not args.no_version_guard:
        cutoff = get_model_deploy_time()
        if cutoff is not None:
            guard_source = f"model mtime ({MODEL_FILE.name})"

    print("=" * 70)
    print("STRONG SIGNAL PERFORMANCE")
    print("=" * 70)
    if cutoff is not None:
        print(f"Cutoff: {cutoff}  (source: {guard_source})")
    else:
        print("Cutoff: NONE (mixing all model versions — use with caution)")

    rows = fetch_signals(cutoff)
    print(f"Signals found: {len(rows)}")

    if len(rows) == 0:
        if cutoff is not None:
            hrs = (datetime.now(tz=timezone.utc) - cutoff).total_seconds() / 3600
            print(f"\nNo signals recorded in the {hrs:.0f}h since cutoff.")
            print("Either the model hasn't triggered any Strong/Moderate alerts yet,")
            print("or the tracked_signals table hasn't received new rows.")
        return

    # Overall by strength
    print("\n── BY STRENGTH ──")
    results = {"cutoff": str(cutoff) if cutoff else None, "guard_source": guard_source}
    for strength in ["Strong", "Moderate"]:
        sub = [r for r in rows if r["strength"] == strength]
        s = summarize_group(sub, strength)
        results[strength.lower()] = s
        if s["filled"] == 0:
            print(f"  {strength:10s}: total={s['total']:3d}  filled=0  (all pending)")
        else:
            print(f"  {strength:10s}: total={s['total']:3d}  filled={s['filled']:3d}  "
                  f"win={s['win_rate']:.3f}  "
                  f"CI=[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  "
                  f"avg_dir_ret={s['avg_return_directional']:+.4f}")

    # By direction (within Strong)
    strong = [r for r in rows if r["strength"] == "Strong"]
    if strong:
        print("\n── STRONG BY DIRECTION ──")
        for direction in ["UP", "DOWN"]:
            sub = [r for r in strong if r["direction"] == direction]
            if not sub:
                continue
            s = summarize_group(sub, f"Strong-{direction}")
            if s["filled"] > 0:
                print(f"  {direction:5s}: total={s['total']:3d}  filled={s['filled']:3d}  "
                      f"win={s['win_rate']:.3f}  "
                      f"CI=[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")

    # By regime
    regimes = sorted({r.get("regime") or "UNKNOWN" for r in strong})
    if len(regimes) > 1 and strong:
        print("\n── STRONG BY REGIME ──")
        for regime in regimes:
            sub = [r for r in strong if (r.get("regime") or "UNKNOWN") == regime]
            s = summarize_group(sub, f"Strong-{regime}")
            if s["filled"] > 0:
                print(f"  {regime:16s}: total={s['total']:3d}  filled={s['filled']:3d}  "
                      f"win={s['win_rate']:.3f}  "
                      f"CI=[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")

    # Verdict for Strong
    print("\n" + "=" * 70)
    print("VERDICT (Strong signals vs 95% target)")
    print("=" * 70)
    s = results.get("strong", {})
    if s.get("filled", 0) < 10:
        print(f"  [INSUFFICIENT] Only {s.get('filled', 0)} filled Strong signals.")
        print("                 Need >= 10 for any read, >= 30 for a CI tight enough to act on.")
    elif s["ci_hi"] < 0.95:
        print(f"  [BELOW TARGET] Win rate {s['win_rate']:.1%} with 95% CI upper bound "
              f"{s['ci_hi']:.1%}")
        print(f"                 CI is entirely below 95% target. Model needs investigation.")
    elif s["ci_lo"] >= 0.95:
        print(f"  [ON TARGET] Win rate {s['win_rate']:.1%} with 95% CI lower bound "
              f"{s['ci_lo']:.1%} >= 95%")
    else:
        print(f"  [UNCLEAR] Win rate {s['win_rate']:.1%}, CI [{s['ci_lo']:.1%}, {s['ci_hi']:.1%}]")
        print(f"            CI spans the 95% target — need more samples to conclude.")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Convert datetimes for JSON
    for r in rows:
        if isinstance(r.get("signal_time"), datetime):
            r["signal_time"] = r["signal_time"].isoformat()
    OUT_JSON.write_text(json.dumps({
        **results,
        "n_signals_checked": len(rows),
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
    }, indent=2, default=str))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
