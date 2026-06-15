"""
Top-k precision sweep for direction model.

Loads walk-forward OOS predictions and computes bidirectional top-k precision:
  - confidence = |prob_up - 0.5|
  - direction  = sign(prob_up - 0.5)
  - precision  = fraction of top-k by confidence where direction matches y_true

Output: precision/freq curve across k in {1,2,5,10,20}% with Wilson CI.

Usage:
    python research/topk_precision_sweep.py
    python research/topk_precision_sweep.py --oos direction_oos_full_expanded
    python research/topk_precision_sweep.py --exclude-contaminated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OOS_DIR = PROJECT_ROOT / "research" / "results" / "dual_model"
OUT = PROJECT_ROOT / "research" / "results" / "topk_precision_sweep.json"

# 2026-04-07 ~ 2026-04-12: cg_bfx_margin / coinbase_premium backfill bug window
CONTAMINATED_START = pd.Timestamp("2026-04-07", tz="UTC")
CONTAMINATED_END = pd.Timestamp("2026-04-12 23:59", tz="UTC")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_topk(df: pd.DataFrame, k_frac: float) -> dict:
    """df must have columns: prob_up, y_true (0/1)."""
    df = df.copy()
    df["confidence"] = (df["prob_up"] - 0.5).abs()
    df["pred_dir"] = (df["prob_up"] >= 0.5).astype(int)
    df["correct"] = (df["pred_dir"] == df["y_true"]).astype(int)

    n_total = len(df)
    k = max(1, int(round(n_total * k_frac)))
    top = df.nlargest(k, "confidence")

    n_correct = int(top["correct"].sum())
    precision = n_correct / k
    ci_lo, ci_hi = wilson_ci(n_correct, k)

    # Frequency: signals per month
    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400
    signals_per_month = k / span_days * 30

    # Mean |return| in directional terms
    ret = top["return_4h"].values
    dir_sign = np.where(top["pred_dir"].values == 1, 1, -1)
    dir_ret = (ret * dir_sign).mean()

    # Min confidence in top-k (this is the threshold)
    min_conf = float(top["confidence"].min())
    # Corresponding prob_up threshold for UP signals and DOWN signals
    thr_up = 0.5 + min_conf
    thr_dn = 0.5 - min_conf

    return {
        "k_frac": k_frac,
        "k_count": k,
        "n_total": n_total,
        "precision": precision,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "signals_per_month": signals_per_month,
        "avg_dir_return": float(dir_ret),
        "min_confidence": min_conf,
        "prob_threshold_up": thr_up,
        "prob_threshold_down": thr_dn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", default="direction_oos_full_expanded",
                    help="OOS parquet stem under research/results/dual_model/")
    ap.add_argument("--exclude-contaminated", action="store_true",
                    help="Drop 2026-04-07 ~ 2026-04-12 (bfx_margin/cb_premium bad window)")
    args = ap.parse_args()

    path = OOS_DIR / f"{args.oos}.parquet"
    if not path.exists():
        raise SystemExit(f"OOS file not found: {path}")

    df = pd.read_parquet(path)
    print("=" * 72)
    print(f"TOP-K PRECISION SWEEP  —  {args.oos}")
    print("=" * 72)
    print(f"Loaded: {len(df)} OOS predictions")
    print(f"Range : {df.index.min()}  →  {df.index.max()}")

    if args.exclude_contaminated:
        mask = ~((df.index >= CONTAMINATED_START) & (df.index <= CONTAMINATED_END))
        dropped = (~mask).sum()
        df = df[mask]
        print(f"Excluded {dropped} rows in contaminated window {CONTAMINATED_START.date()} ~ {CONTAMINATED_END.date()}")
        print(f"After  : {len(df)} predictions")

    # Baseline
    overall_acc = float((df["prob_up"] >= 0.5).astype(int).eq(df["y_true"]).mean())
    overall_up_rate = float(df["y_true"].mean())
    print(f"\nBaseline accuracy (all predictions): {overall_acc:.4f}")
    print(f"Base rate P(y=UP)                  : {overall_up_rate:.4f}")

    print()
    print(f"{'k%':>5} {'count':>6} {'prec':>7} {'CI_lo':>7} {'CI_hi':>7} "
          f"{'/month':>8} {'dir_ret':>9} {'min_conf':>9} {'thr_up':>7} {'thr_dn':>7}")
    print("-" * 72)

    results = []
    for k_frac in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        r = compute_topk(df, k_frac)
        results.append(r)
        print(f"{k_frac*100:>4.0f}% {r['k_count']:>6d} "
              f"{r['precision']:>7.3f} {r['ci_lo']:>7.3f} {r['ci_hi']:>7.3f} "
              f"{r['signals_per_month']:>8.1f} "
              f"{r['avg_dir_return']:>+9.4f} "
              f"{r['min_confidence']:>9.4f} "
              f"{r['prob_threshold_up']:>7.3f} {r['prob_threshold_down']:>7.3f}")

    # Verdict
    print()
    print("=" * 72)
    print("VERDICT: smallest k where precision CI_lo >= target")
    print("=" * 72)
    for target in [0.60, 0.65, 0.70, 0.75, 0.80]:
        hit = next((r for r in results if r["ci_lo"] >= target), None)
        if hit:
            print(f"  target {target:.0%}: k={hit['k_frac']:.0%} "
                  f"(prec={hit['precision']:.3f}, {hit['signals_per_month']:.1f}/month)")
        else:
            print(f"  target {target:.0%}: UNREACHABLE at any tested k")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "oos_file": args.oos,
        "exclude_contaminated": args.exclude_contaminated,
        "n_samples": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "baseline_accuracy": overall_acc,
        "base_rate_up": overall_up_rate,
        "sweep": results,
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
