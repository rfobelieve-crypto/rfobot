"""
Paper-trading robustness check.

The baseline output flagged some sub-segments as positive-EV (Strong/CHOPPY
91.8% WR, Strong/conf=Q4 etc).  Before treating those as real edges we have
to rule out three artifact paths:

  1. Sample-size noise — 61-trade slices can show extreme WR by luck alone.
     Test: bootstrap 95% CI on net_bps; if CI lower bound > 0 the edge is
     statistically defensible.

  2. Time concentration — an "edge" that only exists for one month and never
     repeats is unusable.  Test: split the slice 50/50 by time and check both
     halves are positive (and roughly comparable).

  3. Model-version artifact — production has been retrained multiple times
     (4/9 prune, 4/19 mag retrain, 5/1 dual retrain).  An edge that only
     existed for one model version is ephemeral.  Test: per-version slice.

Also reports CHOPPY-regime time distribution (was the regime label even
recorded for the first half of the dataset?).
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_trading_baseline import fetch_signals, annotate_pnl, COST_BPS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def boot_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Bootstrap 95% CI on the mean of a 1-D array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 5:
        return np.nan, np.nan
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def slice_stats(group: pd.DataFrame, label: str) -> dict:
    if len(group) < 5:
        return None
    e_net = group["endpoint_net"].values
    lo, hi = boot_ci(e_net)
    return {
        "slice": label,
        "n": len(group),
        "wr": float(group["win_endpoint"].mean()),
        "net_bps": float(e_net.mean() * 10000),
        "ci_lo_bps": lo * 10000,
        "ci_hi_bps": hi * 10000,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
    }


def check_segment(df: pd.DataFrame, mask: pd.Series, name: str):
    """Run all three robustness tests on a single segment."""
    seg = df[mask].copy().sort_values("signal_time").reset_index(drop=True)
    n = len(seg)
    print(f"\n{'='*88}")
    print(f"SEGMENT: {name}  (n={n})")
    print(f"{'='*88}")
    if n < 20:
        print(f"  Sample too small to robustness-check meaningfully.")
        return

    overall = slice_stats(seg, "overall")
    print(f"  overall:  WR={overall['wr']*100:>5.1f}%  net={overall['net_bps']:+6.1f} bps  "
          f"95%CI=[{overall['ci_lo_bps']:+.1f}, {overall['ci_hi_bps']:+.1f}]  "
          f"{'**robust**' if overall['ci_excludes_zero'] and overall['net_bps']>0 else '**not robust** ' if overall['net_bps']>0 else ''}")

    # --- 1. Time split (50/50) ---
    mid = len(seg) // 2
    h1, h2 = seg.iloc[:mid], seg.iloc[mid:]
    s1 = slice_stats(h1, "first half")
    s2 = slice_stats(h2, "second half")
    print(f"\n  Time split:")
    print(f"    1st half ({h1['signal_time'].min().date()} → {h1['signal_time'].max().date()}):  "
          f"n={s1['n']}, WR={s1['wr']*100:>5.1f}%, net={s1['net_bps']:+6.1f} bps "
          f"CI=[{s1['ci_lo_bps']:+.1f}, {s1['ci_hi_bps']:+.1f}]")
    print(f"    2nd half ({h2['signal_time'].min().date()} → {h2['signal_time'].max().date()}):  "
          f"n={s2['n']}, WR={s2['wr']*100:>5.1f}%, net={s2['net_bps']:+6.1f} bps "
          f"CI=[{s2['ci_lo_bps']:+.1f}, {s2['ci_hi_bps']:+.1f}]")

    consistent = (
        (s1["net_bps"] > 0) == (s2["net_bps"] > 0)
        and abs(s1["net_bps"] - s2["net_bps"]) < max(abs(overall["net_bps"]), 5) * 2
    )
    print(f"    consistency: {'consistent (same sign, similar magnitude)' if consistent else 'inconsistent (concerning)'}")

    # --- 2. Per-model-version split ---
    print(f"\n  By model_version:")
    for ver, ssg in seg.groupby("model_version", dropna=False, sort=True):
        if len(ssg) < 5:
            continue
        s = slice_stats(ssg, f"v={ver}")
        print(f"    {str(ver)[:25]:<25}  n={s['n']:>4}, WR={s['wr']*100:>5.1f}%, "
              f"net={s['net_bps']:+6.1f} bps "
              f"CI=[{s['ci_lo_bps']:+.1f}, {s['ci_hi_bps']:+.1f}]")


def regime_history(df: pd.DataFrame):
    """Show when each regime label first appeared (catch missing-data periods)."""
    print(f"\n{'='*88}")
    print("REGIME LABEL HISTORY (when did each regime label start being recorded?)")
    print(f"{'='*88}")
    by_reg = df.groupby("regime", dropna=False).agg(
        first=("signal_time", "min"),
        last=("signal_time", "max"),
        n=("signal_time", "count"),
    ).sort_values("first")
    print(by_reg.to_string())


def main():
    logger.info("Loading signals...")
    df = fetch_signals()
    df = annotate_pnl(df)
    logger.info("n=%d, range %s ~ %s", len(df),
                df["signal_time"].min(), df["signal_time"].max())

    print(f"\nCost model: {COST_BPS:.1f} bps round-trip")

    # Run robustness on the segments that looked interesting in baseline
    segments = [
        ("Strong / CHOPPY (golden segment from baseline)",
         (df["strength"] == "Strong") & (df["regime"] == "CHOPPY")),
        ("Strong / conf=Q4 (top quintile but not extreme)",
         (df["strength"] == "Strong") & (
             df["confidence"].between(
                 df.loc[df["strength"] == "Strong", "confidence"].quantile(0.6),
                 df.loc[df["strength"] == "Strong", "confidence"].quantile(0.8),
             )
         )),
        ("Strong overall",
         df["strength"] == "Strong"),
        ("Moderate / UP (the big drain from baseline)",
         (df["strength"] == "Moderate") & (df["direction"] == "UP")),
        ("ALL signals (the broadest view)",
         pd.Series([True] * len(df), index=df.index)),
    ]

    for name, mask in segments:
        check_segment(df, mask, name)

    regime_history(df)


if __name__ == "__main__":
    main()
