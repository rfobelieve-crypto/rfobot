"""
Ablation: Initiation prob filter vs MAG percentile filter — how do they
interact on the walk-forward OOS set?

Reuses research/results/initiation_replay_oos.parquet (dumped by
initiation_replay.py). Reports Strong win rate across a grid of
(init_top_frac, mag_percentile_min) + breakout requirement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from indicator.initiation_features import add_initiation_features

OOS_PATH = PROJECT_ROOT / "research" / "results" / "initiation_replay_oos.parquet"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def side_gate(oos: pd.DataFrame, side: str, top_frac: float,
              mag_min: float, require_bo: bool) -> tuple[int, int]:
    """Return (wins, fires) for one side."""
    prob_col = "p_long" if side == "long" else "p_short"
    y_col = "y_long" if side == "long" else "y_short"
    bo_col = "bo_up" if side == "long" else "bo_dn"

    thr = oos[prob_col].quantile(1 - top_frac)
    mask = oos[prob_col] >= thr
    if mag_min > 0:
        mask &= oos["mag_pct"] >= mag_min
    if require_bo:
        mask &= oos[bo_col] == 1.0

    fires = int(mask.sum())
    if fires == 0:
        return 0, 0
    wins = int(oos.loc[mask, y_col].sum())
    return wins, fires


def main():
    # Load OOS predictions
    oos = pd.read_parquet(OOS_PATH)
    print(f"OOS: {len(oos)} bars")

    # Need bo_up / bo_dn — recompute from cache
    df = load_and_cache_data()
    df = add_initiation_features(df)
    oos["bo_up"] = df.loc[oos.index, "init_bo_up"].fillna(0).values
    oos["bo_dn"] = df.loc[oos.index, "init_bo_dn"].fillna(0).values

    print(f"mag_pct coverage: {oos['mag_pct'].notna().mean()*100:.1f}%")
    print(f"mag_pct mean: {oos['mag_pct'].mean():.3f}\n")

    # --- Grid search -------------------------------------------------------
    top_fracs = [0.03, 0.05, 0.10]
    mag_mins = [0.0, 0.50, 0.65, 0.80]

    print("=" * 95)
    print("  INIT prob top-k × MAG percentile min (with breakout required)")
    print("  Reports: combined WR  |  n fires  |  Wilson CI")
    print("=" * 95)
    print(f"{'top_k':<8}{'mag_min':<10}{'n':>6}{'WR':>10}{'CI':>20}"
          f"{'long_n':>10}{'short_n':>10}")
    print("-" * 95)

    rows = []
    for tf in top_fracs:
        for mm in mag_mins:
            lw, ln = side_gate(oos, "long", tf, mm, require_bo=True)
            sw, sn = side_gate(oos, "short", tf, mm, require_bo=True)
            tw = lw + sw
            tn = ln + sn
            if tn == 0:
                wr = float("nan")
                ci = (0, 0)
            else:
                wr = tw / tn
                ci = wilson_ci(tw, tn)
            rows.append(dict(top_k=tf, mag_min=mm, n=tn, wins=tw, wr=wr,
                              ci_lo=ci[0], ci_hi=ci[1], long_n=ln, short_n=sn))
            ci_str = f"[{ci[0]*100:4.1f}, {ci[1]*100:4.1f}]"
            wr_str = f"{wr*100:5.1f}%" if not np.isnan(wr) else "  ---"
            print(f"{tf*100:>4.0f}%   {mm:<10.2f}{tn:>6}{wr_str:>10}"
                  f"{ci_str:>20}{ln:>10}{sn:>10}")

    # --- Isolate each filter's marginal lift -------------------------------
    print("\n" + "=" * 75)
    print("  Marginal lift breakdown (baseline → + each filter)")
    print("=" * 75)

    # Baseline: no filters (just pos rate)
    n_long_total = len(oos)
    long_pos = oos["y_long"].mean()
    short_pos = oos["y_short"].mean()
    print(f"  Base pos rate:          long={long_pos*100:.2f}%  short={short_pos*100:.2f}%")

    # + init prob top-5% only (no breakout, no mag)
    lw, ln = side_gate(oos, "long", 0.05, 0.0, require_bo=False)
    sw, sn = side_gate(oos, "short", 0.05, 0.0, require_bo=False)
    wr = (lw + sw) / (ln + sn)
    print(f"  + init top-5% only:     WR={wr*100:.1f}%  n={ln+sn}  "
          f"(long {lw}/{ln}={lw/ln*100:.1f}%  short {sw}/{sn}={sw/sn*100:.1f}%)")

    # + init top-5% + breakout
    lw, ln = side_gate(oos, "long", 0.05, 0.0, require_bo=True)
    sw, sn = side_gate(oos, "short", 0.05, 0.0, require_bo=True)
    wr = (lw + sw) / max(ln + sn, 1)
    print(f"  + breakout:             WR={wr*100:.1f}%  n={ln+sn}")

    # + init top-5% + breakout + mag>=0.65
    lw, ln = side_gate(oos, "long", 0.05, 0.65, require_bo=True)
    sw, sn = side_gate(oos, "short", 0.05, 0.65, require_bo=True)
    wr = (lw + sw) / max(ln + sn, 1)
    print(f"  + mag_pct>=0.65:        WR={wr*100:.1f}%  n={ln+sn}")

    # + init top-5% + breakout + mag>=0.80
    lw, ln = side_gate(oos, "long", 0.05, 0.80, require_bo=True)
    sw, sn = side_gate(oos, "short", 0.05, 0.80, require_bo=True)
    wr = (lw + sw) / max(ln + sn, 1)
    print(f"  + mag_pct>=0.80:        WR={wr*100:.1f}%  n={ln+sn}  ← current Strong")

    # --- MAG-only baseline: does MAG alone pick winners? -------------------
    print("\n" + "=" * 75)
    print("  MAG-only filter (no init prob — is MAG predictive alone?)")
    print("=" * 75)
    for mm in [0.65, 0.80, 0.90]:
        mask_long = (oos["mag_pct"] >= mm) & (oos["bo_up"] == 1.0)
        mask_short = (oos["mag_pct"] >= mm) & (oos["bo_dn"] == 1.0)
        ln = int(mask_long.sum())
        sn = int(mask_short.sum())
        if ln + sn == 0:
            continue
        lw = int(oos.loc[mask_long, "y_long"].sum())
        sw = int(oos.loc[mask_short, "y_short"].sum())
        wr = (lw + sw) / (ln + sn)
        print(f"  mag_pct>={mm:.2f} + breakout:  "
              f"WR={wr*100:.1f}%  n={ln+sn}  "
              f"(long {lw}/{ln}={(lw/ln*100) if ln else 0:.0f}%  "
              f"short {sw}/{sn}={(sw/sn*100) if sn else 0:.0f}%)")


if __name__ == "__main__":
    main()
