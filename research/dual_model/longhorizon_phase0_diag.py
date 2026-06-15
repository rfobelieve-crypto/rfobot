"""
Phase-0 market-structure diagnostic for a LONGER-HORIZON / TREND model
aimed at improving LONG-side odds.

Documented premise (memory: project_short_is_the_edge, onchain_etf_overlay_nogo):
  The 4h Dual Model is a MEAN-REVERSION specialist. Short side is the edge
  (+115bps/70% WR). Long side is weak (+23bps/46%) and the root cause is the
  4h HORIZON (entering counter-trend longs that mean-revert), not missing data.

Before building any longer-horizon model (days of work, may NO-GO), answer the
3 cheap questions that decide whether the premise even holds in this data:

  Q1  AUTOCORRELATION FLIP: across horizons H, does trailing momentum
      (close[t]/close[t-L]-1) predict forward return with POSITIVE IC at some
      H? 4h shows negative IC (mean-reversion). If no H turns positive, a trend
      model has nothing to learn -> NO-GO.

  Q2  LONG-SIDE PAYOFF: when trailing momentum > 0 (uptrend), does holding
      longer pay? Win-rate and mean forward return of the long subset by H.

  Q3  (decisive) DOES V7 ALREADY KNOW — JUST HELD TOO SHORT? Take V7's existing
      UP predictions and measure win-rate / mean return at 4h vs 12/24/48/72h.
      If V7's UP calls pay BETTER when held longer, the cheapest fix is a longer
      HOLD on existing longs — no new model needed.

Leakage / power discipline:
  - Forward returns at horizon H overlap (consecutive bars share H-1 of H steps).
    Significance uses MOVING-BLOCK bootstrap with block = H (respects overlap).
  - Independent-sample count printed per H so low-power horizons are visible
    (e.g. 168h -> ~24 non-overlapping samples; read its CI, not its point est).
  - All predictors strictly trailing; all targets strictly forward.

Output: research/results/dual_model/longhorizon_phase0.csv
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, RESULTS_DIR
from research.dual_model.direction_features_v2 import FULL_DIRECTION
from research.dual_model.train_direction_reg_4h import train_direction_reg_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [4, 8, 12, 24, 48, 72, 120, 168]      # 1h bars: 4h .. 1 week
LOOKBACKS = [12, 24, 48, 72, 168]                # trailing momentum windows
EPS = 1e-12


def fwd_return(close: np.ndarray, H: int) -> np.ndarray:
    """Endpoint forward return close[t+H]/close[t]-1 (NaN for last H bars)."""
    n = len(close)
    out = np.full(n, np.nan)
    out[: n - H] = close[H:] / close[: n - H] - 1.0
    return out


def trailing_mom(close: np.ndarray, L: int) -> np.ndarray:
    """Trailing momentum close[t]/close[t-L]-1 (NaN for first L bars)."""
    n = len(close)
    out = np.full(n, np.nan)
    out[L:] = close[L:] / close[: n - L] - 1.0
    return out


def block_boot_ic(x: np.ndarray, y: np.ndarray, block: int,
                  n_boot: int = 2000, seed: int = 42):
    """Moving-block bootstrap CI for Spearman IC. block should be >= horizon."""
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    block = max(2, min(block, n // 3))
    if n < block * 3:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(spearmanr(x, y).correlation)
    nb = int(np.ceil(n / block))
    pool = np.arange(0, n - block + 1)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(pool, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r = spearmanr(x[idx], y[idx]).correlation
        boots[b] = r if np.isfinite(r) else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), n // block      # ~indep samples


def long_subset_stats(signal: np.ndarray, fwd: np.ndarray):
    """Stats for the LONG subset (signal>0): n, win-rate, mean fwd return."""
    m = np.isfinite(signal) & np.isfinite(fwd)
    s, f = signal[m], fwd[m]
    up = s > 0
    n_up = int(up.sum())
    if n_up == 0:
        return dict(n=0, wr=float("nan"), mean_ret=float("nan"))
    return dict(n=n_up, wr=float((f[up] > 0).mean()),
                mean_ret=float(f[up].mean()))


def main() -> int:
    df = load_and_cache_data(limit=4000)
    close = df["close"].values.astype(float)
    idx = df.index
    logger.info("Loaded %d bars (%s ~ %s)", len(df), idx[0], idx[-1])

    fwd = {H: fwd_return(close, H) for H in HORIZONS}
    mom = {L: trailing_mom(close, L) for L in LOOKBACKS}

    # ---------- Q1: autocorrelation flip (mom_L -> fwd_H IC grid) ----------
    print("\n" + "=" * 100)
    print("Q1  MOMENTUM IC GRID  rows=trailing lookback L, cols=forward horizon H")
    print("    (4h baseline expected NEGATIVE = mean-reversion; look for sign flip to + at larger H)")
    print("=" * 100)
    header = "  L\\H  " + "".join(f"{H:>11d}h" for H in HORIZONS)
    print(header)
    q1_rows = []
    for L in LOOKBACKS:
        cells = []
        for H in HORIZONS:
            pt, lo, hi, nind = block_boot_ic(mom[L], fwd[H], block=H)
            sig = "*" if (np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0) else " "
            cells.append(f"{pt:>+9.3f}{sig} ")
            q1_rows.append(dict(kind="mom_ic", L=L, H=H, ic=pt, ci_lo=lo,
                                ci_hi=hi, ci_excl0=(lo * hi > 0
                                if np.isfinite(lo) and np.isfinite(hi) else False),
                                n_indep=nind))
        print(f"  {L:>3d}   " + "".join(cells))
    print("  (* = 95% block-bootstrap CI excludes 0)")

    # ---------- Q2: long-side payoff by horizon ----------
    print("\n" + "=" * 100)
    print("Q2  LONG-SIDE PAYOFF  — when trailing momentum L>0 (uptrend), hold H hours")
    print("=" * 100)
    print(f"  {'mom_L':>6s} {'H':>5s} {'n_up':>7s} {'win%':>8s} {'mean_ret':>11s}")
    q2_rows = []
    for L in (24, 48, 72):
        for H in HORIZONS:
            st = long_subset_stats(mom[L], fwd[H])
            q2_rows.append(dict(kind="long_payoff", L=L, H=H, **st))
            if st["n"]:
                print(f"  {L:>6d} {H:>5d} {st['n']:>7d} {st['wr']*100:>7.1f}% "
                      f"{st['mean_ret']*100:>+10.3f}%")
        print("  " + "-" * 44)

    # ---------- Q3 (decisive): does V7 just need to hold longer? ----------
    logger.info("Q3: regenerating V7 OOS predictions (walk-forward, ~1 min)…")
    oos, metrics, _ = train_direction_reg_walk_forward(
        df, FULL_DIRECTION, objective="mse")
    oos.to_parquet(RESULTS_DIR / "v7_oos_for_horizon.parquet")
    logger.info("V7 OOS IC=%+.4f AUC=%.4f n=%d",
                metrics["spearman_ic"], metrics["auc_sign"], len(oos))

    # align V7 pred to forward returns at every horizon on the SAME bars
    fwd_df = pd.DataFrame({f"fwd_{H}": fwd[H] for H in HORIZONS}, index=idx)
    j = oos[["pred_ret"]].join(fwd_df, how="left")
    pred = j["pred_ret"].values

    # UP subset = top tiers of pred_ret (mirror live: Strong/Moderate longs)
    thr_strong = np.nanpercentile(pred, 95)   # top 5% UP ~ Strong long
    thr_mod = np.nanpercentile(pred, 85)      # top 15% UP ~ Moderate+ long
    print("\n" + "=" * 100)
    print("Q3  V7 EXISTING UP-CALLS — win-rate & mean return when HELD LONGER")
    print(f"    Strong-long thr(pred>= p95)={thr_strong:+.5f}   "
          f"Moderate+long thr(pred>= p85)={thr_mod:+.5f}")
    print("=" * 100)
    print(f"  {'tier':>10s} {'H':>5s} {'n':>6s} {'win%':>8s} {'mean_ret':>11s} "
          f"{'IC(all pred)':>13s}")
    q3_rows = []
    for tier, thr in (("ALL", -np.inf), ("Mod+ p85", thr_mod),
                      ("Strong p95", thr_strong)):
        sel = pred >= thr if np.isfinite(thr) else np.ones_like(pred, bool)
        for H in HORIZONS:
            f = j[f"fwd_{H}"].values
            m = sel & np.isfinite(f)
            n = int(m.sum())
            if n < 5:
                continue
            wr = float((f[m] > 0).mean())
            mr = float(f[m].mean())
            # IC of full pred vs this horizon's forward return (overlap-aware)
            ic_pt, ic_lo, ic_hi, _ = block_boot_ic(pred, j[f"fwd_{H}"].values,
                                                   block=H)
            ic_sig = "*" if (np.isfinite(ic_lo) and ic_lo * ic_hi > 0) else " "
            q3_rows.append(dict(kind="v7_hold", tier=tier, H=H, n=n, wr=wr,
                                mean_ret=mr, full_ic=ic_pt, ic_excl0=(ic_lo*ic_hi>0
                                if np.isfinite(ic_lo) and np.isfinite(ic_hi) else False)))
            print(f"  {tier:>10s} {H:>5d} {n:>6d} {wr*100:>7.1f}% "
                  f"{mr*100:>+10.3f}% {ic_pt:>+12.3f}{ic_sig}")
        print("  " + "-" * 60)

    # ---------- save + verdict ----------
    allrows = q1_rows + q2_rows + q3_rows
    pd.DataFrame(allrows).to_csv(RESULTS_DIR / "longhorizon_phase0.csv", index=False)

    # auto-verdict signals
    mom_flip = [r for r in q1_rows if r["ci_excl0"] and r["ic"] > 0]
    v7_hold = [r for r in q3_rows if r["tier"] == "Strong p95"]
    print("\n" + "=" * 100)
    print("PHASE-0 READ (auto)")
    print("=" * 100)
    if mom_flip:
        print(f"  Q1: momentum turns SIGNIFICANTLY POSITIVE at: "
              f"{sorted(set((r['L'], r['H']) for r in mom_flip))}")
        print("      -> a trend signal EXISTS at longer horizons; trend model has something to learn.")
    else:
        print("  Q1: momentum NEVER significantly positive at any horizon in this sample.")
        print("      -> NO trend autocorrelation; a longer-horizon trend model likely has nothing to learn.")
    if v7_hold:
        best = max(v7_hold, key=lambda r: r["wr"])
        base4 = next((r for r in v7_hold if r["H"] == 4), None)
        if base4:
            print(f"  Q3: V7 Strong-long WR  4h={base4['wr']*100:.1f}%  "
                  f"best={best['wr']*100:.1f}% @ {best['H']}h "
                  f"(meanret {best['mean_ret']*100:+.2f}%)")
            if best["H"] != 4 and best["wr"] - base4["wr"] > 0.03:
                print("      -> V7 longs DO pay better held longer: cheapest win = extend long hold, no new model.")
            else:
                print("      -> Holding V7 longs longer does NOT clearly help.")
    print("=" * 100)
    print(f"\nWrote → {RESULTS_DIR / 'longhorizon_phase0.csv'}")
    print("NOTE: read CIs not point estimates at H>=72 (few independent samples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
