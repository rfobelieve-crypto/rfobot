"""
Power-test for time-series (weekly) momentum on MULTI-YEAR BTC klines.

Phase-0 (5.5mo) showed a suggestive but underpowered weekly-momentum signal
(trailing 168h return -> forward IC rising to +0.27 at 168h, but only ~23
independent windows -> CI included 0). This script answers the real question
with years of data spanning bull AND bear:

  Is weekly time-series momentum a REAL, STABLE, DRIFT-INDEPENDENT long signal?

Discipline (mirrors mistake.md):
  - Overlap: forward-H returns overlap; significance via moving-block bootstrap
    (block = H).
  - DRIFT confound: multi-year BTC has large positive drift, so "long when
    mom>0 wins" can be pure beta. The momentum EDGE is measured as the SPREAD
        spread_H = mean(fwd_H | mom>0) - mean(fwd_H | mom<0)
    which is drift-neutral. Raw long WR is reported but is NOT the edge metric.
  - REGIME: split bull/bear by 30d trailing return sign; momentum-long must
    survive in BOTH (else it is just beta dressed as signal).
  - STABILITY: per-year IC of the headline signal; real momentum is positive in
    most years, not driven by one bull run (the WQ101 outlier lesson).

Pure price signal -> uses long Binance klines only (no Coinglass), independent
of the 4h order-flow model.

Output: research/results/dual_model/ts_momentum_powertest.csv
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

from research.dual_model.shared_data import _fetch_klines_paginated, RESULTS_DIR, CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

N_BARS = 35000          # ~4 years of 1h bars
HORIZONS = [24, 48, 72, 120, 168, 336]      # 1d .. 2w
LOOKBACKS = [72, 120, 168, 336]             # 3d .. 2w trailing momentum
EPS = 1e-12


def fwd_return(close, H):
    n = len(close); out = np.full(n, np.nan)
    out[: n - H] = close[H:] / close[: n - H] - 1.0
    return out


def trail_mom(close, L):
    n = len(close); out = np.full(n, np.nan)
    out[L:] = close[L:] / close[: n - L] - 1.0
    return out


def block_boot(stat_fn, *arrays, block, n_boot=2000, seed=42):
    """Generic moving-block bootstrap CI for a statistic over aligned arrays."""
    rng = np.random.default_rng(seed)
    m = np.all([np.isfinite(a) for a in arrays], axis=0)
    arrays = [a[m] for a in arrays]
    n = len(arrays[0])
    block = max(2, min(block, n // 3))
    if n < block * 3:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(*arrays)
    nb = int(np.ceil(n / block))
    pool = np.arange(0, n - block + 1)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(pool, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        v = stat_fn(*[a[idx] for a in arrays])
        boots[b] = v if np.isfinite(v) else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _ic(x, y):
    r = spearmanr(x, y).correlation
    return r if np.isfinite(r) else 0.0


def _spread(mom, fwd):
    up = mom > 0; dn = mom < 0
    if up.sum() < 5 or dn.sum() < 5:
        return 0.0
    return float(fwd[up].mean() - fwd[dn].mean())


def load_klines():
    cache = CACHE_DIR / "klines_longhist.parquet"
    if cache.exists():
        k = pd.read_parquet(cache)
        logger.info("Loaded cached long klines: %d bars (%s ~ %s)",
                    len(k), k.index[0], k.index[-1])
        return k
    logger.info("Fetching %d 1h klines from Binance (paginated)…", N_BARS)
    k = _fetch_klines_paginated(N_BARS)
    k.to_parquet(cache)
    logger.info("Fetched & cached: %d bars (%s ~ %s)", len(k), k.index[0], k.index[-1])
    return k


def main() -> int:
    k = load_klines()
    close = k["close"].values.astype(float)
    idx = k.index
    fwd = {H: fwd_return(close, H) for H in HORIZONS}
    mom = {L: trail_mom(close, L) for L in LOOKBACKS}

    rows = []

    # ---------- T1: powered momentum IC grid (point estimates) ----------
    print("\n" + "=" * 96)
    print(f"T1  MOMENTUM IC GRID  (n={len(k)} bars, ~{len(k)/8760:.1f} yr)  "
          "rows=lookback L, cols=horizon H")
    print("=" * 96)
    print("  L\\H  " + "".join(f"{H:>11d}h" for H in HORIZONS))
    best_cell = None
    for L in LOOKBACKS:
        cells = []
        for H in HORIZONS:
            ic = _ic(*[a[np.isfinite(mom[L]) & np.isfinite(fwd[H])]
                       for a in (mom[L], fwd[H])])
            cells.append(f"{ic:>+10.3f} ")
            rows.append(dict(test="T1_ic", L=L, H=H, ic=ic))
            if best_cell is None or ic > best_cell[2]:
                best_cell = (L, H, ic)
        print(f"  {L:>3d}  " + "".join(cells))
    print(f"\n  strongest cell: L={best_cell[0]} H={best_cell[1]} IC={best_cell[2]:+.3f}")

    # ---------- T2: bootstrap CI on headline cells (L=168) ----------
    print("\n" + "=" * 96)
    print("T2  SIGNIFICANCE (moving-block bootstrap, block=H) for L=168 momentum")
    print("=" * 96)
    L0 = 168
    for H in HORIZONS:
        ic, lo, hi = block_boot(_ic, mom[L0], fwd[H], block=H, n_boot=1500)
        sp, slo, shi = block_boot(_spread, mom[L0], fwd[H], block=H, n_boot=1500)
        ic_sig = "*" if lo * hi > 0 else " "
        sp_sig = "*" if slo * shi > 0 else " "
        rows.append(dict(test="T2", L=L0, H=H, ic=ic, ic_lo=lo, ic_hi=hi,
                         spread=sp, sp_lo=slo, sp_hi=shi))
        print(f"  H={H:>4d}h  IC={ic:+.3f} CI[{lo:+.3f},{hi:+.3f}]{ic_sig}   "
              f"spread(up-dn)={sp*100:+.3f}% CI[{slo*100:+.2f},{shi*100:+.2f}]%{sp_sig}")
    print("  (spread = drift-neutral momentum edge; * = CI excludes 0)")

    # ---------- T3: per-year stability of headline signal ----------
    H0 = 168
    print("\n" + "=" * 96)
    print(f"T3  PER-YEAR STABILITY  signal=trail{L0}h momentum vs fwd{H0}h  "
          "(real momentum = positive most years)")
    print("=" * 96)
    yr = idx.year
    print(f"  {'year':>6s} {'n':>7s} {'IC':>8s} {'spread%':>9s} {'long_WR%':>9s} {'drift%':>8s}")
    pos_years = 0; tot_years = 0
    for y in sorted(set(yr)):
        sel = yr == y
        mm, ff = mom[L0][sel], fwd[H0][sel]
        msk = np.isfinite(mm) & np.isfinite(ff)
        if msk.sum() < 200:
            continue
        mm, ff = mm[msk], ff[msk]
        ic = _ic(mm, ff)
        sp = _spread(mm, ff)
        up = mm > 0
        lwr = float((ff[up] > 0).mean()) if up.sum() else float("nan")
        drift = float(ff.mean())
        tot_years += 1; pos_years += int(sp > 0)
        rows.append(dict(test="T3_year", year=int(y), n=int(msk.sum()),
                         ic=ic, spread=sp, long_wr=lwr, drift=drift))
        print(f"  {y:>6d} {int(msk.sum()):>7d} {ic:>+8.3f} {sp*100:>+8.2f}% "
              f"{lwr*100:>8.1f}% {drift*100:>+7.2f}%")
    print(f"\n  spread>0 in {pos_years}/{tot_years} years")

    # ---------- T4: regime split (bull vs bear by 30d trailing) ----------
    print("\n" + "=" * 96)
    print("T4  REGIME SPLIT  — does momentum-long survive in BEAR, or is it just beta?")
    print("=" * 96)
    reg = trail_mom(close, 720)        # 30d trailing return as regime proxy
    for label, mask in (("BULL (30d ret>0)", reg > 0), ("BEAR (30d ret<0)", reg < 0)):
        mm, ff = mom[L0], fwd[H0]
        msk = mask & np.isfinite(mm) & np.isfinite(ff)
        mm2, ff2 = mm[msk], ff[msk]
        sp, slo, shi = block_boot(_spread, mm2, ff2, block=H0, n_boot=1500)
        up = mm2 > 0
        lwr = float((ff2[up] > 0).mean()) if up.sum() else float("nan")
        sig = "*" if slo * shi > 0 else " "
        rows.append(dict(test="T4_regime", regime=label, n=int(msk.sum()),
                         spread=sp, sp_lo=slo, sp_hi=shi, long_wr=lwr))
        print(f"  {label:>18s}  n={int(msk.sum()):>6d}  "
              f"spread={sp*100:+.3f}% CI[{slo*100:+.2f},{shi*100:+.2f}]%{sig}  "
              f"long_WR={lwr*100:.1f}%")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ts_momentum_powertest.csv", index=False)

    # ---------- verdict ----------
    t2 = [r for r in rows if r["test"] == "T2"]
    sig_spread = [r for r in t2 if r.get("sp_lo", 0) * r.get("sp_hi", 0) > 0
                  and r["spread"] > 0]
    t4 = [r for r in rows if r["test"] == "T4_regime"]
    bear = next((r for r in t4 if "BEAR" in r["regime"]), None)
    bear_ok = bear and (bear["sp_lo"] * bear["sp_hi"] > 0) and bear["spread"] > 0
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  drift-neutral momentum spread significant & positive at horizons: "
          f"{[r['H'] for r in sig_spread] or 'NONE'}")
    print(f"  positive spread in {pos_years}/{tot_years} years")
    print(f"  survives in BEAR regime: {'YES' if bear_ok else 'NO'}")
    if sig_spread and pos_years >= max(2, int(0.6 * tot_years)) and bear_ok:
        print("\n  >>> GO: weekly TS-momentum is a real, stable, drift-independent")
        print("      long signal. A longer-horizon trend overlay for the long side")
        print("      is justified — build & validate as an independent model.")
    elif sig_spread and pos_years >= max(2, int(0.6 * tot_years)):
        print("\n  >>> PARTIAL: momentum edge is real & stable but BETA-LIKE")
        print("      (fails/weak in bear). Usable only as a bull-regime long filter,")
        print("      not a standalone long-trend alpha. Marginal value over tier-scaling.")
    else:
        print("\n  >>> NO-GO: weekly momentum is not stable/drift-independent enough.")
        print("      The Phase-0 hint was regime/outlier-driven. Trend door closed;")
        print("      long-side answer remains tier-scaling sizing.")
    print("=" * 96)
    print(f"\nWrote → {RESULTS_DIR / 'ts_momentum_powertest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
