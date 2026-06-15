"""
NOVEL feature family: vol_termstructure
=======================================
DVOL term-structure & vol-risk-premium DYNAMICS, screened for MARGINAL
information the deployed 136-feature V7 model has NOT already captured.

Motivation (2026-06-14 feature search):
  The dvol *level* channel is the ONE channel that passed the existing-feature
  screen. But V7 already carries dvol_close/open/high/low, dvol_ma_24h,
  dvol_change_4h, dvol_change_24h, dvol_zscore_72h, and dvol_rv_spread (the VRP
  *level*). The open question is whether there is DYNAMIC alpha BEYOND the
  collinear level:
    - dvol vs realized-vol spread *dynamics* (momentum / z-score / sign of VRP),
      not just the level already in V7 (dvol_rv_spread)
    - dvol acceleration / curvature (2nd difference of the term structure)
    - dvol * funding interaction  (vol premium vs carry)
    - low-vol compression -> expansion *setup* flags
    - dvol percentile regime

HARD RULES honoured:
  1. Trailing-only. Every rolling window is backward-looking (pandas .rolling /
     .ewm / .shift(+k) / .diff). No .shift(-k), no centered/forward windows.
  2. No TA indicators (MACD/EMA/RSI/Bollinger/Stoch). We use raw vol-index
     statistics, funding, realized vol, and price-behavior stats only. (.ewm
     spans used purely as a robust trailing mean of the *vol index*, not as a
     price-trend MACD/EMA crossover signal.)
  3. No feat*sparse_indicator with nonzero fraction < 30%. Setup flags are built
     to have >30% support OR are combined as feat*(1-is_dead_regime).
  4. Daily alt-data is already lagged in the parquet; we build nothing daily and
     never peek same-day on etf_/fg_.

The screen is screen_candidates() from featsearch_lib (identical methodology
across families): conditional IC vs the CLEAN V7 OOS residual, per-fold sign
consistency, max |corr| to any existing column, nonzero fraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import load_features, screen_candidates  # noqa: E402

RESULTS = PROJECT_ROOT / "research" / "results"
OUT_PARQUET = RESULTS / "newfeat_vol_termstructure.parquet"

# bars per year for annualising realized vol (1h bars): 24 * 365
ANN = np.sqrt(24 * 365) * 100.0


def _safe_z(x: pd.Series, win: int, minp: int) -> pd.Series:
    r = x.rolling(win, min_periods=minp)
    mu = r.mean()
    sd = r.std()
    return (x - mu) / sd.replace(0.0, np.nan)


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    dvol = df["dvol_close"].astype(float)
    dvol_hi = df["dvol_high"].astype(float)
    dvol_lo = df["dvol_low"].astype(float)
    rv = df["realized_vol_20b"].astype(float)
    funding = df["cg_funding_close"].astype(float)
    close = df["close"].astype(float)

    # annualised realized vol, same convention as the existing dvol_rv_spread
    rv_ann = rv * ANN
    # the VRP level that V7 ALREADY has (rebuilt locally so we can take dynamics
    # of it WITHOUT exporting the level itself)
    vrp = dvol - rv_ann

    cand = pd.DataFrame(index=df.index)

    # -------------------------------------------------------------------
    # GROUP A — VRP (dvol-rv spread) DYNAMICS, beyond the collinear level
    # -------------------------------------------------------------------
    # A1. VRP momentum: change in the vol-risk-premium over 8h. Captures the
    #     spread *compressing* (fear leaving) or *widening* (fear building),
    #     orthogonal to the V7-held VRP level.
    cand["vts_vrp_mom_8h"] = vrp.diff(8)

    # A2. VRP z-score over a 72h trailing window — is the current premium
    #     stretched relative to its own recent regime? (level-detrended)
    cand["vts_vrp_z_72h"] = _safe_z(vrp, 72, 24)

    # A3. VRP sign-flip persistence: how long has VRP held its current sign
    #     (positive premium = options over realized). A regime-persistence
    #     state feature, cumulative-from-past.
    sign = np.sign(vrp.fillna(0.0))
    grp = (sign != sign.shift(1)).cumsum()
    cand["vts_vrp_sign_streak"] = (
        sign.groupby(grp).cumcount() + 1
    ) * sign  # signed run length

    # A4. realized-vol catching up to implied: ratio of RV momentum to DVOL
    #     momentum over 12h. >1 => realized accelerating faster than implied
    #     (premium about to invert), trailing-only.
    rv_mom = rv_ann.diff(12)
    dv_mom = dvol.diff(12)
    cand["vts_rv_vs_dvol_mom_12h"] = rv_mom - dv_mom

    # -------------------------------------------------------------------
    # GROUP B — DVOL CURVATURE / ACCELERATION (2nd-order term structure)
    # -------------------------------------------------------------------
    # B1. DVOL acceleration: 2nd difference of the vol index over 4h steps.
    #     Convexity of the implied-vol path, not captured by V7's 1st-diff
    #     dvol_change_4h.
    cand["vts_dvol_accel_4h"] = dvol.diff(4) - dvol.diff(4).shift(4)

    # B2. DVOL term-structure slope curvature: (short MA - long MA) - prior,
    #     using purely trailing means of the vol index. Short=6h, long=24h.
    ts_slope = dvol.rolling(6, min_periods=3).mean() - dvol.rolling(24, min_periods=8).mean()
    cand["vts_dvol_slope"] = ts_slope
    cand["vts_dvol_slope_chg_8h"] = ts_slope.diff(8)

    # B3. Intrabar DVOL range expansion: (dvol_high - dvol_low) z-scored over
    #     48h. Spikes in the vol-of-vol that precede regime shifts.
    dvol_rng = (dvol_hi - dvol_lo).clip(lower=0.0)
    cand["vts_dvol_range_z_48h"] = _safe_z(dvol_rng, 48, 16)

    # -------------------------------------------------------------------
    # GROUP C — DVOL * FUNDING  (vol premium vs carry)
    # -------------------------------------------------------------------
    # C1. Vol-premium-vs-carry: signed VRP-zscore times funding-zscore. When
    #     options are richly priced (high VRP) AND carry is extreme, leverage
    #     unwind risk is elevated. Both inputs are dense (>30% nonzero), so no
    #     sparse-indicator violation.
    fund_z = _safe_z(funding, 72, 24)
    cand["vts_vrp_x_funding"] = cand["vts_vrp_z_72h"] * fund_z

    # C2. DVOL change conditioned on funding sign: is implied vol rising while
    #     carry is positive (crowded longs paying up) vs rising while carry
    #     negative? Product of two dense continuous signals.
    cand["vts_dvolchg_x_funding"] = dvol.pct_change(8) * fund_z

    # -------------------------------------------------------------------
    # GROUP D — COMPRESSION -> EXPANSION SETUP  (continuous, dense)
    # -------------------------------------------------------------------
    # D1. Vol-compression score: how deep DVOL sits in its own 96h
    #     min-max range (0 = at trailing low/compressed, 1 = at high).
    #     Continuous, dense — a low value flags a compressed-vol setup.
    win = 96
    dmin = dvol.rolling(win, min_periods=24).min()
    dmax = dvol.rolling(win, min_periods=24).max()
    cand["vts_dvol_compression"] = (dvol - dmin) / (dmax - dmin).replace(0.0, np.nan)

    # D2. Compression-then-expansion impulse: bars-since DVOL was at a 96h
    #     compressed low, multiplied by current 4h DVOL change. Captures the
    #     *release* from compression (expansion just beginning). Built as a
    #     dense continuous interaction, not a sparse flag.
    is_low = (dvol <= (dmin + 0.05 * (dmax - dmin))).astype(float)
    # bars since last compressed-low (cumulative-from-past)
    last_low_grp = is_low.cumsum()
    bars_since_low = is_low.groupby(last_low_grp).cumcount().astype(float)
    cand["vts_expansion_release"] = bars_since_low * dvol.pct_change(4)

    # D3. Realized-vol compression depth (independent of implied): RV percentile
    #     in its own 96h range. Low realized vol historically precedes expansion.
    rmin = rv_ann.rolling(win, min_periods=24).min()
    rmax = rv_ann.rolling(win, min_periods=24).max()
    cand["vts_rv_compression"] = (rv_ann - rmin) / (rmax - rmin).replace(0.0, np.nan)

    # -------------------------------------------------------------------
    # GROUP E — DVOL PERCENTILE REGIME
    # -------------------------------------------------------------------
    # E1. DVOL long-window percentile rank (240h ~ 10d trailing). Regime
    #     position of implied vol — distinct from the 72h zscore V7 holds.
    cand["vts_dvol_pctile_240h"] = dvol.rolling(240, min_periods=72).apply(
        lambda a: (a[-1] >= a).mean(), raw=True
    )

    # E2. VRP percentile rank (is the premium itself in a high/low regime
    #     vs its own 240h history). Captures premium regime, not vol regime.
    cand["vts_vrp_pctile_240h"] = vrp.rolling(240, min_periods=72).apply(
        lambda a: (a[-1] >= a).mean(), raw=True
    )

    return cand


def main() -> None:
    df = load_features()
    print(f"[load] features_all: {df.shape}")
    print(f"[load] index {df.index[0]} .. {df.index[-1]}")

    cand = build_candidates(df)
    n_built = cand.shape[1]
    print(f"[build] {n_built} candidate features: {list(cand.columns)}")

    # -------- leak-safety self-check: trailing-only invariance ----------
    # A trailing feature computed on df[:k] must equal the same feature
    # computed on the full df, at every bar <= k (appending future bars must
    # not change past values). We verify on a mid-frame cut.
    k = int(len(df) * 0.7)
    cand_prefix = build_candidates(df.iloc[:k].copy())
    leak_max = 0.0
    leak_col = ""
    for c in cand.columns:
        a = cand[c].iloc[:k]
        bb = cand_prefix[c]
        both = a.notna() & bb.notna()
        if both.sum() == 0:
            continue
        d = (a[both] - bb[both]).abs().max()
        if pd.notna(d) and d > leak_max:
            leak_max, leak_col = float(d), c
    leak_ok = leak_max < 1e-9
    print(f"[leak-check] max prefix-vs-full diff = {leak_max:.3e} "
          f"({leak_col or 'n/a'}) -> {'PASS (trailing-only)' if leak_ok else 'FAIL'}")

    # -------------------------- screen ---------------------------------
    res = screen_candidates(cand, existing_df=df)
    show = ["feature", "cond_ic", "raw_ic", "frac_fold_consist",
            "max_corr_existing", "nonzero_frac", "n_folds",
            "pass_cond_ic", "pass_consist", "pass_notredundant",
            "pass_notsparse", "SCREEN_PASS"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== SCREEN RESULTS (ranked by |cond_ic|) ===")
    print(res[show].to_string(index=False))

    passers = res[res["SCREEN_PASS"]]
    print(f"\n[result] {len(passers)} / {n_built} candidates pass full screen")

    if len(passers) > 0:
        cols = passers["feature"].tolist()
        cand[cols].to_parquet(OUT_PARQUET)
        print(f"[save] wrote {len(cols)} passing cols -> {OUT_PARQUET}")
    else:
        print("[save] no passers — no parquet written (V7 saturated on this channel)")

    # compact top-5 table for synthesis
    top5 = res.head(5)[["feature", "cond_ic", "frac_fold_consist",
                        "max_corr_existing", "nonzero_frac"]]
    print("\n=== TOP-5 by |cond_ic| ===")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
