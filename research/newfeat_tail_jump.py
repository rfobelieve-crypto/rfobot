"""
Novel feature family: tail_jump  (higher-moment & jump microstructure)
=======================================================================
2026-06-14 feature search.  Build trailing-only candidate features capturing
*asymmetric / jump / exhaustion* price-behavior statistics that the deployed
136-feature V7 model has NOT already captured, then screen them with the SHARED
screen_candidates() against the CLEAN V7 OOS residual.

Design constraints (mistake.md):
  - Trailing-only: a feature at bar t uses ONLY bars <= t.  All rolling windows
    are backward; gap/jump states are cumulative-from-past.  No .shift(-k).
  - NO TA indicators (MACD/EMA/RSI/Bollinger/Stoch).  Pure price-behavior stats
    (returns, semivariance, kurtosis, wicks, jumps) only.
  - NO feat*sparse_indicator with nonzero<30%.  Where a jump indicator is sparse
    we DON'T multiply an existing feature by it; instead we carry the *signed
    jump magnitude* as its own continuous channel (mostly 0 is acceptable ONLY
    if it clears the nonzero>0.30 screen; otherwise it just fails honestly).
  - Daily alt-data already lagged; we touch ONLY raw OHLCV here.

WHY these are distinct from existing V7 cols (verified present):
  V7 already has SYMMETRIC moments: return_kurtosis, return_skew, vol_kurtosis,
  wick_asymmetry, body_ratio, signed_body_ratio, realized_vol_20b.  This family
  deliberately targets the *asymmetric / sequence / jump* structure those miss:
    - downside-only semivariance & semivar share (not symmetric vol)
    - signed jump variation / signed threshold-jump (direction-carrying)
    - gap (open-vs-prevclose) and gap-fill tendency
    - consecutive same-side wick exhaustion run-length (sequence state)
    - body/range compression over a window (coiling), not per-bar body_ratio
    - upside/downside realized-semivol ratio (skew-of-vol)
The redundancy (max_corr<0.90) screen is the guardrail against accidental dup.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # screen runs per-fold spearman on some
# near-constant slices (sparse jump cols) -> harmless SpearmanRConstantInput

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.featsearch_lib import (  # noqa: E402
    load_features, screen_candidates,
)

RESULTS = PROJECT_ROOT / "research" / "results"
OUT_PARQUET = RESULTS / "newfeat_tail_jump.parquet"


def _safe_div(a, b):
    b = b.replace(0.0, np.nan)
    return a / b


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """All trailing-only.  Returns DataFrame indexed by df.index."""
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # log returns (bar-to-bar), trailing
    ret = np.log(c).diff()
    prev_c = c.shift(1)                       # bar t-1 close (past) -> trailing
    rng = (h - l)                             # current-bar range
    body = (c - o).abs()
    cand = pd.DataFrame(index=df.index)

    # ---- ATR-style scale (trailing true range, NOT an EMA/TA indicator) -----
    # Wilder ATR is an EMA -> banned.  We use a SIMPLE trailing mean of true
    # range = a plain dispersion scale (price-behavior stat), window 24 bars.
    tr = pd.concat([(h - l),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    atr24 = tr.rolling(24, min_periods=12).mean()

    # =====================================================================
    # 1) DOWNSIDE SEMIVARIANCE (asymmetric dispersion, 24-bar trailing)
    #    Only negative returns contribute.  V7 has symmetric realized_vol_20b;
    #    this is the *downside-only* second moment.
    # =====================================================================
    neg = ret.where(ret < 0, 0.0)
    pos = ret.where(ret > 0, 0.0)
    semivar_dn = (neg ** 2).rolling(24, min_periods=12).mean()
    semivar_up = (pos ** 2).rolling(24, min_periods=12).mean()
    cand["tj_semivar_dn_24"] = np.sqrt(semivar_dn)

    # 2) SEMIVAR SHARE: fraction of total variance that is downside.
    #    Skew-of-vol proxy in [0,1].  >0.5 = downside-heavy dispersion.
    cand["tj_semivar_dn_share_24"] = _safe_div(semivar_dn, (semivar_dn + semivar_up))

    # 3) SEMI-VOL RATIO (up vs down realized vol, log).  Directional dispersion
    #    asymmetry distinct from return_skew (which is skew of returns, not vol).
    cand["tj_semivol_ratio_24"] = np.log(_safe_div(np.sqrt(semivar_up),
                                                   np.sqrt(semivar_dn)))

    # =====================================================================
    # 4) SIGNED JUMP VARIATION (Barndorff-Nielsen style):  realized var minus
    #    bipower variation, carrying the sign of the net jump direction.
    #    Captures discontinuous moves V7's smooth-vol features miss.
    # =====================================================================
    w = 24
    rv = (ret ** 2).rolling(w, min_periods=12).sum()
    bpv = (np.pi / 2.0) * (ret.abs() * ret.abs().shift(1)).rolling(
        w, min_periods=12).sum()
    jump_var = (rv - bpv).clip(lower=0.0)               # jump component >= 0
    # carry sign: net directional pull of the largest-|ret| bars in the window
    net_dir = np.sign(ret).rolling(w, min_periods=12).apply(
        lambda x: np.sign(np.nansum(x)), raw=True)
    cand["tj_signed_jumpvar_24"] = np.sqrt(jump_var) * net_dir

    # =====================================================================
    # 5) SIGNED THRESHOLD-JUMP (|ret|/ATR over threshold, carrying sign).
    #    The classic "jump indicator" but we keep it CONTINUOUS+SIGNED so it's
    #    not a sparse 0/1 multiplier (mistake.md sparsity rule).  Value = the
    #    z-scaled return only when it exceeds 1.5x its own bar-scale, else 0.
    # =====================================================================
    bar_scale = _safe_div(tr, atr24)            # current TR relative to ATR
    ret_over_atr = _safe_div(ret, atr24 / c)    # return in ATR units (approx)
    is_jump = (bar_scale > 1.5)
    cand["tj_signed_jump_15_24"] = ret_over_atr.where(is_jump, 0.0)

    # 6) ROLLING SIGNED-JUMP INTENSITY: trailing sum of signed jumps over 12 bars
    #    (a smoother, denser version -> better nonzero coverage).
    cand["tj_jump_intensity_12"] = cand["tj_signed_jump_15_24"].rolling(
        12, min_periods=6).sum()

    # =====================================================================
    # 7) GAP (open vs previous close), in ATR units.  Overnight/inter-bar gap.
    #    V7 has no open-vs-prevclose channel.  Continuous & signed.
    # =====================================================================
    gap = _safe_div((o - prev_c), atr24)
    cand["tj_gap_atr"] = gap

    # 8) INTRABAR OVERSHOOT-REVERSAL (exhaustion / mean-reversion of the bar's
    #    extreme).  NOTE: true open-vs-prevclose GAPS do not exist on continuous
    #    BTC perp (51% of bars have open==prev_close exactly, max gap ~7e-4*ATR;
    #    verified) -> a gap-fill feature is structurally null here.  Instead we
    #    measure how far price REVERTED from the bar's own extreme back to the
    #    close, signed by which extreme.  + = closed back down from the high
    #    (up-rejection), - = closed back up from the low (down-rejection).
    #    Trailing 12-bar mean -> persistent intrabar reversal pressure.
    rev_from_high = _safe_div((h - c), rng)      # how far below the high it closed
    rev_from_low = _safe_div((c - l), rng)       # how far above the low it closed
    overshoot_rev = (rev_from_high - rev_from_low)  # +ve = up-rejection dominant
    cand["tj_overshoot_reversal_12"] = overshoot_rev.rolling(
        12, min_periods=6).mean()

    # =====================================================================
    # 9) CONSECUTIVE WICK EXHAUSTION (upper).  Run-length of bars where the
    #    upper shadow dominates (rejection of highs) -> exhaustion of up-moves.
    #    Sequence STATE, cumulative-from-past, distinct from per-bar wick_asym.
    # =====================================================================
    upper_wick = (h - np.maximum(o, c))
    lower_wick = (np.minimum(o, c) - l)
    upper_dom = (upper_wick > (1.5 * lower_wick + 1e-9)) & (upper_wick > 0.3 * rng)
    lower_dom = (lower_wick > (1.5 * upper_wick + 1e-9)) & (lower_wick > 0.3 * rng)

    def run_length(mask: pd.Series) -> pd.Series:
        # trailing consecutive-True count ending at bar t
        grp = (~mask).cumsum()
        return mask.groupby(grp).cumsum()

    up_exh = run_length(upper_dom.fillna(False))
    dn_exh = run_length(lower_dom.fillna(False))
    # signed exhaustion: + = upper-wick run (bearish exhaustion), - = lower run
    cand["tj_wick_exhaustion_signed"] = (up_exh - dn_exh).astype(float)

    # 10) WICK-DOMINANCE PRESSURE (12-bar trailing mean of signed wick share).
    #     Denser continuous version: per-bar (lower-upper)/range, smoothed.
    wick_press = _safe_div((lower_wick - upper_wick), rng)
    cand["tj_wick_pressure_12"] = wick_press.rolling(12, min_periods=6).mean()

    # =====================================================================
    # 11) BODY/RANGE COMPRESSION (coiling) over 12 bars.  Mean body/range ratio
    #     trailing -> low value = lots of indecision/coiling.  Distinct from
    #     per-bar body_ratio (V7 has single-bar only, not the window aggregate).
    # =====================================================================
    body_range = _safe_div(body, rng)
    cand["tj_body_compression_12"] = body_range.rolling(12, min_periods=6).mean()

    # 12) RANGE COMPRESSION RATIO: current 6-bar avg range vs 48-bar avg range.
    #     < 1 = compressed (quiet) relative to recent history.  Pure dispersion.
    rng_short = rng.rolling(6, min_periods=3).mean()
    rng_long = rng.rolling(48, min_periods=20).mean()
    cand["tj_range_compress_6_48"] = _safe_div(rng_short, rng_long)

    # =====================================================================
    # 13) TAIL-RETURN ASYMMETRY: trailing share of |ret| coming from the worst
    #     downside tail vs best upside tail over 48 bars (CVaR-style asymmetry).
    #     Heavy-downside-tail signature distinct from symmetric kurtosis.
    # =====================================================================
    def tail_asym(x: np.ndarray) -> float:
        if np.isnan(x).all():
            return np.nan
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return np.nan
        q = np.quantile(np.abs(x), 0.80)
        up = x[(x > 0) & (np.abs(x) >= q)].sum()
        dn = -x[(x < 0) & (np.abs(x) >= q)].sum()
        tot = up + dn
        if tot <= 0:
            return 0.0
        return (up - dn) / tot      # +1 all upside tail, -1 all downside tail
    cand["tj_tail_asym_48"] = ret.rolling(48, min_periods=20).apply(
        tail_asym, raw=True)

    # 14) MAX DRAWUP/DRAWDOWN ASYMMETRY over 24 bars (path shape, trailing).
    #     Largest peak-to-trough fall vs trough-to-peak rise within window.
    def du_dd_asym(x: np.ndarray) -> float:
        # x = close path; compute max run-up and max run-down magnitudes
        if np.isnan(x).any() or len(x) < 6:
            return np.nan
        lx = np.log(x)
        # max drawdown
        run_max = np.maximum.accumulate(lx)
        dd = (lx - run_max).min()
        # max drawup
        run_min = np.minimum.accumulate(lx)
        du = (lx - run_min).max()
        tot = du + (-dd)
        if tot <= 0:
            return 0.0
        return (du + dd) / tot       # >0 upside path dominates, <0 downside
    cand["tj_drawup_dd_asym_24"] = c.rolling(24, min_periods=12).apply(
        du_dd_asym, raw=True)

    # 15) CLOSE-LOCATION-VALUE persistence: trailing mean of where close sits in
    #     the bar range ((c-l)/(h-l) -> 0..1).  Smoothed buying/selling close.
    #     Distinct from signed_body_ratio (that's (c-o)/range, this is (c-l)).
    clv = _safe_div((c - l), rng)
    cand["tj_clv_persistence_12"] = (clv.rolling(12, min_periods=6).mean() - 0.5)

    return cand


def main():
    df = load_features()
    print(f"[load] features {df.shape}, index {df.index.min()} -> {df.index.max()}")

    cand = build_candidates(df)
    print(f"[build] {cand.shape[1]} candidates: {list(cand.columns)}")

    # ---- leak-safety self-check: rebuild on a TRUNCATED frame, the overlap of
    #      the truncated feature values must EXACTLY equal the full-frame values
    #      on the shared bars (trailing-only => appending future bars cannot
    #      change a past value).  This catches any accidental forward leakage.
    cut = int(len(df) * 0.7)
    df_trunc = df.iloc[:cut]
    cand_trunc = build_candidates(df_trunc)
    common_idx = cand_trunc.index
    full_on_common = cand.loc[common_idx]
    leak_bad = []
    for col in cand.columns:
        a = cand_trunc[col]
        b = full_on_common[col]
        both = a.notna() & b.notna()
        if both.sum() == 0:
            continue
        diff = (a[both] - b[both]).abs()
        # tolerance for float; rolling.apply can have tiny fp noise
        if diff.max() > 1e-8:
            leak_bad.append((col, float(diff.max())))
    if leak_bad:
        print(f"[LEAK-CHECK] *** FAILED *** {leak_bad}")
    else:
        print(f"[LEAK-CHECK] PASS — all {cand.shape[1]} cols trailing-only "
              f"(truncated rebuild matches full frame on {len(common_idx)} bars)")

    # coverage report
    print("\n[coverage] nonnull / nonzero per candidate:")
    for col in cand.columns:
        s = cand[col]
        nn = s.notna().mean()
        nz = (s.abs() > 1e-12).mean()
        print(f"  {col:30s} nonnull={nn:.2f} nonzero={nz:.2f} "
              f"std={np.nanstd(s):.4g}")

    # ---- THE SHARED SCREEN ------------------------------------------------
    res = screen_candidates(cand, existing_df=df)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    show = ["feature", "cond_ic", "raw_ic", "frac_fold_consist",
            "nonzero_frac", "max_corr_existing", "SCREEN_PASS"]
    print("\n[screen] ranked by |cond_ic|:")
    print(res[show].to_string(index=False))

    passers = res[res["SCREEN_PASS"]].copy()
    print(f"\n[result] {len(passers)} / {len(res)} candidates PASS the screen.")

    if len(passers):
        keep = passers["feature"].tolist()
        cand[keep].to_parquet(OUT_PARQUET)
        print(f"[save] wrote {len(keep)} passing cols -> {OUT_PARQUET}")
    else:
        print("[save] no passers — nothing written (parquet_path = '')")

    # always dump full screen table for the synthesis record
    res.to_csv(RESULTS / "newfeat_tail_jump_screen.csv", index=False)
    print(f"[save] full screen table -> {RESULTS / 'newfeat_tail_jump_screen.csv'}")


if __name__ == "__main__":
    main()
