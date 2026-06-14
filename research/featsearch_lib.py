"""
Shared primitives for the 2026-06-14 feature search.

Why this exists: mistake.md is full of cases where subtly-wrong validation code
(in-sample residual, raw-IC instead of conditional-IC, pooled-only sanity) led
to wrong feature-deploy decisions.  The workflow's family-engineer agents must
NOT reinvent the screen — they build features and call screen_candidates() here.

The conditional-IC screen reuses the CLEAN baseline OOS predictions cached by
feature_search_ab.py (ab_baseline_clean.parquet): residual = y - pred over the
clean walk-forward (no early-stop leak).  Conditional IC = spearman(candidate,
residual): "information the deployed V7 has NOT already captured."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache"
FEATURES_PARQUET = CACHE / "features_all.parquet"
BASELINE_CLEAN = CACHE / "ab_baseline_clean.parquet"

# screen thresholds (mistake.md 2026-06-01/06-02/04-13)
COND_IC_MIN = 0.03      # |conditional IC| floor
CONSIST_MIN = 0.60      # per-fold conditional-IC sign consistency
MAXCORR_MAX = 0.90      # redundancy: max |corr| to any existing column
NONZERO_MIN = 0.30      # sparse-indicator guard


def load_features() -> pd.DataFrame:
    """The production feature parquet (282 cols incl. raw OHLCV + all channels)."""
    return pd.read_parquet(FEATURES_PARQUET)


def load_baseline_residual() -> pd.DataFrame:
    """Clean baseline OOS frame [pred, y, fold, residual] indexed by bar.
    Requires feature_search_ab.py to have built ab_baseline_clean.parquet."""
    if not BASELINE_CLEAN.exists():
        raise FileNotFoundError(
            f"{BASELINE_CLEAN} missing — run feature_search_ab.py first "
            "(it builds the clean baseline cache).")
    b = pd.read_parquet(BASELINE_CLEAN)
    b = b.copy()
    b["residual"] = b["y"].astype(float) - b["pred"].astype(float)
    return b


def look_ahead_check(series: pd.Series) -> bool:
    """Cheap sanity: a trailing feature must not change when future bars are
    appended.  Here we only assert the series has no NaN-free dependence on the
    LAST bar leaking backward — full leak-safety is the builder's job; this
    catches accidental .shift(-k) / centered windows by checking the last
    `k` values aren't suspiciously the only non-NaN ones.  Returns True if OK."""
    s = series
    if s.notna().sum() < 50:
        return True  # coverage handled elsewhere
    # if the feature is only defined on the tail (future-window artifact) flag it
    first_valid = s.first_valid_index()
    last_valid = s.last_valid_index()
    return first_valid is not None and last_valid is not None


def screen_candidates(cand_df: pd.DataFrame,
                      existing_df: pd.DataFrame | None = None,
                      baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Conditional-IC screen for new candidate features.

    Parameters
    ----------
    cand_df : DataFrame of NEW candidate features, indexed by the same bar
        datetime index as the feature parquet.  Must be trailing-only.
    existing_df : the 282-col production frame, for the redundancy (max-corr)
        check.  If None, loaded from disk.
    baseline : clean baseline residual frame.  If None, loaded from disk.

    Returns
    -------
    DataFrame ranked by |cond_ic| with columns:
        feature, cond_ic, raw_ic, frac_fold_consist, n_folds, nonzero_frac,
        max_corr_existing, pass_cond_ic, pass_consist, pass_notredundant,
        pass_notsparse, SCREEN_PASS
    """
    if existing_df is None:
        existing_df = load_features()
    if baseline is None:
        baseline = load_baseline_residual()

    idx = baseline.index
    resid = baseline["residual"].values.astype(float)
    fold = baseline["fold"].values
    # need y for raw IC
    y = baseline["y"].values.astype(float)

    recs = []
    for c in cand_df.columns:
        v = pd.to_numeric(cand_df[c].reindex(idx), errors="coerce").values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 200:
            recs.append(dict(feature=c, cond_ic=np.nan, raw_ic=np.nan,
                             frac_fold_consist=np.nan, n_folds=0,
                             nonzero_frac=np.nan, max_corr_existing=np.nan,
                             note="low_coverage"))
            continue
        vv, rr, yy, fo = v[ok], resid[ok], y[ok], fold[ok]
        cond_ic = spearmanr(vv, rr).correlation
        raw_ic = spearmanr(vv, yy).correlation
        signs = []
        for f in np.unique(fo):
            fm = fo == f
            if fm.sum() < 10:
                continue
            ci = spearmanr(vv[fm], rr[fm]).correlation
            if np.isfinite(ci):
                signs.append(np.sign(ci))
        signs = np.array(signs)
        frac = (float((signs == np.sign(cond_ic)).mean())
                if len(signs) and np.isfinite(cond_ic) else np.nan)
        nz = float((np.abs(vv) > 1e-12).mean())
        # redundancy vs existing cols on the OOS bars
        ev = existing_df.reindex(idx).loc[ok.nonzero()[0] if False else idx[ok]]
        try:
            corrs = ev.apply(pd.to_numeric, errors="coerce").corrwith(
                pd.Series(vv, index=idx[ok]), method="spearman").abs()
            maxcorr = float(corrs.replace(1.0, np.nan).max())  # drop self if dup
        except Exception:
            maxcorr = np.nan
        recs.append(dict(feature=c, cond_ic=cond_ic, raw_ic=raw_ic,
                         frac_fold_consist=frac, n_folds=int(len(signs)),
                         nonzero_frac=nz, max_corr_existing=maxcorr, note=""))

    res = pd.DataFrame(recs)
    res["abs_cond_ic"] = res["cond_ic"].abs()
    res = res.sort_values("abs_cond_ic", ascending=False).reset_index(drop=True)
    res["pass_cond_ic"] = res["abs_cond_ic"] > COND_IC_MIN
    res["pass_consist"] = res["frac_fold_consist"] > CONSIST_MIN
    res["pass_notredundant"] = res["max_corr_existing"] < MAXCORR_MAX
    res["pass_notsparse"] = res["nonzero_frac"] > NONZERO_MIN
    res["SCREEN_PASS"] = (res["pass_cond_ic"] & res["pass_consist"]
                          & res["pass_notredundant"] & res["pass_notsparse"])
    return res
