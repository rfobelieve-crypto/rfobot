"""
V7 multicoin — ETH feature elimination screen (2026-07-23, 2nd follow-up).

Two prior attempts (eth_direction_gate_a.py, eth_retune_ab.py) both confirmed
NO-GO with the full 136-feature BTC-designed set: neither the raw port nor
hyperparameter retuning + dropping thin-coverage columns produced a signal
close to the 0.54 gate. The user asked to try feature ELIMINATION next: is
there a small subset of the existing 136 features that individually carries
real ETH signal, even if the full ensemble (dominated by BTC-irrelevant
noise features) doesn't?

Methodology (deliberately conservative — screening 136 features is itself a
multiple-comparisons problem; mistake.md 2026-06-01/06-02/06-20 are all
variations of "found something via search, it was noise"):

  1. Univariate raw-IC screen: spearman(feature, y_path_ret_4h) for each of
     the 136 features, computed ONLY on WF test-fold bars (same 77 folds as
     everywhere else — never train-fold bars, to avoid any in-sample leak
     into the screen itself).
  2. Per-fold sign consistency: what fraction of the 77 folds have the same
     IC sign as the pooled IC. A feature with real, stable signal should be
     consistent across most folds; a feature that's "significant" only
     because of 2-3 lucky folds is noise.
  3. Strict compound bar (stricter than featsearch_lib.py's ADD-candidate
     screen, because this is a bigger search over more features):
         |pooled IC| >= 0.05  AND  fold_consistency >= 0.65
  4. Whatever survives gets ONE retrain-and-test pass (not iterative
     re-screening) through the SAME 4-condition sanity gate used for BTC's
     own feature A/B and the ETH retune follow-up, so a "found a working
     subset" claim has to survive the identical bar that would catch an
     outlier-driven mirage.

Run: python research/multicoin/eth_feature_elimination.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import walk_forward_splits
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available
from research.feature_search_ab import BASE_PARAMS, _pooled, _verdict_block
from research.multicoin.eth_retune_ab import _per_fold_oos_custom

CACHE = PROJECT_ROOT / "research" / "multicoin" / ".cache" / "eth_features_all.parquet"
RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "multicoin"

IC_MIN = 0.05
CONSIST_MIN = 0.65


def screen_univariate(df: pd.DataFrame, features: list[str],
                      splits: list[tuple[list[int], list[int]]]) -> pd.DataFrame:
    """IC + fold-sign-consistency for each feature, computed ONLY on test-fold
    bars (never train bars) across all WF folds — avoids any in-sample leak
    into the screen."""
    test_idx = sorted(set(i for _, te in splits for i in te))
    test_df = df.iloc[test_idx]
    y = test_df["y_path_ret_4h"].values.astype(float)
    y_valid = np.isfinite(y)

    # map each test bar to its fold number for per-fold consistency
    fold_of = np.full(len(df), -1)
    for fi, (_, te) in enumerate(splits):
        for i in te:
            fold_of[i] = fi
    test_fold = fold_of[test_idx]

    recs = []
    for feat in features:
        v = pd.to_numeric(test_df[feat], errors="coerce").values.astype(float)
        ok = y_valid & np.isfinite(v)
        if ok.sum() < 200:
            recs.append(dict(feature=feat, ic=np.nan, consist=np.nan, n=int(ok.sum())))
            continue
        vv, yy, fo = v[ok], y[ok], test_fold[ok]
        ic = spearmanr(vv, yy).correlation
        signs = []
        for f in np.unique(fo):
            fm = fo == f
            if fm.sum() < 10:
                continue
            fic = spearmanr(vv[fm], yy[fm]).correlation
            if np.isfinite(fic):
                signs.append(np.sign(fic))
        signs = np.array(signs)
        consist = (float((signs == np.sign(ic)).mean())
                  if len(signs) and np.isfinite(ic) else np.nan)
        recs.append(dict(feature=feat, ic=ic, consist=consist, n=int(ok.sum()),
                         n_folds=len(signs)))
    res = pd.DataFrame(recs)
    res["abs_ic"] = res["ic"].abs()
    res = res.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    res["pass_ic"] = res["abs_ic"] >= IC_MIN
    res["pass_consist"] = res["consist"] >= CONSIST_MIN
    res["SCREEN_PASS"] = res["pass_ic"] & res["pass_consist"]
    return res


def main():
    print("=" * 72)
    print("  ETH FEATURE ELIMINATION SCREEN")
    print("=" * 72)

    df = pd.read_parquet(CACHE)
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]
    deployed = filter_available(FULL_DIRECTION, list(df.columns))
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    print(f"  ETH bars={len(df)}  features={len(deployed)}  folds={len(splits)}")

    print(f"\n  Univariate screen (test-fold-only IC + sign consistency), "
          f"bar: |IC|>={IC_MIN} AND consist>={CONSIST_MIN}")
    screen = screen_univariate(df, deployed, splits)
    n_expected_false_pos = len(deployed) * 0.05
    print(f"  (n={len(deployed)} features screened — at p~0.05 per-feature, "
          f"expect ~{n_expected_false_pos:.1f} false positives by chance alone; "
          f"the compound IC+consistency bar is deliberately stricter than that)")

    survivors = screen[screen["SCREEN_PASS"]]
    print(f"\n  SCREEN_PASS: {len(survivors)}/{len(deployed)} features")
    if len(survivors) > 0:
        print(survivors[["feature", "ic", "consist", "n_folds"]].to_string(index=False))
    else:
        print("  none — no individual feature clears the strict bar.")

    print("\n  Top 15 by |IC| regardless of pass/fail (for visibility):")
    print(screen.head(15)[["feature", "ic", "consist", "pass_ic", "pass_consist"]].to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    screen.to_csv(RESULTS_DIR / "eth_feature_screen.csv", index=False)
    print(f"\n  saved full screen -> {RESULTS_DIR / 'eth_feature_screen.csv'}")

    if len(survivors) < 3:
        print(f"\n  Only {len(survivors)} survivor(s) — too few for a meaningful reduced-"
              f"feature ensemble. Stopping here (retraining on <3 features would just be "
              f"a near-univariate model, not a real ensemble test).")
        return

    # One retrain-and-test pass on survivors, judged against the SAME 4-condition
    # gate as everything else this project ships as an "improvement."
    surv_feats = survivors["feature"].tolist()
    print(f"\n  Retraining on {len(surv_feats)} survivor features "
          f"(single pass, no iterative re-screening)...")

    baseline_folds = _per_fold_oos_custom(df, deployed, dict(BASE_PARAMS))
    b_auc, b_ic = _pooled(baseline_folds)

    t0 = time.time()
    reduced_folds = _per_fold_oos_custom(df, surv_feats, dict(BASE_PARAMS))
    r_auc, r_ic = _pooled(reduced_folds)
    verdict = _verdict_block(baseline_folds, reduced_folds, "auc")
    cond = verdict["conditions"]

    print(f"\n  baseline(136 feat) AUC={b_auc:.4f} IC={b_ic:+.4f}")
    print(f"  reduced({len(surv_feats)} feat) AUC={r_auc:.4f} IC={r_ic:+.4f} "
          f"({time.time()-t0:.0f}s)")
    print(f"  4-cond vs baseline: agg_lift={verdict['agg_lift']:+.4f} "
          f"mean_fold_lift={verdict['mean_fold_lift']:+.5f} "
          f"frac_pos={verdict['frac_pos']:.2f} boot_p_le0={verdict['boot_p_le0']:.4f}")
    print(f"  => {'REAL IMPROVEMENT' if verdict['DEPLOY'] else 'no significant lift'}")
    print(f"  absolute vs BTC 0.5412 gate(>=0.54): "
          f"{'PASS' if r_auc >= 0.54 else 'still FAIL'} ({r_auc:.4f})")


if __name__ == "__main__":
    main()
