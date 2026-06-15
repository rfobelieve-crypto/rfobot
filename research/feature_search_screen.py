"""
Feature-search SCREEN (2026-06-14, ultracode re-run of feature engineering).

Goal: of the ~137 already-computed-but-not-deployed columns in the production
feature parquet, which (if any) carry MARGINAL information the current 136-feature
V7 direction model has NOT already absorbed?

Discipline enforced (from mistake.md):
  - 2026-06-01 / 2026-06-02: screen by CONDITIONAL IC vs the V7 OOS residual,
    NOT raw univariate IC.  Raw IC high only proves "correlated with target";
    conditional IC proves "V7 hasn't captured it".
  - 2026-04-13 in-sample vs OOS: the residual MUST come from a clean
    walk-forward (no early-stop-on-test leak), never an in-sample prediction.
  - 2026-04-13 sparse-indicator: flag candidates whose non-zero fraction < 30%.
  - feature redundancy: report max |corr| to any deployed feature; a candidate
    that is ~collinear with a deployed one cannot add ensemble lift.

This script does NOT decide deployment.  It produces a ranked candidate list;
the ensemble A/B (feature_search_ab.py) + per-fold sanity is the deploy gate.

Run:  python research/feature_search_screen.py
Out:  research/results/feature_search_screen.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_direction_reg_labels import build_direction_reg_labels
from research.dual_model.direction_features_v2 import FULL_DIRECTION, filter_available

RESULTS = PROJECT_ROOT / "research" / "results"

BASE_PARAMS = dict(
    objective="reg:squarederror", eval_metric="mae",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0, n_jobs=4,
)

RAW_META = {"open", "high", "low", "close", "volume", "taker_buy_vol",
            "taker_buy_quote", "trade_count", "quote_vol", "close_time",
            "ignore", "ts_open", "y_path_ret_4h"}


def clean_walk_forward(df, features):
    """Clean per-fold XGBRegressor on y_path_ret_4h: NO early-stop-on-test.
    Returns OOS frame [pred_ret, y, fold] indexed by bar."""
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    rows = []
    t0 = time.time()
    for fi, (tr, te) in enumerate(splits):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        m_tr = tr_df["y_path_ret_4h"].notna()
        m_te = te_df["y_path_ret_4h"].notna()
        Xtr = tr_df.loc[m_tr, features].fillna(0)
        ytr = tr_df.loc[m_tr, "y_path_ret_4h"].values.astype(float)
        Xte = te_df.loc[m_te, features].fillna(0)
        yte = te_df.loc[m_te, "y_path_ret_4h"].values.astype(float)
        if len(ytr) < 50 or len(yte) < 5:
            continue
        model = xgb.XGBRegressor(**BASE_PARAMS)
        model.fit(Xtr, ytr)
        rows.append(pd.DataFrame(
            {"pred_ret": model.predict(Xte), "y": yte, "fold": fi},
            index=te_df.loc[m_te].index))
    oos = pd.concat(rows).sort_index()
    print(f"  clean WF baseline: {len(oos)} OOS bars, {oos['fold'].nunique()} folds, "
          f"{time.time()-t0:.0f}s")
    return oos


def main():
    print("=" * 72)
    print("  FEATURE-SEARCH SCREEN — conditional IC vs V7 OOS residual")
    print("=" * 72)
    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]

    deployed = filter_available(FULL_DIRECTION, list(df.columns))
    print(f"  bars={len(df)}  span={df.index.min()} ~ {df.index.max()}")
    print(f"  deployed V7 features: {len(deployed)}")

    # candidate pool: every numeric col not deployed, not raw/meta/target,
    # coverage >= 200 non-NaN, non-constant
    cand = []
    for c in df.columns:
        if c in deployed or c in RAW_META or c.startswith("y_"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 200 or s.nunique(dropna=True) <= 1:
            continue
        cand.append(c)
    print(f"  candidate pool (non-deployed real features): {len(cand)}")

    # --- clean baseline residual ---
    oos = clean_walk_forward(df, deployed)
    y = oos["y"].values.astype(float)
    pred = oos["pred_ret"].values.astype(float)
    resid = y - pred

    mask = y != 0
    base_auc = roc_auc_score((y[mask] > 0).astype(int), pred[mask])
    base_ic = spearmanr(pred, y).correlation
    print(f"\n  TODAY's clean V7 baseline (no early-stop leak):")
    print(f"    sign-AUC = {base_auc:.4f}    Spearman IC = {base_ic:+.4f}")
    print(f"    (contrast: leaky/deployed canonical ~0.59 AUC / 0.17 IC is "
          f"early-stop-inflated)")

    oos_idx = oos.index
    fold_of = oos["fold"]

    # --- per-candidate conditional IC ---
    cand_vals = df.loc[oos_idx, cand].apply(pd.to_numeric, errors="coerce")
    recs = []
    for c in cand:
        v = cand_vals[c].values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 200:
            continue
        vv, rr, yy = v[ok], resid[ok], y[ok]
        cond_ic = spearmanr(vv, rr).correlation
        raw_ic = spearmanr(vv, yy).correlation
        # per-fold conditional IC sign consistency
        fo = fold_of.values[ok]
        signs = []
        for f in np.unique(fo):
            fm = fo == f
            if fm.sum() < 10:
                continue
            ci = spearmanr(vv[fm], rr[fm]).correlation
            if np.isfinite(ci):
                signs.append(np.sign(ci))
        signs = np.array(signs)
        frac_consist = (float((signs == np.sign(cond_ic)).mean())
                        if len(signs) else np.nan)
        nz_frac = float((np.abs(vv) > 1e-12).mean())
        recs.append(dict(feature=c, cond_ic=cond_ic, raw_ic=raw_ic,
                         abs_cond_ic=abs(cond_ic), frac_fold_consist=frac_consist,
                         n_folds=len(signs), nonzero_frac=nz_frac, n=int(ok.sum())))

    res = pd.DataFrame(recs)

    # max |corr| to any deployed feature (redundancy check), spearman on OOS bars
    dep_vals = df.loc[oos_idx, deployed].apply(pd.to_numeric, errors="coerce")
    combo = pd.concat([cand_vals, dep_vals], axis=1)
    corr = combo.corr(method="spearman")
    max_dep_corr = {}
    for c in res["feature"]:
        if c in corr.index:
            row = corr.loc[c, deployed].abs()
            max_dep_corr[c] = float(row.max()) if len(row) else np.nan
        else:
            max_dep_corr[c] = np.nan
    res["max_corr_deployed"] = res["feature"].map(max_dep_corr)

    res["family"] = res["feature"].str.split("_").str[0]
    res = res.sort_values("abs_cond_ic", ascending=False).reset_index(drop=True)

    # screen flags (mistake.md thresholds)
    res["pass_cond_ic"] = res["abs_cond_ic"] > 0.03
    res["pass_consist"] = res["frac_fold_consist"] > 0.60
    res["pass_notredundant"] = res["max_corr_deployed"] < 0.90
    res["pass_notsparse"] = res["nonzero_frac"] > 0.30
    res["SCREEN_PASS"] = (res["pass_cond_ic"] & res["pass_consist"]
                          & res["pass_notredundant"] & res["pass_notsparse"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "feature_search_screen.csv"
    res.to_csv(out, index=False)

    print(f"\n  === SCREEN RESULTS (ranked by |conditional IC|) ===")
    print(f"  {'feature':<34}{'condIC':>8}{'rawIC':>8}{'consist':>8}"
          f"{'maxdep':>8}{'nz%':>6}  PASS")
    for _, r in res.head(35).iterrows():
        print(f"  {r['feature']:<34}{r['cond_ic']:>+8.3f}{r['raw_ic']:>+8.3f}"
              f"{r['frac_fold_consist']:>8.2f}{r['max_corr_deployed']:>8.2f}"
              f"{r['nonzero_frac']*100:>5.0f}%  {'YES' if r['SCREEN_PASS'] else ''}")

    n_pass = int(res["SCREEN_PASS"].sum())
    print(f"\n  candidates passing full screen: {n_pass} / {len(res)}")
    if n_pass:
        passers = res[res["SCREEN_PASS"]]
        print("  by family:",
              dict(passers["family"].value_counts()))
        print("  PASSERS:", list(passers["feature"]))
    print(f"\n  saved -> {out}")
    print("  NOTE: screen pass = 'worth an ensemble A/B', NOT deploy. "
          "A/B + 4-cond per-fold sanity is the gate.")


if __name__ == "__main__":
    main()
