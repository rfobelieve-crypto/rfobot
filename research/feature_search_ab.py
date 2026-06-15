"""
Feature-search ENSEMBLE A/B (2026-06-14).

Decides whether adding candidate feature(s) to V7 (136 deployed direction
features) produces a REAL OOS AUC/IC lift — not an outlier-driven mirage.

The deploy gate is the 4-condition per-fold sanity from mistake.md 2026-06-02:
    1. aggregate (pooled) lift  > +0.005
    2. per-fold mean lift       > +0.001
    3. frac_positive folds      > 0.55
    4. bootstrap p(lift<=0)     < 0.05   (per-fold lifts resampled)
ALL FOUR for AUC, under the CLEAN walk-forward (no early-stop-on-test leak),
is required to call DEPLOY.  We also report the LEAKY prod-trainer numbers for
comparability with historical A/B (WQ101 etc.), but the clean numbers gate.

Baseline per-fold OOS is computed once and cached; each candidate set only
retrains the "new" model — paired on identical folds & test bars.

Run:  python research/feature_search_ab.py                 # default battery
      python research/feature_search_ab.py --add dvol_ma_24h,dvol_low --label dvol2
Out:  research/results/feature_search_ab.json   (appended verdicts)
"""
from __future__ import annotations

import sys
import json
import time
import argparse
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
CACHE = PROJECT_ROOT / "research" / "dual_model" / ".cache"

BASE_PARAMS = dict(
    objective="reg:squarederror", eval_metric="mae",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0, n_jobs=4,
)

# Candidate sets for the default battery (derived from feature_search_screen.csv)
BATTERY = {
    # the only family that PASSED the conditional-IC screen
    "dvol_level_all": ["dvol_ma_24h", "dvol_low", "dvol_open", "dvol_close",
                       "dvol_high"],
    "dvol_best_single": ["dvol_ma_24h"],
    # dvol dynamics (not just the collinear level) — does change/spread help?
    "dvol_dynamics": ["dvol_ma_24h", "dvol_rv_spread", "dvol_change_24h",
                      "dvol_zscore_72h"],
    # NEGATIVE CONTROLS: high |cond_ic| but FAILED fold-consistency (<0.4).
    # mistake.md predicts these add nothing despite the high univariate number.
    "dfp_highcondic": ["dfp_fcvd_cum_24h_rank", "dfp_scvd_cum_24h_rank",
                       "dfp_flow_consensus_persist_8h", "dfp_net_build_8h"],
    "init_highcondic": ["init_break_up_atr", "init_break_dn_atr",
                        "init_liq_cluster_imb", "init_liq_long_cluster"],
    "imb_multibar": ["imb_8b", "imb_5b", "imb_3b"],
    # kitchen sink: top-12 by |cond_ic| regardless of screen — exposes whether
    # piling on raw-correlated features helps or (as expected) hurts.
    "top12_condic": ["dfp_fcvd_cum_24h_rank", "init_break_up_atr",
                     "cg_liq_long_cum_8h", "imb_8b", "funding_pain_24h",
                     "funding_pain_48h", "init_liq_long_cluster",
                     "init_liq_cluster_imb", "dvol_ma_24h", "taker_delta_ma_4h",
                     "cg_liq_long_cum_4h", "init_break_dn_atr"],
}


def _per_fold_oos(df, features, *, leaky: bool):
    """Per-fold WF; returns dict fold -> DataFrame[pred, y] on test bars.
    leaky=True reproduces the production early-stop-on-test."""
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    params = dict(BASE_PARAMS)
    if leaky:
        params["early_stopping_rounds"] = 30
    out = {}
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
        m = xgb.XGBRegressor(**params)
        if leaky:
            m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        else:
            m.fit(Xtr, ytr)
        out[fi] = pd.DataFrame({"pred": m.predict(Xte), "y": yte},
                               index=te_df.loc[m_te].index)
    return out


def _fold_auc(d):
    y = d["y"].values
    mask = y != 0
    if mask.sum() < 8:
        return np.nan
    return roc_auc_score((y[mask] > 0).astype(int), d["pred"].values[mask])


def _pooled(folds):
    allp = pd.concat(folds.values())
    y = allp["y"].values
    mask = y != 0
    auc = roc_auc_score((y[mask] > 0).astype(int), allp["pred"].values[mask])
    ic = spearmanr(allp["pred"].values, y).correlation
    return auc, ic


def _boot_p(lifts, n=4000):
    """Bootstrap over per-fold lifts; return (ci_lo, ci_hi, p(lift<=0))."""
    lifts = np.asarray([x for x in lifts if np.isfinite(x)], dtype=float)
    if len(lifts) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(42)
    means = np.array([lifts[rng.integers(0, len(lifts), len(lifts))].mean()
                      for _ in range(n)])
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
            float((means <= 0).mean()))


def _verdict_block(base_folds, new_folds, metric_name):
    """4-condition sanity for one metric ('auc' or 'ic')."""
    common = sorted(set(base_folds) & set(new_folds))
    if metric_name == "auc":
        b = {f: _fold_auc(base_folds[f]) for f in common}
        n = {f: _fold_auc(new_folds[f]) for f in common}
        pooled_b = _pooled(base_folds)[0]
        pooled_n = _pooled(new_folds)[0]
    else:
        def fic(d):
            return spearmanr(d["pred"].values, d["y"].values).correlation
        b = {f: fic(base_folds[f]) for f in common}
        n = {f: fic(new_folds[f]) for f in common}
        pooled_b = _pooled(base_folds)[1]
        pooled_n = _pooled(new_folds)[1]
    lifts = [n[f] - b[f] for f in common
             if np.isfinite(n[f]) and np.isfinite(b[f])]
    lifts = np.array(lifts)
    agg_lift = pooled_n - pooled_b
    mean_lift = float(np.mean(lifts)) if len(lifts) else np.nan
    frac_pos = float((lifts > 0).mean()) if len(lifts) else np.nan
    ci_lo, ci_hi, p_le0 = _boot_p(lifts)
    cond = dict(
        c1_agg=bool(agg_lift > 0.005),
        c2_mean=bool(mean_lift > 0.001),
        c3_frac=bool(frac_pos > 0.55),
        c4_boot=bool(p_le0 < 0.05),
    )
    return dict(
        metric=metric_name, pooled_base=round(pooled_b, 4),
        pooled_new=round(pooled_n, 4), agg_lift=round(float(agg_lift), 5),
        mean_fold_lift=round(mean_lift, 5), frac_pos=round(frac_pos, 3),
        boot_ci=[round(ci_lo, 5), round(ci_hi, 5)], boot_p_le0=round(p_le0, 4),
        n_folds=len(lifts), conditions=cond,
        DEPLOY=bool(all(cond.values())),
    )


def _build_baseline(df, deployed):
    """Compute & cache baseline per-fold OOS for clean + leaky."""
    cache = {}
    for mode, leaky in [("clean", False), ("leaky", True)]:
        path = CACHE / f"ab_baseline_{mode}.parquet"
        if path.exists():
            raw = pd.read_parquet(path)
            cache[mode] = {f: g[["pred", "y"]] for f, g in raw.groupby("fold")}
            print(f"  baseline[{mode}] loaded from cache ({len(cache[mode])} folds)")
            continue
        t0 = time.time()
        folds = _per_fold_oos(df, deployed, leaky=leaky)
        flat = pd.concat([d.assign(fold=f) for f, d in folds.items()])
        flat.to_parquet(path)
        cache[mode] = folds
        a, i = _pooled(folds)
        print(f"  baseline[{mode}] built: {len(folds)} folds AUC={a:.4f} IC={i:+.4f} "
              f"({time.time()-t0:.0f}s)")
    return cache


def run_set(df, deployed, baseline, label, adds):
    adds = filter_available(adds, list(df.columns))
    feats = deployed + [a for a in adds if a not in deployed]
    rec = dict(label=label, adds=adds, n_new=len(adds))
    for mode in ("clean", "leaky"):
        t0 = time.time()
        new_folds = _per_fold_oos(df, feats, leaky=(mode == "leaky"))
        rec[mode] = dict(
            auc=_verdict_block(baseline[mode], new_folds, "auc"),
            ic=_verdict_block(baseline[mode], new_folds, "ic"),
            secs=round(time.time() - t0, 0),
        )
    # overall deploy = clean AUC passes all 4
    rec["DEPLOY"] = rec["clean"]["auc"]["DEPLOY"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", default="", help="comma-separated features (one set)")
    ap.add_argument("--label", default="custom")
    ap.add_argument("--extra-parquet", default="",
                    help="comma-separated parquet paths of NEW feature columns "
                         "to merge into df before A/B (index = bar datetime)")
    args = ap.parse_args()

    print("=" * 72)
    print("  FEATURE-SEARCH ENSEMBLE A/B — 4-condition per-fold sanity")
    print("=" * 72)
    df = load_and_cache_data()
    labels = build_direction_reg_labels(df)
    df = df.copy()
    df["y_path_ret_4h"] = labels["y_path_ret_4h"]

    for pth in [p.strip() for p in args.extra_parquet.split(",") if p.strip()]:
        extra = pd.read_parquet(pth)
        new_cols = [c for c in extra.columns if c not in df.columns]
        df = df.join(extra[new_cols].reindex(df.index))
        print(f"  merged {len(new_cols)} new cols from {pth}")

    deployed = filter_available(FULL_DIRECTION, list(df.columns))
    print(f"  bars={len(df)}  deployed={len(deployed)}")

    baseline = _build_baseline(df, deployed)

    sets = ({args.label: [s.strip() for s in args.add.split(",") if s.strip()]}
            if args.add else BATTERY)

    results = []
    for label, adds in sets.items():
        print(f"\n  --- A/B set: {label}  (+{len(adds)} features) ---")
        rec = run_set(df, deployed, baseline, label, adds)
        c, lk = rec["clean"], rec["leaky"]
        print(f"    CLEAN  AUC {c['auc']['pooled_base']:.4f}->{c['auc']['pooled_new']:.4f} "
              f"agg{c['auc']['agg_lift']:+.4f} mean{c['auc']['mean_fold_lift']:+.4f} "
              f"fpos{c['auc']['frac_pos']:.2f} p{c['auc']['boot_p_le0']:.3f} "
              f"| IC agg{c['ic']['agg_lift']:+.4f}")
        print(f"    LEAKY  AUC {lk['auc']['pooled_base']:.4f}->{lk['auc']['pooled_new']:.4f} "
              f"agg{lk['auc']['agg_lift']:+.4f} mean{lk['auc']['mean_fold_lift']:+.4f} "
              f"fpos{lk['auc']['frac_pos']:.2f} p{lk['auc']['boot_p_le0']:.3f}")
        conds = c["auc"]["conditions"]
        print(f"    clean-AUC 4-cond: agg>{0.005}={conds['c1_agg']} "
              f"mean>0.001={conds['c2_mean']} fpos>0.55={conds['c3_frac']} "
              f"bootp<0.05={conds['c4_boot']}  =>  "
              f"{'*** DEPLOY ***' if rec['DEPLOY'] else 'NO-GO'}")
        results.append(rec)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "feature_search_ab.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  saved -> {out}")
    gos = [r["label"] for r in results if r["DEPLOY"]]
    print(f"  DEPLOY verdicts (clean AUC, all 4 conditions): "
          f"{gos if gos else 'NONE — V7 confirmed saturated on this candidate pool'}")


if __name__ == "__main__":
    main()
