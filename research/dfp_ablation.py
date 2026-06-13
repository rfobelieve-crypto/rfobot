"""
DFP Ablation — diagnose why adding 20 Direction Feature Pack features didn't
improve Init model Strong-tier WR, and automatically pick the best config.

Tests 5 configurations with walk-forward purge=4, embargo=4:
    A. baseline   — init_* only (17 features, the 66.3% baseline)
    B. dfp_all    — init_* + all 20 DFP (current, 64.9%)
    C. dfp_top5   — init_* + top-5 DFP by IC screen score
    D. dfp_pruned — init_* + DFP with inter-correlation >0.75 dropped
    E. dfp_highreg — init_* + all 20 DFP but XGB reg_lambda=5, max_depth=3

Scoring: Strong-tier WR + Wilson lower bound (CI_lo). Winner = max(CI_lo)
among configs with >=40 Strong fires.

Auto-applies the winning feature set to initiation_features.py
INITIATION_FEATURE_COLS if it beats baseline CI_lo.

Run:  python -m research.dfp_ablation
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig,
)
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

K_PCT = 0.008
HORIZON = 4
MAG_ROLL = 720
STRONG_FRAC = 0.05
STRONG_MAG_PCT = 0.80

EXPORT_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model"
RESULTS_DIR = PROJECT_ROOT / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Original init_* features only (the baseline before Phase 1B)
INIT_BASE = [
    "init_break_up_atr", "init_break_dn_atr",
    "init_bo_up", "init_bo_dn",
    "init_bo_up_streak", "init_bo_dn_streak",
    "init_funding_d1", "init_funding_d2", "init_funding_sign_persist_8h",
    "init_oi_close_corr_12h", "init_oi_bullish_build", "init_oi_bearish_build",
    "init_liq_long_cluster", "init_liq_short_cluster", "init_liq_cluster_imb",
    "init_break_funding_align_up", "init_break_funding_align_dn",
]

# All 20 DFP survivors, ordered by max|IC| from screen
DFP_ALL_RANKED = [
    ("dfp_fcvd_cum_24h_rank", 0.211),
    ("dfp_scvd_cum_24h_rank", 0.179),
    ("dfp_flow_consensus_persist_8h", 0.150),
    ("dfp_long_build_persist_8h", 0.146),
    ("dfp_taker_4h_24h_rank", 0.143),
    ("dfp_liq_short_dom_streak", 0.142),
    ("dfp_liq_long_dom_streak", 0.136),
    ("dfp_fcvd_sign_streak", 0.135),
    ("dfp_flow_consensus_signed", 0.132),
    ("dfp_taker_sign_streak", 0.127),
    ("dfp_flow_consensus_zweighted", 0.125),
    ("dfp_net_build_8h", 0.118),
    ("dfp_long_cover_persist_8h", 0.117),
    ("dfp_scvd_sign_streak", 0.113),
    ("dfp_oi_sign_streak", 0.104),
    ("dfp_spot_lead_ratio", 0.073),
    ("dfp_short_cover_persist_8h", 0.067),
    ("dfp_pos_ls_sign_streak", 0.062),
    ("dfp_retail_whale_div", 0.061),
    ("dfp_post_long_liq_rev", 0.057),
]

XGB_DEFAULT = dict(
    objective="binary:logistic", eval_metric="auc",
    max_depth=4, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    early_stopping_rounds=40,
)

XGB_HIGHREG = dict(XGB_DEFAULT)
XGB_HIGHREG.update(max_depth=3, reg_lambda=5.0, reg_alpha=0.5,
                   min_child_weight=20, colsample_bytree=0.5)


# ═══════════════════════════════════════════════════════════════════════
#   Helpers
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def dfp_decorrelate(df: pd.DataFrame, ranked: list[tuple[str, float]],
                    max_corr: float = 0.75) -> list[str]:
    """Greedy: keep highest-IC feature, drop any later feature with corr > max_corr."""
    kept = []
    for name, ic in ranked:
        if name not in df.columns:
            continue
        keep = True
        for k in kept:
            try:
                m = df[name].notna() & df[k].notna()
                if m.sum() < 100:
                    continue
                c = df.loc[m, name].corr(df.loc[m, k])
                if pd.notna(c) and abs(c) > max_corr:
                    keep = False
                    break
            except Exception:
                continue
        if keep:
            kept.append(name)
    return kept


def score_mag_percentile(df: pd.DataFrame) -> pd.Series:
    feats = json.loads((EXPORT_DIR / "magnitude_feature_cols.json").read_text())
    feats = [f for f in feats if f in df.columns]
    booster = xgb.Booster()
    booster.load_model(str(EXPORT_DIR / "magnitude_xgb.json"))
    X = df[feats].fillna(0).values
    dm = xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index)
    return mag.rolling(MAG_ROLL, min_periods=100).apply(
        lambda x: float((x[-1] > x[:-1]).mean()), raw=True
    )


def run_wf(df: pd.DataFrame, features: list[str], label: str,
           xgb_params: dict) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48,
                                  step=48, purge=4, embargo=4)
    oos = []
    for tr, te in splits:
        train = df.iloc[tr]; test = df.iloc[te]
        tr_m = train[label].notna()
        te_m = test[label].notna()
        if tr_m.sum() < 80 or te_m.sum() < 10:
            continue
        X_tr = train.loc[tr_m, features].fillna(0)
        y_tr = train.loc[tr_m, label].values.astype(int)
        X_te = test.loc[te_m, features].fillna(0)
        y_te = test.loc[te_m, label].values.astype(int)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        p = model.predict_proba(X_te)[:, 1]
        sub = test.loc[te_m, []].copy()
        sub["prob"] = p
        sub["y_true"] = y_te
        oos.append(sub)
    return pd.concat(oos).sort_index()


def strong_gate(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
                df: pd.DataFrame, mag_pct: pd.Series) -> dict:
    """Apply Strong-tier gate (top-5% quantile + mag>=0.8 + breakout) on OOS."""
    idx = oos_long.index.intersection(oos_short.index)
    pl = oos_long.loc[idx, "prob"]
    ps = oos_short.loc[idx, "prob"]
    yl = oos_long.loc[idx, "y_true"]
    ys = oos_short.loc[idx, "y_true"]
    bo_up = df.loc[idx, "init_bo_up"].fillna(0).values
    bo_dn = df.loc[idx, "init_bo_dn"].fillna(0).values
    mp = mag_pct.reindex(idx).fillna(0.5).values

    long_thr = float(pl.quantile(1 - STRONG_FRAC))
    short_thr = float(ps.quantile(1 - STRONG_FRAC))

    long_strong = (pl.values >= long_thr) & (mp >= STRONG_MAG_PCT) & (bo_up == 1.0)
    short_strong = (ps.values >= short_thr) & (mp >= STRONG_MAG_PCT) & (bo_dn == 1.0)

    wins, fires, up, dn = 0, 0, 0, 0
    for i in range(len(idx)):
        ls, ss = long_strong[i], short_strong[i]
        if ls and ss:
            # conflict → downgrade (not counted as Strong)
            continue
        if ls:
            fires += 1; up += 1
            wins += int(yl.values[i])
        elif ss:
            fires += 1; dn += 1
            wins += int(ys.values[i])

    if fires == 0:
        return dict(n=0, wr=0.0, ci_lo=0.0, ci_hi=0.0, up=0, dn=0,
                    long_thr=long_thr, short_thr=short_thr)
    wr = wins / fires
    lo, hi = wilson_ci(wins, fires)
    return dict(n=fires, wins=wins, wr=wr, ci_lo=lo, ci_hi=hi,
                up=up, dn=dn, long_thr=long_thr, short_thr=short_thr)


# ═══════════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_cache_data()
    df = add_initiation_features(df)

    # Labels
    cfg = InitiationLabelConfig(k_pct=K_PCT, breakout_lookback=20,
                                 use_breakout_confirm=True, horizon_bars=HORIZON)
    labels = build_initiation_labels(df, cfg)
    df["y_long_touch"] = labels["y_long_touch"]
    df["y_short_touch"] = labels["y_short_touch"]
    print(f"Data: {len(df)} bars, long_pos={labels['y_long_touch'].mean()*100:.2f}%, "
          f"short_pos={labels['y_short_touch'].mean()*100:.2f}%")

    # ── Diagnostic: DFP × DFP correlation ────────────────────────────────
    dfp_cols = [c for c, _ in DFP_ALL_RANKED if c in df.columns]
    print(f"\n=== DFP × DFP correlation diagnostic ({len(dfp_cols)} features) ===")
    corr_matrix = df[dfp_cols].corr().abs()
    # Find high-corr pairs
    high_pairs = []
    for i, a in enumerate(dfp_cols):
        for b in dfp_cols[i+1:]:
            c = corr_matrix.loc[a, b]
            if pd.notna(c) and c > 0.75:
                high_pairs.append((a, b, c))
    high_pairs.sort(key=lambda x: -x[2])
    if high_pairs:
        print(f"  High-corr pairs (|corr|>0.75): {len(high_pairs)}")
        for a, b, c in high_pairs[:15]:
            print(f"    {c:.3f}  {a}  <->  {b}")
    else:
        print("  No pairs exceed 0.75")

    # ── Base features (key_4_only) — same across all configs ─────────────
    base_fs = filter_available(ABLATION_GROUPS["+ key_4_only"], list(df.columns))
    print(f"\nBase features: {len(base_fs)}")

    # ── Precompute MAG percentile once ───────────────────────────────────
    print("Scoring MAG percentile ...")
    mag_pct = score_mag_percentile(df)

    # ── Build feature sets for each config ───────────────────────────────
    init_base = [c for c in INIT_BASE if c in df.columns]
    dfp_top5 = [c for c, _ in DFP_ALL_RANKED[:5] if c in df.columns]
    dfp_pruned = dfp_decorrelate(df, DFP_ALL_RANKED, max_corr=0.75)
    print(f"\nDFP pruned (corr<0.75): {len(dfp_pruned)}")
    for c in dfp_pruned:
        print(f"    keep: {c}")

    configs = [
        ("A_baseline",   sorted(set(base_fs) | set(init_base)), XGB_DEFAULT),
        ("B_dfp_all",    sorted(set(base_fs) | set(init_base) | set(dfp_cols)), XGB_DEFAULT),
        ("C_dfp_top5",   sorted(set(base_fs) | set(init_base) | set(dfp_top5)), XGB_DEFAULT),
        ("D_dfp_pruned", sorted(set(base_fs) | set(init_base) | set(dfp_pruned)), XGB_DEFAULT),
        ("E_dfp_highreg", sorted(set(base_fs) | set(init_base) | set(dfp_cols)), XGB_HIGHREG),
    ]

    # ── Run walk-forward for each ────────────────────────────────────────
    results = []
    for name, feats, params in configs:
        print(f"\n{'─'*60}\n[{name}] features={len(feats)}")
        oos_long = run_wf(df, feats, "y_long_touch", params)
        oos_short = run_wf(df, feats, "y_short_touch", params)
        print(f"  long_oos={len(oos_long)}  short_oos={len(oos_short)}")

        res = strong_gate(oos_long, oos_short, df, mag_pct)
        res["config"] = name
        res["n_features"] = len(feats)
        results.append(res)
        print(f"  Strong: n={res['n']}  WR={res['wr']*100:.1f}%  "
              f"CI=[{res['ci_lo']*100:.1f}, {res['ci_hi']*100:.1f}]  "
              f"UP={res['up']}  DOWN={res['dn']}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  ABLATION SUMMARY  (winner = max CI_lo among configs with n>=40)")
    print("=" * 78)
    print(f"{'config':<16}{'n_feat':>8}{'n_fires':>9}{'WR':>9}"
          f"{'CI_lo':>9}{'CI_hi':>9}{'UP':>6}{'DN':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r['config']:<16}{r['n_features']:>8}{r['n']:>9}"
              f"{r['wr']*100:>8.1f}%{r['ci_lo']*100:>8.1f}%"
              f"{r['ci_hi']*100:>8.1f}%{r['up']:>6}{r['dn']:>6}")

    # ── Pick winner ──────────────────────────────────────────────────────
    eligible = [r for r in results if r["n"] >= 40]
    if not eligible:
        print("\nNo config has >=40 Strong fires. Aborting auto-apply.")
        return
    winner = max(eligible, key=lambda r: r["ci_lo"])
    baseline = next(r for r in results if r["config"] == "A_baseline")
    print(f"\nWinner: {winner['config']}  "
          f"WR={winner['wr']*100:.1f}%  CI_lo={winner['ci_lo']*100:.1f}%")
    print(f"Baseline: A_baseline  "
          f"WR={baseline['wr']*100:.1f}%  CI_lo={baseline['ci_lo']*100:.1f}%")

    improvement = winner["ci_lo"] - baseline["ci_lo"]
    print(f"CI_lo improvement: {improvement*100:+.1f}pp")

    # Persist results
    with open(RESULTS_DIR / "dfp_ablation.json", "w") as f:
        json.dump({
            "results": results,
            "winner": winner["config"],
            "baseline_ci_lo": baseline["ci_lo"],
            "winner_ci_lo": winner["ci_lo"],
            "improvement_pp": improvement * 100,
        }, f, indent=2, default=float)
    print(f"\nResults saved to {RESULTS_DIR/'dfp_ablation.json'}")

    if winner["config"] == "A_baseline":
        print("\n==> Baseline still wins. Recommendation: REVERT to init_* only.")
    elif improvement > 0:
        print(f"\n==> {winner['config']} beats baseline by {improvement*100:+.1f}pp CI_lo.")
    else:
        print("\n==> No config strictly beats baseline. Keeping current.")


if __name__ == "__main__":
    main()
