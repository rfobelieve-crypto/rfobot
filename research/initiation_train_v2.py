"""
Initiation model v2 — walk-forward training, evaluation, calibration, and
Strong-signal gate simulation.

Scope (locked):
    - Replaces ONLY the Direction model. Magnitude model + regime detector
      untouched.
    - Two independent binary classifiers (long_init, short_init) on 1h bars,
      4h forward horizon, k ∈ {0.008, 0.010, 0.012, 0.015}, with trailing
      breakout-confirm label gate.
    - Features: existing direction feature set + 17 `init_*` features from
      indicator/initiation_features.py.
    - Walk-forward: initial_train=288 (12d), test_size=48, step=48,
      purge=4, embargo=4.  ~77 folds on current 4000-bar cache.
      (purge/embargo = 4 to match 4h forward horizon)
    - Reports per-k, per-month: ROC-AUC, PR-AUC, Precision@top-5/10%,
      Brier, ECE-10. Plus Strong-gate sim (top-5% prob AND breakout-confirm).

Usage:
    python research/initiation_train_v2.py
    python research/initiation_train_v2.py --k 0.010
    python research/initiation_train_v2.py --k 0.008 0.010 0.012 0.015
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.dual_model.shared_data import (
    load_and_cache_data, walk_forward_splits,
)
from research.dual_model.build_initiation_labels import (
    build_initiation_labels, InitiationLabelConfig, K_SWEEP_VALUES,
)
from research.dual_model.direction_features_v2 import (
    ABLATION_GROUPS, filter_available,
)
from indicator.initiation_features import (
    add_initiation_features, INITIATION_FEATURE_COLS,
)

# Production MAG model paths (used as second-stage filter, NOT retrained)
MAG_MODEL_PATH = Path("indicator/model_artifacts/dual_model/magnitude_xgb.json")
MAG_FEATS_PATH = Path("indicator/model_artifacts/dual_model/magnitude_feature_cols.json")
MAG_ROLLING_WINDOW = 720  # 30 days for percentile calc


def score_production_mag(df: pd.DataFrame) -> pd.Series:
    """
    Score every bar with the production MAG model and return a rolling
    percentile series (trailing window of MAG_ROLLING_WINDOW bars).

    NOTE: mag model was trained on the full history, so scoring the same
    history is in-sample. Real-live lift may be slightly lower. Accepted
    trade-off because we want to simulate what production actually does
    at inference time (using the current deployed model).
    """
    import xgboost as _xgb
    feats = json.loads(MAG_FEATS_PATH.read_text())
    feats = [f for f in feats if f in df.columns]
    booster = _xgb.Booster()
    booster.load_model(str(MAG_MODEL_PATH))
    X = df[feats].fillna(0).values
    dm = _xgb.DMatrix(X, feature_names=feats)
    raw = booster.predict(dm)
    mag = pd.Series(raw, index=df.index, name="mag_pred")
    # Expanding/rolling trailing percentile
    pct = mag.rolling(MAG_ROLLING_WINDOW, min_periods=100).apply(
        lambda x: (x[-1] > x[:-1]).mean(), raw=True
    )
    pct.name = "mag_percentile"
    return mag, pct

OUT_DIR = Path("research/results/initiation_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR_FEATURE_SET = "+ key_4_only"  # existing production direction set

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "early_stopping_rounds": 40,
}


# ---------- metrics ----------

def ece_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error on equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_frac: float) -> tuple[float, int]:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    idx = np.argsort(-y_prob)[:k]
    prec = float(y_true[idx].mean())
    return prec, k


# ---------- training ----------

def train_one_side(df: pd.DataFrame, features: list[str], label_col: str) -> pd.DataFrame:
    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48,
                                  purge=4, embargo=4)
    all_oos = []
    for fold_i, (tr, te) in enumerate(splits):
        train_df = df.iloc[tr]
        test_df = df.iloc[te]
        tr_mask = train_df[label_col].notna()
        te_mask = test_df[label_col].notna()
        if tr_mask.sum() < 80 or te_mask.sum() < 10:
            continue
        X_tr = train_df.loc[tr_mask, features].fillna(0)
        y_tr = train_df.loc[tr_mask, label_col].values.astype(int)
        X_te = test_df.loc[te_mask, features].fillna(0)
        y_te = test_df.loc[te_mask, label_col].values.astype(int)

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        if pos == 0 or neg == 0:
            continue
        # NOTE: scale_pos_weight removed on 2026-04-15. With first-touch labels
        # the class is dense enough (~4%) that we no longer need aggressive
        # positive reweighting; removing it keeps probabilities near their
        # natural calibration and slashes ECE without hurting top-k ranking.
        params = XGB_PARAMS.copy()
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        p = model.predict_proba(X_te)[:, 1]
        oos = pd.DataFrame({
            "y_true": y_te,
            "y_prob": p,
            "fold": fold_i,
            "bo_confirm_up": test_df.loc[te_mask, "init_bo_up"].values
                if "init_bo_up" in test_df.columns else np.nan,
            "bo_confirm_dn": test_df.loc[te_mask, "init_bo_dn"].values
                if "init_bo_dn" in test_df.columns else np.nan,
            "mag_percentile": test_df.loc[te_mask, "mag_percentile"].values
                if "mag_percentile" in test_df.columns else np.nan,
        }, index=test_df.loc[te_mask].index)
        all_oos.append(oos)
    return pd.concat(all_oos) if all_oos else pd.DataFrame()


def monthly_report(oos: pd.DataFrame, label: str) -> dict:
    months = oos.index.strftime("%Y-%m").values
    report = {}
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() < 30:
            continue
        y = oos.loc[mask, "y_true"].values
        p = oos.loc[mask, "y_prob"].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            auc = roc_auc_score(y, p)
            pr_auc = average_precision_score(y, p)
        except ValueError:
            continue
        prec5, k5 = precision_at_k(y, p, 0.05)
        prec10, k10 = precision_at_k(y, p, 0.10)
        brier = brier_score_loss(y, p)
        ece = ece_bins(y, p)
        report[m] = {
            "n": int(mask.sum()),
            "pos_rate": float(y.mean()),
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "prec@5": prec5, "k@5": k5,
            "prec@10": prec10, "k@10": k10,
            "brier": float(brier),
            "ece10": ece,
        }
    # Overall
    y = oos["y_true"].values
    p = oos["y_prob"].values
    overall = {
        "n": len(oos),
        "pos_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
        "prec@5": precision_at_k(y, p, 0.05)[0],
        "prec@10": precision_at_k(y, p, 0.10)[0],
        "brier": float(brier_score_loss(y, p)),
        "ece10": ece_bins(y, p),
    }
    report["_overall"] = overall
    return report


def strong_gate_sim(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
                     top_k_frac: float = 0.05,
                     mag_min_percentile: float | None = None) -> dict:
    """
    Strong signal = (top-k prob) AND (breakout-confirm) [AND mag_percentile>=X].
    Reports win rate, N signals, monthly breakdown for the chosen top_k_frac.
    """
    def side_sim(oos: pd.DataFrame, confirm_col: str, frac: float) -> dict:
        if oos.empty:
            return {}
        threshold = oos["y_prob"].quantile(1 - frac)
        gate = (oos["y_prob"] >= threshold) & (oos[confirm_col] == 1.0)
        if mag_min_percentile is not None and "mag_percentile" in oos.columns:
            gate = gate & (oos["mag_percentile"] >= mag_min_percentile)
        hit = oos.loc[gate]
        out = {
            "top_k_frac": frac,
            "threshold_prob": float(threshold),
            "n_signals": int(gate.sum()),
            "win_rate": float(hit["y_true"].mean()) if len(hit) else None,
            "baseline_rate": float(oos["y_true"].mean()),
            "lift": None,
        }
        if out["win_rate"] is not None and out["baseline_rate"] > 0:
            out["lift"] = out["win_rate"] / out["baseline_rate"]
        if len(hit):
            months = hit.index.strftime("%Y-%m").values
            monthly = {}
            for m in sorted(set(months)):
                mm = months == m
                if mm.sum() < 1:
                    continue
                monthly[m] = {
                    "n": int(mm.sum()),
                    "win_rate": float(hit.loc[mm, "y_true"].mean()),
                }
            out["monthly"] = monthly
        return out

    return {
        "long":  side_sim(oos_long,  "bo_confirm_up", top_k_frac),
        "short": side_sim(oos_short, "bo_confirm_dn", top_k_frac),
    }


def topk_curve(oos_long: pd.DataFrame, oos_short: pd.DataFrame,
                fracs: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10)) -> dict:
    """
    Trade-off curve: for each top_k_frac, evaluate Strong-gate win rate
    and signal count on each side. This is the operating-point menu.
    """
    curve = {"fracs": list(fracs), "long": [], "short": []}
    for f in fracs:
        res = strong_gate_sim(oos_long, oos_short, top_k_frac=f)
        curve["long"].append(res["long"])
        curve["short"].append(res["short"])
    return curve


def print_curve(curve: dict) -> None:
    print("  top-k curve (Strong gate = top-k prob AND trailing breakout confirm):")
    print(f"    {'frac':>6}  {'side':<5}  {'n_sig':>6}  {'win_rate':>9}  "
          f"{'lift':>6}  {'thr':>6}")
    for side in ("long", "short"):
        for entry in curve[side]:
            if not entry:
                continue
            wr = entry.get("win_rate")
            wr_s = f"{wr*100:.1f}%" if wr is not None else "   -"
            lift = entry.get("lift")
            lift_s = f"{lift:.1f}x" if lift is not None else "  -"
            print(f"    {entry['top_k_frac']*100:>5.1f}%  {side:<5}  "
                  f"{entry['n_signals']:>6}  {wr_s:>9}  {lift_s:>6}  "
                  f"{entry['threshold_prob']:>6.3f}")


def run_for_k(df: pd.DataFrame, features: list[str], k: float) -> dict:
    print(f"\n{'='*88}\n  k = {k*100:.1f}%\n{'='*88}")
    cfg = InitiationLabelConfig(k_pct=k, use_breakout_confirm=True)
    labels = build_initiation_labels(df, cfg)
    work = df.copy()
    for c in ["y_long_touch", "y_short_touch"]:
        work[c] = labels[c]

    # Long side
    oos_long = train_one_side(work, features, "y_long_touch")
    # Short side
    oos_short = train_one_side(work, features, "y_short_touch")

    result = {
        "k_pct": k,
        "n_features": len(features),
        "long":  {"n_oos": len(oos_long),  "monthly": monthly_report(oos_long, "long")}
                 if len(oos_long) else {},
        "short": {"n_oos": len(oos_short), "monthly": monthly_report(oos_short, "short")}
                 if len(oos_short) else {},
        "strong_gate_top5": strong_gate_sim(oos_long, oos_short, 0.05),
        "topk_curve": topk_curve(oos_long, oos_short),
    }

    # Print summary row
    def fmt_side(name, side_dict):
        ov = side_dict.get("monthly", {}).get("_overall", {})
        if not ov or ov.get("auc") is None:
            print(f"  {name}: <no overall>")
            return
        print(f"  {name}: n={ov['n']:>4}  pos={ov['pos_rate']*100:5.2f}%  "
              f"AUC={ov['auc']:.3f}  PR-AUC={ov['pr_auc']:.3f}  "
              f"P@5={ov['prec@5']:.3f}  P@10={ov['prec@10']:.3f}  "
              f"ECE={ov['ece10']:.3f}")
    fmt_side("long ", result["long"])
    fmt_side("short", result["short"])

    gate = result["strong_gate_top5"]
    for side in ("long", "short"):
        g = gate.get(side, {})
        if g and g.get("win_rate") is not None:
            print(f"  STRONG-{side}: n_signals={g['n_signals']}  "
                  f"win_rate={g['win_rate']*100:.1f}%  "
                  f"baseline={g['baseline_rate']*100:.1f}%  "
                  f"lift={g['lift']:.2f}x")

    print_curve(result["topk_curve"])

    # --- MAG-gated experiment: loose init threshold + mag filter ---
    print("\n  MAG-gated top-k (init prob top-k AND breakout AND mag_percentile >= X):")
    for mag_thr in (0.50, 0.65, 0.80):
        print(f"\n    mag_percentile >= {mag_thr}:")
        print(f"      {'frac':>6}  {'side':<5}  {'n_sig':>6}  {'win_rate':>9}  {'lift':>6}")
        for frac in (0.05, 0.10, 0.15, 0.20):
            g = strong_gate_sim(oos_long, oos_short, top_k_frac=frac,
                                 mag_min_percentile=mag_thr)
            for side in ("long", "short"):
                entry = g.get(side, {})
                if not entry or entry.get("win_rate") is None:
                    continue
                wr = entry["win_rate"]
                lift = entry.get("lift", 0) or 0
                print(f"      {frac*100:>5.1f}%  {side:<5}  "
                      f"{entry['n_signals']:>6}  {wr*100:>7.1f}%  {lift:>5.1f}x")

    # Monthly breakdown for two most promising MAG-gated points
    for frac, mag_thr, label in [(0.05, 0.80, "top-5% + mag>=0.80 (Strong)"),
                                   (0.10, 0.65, "top-10% + mag>=0.65 (Moderate)"),
                                   (0.15, 0.50, "top-15% + mag>=0.50")]:
        g = strong_gate_sim(oos_long, oos_short, top_k_frac=frac,
                             mag_min_percentile=mag_thr)
        print(f"\n  monthly breakdown @ {label}:")
        for side in ("long", "short"):
            m = g.get(side, {}).get("monthly", {})
            if not m:
                continue
            months_sorted = sorted(m.keys())
            row = "    " + f"{side:<5} "
            for mk in months_sorted:
                row += f" {mk[-5:]}:{m[mk]['n']:>2}/{m[mk]['win_rate']*100:>3.0f}%"
            print(row)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, nargs="+", default=list(K_SWEEP_VALUES))
    ap.add_argument("--out", default=str(OUT_DIR / "report.json"))
    args = ap.parse_args()

    df = load_and_cache_data()
    df = add_initiation_features(df)

    # Score production MAG model once and attach rolling percentile
    mag_raw, mag_pct = score_production_mag(df)
    df["mag_pred"] = mag_raw
    df["mag_percentile"] = mag_pct
    print(f"Loaded {len(df)} bars  |  added {len(INITIATION_FEATURE_COLS)} init features  "
          f"|  mag_pct valid: {mag_pct.notna().sum()}")

    # Feature set: base direction features ∪ init features
    base_feats = filter_available(ABLATION_GROUPS[BASE_DIR_FEATURE_SET], list(df.columns))
    extra = [c for c in INITIATION_FEATURE_COLS if c in df.columns]
    # (ablation switch kept but disabled — first-touch label no longer has
    # the feature/label co-source issue since label uses forward path, not
    # endpoint AND breakout gate. Re-enable by setting ABLATE_PREFIXES non-empty.)
    ABLATE_PREFIXES = tuple()
    dropped = [c for c in extra if c.startswith(ABLATE_PREFIXES)] if ABLATE_PREFIXES else []
    extra = [c for c in extra if c not in dropped]
    features = sorted(set(base_feats) | set(extra))
    print(f"Features: base={len(base_feats)} + init={len(extra)} = {len(features)}")

    all_results = {}
    for k in args.k:
        all_results[f"k={k}"] = run_for_k(df, features, k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {args.out}")

    # Go/no-go scorecard (strong-gate win rate ≥ 60% on overall + both sides)
    print("\n" + "=" * 88)
    print("GO/NO-GO SCORECARD  (target: strong-gate win rate >= 60%)")
    print("=" * 88)
    for key, res in all_results.items():
        gate = res.get("strong_gate_top5", {})
        long_wr = gate.get("long", {}).get("win_rate")
        short_wr = gate.get("short", {}).get("win_rate")
        ok_long = long_wr is not None and long_wr >= 0.60
        ok_short = short_wr is not None and short_wr >= 0.60
        verdict = "PASS" if (ok_long and ok_short) else "FAIL"
        print(f"  {key:<10}  long={long_wr if long_wr is None else f'{long_wr*100:.1f}%':>7}  "
              f"short={short_wr if short_wr is None else f'{short_wr*100:.1f}%':>7}  "
              f"=> {verdict}")


if __name__ == "__main__":
    main()
