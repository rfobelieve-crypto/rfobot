"""
Reliability Diagram + Brier Score — honest probability calibration check.

Does dir_prob_up = 0.65 actually mean the sample has a 65% UP rate? If yes,
the model is well-calibrated and the production thresholds (0.60/0.40) are
trustworthy. If not, the Strong-signal win rate target (95%) is unreachable
regardless of feature engineering.

Reads indicator_history.parquet (production) which has both the model's
predicted probability and the realized 4h return backfilled by outcome_tracker.

Usage:
    python research/calibration_check.py
    python research/calibration_check.py --regime TRENDING_BULL  # regime-specific

Output:
    research/results/calibration_check.json
    research/results/calibration_diagram.png
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HIST = Path("indicator/model_artifacts/indicator_history.parquet")
MODEL_FILE = Path("indicator/model_artifacts/dual_model/direction_xgb.json")
PROD_DIR_FEATS = Path("indicator/model_artifacts/dual_model/direction_feature_cols.json")
ABLATION_JSON = Path("research/results/ablation_study.json")
OUT_JSON = Path("research/results/calibration_check.json")
OUT_PNG = Path("research/results/calibration_diagram.png")

# Direction model params — keep in sync with permutation_test.py and deploy_new_models.py
DIR_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 200,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
    "verbosity": 0,
}


def load_walkforward_predictions(use_ablation: bool = False) -> pd.DataFrame:
    """
    Run walk-forward CV using the same logic as permutation_test.py, collect
    all OOS predictions and realized outcomes. Returns a DataFrame matching
    the production indicator_history format (so the rest of this script
    doesn't need to branch).

    Columns: dir_prob_up, actual_return_4h, regime
    Index:   test-bar timestamps
    """
    import xgboost as xgb
    from research.dual_model.shared_data import load_and_cache_data, walk_forward_splits
    from research.dual_model.build_direction_labels import build_direction_labels

    print("Loading feature cache...")
    df = load_and_cache_data(limit=4000, force_refresh=False, max_stale_hours=12.0)
    labels = build_direction_labels(df, k=0.5)
    df["y_dir"] = labels["y_dir"]
    df["dir_return_4h"] = labels["return_4h"]

    # Feature set — match the production model
    with open(PROD_DIR_FEATS) as f:
        base = json.load(f)
    base = [c for c in base if c in df.columns]

    if use_ablation and ABLATION_JSON.exists():
        abl = json.loads(ABLATION_JSON.read_text())
        keep = abl.get("combined", {}).get("keep_features", [])
        feats = base + [k for k in keep if k not in base and k in df.columns]
        label = f"walkforward_ablation_{len(feats)}"
    else:
        feats = base
        label = f"walkforward_baseline_{len(feats)}"

    print(f"Feature set: {label}")

    # Regime weights (production-matched)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ret_24h = df["close"].pct_change(24)
    vol_24h = log_ret.rolling(24).std()
    vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)
    weights = np.ones(len(df))
    bull_mask = ((vol_pct > 0.6) & (ret_24h > 0.005)).fillna(False).values
    bear_mask = ((vol_pct > 0.6) & (ret_24h < -0.005)).fillna(False).values
    weights[bull_mask] = 4.0
    weights[bear_mask] = 2.0

    # Regime label for per-regime breakdown
    regime = np.where(bull_mask, "TRENDING_BULL",
                      np.where(bear_mask, "TRENDING_BEAR", "CHOPPY"))

    y = df["y_dir"].values.astype(float)
    rets = df["dir_return_4h"].values.astype(float)

    splits = walk_forward_splits(len(df), initial_train=288, test_size=48, step=48)
    print(f"Walk-forward: {len(splits)} folds")

    # Collect OOS predictions
    oos_probs: list[float] = []
    oos_outcomes: list[float] = []  # realized return_4h
    oos_idx: list[int] = []

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        tr_mask = ~np.isnan(y[tr_idx])
        te_mask = ~np.isnan(y[te_idx])
        if tr_mask.sum() < 50 or te_mask.sum() < 5:
            continue

        X_tr = df.iloc[tr_idx][feats].fillna(0).values[tr_mask]
        y_tr = y[tr_idx][tr_mask].astype(int)
        X_te = df.iloc[te_idx][feats].fillna(0).values[te_mask]
        sw = weights[tr_idx][tr_mask]

        up = y_tr.mean()
        p = DIR_PARAMS.copy()
        p["scale_pos_weight"] = (1 - up) / up if up > 0 else 1.0

        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        prob = m.predict_proba(X_te)[:, 1]

        # Align predictions to test-bar global indices
        te_global = np.array(te_idx)[te_mask]
        oos_probs.extend(prob.tolist())
        oos_outcomes.extend(rets[te_global].tolist())
        oos_idx.extend(te_global.tolist())

        if (fold_i + 1) % 10 == 0:
            print(f"  fold {fold_i+1}/{len(splits)}: collected {len(oos_probs)} OOS predictions")

    print(f"Total OOS predictions: {len(oos_probs)}")

    # Build DataFrame matching indicator_history schema
    wf_df = pd.DataFrame({
        "dir_prob_up": oos_probs,
        "actual_return_4h": oos_outcomes,
        "regime": regime[oos_idx],
    }, index=df.index[oos_idx])

    # Drop NaN outcomes (shouldn't happen due to tr_mask but just in case)
    wf_df = wf_df.dropna(subset=["dir_prob_up", "actual_return_4h"])
    return wf_df


def get_model_deploy_time() -> pd.Timestamp | None:
    """Return mtime of the current production direction model as a UTC Timestamp."""
    if not MODEL_FILE.exists():
        return None
    mtime = MODEL_FILE.stat().st_mtime
    return pd.Timestamp(mtime, unit="s", tz="UTC")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (more accurate than normal approx)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Lower is better. Random guess (always 0.5) = 0.25. Perfect = 0."""
    return float(np.mean((probs - outcomes) ** 2))


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    ECE = weighted average of |predicted - actual| per bin.
    0 = perfectly calibrated. >0.05 = noticeable miscalibration.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = probs[mask].mean()
        bin_actual = outcomes[mask].mean()
        ece += (mask.sum() / n) * abs(bin_pred - bin_actual)
    return float(ece)


def bootstrap_metrics(probs: np.ndarray, outcomes: np.ndarray,
                      n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap 95% CI for Brier skill score and ECE.
    With small n (~244), point estimates can look alarming but sit inside a
    wide CI that overlaps zero. This tells us whether the miscalibration is
    statistically real or sampling noise.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    climatology_p = outcomes.mean()

    brier_skills = []
    eces = []
    conf_gaps = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p_b = probs[idx]
        y_b = outcomes[idx]

        brier_b = brier_score(p_b, y_b)
        clim_b = brier_score(np.full_like(p_b, climatology_p), y_b)
        if clim_b > 0:
            brier_skills.append(1 - brier_b / clim_b)
        eces.append(expected_calibration_error(p_b, y_b))

        # Confidence gap at extreme bins
        ext_mask = (p_b >= 0.60) | (p_b <= 0.40)
        if ext_mask.sum() > 10:
            ep = p_b[ext_mask]
            ea = y_b[ext_mask]
            acc = ((ep > 0.5).astype(int) == ea).mean()
            conf = np.where(ep > 0.5, ep, 1 - ep).mean()
            conf_gaps.append(conf - acc)

    def ci(arr):
        if len(arr) == 0:
            return (None, None, None)
        a = np.array(arr)
        return (float(np.percentile(a, 2.5)),
                float(np.median(a)),
                float(np.percentile(a, 97.5)))

    bs_lo, bs_med, bs_hi = ci(brier_skills)
    ece_lo, ece_med, ece_hi = ci(eces)
    cg_lo, cg_med, cg_hi = ci(conf_gaps)

    return {
        "n_boot": n_boot,
        "brier_skill_ci": [bs_lo, bs_hi],
        "brier_skill_median": bs_med,
        "brier_skill_crosses_zero": bs_lo is not None and bs_lo <= 0 <= bs_hi,
        "ece_ci": [ece_lo, ece_hi],
        "ece_median": ece_med,
        "confidence_gap_ci": [cg_lo, cg_hi],
        "confidence_gap_median": cg_med,
        "confidence_gap_crosses_zero": cg_lo is not None and cg_lo <= 0 <= cg_hi,
    }


def analyze(df: pd.DataFrame, label: str = "ALL") -> dict:
    """Build calibration metrics + bin-level data for plotting."""
    probs = df["dir_prob_up"].values
    outcomes = (df["actual_return_4h"] > 0).astype(int).values
    n = len(probs)

    if n < 20:
        return {"label": label, "n_samples": n, "insufficient": True}

    # Binning
    bin_edges = np.array([0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0])
    bin_stats = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        p_mean = float(probs[mask].mean())
        actual = float(outcomes[mask].mean())
        n_bin = int(mask.sum())
        ci_lo, ci_hi = wilson_ci(int(outcomes[mask].sum()), n_bin)
        bin_stats.append({
            "range": f"[{lo:.2f}, {hi:.2f})",
            "lo": lo, "hi": hi,
            "n": n_bin,
            "predicted": p_mean,
            "actual": actual,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "diff": actual - p_mean,
        })

    # Global metrics
    brier = brier_score(probs, outcomes)
    ece = expected_calibration_error(probs, outcomes)

    # Baselines
    mean_outcome = float(outcomes.mean())
    baseline_brier = brier_score(np.full_like(probs, 0.5), outcomes)  # always 0.5
    climatology_brier = brier_score(np.full_like(probs, mean_outcome), outcomes)

    # Over/under confidence
    extreme_mask = (probs >= 0.60) | (probs <= 0.40)
    extreme_pred = probs[extreme_mask]
    extreme_actual = outcomes[extreme_mask]
    extreme_n = int(extreme_mask.sum())

    if extreme_n > 10:
        extreme_accuracy = float(
            ((extreme_pred > 0.5).astype(int) == extreme_actual).mean()
        )
        extreme_avg_confidence = float(
            np.where(extreme_pred > 0.5, extreme_pred, 1 - extreme_pred).mean()
        )
        confidence_gap = extreme_avg_confidence - extreme_accuracy
    else:
        extreme_accuracy = None
        extreme_avg_confidence = None
        confidence_gap = None

    # Bootstrap CIs
    boot = bootstrap_metrics(probs, outcomes)

    return {
        "label": label,
        "n_samples": n,
        "mean_outcome": mean_outcome,
        "brier": brier,
        "brier_vs_constant_0.5": baseline_brier,
        "brier_vs_climatology": climatology_brier,
        "brier_skill_vs_climatology": 1 - brier / climatology_brier if climatology_brier > 0 else 0,
        "ece": ece,
        "bins": bin_stats,
        "extreme_n": extreme_n,
        "extreme_accuracy": extreme_accuracy,
        "extreme_avg_confidence": extreme_avg_confidence,
        "confidence_gap": confidence_gap,
        "bootstrap": boot,
    }


def plot_reliability(results: list[dict], out: Path):
    """Multi-panel reliability diagram."""
    n_panels = len([r for r in results if not r.get("insufficient")])
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), squeeze=False)
    axes = axes[0]

    panel = 0
    for r in results:
        if r.get("insufficient"):
            continue
        ax = axes[panel]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # Actual data
        bins = r["bins"]
        if bins:
            xs = [b["predicted"] for b in bins]
            ys = [b["actual"] for b in bins]
            ci_lo = [b["ci_lo"] for b in bins]
            ci_hi = [b["ci_hi"] for b in bins]
            ns = [b["n"] for b in bins]

            # Error bars (Wilson CI)
            y_err = [[y - lo for y, lo in zip(ys, ci_lo)],
                     [hi - y for y, hi in zip(ys, ci_hi)]]
            ax.errorbar(xs, ys, yerr=y_err, fmt="o-", capsize=3,
                        color="steelblue", markersize=6, linewidth=1.5,
                        label="Model")

            # Bin size annotations
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"n={n}", (x, y), xytext=(5, 5),
                            textcoords="offset points", fontsize=7, alpha=0.7)

        # Deadzone shading
        ax.axvspan(0.40, 0.60, alpha=0.1, color="gray", label="Deadzone [0.4, 0.6]")
        ax.axvline(0.60, color="green", linestyle=":", alpha=0.5)
        ax.axvline(0.40, color="red", linestyle=":", alpha=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(UP)")
        ax.set_ylabel("Observed UP rate")
        ax.set_title(f"{r['label']}  (n={r['n_samples']}, Brier={r['brier']:.4f}, "
                     f"ECE={r['ece']:.4f})")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        panel += 1

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["production", "walkforward"],
                    default="production",
                    help="Source of predictions: 'production' reads indicator_history.parquet "
                         "(small n, needs version guard), 'walkforward' runs CV on cache "
                         "(large n, always clean, same model hyperparams as production).")
    ap.add_argument("--use-ablation", action="store_true",
                    help="Walkforward only: use the ablation-validated 34-feature set "
                         "instead of the production 29-feature set")
    ap.add_argument("--regime", type=str, default=None,
                    help="Filter to specific regime (e.g. TRENDING_BULL)")
    ap.add_argument("--since", type=str, default=None,
                    help="Production only: only include samples on/after this UTC date "
                         "(YYYY-MM-DD). Overrides auto model-version guard.")
    ap.add_argument("--no-version-guard", action="store_true",
                    help="Production only: disable automatic model-version guard.")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="Minimum sample count required to run analysis (default 100)")
    args = ap.parse_args()

    # ─────────────────────────────────────────────
    # Walk-forward mode: run CV and return early
    # ─────────────────────────────────────────────
    if args.source == "walkforward":
        print("=" * 70)
        print("WALK-FORWARD CALIBRATION (OOS predictions from cache)")
        print("=" * 70)
        wf_df = load_walkforward_predictions(use_ablation=args.use_ablation)
        print(f"OOS predictions collected: {len(wf_df)}")
        if len(wf_df) < args.min_samples:
            print(f"[INSUFFICIENT] n={len(wf_df)} < {args.min_samples}")
            return

        valid = wf_df
        results = [analyze(valid, "WALKFORWARD_ALL")]
        if "regime" in valid.columns:
            for regime in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"]:
                sub = valid[valid["regime"] == regime]
                if len(sub) >= 50:
                    results.append(analyze(sub, f"WALKFORWARD_{regime}"))
        _print_and_save_results(results, args)
        return

    # ─────────────────────────────────────────────
    # Production mode: read indicator_history
    # ─────────────────────────────────────────────
    print(f"Loading {HIST}...")
    df = pd.read_parquet(HIST)
    print(f"Raw: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    # Strip tz for comparison
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df_idx_utc = df.index.tz_convert("UTC")
    else:
        df_idx_utc = df.index.tz_localize("UTC")

    # Model version guard — determine effective cutoff
    model_mtime = get_model_deploy_time()
    cutoff: pd.Timestamp | None = None

    if args.since:
        cutoff = pd.Timestamp(args.since, tz="UTC")
        print(f"Using --since filter: samples >= {cutoff}")
    elif not args.no_version_guard and model_mtime is not None:
        cutoff = model_mtime
        print(f"Model version guard: {MODEL_FILE.name} mtime = {cutoff}")
        print(f"  (samples before this date are from a DIFFERENT model version)")
    elif args.no_version_guard:
        print("WARNING: model version guard disabled. Results may mix model versions.")

    if cutoff is not None:
        before = len(df)
        df = df[df_idx_utc >= cutoff]
        print(f"  Filtered: {before} -> {len(df)} bars after {cutoff}")

    # Filter to realized predictions
    valid = df[df["actual_return_4h"].notna() & df["dir_prob_up"].notna()].copy()
    print(f"Valid (dir_prob_up + actual_return_4h): {len(valid)}")

    if len(valid) < args.min_samples:
        print(f"\n[INSUFFICIENT DATA] n={len(valid)} < min_samples={args.min_samples}")
        print(f"  Calibration estimates with n<{args.min_samples} are too noisy to act on.")
        if cutoff is not None:
            days_since = (pd.Timestamp.now(tz="UTC") - cutoff).total_seconds() / 86400
            print(f"  Hours since model deploy: {days_since*24:.0f}h "
                  f"({days_since:.1f} days)")
            # Rough estimate: indicator runs 1 pred/h, 4h forward → ~20/day after warmup
            est_days_needed = max(1, (args.min_samples - len(valid)) / 20)
            print(f"  Estimated wait time: ~{est_days_needed:.0f} more days to reach n={args.min_samples}")
        print(f"  Use --since <earlier_date> to widen the window (WARNING: may mix models)")
        return

    # Overall calibration
    results = [analyze(valid, "ALL")]

    # Per-regime
    if "regime" in valid.columns:
        for regime in ["CHOPPY", "TRENDING_BULL", "TRENDING_BEAR"]:
            sub = valid[valid["regime"] == regime]
            if len(sub) >= 50:
                results.append(analyze(sub, regime))

    _print_and_save_results(results, args)


def _print_and_save_results(results: list[dict], args):
    # Report
    print("\n" + "=" * 80)
    print("CALIBRATION REPORT")
    print("=" * 80)

    for r in results:
        if r.get("insufficient"):
            continue
        print(f"\n── {r['label']} (n={r['n_samples']}) ──")
        print(f"  Base rate (actual UP%):  {r['mean_outcome']:.3f}")
        print(f"  Brier score:             {r['brier']:.4f}  "
              f"(vs 0.5 baseline: {r['brier_vs_constant_0.5']:.4f}, "
              f"vs climatology: {r['brier_vs_climatology']:.4f})")
        print(f"  Brier skill score:       {r['brier_skill_vs_climatology']:+.4f}  "
              "(positive = beats climatology)")
        print(f"  ECE (lower=better):      {r['ece']:.4f}")

        # Bootstrap 95% CIs
        b = r.get("bootstrap", {})
        if b:
            bs_ci = b["brier_skill_ci"]
            ece_ci = b["ece_ci"]
            print(f"  Bootstrap 95% CI (n_boot={b['n_boot']}):")
            print(f"    Brier skill:  [{bs_ci[0]:+.4f}, {bs_ci[1]:+.4f}]  ", end="")
            if b["brier_skill_crosses_zero"]:
                print("(CROSSES ZERO — not statistically distinguishable from climatology)")
            elif bs_ci[1] < 0:
                print("(entirely BELOW zero — significantly worse than climatology)")
            else:
                print("(entirely ABOVE zero — significantly better than climatology)")
            print(f"    ECE:          [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]")
            cg_ci = b["confidence_gap_ci"]
            if cg_ci[0] is not None:
                print(f"    Conf gap:     [{cg_ci[0]:+.4f}, {cg_ci[1]:+.4f}]  ", end="")
                if b["confidence_gap_crosses_zero"]:
                    print("(CROSSES ZERO — over/under confidence not statistically real)")
                elif cg_ci[0] > 0:
                    print("(significantly OVER-confident)")
                else:
                    print("(significantly UNDER-confident)")

        if r["confidence_gap"] is not None:
            print(f"  Extreme bins (p<=0.4 or p>=0.6): n={r['extreme_n']}")
            print(f"    Avg confidence:        {r['extreme_avg_confidence']:.3f}")
            print(f"    Actual accuracy:       {r['extreme_accuracy']:.3f}")
            print(f"    Confidence gap:        {r['confidence_gap']:+.3f}  ", end="")
            if r['confidence_gap'] > 0.03:
                print("(OVER-confident — predictions too strong)")
            elif r['confidence_gap'] < -0.03:
                print("(UNDER-confident — predictions too weak)")
            else:
                print("(well-calibrated)")

        print(f"  Bin-level:")
        print(f"    {'Range':18s}  {'n':>5s}  {'pred':>6s}  {'actual':>7s}  "
              f"{'diff':>7s}  95% CI")
        for b in r["bins"]:
            marker = ""
            if b["actual"] < b["ci_lo"] or b["actual"] > b["ci_hi"]:
                marker = ""  # within CI by construction
            # Flag bins where predicted is outside actual's CI (severe miscalibration)
            if b["predicted"] < b["ci_lo"] or b["predicted"] > b["ci_hi"]:
                marker = " MISCAL"
            print(f"    {b['range']:18s}  {b['n']:>5d}  {b['predicted']:>6.3f}  "
                  f"{b['actual']:>7.3f}  {b['diff']:>+7.3f}  "
                  f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]{marker}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    overall = results[0]
    boot = overall.get("bootstrap", {})
    # If bootstrap CI crosses zero, downgrade FAIL to INCONCLUSIVE
    if overall["brier_skill_vs_climatology"] <= 0 and boot.get("brier_skill_crosses_zero"):
        print("  [INCONCLUSIVE] Brier skill point estimate is negative, but 95% CI")
        print(f"                 [{boot['brier_skill_ci'][0]:+.4f}, {boot['brier_skill_ci'][1]:+.4f}] "
              "crosses zero.")
        print(f"                 With n={overall['n_samples']} samples, we cannot distinguish")
        print("                 the model from climatology. Accumulate more outcome_tracker")
        print("                 backfill (target n>=500) before concluding miscalibration.")
    elif overall["brier_skill_vs_climatology"] <= 0:
        print("  [FAIL] Model does not beat climatology baseline.")
        print("         Predicting the historical UP rate constantly would be as good.")
        print(f"         95% CI: [{boot['brier_skill_ci'][0]:+.4f}, {boot['brier_skill_ci'][1]:+.4f}]  "
              "(entirely below zero)")
    elif overall["ece"] > 0.05:
        print(f"  [POOR] ECE = {overall['ece']:.4f} > 0.05 — model poorly calibrated.")
        print("         Consider Platt scaling or isotonic regression on dir_prob_up.")
    elif overall["confidence_gap"] and abs(overall["confidence_gap"]) > 0.05:
        direction = "OVER" if overall["confidence_gap"] > 0 else "UNDER"
        print(f"  [WARN] Model is {direction}-confident at extreme bins "
              f"(gap = {overall['confidence_gap']:+.3f}).")
        if direction == "OVER":
            print("         Strong-signal win rate will be LOWER than dir_prob suggests.")
        else:
            print("         Strong-signal threshold may be TOO STRICT — missing good signals.")
    else:
        print("  [OK]   Model is reasonably calibrated for production use.")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    plot_reliability(results, OUT_PNG)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
